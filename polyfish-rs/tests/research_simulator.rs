use polyfish::Coords;
use polyfish::actions::city::{add_population, capture_city};
use polyfish::actions::structure::destroy_structure;
use polyfish::functions::{get_city_at, get_city_production};
use polyfish::game::Game;
use polyfish::moves::{BuildMove, EndTurnMove, RewardMove};
use polyfish::research::{
    self, EXPERT_BUDGET, MatchConfig, MatchResult, Opponent, OpponentPreset, observation,
};
use polyfish::settings::technology::get_technology_setting;
use polyfish::states::{CityState, GameState};
use polyfish::types::{
    CityRewardType, MapSize, ModeType, ResourceType, StructureType, TerrainType, TribeType,
};

fn fixture(size: MapSize) -> Game {
    research::new_match(
        size,
        [TribeType::Imperius, TribeType::Bardur],
        42,
        MatchConfig::new(Opponent::Expert, 2),
    )
}

#[test]
fn standard_tech_branches_agree_on_prerequisites_and_tiers() {
    use polyfish::types::TechnologyType::{Aquatism, Basic, Fishing, Navigation, Ramming, Sailing};

    let mut pending = vec![Basic];
    while let Some(parent) = pending.pop() {
        let setting = get_technology_setting(parent);
        for child in setting.next {
            let child_setting = get_technology_setting(child);
            assert_eq!(child_setting.requires, (parent != Basic).then_some(parent));
            assert_eq!(child_setting.tier.unwrap(), setting.tier.unwrap() + 1);
            pending.push(child);
        }
    }

    assert_eq!(get_technology_setting(Fishing).next, vec![Sailing, Ramming]);
    assert_eq!(get_technology_setting(Sailing).next, vec![Navigation]);
    assert_eq!(get_technology_setting(Ramming).next, vec![Aquatism]);
}

#[test]
fn drylands_uses_the_selected_climates_without_open_water() {
    for size in [MapSize::Tiny, MapSize::Small, MapSize::Normal] {
        let game = fixture(size);
        for tile in game.state.tiles.values() {
            assert!(matches!(
                tile.climate,
                TribeType::Imperius | TribeType::Bardur
            ));
            assert!(!matches!(
                tile.terrain_type,
                TerrainType::Water | TerrainType::Ocean
            ));
            if tile.owner > 0 {
                assert_eq!(tile.climate, game.state.tribes[&tile.owner].tribe_type);
            }
        }
    }
}

#[test]
fn bardur_drylands_has_fields_and_forests_at_each_size() {
    for size in [MapSize::Tiny, MapSize::Small, MapSize::Normal] {
        for seed in [0, 1, 42, -6_231_624_982_178_099_917] {
            let game = research::new_match(
                size,
                [TribeType::Bardur; 2],
                seed,
                MatchConfig::new(Opponent::Human, 2),
            );
            let tiles = &game.state.tiles;
            let count = |terrain| {
                tiles
                    .values()
                    .filter(|tile| tile.terrain_type == terrain)
                    .count()
            };
            let forest = count(TerrainType::Forest);
            assert!(
                forest > tiles.len() / 8 && forest < tiles.len() / 2,
                "{size:?}, seed {seed}: {forest}"
            );
            assert!(
                count(TerrainType::Field) > tiles.len() / 4,
                "{size:?}, seed {seed}"
            );
        }
    }
}

#[test]
fn workshop_adds_one_star_per_turn_and_is_paid_after_loading() {
    let mut game = fixture(MapSize::Tiny);
    let capital = game.state.tribes[&1].cities[0].idx;
    let _ = add_population(&mut game.state, capital, 2);
    let income =
        |state: &GameState| get_city_production(state, get_city_at(state, capital).unwrap());
    let before = income(&game.state);
    let undo = game
        .play_move(&RewardMove::new(capital, CityRewardType::Workshop))
        .unwrap();
    assert_eq!(income(&game.state), before + 1);
    undo(&mut game.state);
    assert_eq!(income(&game.state), before);
    let _ = game
        .play_move(&RewardMove::new(capital, CityRewardType::Workshop))
        .unwrap();

    let saved = game.to_json().unwrap();
    let mut reloaded = Game::from_json(&saved).unwrap();
    assert_eq!(income(&reloaded.state), before + 1);
    let stars = reloaded.state.tribes[&1].stars;
    let _ = reloaded.play_move(&EndTurnMove).unwrap();
    let _ = reloaded.play_move(&EndTurnMove).unwrap();
    assert_eq!(reloaded.state.tribes[&1].stars, stars + before + 1);
}

#[test]
fn lumber_huts_feed_a_friendly_sawmill_across_city_borders() {
    let mut game = fixture(MapSize::Tiny);
    let home = game.state.tribes[&1].cities[0].idx;
    let (neighbor_city, mill, hut, enemy_hut) = (60, 59, 58, 48);
    game.state
        .tribes
        .get_mut(&1)
        .unwrap()
        .cities
        .push(CityState {
            idx: neighbor_city,
            owner: 1,
            production: 1,
            ..CityState::default()
        });
    game.state.tribes.get_mut(&1).unwrap().stars = 100;
    game.state.settings._fow = false;

    let enemy_city = game.state.tribes[&2].cities[0].idx;
    for (index, owner, city, terrain) in [
        (mill, 1, neighbor_city, TerrainType::Field),
        (hut, 1, home, TerrainType::Forest),
        (enemy_hut, 2, enemy_city, TerrainType::Forest),
    ] {
        let tile = game.state.tiles.get_mut(&index).unwrap();
        tile.owner = owner;
        tile.ruling_city_coords = Some(Coords::from_index(city, 11));
        tile.terrain_type = terrain;
        game.state.structures.shift_remove(&index);
    }
    let _ = polyfish::actions::structure::create_structure(
        &mut game.state,
        enemy_hut,
        StructureType::LumberHut,
        1,
    );
    let population = |state: &GameState, city| get_city_at(state, city).unwrap().population;
    let home_pop = population(&game.state, home);

    let undo_mill = game
        .play_move(&BuildMove::new(mill, StructureType::Sawmill))
        .unwrap();
    assert_eq!(population(&game.state, neighbor_city), 0);
    let undo_hut = game
        .play_move(&BuildMove::new(hut, StructureType::LumberHut))
        .unwrap();
    assert_eq!(population(&game.state, home), home_pop + 1);
    assert_eq!(population(&game.state, neighbor_city), 1);
    let undo_destroy = destroy_structure(&mut game.state, hut);
    assert_eq!(population(&game.state, home), home_pop);
    assert_eq!(population(&game.state, neighbor_city), 0);
    undo_destroy(&mut game.state);
    assert_eq!(population(&game.state, neighbor_city), 1);

    undo_hut(&mut game.state);
    undo_mill(&mut game.state);
    let _ = game
        .play_move(&BuildMove::new(hut, StructureType::LumberHut))
        .unwrap();
    let _ = game
        .play_move(&BuildMove::new(mill, StructureType::Sawmill))
        .unwrap();
    assert_eq!(population(&game.state, neighbor_city), 1);
}

#[test]
fn drylands_keeps_two_fishing_tiles_for_water_tribes() {
    for tribe in [TribeType::Kickoo, TribeType::Aquarion] {
        for seed in 0..12 {
            let game = research::new_match(
                MapSize::Tiny,
                [TribeType::Imperius, tribe],
                seed,
                MatchConfig::new(Opponent::Human, 2),
            );
            let water: Vec<_> = game
                .state
                .tiles
                .values()
                .filter(|tile| matches!(tile.terrain_type, TerrainType::Water | TerrainType::Ocean))
                .collect();
            assert_eq!(water.len(), 2, "{tribe:?}, seed {seed}");
            for tile in water {
                let index = tile.coords.idx;
                assert_eq!(tile.owner, 2);
                assert_eq!(tile.terrain_type, TerrainType::Water);
                assert_eq!(
                    game.state.resources[&index].as_ref().unwrap().resource_type,
                    ResourceType::Fish
                );
                assert_ne!(
                    game.state
                        .structures
                        .get(&index)
                        .and_then(Option::as_ref)
                        .map(|structure| structure.structure_type),
                    Some(StructureType::Lighthouse)
                );
            }
        }
    }
}

fn turn(game: &mut Game, preset: OpponentPreset) -> Vec<serde_json::Value> {
    let pov = game.current_player_id();
    let mut actions = Vec::new();
    while game.current_player_id() == pov && research::result(game) == MatchResult::InProgress {
        let action = research::choose_move(game, preset).expect("a legal bot move");
        actions.push(action.serialize());
        let _undo = game.play_move(action.as_ref()).expect("move accepted");
        assert!(actions.len() <= 100, "bot did not finish the turn");
    }
    actions
}

#[test]
fn all_presets_finish_the_same_turn_after_a_save_load() {
    for size in [MapSize::Tiny, MapSize::Small, MapSize::Normal] {
        for preset in [
            OpponentPreset::Easy,
            OpponentPreset::Normal,
            OpponentPreset::Expert,
        ] {
            let mut game = fixture(size);
            let saved = game.to_json().unwrap();
            let mut loaded = Game::from_json(&saved).unwrap();
            assert_eq!(
                loaded.state.settings.research.unwrap().preset_version,
                research::PRESET_VERSION
            );
            assert_eq!(loaded.tile_count(), size.get_size().pow(2));
            assert_eq!(turn(&mut game, preset), turn(&mut loaded, preset));
            assert_eq!(game.current_player_id(), 2);
        }
    }
}

#[test]
fn expert_uses_the_budget_and_ignores_hidden_state() {
    let game = fixture(MapSize::Tiny);
    let mut changed = game.clone();
    let hidden = game
        .state
        .tiles
        .values()
        .find(|tile| !tile.explorers.contains(&1))
        .unwrap()
        .coords
        .idx;
    let visible = game
        .state
        .tiles
        .values()
        .find(|tile| {
            tile.explorers.contains(&1)
                && !game
                    .state
                    .structures
                    .get(&tile.coords.idx)
                    .is_some_and(Option::is_some)
                && !game
                    .state
                    .resources
                    .get(&tile.coords.idx)
                    .is_some_and(Option::is_some)
        })
        .unwrap()
        .coords
        .idx;
    changed.state.tiles[&hidden].terrain_type = TerrainType::Mountain;
    changed.state.tiles[&hidden].has_road = true;
    changed.state.tribes[&2].stars += 1000;
    changed.state.resources.insert(
        visible,
        Some(polyfish::states::ResourceState {
            resource_type: ResourceType::Metal,
        }),
    );
    assert_eq!(
        observation(&game, 1).state.tiles[&hidden].terrain_type,
        TerrainType::Field
    );
    assert!(observation(&changed, 1).state.resources[&visible].is_none());
    for preset in [
        OpponentPreset::Easy,
        OpponentPreset::Normal,
        OpponentPreset::Expert,
    ] {
        let (expected, advances) = research::choose_with_budget(&game, preset, EXPERT_BUDGET);
        let (actual, changed_advances) =
            research::choose_with_budget(&changed, preset, EXPERT_BUDGET);
        assert_eq!(expected.unwrap().serialize(), actual.unwrap().serialize());
        assert_eq!(advances, changed_advances);
        assert_eq!(
            advances,
            if preset == OpponentPreset::Expert {
                EXPERT_BUDGET
            } else {
                0
            }
        );
    }
}

#[test]
fn might_capital_capture_ends_the_match_and_undo_restores_it() {
    let mut game = fixture(MapSize::Tiny);
    let original_capital = game.state.tribes[&2].starting_tile_coords.idx;
    let village = game
        .state
        .structures
        .iter()
        .find_map(|(&idx, structure)| {
            structure
                .as_ref()
                .filter(|s| s.structure_type == polyfish::StructureType::Village)
                .filter(|_| game.state.tiles[&idx].owner == 0)
                .map(|_| idx)
        })
        .unwrap();
    game.state.settings.current_player_turn_id = 2;
    let _found_city = capture_city(&mut game.state, village).unwrap();
    game.state.settings.current_player_turn_id = 1;

    let undo = capture_city(&mut game.state, original_capital).unwrap();
    assert_eq!(game.state.tribes[&2].cities.len(), 1);
    assert_eq!(research::result(&game), MatchResult::Won { winner: 1 });
    assert!(game.legal_moves().is_empty());
    let loaded = Game::from_json(&game.to_json().unwrap()).unwrap();
    assert_eq!(research::result(&loaded), MatchResult::Won { winner: 1 });

    undo(&mut game.state);
    assert_eq!(research::result(&game), MatchResult::InProgress);
    assert_eq!(game.state.tribes[&2].cities.len(), 2);
    game.state.settings.mode = ModeType::Domination;
    let _capture = capture_city(&mut game.state, original_capital).unwrap();
    assert_eq!(research::result(&game), MatchResult::InProgress);
}

#[test]
fn research_turn_limit_is_a_truncation() {
    let mut game = fixture(MapSize::Tiny);
    game.state.settings.max_turns = game.turn();
    let _first = game.play_move(&EndTurnMove).unwrap();
    let _second = game.play_move(&EndTurnMove).unwrap();
    assert_eq!(research::result(&game), MatchResult::Truncated);
    assert_eq!(game.state._history.len(), 2);
}

#[test]
fn a_stale_territory_list_does_not_duplicate_build_actions() {
    use polyfish::states::TechnologyState;
    use polyfish::types::{MoveType, StructureType, TechnologyType};

    let mut game = fixture(MapSize::Tiny);
    let tribe = game.state.tribes.get_mut(&1).unwrap();
    let city = &tribe.cities[0];
    let target = *city
        ._territory
        .iter()
        .find(|&&idx| idx != city.idx)
        .unwrap();
    let mut stale = city.clone();
    stale.idx = 120;
    stale._territory = vec![target];
    tribe.cities.push(stale);
    tribe.stars = 10;
    tribe.tech_vanilla.push(TechnologyState {
        tech_type: TechnologyType::Forestry,
        discovered: true,
        discovered_turn: 0,
    });
    game.state.tiles[&target].terrain_type = TerrainType::Forest;
    game.state.structures.insert(target, None);
    game.state.resources.insert(target, None);

    let builds = game
        .legal_moves()
        .iter()
        .filter(|action| {
            action.move_type() == MoveType::Build
                && action.target_idx().ok() == Some(target as usize)
                && action.structure_type().ok() == Some(StructureType::LumberHut)
        })
        .count();
    assert_eq!(builds, 1);
}
