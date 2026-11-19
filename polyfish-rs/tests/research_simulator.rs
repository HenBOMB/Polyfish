use polyfish::actions::city::capture_city;
use polyfish::game::Game;
use polyfish::moves::EndTurnMove;
use polyfish::research::{
    self, EXPERT_BUDGET, MatchConfig, MatchResult, Opponent, OpponentPreset, observation,
};
use polyfish::settings::technology::get_technology_setting;
use polyfish::types::{MapSize, ModeType, ResourceType, StructureType, TerrainType, TribeType};

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
