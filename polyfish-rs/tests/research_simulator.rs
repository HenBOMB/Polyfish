use polyfish::Coords;
use polyfish::actions::city::{add_population, capture_city};
use polyfish::actions::structure::{create_structure, destroy_structure};
use polyfish::functions::{get_city_at, get_city_production, get_unit_at};
use polyfish::game::Game;
use polyfish::moves::{
    AttackMove, BuildMove, CaptureMove, EndTurnMove, Move, ResearchMove, RewardMove, StepMove,
    UpgradeMove,
};
use polyfish::replay::ReplayCommand;
use polyfish::research::{
    self, EXPERT_BUDGET, MatchConfig, MatchResult, Opponent, OpponentPreset, observation,
};
use polyfish::settings::technology::get_technology_setting;
use polyfish::states::{CityState, GameState, StructureState, UnitState};
use polyfish::types::{
    CityRewardType, MapSize, MapType, ModeType, ResourceType, StructureType, TechnologyType,
    TerrainType, TribeType, UnitType,
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
fn neutral_roads_require_technology_and_visibility() {
    let mut game = fixture(MapSize::Tiny);
    let idx = *game
        .state
        .tiles
        .iter()
        .find(|(_, tile)| tile.owner == 0)
        .unwrap()
        .0;
    let road = BuildMove::new(idx, StructureType::Road);
    let tile = game.state.tiles.get_mut(&idx).unwrap();
    tile.terrain_type = TerrainType::Field;
    tile.explorers.insert(1);
    assert!(!is_legal(&game, &road));

    let tribe = game.state.tribes.get_mut(&1).unwrap();
    tribe.stars = 20;
    tribe.tech_vanilla.push(polyfish::states::TechnologyState {
        tech_type: TechnologyType::Roads,
        discovered: true,
        discovered_turn: 0,
    });
    assert!(is_legal(&game, &road));

    game.state.tiles[&idx].explorers.remove(&1);
    assert!(!is_legal(&game, &road));
    game.state.settings._fow = false;
    assert!(is_legal(&game, &road));
}

#[test]
fn first_contact_respects_fog_and_invisibility() {
    use polyfish::actions::try_discover_other_tribes;
    use polyfish::types::UnitEffect;

    let mut game = fixture(MapSize::Tiny);
    let idx = game.state.tribes[&2].units[0].coords.idx;
    game.state.tiles[&idx].explorers.remove(&1);
    let _ = try_discover_other_tribes(&mut game.state);
    assert!(game.state.tribes[&1].known_players.is_empty());

    game.state.tiles[&idx].explorers.insert(1);
    game.state.tribes[&2].units[0]
        .effects
        .insert(UnitEffect::Invisible);
    let _ = try_discover_other_tribes(&mut game.state);
    assert!(game.state.tribes[&1].known_players.is_empty());

    game.state.tribes[&2].units[0]
        .effects
        .remove(&UnitEffect::Invisible);
    let stars = game.state.tribes[&1].stars;
    let reward = polyfish::functions::get_star_exchange(&game.state, 2);
    let undo = try_discover_other_tribes(&mut game.state);
    assert!(game.state.tribes[&1].known_players.contains(&2));
    assert_eq!(game.state.tribes[&1].stars, stars + reward);
    undo(&mut game.state);
    assert!(game.state.tribes[&1].known_players.is_empty());
    assert_eq!(game.state.tribes[&1].stars, stars);
}

#[test]
fn all_research_maps_have_valid_seeded_starts() {
    for size in research::BOARD_SIZES {
        for (kind, name, _) in research::MAP_TYPES {
            let tiles = size.get_size().pow(2);
            assert_eq!(research::board_size(tiles as u16), Some(size));
            assert_eq!(research::map_type(name), Some(kind));
            for seed in [0, 42, i64::MAX] {
                let start = || {
                    research::new_match_on(
                        size,
                        kind,
                        [TribeType::Kickoo, TribeType::Hoodrick],
                        seed,
                        MatchConfig::new(Opponent::Human, 2),
                    )
                };
                let game = start();
                let state = &game.state;
                assert_eq!(state.tiles.len(), tiles as usize);
                assert_eq!(state.settings.map_type, kind);
                assert_eq!(state.initial_seed, seed);
                for (id, tribe) in &state.tribes {
                    let capital = tribe.starting_tile_coords.idx;
                    assert_eq!(state.tiles[&capital].capital_of, *id);
                    assert_eq!(state.tiles[&capital].owner, *id);
                    assert_eq!(state.tiles[&capital].terrain_type, TerrainType::Field);
                    assert_eq!(tribe.cities[0].idx, capital);
                }
                assert!(
                    state
                        .tiles
                        .values()
                        .any(|tile| tile.terrain_type == TerrainType::Water)
                );
                let other = start();
                for (idx, tile) in &state.tiles {
                    assert_eq!(tile.terrain_type, other.state.tiles[idx].terrain_type);
                    assert_eq!(tile.climate, other.state.tiles[idx].climate);
                }
                let reload = Game::from_json(&game.to_json().unwrap()).unwrap();
                assert_eq!(reload.state.settings.map_type, kind);
                assert_eq!(reload.state.initial_seed, seed);
            }
        }
    }
}

fn land_components(state: &GameState) -> Vec<Vec<i32>> {
    let size = state.settings.size;
    let mut seen = vec![false; (size * size) as usize];
    let mut components = Vec::new();
    for start in 0..size * size {
        if seen[start as usize] || state.tiles[&start].is_water_terrain() {
            continue;
        }
        seen[start as usize] = true;
        let mut frontier = vec![start];
        let mut component = Vec::new();
        while let Some(idx) = frontier.pop() {
            component.push(idx);
            for near in polyfish::functions::get_adjacent_indices(state, idx, 1) {
                if !seen[near as usize] && !state.tiles[&near].is_water_terrain() {
                    seen[near as usize] = true;
                    frontier.push(near);
                }
            }
        }
        components.push(component);
    }
    components.sort_by_key(|component| std::cmp::Reverse(component.len()));
    components
}

#[test]
fn water_maps_respect_their_geography_after_resources_are_placed() {
    let maps = [
        MapType::Lakes,
        MapType::Continents,
        MapType::Pangea,
        MapType::Archipelago,
        MapType::WaterWorld,
    ];
    for (size, island_count) in research::BOARD_SIZES.into_iter().zip([0, 1, 2, 3, 4, 9]) {
        for kind in maps {
            for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 42, -999, i64::MAX] {
                for tribes in [
                    [TribeType::Imperius, TribeType::Bardur],
                    [TribeType::Kickoo, TribeType::Imperius],
                    [TribeType::Bardur, TribeType::Kickoo],
                    [TribeType::Kickoo; 2],
                ] {
                    let game = research::new_match_on(
                        size,
                        kind,
                        tribes,
                        seed,
                        MatchConfig::new(Opponent::Human, 2),
                    );
                    assert_water_geography(
                        &game.state,
                        kind,
                        island_count,
                        &format!("{kind:?} {size:?} seed {seed} {tribes:?}"),
                    );
                }
            }
        }
    }
}

fn assert_water_geography(state: &GameState, kind: MapType, island_count: usize, case: &str) {
    use polyfish::functions::{get_adjacent_indices, get_plus_sign_indices, get_structure_type_at};

    let size = state.settings.size;
    let components = land_components(state);
    let capital = |id| state.tribes[&id].starting_tile_coords.idx;
    let village = |idx| get_structure_type_at(state, idx) == Some(StructureType::Village);
    let settlements = state.tiles.keys().copied().filter(|&idx| village(idx));
    let edge = state.tiles.values().filter(|tile| {
        let (x, y) = (tile.coords.x, tile.coords.y);
        x == 0 || y == 0 || x == size - 1 || y == size - 1
    });
    let land_pct = 100 * components.iter().map(Vec::len).sum::<usize>() / state.tiles.len();
    let has_coast = |idx| {
        get_plus_sign_indices(idx, size)
            .into_iter()
            .any(|near| state.tiles[&near].terrain_type == TerrainType::Water)
    };
    for idx in settlements.clone() {
        assert!(
            !state.tiles[&idx].is_water_terrain(),
            "settlement under water: {case}"
        );
    }
    for tribe in state
        .tribes
        .values()
        .filter(|tribe| tribe.tribe_type == TribeType::Kickoo)
    {
        let fish = get_adjacent_indices(state, tribe.starting_tile_coords.idx, 1)
            .into_iter()
            .filter(|idx| {
                state.tiles[idx].terrain_type == TerrainType::Water
                    && state.tiles[idx].owner == tribe.id
                    && state
                        .resources
                        .get(idx)
                        .and_then(Option::as_ref)
                        .is_some_and(|resource| resource.resource_type == ResourceType::Fish)
            })
            .count();
        assert!(
            fish >= 2,
            "Kickoo needs two shallow fish: {case}, found {fish}"
        );
    }
    match kind {
        MapType::Lakes => {
            assert!(
                edge.clone().all(|tile| !tile.is_water_terrain()),
                "wet outer border: {case}"
            );
            assert!((67..=73).contains(&land_pct), "{case}: {land_pct}% land");
            assert_eq!(components.len(), 1, "disconnected land: {case}");
            for id in [1, 2] {
                let home = components
                    .iter()
                    .find(|component| component.contains(&capital(id)))
                    .unwrap();
                let reachable = home
                    .iter()
                    .filter(|&&idx| village(idx) && state.tiles[&idx].owner == 0)
                    .count();
                assert!(
                    reachable >= 2,
                    "fewer than two land-connected villages: {case}"
                );
            }
        }
        MapType::Pangea => {
            assert!(
                edge.clone().all(|tile| tile.is_water_terrain()),
                "land on the outer edge: {case}"
            );
            assert!((44..=57).contains(&land_pct), "{case}: {land_pct}% land");
            assert!(
                components[0].contains(&((size / 2) * (size + 1))),
                "center is off the mainland: {case}"
            );
            assert!(
                [1, 2]
                    .into_iter()
                    .all(|id| components[0].contains(&capital(id))),
                "capital off mainland: {case}"
            );
            assert_eq!(
                components
                    .iter()
                    .skip(1)
                    .filter(|group| group.len() == 1 && village(group[0]))
                    .count(),
                island_count,
                "wrong number of island villages: {case}"
            );
        }
        MapType::Continents => {
            assert!((44..=57).contains(&land_pct), "{case}: {land_pct}% land");
            for id in [1, 2] {
                let home = components
                    .iter()
                    .find(|component| component.contains(&capital(id)))
                    .unwrap();
                assert!(
                    home.len() >= state.tiles.len() / 9,
                    "capital on a tiny island: {case}"
                );
                assert!(
                    !home.contains(&capital(3 - id)),
                    "capitals share a landmass: {case}"
                );
            }
            assert_eq!(
                components
                    .iter()
                    .filter(|group| group.len() == 1 && village(group[0]))
                    .count(),
                island_count,
                "wrong number of island villages: {case}"
            );
        }
        MapType::Archipelago | MapType::WaterWorld => {
            let land_range = if kind == MapType::Archipelago {
                32..=40
            } else {
                11..=17
            };
            assert!(land_range.contains(&land_pct), "{case}: {land_pct}% land");
            assert!(components.len() >= 3, "missing separate islands: {case}");
            for idx in settlements {
                assert!(has_coast(idx), "settlement {idx} off the coast: {case}");
            }
            let mut open_water: std::collections::HashSet<_> = edge
                .filter(|tile| tile.is_water_terrain())
                .map(|tile| tile.coords.idx)
                .collect();
            let mut frontier: Vec<_> = open_water.iter().copied().collect();
            while let Some(idx) = frontier.pop() {
                for near in get_adjacent_indices(state, idx, 1) {
                    if state.tiles[&near].is_water_terrain() && open_water.insert(near) {
                        frontier.push(near);
                    }
                }
            }
            for id in [1, 2] {
                let has_harbor =
                    get_adjacent_indices(state, capital(id), 1)
                        .into_iter()
                        .any(|idx| {
                            state.tiles[&idx].terrain_type == TerrainType::Water
                                && state.tiles[&idx].owner == id
                                && open_water.contains(&idx)
                        });
                assert!(has_harbor, "capital lacks a viable harbor: {case}");
            }
            if kind == MapType::WaterWorld {
                assert!(
                    components.iter().all(|group| group
                        .iter()
                        .filter(|&&idx| village(idx))
                        .count()
                        <= 2),
                    "more than two settlements on an island: {case}"
                );
            }
        }
        _ => unreachable!(),
    }
}

#[test]
fn regular_tribes_use_the_expected_starting_units_and_tech() {
    use TechnologyType::*;
    use UnitType::*;
    let expected = [
        (TribeType::AiMo, MindBender, Some(Philosophy)),
        (TribeType::Bardur, Warrior, Some(Hunting)),
        (TribeType::Hoodrick, Archer, Some(Archery)),
        (TribeType::Imperius, Warrior, Some(Organization)),
        (TribeType::Kickoo, Warrior, Some(Fishing)),
        (TribeType::Luxidoor, Warrior, Option::None),
        (TribeType::Oumaji, Rider, Some(Riding)),
        (TribeType::Quetzali, Defender, Some(Strategy)),
        (TribeType::Vengir, Swordsman, Some(Smithery)),
        (TribeType::XinXi, Warrior, Some(Climbing)),
        (TribeType::Yadakk, Warrior, Some(Roads)),
        (TribeType::Zebasi, Warrior, Some(Farming)),
    ];
    for ((tribe, value, label), (expected, unit, tech)) in
        research::REGULAR_TRIBES.into_iter().zip(expected)
    {
        assert_eq!(research::research_tribe(value), Some(expected));
        assert_eq!(research::research_tribe(label), Some(expected));
        let game = research::new_match_on(
            MapSize::Tiny,
            MapType::Archipelago,
            [tribe, TribeType::Kickoo],
            42,
            MatchConfig::new(Opponent::Human, 2),
        );
        let player = &game.state.tribes[&1];
        assert_eq!(player.tribe_type, expected);
        assert_eq!(player.units[0].unit_type, unit);
        if let Some(tech) = tech {
            assert!(get_technology_setting(tech).tier.is_some());
            assert!(
                player
                    .tech_vanilla
                    .iter()
                    .any(|entry| entry.tech_type == tech)
            );
        }
        if tribe == TribeType::Luxidoor {
            assert_eq!(player.cities[0].level, 3);
        }
    }
    assert!(research::research_tribe("Aquarion").is_none());
}

const WEST_CAPITAL: i32 = 57;
const WEST_PORT: i32 = 58;
const EAST_CAPITAL: i32 = 63;

fn strait() -> Game {
    let mut game = research::new_match_on(
        MapSize::Tiny,
        MapType::Archipelago,
        [TribeType::Kickoo, TribeType::Imperius],
        42,
        MatchConfig::new(Opponent::Human, 2),
    );
    game.state.resources.clear();
    game.state.structures.clear();
    for capital in [WEST_CAPITAL, EAST_CAPITAL] {
        game.state.structures.insert(
            capital,
            Some(StructureState {
                structure_type: StructureType::Village,
                ..StructureState::default()
            }),
        );
    }
    for (&idx, tile) in &mut game.state.tiles {
        let x = idx % 11;
        tile.terrain_type = match x {
            3 | 7 => TerrainType::Water,
            4..=6 => TerrainType::Ocean,
            _ => TerrainType::Field,
        };
        tile.capital_of = if idx == WEST_CAPITAL {
            1
        } else if idx == EAST_CAPITAL {
            2
        } else {
            0
        };
        tile.owner = if idx == WEST_CAPITAL || idx == WEST_PORT {
            1
        } else {
            tile.capital_of
        };
        tile.ruling_city_coords = match tile.owner {
            1 => Some(Coords::from_index(WEST_CAPITAL, 11)),
            2 => Some(Coords::from_index(EAST_CAPITAL, 11)),
            _ => None,
        };
        tile._unit_owner_id = None;
        tile.explorers.extend([1, 2]);
    }
    for (id, capital) in [(1, WEST_CAPITAL), (2, EAST_CAPITAL)] {
        let tribe = game.state.tribes.get_mut(&id).unwrap();
        tribe.starting_tile_coords = Coords::from_index(capital, 11);
        tribe.cities[0].idx = capital;
        tribe.stars = 30;
        tribe.units[0].coords = tribe.starting_tile_coords;
        tribe.units[0].prev_coords = tribe.starting_tile_coords;
        tribe.units[0].home_coords = Some(tribe.starting_tile_coords);
    }
    game.state.tribes.get_mut(&2).unwrap().units.clear();
    game.post_load();
    game
}

fn is_legal(game: &Game, move_: &dyn Move) -> bool {
    game.legal_moves()
        .iter()
        .any(|candidate| candidate.serialize() == move_.serialize())
}

fn play_legal(game: &mut Game, move_: &dyn Move) {
    assert!(is_legal(game, move_), "{}", move_.describe(&game.state));
    let _ = game.play_move(move_).unwrap();
}

fn next_round(game: &mut Game) {
    play_legal(game, &EndTurnMove);
    play_legal(game, &EndTurnMove);
}

#[test]
fn naval_crossing_upgrades_fights_lands_and_wins_might() {
    let mut game = strait();
    play_legal(&mut game, &BuildMove::new(WEST_PORT, StructureType::Port));
    play_legal(&mut game, &StepMove::new(WEST_CAPITAL, WEST_PORT));
    let raft = get_unit_at(&game.state, WEST_PORT).unwrap();
    assert_eq!(raft.unit_type, UnitType::Raft);
    assert_eq!(raft.passenger_type, Some(UnitType::Warrior));
    next_round(&mut game);
    let ocean = WEST_PORT + 2;
    assert!(!is_legal(&game, &StepMove::new(WEST_PORT, ocean)));

    let enemy = UnitState {
        coords: Coords::from_index(ocean, 11),
        owner: 2,
        unit_type: UnitType::Raft,
        health: 1.0,
        ..UnitState::default()
    };
    game.state.tribes.get_mut(&2).unwrap().units.push(enemy);
    game.post_load();
    assert!(!is_legal(&game, &AttackMove::new(WEST_PORT, ocean)));
    play_legal(&mut game, &ResearchMove::new(TechnologyType::Sailing));
    let upgrade = UpgradeMove::new(WEST_PORT, UnitType::Scoutship);
    assert!(matches!(
        ReplayCommand::from_move(&upgrade).unwrap(),
        ReplayCommand::Upgrade { .. }
    ));
    play_legal(&mut game, &upgrade);
    play_legal(&mut game, &AttackMove::new(WEST_PORT, ocean));
    assert!(get_unit_at(&game.state, ocean).is_none());

    next_round(&mut game);
    play_legal(&mut game, &StepMove::new(WEST_PORT, ocean));
    next_round(&mut game);
    play_legal(&mut game, &StepMove::new(ocean, EAST_CAPITAL));
    let passenger = get_unit_at(&game.state, EAST_CAPITAL).unwrap();
    assert_eq!(passenger.unit_type, UnitType::Warrior);
    assert!(passenger.moved && passenger.attacked);
    assert!(!is_legal(&game, &CaptureMove::new(EAST_CAPITAL)));
    next_round(&mut game);
    play_legal(&mut game, &CaptureMove::new(EAST_CAPITAL));
    assert_eq!(research::result(&game), MatchResult::Won { winner: 1 });
}

#[test]
fn giant_returns_to_land_as_a_giant() {
    let mut game = strait();
    let giant = &mut game.state.tribes[&1].units[0];
    giant.unit_type = UnitType::Giant;
    giant.health = 40.0;
    play_legal(&mut game, &BuildMove::new(WEST_PORT, StructureType::Port));
    play_legal(&mut game, &StepMove::new(WEST_CAPITAL, WEST_PORT));
    assert_eq!(
        get_unit_at(&game.state, WEST_PORT).unwrap().unit_type,
        UnitType::Juggernaut
    );
    next_round(&mut game);
    play_legal(&mut game, &StepMove::new(WEST_PORT, WEST_CAPITAL));
    assert_eq!(
        get_unit_at(&game.state, WEST_CAPITAL).unwrap().unit_type,
        UnitType::Giant
    );
}

#[test]
fn all_presets_can_take_a_capital_across_a_strait() {
    for preset in [
        OpponentPreset::Easy,
        OpponentPreset::Normal,
        OpponentPreset::Expert,
    ] {
        let mut game = strait();
        game.state.settings.max_turns = 20;
        let mut navigated_ocean = false;
        for _ in 0..160 {
            if research::result(&game) != MatchResult::InProgress {
                break;
            }
            if game.current_player_id() == 2 {
                play_legal(&mut game, &EndTurnMove);
                continue;
            }
            let action = research::choose_move(&game, preset).unwrap();
            play_legal(&mut game, action.as_ref());
            navigated_ocean |= game.state.tribes[&1].units.iter().any(|unit| {
                unit.unit_type.is_naval()
                    && game.state.tiles[&unit.coords.idx].terrain_type == TerrainType::Ocean
            });
        }
        assert!(navigated_ocean, "{preset:?} never entered the ocean");
        assert_eq!(
            research::result(&game),
            MatchResult::Won { winner: 1 },
            "{preset:?}"
        );
    }
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

#[test]
fn loading_a_save_preserves_market_income_across_a_city_border() {
    let mut game = fixture(MapSize::Tiny);
    let size = game.map_size();
    let home = game.state.tribes[&1].cities[0].idx;
    let step = if home % size < size / 2 { 1 } else { -1 };
    let (market, sawmill, neighbor) = (home + step, home + 2 * step, home + 3 * step);

    let tribe = game.state.tribes.get_mut(&1).unwrap();
    tribe.cities[0]._territory = vec![market];
    tribe.cities.push(CityState {
        idx: neighbor,
        owner: 1,
        production: 1,
        _territory: vec![sawmill],
        ..CityState::default()
    });
    for (idx, city) in [(market, home), (sawmill, neighbor), (neighbor, neighbor)] {
        let tile = game.state.tiles.get_mut(&idx).unwrap();
        tile.owner = 1;
        tile.ruling_city_coords = Some(Coords::from_index(city, size));
    }
    let _ = create_structure(&mut game.state, market, StructureType::Market, 1);
    let _ = create_structure(&mut game.state, sawmill, StructureType::Sawmill, 1);
    let income =
        |state: &GameState, idx| get_city_production(state, get_city_at(state, idx).unwrap());
    let home_income = income(&game.state, home);
    assert_eq!(home_income, 3); // Base, capital bonus, and one adjacent Sawmill.
    assert_eq!(income(&game.state, neighbor), 1);

    for ruling_city in [Some(Coords::from_index(home, size)), None] {
        game.state.tiles[&market].ruling_city_coords = ruling_city;
        let reloaded = Game::from_json(&game.to_json().unwrap()).unwrap();
        assert_eq!(income(&reloaded.state, home), home_income);
        assert_eq!(income(&reloaded.state, neighbor), 1);
    }
}
