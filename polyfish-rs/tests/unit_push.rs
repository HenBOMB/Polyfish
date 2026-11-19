use polyfish::Coords;
use polyfish::actions::units::{push_unit, spawn_unit};
use polyfish::functions::{calculate_pushable_position, get_adjacent_indices, get_true_unit_at};
use polyfish::game::Game;
use polyfish::moves::{AttackMove, EndTurnMove, RewardMove, StepMove, SummonMove};
use polyfish::states::{CityState, GameState, TileState, TribeState};
use polyfish::types::{CityRewardType, TerrainType, TribeType, UnitEffect, UnitType};

fn flat_board(size: i32, capital: (i32, i32)) -> Game {
    let mut game = Game::new();
    let state = &mut game.state;
    state.settings.size = size;
    state.settings._fow = false;
    state.settings.version = 115;
    state.settings.current_player_turn_id = 1;
    for idx in 0..size * size {
        state.tiles.insert(
            idx,
            TileState {
                coords: Coords::from_index(idx, size),
                terrain_type: TerrainType::Field,
                explorers: [1, 2].into_iter().collect(),
                ..TileState::default()
            },
        );
    }
    for (id, (x, y)) in [(1, capital), (2, (size - 2, size - 2))] {
        let idx = Coords::from_xy(x, y, size).idx;
        let tile = state.tiles.get_mut(&idx).unwrap();
        tile.capital_of = id;
        tile.owner = id;
        tile.ruling_city_coords = Some(tile.coords);
        state.tribes.insert(
            id,
            TribeState {
                id,
                tribe_type: TribeType::Imperius,
                stars: 20,
                cities: vec![CityState {
                    idx,
                    owner: id,
                    level: 5,
                    production: 5,
                    rewards: vec![
                        CityRewardType::Workshop,
                        CityRewardType::Resources,
                        CityRewardType::BorderGrowth,
                    ],
                    ..CityState::default()
                }],
                ..TribeState::default()
            },
        );
    }
    game.post_load();
    game
}

fn destination(state: &GameState, start: i32) -> Option<i32> {
    calculate_pushable_position(state, get_true_unit_at(state, start).unwrap())
}

#[test]
fn freshly_trained_catapult_heads_toward_center_even_after_another_unit_leaves() {
    let mut game = flat_board(11, (4, 6));
    let capital = 70; // (4, 6)
    let center = 60; // (5, 5), screen east of the capital
    let north = 82; // (5, 7)
    let _ = spawn_unit(&mut game.state, 1, UnitType::Warrior, capital, false);
    let _ = game.play_move(&StepMove::new(capital, north)).unwrap();
    let _ = game
        .play_move(&SummonMove::new(capital, UnitType::Catapult))
        .unwrap();
    let saved = game.to_json().unwrap();
    let mut loaded = Game::from_json(&saved).unwrap();
    let undo = loaded
        .play_move(&RewardMove::new(capital, CityRewardType::SuperUnit))
        .unwrap();
    assert_eq!(
        get_true_unit_at(&loaded.state, center).unwrap().unit_type,
        UnitType::Catapult
    );
    assert_eq!(
        get_true_unit_at(&loaded.state, north).unwrap().unit_type,
        UnitType::Warrior
    );
    assert_eq!(
        get_true_unit_at(&loaded.state, capital).unwrap().unit_type,
        UnitType::Giant
    );
    undo(&mut loaded.state);
    assert_eq!(
        get_true_unit_at(&loaded.state, capital).unwrap().unit_type,
        UnitType::Catapult
    );
}

#[test]
fn unmoved_units_use_screen_south_at_center_and_the_true_midpoint_on_even_boards() {
    let mut odd = flat_board(11, (5, 5));
    let _ = spawn_unit(&mut odd.state, 1, UnitType::Warrior, 60, false);
    assert_eq!(destination(&odd.state, 60), Some(48)); // (5,5) -> (4,4), screen south

    let mut even = flat_board(14, (6, 7));
    let _ = spawn_unit(&mut even.state, 1, UnitType::Warrior, 104, false);
    assert_eq!(destination(&even.state, 104), Some(91)); // (6,7) -> (7,6)
}

#[test]
fn friends_follow_their_last_movement_and_enemy_siegers_reverse_it() {
    for (owner, expected) in [(1, 72), (2, 48)] {
        let mut game = flat_board(11, (5, 5));
        game.state.settings.current_player_turn_id = owner;
        let _ = spawn_unit(&mut game.state, owner, UnitType::Warrior, 48, false);
        let _ = game.play_move(&StepMove::new(48, 60)).unwrap();
        game.state.settings.current_player_turn_id = 1;
        assert_eq!(destination(&game.state, 60), Some(expected));
    }
}

#[test]
fn ranged_facing_survives_turns_and_is_replaced_by_a_later_move() {
    let mut game = flat_board(11, (4, 4));
    let _ = spawn_unit(&mut game.state, 1, UnitType::Catapult, 48, false);
    let _ = spawn_unit(&mut game.state, 2, UnitType::Defender, 28, false);
    let _ = game.play_move(&AttackMove::new(48, 28)).unwrap();
    assert_eq!(destination(&game.state, 48), Some(38)); // (4,4) facing (6,2)
    let _ = game.play_move(&EndTurnMove).unwrap();
    let _ = game.play_move(&EndTurnMove).unwrap();
    assert_eq!(destination(&game.state, 48), Some(38));
    let undo = game.play_move(&StepMove::new(48, 60)).unwrap();
    assert_eq!(destination(&game.state, 60), Some(72));
    undo(&mut game.state);
    assert_eq!(destination(&game.state, 48), Some(38));
}

#[test]
fn blocked_pushes_alternate_left_then_right_and_can_remove_a_unit() {
    let mut game = flat_board(11, (5, 5));
    let _ = spawn_unit(&mut game.state, 1, UnitType::Warrior, 48, false);
    let _ = game.play_move(&StepMove::new(48, 60)).unwrap();
    let _ = spawn_unit(&mut game.state, 2, UnitType::Cloak, 72, false);
    game.state.tribes.get_mut(&2).unwrap().units[0]
        .effects
        .insert(UnitEffect::Invisible);

    assert_eq!(destination(&game.state, 60), Some(71)); // screen northwest
    game.state.tiles.get_mut(&71).unwrap().terrain_type = TerrainType::Mountain;
    assert_eq!(destination(&game.state, 60), Some(61)); // screen northeast

    let pushed = push_unit(&mut game.state, 60).unwrap();
    assert_eq!(destination(&game.state, 61), Some(62)); // a push also changes facing
    (pushed.undo)(&mut game.state);
    assert_eq!(destination(&game.state, 60), Some(61));

    for idx in get_adjacent_indices(&game.state, 60, 1) {
        game.state.tiles.get_mut(&idx).unwrap().terrain_type = TerrainType::Mountain;
    }
    assert_eq!(destination(&game.state, 60), None);
    let undo = game
        .play_move(&RewardMove::new(60, CityRewardType::SuperUnit))
        .unwrap();
    assert_eq!(game.state.tribes[&1].units.len(), 1);
    assert_eq!(
        get_true_unit_at(&game.state, 60).unwrap().unit_type,
        UnitType::Giant
    );
    assert_eq!(game.state.tribes[&2].kills, 0);
    undo(&mut game.state);
    assert_eq!(
        get_true_unit_at(&game.state, 60).unwrap().unit_type,
        UnitType::Warrior
    );
}
