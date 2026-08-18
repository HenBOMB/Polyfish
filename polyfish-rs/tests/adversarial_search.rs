//! Regressions for adversarial in-tree search: `simulate_move(EndTurn)` must
//! hand control to the next player instead of deleting its turn, and the
//! lookahead horizon must respect the game's real length.

use polyfish::ai::brain::{MAX_TURNS_AHEAD, MIN_TURNS_AHEAD, max_turns_ahead};
use polyfish::game::{Game, adversarial_search, set_adversarial_search};
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, MoveType, TerrainType, TribeType};
use std::collections::HashSet;
use std::sync::{Mutex, MutexGuard};

/// The adversarial switch is process-wide, so mode-sensitive tests must not
/// run concurrently. Every such test holds this for its whole body.
static MODE: Mutex<()> = Mutex::new(());

struct ModeGuard(#[allow(dead_code)] MutexGuard<'static, ()>);

impl ModeGuard {
    fn set(on: bool) -> Self {
        let g = MODE.lock().unwrap_or_else(|e| e.into_inner());
        set_adversarial_search(on);
        Self(g)
    }
}

impl Drop for ModeGuard {
    fn drop(&mut self) {
        set_adversarial_search(false);
    }
}

fn make_game(seed: i64) -> Game {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    });
    game.post_load();
    game
}

fn end_turn_move(game: &Game) -> Box<dyn polyfish::moves::Move> {
    game.legal_moves()
        .into_iter()
        .find(|m| m.move_type() == MoveType::EndTurn)
        .expect("EndTurn must always be legal")
}

#[test]
fn the_switch_round_trips() {
    let _g = ModeGuard::set(true);
    assert!(adversarial_search());
    set_adversarial_search(false);
    assert!(!adversarial_search());
}

#[test]
fn sim_end_turn_hands_over_when_adversarial() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(42);
    let mover = game.state.settings.current_player_turn_id;
    let turn = game.state.settings.turn;

    let m = end_turn_move(&game);
    let undo = game.simulate_move(m.as_ref()).expect("EndTurn simulated");

    assert_ne!(
        game.state.settings.current_player_turn_id, mover,
        "simulate_move(EndTurn) must hand control to the next player"
    );
    assert_eq!(
        game.state.settings.turn, turn,
        "handing over inside the turn order must not advance the game turn"
    );

    undo(&mut game.state);
    assert_eq!(game.state.settings.current_player_turn_id, mover);
    assert_eq!(game.state.settings.turn, turn);
}

#[test]
fn sim_end_turn_skips_opponent_when_not_adversarial() {
    let _g = ModeGuard::set(false);
    let mut game = make_game(42);
    let mover = game.state.settings.current_player_turn_id;

    let m = end_turn_move(&game);
    let undo = game.simulate_move(m.as_ref()).expect("EndTurn simulated");

    assert_eq!(
        game.state.settings.current_player_turn_id, mover,
        "legacy single-player search must cycle straight back to the mover"
    );
    undo(&mut game.state);
}

/// Walk the greedy-legal move list far enough to cross several turn
/// boundaries and record who was to move at every ply.
fn descent_players(game: &mut Game, plies: usize) -> (HashSet<i32>, usize) {
    let mut seen = HashSet::new();
    let mut undos = Vec::new();
    let mut end_turns = 0;

    for _ in 0..plies {
        if game.state.settings._game_over {
            break;
        }
        seen.insert(game.state.settings.current_player_turn_id);
        let mut moves = game.legal_moves();
        if moves.is_empty() {
            break;
        }
        // Mirror the tree's own rule: EndTurn only once nothing else is left.
        let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
        if has_other {
            moves.retain(|m| m.move_type() != MoveType::EndTurn);
        }
        let m = moves.remove(0);
        if m.move_type() == MoveType::EndTurn {
            end_turns += 1;
        }
        match game.simulate_move(m.as_ref()) {
            Some(u) => undos.push(u),
            None => break,
        }
    }

    while let Some(u) = undos.pop() {
        u(&mut game.state);
    }
    (seen, end_turns)
}

#[test]
fn descent_visits_both_players_when_adversarial() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(7);
    let (seen, end_turns) = descent_players(&mut game, 80);

    assert!(
        end_turns >= 2,
        "probe must cross at least two turn boundaries, saw {end_turns}"
    );
    assert!(
        seen.contains(&1) && seen.contains(&2),
        "both players must appear on an in-tree path, saw {seen:?}"
    );
}

#[test]
fn descent_never_visits_opponent_when_not_adversarial() {
    let _g = ModeGuard::set(false);
    let mut game = make_game(7);
    let (seen, end_turns) = descent_players(&mut game, 80);

    assert!(end_turns >= 2, "probe must cross turn boundaries");
    assert_eq!(
        seen,
        HashSet::from([1]),
        "legacy search must only ever see the root player"
    );
}

/// The undo chain has to restore the opponent's turn too, not just ours.
#[test]
fn adversarial_descent_undo_round_trips() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(11);
    let before = serde_json::to_value(&game.state).unwrap();
    let (_seen, _) = descent_players(&mut game, 60);
    let after = serde_json::to_value(&game.state).unwrap();
    assert_eq!(before, after, "adversarial descent did not undo cleanly");
}

#[test]
fn max_turns_ahead_uses_its_argument() {
    // The bug this replaces hard-coded a 20-turn game and ignored max_turns.
    let short: Vec<i32> = (0..30).map(|t| max_turns_ahead(t, 10)).collect();
    let long: Vec<i32> = (0..30).map(|t| max_turns_ahead(t, 45)).collect();
    assert_ne!(short, long, "horizon must depend on max_turns");

    // Never look past the game's own end (above the floor).
    for t in 0..60 {
        for &m in &[10, 20, 45, 100] {
            let h = max_turns_ahead(t, m);
            assert!(h >= MIN_TURNS_AHEAD, "horizon {h} below floor at t={t} m={m}");
            assert!(h <= MAX_TURNS_AHEAD, "horizon {h} above cap at t={t} m={m}");
            if m - t >= MIN_TURNS_AHEAD {
                assert!(h <= m - t, "horizon {h} looks past turn {m} from {t}");
            }
        }
    }
}

#[test]
fn max_turns_ahead_is_non_increasing() {
    for &m in &[10, 20, 45, 100] {
        for t in 0..(m + 5) {
            assert!(
                max_turns_ahead(t + 1, m) <= max_turns_ahead(t, m),
                "horizon grew from turn {t} to {} (max_turns {m})",
                t + 1
            );
        }
    }
}

/// The late game is where the old hard-coded 20 collapsed the horizon to 2.
#[test]
fn max_turns_ahead_survives_the_late_game() {
    assert!(
        max_turns_ahead(30, 45) > MIN_TURNS_AHEAD,
        "a 45-turn game still has real lookahead left at turn 30"
    );
}

/// Tactical regression, deterministic substitute for "the search must not walk
/// the defender away": with an enemy unit adjacent to an undefended city, an
/// in-tree EndTurn must expose the enemy's move onto that city. Under the
/// legacy search that line is unreachable, so the threat is invisible and no
/// amount of training can teach the defender to stay.
#[test]
fn in_tree_opponent_can_threaten_an_undefended_city() {
    let _g = ModeGuard::set(true);
    let mut game = make_game(42);
    let size = game.state.settings.size;

    let capital = game.state.tribes[&1].cities[0].idx;

    // Walk our own defender off the capital; that is the move the search is
    // supposed to reject once it can see the consequence.
    let far = (0..size * size)
        .find(|idx| {
            *idx != capital
                && polyfish::functions::get_adjacent_indices(&game.state, capital, 3)
                    .iter()
                    .all(|a| a != idx)
                && matches!(
                    game.state.tiles[idx].terrain_type,
                    TerrainType::Field | TerrainType::Forest
                )
        })
        .expect("map must have a distant land tile");

    // Park the enemy on a land tile next to our capital.
    let adjacent = polyfish::functions::get_adjacent_indices(&game.state, capital, 1)
        .into_iter()
        .find(|idx| {
            *idx != capital
                && matches!(
                    game.state.tiles[idx].terrain_type,
                    TerrainType::Field | TerrainType::Forest
                )
        })
        .expect("capital must have a land neighbour");

    {
        let p1 = game.state.tribes.get_mut(&1).unwrap();
        assert!(!p1.units.is_empty(), "P1 starts with a unit");
        p1.units.truncate(1);
        p1.units[0].coords = polyfish::coords::Coords::from_index(far, size);
        p1.units[0].prev_coords = p1.units[0].coords.clone();
    }
    {
        let p2 = game.state.tribes.get_mut(&2).unwrap();
        assert!(!p2.units.is_empty(), "P2 starts with a unit");
        p2.units.truncate(1);
        p2.units[0].coords = polyfish::coords::Coords::from_index(adjacent, size);
        p2.units[0].prev_coords = p2.units[0].coords.clone();
    }
    for tile in game.state.tiles.values_mut() {
        tile._unit_owner_id = None;
    }
    game.post_load();

    // Search the same obscured view self_play searches. The enemy sits inside
    // our vision, so the belief-state opponent still owns it.
    let mut view = game.clone_for_mcts(1);
    assert!(
        view.state.tribes[&2].units.iter().any(|u| u.coords.idx == adjacent),
        "the adjacent enemy must survive fog obscuring"
    );

    let m = end_turn_move(&view);
    let _undo = view.simulate_move(m.as_ref()).expect("EndTurn simulated");
    assert_eq!(view.state.settings.current_player_turn_id, 2);

    let threats: Vec<(MoveType, String)> = view
        .legal_moves()
        .iter()
        .filter(|mv| mv.target_idx().map_or(false, |t| t as i32 == capital))
        .map(|mv| (mv.move_type(), mv.describe(&view.state)))
        .collect();
    println!("in-tree enemy threats on the capital: {threats:?}");

    assert!(
        threats
            .iter()
            .any(|(t, _)| matches!(t, MoveType::Step | MoveType::Attack | MoveType::Capture)),
        "in-tree opponent must be able to move onto the undefended capital \
         at {capital} from {adjacent}"
    );
}
