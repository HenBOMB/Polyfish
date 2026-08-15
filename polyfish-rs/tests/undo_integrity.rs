//! Fuzz probe for MCTS state-restoration bugs: walks random game paths and
//! verifies that (a) undoing a simulated move restores the exact prior state,
//! and (b) re-simulating the same move reproduces the exact same result.
//! Either failing explains "Insufficient stars" errors during tree descent.

use polyfish::game::Game;
use polyfish::types::{MapSize, MapType};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn snap(game: &Game) -> serde_json::Value {
    serde_json::to_value(&game.state).unwrap()
}

/// Return the paths of the first few differing fields between two JSON values.
fn diff_paths(a: &serde_json::Value, b: &serde_json::Value, path: String, out: &mut Vec<String>) {
    if out.len() >= 5 {
        return;
    }
    match (a, b) {
        (serde_json::Value::Object(ma), serde_json::Value::Object(mb)) => {
            for k in ma.keys().chain(mb.keys()) {
                let pa = ma.get(k).unwrap_or(&serde_json::Value::Null);
                let pb = mb.get(k).unwrap_or(&serde_json::Value::Null);
                if pa != pb {
                    diff_paths(pa, pb, format!("{}.{}", path, k), out);
                }
            }
        }
        (serde_json::Value::Array(va), serde_json::Value::Array(vb)) => {
            if va.len() != vb.len() {
                out.push(format!("{}: array len {} vs {}", path, va.len(), vb.len()));
                return;
            }
            for (i, (pa, pb)) in va.iter().zip(vb.iter()).enumerate() {
                if pa != pb {
                    diff_paths(pa, pb, format!("{}[{}]", path, i), out);
                }
            }
        }
        _ => out.push(format!("{}: {} vs {}", path, a, b)),
    }
}

fn assert_same(before: &serde_json::Value, after: &serde_json::Value, ctx: &str) {
    if before != after {
        let mut diffs = Vec::new();
        diff_paths(before, after, String::new(), &mut diffs);
        panic!("STATE MISMATCH {}\nDiffs:\n  {}", ctx, diffs.join("\n  "));
    }
}

/// Recursively simulate a random move, descend deeper, then unwind and verify
/// this level's undo restores the exact prior state, and that replaying the
/// move is deterministic. Inner levels verify themselves first, so a mismatch
/// always implicates this level's move.
fn descent_check(game: &mut Game, rng: &mut StdRng, depth: u32) {
    if depth == 0 || game.state.settings._game_over {
        return;
    }
    let moves = game.legal_moves();
    if moves.is_empty() {
        return;
    }
    let m = &moves[rng.random_range(0..moves.len())];
    let desc = m.describe(&game.state);
    let turn = game.state.settings.turn;

    let before = snap(game);
    let Some(undo) = game.simulate_move(m.as_ref()) else {
        panic!(
            "LEGAL MOVE REFUSED in simulation: {} (turn {})",
            desc, turn
        );
    };
    let after_first = snap(game);

    descent_check(game, rng, depth - 1);

    undo(&mut game.state);
    assert_same(
        &before,
        &snap(game),
        &format!("after undoing [{}] (turn {})", desc, turn),
    );

    // Replay determinism: same state + same move must give the same result.
    let Some(undo2) = game.simulate_move(m.as_ref()) else {
        panic!(
            "REPLAY REFUSED: {} executed once but not twice (turn {})",
            desc, turn
        );
    };
    assert_same(
        &after_first,
        &snap(game),
        &format!("replaying [{}] (turn {})", desc, turn),
    );
    undo2(&mut game.state);
    assert_same(
        &before,
        &snap(game),
        &format!("after second undo of [{}] (turn {})", desc, turn),
    );
}

// Heavy fuzz probe (~6s release, minutes in debug) — run on demand:
//   cargo test --release --test undo_integrity -- --ignored
#[test]
#[ignore]
fn test_simulate_undo_integrity() {
    for game_seed in 0..30i64 {
        let gen_settings = polyfish::mapgen::MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            seed: game_seed,
            ..Default::default()
        };
        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(gen_settings);
        game.state.settings.max_turns = 12;
        game.post_load();

        let mut rng = StdRng::seed_from_u64(game_seed as u64);

        // Random real playout; at every step fuzz simulated descents first.
        for _step in 0..60 {
            if game.state.settings._game_over {
                break;
            }
            descent_check(&mut game, &mut rng, 5);

            let moves = game.legal_moves();
            if moves.is_empty() {
                break;
            }
            let m = &moves[rng.random_range(0..moves.len())];
            if game.play_move(m.as_ref()).is_none() {
                break;
            }
        }
        eprintln!("game seed {} ok (turn {})", game_seed, game.state.settings.turn);
    }
}
