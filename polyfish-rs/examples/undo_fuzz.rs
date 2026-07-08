//! Perft-style make/unmake verifier for the MCTS simulate/undo path.
//!
//! For each seed: build a game exactly like self_play, then interleave
//! random real moves with random MCTS-like descents (simulate_move × depth,
//! undo all). After every descent the full state is deep-compared against
//! the pre-descent snapshot; the first mismatch prints the offending move
//! sequence and the differing JSON paths, then exits non-zero.
//!
//! Usage: cargo run --release --example undo_fuzz -- [num_seeds] [start_seed] [max_depth]

use polyfish::game::Game;
use polyfish::types::TribeType;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;

const TRIBES: [TribeType; 12] = [
    TribeType::Imperius,
    TribeType::Bardur,
    TribeType::Oumaji,
    TribeType::XinXi,
    TribeType::Vengir,
    TribeType::Hoodrick,
    TribeType::Luxidoor,
    TribeType::Zebasi,
    TribeType::AiMo,
    TribeType::Quetzali,
    TribeType::Yadakk,
    TribeType::Kickoo,
];

fn json_diff(prefix: String, a: &Value, b: &Value, out: &mut Vec<String>) {
    if out.len() >= 20 {
        return;
    }
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            for (k, va) in ma {
                match mb.get(k) {
                    Some(vb) => json_diff(format!("{}/{}", prefix, k), va, vb, out),
                    None => out.push(format!("{}/{}: {} -> <missing>", prefix, k, va)),
                }
            }
            for (k, vb) in mb {
                if !ma.contains_key(k) {
                    out.push(format!("{}/{}: <missing> -> {}", prefix, k, vb));
                }
            }
        }
        (Value::Array(aa), Value::Array(ab)) => {
            if aa.len() != ab.len() {
                out.push(format!("{}: array len {} -> {}", prefix, aa.len(), ab.len()));
            }
            for (i, (va, vb)) in aa.iter().zip(ab.iter()).enumerate() {
                json_diff(format!("{}[{}]", prefix, i), va, vb, out);
            }
        }
        _ => {
            if a != b {
                out.push(format!("{}: {} -> {}", prefix, a, b));
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_seeds: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let start_seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let max_depth: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);

    let mut total_descents = 0u64;
    let mut total_sim_moves = 0u64;

    for seed in start_seed..start_seed + num_seeds {
        let mut rng = StdRng::seed_from_u64(seed);
        let t1 = TRIBES[rng.gen_range(0..TRIBES.len())];
        let mut t2 = TRIBES[rng.gen_range(0..TRIBES.len())];
        while t2 == t1 {
            t2 = TRIBES[rng.gen_range(0..TRIBES.len())];
        }

        let gen_settings = polyfish::mapgen::MapGenSettings {
            size: polyfish::types::MapSize::Tiny,
            map_type: polyfish::types::MapType::Drylands,
            tribes: vec![t1, t2],
            seed: seed as i64,
            ..Default::default()
        };

        let mut game = Game::new();
        game.state = polyfish::mapgen::generate(gen_settings);
        game.state.settings.mode = polyfish::types::ModeType::Perfection;
        game.state.settings.max_turns = 30;
        game.post_load();

        let mut real_moves = 0usize;
        'game: while !game.state.settings._game_over && real_moves < 400 {
            // A few MCTS-like descents from the current position.
            // Snapshot before every simulated move so a mismatch can be
            // pinned to the exact move whose undo failed to restore state.
            for _ in 0..4 {
                let mut snapshots: Vec<Value> =
                    vec![serde_json::to_value(&game.state).expect("serialize state")];
                let mut undos = Vec::new();
                let mut seq: Vec<String> = Vec::new();

                let depth = rng.gen_range(1..=max_depth);
                for _ in 0..depth {
                    if game.state.settings._game_over {
                        break;
                    }
                    let legal = game.legal_moves();
                    if legal.is_empty() {
                        break;
                    }
                    let m = &legal[rng.gen_range(0..legal.len())];
                    let desc = m.describe(&game.state);
                    match game.simulate_move(m.as_ref()) {
                        Some(u) => {
                            seq.push(desc);
                            undos.push(u);
                            snapshots
                                .push(serde_json::to_value(&game.state).expect("serialize state"));
                        }
                        None => {
                            println!("SIMULATE FAILED (legal move rejected)");
                            println!("  seed: {}", seed);
                            println!("  path: {:?}", seq);
                            println!("  move: {}", desc);
                            std::process::exit(2);
                        }
                    }
                }
                total_descents += 1;
                total_sim_moves += seq.len() as u64;

                while let Some(u) = undos.pop() {
                    u(&mut game.state);
                    snapshots.pop();
                    let expected = snapshots.last().expect("snapshot underflow");
                    let restored = serde_json::to_value(&game.state).expect("serialize state");
                    if expected != &restored {
                        let bad_idx = undos.len();
                        let mut diffs = Vec::new();
                        json_diff(String::new(), expected, &restored, &mut diffs);
                        println!("UNDO MISMATCH");
                        println!("  seed: {}  (tribes {:?} vs {:?})", seed, t1, t2);
                        println!("  after real move #{}", real_moves);
                        println!("  descent path ({} moves):", seq.len());
                        for (i, d) in seq.iter().enumerate() {
                            let marker = if i == bad_idx { "  <-- BAD UNDO" } else { "" };
                            println!("    {:2}. {}{}", i, d, marker);
                        }
                        println!("  differing fields (max 20):");
                        for d in &diffs {
                            println!("    {}", d);
                        }
                        std::process::exit(1);
                    }
                }
            }

            // Advance the real game with one random legal move.
            let legal = game.legal_moves();
            if legal.is_empty() {
                break 'game;
            }
            let m = &legal[rng.gen_range(0..legal.len())];
            if game.play_move(m.as_ref()).is_none() {
                println!(
                    "REAL MOVE FAILED: seed {} move {}",
                    seed,
                    m.describe(&game.state)
                );
                break 'game;
            }
            real_moves += 1;
        }
        println!(
            "seed {} ok ({} real moves, {} descents so far, {} sim moves)",
            seed, real_moves, total_descents, total_sim_moves
        );
    }
    println!(
        "ALL CLEAN: {} descents, {} simulated moves verified",
        total_descents, total_sim_moves
    );
}
