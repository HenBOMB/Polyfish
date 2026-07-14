//! Repro/regression for illegal-move attempts during Gumbel MCTS descent.
//!
//! Root cause: the agent reuses its search tree across real moves, but real
//! `play_move` explores tiles that `simulate_move` deliberately does not.
//! A reused subtree containing a ruin `Capture` can therefore re-roll a
//! different reward on replay (the eligible-reward list depends on nearby
//! fog), making cached grandchildren (e.g. an affordable Research) illegal.
//! The search must tolerate and heal this, never spamming execute errors.

use candle_core::Device;
use candle_nn::VarMap;
use polyfish::ai::brain::{Brain, SearchBackend};
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::{Game, SIM_MOVE_FAILURES};
use polyfish::mapgen::{generate, MapGenSettings};
use polyfish::types::{MapSize, MapType, TribeType};
use std::sync::atomic::Ordering;
use std::sync::Arc;

fn make_game(seed: i64) -> Game {
    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Bardur],
        seed,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.state.settings.max_turns = 8;
    game.post_load();
    game
}

// Heavy end-to-end probe (~2.5 min release; loads model.safetensors when
// present) — run on demand:
//   cargo test --release --test gumbel_reuse_integrity -- --ignored --nocapture
#[test]
#[ignore]
fn test_gumbel_search_never_attempts_illegal_moves() {
    let device = Device::Cpu;
    // Use the real trained weights when present (sharper policy reuses deeper
    // lines, which is where stale-tree bugs live); random weights otherwise.
    let network = if std::path::Path::new("model.safetensors").exists() {
        let vs = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &["model.safetensors"],
                candle_core::DType::F32,
                &device,
            )
            .unwrap()
        };
        Arc::new(PolyZeroNet::new(vs).unwrap())
    } else {
        let varmap = VarMap::new();
        let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
        Arc::new(PolyZeroNet::new(vs).unwrap())
    };
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(network));

    SIM_MOVE_FAILURES.store(0, Ordering::Relaxed);

    for seed in 0..20i64 {
        let mut game = make_game(seed);

        // Mirror self-play: gumbel backend with production search params, one
        // persistent agent per player so tree reuse is exercised.
        let mut agent1 = Brain::with_backend(&evaluator, 64, SearchBackend::Gumbel { k: 16 })
            .with_prior_heuristic_weight(0.1)
            .with_policy_target_q_weight(1.0)
            .with_tree_q_weight(1.0);
        let mut agent2 = Brain::with_backend(&evaluator, 64, SearchBackend::Gumbel { k: 16 })
            .with_prior_heuristic_weight(0.1)
            .with_policy_target_q_weight(1.0)
            .with_tree_q_weight(1.0);

        let mut move_count = 0usize;
        while !polyfish::functions::is_game_over(&game.state) && move_count < 3000 {
            let pov = game.state.settings.current_player_turn_id;
            let agent = if pov == 1 { &mut agent1 } else { &mut agent2 };
            let (best_move, _) = agent.think_decomposed(&game, move_count);
            let m = match best_move {
                Some(m) => m,
                None => Box::new(polyfish::moves::EndTurnMove),
            };
            if game.play_move(m.as_ref()).is_none() {
                break;
            }
            move_count += 1;
        }
        let failures = SIM_MOVE_FAILURES.load(Ordering::Relaxed);
        eprintln!(
            "seed {} done: {} real moves, turn {}, sim-failures so far: {}",
            seed, move_count, game.state.settings.turn, failures
        );
    }

    let failures = SIM_MOVE_FAILURES.load(Ordering::Relaxed);
    assert_eq!(
        failures, 0,
        "MCTS attempted {} illegal simulated moves during descent",
        failures
    );
}
