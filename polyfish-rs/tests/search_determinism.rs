//! Search reproducibility (audit T3).
//!
//! Two things had to be true before a search experiment could be replayed, and
//! neither was. The Gumbel agent drew its root noise and its opening-temperature
//! sample from the thread-local RNG, so nothing could pin the noise. And
//! `generate_legal_moves` returned the same moves in a different ORDER on every
//! run — two hash containers in movegen were iterated to emit moves, and Rust
//! seeds each map instance separately. Order decides which move receives which
//! Gumbel draw, so a permuted list is a different search.
//!
//! Both are now fixed, and these hold them fixed. Note the second one is not
//! visible if you compare move *types*: the permutation is within a type.

use candle_core::Device;
use candle_nn::VarMap;
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::gumbel_mcts::GumbelMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, TribeType};
use std::sync::Arc;

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

fn make_evaluator() -> (Arc<PolyZeroNet>, Evaluator) {
    let varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);
    let net = Arc::new(PolyZeroNet::new(vs).unwrap());
    let eval = Evaluator::Inline(InlineEvalHandle::new(net.clone()));
    (net, eval)
}

/// Full move identity, not just the move type — the ordering bug this guards
/// against permutes moves *within* a type and is invisible to a type-only
/// comparison.
fn move_list(game: &Game) -> Vec<String> {
    game.legal_moves().iter().map(|m| format!("{m:?}")).collect()
}

#[test]
fn legal_move_order_is_stable_across_identical_states() {
    let first = move_list(&make_game(4242));
    assert!(first.len() > 3, "position is too simple to prove anything");
    for i in 0..30 {
        assert_eq!(
            first,
            move_list(&make_game(4242)),
            "legal move order varies between identical states (attempt {i})"
        );
    }
}

#[test]
fn legal_move_order_is_stable_after_play() {
    // Later positions have more units and more researchable techs, which is
    // where the two offending containers actually had entries.
    let mut game = make_game(4242);
    for _ in 0..6 {
        let moves = game.legal_moves();
        let pick = moves.len() / 2;
        if game.play_move(moves[pick].as_ref()).is_none() {
            break;
        }
    }
    let first = move_list(&game);
    for i in 0..30 {
        assert_eq!(first, move_list(&game), "order varies on re-query (attempt {i})");
    }
}

/// The move sequence a seeded agent plays over `plies` moves.
fn play(evaluator: &Evaluator, seed: u64, plies: usize) -> Vec<String> {
    let mut game = make_game(4242);
    let mut agent = GumbelMctsAgent::new(evaluator, 8, 4).with_search_seed(seed);
    let mut out = Vec::new();
    for i in 0..plies {
        // move_count below TEMPERATURE_MOVE_THRESHOLD also exercises the
        // visit-weighted sampling path, the third RNG draw site.
        match agent.select_move_with_decomposed_visits(&mut game, i) {
            (Some(m), visits) => {
                let shape: Vec<f32> = visits.iter().map(|v| v.visits).collect();
                out.push(format!("{m:?} {shape:?}"));
                if game.play_move(m.as_ref()).is_none() {
                    break;
                }
            }
            (None, _) => break,
        }
    }
    out
}

#[test]
fn same_seed_replays_the_same_search() {
    let (_net, evaluator) = make_evaluator();
    let a = play(&evaluator, 20260818, 6);
    let b = play(&evaluator, 20260818, 6);
    assert!(!a.is_empty(), "search produced no moves — test proves nothing");
    assert_eq!(a, b, "same seed must replay the same move sequence");
}

#[test]
fn the_seed_is_actually_wired_to_the_noise() {
    // Guards the vacuous pass: if no seed reached the Gumbel draw, every seed
    // would agree and the test above would hold for the wrong reason.
    let (_net, evaluator) = make_evaluator();
    let mut seen = std::collections::HashSet::new();
    for seed in 0..10u64 {
        let mut game = make_game(4242);
        let mut agent = GumbelMctsAgent::new(&evaluator, 8, 4).with_search_seed(seed);
        let (_m, visits) = agent.select_move_with_decomposed_visits(&mut game, 0);
        seen.insert(format!("{:?}", visits.iter().map(|v| v.visits).collect::<Vec<_>>()));
    }
    assert!(
        seen.len() > 1,
        "every seed searched identically — the seed is not reaching the Gumbel noise"
    );
}

#[test]
fn agents_do_not_share_a_stream_by_default() {
    // Self-play runs many actors. If they shared a stream every actor would
    // play the same game, so the unseeded default must still vary per agent.
    let (_net, evaluator) = make_evaluator();
    let mut seen = std::collections::HashSet::new();
    for _ in 0..10 {
        let mut game = make_game(4242);
        let mut agent = GumbelMctsAgent::new(&evaluator, 8, 4);
        let (_m, visits) = agent.select_move_with_decomposed_visits(&mut game, 0);
        seen.insert(format!("{:?}", visits.iter().map(|v| v.visits).collect::<Vec<_>>()));
    }
    assert!(seen.len() > 1, "freshly constructed agents all searched identically");
}

/// The other three search agents own their streams too. `gumbel_mcts` is what
/// training runs, but `heuristic_mcts` supplies the greedy teacher (so its
/// randomness reaches training data) and `RandomAgent` is the ladder's Elo-0
/// floor, so a gauge reading depends on it.
mod other_agents {
    use super::make_game;
    use polyfish::ai::heuristic_mcts::{GreedyHeuristicAgent, HeuristicMctsAgent, RandomAgent};

    fn play_random(seed: u64, plies: usize) -> Vec<String> {
        let mut game = make_game(4242);
        let agent = RandomAgent::new().with_search_seed(seed);
        let mut out = Vec::new();
        for _ in 0..plies {
            match agent.select_move(&mut game) {
                Some(m) => {
                    out.push(format!("{m:?}"));
                    if game.play_move(m.as_ref()).is_none() {
                        break;
                    }
                }
                None => break,
            }
        }
        out
    }

    #[test]
    fn random_agent_replays_under_a_seed() {
        let a = play_random(7, 20);
        assert!(a.len() > 3, "too few moves to prove anything");
        assert_eq!(a, play_random(7, 20));
    }

    #[test]
    fn random_agent_actually_uses_its_seed() {
        let seen: std::collections::HashSet<_> = (0..12).map(|s| play_random(s, 20)).collect();
        assert!(seen.len() > 1, "every seed played the same game");
    }

    #[test]
    fn heuristic_agent_replays_under_a_seed() {
        let run = |seed: u64| {
            let mut game = make_game(4242);
            let agent = HeuristicMctsAgent::new(24).with_search_seed(seed);
            (0..4)
                .filter_map(|_| {
                    let m = agent.select_move(&mut game)?;
                    let s = format!("{m:?}");
                    game.play_move(m.as_ref())?;
                    Some(s)
                })
                .collect::<Vec<_>>()
        };
        let a = run(99);
        assert!(!a.is_empty());
        assert_eq!(a, run(99));
    }

    #[test]
    fn greedy_teacher_replays_under_a_seed() {
        let run = |seed: u64| {
            let mut game = make_game(4242);
            let agent = GreedyHeuristicAgent::new().with_search_seed(seed);
            (0..6)
                .filter_map(|_| {
                    let m = agent.select_move(&mut game)?;
                    let s = format!("{m:?}");
                    game.play_move(m.as_ref())?;
                    Some(s)
                })
                .collect::<Vec<_>>()
        };
        let a = run(5);
        assert!(!a.is_empty());
        assert_eq!(a, run(5));
    }
}
