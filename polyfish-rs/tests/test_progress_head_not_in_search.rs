//! The `v_progress` head must not reach the search (#33).
//!
//! `GumbelNode::q_value()` used to return `value_sum/visits + own_progress`,
//! and the root q_value is the TD bootstrap for training labels
//! (`last_root_value` -> `self_play`'s `td_lambda_labels`). Only candle
//! computes that head — tch and metal stub it to 0 — so identical games
//! produced different training data depending on which box generated them.
//! It is also the node's own mover's quantity while `value_sum` is in the
//! parent's perspective, so under adversarial search a handover child gained
//! the opponent's progress un-negated.
//!
//! The head is still trained (`train.py` MSEs the `progress` target written by
//! `self_play`); it is aux-only now. These pin that: perturbing `v_progress`
//! must move the evaluator's progress output and change nothing else.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use polyfish::ai::eval_server::{Evaluator, InlineEvalHandle};
use polyfish::ai::features::state_to_cpu_features;
use polyfish::ai::gumbel_mcts::GumbelMctsAgent;
use polyfish::ai::network::PolyZeroNet;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, TribeType};
use std::sync::Arc;

const SEARCH_SEED: u64 = 20260822;

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

/// Everything one seeded search contributes to training data and to play: the
/// move played, the root value (the TD bootstrap), and the exported policy
/// target pi' — the last because `extract_policy_targets` builds pi' from the
/// same `q_value()`s, so the head reached the policy label too.
struct SearchResult {
    chosen: String,
    root_value: f32,
    policy_target: Vec<(String, f32)>,
}

fn seeded_search(evaluator: &Evaluator) -> SearchResult {
    let mut game = make_game(7);
    let mut agent = GumbelMctsAgent::new(evaluator, 16, 4).with_search_seed(SEARCH_SEED);
    let (chosen, visits) = agent.select_move_with_decomposed_visits(&mut game, 0);
    let root_value = agent
        .last_root_value()
        .expect("the search ran, so it has a root value");
    SearchResult {
        chosen: format!("{chosen:?}"),
        root_value,
        policy_target: visits
            .iter()
            .map(|v| {
                (
                    format!("{:?}/{:?}/{:?}", v.move_type, v.source_idx, v.target_idx),
                    v.visits,
                )
            })
            .collect(),
    }
}

/// The evaluator's `(value, progress)` for a fixed position, through the same
/// seam the search reads.
fn evaluate_once(evaluator: &Evaluator) -> (f32, f32) {
    let game = make_game(7);
    let features = state_to_cpu_features(&game.state, game.state.settings.current_player_turn_id)
        .expect("features encode");
    let results = evaluator.evaluate(vec![features]);
    (results[0].0, results[0].1)
}

#[test]
fn perturbing_the_progress_head_does_not_move_the_search() {
    let mut varmap = VarMap::new();
    let vs = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let net = Arc::new(PolyZeroNet::new(vs).unwrap());
    let evaluator = Evaluator::Inline(InlineEvalHandle::new(net));

    let before = seeded_search(&evaluator);
    assert!(
        before.policy_target.len() > 1,
        "position is too simple to prove anything about the policy target"
    );
    let (value_before, progress_before) = evaluate_once(&evaluator);

    let filters = {
        let data = varmap.data().lock().unwrap();
        data.get("v_progress.weight")
            .expect("the aux progress head is still in the network")
            .dims()[1]
    };
    varmap
        .set_one(
            "v_progress.weight",
            Tensor::full(5.0f32, (1, filters), &Device::Cpu).unwrap(),
        )
        .unwrap();
    varmap
        .set_one(
            "v_progress.bias",
            Tensor::full(3.0f32, (1,), &Device::Cpu).unwrap(),
        )
        .unwrap();

    let (value_after, progress_after) = evaluate_once(&evaluator);

    // Guards the vacuous pass: if the perturbation never reached the network,
    // everything below would match for the wrong reason.
    assert!(
        (progress_after - progress_before).abs() > 1e-3,
        "the perturbation did not reach the progress head \
         (before {progress_before}, after {progress_after}) — this test proves nothing"
    );
    assert!(
        (value_after - value_before).abs() < 1e-6,
        "perturbing v_progress moved the win value too, so the two heads are \
         not independent and this test cannot isolate progress"
    );

    let after = seeded_search(&evaluator);

    assert_eq!(
        before.chosen, after.chosen,
        "the progress head changed which move the search chose"
    );
    assert_eq!(
        before.root_value, after.root_value,
        "the progress head changed the root value — that value is the TD \
         bootstrap for training labels, so labels are backend-dependent again"
    );
    assert_eq!(
        before.policy_target, after.policy_target,
        "the progress head changed the exported policy target pi' — it is built \
         from the same q_value()s, so the policy label is backend-dependent again"
    );
}
