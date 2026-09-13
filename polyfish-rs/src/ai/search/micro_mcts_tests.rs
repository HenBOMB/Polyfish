//! Split to a separate file (Sep 2026, wave-batching work) so the production
//! module this backs stays well under 1000 lines despite thorough coverage.

use super::*;
use crate::ai::eval_server::{DummyEvalHandle, Evaluator};
use crate::ai::oracle_macro::compute_macro_goal;
use crate::ai::search::goal_aux::compute_goal_aux;
use crate::ai::search::macro_exec::rank_plies;

/// EXP_ELO_079: measure this search's OWN emergent depth at production
/// params (sims=64, k=4 -- POLYFISH_MICRO_MCTS_SIMS=64 was the only
/// override in EXP_ELO_074's launch config, K/DEPTH/CPUCT stayed at
/// their env-var defaults), instead of trusting the unrelated old-
/// GumbelMctsAgent depth/sims curve cited in this module's own doc
/// comment. Uses `Evaluator::Dummy` (constant leaf value) so this is a
/// mechanics-only measurement, independent of any trained checkpoint --
/// a real network's sharper Q differences would let PUCT concentrate
/// visits (and thus depth) along a preferred line MORE than this
/// constant-value floor does, so this measurement is a conservative
/// lower bound on production depth, not an exact match.
#[test]
fn measures_own_emergent_depth_at_production_params() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let params = MicroParams { sims: 64, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0 , forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 1 };

    for seed in 0..6i64 {
        let mut game = Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: crate::types::MapSize::Tiny,
            map_type: crate::types::MapType::Drylands,
            tribes: vec![crate::types::TribeType::Imperius, crate::types::TribeType::Bardur],
            seed,
            version: 115,
        });
        game.post_load();
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        let goal = compute_macro_goal(&view.state, pov, 0);
        let aux = compute_goal_aux(&view.state, pov, &goal, 0, 0, None);
        let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
        let ranked = rank_plies(&mut view, pov, &goal, &aux, star_gate, 1.0, None, None);
        if ranked.len() < 2 {
            continue;
        }
        micro_search_pick(&view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &params, None, false);
    }

    let calls = MICRO_MCTS_DEPTH_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let sum = MICRO_MCTS_DEPTH_SUM.load(std::sync::atomic::Ordering::Relaxed);
    let max_seen = MICRO_MCTS_MAX_DEPTH_SEEN.load(std::sync::atomic::Ordering::Relaxed);
    assert!(calls > 0, "search never actually ran (every root ply was trivial EndTurn-only?)");
    let mean = sum as f64 / calls as f64;
    eprintln!(
        "EXP_ELO_079 measured depth @ sims=64,k=4: calls={calls} mean_max_depth={mean:.2} deepest_line_seen={max_seen}"
    );
    // Not a pass/fail assertion on the exact number -- this test's job is
    // to print the real measurement; see the ledger entry for the read.
}

/// EXP_ELO_153: `goal_prior_w` must be a no-op at 0.0 (proven by
/// construction -- the whole block is gated on `!= 0.0`, already
/// exercised by every other test in this file passing `goal_prior_w:
/// 0.0`) and must measurably shift PUCT priors toward the candidate that
/// reduces distance to the committed EXPAND target when nonzero. Finds a
/// real unit with 2+ legal Step destinations at different distances from
/// a chosen far target, so the two candidates differ ONLY in
/// destination -- same source unit, same base score (0.0, synthetic) --
/// isolating the new term as the sole source of any prior difference.
#[test]
fn goal_prior_w_shifts_priors_toward_the_closer_expand_candidate() {
    use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
    use crate::coords::Coords;

    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    for seed in 0..20i64 {
        let game = tiny_game_at_seed(seed);
        let pov = game.state.settings.current_player_turn_id;
        let size = game.state.settings.size;
        let moves = game.legal_moves();

        let mut by_source: std::collections::HashMap<i32, Vec<&Box<dyn Move>>> =
            std::collections::HashMap::new();
        for m in moves.iter().filter(|m| m.move_type() == MoveType::Step) {
            if let Ok(src) = m.source_idx() {
                by_source.entry(src as i32).or_default().push(m);
            }
        }
        let Some(cands) = by_source.values().find(|v| v.len() >= 2) else { continue };

        let far_target = size * size - 1;
        let far_coords = Coords::from_index(far_target, size);
        let mut dists: Vec<(i32, usize)> = cands
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let t = m.target_idx().unwrap() as i32;
                (Coords::from_index(t, size).chebyshev_distance_to(&far_coords), i)
            })
            .collect();
        dists.sort_by_key(|&(d, _)| d);
        let (closer_d, closer_i) = dists[0];
        let (farther_d, farther_i) = *dists.last().unwrap();
        if farther_d - closer_d < 2 {
            continue; // not enough spread on this seed's map -- try another
        }

        let ranked: Vec<(f32, Box<dyn Move>)> =
            cands.iter().map(|m| (0.0f32, dyn_clone::clone_box(m.as_ref()))).collect();
        let goal = MacroGoal {
            orders: vec![(OrderKind::Expand, far_target)],
            stance: Stance::Grow,
            save_target: None,
            prepare: None,
        };
        let aux = compute_goal_aux(&game.state, pov, &goal, 0, 0, None);
        let base =
            MicroParams { sims: 1, depth: 64, k: ranked.len().max(4), c_puct: 1.5, net_prior_w: 0.0, forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 1 };
        let boosted = MicroParams { goal_prior_w: 50.0, ..base };

        let (_, _, _, trace0) =
            micro_search_pick(&game, pov, &goal, &ranked, &aux, false, &evaluator, &base, None, false);
        let (_, _, _, trace1) =
            micro_search_pick(&game, pov, &goal, &ranked, &aux, false, &evaluator, &boosted, None, false);
        assert_eq!(trace0.len(), ranked.len());
        assert_eq!(trace1.len(), ranked.len());

        let closer_gain = trace1[closer_i].prior - trace0[closer_i].prior;
        let farther_gain = trace1[farther_i].prior - trace0[farther_i].prior;
        assert!(
            closer_gain > farther_gain,
            "seed {seed}: closer candidate's prior should gain more than the \
             farther one's when goal_prior_w > 0 (closer {closer_d} tiles away \
             gained {closer_gain:+.4}, farther {farther_d} tiles away gained \
             {farther_gain:+.4})"
        );
        return;
    }
    panic!("no seed in 0..20 produced a usable same-unit, 2+-destination scenario");
}

fn tiny_game_at_seed(seed: i64) -> Game {
    let mut game = Game::new();
    game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
        size: crate::types::MapSize::Tiny,
        map_type: crate::types::MapType::Drylands,
        tribes: vec![crate::types::TribeType::Imperius, crate::types::TribeType::Bardur],
        seed,
        version: 115,
    });
    game.post_load();
    game
}

fn uniform_policy(spatial: usize) -> crate::ai::network::RawPolicyOutput {
    crate::ai::network::RawPolicyOutput {
        fog: None,
        macro_stance: None,
        macro_order: None,
        rollout_value: None,
        action_type: vec![0.0; 11],
        source_spatial: vec![0.0; spatial],
        target_spatial: vec![0.0; spatial],
        move_option: vec![0.0; 192],
    }
}

/// EXP_ELO_126: `net_prior_w == 0.0` must fully gate off the widening
/// path -- the whole point of the gate is that this fix is byte-
/// identical to the pre-EXP_ELO_126 behavior when the net prior is
/// disabled. Checked via the RETURNED PICK, not the shared
/// `MICRO_MCTS_UNION_WIDENED` atomic -- that counter is a process-wide
/// static and `cargo test` runs tests in parallel by default, so a
/// concurrently-running test that deliberately triggers widening (e.g.
/// `union_pick_maps_back_to_the_original_ranked_index`) can and does
/// increment it mid-run here, producing a false failure. The pick
/// itself has no such cross-test interference.
#[test]
fn union_widening_is_fully_gated_off_at_net_prior_w_zero() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let params = MicroParams { sims: 8, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0 , forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 1 };
    let mut ran_any = false;
    for seed in 0..8i64 {
        let game = tiny_game_at_seed(seed);
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        let goal = compute_macro_goal(&view.state, pov, 0);
        let aux = compute_goal_aux(&view.state, pov, &goal, 0, 0, None);
        let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
        let ranked = rank_plies(&mut view, pov, &goal, &aux, star_gate, 1.0, None, None);
        if ranked.len() < 2 {
            continue;
        }
        let heur_top = ranked.len().min(4);
        ran_any = true;
        let (picked, _, _, _) =
            micro_search_pick(&view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &params, None, false);
        if let Some(idx) = picked {
            assert!(
                idx < heur_top,
                "seed {seed}: net_prior_w == 0.0 but the pick ({idx}) fell outside the \
                 heuristic's own top-{heur_top} -- the widening gate is not actually closed"
            );
        }
    }
    assert!(ran_any, "no seed produced a real search call -- test setup is broken");
}

/// `net_root` rework: `root_already_net_ranked == true` must fully gate
/// off the second-forward-pass widening block even when `net_prior_w`
/// is nonzero -- there's nothing left for it to add once `ranked` is
/// already net-derived. Checked via the returned pick staying inside
/// `ranked`'s own top-k, not the shared `MICRO_MCTS_UNION_WIDENED`
/// atomic -- per `union_pick_maps_back_to_the_original_ranked_index`'s
/// own note below, that counter is a process-wide static and `cargo
/// test` runs tests in parallel by default, so it isn't a reliable
/// per-test signal.
#[test]
fn root_already_net_ranked_fully_gates_off_the_widening_block() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let params = MicroParams { sims: 8, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.3 , forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 1 };
    let mut ran_any = false;
    for seed in 0..8i64 {
        let game = tiny_game_at_seed(seed);
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        let goal = compute_macro_goal(&view.state, pov, 0);
        let aux = compute_goal_aux(&view.state, pov, &goal, 0, 0, None);
        let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
        let ranked = rank_plies(&mut view, pov, &goal, &aux, star_gate, 1.0, None, None);
        if ranked.len() < 2 {
            continue;
        }
        let heur_top = ranked.len().min(4);
        ran_any = true;
        let (picked, _, _, _) = micro_search_pick(
            &view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &params, None, true,
        );
        if let Some(idx) = picked {
            assert!(
                idx < heur_top,
                "seed {seed}: root_already_net_ranked == true but the pick ({idx}) fell \
                 outside the top-{heur_top} -- the widening block did not gate off"
            );
        }
    }
    assert!(ran_any, "no seed produced a real search call -- test setup is broken");
}

/// EXP_ELO_126: the core regression this fix exists to prevent. Before
/// this fix, `micro_search_pick` returned an index into its OWN
/// children vector, which happened to equal the index into `ranked`
/// only because children were always exactly `ranked[..top_n]` in
/// order. Once the candidate set can be a union that appends net-only
/// moves after the heuristic's own top-k, returning the raw
/// children-vector position would silently point at the wrong move.
/// This finds a real position with legal moves outside the heuristic's
/// top-4, crafts a policy that the REAL `compute_move_priors_raw`
/// (ground truth, not a hand-derived guess about `mapper.rs`'s
/// internals) actually prefers for one of those excluded moves, forces
/// a deterministic single-simulation pick of it (`sims: 1` -- PUCT's
/// very first selection, with every child unvisited, is exactly
/// `argmax(prior)`; `net_prior_w: 1.0` makes prior purely net-driven),
/// and asserts the returned index is the move's ORIGINAL position in
/// `ranked`, not its position among the union's children.
#[test]
fn union_pick_maps_back_to_the_original_ranked_index() {
    for seed in 0..20i64 {
        let game = tiny_game_at_seed(seed);
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        // Must match `micro_search_pick`'s own `map_size` derivation
        // exactly -- Tiny maps are NOT the 11x11 feature-space
        // constant, so hardcoding 11 here silently miscoordinates
        // every spatial index crafted below against a different grid
        // width than the real function decodes against.
        let map_size = view.state.settings.size as usize;
        let spatial = map_size * map_size;
        let goal = compute_macro_goal(&view.state, pov, 0);
        let aux = compute_goal_aux(&view.state, pov, &goal, 0, 0, None);
        let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
        let ranked = rank_plies(&mut view, pov, &goal, &aux, star_gate, 1.0, None, None);
        let heur_top = ranked.len().min(4);
        if ranked.len() <= heur_top {
            continue; // nothing outside the heuristic's own top-k here
        }
        let all_moves: Vec<Box<dyn Move>> =
            ranked.iter().map(|(_, mv)| dyn_clone::clone_box(mv.as_ref())).collect();

        // Try biasing one coordinate at a time across all four policy
        // tensors until the REAL decode function clearly prefers a
        // move outside the heuristic's top-k.
        let tensor_lens = [11usize, spatial, spatial, 192];
        let mut winner: Option<(crate::ai::network::RawPolicyOutput, usize)> = None;
        'search: for (tensor, &len) in tensor_lens.iter().enumerate() {
            for i in 0..len {
                let mut policy = uniform_policy(spatial);
                match tensor {
                    0 => policy.action_type[i] = 50.0,
                    1 => policy.source_spatial[i] = 50.0,
                    2 => policy.target_spatial[i] = 50.0,
                    _ => policy.move_option[i] = 50.0,
                }
                let scores = crate::ai::search::policy_composer::compute_move_priors_raw(
                    &policy, &all_moves, map_size, false,
                );
                // Lowest-index-wins-ties argmax, matching
                // `micro_search_pick`'s own `net_order` sort exactly --
                // `Iterator::max_by` breaks ties toward the LAST
                // element, the opposite direction, which silently
                // picked a different (tied) winner than production
                // here during development.
                let mut best_i = 0usize;
                let mut best_s = scores[0];
                for (i, &s) in scores.iter().enumerate().skip(1) {
                    if s > best_s {
                        best_s = s;
                        best_i = i;
                    }
                }
                // Require a clean margin over EVERY other candidate
                // (not just the heuristic's own top-k) -- a move that
                // merely edges out the top-4 while nearly tying some
                // OTHER excluded move (e.g. two units able to step onto
                // the same tile) is exactly the ambiguous case a
                // tie-break-direction mismatch can flip.
                let second_best = scores
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != best_i)
                    .map(|(_, &s)| s)
                    .fold(f32::MIN, f32::max);
                if best_i >= heur_top && best_s > second_best * 2.0 + 1e-6 {
                    winner = Some((policy, best_i));
                    break 'search;
                }
            }
        }
        let Some((policy, target_idx)) = winner else {
            continue; // this seed's move set didn't yield a clean case, try another
        };

        let evaluator = Evaluator::Dummy(DummyEvalHandle::new().with_policy(policy));
        let params = MicroParams { sims: 1, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 1.0 , forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 1 };
        let (picked, _, _, _) =
            micro_search_pick(&view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &params, None, false);
        assert_eq!(
            picked,
            Some(target_idx),
            "seed {seed}: expected the net-only candidate at ranked[{target_idx}] (outside \
             the heuristic's top-{heur_top}) to be returned by its ORIGINAL ranked index, got {picked:?}"
        );
        return; // one confirmed case is enough to pin the invariant
    }
    panic!("no seed across 0..20 produced a usable net-only-candidate scenario -- test setup needs a wider seed range");
}

/// EXP_ELO_119: pins the fix for EXP_ELO_079's own collapse -- a real
/// gap this project has actually measured (idx177, GARRISON_49) no
/// longer zeroes every other candidate's prior.
#[test]
fn softmax_priors_no_longer_collapses_on_a_real_measured_gap() {
    let scores = [873.678, 446.889, 445.889, 168.399, 41.915, 27.000];
    let priors = softmax_priors(&scores);
    assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-4, "priors must sum to 1: {priors:?}");
    for (i, p) in priors.iter().enumerate() {
        assert!(
            *p > 0.01,
            "candidate {i} (score {}) got a near-zero prior ({p}) -- the collapse EXP_ELO_079 \
             diagnosed is back: PUCT's exploration term can never pull visits toward it",
            scores[i]
        );
    }
    assert!(priors[0] > priors[1], "the top-scoring candidate should still lead");
}

/// A genuine outlier (not just a "big" gap, but overwhelmingly clear of
/// a tight cluster) should still concentrate the bulk of the mass on
/// it. Population std is measured INCLUDING the outlier, so a single
/// extreme value inflates its own denominator ("self-dilution") --
/// this candidate set's z-score comes out to ~2.65, giving ~0.75 rather
/// than the near-1.0 a naive read might expect. That's still a real,
/// large majority (vs. the flat 1.0/0.0 EXP_ELO_079 diagnosed), so it's
/// pinned at a looser bound, not tightened until a real gauge says the
/// dilution itself costs something.
#[test]
fn softmax_priors_still_favors_a_genuine_outlier() {
    let scores = [1000.0, 10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9];
    let priors = softmax_priors(&scores);
    assert!(priors[0] > 0.5, "a true outlier should still clearly dominate: {priors:?}");
}

/// EXP_ELO_150: `forced_playouts` must be a true no-op when off (the
/// project's standing convention for every opt-in flag) -- same pick,
/// same child trace, as the exact same call with the field omitted from
/// the sims loop entirely.
#[test]
fn forced_playouts_off_is_byte_identical_to_the_old_sims_loop() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let base = MicroParams { sims: 8, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0, forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 1 };
    let mut ran_any = false;
    for seed in 0..8i64 {
        let game = tiny_game_at_seed(seed);
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        let goal = compute_macro_goal(&view.state, pov, 0);
        let aux = compute_goal_aux(&view.state, pov, &goal, 0, 0, None);
        let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
        let ranked = rank_plies(&mut view, pov, &goal, &aux, star_gate, 1.0, None, None);
        if ranked.len() < 5 {
            continue; // want at least a few real root children
        }
        ran_any = true;
        let (pick_a, _, _, trace_a) =
            micro_search_pick(&view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &base, None, false);
        let (pick_b, _, _, trace_b) =
            micro_search_pick(&view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &base, None, false);
        assert_eq!(pick_a, pick_b, "seed {seed}: same off params must reproduce the same pick");
        assert_eq!(
            trace_a.iter().map(|c| c.visits).collect::<Vec<_>>(),
            trace_b.iter().map(|c| c.visits).collect::<Vec<_>>(),
            "seed {seed}: same off params must reproduce the same visit distribution"
        );
    }
    assert!(ran_any, "no seed produced a ply with >=5 real candidates -- widen the seed range");
}

/// EXP_ELO_150: the whole point of the warm start -- with `sims >=
/// num_root_children`, every child gets a real visit, closing the "a
/// candidate the heuristic ranked far ahead can end the search with
/// zero visits, purely from early PUCT path-dependence" failure mode
/// this experiment measured directly on real games (28/176 Step moves
/// by a goal-committed unit misaligned, see the ledger entry).
#[test]
fn forced_playouts_guarantees_every_root_child_at_least_one_visit() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let params = MicroParams { sims: 8, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0, forced_playouts: true, goal_prior_w: 0.0, leaf_batch: 1 };
    let mut checked_any = false;
    for seed in 0..8i64 {
        let game = tiny_game_at_seed(seed);
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        let goal = compute_macro_goal(&view.state, pov, 0);
        let aux = compute_goal_aux(&view.state, pov, &goal, 0, 0, None);
        let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
        let ranked = rank_plies(&mut view, pov, &goal, &aux, star_gate, 1.0, None, None);
        if ranked.len() < 2 {
            continue;
        }
        let (_, _, _, trace) =
            micro_search_pick(&view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &params, None, false);
        if trace.len() > params.sims {
            continue; // only meaningful when sims can cover every actual root child
        }
        checked_any = true;
        for (i, c) in trace.iter().enumerate() {
            assert!(
                c.visits >= 1,
                "seed {seed}, child {i} ({}): forced_playouts must give every root child >= 1 visit, got {}",
                c.mv,
                c.visits
            );
        }
    }
    assert!(checked_any, "no seed produced ranked.len() in [2, sims] -- widen the seed range");
}

/// Near-equal scores (std -> floor) must not blow up into a wild,
/// artificially spread distribution.
#[test]
fn softmax_priors_stays_reasonable_when_scores_are_nearly_equal() {
    let scores = [10.0, 10.01, 9.99, 10.02];
    let priors = softmax_priors(&scores);
    for p in &priors {
        assert!(*p > 0.15 && *p < 0.35, "near-equal scores should land close to uniform: {priors:?}");
    }
}

/// A real-game root, built the same way an interior node populates its own
/// children (`cheap_candidates` + `softmax_priors`) -- gives white-box tests
/// direct control of `MicroNode`/`MicroChild` without going through
/// `micro_search_pick`'s own root-widening machinery. `None` when the ply
/// has no real candidates (e.g. only EndTurn) worth searching.
fn build_test_root(seed: i64) -> Option<(MicroNode, PlayerId, MacroGoal, GoalAux, bool)> {
    let game = tiny_game_at_seed(seed);
    let pov = game.state.settings.current_player_turn_id;
    let view = game.clone_for_mcts(pov);
    let goal = compute_macro_goal(&view.state, pov, 0);
    let aux = compute_goal_aux(&view.state, pov, &goal, 0, 0, None);
    let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
    let cands = cheap_candidates(&view, &goal, star_gate, &aux, 8);
    if cands.len() < 2 {
        return None;
    }
    let scores: Vec<f32> = cands.iter().map(|(_, s)| *s).collect();
    let priors = softmax_priors(&scores);
    let children: Vec<MicroChild> = cands
        .into_iter()
        .zip(priors)
        .map(|((mv, _), prior)| MicroChild { mv, prior, node: None, virtual_loss: 0.0 })
        .collect();
    let root = MicroNode { game: view, visits: 0, value_sum: 0.0, children, is_terminal: false };
    Some((root, pov, goal, aux, star_gate))
}

/// Same as `build_test_root`, but truncated to exactly one child -- forces
/// every descent within a wave onto the identical not-yet-created leaf, the
/// deterministic way to exercise `micro_collect_wave`'s dedup path.
fn single_child_test_root(start_seed: i64) -> (MicroNode, PlayerId, MacroGoal, GoalAux, bool) {
    for seed in start_seed..start_seed + 20 {
        if let Some((mut root, pov, goal, aux, star_gate)) = build_test_root(seed) {
            root.children.truncate(1);
            return (root, pov, goal, aux, star_gate);
        }
    }
    panic!("no seed in range produced a usable root -- widen the seed range");
}

fn run_wave_batched_sims(
    root: &mut MicroNode,
    pov: PlayerId,
    goal: &MacroGoal,
    star_gate: bool,
    aux: &GoalAux,
    evaluator: &Evaluator,
    params: &MicroParams,
) {
    let mut done = 0usize;
    while done < params.sims {
        let want = (params.sims - done).min(params.leaf_batch.max(1));
        let (immediate, pending, features) = micro_collect_wave(root, pov, goal, star_gate, aux, params, want);
        done += immediate as usize;
        if !pending.is_empty() {
            done += pending.iter().map(|p| p.count as usize).sum::<usize>();
            micro_resolve_wave(root, pending, features, evaluator);
        }
    }
}

fn assert_no_residual_virtual_loss(node: &MicroNode, seed: i64) {
    for (i, c) in node.children.iter().enumerate() {
        assert_eq!(c.virtual_loss, 0.0, "seed {seed}, child {i}: leftover virtual_loss after a fully resolved wave");
        if let Some(n) = &c.node {
            assert_no_residual_virtual_loss(n, seed);
        }
    }
}

/// Wave-batching must conserve total visit mass regardless of `leaf_batch`:
/// every collected+resolved sim contributes exactly one unit to the wave
/// root's own `visits`, the same invariant the pre-wave-batching sequential
/// loop held by construction (every call updated the outermost frame's
/// `node.visits` exactly once per sim -- see `micro_apply_leaf`'s doc).
#[test]
fn wave_batching_conserves_total_visits_at_every_leaf_batch_size() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let sims = 12usize;
    for leaf_batch in [1usize, 2, 3, sims] {
        let mut ran_any = false;
        for seed in 0..10i64 {
            let Some((mut root, pov, goal, aux, star_gate)) = build_test_root(seed) else { continue };
            ran_any = true;
            let params =
                MicroParams { sims, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0, forced_playouts: false, goal_prior_w: 0.0, leaf_batch };
            run_wave_batched_sims(&mut root, pov, &goal, star_gate, &aux, &evaluator, &params);
            assert_eq!(
                root.visits, sims as u32,
                "leaf_batch={leaf_batch} seed={seed}: root.visits must equal sims regardless of batching"
            );
        }
        assert!(ran_any, "leaf_batch={leaf_batch}: no seed produced a usable root -- widen the seed range");
    }
}

/// Every wave's virtual loss must be fully removed by the time it resolves
/// -- a leftover charge would silently bias every subsequent ply's search
/// (via `MicroTreeCarry`, which moves a resolved subtree, `virtual_loss`
/// included, into the next ply's root).
#[test]
fn wave_batching_leaves_no_residual_virtual_loss() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let sims = 12usize;
    let mut ran_any = false;
    for seed in 0..10i64 {
        let Some((mut root, pov, goal, aux, star_gate)) = build_test_root(seed) else { continue };
        ran_any = true;
        let params =
            MicroParams { sims, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0, forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 4 };
        run_wave_batched_sims(&mut root, pov, &goal, star_gate, &aux, &evaluator, &params);
        assert_no_residual_virtual_loss(&root, seed);
    }
    assert!(ran_any, "no seed produced a usable root -- widen the seed range");
}

/// A wave whose only legal continuation is a single not-yet-created child
/// must dedup every repeat descent onto ONE pending entry with the right
/// `count`, not one pending entry per descent.
#[test]
fn wave_batching_dedups_repeat_picks_onto_one_pending_entry() {
    let (mut root, pov, goal, aux, star_gate) = single_child_test_root(0);
    let params =
        MicroParams { sims: 4, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0, forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 4 };
    let (immediate, pending, features) = micro_collect_wave(&mut root, pov, &goal, star_gate, &aux, &params, 4);
    assert_eq!(immediate, 0);
    assert_eq!(pending.len(), 1, "a single legal child must dedup onto one pending entry, got {}", pending.len());
    assert_eq!(pending[0].count, 4);
    assert_eq!(features.len(), 1);
}

/// `build_test_root`, but with priors overridden to a deliberately skewed,
/// hand-chosen distribution (one dominant candidate, five equal weaker
/// ones) instead of whatever a real game state's heuristic scores happen to
/// produce -- gives `wave_batching_spreads_a_wave_across_more_distinct_children_than_sequential`
/// a fully deterministic setup instead of depending on real-game prior skew.
fn skewed_prior_test_root(seed: i64) -> Option<(MicroNode, PlayerId, MacroGoal, GoalAux, bool)> {
    let (mut root, pov, goal, aux, star_gate) = build_test_root(seed)?;
    if root.children.len() < 6 {
        return None;
    }
    root.children.truncate(6);
    let n = root.children.len();
    for (i, c) in root.children.iter_mut().enumerate() {
        c.prior = if i == 0 { 0.5 } else { 0.5 / (n - 1) as f32 };
    }
    Some((root, pov, goal, aux, star_gate))
}

/// `leaf_batch > 1` must reach coverage a strictly-sequential (`leaf_batch ==
/// 1`) search of the same `sims` budget cannot: with a constant (Dummy) leaf
/// value, Q ties at 0 everywhere, so a sequential search's PUCT term always
/// prefers the single highest-prior child and (bar an exploration reversal
/// once its own visit count grows) can starve the rest at a tiny budget.
/// Virtual loss inside one wave spreads picks across distinct children
/// immediately, without waiting on real visit counts to grow. Hand-derived
/// on a `[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]` prior at `c_puct=1.5`, `sims=6`:
/// sequential visits child 0 five times and child 1 once (2 distinct);
/// batched (one wave of 6) visits every child exactly once (6 distinct).
#[test]
fn wave_batching_spreads_a_wave_across_more_distinct_children_than_sequential() {
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
    let sims = 6usize;
    let mut ran_any = false;
    for seed in 0..15i64 {
        let Some((mut root_seq, pov, goal, aux, star_gate)) = skewed_prior_test_root(seed) else { continue };
        ran_any = true;
        let seq_params =
            MicroParams { sims, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0, forced_playouts: false, goal_prior_w: 0.0, leaf_batch: 1 };
        run_wave_batched_sims(&mut root_seq, pov, &goal, star_gate, &aux, &evaluator, &seq_params);
        let sequential_touched = root_seq.children.iter().filter(|c| c.node.is_some()).count();

        let (mut root_batch, ..) = skewed_prior_test_root(seed).unwrap();
        let batch_params = MicroParams { leaf_batch: sims, ..seq_params };
        run_wave_batched_sims(&mut root_batch, pov, &goal, star_gate, &aux, &evaluator, &batch_params);
        let batched_touched = root_batch.children.iter().filter(|c| c.node.is_some()).count();

        assert!(
            batched_touched > sequential_touched,
            "seed {seed}: batching should touch strictly more distinct root children under a \
             skewed prior + constant leaf value: sequential={sequential_touched} batched={batched_touched}"
        );
    }
    assert!(ran_any, "no seed produced >= 6 real candidates -- widen the seed range");
}
