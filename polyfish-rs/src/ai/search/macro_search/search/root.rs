use super::super::config::MacroParams;
use super::super::node::Node;
use super::MacroMctsSearch;
use crate::ai::macro_exec::TurnCounters;
use crate::ai::oracle_macro::{LaneState, MacroGoal};
use crate::ai::search::macro_mcts::MacroMctsStats;
use crate::game::Game;
use crate::states::PlayerId;

impl<'a> MacroMctsSearch<'a> {
    /// Run `sims` simulations from `root_game` (the acting player's fogged
    /// view) and return the winning root directive index. Root candidate 0
    /// must be the committed script base; ties break toward it.
    pub(crate) fn run(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
    ) -> (usize, MacroMctsStats) {
        Self::run_with_branch_belief(
            root_game,
            pov,
            root_candidates,
            own_counters,
            own_lane_state,
            params,
            evaluator,
            None,
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_with_belief_world(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
        belief: crate::ai::belief::BeliefState,
    ) -> (usize, MacroMctsStats) {
        let max_particles = params.map_particles.max(1);
        let min_sims_per_particle = root_candidates.len().saturating_add(1).max(1);
        let particle_count = max_particles.min((params.sims.max(1) / min_sims_per_particle).max(1));
        if !params.tree_belief_materialization || particle_count == 1 {
            return Self::run_with_branch_belief(
                root_game,
                pov,
                root_candidates,
                own_counters,
                own_lane_state,
                params,
                evaluator,
                Some(belief),
                |_| {},
            );
        }

        let total_sims = params.sims.max(1);
        let base_sims = total_sims / particle_count;
        let extra_sims = total_sims % particle_count;
        let mut total = MacroMctsStats::default();
        let mut q_rows = Vec::with_capacity(particle_count);
        for particle in 0..particle_count {
            let mut particle_params = *params;
            particle_params.sims = base_sims + usize::from(particle < extra_sims);
            particle_params.map_particles = 1;
            let mut q_row = Vec::new();
            let (_, sample) = Self::run_with_branch_belief_particle(
                root_game,
                pov,
                root_candidates.clone(),
                own_counters,
                own_lane_state,
                &particle_params,
                evaluator,
                Some(belief.clone()),
                particle as i32 + 1,
                |search| {
                    let root = &search.nodes[0];
                    q_row = root.edge_q_values();
                },
            );
            merge_particle_stats(&mut total, &sample);
            q_rows.push(q_row);
        }

        let means = mean_particle_q(&q_rows);
        let best = best_particle_q(&means);
        total.root_q = means.get(best).and_then(|q| q.map(|q| q.clamp(-1.0, 1.0)));
        let backed: Vec<f32> = means.iter().filter_map(|q| *q).collect();
        total.root_q_spread = if backed.len() > 1 {
            Some(
                backed.iter().copied().fold(f32::MIN, f32::max)
                    - backed.iter().copied().fold(f32::MAX, f32::min),
            )
        } else {
            None
        };
        let visits: f32 = total.root_visits.iter().sum();
        total.root_visit_max_share = if visits > 0.0 {
            total.root_visits.iter().copied().fold(0.0, f32::max) / visits
        } else {
            0.0
        };
        return (best, total);
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_with(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
        inspect: impl FnOnce(&MacroMctsSearch),
    ) -> (usize, MacroMctsStats) {
        Self::run_with_branch_belief(
            root_game,
            pov,
            root_candidates,
            own_counters,
            own_lane_state,
            params,
            evaluator,
            None,
            inspect,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_with_branch_belief(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
        belief: Option<crate::ai::belief::BeliefState>,
        inspect: impl FnOnce(&MacroMctsSearch),
    ) -> (usize, MacroMctsStats) {
        Self::run_with_branch_belief_particle(
            root_game,
            pov,
            root_candidates,
            own_counters,
            own_lane_state,
            params,
            evaluator,
            belief,
            0,
            inspect,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_with_branch_belief_particle(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
        belief: Option<crate::ai::belief::BeliefState>,
        particle_salt: i32,
        inspect: impl FnOnce(&MacroMctsSearch),
    ) -> (usize, MacroMctsStats) {
        let mut sanitized_root;
        let root_game = if params.tree_belief_materialization && belief.is_some() {
            sanitized_root = root_game.clone();
            r#macro::belief::BranchBelief::sanitize_fog_view(&mut sanitized_root.state, pov);
            &sanitized_root
        } else {
            root_game
        };
        debug_assert_eq!(
            root_game.state.tribes.len(),
            2,
            "macro MCTS is 2-player only"
        );
        let root_turn = root_game.state.settings.turn;
        let mut counters = [TurnCounters::default(); 2];
        counters[seat(pov)] = own_counters;
        counters[seat(other(pov))] = derive_counters(&root_game.state, other(pov));
        let mut lane_states: [LaneState; 2] = Default::default();
        lane_states[seat(pov)] = own_lane_state.clone();

        let leaf = params.leaf;
        // The root has no incoming edge, so no committed directive is known
        // for it — every leaf read there falls back to the scripted painting.
        let leaf_fn = move |s: &crate::states::GameState, p: PlayerId, t3: u32| {
            leaf_value(evaluator, leaf, s, p, t3, None)
        };
        // `own_last_goal: None` -- the root has no in-tree ancestor to
        // continue from, and its candidate list is unconditionally
        // overwritten by `root.candidates = root_candidates` immediately
        // below, so computing a real continuation candidate here would be
        // pure waste even if it had one.
        let mut root = Node::new(
            root_game.clone(),
            pov,
            counters,
            lane_states,
            root_turn,
            params.k,
            None,
            &leaf_fn,
            None,
            None,
        );
        root.candidates = root_candidates;
        let has_branch_belief = params.tree_belief_materialization && belief.is_some();
        if has_branch_belief {
            root.branch_belief = belief.map(|belief| {
                macro_search::belief::BranchBelief::new_with_salt(
                    belief,
                    &root.game.state,
                    particle_salt,
                )
            });
        }

        // War-room item 3 + EXP_ELO_165: one shared eval call for both root
        // PUCT-prior injection and net-candidate synthesis — either wanting
        // it is enough to pay for it. EXP_ELO_066: the first version of this
        // painted the scripted base goal here, matching `leaf_value`'s
        // fallback convention — but that convention was WRONG for this
        // specific head. Every existing macro_stance/macro_order training
        // row is painted with the search's own COMMITTED (post-search,
        // already-chosen) goal, so the head partly learned to echo its own
        // input rather than predict blind, and painting anything at the
        // root (which by definition has no committed goal yet) fed it an
        // out-of-distribution input — measured as a real −18.75pp
        // regression, not a weak prior. The fix is training-side (repaint
        // those label rows `None` = goal-blind) and this call must paint
        // the SAME way for the two to agree once retrained. Both weights at
        // 0.0 skips the eval call entirely, byte-identical to plain UCT.
        let mut net_eval: Option<(Vec<f32>, Vec<f32>)> = None;
        if (params.root_prior_w > 0.0 || params.net_candidates_w > 0.0)
            && !root.candidates.is_empty()
        {
            if let Ok(feats) =
                crate::ai::features::state_to_cpu_features_goal(&root_game.state, pov, None, None)
            {
                if let Some(result) = evaluator.evaluate(vec![feats]).into_iter().next() {
                    if let (Some(stance), Some(order)) =
                        (result.2.macro_stance.clone(), result.2.macro_order.clone())
                    {
                        net_eval = Some((stance, order));
                    }
                }
            }
        }

        // EXP_ELO_165: synthesize one new candidate from the net's own
        // (stance, order) prediction and add it to the ballot BEFORE sizing
        // the per-edge arrays below, so it's a real, fully-searched root
        // edge like every scripted candidate — not a post-hoc addition.
        let mut net_candidate_index: Option<usize> = None;
        if params.net_candidates_w > 0.0 {
            if let Some((stance, order)) = &net_eval {
                let map_size = root_game.state.settings.size as usize;
                if let Some(candidate) =
                    net_proposed_candidate(stance, order, &root.candidates, map_size)
                {
                    net_candidate_index = Some(root.candidates.len());
                    root.candidates.push(candidate);
                }
            }
        }

        let n = root.candidates.len();
        root.children = vec![None; n];
        root.edge_visits = vec![0.0; n];
        root.edge_values = vec![0.0; n];
        root.edge_shape = vec![0.0; n];
        root.edge_frozen = vec![None; n];
        root.edge_virtual_loss = vec![0.0; n];
        root.virtual_visits = 0.0;

        if params.root_prior_w > 0.0 && n > 0 {
            if let Some((stance, order)) = &net_eval {
                let map_size = root_game.state.settings.size as usize;
                let prior = decode_macro_prior(stance, order, &root.candidates, map_size);
                root.edge_prior = prior.iter().map(|p| p * params.root_prior_w).collect();
            }
        }

        let mut search = MacroMctsSearch {
            nodes: vec![root],
            pov,
            eval: evaluator,
            leaf,
            stats: MacroMctsStats {
                net_candidate_index,
                effective_map_particles: u32::from(has_branch_belief),
                ..MacroMctsStats::default()
            },
        };
        // leaf_batch=1 (the default) makes this identical to a plain
        // `for _ in 0..sims { search.simulate(...) }` loop, one wave per
        // sim. leaf_batch>1 collects that many Path-B leaves per wave
        // before making one batched eval call -- see `collect_wave`'s doc.
        let total_sims = params.sims.max(1);
        let batch = params.leaf_batch.max(1);
        let mut done = 0usize;
        while done < total_sims {
            let want = (total_sims - done).min(batch);
            let (immediate, pending, features) = search.collect_wave(0, root_turn, params, want);
            done += immediate as usize;
            if !pending.is_empty() {
                done += pending.iter().map(|p| p.count as usize).sum::<usize>();
                search.resolve_wave(pending, features, root_turn, params);
            }
        }
        inspect(&search);

        let root = &search.nodes[0];
        let mut best = 0;
        for i in 1..root.edge_visits.len() {
            if root.edge_visits[i] > root.edge_visits[best] {
                best = i;
            }
        }
        search.stats.nodes = search.nodes.len();
        search.stats.root_visit_max_share = if root.visits > 0.0 {
            root.edge_visits.iter().cloned().fold(0.0, f32::max) / root.visits
        } else {
            0.0
        };
        search.stats.root_q = if root.edge_visits[best] > 0.0 {
            Some((root.edge_values[best] / root.edge_visits[best]).clamp(-1.0, 1.0))
        } else {
            None
        };
        let backed: Vec<f32> = (0..root.candidates.len())
            .filter(|&i| root.edge_visits[i] > 0.0)
            .map(|i| root.edge_values[i] / root.edge_visits[i])
            .collect();
        search.stats.root_q_spread = if backed.len() > 1 {
            let hi = backed.iter().cloned().fold(f32::MIN, f32::max);
            let lo = backed.iter().cloned().fold(f32::MAX, f32::min);
            Some(hi - lo)
        } else {
            None
        };
        search.stats.root_candidates = root.candidates.clone();
        search.stats.root_visits = root.edge_visits.clone();
        (best, search.stats)
    }

    /// `run` plus a per-edge root dump on stdout (smoke instrumentation only).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_probed(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
    ) -> (usize, MacroMctsStats) {
        let cands_dbg: Vec<String> = root_candidates
            .iter()
            .map(|c| format!("{:?}/{}ord", c.stance, c.orders.len()))
            .collect();
        let (best, stats) = Self::run_with(
            root_game,
            pov,
            root_candidates,
            own_counters,
            own_lane_state,
            params,
            evaluator,
            |s| {
                let root = &s.nodes[0];
                for i in 0..root.candidates.len() {
                    let q = if root.edge_visits[i] > 0.0 {
                        root.edge_values[i] / root.edge_visits[i]
                    } else {
                        f32::NAN
                    };
                    println!(
                        "    edge {i} [{}]: visits={} q={q:+.4} shape={:+.4}",
                        cands_dbg[i], root.edge_visits[i], root.edge_shape[i]
                    );
                }
            },
        );
        (best, stats)
    }
}
