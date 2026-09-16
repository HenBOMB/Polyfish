use super::super::config::MacroParams;
use super::super::node::Node;
use super::MacroMctsSearch;
use super::wave::{DescendOutcome, PendingLeaf};
use crate::ai::macro_exec::TurnCounters;
use crate::ai::oracle_macro::{LaneState, MacroGoal, compute_macro_goal};
use crate::ai::search::macro_mcts::MacroMctsStats;
use crate::game::Game;
use crate::states::PlayerId;

impl<'a> MacroMctsSearch<'a> {
    /// One descent: walk UCT edges until an unexpanded edge, expand it
    /// then back the child's value up the path.
    fn simulate(&mut self, root_idx: usize, root_turn: i32, params: &MacroParams) {
        let (_immediate, pending, features) = self.collect_wave(root_idx, root_turn, params, 1);
        if !pending.is_empty() {
            self.resolve_wave(pending, features, root_turn, params);
        }
    }

    /// Back up `times` values along the path, removing `VIRTUAL_LOSS` charges to mirror descent.
    fn backup(&mut self, path: &[(usize, usize)], mut value: f32, times: u32) {
        // Traverse path backwards and backup the value
        for &(pidx, e) in path.iter().rev() {
            value = self.nodes[pidx].backup(e, value, times as f32);
        }

        self.stats.max_depth = self.stats.max_depth.max(path.len());
    }
    /// Walks tree with virtual-loss UCT; either resolves immediately (frozen/leaf/cached/sim),
    /// or queues a PendingLeaf if needing eval; duplicate edges in this wave just bump count.
    fn descend_once(
        &mut self,
        root_idx: usize,
        root_turn: i32,
        params: &MacroParams,
        dedup: &mut std::collections::HashMap<(usize, usize), usize>,
        pending: &mut Vec<PendingLeaf>,
        features: &mut Vec<crate::ai::features::RawFeatures>,
    ) -> (Vec<(usize, usize)>, DescendOutcome) {
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut idx = root_idx;
        loop {
            if let Some(v) = self.nodes[idx].frozen_value {
                return (path, DescendOutcome::Resolved(v));
            }
            if self.nodes[idx].candidates.is_empty() {
                let n = &self.nodes[idx];
                let v = leaf_value(
                    self.eval,
                    self.leaf,
                    &n.game.state,
                    n.player,
                    n.counters[seat(n.player)].tier3_bought,
                    n.from.as_ref().map(|(p, g)| (*p, g)),
                );
                return (path, DescendOutcome::Resolved(v));
            }
            let e = self.nodes[idx].select_edge();
            path.push((idx, e));
            let node = &mut self.nodes[idx];
            node.edge_virtual_loss[e] += VIRTUAL_LOSS;
            node.virtual_visits += VIRTUAL_LOSS;
            if let Some(child) = self.nodes[idx].children[e] {
                idx = child;
                continue;
            }
            // EXP_ELO_125 (piece 4): an edge frozen by the cheap NN
            // estimator on a previous visit -- reuse the cached value
            // instead of re-querying the eval server every time.
            if let Some(v) = self.nodes[idx].edge_frozen[e] {
                return (path, DescendOutcome::Resolved(v));
            }
            if let Some(&pi) = dedup.get(&(idx, e)) {
                pending[pi].count += 1;
                return (path, DescendOutcome::Deferred);
            }
            let depth = path.len();
            if let Some(feat) = self.try_freeze_rollout(idx, e, depth, params) {
                let offset = features.len();
                features.push(feat);
                dedup.insert((idx, e), pending.len());
                pending.push(PendingLeaf {
                    parent: idx,
                    edge: e,
                    path: path.clone(),
                    feat_offset: offset,
                    count: 1,
                });
                return (path, DescendOutcome::Deferred);
            }
            let child = self.expand_execute(idx, e, root_turn, params);
            let cn = &self.nodes[child];
            let v = cn.frozen_value.unwrap_or_else(|| {
                leaf_value(
                    self.eval,
                    self.leaf,
                    &cn.game.state,
                    cn.player,
                    cn.counters[seat(cn.player)].tier3_bought,
                    cn.from.as_ref().map(|(p, g)| (*p, g)),
                )
            });
            return (path, DescendOutcome::Resolved(v));
        }
    }

    /// Collects up to `budget` simulations' worth of work in one wave:
    /// repeated `descend_once` calls, backing up immediate resolutions in
    /// place and accumulating Path-B-eligible leaves (plus their feature
    /// rows) for one batched eval call. Path C (`expand_execute`) is
    /// untouched by wave-batching — it resolves synchronously inside
    /// `descend_once` exactly as before. Returns the count of immediately-
    /// resolved sims plus whatever's left pending for `resolve_wave`.
    fn collect_wave(
        &mut self,
        root_idx: usize,
        root_turn: i32,
        params: &MacroParams,
        budget: usize,
    ) -> (u32, Vec<PendingLeaf>, Vec<crate::ai::features::RawFeatures>) {
        let mut immediate = 0u32;
        let mut pending: Vec<PendingLeaf> = Vec::new();
        let mut features: Vec<crate::ai::features::RawFeatures> = Vec::new();
        let mut dedup: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        loop {
            let done = immediate as usize + pending.iter().map(|p| p.count as usize).sum::<usize>();
            if done >= budget {
                break;
            }
            let (path, outcome) = self.descend_once(
                root_idx,
                root_turn,
                params,
                &mut dedup,
                &mut pending,
                &mut features,
            );
            if let DescendOutcome::Resolved(v) = outcome {
                self.backup(&path, v, 1);
                immediate += 1;
            }
        }
        (immediate, pending, features)
    }

    /// The one batched `evaluator.evaluate()` call for a wave's pending
    /// Path-B leaves, then backs up each one's real result (`leaf.count`
    /// times, undoing exactly the virtual loss its `count` descents
    /// charged). A checkpoint that predates the rollout-value head returns
    /// `None` for every row — falls back to `expand_execute`'s full
    /// simulation per repeat pick, mirroring `descend_once`'s own
    /// Path-C-fallthrough for the non-wave-batched case.
    fn resolve_wave(
        &mut self,
        pending: Vec<PendingLeaf>,
        features: Vec<crate::ai::features::RawFeatures>,
        root_turn: i32,
        params: &MacroParams,
    ) {
        let results = self.eval.evaluate(features);
        for leaf in pending {
            match results
                .get(leaf.feat_offset)
                .and_then(|r| r.2.rollout_value)
            {
                Some(raw_value) => {
                    let v = -raw_value;
                    self.nodes[leaf.parent].edge_frozen[leaf.edge] = Some(v);
                    self.backup(&leaf.path, v, leaf.count);
                }
                None => {
                    for _ in 0..leaf.count {
                        let child = self.expand_execute(leaf.parent, leaf.edge, root_turn, params);
                        let cn = &self.nodes[child];
                        let v = cn.frozen_value.unwrap_or_else(|| {
                            leaf_value(
                                self.eval,
                                self.leaf,
                                &cn.game.state,
                                cn.player,
                                cn.counters[seat(cn.player)].tier3_bought,
                                cn.from.as_ref().map(|(p, g)| (*p, g)),
                            )
                        });
                        self.backup(&leaf.path, v, 1);
                    }
                }
            }
        }
    }

    /// EXP_ELO_125 (piece 4) gate + feature extraction only -- no eval call,
    /// no mutation. Returns the one feature row to price if `edge` is
    /// eligible for the cheap rollout estimator (depth-gated variance
    /// control; a candidate's goal is exactly as "uncommitted" here as at
    /// the root, so root-adjacent edges always get full `execute_turn`
    /// simulation instead). `None` when ineligible -- caller falls through
    /// to `expand_execute`'s full simulation, same as an eval/feature
    /// failure once the caller actually queries the evaluator.
    fn try_freeze_rollout(
        &self,
        parent: usize,
        edge: usize,
        depth: usize,
        params: &MacroParams,
    ) -> Option<crate::ai::features::RawFeatures> {
        if !(params.rollout_nn_w > 0.0 && depth > params.rollout_nn_min_depth) {
            return None;
        }
        let p = &self.nodes[parent];
        crate::ai::features::state_to_cpu_features_goal(
            &p.game.state,
            p.player,
            None,
            Some(&p.candidates[edge]),
        )
        .ok()
    }

    /// Full `execute_turn_net_greedy` simulation of `edge` off `parent`'s
    /// state, creating and linking in a real child `Node`. This is Path C
    /// (untouched by wave-batching) -- runs whenever `try_freeze_rollout`
    /// didn't apply or its eval call came back empty.
    fn expand_execute(
        &mut self,
        parent: usize,
        edge: usize,
        root_turn: i32,
        params: &MacroParams,
    ) -> usize {
        let (mut game, player, mut counters, mut lane_states, goal, parent_from, mut branch_belief) = {
            let p = &self.nodes[parent];
            (
                p.game.clone(),
                p.player,
                p.counters,
                p.lane_states.clone(),
                p.candidates[edge].clone(),
                p.from.clone(),
                p.branch_belief
                    .as_ref()
                    .map(macro_search::belief::BranchBelief::fork),
            )
        };
        // EXP_ELO_170/171: `parent.from` already records (who acted, what
        // they committed) for the edge that produced `parent`. `Node.player`
        // strictly alternates by construction (`other(player)`, independent
        // of game state — see `Node`'s own doc comment), so the child about
        // to be created here (`other(player)`) is exactly the player named
        // in `parent_from`, whenever `parent_from` is `Some` -- i.e. this IS
        // "what did the child's own player commit to two plies back", for
        // free, with zero new persistent state or parent pointer.
        let child_own_last_goal: Option<&MacroGoal> = if params.tree_continuation {
            parent_from.as_ref().map(|(_, g)| g)
        } else {
            None
        };
        let s = seat(player);
        // EXP_ELO_036b: pre-move potential of THIS edge's directive, with one
        // GoalAux for both sides of the difference (the executor's
        // edge_snapshot pattern) — mixing directives or auxes across the
        // difference would mint reward on switches, not approach.
        let shape_pre = if params.shape_w != 0.0 && player == self.pov {
            let aux = crate::ai::oracle_macro::compute_goal_aux(
                &game.state,
                player,
                &goal,
                counters[s].techs_bought,
                counters[s].tier3_bought,
                Some(&lane_states[s]),
            );
            Some((
                crate::ai::reward::goal_potential(&game.state, player, &goal, Some(&aux)),
                aux,
            ))
        } else {
            None
        };
        // An executor anomaly leaves the state where it stopped; the node is
        // still scoreable, so treat it like any other boundary.
        //
        // Verdi, Sep 7 2026: rank_plies is gone from macro-mcts's own
        // root-candidate rollouts, unconditionally -- no env-var, no CPU
        // fallback. `execute_turn`/`execute_turn_recorded` (the old
        // rank_plies-driven path) stay untouched as functions since they're
        // still shared by `belief/mod.rs`'s fog-of-war rollouts and
        // `MacroLookaheadAgent::replan`, but this call site no longer uses
        // them at all.
        let _ = crate::ai::search::net_root::execute_turn_net_greedy(
            &mut game,
            player,
            &goal,
            &mut lane_states[s],
            &mut counters[s],
            self.eval,
        );
        if let Some(branch) = &mut branch_belief {
            let (stats, observations) = branch.materialize_child(&mut game);
            self.stats.branch_capital_materializations += stats.capital as u32;
            self.stats.branch_village_materializations += stats.village as u32;
            self.stats.branch_revealed_tiles += observations.revealed;
            self.stats.branch_resource_reveals += observations.resource_revealed;
            self.stats.branch_capital_refutations += observations.capital_refuted;
            self.stats.branch_capital_confirmations += observations.capital_confirmed;
            self.stats.branch_village_confirmations += observations.village_confirmed;
            self.stats.branch_unit_materializations += observations.unit_materialized;
        }
        let shape = match &shape_pre {
            Some((pre, aux)) => {
                let post = crate::ai::reward::goal_potential(&game.state, player, &goal, Some(aux));
                params.shape_w * (post - pre)
            }
            None => 0.0,
        };
        let leaf = self.leaf;
        let eval = self.eval;
        // The child's depth-capped value (computed inside `Node::new`) gets the
        // same edge context every other leaf read of this node will get.
        let from = (player, goal.clone());
        let leaf_fn = move |s: &crate::states::GameState, p: PlayerId, t3: u32| {
            leaf_value(eval, leaf, s, p, t3, Some((from.0, &from.1)))
        };
        let child = Node::new(
            game,
            other(player),
            counters,
            lane_states,
            root_turn,
            params.k,
            Some((player, goal.clone())),
            &leaf_fn,
            child_own_last_goal,
            branch_belief,
        );
        if child.branch_belief.is_some() && child.player == self.pov {
            self.stats.branch_belief_candidate_nodes += 1;
            self.stats.branch_belief_candidates += child.candidates.len() as u32;
        }
        let child_idx = self.nodes.len();
        self.nodes.push(child);
        self.nodes[parent].children[edge] = Some(child_idx);
        self.nodes[parent].edge_shape[edge] = shape;
        if let Some(path) = macro_rollout_trace_path() {
            let child_state = &self.nodes[child_idx].game.state;
            dump_rollout_node(
                path,
                child_idx,
                parent,
                child_state.settings.turn,
                player,
                &goal,
                self.pov,
                child_state,
            );
        }
        child_idx
    }
}

/// Returns leaf value from `player`'s POV. Net mode paints the base goal and reads win_value only.
/// Falls back to heuristics on any failure. Committed directive only known at decision, else approximate.
pub(super) fn compute_leaf_value(
    eval: &crate::ai::eval_server::Evaluator,
    leaf: crate::ai::macro_agent::MacroLeaf,
    state: &crate::states::GameState,
    player: PlayerId,
    tier3: u32,
    // The edge that produced this state: (the player who executed the turn,
    // the directive they executed). `None` at the root. Only `NetAsymPaint`
    // reads it — for that player the committed directive is KNOWN, which is
    // the one painting inference can align with training for free.
    from: Option<(PlayerId, &MacroGoal)>,
) -> f32 {
    use crate::ai::macro_agent::MacroLeaf;
    let aligned = leaf == MacroLeaf::NetAsymPaint;
    let net = |p: PlayerId| -> Option<f32> {
        let scripted;
        let goal = match from {
            Some((mover, g)) if aligned && mover == p => g,
            _ => {
                scripted = compute_macro_goal(state, p, tier3);
                &scripted
            }
        };
        crate::ai::features::state_to_cpu_features_goal(state, p, None, Some(goal))
            .ok()
            .and_then(|f| eval.evaluate(vec![f]).first().map(|r| r.0))
    };
    match leaf {
        MacroLeaf::Net => {
            if let Some(v) = net(player) {
                return v;
            }
        }
        // Both perspectives, halved: makes the zero-sum identity the negamax
        // backup assumes hold BY CONSTRUCTION, at two forwards per leaf.
        MacroLeaf::NetAsym | MacroLeaf::NetAsymPaint => {
            let opp: PlayerId = if player == 1 { 2 } else { 1 };
            if let (Some(a), Some(b)) = (net(player), net(opp)) {
                return (a - b) / 2.0;
            }
        }
        MacroLeaf::Heuristic => {}
        MacroLeaf::HeuristicV2 => return crate::ai::evaluate_state_v2(state, player),
    }
    crate::ai::evaluate_state(state, player)
}
