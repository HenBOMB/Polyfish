use super::super::belief::BranchBelief;
use super::super::config::MacroParams;
use super::super::node::Node;
use super::MacroMctsSearch;
use super::edge_shaper::{EdgeShaper, EdgeShaperSnapshot};
use super::wave::{DescendOutcome, PendingLeaf};
use crate::ai::features::RawFeatures;
use crate::ai::macro_exec::TurnCounters;
use crate::ai::oracle_macro::{LaneState, MacroGoal, compute_macro_goal};
use crate::ai::search::macro_mcts::{dump_rollout_node, macro_rollout_trace_path};
use crate::game::Game;
use crate::states::PlayerId;
use crate::utils::converter::{opponent_player_id, player_id_to_usize};
use std::collections::HashMap;

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
        dedup: &mut HashMap<(usize, usize), usize>,
        pending: &mut Vec<PendingLeaf>,
        features: &mut Vec<RawFeatures>,
    ) -> (Vec<(usize, usize)>, DescendOutcome) {
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut idx = root_idx;
        loop {
            let node = &mut self.nodes[idx];
            if let Some(value) = node.frozen_value() {
                return (path, DescendOutcome::Resolved(value)); // cache hit
            }

            if node.candidates().is_empty() {
                let value = compute_leaf_value(
                    self.eval,
                    self.leaf,
                    node.game_state(),
                    node.player(),
                    node.tier_3_bought(),
                    node.from(),
                );
                return (path, DescendOutcome::Resolved(value));
            }

            let edge = node.prepare_next_for_descend();
            path.push((idx, edge));
            if let Some(child_idx) = node.get_child_by_edge(edge) {
                idx = child_idx;
                continue;
            }

            // Reuse cached value if available.
            if let Some(v) = node.get_edge_frozen(edge) {
                return (path, DescendOutcome::Resolved(v));
            }

            // Skip if already queued.
            if let Some(&pi) = dedup.get(&(idx, edge)) {
                pending[pi].count += 1;
                return (path, DescendOutcome::Deferred);
            }

            let depth = path.len();
            if let Some(feat) = self.try_freeze_rollout(idx, edge, depth, params) {
                let offset = features.len();
                features.push(feat);
                dedup.insert((idx, edge), pending.len());
                pending.push(PendingLeaf {
                    parent: idx,
                    edge: edge,
                    path: path.clone(),
                    feat_offset: offset,
                    count: 1,
                });
                return (path, DescendOutcome::Deferred);
            }

            // Expand the edge.
            let child_idx = self.expand_execute(idx, edge, root_turn, params);
            let child_node = &mut self.nodes[child_idx];
            let value = child_node.frozen_value().unwrap_or_else(|| {
                compute_leaf_value(
                    self.eval,
                    self.leaf,
                    child_node.game_state(),
                    child_node.player(),
                    child_node.tier_3_bought(),
                    child_node.from(),
                )
            });
            return (path, DescendOutcome::Resolved(value));
        }
    }

    /// Runs up to `budget` descents, batching Path-B leaves for eval and backing up resolved sims,
    /// skipping Path C which always resolves inline.
    /// Returns count of resolved sims plus pending leaves and features.
    fn collect_wave(
        &mut self,
        root_idx: usize,
        root_turn: i32,
        params: &MacroParams,
        budget: usize,
    ) -> (u32, Vec<PendingLeaf>, Vec<RawFeatures>) {
        let mut immediate = 0u32;
        let mut pending: Vec<PendingLeaf> = Vec::new();
        let mut features: Vec<RawFeatures> = Vec::new();
        let mut dedup: HashMap<(usize, usize), usize> = HashMap::new();
        loop {
            let done = immediate + pending.iter().map(|p| p.count).sum::<u32>();
            if done >= budget as u32 {
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

    /// Batched eval for all pending Path-B leaves; backs up each result or
    /// falls back to full sim if the value head is missing.
    fn resolve_wave(
        &mut self,
        pending: Vec<PendingLeaf>,
        features: Vec<RawFeatures>,
        root_turn: i32,
        params: &MacroParams,
    ) {
        // neural net eval for all pending leaves.
        let results = self.eval.evaluate(features);

        // Back up value head or run full sims when missing.
        for leaf in pending {
            match results
                .get(leaf.feat_offset)
                .and_then(|r| r.2.rollout_value)
            {
                Some(raw_value) => {
                    // Backups value head
                    let v = -raw_value;
                    self.nodes[leaf.parent].set_edge_frozen(leaf.edge, v);
                    self.backup(&leaf.path, v, leaf.count);
                }
                None => {
                    // Run full sim for each repeat pick.
                    for _ in 0..leaf.count {
                        let idx = self.expand_execute(leaf.parent, leaf.edge, root_turn, params);
                        let child_node = &mut self.nodes[idx];
                        let value = child_node.frozen_value().unwrap_or_else(|| {
                            compute_leaf_value(
                                self.eval,
                                self.leaf,
                                child_node.game_state(),
                                child_node.player(),
                                child_node.tier_3_bought(),
                                child_node.from(),
                            )
                        });
                        self.backup(&leaf.path, value, 1);
                    }
                }
            }
        }
    }

    /// Returns feature row for rollout value if eligible; else None.
    /// No eval/mutation; depth-gated. Root-adjacent edges always simulate.
    /// Falls through to expand_execute when ineligible or failed.
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
        let node = &self.nodes[parent];
        crate::ai::features::state_to_cpu_features_goal(
            &node.game_state(),
            node.player(),
            None,
            Some(node.candidate_for_edge(edge)),
        )
        .ok()
    }

    fn snapshot_parent_edge(&self, node_idx: usize, edge: usize) -> EdgeShaperSnapshot {
        let node = &self.nodes[node_idx];
        EdgeShaperSnapshot {
            game: node.game().clone(),
            player: node.player(),
            counters: node.counters().clone(),
            lane_states: node.lane_states().clone(),
            goal: node.candidate_for_edge(edge).clone(),
            parent_from: node.from().map(|(p, g)| (p, g.clone())),
            branch_belief: node.branch_belief().map(BranchBelief::fork),
        }
    }

    /// Executes the turn and materializes the child.
    fn execute_and_materialize(
        &mut self,
        game: &mut Game,
        player: PlayerId,
        goal: &MacroGoal,
        lane_states: &mut [LaneState; 2],
        counters: &mut [TurnCounters; 2],
        branch_belief: &mut Option<BranchBelief>,
    ) {
        // Node is always scoreable here; execute_turn_net_greedy is now used unconditionally.
        // Legacy rank_plies paths remain only for belief rollouts and MacroLookaheadAgent::replan.
        let seat_idx = player_id_to_usize(player);
        let _ = crate::ai::search::net_root::execute_turn_net_greedy(
            game,
            player,
            &goal,
            &mut lane_states[seat_idx],
            &mut counters[seat_idx],
            self.eval,
        );
        if let Some(branch) = branch_belief {
            let (stats, observations) = branch.materialize_child(game);
            self.stats.branch_capital_materializations += stats.capital as u32;
            self.stats.branch_village_materializations += stats.village as u32;
            self.stats.branch_revealed_tiles += observations.revealed;
            self.stats.branch_resource_reveals += observations.resource_revealed;
            self.stats.branch_capital_refutations += observations.capital_refuted;
            self.stats.branch_capital_confirmations += observations.capital_confirmed;
            self.stats.branch_village_confirmations += observations.village_confirmed;
            self.stats.branch_unit_materializations += observations.unit_materialized;
        }
    }

    /// Runs `execute_turn_net_greedy` for `edge` off `parent`, creating a new child `Node`.
    fn expand_execute(
        &mut self,
        parent: usize,
        edge: usize,
        root_turn: i32,
        params: &MacroParams,
    ) -> usize {
        let mut snapshot = self.snapshot_parent_edge(parent, edge);

        let shaper = EdgeShaper::start(&params, &snapshot, self.pov);
        self.execute_and_materialize(
            &mut snapshot.game,
            snapshot.player,
            &snapshot.goal,
            &mut snapshot.lane_states,
            &mut snapshot.counters,
            &mut snapshot.branch_belief,
        );
        let shape = shaper.finish(&snapshot.game.state);
        self.build_and_link_child(parent, edge, snapshot, root_turn, params, shape)
    }

    /// Builds the child `Node` from the post-execution snapshot, links it
    /// into the tree under `parent`/`edge` with `shape`, folds the
    /// belief-candidate stats, and dumps a trace row if tracing is on.
    fn build_and_link_child(
        &mut self,
        parent: usize,
        edge: usize,
        snapshot: EdgeShaperSnapshot,
        root_turn: i32,
        params: &MacroParams,
        shape: f32,
    ) -> usize {
        // EXP_ELO_170/171: `parent.from` gives the child's own player's last
        // committed goal (two plies back), with no extra state — strict
        // alternation makes `Node.player` always `other(parent.player)`.
        let child_own_last_goal: Option<&MacroGoal> = if params.tree_continuation {
            snapshot.parent_from.as_ref().map(|(_, g)| g)
        } else {
            None
        };

        let (leaf, eval) = (self.leaf, self.eval);
        // The child's depth-capped value (computed inside `Node::new`) gets the
        // same edge context every other leaf read of this node will get.
        let from = (snapshot.player, snapshot.goal.clone());
        let leaf_fn = move |s: &crate::states::GameState, p: PlayerId, t3: u32| {
            compute_leaf_value(eval, leaf, s, p, t3, Some((from.0, &from.1)))
        };

        let child = Node::new(
            snapshot.game,
            opponent_player_id(snapshot.player),
            snapshot.counters,
            snapshot.lane_states,
            root_turn,
            params.k,
            Some((snapshot.player, snapshot.goal.clone())),
            &leaf_fn,
            child_own_last_goal,
            snapshot.branch_belief,
        );

        if child.branch_belief().is_some() && child.player() == self.pov {
            self.stats.branch_belief_candidate_nodes += 1;
            self.stats.branch_belief_candidates += child.candidates().len() as u32;
        }

        let child_idx = self.nodes.len();
        self.nodes.push(child);
        {
            let parent_node = &mut self.nodes[parent];
            parent_node.set_child_by_edge(edge, Some(child_idx));
            parent_node.set_edge_shape(edge, shape);
        }

        // For debugging purposes
        if let Some(path) = macro_rollout_trace_path() {
            let child_state = &self.nodes[child_idx].game_state();
            dump_rollout_node(
                path,
                child_idx,
                parent,
                child_state.settings.turn,
                snapshot.player,
                &snapshot.goal,
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
    tier3_bought_count: u32,
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
                scripted = compute_macro_goal(state, p, tier3_bought_count);
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
