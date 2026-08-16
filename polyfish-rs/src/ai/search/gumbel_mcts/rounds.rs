//! Sequential-Halving round mechanics and turn-crossing (Aug 2026 taxonomy
//! split out of gumbel_mcts.rs to keep every file under ~1000 lines):
//! `run_search`'s per-round driver loop, the score used to re-rank survivors,
//! one round-robin batch of leaf collection, and crossing an EndTurn edge
//! (with or without giving the opponent a real ghost turn). A second
//! `impl<'a> GumbelMctsAgent<'a>` block — see trace.rs's note.

use crate::ai::brain::max_turns_ahead;
use crate::ai::gumbel_qtransform::{self, sequence_of_considered_visits};
use crate::ai::mcts_common::{self, backpropagate_return_with_rewards, get_node_by_path_mut};
use crate::ai::reward;
use crate::game::Game;
use crate::moves::EndTurnMove;
use crate::types::MoveType;

use super::{GumbelLeaf, GumbelMctsAgent, GumbelNode};

impl<'a> GumbelMctsAgent<'a> {
    /// Sequential Halving over `in_cut`. Each round considers the top
    /// `round_considered` candidates (by current score) and gives each
    /// exactly `visits_per_candidate` new visits via round-robin batching.
    pub(super) fn run_search(
        &self,
        game: &mut Game,
        root: &mut GumbelNode,
        in_cut: &mut Vec<usize>,
        start_turn: i32,
    ) {
        if in_cut.is_empty() {
            return;
        }
        let max_considered = in_cut.len();
        let table = sequence_of_considered_visits(max_considered, self.iterations);
        for (round_idx, round_considered, visits_per_candidate) in table {
            let round_considered = round_considered.min(in_cut.len());
            if round_considered <= 1 {
                break;
            }
            // Round 0 keeps the initial (logit + gumbel) order; later rounds
            // re-rank survivors by current score so the best
            // `round_considered` stay in play.
            if round_idx > 0 {
                self.rerank_in_cut(root, in_cut);
            }
            self.run_round_robin_round(
                game,
                root,
                in_cut,
                round_considered,
                visits_per_candidate,
                start_turn,
            );
            self.record_round_snapshot(root, in_cut, round_idx, round_considered, visits_per_candidate);
        }
    }
    /// Re-sort `in_cut` by current score `gumbel + logit + sigma(completed-Q)`
    /// (descending), so the strongest candidates occupy the front positions.
    pub(super) fn rerank_in_cut(&self, root: &GumbelNode, in_cut: &mut Vec<usize>) {
        let sigma_q = self.sigma_q_for(root, in_cut);
        let mut scored: Vec<(usize, f32)> = in_cut
            .iter()
            .enumerate()
            .map(|(pos, &i)| {
                (
                    i,
                    root.children[i].gumbel + root.children[i].logit + sigma_q[pos],
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        *in_cut = scored.into_iter().map(|(i, _)| i).collect();
    }
    /// sigma(completed-Q) over the candidates referenced by `child_indices`,
    /// returned as a `Vec<f32>` aligned with `child_indices` (i.e. entry `pos`
    /// corresponds to `root.children[child_indices[pos]]`), scaled by
    /// `tree_q_weight` so trust-gating applies to every selection consumer.
    pub(super) fn sigma_q_for(&self, root: &GumbelNode, child_indices: &[usize]) -> Vec<f32> {
        let q: Vec<f32> = child_indices.iter().map(|&i| root.children[i].q_value()).collect();
        let visits: Vec<f32> = child_indices
            .iter()
            .map(|&i| root.children[i].visits)
            .collect();
        let priors: Vec<f32> = child_indices
            .iter()
            .map(|&i| root.children[i].logit)
            .collect();
        let mut sq =
            gumbel_qtransform::sigma_completed_q(root.own_value, &priors, &q, &visits, true);
        for s in &mut sq {
            *s *= self.tree_q_weight;
        }
        sq
    }
    /// Run one Sequential-Halving round: give each of the first
    /// `round_considered` candidates exactly `visits_per_candidate` new
    /// visits, collected round-robin and batched into one NN call per wave.
    pub(super) fn run_round_robin_round(
        &self,
        game: &mut Game,
        root: &mut GumbelNode,
        in_cut: &[usize],
        round_considered: usize,
        visits_per_candidate: usize,
        start_turn: i32,
    ) {
        let turn_horizon =
            start_turn + max_turns_ahead(start_turn, game.state.settings.max_turns);

        let total_needed = round_considered * visits_per_candidate;
        let mut collected_per_candidate = vec![0usize; round_considered];
        let mut total_collected = 0;

        while total_collected < total_needed {
            let mut leaves: Vec<GumbelLeaf> = Vec::with_capacity(self.batch_size);

            // One wave: cycle through the in-play candidates in order, taking
            // one leaf from each (that hasn't hit its quota) until the batch
            // is full or no candidate can make progress this pass.
            'wave: loop {
                let mut made_progress = false;
                for cand in 0..round_considered {
                    if leaves.len() >= self.batch_size {
                        break 'wave;
                    }
                    if collected_per_candidate[cand] >= visits_per_candidate {
                        continue;
                    }
                    match self.select_and_extract_leaf_under_candidate(
                        root,
                        in_cut[cand],
                        game,
                        turn_horizon,
                    ) {
                        Some(leaf) => {
                            leaves.push(leaf);
                            collected_per_candidate[cand] += 1;
                            made_progress = true;
                        }
                        None => {
                            // Candidate's subtree is terminal / dead-ended.
                            // Mark its quota filled so we don't spin on it.
                            collected_per_candidate[cand] = visits_per_candidate;
                        }
                    }
                }
                if !made_progress {
                    break 'wave;
                }
            }

            if leaves.is_empty() {
                break;
            }
            total_collected += leaves.len();

            let values = self.batched_evaluate_and_expand(root, &leaves);
            for (leaf, &(value, _progress)) in leaves.iter().zip(values.iter()) {
                backpropagate_return_with_rewards(
                    root,
                    &leaf.data.path_indices,
                    &leaf.data.path_players,
                    &leaf.rewards,
                    &leaf.turn_deltas,
                    mcts_common::VIRTUAL_LOSS,
                    value,
                    reward::GAMMA_TURN,
                );
            }

            // Heal nodes whose cached children proved stale on replay (see
            // GumbelLeaf::stale_path): drop the children and mark unexpanded
            // so the next visit re-expands from the true replayed state.
            // Deferred to after backprop because other leaves in this wave may
            // hold path indices through a node being cleared.
            for leaf in &leaves {
                if let Some(path) = &leaf.stale_path {
                    if let Some(node) = get_node_by_path_mut(root, path) {
                        node.children.clear();
                        node.is_expanded = false;
                    }
                }
            }
        }
    }
    /// Cross an `EndTurn` edge. With `unfreeze_opponent` off (default), this
    /// is exactly `game.simulate_move(&EndTurnMove)` — the engine's built-in
    /// auto-skip past every opponent back to the searching player, bit-exact
    /// with all prior behavior. With it on (EXP_ELO_017), the searcher's own
    /// turn ends once, then each intervening opponent plays a REAL turn via
    /// the deterministic (argmax) Greedy heuristic — cheap, NN-free — until
    /// control returns to the searcher. Two-player only: asserts that
    /// happens within `MAX_GHOST_MOVES`, loud rather than silently wrong on
    /// a future 3+-player run.
    /// Edge-reward snapshot: the shared shaped snapshot plus the goal
    /// potential on the root player's own edges (their goal is the only one
    /// this search knows). Pre/post of an edge always use the same (mover,
    /// root) pair, so the added term stays a consistent potential.
    pub(super) fn edge_snapshot(
        &self,
        state: &crate::states::GameState,
        mover: i32,
        root_player: i32,
    ) -> (f32, f32) {
        let (mut my, opp) =
            reward::shaped_snapshot(state, mover, self.reward_shape_w, self.pursuit_shape_w);
        if self.goal_shape_w != 0.0 && mover == root_player {
            if let Some(goal) = &self.macro_goal {
                my += self.goal_shape_w
                    * reward::goal_potential(state, mover, goal, self.goal_aux.as_ref());
            }
        }
        (my, opp)
    }
    pub(super) fn cross_end_turn(game: &mut Game, unfreeze_opponent: bool) -> Option<crate::actions::UndoCallback> {
        if !unfreeze_opponent {
            return game.simulate_move(&EndTurnMove);
        }
        // Snapshot the whole state and restore it wholesale on undo, rather
        // than composing per-move undo callbacks across the ghost turn. A ghost
        // turn can include combat, and the kill-undo's index-patching
        // (units.rs) does NOT compose when several moves are bundled and
        // unwound in reverse — it panicked on `Vec::insert` out-of-bounds.
        // Restoring a clone is correct under the wave loop's LIFO unwind: by
        // the time this undo runs, every later edge is already undone, so the
        // live state equals the end-of-ghost-turn state and overwriting it with
        // the pre-ghost-turn snapshot restores exactly this edge. Costs one
        // GameState clone per turn crossing — unfreeze is the expensive-but-
        // robust training path; gauges/arena never cross unfrozen.
        const MAX_GHOST_MOVES: usize = 64;
        let searcher = game.state.settings.current_player_turn_id;
        let snapshot = game.state.clone();
        game.simulate_single_end_turn();
        let mut n = 0usize;
        while game.state.settings.current_player_turn_id != searcher
            && !game.state.settings._game_over
        {
            // Any ghost-turn anomaly (Greedy can't move, an illegal move, or a
            // runaway loop) degrades gracefully to a stale-node signal instead
            // of panicking mid-search and discarding the whole game's data.
            let bail = n >= MAX_GHOST_MOVES
                || match crate::ai::heuristic_mcts::GreedyHeuristicAgent.select_move(game) {
                    None => true,
                    Some(mv) if mv.move_type() == MoveType::EndTurn => {
                        game.simulate_single_end_turn();
                        false
                    }
                    Some(mv) => game.simulate_move(mv.as_ref()).is_none(),
                };
            if bail {
                game.state = snapshot.clone();
                return None;
            }
            n += 1;
        }
        Some(Box::new(move |s: &mut crate::states::GameState| {
            *s = snapshot;
        }))
    }
}
