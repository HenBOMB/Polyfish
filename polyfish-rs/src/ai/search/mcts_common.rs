//! Shared MCTS helpers used by both the AlphaZero-style (`ZeroMctsAgent`) and
//! Gumbel MuZero (`GumbelMctsAgent`) search agents.
//!
//! The two agents share a fair amount of structural code that, historically,
//! was duplicated byte-for-byte across `mcts_zero.rs` and `gumbel_mcts.rs`.
//! The most load-bearing piece is the player-aware value-backpropagation
//! sign-flip logic: a future fix to one copy and not the other would silently
//! diverge. That logic lives here exactly once, behind the `BackpropNode`
//! trait, so both agents are guaranteed to use the same code path.

use crate::ai::features::{RawFeatures, state_to_cpu_features_goal};
use crate::game::Game;
use crate::moves::Move;
use std::cell::RefCell;

/// Default virtual-loss amount applied during parallel leaf collection.
#[allow(dead_code)] // consumed by GumbelMctsAgent in phase (b)
pub(crate) const VIRTUAL_LOSS: f32 = 1.0;

/// Default number of leaves coalesced into a single batched NN call.
#[allow(dead_code)] // consumed by GumbelMctsAgent in phase (b)
pub(crate) const DEFAULT_BATCH_SIZE: usize = 24;

/// Pre-computed data extracted at a leaf node before undoing moves.
///
/// This owns everything the batched-evaluation phase needs, so the caller is
/// free to undo all simulated moves back to the root before the (potentially
/// async / batched) NN call happens.
pub(crate) struct LeafData {
    /// Child indices walked from the root to reach this leaf.
    pub path_indices: Vec<usize>,
    /// Player ID at each node along the path (length = path_indices + 1;
    /// `path_players[0]` is the root player, `path_players[i+1]` is the player
    /// after descending into `path_indices[i]`).
    pub path_players: Vec<i32>,
    /// Device-free features for NN evaluation (`None` if terminal / horizon-only).
    pub features: Option<RawFeatures>,
    /// Legal moves at this leaf, wrapped in `RefCell` for interior mutability
    /// during `take()` in the batched-expansion phase.
    pub legal_moves: RefCell<Vec<Box<dyn Move>>>,
    /// Heuristic scores parallel to `legal_moves`, computed at the leaf state
    /// when in-tree prior blending is enabled. `None` otherwise.
    pub heuristic_scores: Option<Vec<f32>>,
    /// Map size at the leaf state.
    pub map_size: usize,
    /// Terminal outcome in `[-1, 1]` from the leaf player's perspective, if
    /// this is a game-over state. `None` for non-terminal leaves.
    pub terminal_value: Option<f32>,
}

/// Convert a terminal game state into a signed outcome in `[-1, 1]` from the
/// perspective of the player whose turn it currently is.
///
/// Extracted verbatim from the duplicated blocks in `mcts_zero.rs` and
/// `gumbel_mcts.rs`.
pub(crate) fn compute_terminal_outcome(game: &Game) -> f32 {
    let current_player = game.state.settings.current_player_turn_id;
    let my_score = game
        .state
        .tribes
        .get(&current_player)
        .map(|t| t.score)
        .unwrap_or(0);

    let opp_best_score = game
        .state
        .tribes
        .iter()
        .filter(|(id, _)| **id != current_player)
        .map(|(_, t)| t.score)
        .max()
        .unwrap_or(0);

    if my_score > opp_best_score {
        1.0 // Win
    } else if my_score < opp_best_score {
        -1.0 // Loss
    } else {
        0.0 // Draw
    }
}

/// Build the `LeafData` for the current (post-descent, pre-undo) `game`
/// state. This is the three-way branch shared by both agents:
///   - terminal: outcome from `compute_terminal_outcome`, no features / moves
///   - needs expansion: features + legal moves (EndTurn included alongside
///     every other legal move — Verdi, Aug 2026: the early-training
///     passivity risk this used to guard against no longer needs a hard
///     gate; the model should learn when a turn is actually done)
///   - horizon: features only (evaluated by NN but not expanded)
///
/// `indices_stack` and `path_players` are moved into the returned `LeafData`.
/// `needs_expansion` is whatever the caller's descent loop decided: true when
/// the leaf is an unexpanded interior node, false for terminal / horizon.
/// Undoing the simulated moves remains the caller's responsibility.
pub(crate) fn extract_leaf_data(
    game: &Game,
    indices_stack: Vec<usize>,
    path_players: Vec<i32>,
    needs_expansion: bool,
    pursuit_focus: Option<i32>,
    macro_goal: Option<&crate::ai::oracle_macro::MacroGoal>,
) -> LeafData {
    let map_size = game.state.settings.size as usize;

    if game.state.settings._game_over {
        let outcome = compute_terminal_outcome(game);
        return LeafData {
            path_indices: indices_stack,
            path_players,
            features: None,
            legal_moves: RefCell::new(Vec::new()),
            heuristic_scores: None,
            map_size,
            terminal_value: Some(outcome),
        };
    }

    if needs_expansion {
        let feat = state_to_cpu_features_goal(
            &game.state,
            game.state.settings.current_player_turn_id,
            pursuit_focus,
            macro_goal,
        )
        .expect("BUG: Failed to create features at MCTS leaf");

        let legal_moves = game.legal_moves();

        return LeafData {
            path_indices: indices_stack,
            path_players,
            features: Some(feat),
            legal_moves: RefCell::new(legal_moves),
            heuristic_scores: None,
            map_size,
            terminal_value: None,
        };
    }

    // Horizon: evaluate with NN but do not expand.
    let feat = state_to_cpu_features_goal(
        &game.state,
        game.state.settings.current_player_turn_id,
        pursuit_focus,
        macro_goal,
    )
    .ok();

    LeafData {
        path_indices: indices_stack,
        path_players,
        features: feat,
        legal_moves: RefCell::new(Vec::new()),
        heuristic_scores: None,
        map_size,
        terminal_value: None,
    }
}

/// Tree-shape trait: any node that exposes a slice of children of the same
/// type. Used by the generic `get_node_by_path` / `get_node_by_path_mut`
/// helpers so both agents share the same path-walking logic.
pub(crate) trait TreeNode {
    fn children(&self) -> &[Self]
    where
        Self: Sized;
    fn children_mut(&mut self) -> &mut [Self]
    where
        Self: Sized;
}

/// Walk `indices` from `root` and return the referenced node, if it exists.
pub(crate) fn get_node_by_path<'b, N: TreeNode>(
    root: &'b N,
    indices: &[usize],
) -> Option<&'b N> {
    let mut current = root;
    for &idx in indices {
        current = current.children().get(idx)?;
    }
    Some(current)
}

/// Walk `indices` from `root` and return the referenced node mutably.
pub(crate) fn get_node_by_path_mut<'b, N: TreeNode>(
    root: &'b mut N,
    indices: &[usize],
) -> Option<&'b mut N> {
    let mut current = root;
    for &idx in indices {
        current = current.children_mut().get_mut(idx)?;
    }
    Some(current)
}

/// Node trait that participates in player-aware value backpropagation.
///
/// `BackpropNode` extends `TreeNode` so the shared backprop walker can descend
/// through children generically. Each concrete node type implements trivial
/// field accessors for `visits` / `value_sum` / `virtual_loss`.
pub(crate) trait BackpropNode: TreeNode {
    fn visits_mut(&mut self) -> &mut f32;
    fn value_sum_mut(&mut self) -> &mut f32;
    fn virtual_loss(&self) -> &RefCell<f32>;
}

/// Backpropagate `value` (in the leaf player's perspective) along the path
/// `root -> indices`, removing `virtual_loss_amount` from every node on the
/// path and adding one visit + the (sign-flipped) value to each node's
/// running sums.
///
/// Sign flipping: the value is negated each time the player to move changes
/// between consecutive nodes on the path (i.e. across an `EndTurn` boundary).
/// The root is handled explicitly relative to the leaf player; interior nodes
/// flip relative to their parent.
///
/// This is the single load-bearing correctness path that was previously
/// duplicated across `mcts_zero.rs` and `gumbel_mcts.rs`. Both agents now
/// route through this one implementation.
pub(crate) fn backpropagate_and_remove_virtual_loss<N: BackpropNode>(
    root: &mut N,
    indices: &[usize],
    path_players: &[i32],
    virtual_loss_amount: f32,
    mut value: f32,
) {
    // Anchor: the leaf player's perspective.
    let leaf_player = path_players.last().copied().unwrap_or(1);
    // Root player is the first entry in the path.
    let root_player = path_players.first().copied().unwrap_or(leaf_player);

    // Root's value, from the root's perspective.
    let root_value = if root_player != leaf_player {
        -value
    } else {
        value
    };

    *root.virtual_loss().borrow_mut() -= virtual_loss_amount;
    *root.visits_mut() += 1.0;
    *root.value_sum_mut() += root_value;

    let mut current: &mut N = root;
    let mut prev_player = root_player;

    for (i, &idx) in indices.iter().enumerate() {
        let child = match current.children_mut().get_mut(idx) {
            Some(c) => c,
            None => break,
        };

        // `path_players[i]` is the parent's player, `path_players[i+1]` is the
        // child's player (i.e. the player after the move that produced `child`).
        let child_player = if i + 1 < path_players.len() {
            path_players[i + 1]
        } else {
            leaf_player
        };

        // If the player to move changed across this edge, flip the value sign.
        if child_player != prev_player {
            value = -value;
        }

        *child.virtual_loss().borrow_mut() -= virtual_loss_amount;
        *child.visits_mut() += 1.0;
        *child.value_sum_mut() += value;

        current = child;
        prev_player = child_player;
    }
}

/// Reward-aware (MuZero-style) backpropagation. Each node's accumulator
/// holds the action-value of the edge that produced it — `credited[k] =
/// sv[k-1]` where `sv[d] = γ^Δt[d] * (r[d] + sv[d+1])` is the discounted
/// state-value-from-d computed backward from the leaf (`sv[n] =
/// leaf_value`), and `credited[0] = sv[0]` for the root (which has no
/// incoming edge of its own). Concretely: node `k`'s stored value bakes in
/// the reward of the edge *into* `k`, not the edge leaving it — that's what
/// makes `root.children[i].q_value()` an action value comparable across
/// siblings (a capture's `+reward` enters search as an exact number at
/// depth 1, rather than relying on the noisy, future-only value head to have
/// learned that capturing beats not-yet-capturing).
///
/// `path_rewards[d]` / `path_turn_deltas[d]` describe edge `d` (node `d` ->
/// node `d+1`), in the perspective of `path_players[d]` (the mover of that
/// edge) — same convention `extract_policy_targets`'s callers already use
/// for `path_players`. Length `n` (one entry per edge), vs. `path_players`'s
/// `n+1` (one entry per node). The player-aware sign flip generalizes
/// `backpropagate_and_remove_virtual_loss`'s (a value is negated when
/// crossing an edge where the mover changes) but is applied per-edge in
/// each node's own local perspective rather than accumulated as parity from
/// the root; the two conventions coincide whenever the path has at most one
/// player change (root vs. leaf) — true of every real search tree in this
/// codebase (single-player; see module docs) and this function's own
/// same-player tests — and only diverge on synthetic multi-flip paths
/// nothing here actually produces.
///
/// Setting every reward to 0 does **not** collapse to the plain-value
/// function's behavior unless `gamma == 1.0` — discounting the whole future
/// return (not just rewards) across turn boundaries is the point: it's what
/// makes a state further from the root in turns worth structurally less,
/// independent of whether any single edge banked a reward.
#[allow(clippy::too_many_arguments)]
pub(crate) fn backpropagate_return_with_rewards<N: BackpropNode>(
    root: &mut N,
    indices: &[usize],
    path_players: &[i32],
    path_rewards: &[f32],
    path_turn_deltas: &[i32],
    virtual_loss_amount: f32,
    leaf_value: f32,
    gamma: f32,
) {
    let n = indices.len();
    debug_assert_eq!(path_players.len(), n + 1);
    debug_assert_eq!(path_rewards.len(), n);
    debug_assert_eq!(path_turn_deltas.len(), n);

    // Backward pass: returns[d] = discounted return from node d onward, in
    // node d's own player-to-move perspective. returns[n] is the leaf value,
    // already in the leaf player's perspective.
    let mut returns = vec![0.0f32; n + 1];
    returns[n] = leaf_value;
    for d in (0..n).rev() {
        let mut next = returns[d + 1];
        if path_players[d + 1] != path_players[d] {
            next = -next;
        }
        let discount = gamma.powi(path_turn_deltas[d]);
        returns[d] = discount * (path_rewards[d] + next);
    }

    // Forward walk: identical bookkeeping shape to
    // `backpropagate_and_remove_virtual_loss`. The root has no incoming edge
    // of its own, so it's credited with `returns[0]` (= sv[0], its own
    // state-value). Every other node on the path is credited with
    // `returns[i]`, NOT `returns[i+1]` — node `i+1` (reached via edge `i`)
    // must bake in edge `i`'s reward, which lives at `returns[i]` per the
    // `credited[k] = sv[k-1]` shift documented above. Crediting `i+1` with
    // `returns[i+1]` instead would attribute the reward of the edge LEAVING
    // a node to that node itself, silently reintroducing a one-edge-late
    // version of the exact banked-vs-pending inversion this function exists
    // to fix.
    *root.virtual_loss().borrow_mut() -= virtual_loss_amount;
    *root.visits_mut() += 1.0;
    *root.value_sum_mut() += returns[0];

    let mut current: &mut N = root;
    for (i, &idx) in indices.iter().enumerate() {
        let child = match current.children_mut().get_mut(idx) {
            Some(c) => c,
            None => break,
        };
        *child.virtual_loss().borrow_mut() -= virtual_loss_amount;
        *child.visits_mut() += 1.0;
        *child.value_sum_mut() += returns[i];
        current = child;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal node for exercising `backpropagate_and_remove_virtual_loss`
    /// in pure isolation from `Game` / `Move` / network machinery.
    struct TestNode {
        visits: f32,
        value_sum: f32,
        virtual_loss: RefCell<f32>,
        children: Vec<TestNode>,
    }

    impl TestNode {
        fn new() -> Self {
            Self {
                visits: 0.0,
                value_sum: 0.0,
                virtual_loss: RefCell::new(0.0),
                children: Vec::new(),
            }
        }
        fn with_vl(vl: f32) -> Self {
            Self {
                visits: 0.0,
                value_sum: 0.0,
                virtual_loss: RefCell::new(vl),
                children: Vec::new(),
            }
        }
    }

    impl TreeNode for TestNode {
        fn children(&self) -> &[Self] {
            &self.children
        }
        fn children_mut(&mut self) -> &mut [Self] {
            &mut self.children
        }
    }

    impl BackpropNode for TestNode {
        fn visits_mut(&mut self) -> &mut f32 {
            &mut self.visits
        }
        fn value_sum_mut(&mut self) -> &mut f32 {
            &mut self.value_sum
        }
        fn virtual_loss(&self) -> &RefCell<f32> {
            &self.virtual_loss
        }
    }

    fn build_two_ply_tree() -> TestNode {
        // root -> child0 -> grandchild0
        // Same player on both edges (no sign flip).
        let mut root = TestNode::new();
        root.children.push(TestNode::new());
        root.children[0].children.push(TestNode::new());
        root
    }

    #[test]
    fn backprop_same_player_no_flip() {
        // path_players: [1, 1, 1] — root, child, grandchild all player 1.
        // Leaf value +0.5 from leaf (player 1) perspective.
        let mut root = build_two_ply_tree();
        // Pre-charge virtual loss on every node.
        *root.virtual_loss.borrow_mut() = VIRTUAL_LOSS;
        *root.children[0].virtual_loss.borrow_mut() = VIRTUAL_LOSS;
        *root.children[0].children[0].virtual_loss.borrow_mut() = VIRTUAL_LOSS;

        backpropagate_and_remove_virtual_loss(
            &mut root,
            &[0, 0],
            &[1, 1, 1],
            VIRTUAL_LOSS,
            0.5,
        );

        assert!((root.visits - 1.0).abs() < 1e-6);
        assert!((root.value_sum - 0.5).abs() < 1e-6);
        assert!((*root.virtual_loss.borrow() - 0.0).abs() < 1e-6);

        let child = &root.children[0];
        assert!((child.visits - 1.0).abs() < 1e-6);
        assert!((child.value_sum - 0.5).abs() < 1e-6);
        assert!((*child.virtual_loss.borrow() - 0.0).abs() < 1e-6);

        let gc = &root.children[0].children[0];
        assert!((gc.visits - 1.0).abs() < 1e-6);
        assert!((gc.value_sum - 0.5).abs() < 1e-6);
    }

    #[test]
    fn backprop_flips_across_player_change() {
        // path_players = [1, 2, 2], indices = [0, 0], leaf value = +0.5
        // (leaf player = 2). This locks in the exact sign sequence produced
        // by the pre-extraction reference implementation in `mcts_zero.rs`:
        //   root_value  = -(0.5) = -0.5            (root player 1 != leaf 2)
        //   i=0: child_player 2 != prev 1 -> flip  -> child      = -0.5
        //   i=1: child_player 2 == prev 2 -> no flip -> grandchild = -0.5
        // The shared fn must reproduce these values byte-for-byte.
        let mut root = build_two_ply_tree();

        backpropagate_and_remove_virtual_loss(
            &mut root,
            &[0, 0],
            &[1, 2, 2],
            VIRTUAL_LOSS,
            0.5,
        );

        assert!((root.value_sum - (-0.5)).abs() < 1e-6, "root value_sum");
        assert!(
            (root.children[0].value_sum - (-0.5)).abs() < 1e-6,
            "child value_sum"
        );
        assert!(
            (root.children[0].children[0].value_sum - (-0.5)).abs() < 1e-6,
            "grandchild value_sum"
        );
    }

    #[test]
    fn backprop_double_flip_two_end_turns() {
        // root player 1, child player 2, grandchild player 1.
        // Leaf value +0.5 from leaf (player 1) perspective.
        // root == leaf player -> root_value = +0.5.
        // i=0: prev=1, child=2 -> flip -> value=-0.5. child gets -0.5.
        // i=1: prev=2, child=1 -> flip -> value=+0.5. grandchild gets +0.5.
        let mut root = build_two_ply_tree();

        backpropagate_and_remove_virtual_loss(
            &mut root,
            &[0, 0],
            &[1, 2, 1],
            VIRTUAL_LOSS,
            0.5,
        );

        assert!((root.value_sum - 0.5).abs() < 1e-6, "root value_sum");
        assert!((root.children[0].value_sum - (-0.5)).abs() < 1e-6, "child");
        assert!(
            (root.children[0].children[0].value_sum - 0.5).abs() < 1e-6,
            "grandchild"
        );
    }

    #[test]
    fn backprop_removes_virtual_loss_only_on_path() {
        // Two children under root; only the path through child 0 should have
        // its virtual loss decremented; child 1 is untouched.
        let mut root = TestNode::new();
        root.children.push(TestNode::with_vl(VIRTUAL_LOSS));
        root.children.push(TestNode::with_vl(VIRTUAL_LOSS));
        *root.virtual_loss.borrow_mut() = VIRTUAL_LOSS;

        backpropagate_and_remove_virtual_loss(
            &mut root,
            &[0],
            &[1, 1],
            VIRTUAL_LOSS,
            0.0,
        );

        assert!((*root.virtual_loss.borrow() - 0.0).abs() < 1e-6);
        assert!((*root.children[0].virtual_loss.borrow() - 0.0).abs() < 1e-6);
        assert!(
            (*root.children[1].virtual_loss.borrow() - VIRTUAL_LOSS).abs() < 1e-6,
            "off-path child must keep its virtual loss"
        );
    }

    #[test]
    fn td_backup_matches_plain_when_rewards_zero_and_gamma_one() {
        // Sanity check: r=0 everywhere and gamma=1 must reproduce the plain
        // (non-reward-aware) backup exactly, same-player path.
        let mut root = build_two_ply_tree();
        backpropagate_return_with_rewards(
            &mut root,
            &[0, 0],
            &[1, 1, 1],
            &[0.0, 0.0],
            &[0, 0],
            VIRTUAL_LOSS,
            0.5,
            1.0,
        );
        assert!((root.value_sum - 0.5).abs() < 1e-6, "root");
        assert!((root.children[0].value_sum - 0.5).abs() < 1e-6, "child");
        assert!(
            (root.children[0].children[0].value_sum - 0.5).abs() < 1e-6,
            "grandchild"
        );
    }

    #[test]
    fn td_backup_credits_banked_reward_at_the_edge_that_earned_it() {
        // root -> child (edge 0: the capture itself, reward +0.3) -> leaf
        // (edge 1: no further reward, value 0.0). `child`'s credited value
        // is the action-value of edge 0 (the edge that produced it), so it
        // must carry the +0.3 regardless of what happens further down;
        // `leaf`'s credited value is the action-value of edge 1 (reward
        // 0.0), so it must NOT inherit the earlier capture.
        let mut root = TestNode::new();
        root.children.push(TestNode::new());
        root.children[0].children.push(TestNode::new());

        backpropagate_return_with_rewards(
            &mut root,
            &[0, 0],
            &[1, 1, 1],
            &[0.3, 0.0],
            &[0, 0],
            VIRTUAL_LOSS,
            0.0,
            0.9,
        );
        // sv[2]=0.0 (leaf). sv[1]=0.9^0*(0.0+0.0)=0.0. sv[0]=0.9^0*(0.3+0.0)=0.3.
        // credited[root]=sv[0]=0.3. credited[child]=sv[0]=0.3. credited[leaf]=sv[1]=0.0.
        assert!(
            (root.children[0].value_sum - 0.3).abs() < 1e-6,
            "child (capture edge) should be credited with its own banked reward, got {}",
            root.children[0].value_sum
        );
        assert!((root.value_sum - 0.3).abs() < 1e-6, "root");
        assert!(
            (root.children[0].children[0].value_sum - 0.0).abs() < 1e-6,
            "leaf's own edge banked nothing — must not inherit the earlier capture, got {}",
            root.children[0].children[0].value_sum
        );
    }

    #[test]
    fn td_backup_banked_now_beats_pending_later_with_equal_totals() {
        // The motivating case: two 2-edge paths accumulate the SAME total
        // reward (+0.3) and cross the SAME single turn boundary, but differ
        // in WHICH edge carries which:
        //   capture_now:   edge0 = capture (r=0.3, same turn), edge1 = the
        //                  turn boundary passing normally (r=0.0, dt=1).
        //   capture_later: edge0 = a wasted move that crosses the turn
        //                  boundary without capturing (r=0.0, dt=1), edge1 =
        //                  the capture next turn (r=0.3, same turn).
        // Both reach an identically-valued leaf (0.2). A flat MC sum over
        // the path can't tell these apart; discounting must: the capture-now
        // child (the live decision at the shared root) must strictly beat
        // the capture-later child, since the later path's reward gets
        // discounted by one extra turn on its way back to the root.
        let mut capture_now = TestNode::new();
        capture_now.children.push(TestNode::new());
        capture_now.children[0].children.push(TestNode::new());
        backpropagate_return_with_rewards(
            &mut capture_now,
            &[0, 0],
            &[1, 1, 1],
            &[0.3, 0.0], // capture now, then an ordinary turn passes
            &[0, 1],     // turn boundary is on the SECOND edge
            VIRTUAL_LOSS,
            0.2,
            0.9,
        );

        let mut capture_later = TestNode::new();
        capture_later.children.push(TestNode::new());
        capture_later.children[0].children.push(TestNode::new());
        backpropagate_return_with_rewards(
            &mut capture_later,
            &[0, 0],
            &[1, 1, 1],
            &[0.0, 0.3], // wasted move, then capture next turn
            &[1, 0],     // turn boundary is on the FIRST edge
            VIRTUAL_LOSS,
            0.2,
            0.9,
        );

        // capture_now: sv[2]=0.2, sv[1]=0.9^1*(0.0+0.2)=0.18, sv[0]=0.9^0*(0.3+0.18)=0.48.
        // capture_later: sv[2]=0.2, sv[1]=0.9^0*(0.3+0.2)=0.5, sv[0]=0.9^1*(0.0+0.5)=0.45.
        assert!(
            capture_now.children[0].value_sum > capture_later.children[0].value_sum,
            "capture-now child ({}) should beat capture-later child ({})",
            capture_now.children[0].value_sum,
            capture_later.children[0].value_sum
        );
        assert!(
            (capture_now.children[0].value_sum - 0.48).abs() < 1e-5,
            "got {}",
            capture_now.children[0].value_sum
        );
        assert!(
            (capture_later.children[0].value_sum - 0.45).abs() < 1e-5,
            "got {}",
            capture_later.children[0].value_sum
        );
    }

    #[test]
    fn td_backup_discounts_across_turn_boundary() {
        // Single edge, reward 0, leaf value 1.0, turn delta 1: return should
        // be gamma * 1.0, not 1.0.
        let mut root = TestNode::new();
        root.children.push(TestNode::new());
        backpropagate_return_with_rewards(
            &mut root,
            &[0],
            &[1, 1],
            &[0.0],
            &[1],
            VIRTUAL_LOSS,
            1.0,
            0.9,
        );
        assert!(
            (root.children[0].value_sum - 0.9).abs() < 1e-6,
            "got {}",
            root.children[0].value_sum
        );
    }

    #[test]
    fn td_backup_flips_across_player_change() {
        // path_players = [1, 2, 2]: root is player 1, child and leaf are
        // both player 2. Leaf value +0.5 in the leaf's (player 2)
        // perspective, no rewards, gamma 1.0.
        //
        // root is credited with sv[0] = the value of edge0 (root's move,
        // player 1) in player 1's frame: player 1 != leaf's player 2, so it
        // flips to -0.5 — matches the plain backup's root value exactly.
        //
        // child is credited with sv[0] too (edge0's action-value, same
        // number as root — coincidence of a single-edge-worth of
        // information, same as the plain backup's no-flip case) = -0.5.
        //
        // leaf is credited with sv[1] = the value of edge1 (child's move,
        // player 2) in player 2's frame: child and leaf are BOTH player 2,
        // no flip needed, so it stays +0.5 — this is where this function
        // intentionally diverges from `backpropagate_and_remove_virtual_loss`
        // (which gives -0.5 here): that function accumulates sign flips as
        // parity-from-root rather than locally per edge, so a node 2+ hops
        // past the one real perspective change gets a stale sign. Per-edge
        // local perspective is the mathematically consistent choice (and
        // the two conventions agree on every real search tree, which is
        // always single-player — see module docs).
        let mut root = build_two_ply_tree();
        backpropagate_return_with_rewards(
            &mut root,
            &[0, 0],
            &[1, 2, 2],
            &[0.0, 0.0],
            &[0, 0],
            VIRTUAL_LOSS,
            0.5,
            1.0,
        );
        assert!((root.value_sum - (-0.5)).abs() < 1e-6, "root value_sum");
        assert!(
            (root.children[0].value_sum - (-0.5)).abs() < 1e-6,
            "child value_sum"
        );
        assert!(
            (root.children[0].children[0].value_sum - 0.5).abs() < 1e-6,
            "grandchild value_sum, got {}",
            root.children[0].children[0].value_sum
        );
    }

    #[test]
    fn td_backup_removes_virtual_loss_only_on_path() {
        let mut root = TestNode::new();
        root.children.push(TestNode::with_vl(VIRTUAL_LOSS));
        root.children.push(TestNode::with_vl(VIRTUAL_LOSS));
        *root.virtual_loss.borrow_mut() = VIRTUAL_LOSS;

        backpropagate_return_with_rewards(
            &mut root,
            &[0],
            &[1, 1],
            &[0.0],
            &[0],
            VIRTUAL_LOSS,
            0.0,
            0.9,
        );

        assert!((*root.virtual_loss.borrow() - 0.0).abs() < 1e-6);
        assert!((*root.children[0].virtual_loss.borrow() - 0.0).abs() < 1e-6);
        assert!(
            (*root.children[1].virtual_loss.borrow() - VIRTUAL_LOSS).abs() < 1e-6,
            "off-path child must keep its virtual loss"
        );
    }

    #[test]
    fn get_node_by_path_walks_children() {
        let mut root = TestNode::new();
        root.children.push(TestNode::new());
        root.children[0].children.push(TestNode::new());

        assert!(get_node_by_path(&root, &[]).is_some());
        assert!(get_node_by_path(&root, &[0]).is_some());
        assert!(get_node_by_path(&root, &[0, 0]).is_some());
        assert!(get_node_by_path(&root, &[1]).is_none());
        assert!(get_node_by_path(&root, &[0, 1]).is_none());
    }
}
