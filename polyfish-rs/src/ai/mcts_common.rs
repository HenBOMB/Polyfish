//! Shared MCTS helpers used by both the AlphaZero-style (`ZeroMctsAgent`) and
//! Gumbel MuZero (`GumbelMctsAgent`) search agents.
//!
//! The two agents share a fair amount of structural code that, historically,
//! was duplicated byte-for-byte across `mcts_zero.rs` and `gumbel_mcts.rs`.
//! The most load-bearing piece is the player-aware value-backpropagation
//! sign-flip logic: a future fix to one copy and not the other would silently
//! diverge. That logic lives here exactly once, behind the `BackpropNode`
//! trait, so both agents are guaranteed to use the same code path.

use crate::ai::features::{RawFeatures, state_to_cpu_features};
use crate::game::Game;
use crate::moves::Move;
use crate::types::MoveType;
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
///   - needs expansion: features + legal moves (EndTurn filtered out when
///     other moves exist, since batched expansion always allows EndTurn)
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
) -> LeafData {
    let map_size = game.state.settings.size as usize;

    if game.state.settings._game_over {
        let outcome = compute_terminal_outcome(game);
        return LeafData {
            path_indices: indices_stack,
            path_players,
            features: None,
            legal_moves: RefCell::new(Vec::new()),
            map_size,
            terminal_value: Some(outcome),
        };
    }

    if needs_expansion {
        let feat = state_to_cpu_features(&game.state, game.state.settings.current_player_turn_id)
            .expect("BUG: Failed to create features at MCTS leaf");

        let mut legal_moves = game.legal_moves();
        // In batched expansion we always allow EndTurn (allow_end_turn=true),
        // so only keep EndTurn if it is the sole option.
        let has_other = legal_moves
            .iter()
            .any(|m| m.move_type() != MoveType::EndTurn);
        if has_other {
            legal_moves.retain(|m| m.move_type() != MoveType::EndTurn);
        }

        return LeafData {
            path_indices: indices_stack,
            path_players,
            features: Some(feat),
            legal_moves: RefCell::new(legal_moves),
            map_size,
            terminal_value: None,
        };
    }

    // Horizon: evaluate with NN but do not expand.
    let feat = state_to_cpu_features(&game.state, game.state.settings.current_player_turn_id).ok();

    LeafData {
        path_indices: indices_stack,
        path_players,
        features: feat,
        legal_moves: RefCell::new(Vec::new()),
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
