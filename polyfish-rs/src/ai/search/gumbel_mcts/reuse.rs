//! Tree-reuse validation and prior-blending helpers (Aug 2026 taxonomy
//! split out of gumbel_mcts.rs to keep every file under ~1000 lines): reset
//! stats for re-rooting, clone a chosen move out of the tree, blend a
//! heuristic/goal prior into logits, and check a reused subtree's cached
//! children against the real legal set.

use super::gate::gate_block;

use crate::ai::features::{self};
use crate::ai::gumbel_qtransform::softmax;
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::types::MoveType;

use super::GumbelNode;

/// Recursively zero `visits` / `value_sum` / `virtual_loss` across the
/// subtree, keeping `is_expanded`, `children`, `logit`, `own_value`, and
/// `move_to_here` intact. Used by structure-only root-shift reuse so the new
/// search's Sequential Halving runs on a clean statistical slate while the
/// expanded structure and cached NN policy/value are retained.
pub(super) fn reset_stats_recursive(node: &mut GumbelNode) {
    node.visits = 0.0;
    node.value_sum = 0.0;
    *node.virtual_loss.borrow_mut() = 0.0;
    for c in &mut node.children {
        reset_stats_recursive(c);
    }
}
/// Clone the chosen child's move out of the tree without removing the child,
/// so the subtree below it stays available for next-call root-shift reuse.
pub(super) fn clone_child_move(root: &GumbelNode, idx: usize) -> Option<Box<dyn Move>> {
    root.children
        .get(idx)
        .and_then(|c| c.move_to_here.as_ref())
        .map(|m| dyn_clone::clone_box(&**m))
}
/// Blend a heuristic prior into a raw logit slice in place.
/// Formula `p' = (1-w)*p_net + w*p_heur`, `p_heur = softmax(heur_scores / TEMP)`.
/// Shared by the root blend (`blend_heuristic_prior`) and in-tree expansion.
pub(super) fn blend_heuristic_into_logits(logits: &mut [f32], heur_scores: &[f32], weight: f32) {
    const HEURISTIC_TEMP: f32 = 20.0;
    if logits.is_empty() || logits.len() != heur_scores.len() {
        return;
    }

    let p_net = softmax(logits);
    let scaled: Vec<f32> = heur_scores.iter().map(|s| s / HEURISTIC_TEMP).collect();
    let p_heur = softmax(&scaled);

    for (i, l) in logits.iter_mut().enumerate() {
        let p = (1.0 - weight) * p_net[i] + weight * p_heur[i];
        // Add a small epsilon to prevent log(0)
        *l = (p + 1e-9).ln();
    }
}
/// Weight on the goal prior: mass reserved for moves that advance a SAVE
/// batch. Sized against the measured deficit — the net's prior on the lane
/// tech is ~3e-6 (median, 519 traced Smithery decisions), an 11.3-nat gap to
/// the chosen move, which no amount of σ(Q) can bridge inside sequential
/// halving. This buys the plan's own move enough visits for Q to arbitrate; it
/// is not meant to make it win. FIRST FIT — dial against the measured cut-entry
/// and eligibility rates before trusting it.
pub(super) fn goal_prior_weight() -> f32 {
    static W: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("GOAL_PRIOR_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.15)
    })
}
/// Mix mass onto the moves that advance the SAVE batch, in place.
///
/// The macro names a lane (tech + structure); until v10 it kept only the
/// price, so the plan could not raise the prior on the one move it existed to
/// reach. Applied before the top-k cut is built so the cut ranks on the
/// boosted prior — the gates learned the hard way that a root-only fixup which
/// skips `finish_reused_root` is inert on 7 of every 8 plies.
pub(super) fn blend_goal_prior(
    game: &Game,
    children: &mut [GumbelNode],
    goal: Option<&crate::ai::oracle_macro::MacroGoal>,
) {
    let w = goal_prior_weight();
    if w <= 0.0 || children.is_empty() {
        return;
    }
    let Some(lane) = goal.and_then(|g| g.save_target.as_ref()) else {
        return;
    };
    let pov = game.state.settings.current_player_turn_id;
    let Some(tribe) = game.state.tribes.get(&pov) else {
        return;
    };
    let hits: Vec<bool> = children
        .iter()
        .map(|c| {
            c.move_to_here.as_ref().map_or(false, |m| {
                crate::ai::oracle_macro::advances_save_plan(m.as_ref(), lane, tribe)
            })
        })
        .collect();
    let n_hit = hits.iter().filter(|&&h| h).count();
    if n_hit == 0 {
        return;
    }
    let mut logits: Vec<f32> = children.iter().map(|c| c.logit).collect();
    let p_net = softmax(&logits);
    let share = w / n_hit as f32;
    for (i, l) in logits.iter_mut().enumerate() {
        let p = (1.0 - w) * p_net[i] + if hits[i] { share } else { 0.0 };
        *l = (p + 1e-9).ln();
    }
    for (child, l) in children.iter_mut().zip(logits.into_iter()) {
        child.logit = l;
    }
}
pub(super) fn blend_heuristic_prior(game: &Game, children: &mut [GumbelNode], weight: f32) {
    if children.is_empty() {
        return;
    }
    let mut logits: Vec<f32> = children.iter().map(|c| c.logit).collect();
    let scores: Vec<f32> = children.iter()
        .map(|c| c.move_to_here.as_ref()
            .map_or(0.0, |m| crate::ai::scoring::score_move(game, m.as_ref())))
        .collect();
    blend_heuristic_into_logits(&mut logits, &scores, weight);
    for (child, l) in children.iter_mut().zip(logits.into_iter()) {
        child.logit = l;
    }
}
/// Multiset-compare a reused root's cached child moves against the real
/// state's legal moves. Any mismatch means the sim-built cache is stale.
///
/// Every expansion path (`build_fresh_root`, and `extract_leaf_data` in
/// `mcts_common.rs` for interior nodes) drops `EndTurn` from a node's
/// children whenever another move is legal, keeping it only when it's the
/// sole option. The cached `children` were built through one of those paths,
/// so the comparison must apply the same normalization to `game.legal_moves()`
/// — otherwise EndTurn's presence in the raw legal set spuriously fails the
/// match on every reuse attempt where any other move exists.
pub(super) fn reused_children_match_legal(
    game: &Game,
    children: &[GumbelNode],
    star_gate: bool,
    stance: Option<crate::ai::oracle_macro::Stance>,
    goal_aux: Option<&crate::ai::oracle_macro::GoalAux>,
) -> bool {
    let mut legal = game.legal_moves();
    if star_gate || goal_aux.is_some() {
        // Consistency check, not a real gating decision — same predicate, but
        // deliberately unrecorded so the stats count each ply exactly once.
        legal.retain(|m| {
            m.move_type() == MoveType::EndTurn
                || gate_block(&game.state, m.as_ref(), star_gate, stance, goal_aux).is_none()
        });
    }
    let has_other = legal.iter().any(|m| m.move_type() != MoveType::EndTurn);
    if has_other {
        legal.retain(|m| m.move_type() != MoveType::EndTurn);
    }
    if legal.len() != children.len() {
        return false;
    }
    let mut remaining: Vec<serde_json::Value> = legal.iter().map(|m| m.serialize()).collect();
    for child in children {
        let Some(m) = child.move_to_here.as_ref() else {
            return false;
        };
        let v = m.serialize();
        match remaining.iter().position(|r| *r == v) {
            Some(i) => {
                remaining.swap_remove(i);
            }
            None => return false,
        }
    }
    true
}
/// Apply `m` to `game` (assumed to be at the root state, with all search
/// undos applied) via `play_move` — the same path the real game loop uses —
/// and hash the resulting state's features. This is the hash the *next*
/// search's root must match to re-root into this child. Using `play_move`
/// (not `simulate_move`) is load-bearing: the real game loop advances via
/// `play_move`, which updates `_history` and runs FOW discovery that
/// `simulate_move` skips, so a simulate-derived hash would never match the
/// next call's play-derived features. The `game` here is the per-call clone
/// the Brain passes in, which is discarded after we return, so no undo is
/// needed. `None` if the move can't be applied or features can't be built,
/// in which case the next call simply builds fresh.
pub(super) fn next_root_hash_for(
    game: &mut Game,
    m: Option<&dyn Move>,
    pursuit_focus: Option<i32>,
    macro_goal: Option<&crate::ai::oracle_macro::MacroGoal>,
) -> Option<u64> {
    let m = m?;
    // The undo callback is intentionally dropped because the game is a per-call clone
    let _ = game.play_move(m)?;
    let feat = features::state_to_cpu_features_goal(
        &game.state,
        game.state.settings.current_player_turn_id,
        pursuit_focus,
        macro_goal,
    )
    .ok()?;
    Some(feat.hash())
}
pub(super) fn move_or_end_turn(best_move: Option<Box<dyn Move>>) -> Option<Box<dyn Move>> {
    if best_move.is_none() {
        Some(Box::new(EndTurnMove))
    } else {
        best_move
    }
}
