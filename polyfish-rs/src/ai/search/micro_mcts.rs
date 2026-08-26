//! Micro-mcts: a bounded within-turn PUCT search below the committed
//! `MacroGoal`, real-trajectory-only (never inside macro-mcts's own turn-
//! level rollouts, which stay on the cheap greedy `rank_plies` unchanged).
//!
//! Follows EXP_ELO_071 (Phase 0 throughput probe), which measured that
//! pricing every candidate — interior nodes included — with the full
//! `rank_view`/Δφ pipeline costs more than it buys (−51% to −80% moves/sec,
//! worsening with scale). Two corrections from that finding:
//! - Interior-node child priors use the CHEAP `score_move` heuristic only
//!   (no `simulate_move`/Δφ) — the expensive pipeline is never called below
//!   the root.
//! - Leaves are evaluated by the trained network's value head via
//!   `eval_server`, not `goal_potential` — so search quality is learnable
//!   instead of bound to a fixed hand-authored heuristic (Verdi's explicit
//!   correction earlier in this design: a Φ-based leaf just searches deeper
//!   into the same brittleness it's meant to escape).
//!
//! The root's own children are `rank_view`'s already-computed candidates
//! (real per-ply cost, paid regardless of whether this search runs) — only
//! nodes BELOW the root use the cheap path.
//!
//! Effective search depth is emergent from `sims`, not a separate dial: the
//! tree grows one new ply per simulation along the PUCT-selected path and
//! stops naturally at the real turn boundary (`is_terminal`). `MicroParams::
//! depth` is only a defensive recursion ceiling, not a target.

use crate::ai::eval_server::Evaluator;
use crate::ai::oracle_macro::MacroGoal;
use crate::ai::search::goal_aux::GoalAux;
use crate::ai::search::macro_exec::gate_ok;
use crate::ai::scoring::score_move;
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::states::PlayerId;
use crate::types::MoveType;

#[derive(Clone, Copy, Debug)]
pub struct MicroParams {
    pub sims: usize,
    /// NOT a target depth -- `select_and_expand` already stops naturally at
    /// the real turn boundary (`is_terminal`: only-EndTurn-remains, or a
    /// simulated move hands the turn to the other player). This is purely a
    /// defensive ceiling against pathological runaway recursion; `sims` is
    /// what actually determines how deep any explored line gets, exactly
    /// like the old `GumbelMctsAgent` (depth grew 4.09→26.32 plies as sims
    /// went 64→2048, per Addendum 2). A previous default of 4 silently
    /// capped every simulation inside the flat, no-new-knowledge regime that
    /// data showed (override rate only rises past ~9 plies) -- raised to 64
    /// so it never binds in practice.
    pub depth: usize,
    pub k: usize,
    /// PUCT exploration constant. Uncalibrated first-fit (a common
    /// AlphaZero-style default) -- dial against a measured root q-spread
    /// before trusting it, per this codebase's own convention for every
    /// other search constant (see macro_mcts::EXPLORATION's own history).
    pub c_puct: f32,
}

impl Default for MicroParams {
    fn default() -> Self {
        Self { sims: 16, depth: 64, k: 4, c_puct: 1.5 }
    }
}

/// Real-trajectory-only env gate, mirroring `ply_trace_path`'s style. Unset
/// (the default in every existing invocation) costs one cached read and
/// changes nothing -- `select_move` skips the search entirely.
pub fn micro_mcts_params() -> Option<MicroParams> {
    static PARAMS: std::sync::OnceLock<Option<MicroParams>> = std::sync::OnceLock::new();
    *PARAMS.get_or_init(|| {
        let sims: usize = std::env::var("POLYFISH_MICRO_MCTS_SIMS").ok()?.parse().ok()?;
        let depth: usize = std::env::var("POLYFISH_MICRO_MCTS_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64);
        let k: usize = std::env::var("POLYFISH_MICRO_MCTS_K")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        let c_puct: f32 = std::env::var("POLYFISH_MICRO_MCTS_CPUCT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1.5);
        Some(MicroParams { sims, depth, k, c_puct })
    })
}

/// Diagnostic (temporary, not a standing feature): how often the tree's
/// argmax-visits pick actually disagrees with `rank_view`'s own top-ranked
/// candidate (index 0). If this stays at 0 across real games, the search is
/// not influencing move selection at all in practice, regardless of sims.
pub static MICRO_MCTS_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MICRO_MCTS_OVERRIDES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Diagnostic: how often a carried-in subtree (see `MicroTreeCarry`) was
/// actually spliced into this ply's root vs. discarded (predicted move
/// wasn't the one played, or a new turn started). `ATTEMPTS` counts calls
/// where a carry was offered at all; `HITS` counts the ones that matched.
pub static MICRO_CARRY_ATTEMPTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static MICRO_CARRY_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Root-advancement warm start, carried by the caller (`MacroMctsAgent`)
/// from one real ply to the next. `mv_key` identifies the move this search
/// picked -- the caller compares it against what actually got played and
/// only hands `children` back on a match (intra-turn plies are
/// deterministic and self-only, so a key match means the position is exact,
/// not approximate). `children` are the grandchildren already explored one
/// level below that move -- i.e. already-explored options for the NEXT
/// ply's decision, not the move itself (which is now consumed and can't
/// reappear as a candidate). The caller must drop this (pass `None`) across
/// a turn boundary or whenever the predicted move didn't end up executed.
pub struct MicroTreeCarry {
    mv_key: serde_json::Value,
    children: Vec<MicroChild>,
}

struct MicroChild {
    mv: Box<dyn Move>,
    prior: f32,
    node: Option<MicroNode>,
}

struct MicroNode {
    game: Game,
    visits: u32,
    value_sum: f32,
    children: Vec<MicroChild>,
    is_terminal: bool,
}

impl MicroNode {
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f32
        }
    }
}

fn softmax_priors(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let n = scores.len() as f32;
        return vec![1.0 / n; scores.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Cheap interior-node candidate generation: `legal_moves` + `gate_ok` +
/// the static `score_move` heuristic. No `simulate_move`, no Δφ, no belief/
/// threats -- this is the whole point (see module doc). Truncates to the
/// top `k` by raw score before the caller turns scores into PUCT priors.
fn cheap_candidates(
    game: &Game,
    goal: &MacroGoal,
    star_gate: bool,
    aux: &GoalAux,
    k: usize,
) -> Vec<(Box<dyn Move>, f32)> {
    let mut moves = game.legal_moves();
    moves.retain(|m| gate_ok(&game.state, m.as_ref(), star_gate, Some(goal.stance), Some(aux)));
    let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
    if has_other {
        moves.retain(|m| m.move_type() != MoveType::EndTurn);
    }
    if moves.is_empty() {
        return vec![(Box::new(EndTurnMove) as Box<dyn Move>, 0.0)];
    }
    let mut scored: Vec<(Box<dyn Move>, f32)> = moves
        .into_iter()
        .map(|m| {
            let s = score_move(game, m.as_ref());
            (m, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k.max(1));
    scored
}

/// The only place this module touches `eval_server`. One position, one
/// forward -- batching multiple simulations' leaves into a single
/// `evaluate(vec![...])` call is a real, worthwhile fast-follow (the walks
/// to reach each leaf are already independent) but not built here; this
/// mirrors macro-mcts's own `leaf_value`, which is also one-at-a-time.
fn leaf_value(game: &Game, pov: PlayerId, goal: &MacroGoal, evaluator: &Evaluator) -> f32 {
    crate::ai::features::state_to_cpu_features_goal(&game.state, pov, None, Some(goal))
        .ok()
        .and_then(|f| evaluator.evaluate(vec![f]).into_iter().next().map(|r| r.0))
        .unwrap_or(0.0)
}

#[allow(clippy::too_many_arguments)]
fn select_and_expand(
    node: &mut MicroNode,
    pov: PlayerId,
    goal: &MacroGoal,
    star_gate: bool,
    aux: &GoalAux,
    evaluator: &Evaluator,
    params: &MicroParams,
    depth_remaining: usize,
) -> f32 {
    if node.is_terminal || depth_remaining == 0 {
        return leaf_value(&node.game, pov, goal, evaluator);
    }
    if node.children.is_empty() {
        let cands = cheap_candidates(&node.game, goal, star_gate, aux, params.k);
        if cands.len() == 1 && cands[0].0.move_type() == MoveType::EndTurn {
            // Turn is genuinely over: nothing left to explore, and we never
            // simulate past our own EndTurn into the opponent's turn.
            node.is_terminal = true;
            return leaf_value(&node.game, pov, goal, evaluator);
        }
        let scores: Vec<f32> = cands.iter().map(|(_, s)| *s).collect();
        let priors = softmax_priors(&scores);
        node.children = cands
            .into_iter()
            .zip(priors)
            .map(|((mv, _), prior)| MicroChild { mv, prior, node: None })
            .collect();
    }

    let total_visits: u32 =
        node.children.iter().map(|c| c.node.as_ref().map_or(0, |n| n.visits)).sum();
    let mut best_idx = 0;
    let mut best_score = f32::MIN;
    for (i, c) in node.children.iter().enumerate() {
        let (q, n) = c.node.as_ref().map_or((0.0, 0u32), |n| (n.q(), n.visits));
        let u = params.c_puct * c.prior * (total_visits.max(1) as f32).sqrt() / (1.0 + n as f32);
        let s = q + u;
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    let value = if node.children[best_idx].node.is_none() {
        let mut child_game = node.game.clone();
        let ok = child_game.simulate_move(node.children[best_idx].mv.as_ref()).is_some();
        let terminal = !ok || child_game.state.settings.current_player_turn_id != pov;
        let v = leaf_value(&child_game, pov, goal, evaluator);
        node.children[best_idx].node = Some(MicroNode {
            game: child_game,
            visits: 1,
            value_sum: v,
            children: Vec::new(),
            is_terminal: terminal,
        });
        v
    } else {
        let child = node.children[best_idx].node.as_mut().unwrap();
        select_and_expand(child, pov, goal, star_gate, aux, evaluator, params, depth_remaining - 1)
    };
    node.visits += 1;
    node.value_sum += value;
    value
}

/// Root children are `rank_view`'s own top candidates (already paid for,
/// full Δφ fidelity) -- only nodes below the root use `cheap_candidates`.
/// `carry`, if it matches one of this ply's candidates by move identity, is
/// spliced in as that child's already-explored subtree (root advancement --
/// a free warm start instead of discarding a ply's search every ply).
/// Returns `(index into `ranked` the search prefers, subtree to carry into
/// the NEXT ply if the caller ends up actually playing that pick)`. The
/// index is `None` when there's nothing to search (a lone EndTurn, or too
/// few candidates); the carry is `None` whenever no search ran.
#[allow(clippy::too_many_arguments)]
pub fn micro_search_pick(
    view: &Game,
    pov: PlayerId,
    goal: &MacroGoal,
    ranked: &[(f32, Box<dyn Move>)],
    aux: &GoalAux,
    star_gate: bool,
    evaluator: &Evaluator,
    params: &MicroParams,
    carry: Option<MicroTreeCarry>,
) -> (Option<usize>, Option<MicroTreeCarry>) {
    if ranked.len() < 2 || ranked[0].1.move_type() == MoveType::EndTurn {
        return (None, None);
    }
    if carry.is_some() {
        MICRO_CARRY_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let top_n = ranked.len().min(params.k.max(4));
    let scores: Vec<f32> = ranked.iter().take(top_n).map(|(s, _)| *s).collect();
    let priors = softmax_priors(&scores);
    // The carry's own `mv_key` is the move that was just played -- already
    // consumed, and can't reappear here. What can reappear (and is worth
    // matching) is the set of options already explored one ply below it.
    let mut carried_children = carry.map(|c| c.children).unwrap_or_default();
    let children: Vec<MicroChild> = ranked
        .iter()
        .take(top_n)
        .zip(priors)
        .map(|((_, mv), prior)| {
            let key = mv.serialize();
            let node = if let Some(pos) =
                carried_children.iter().position(|c| c.mv.serialize() == key)
            {
                MICRO_CARRY_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                carried_children.remove(pos).node
            } else {
                None
            };
            MicroChild { mv: dyn_clone::clone_box(mv.as_ref()), prior, node }
        })
        .collect();
    let mut root = MicroNode {
        game: view.clone(),
        visits: 0,
        value_sum: 0.0,
        children,
        is_terminal: false,
    };
    for _ in 0..params.sims {
        select_and_expand(&mut root, pov, goal, star_gate, aux, evaluator, params, params.depth);
    }
    let mut best_idx = 0;
    let mut best_visits: i64 = -1;
    for (i, c) in root.children.iter().enumerate() {
        let v = c.node.as_ref().map_or(0, |n| n.visits) as i64;
        if v > best_visits {
            best_visits = v;
            best_idx = i;
        }
    }
    MICRO_MCTS_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if best_idx != 0 {
        MICRO_MCTS_OVERRIDES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let mv_key = root.children[best_idx].mv.serialize();
    let grandchildren = root.children[best_idx].node.take().map(|node| node.children).unwrap_or_default();
    let next_carry =
        if grandchildren.is_empty() { None } else { Some(MicroTreeCarry { mv_key, children: grandchildren }) };
    (Some(best_idx), next_carry)
}
