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
    /// Weight on a net-derived root prior, blended with the existing
    /// `rank_view`/Δφ-score softmax prior (0.0 = off, current behavior).
    /// Mirrors `MacroParams::root_prior_w`'s convention. The goal painted for
    /// this eval call is the ply's own COMMITTED macro goal -- the same
    /// convention real training rows already use (`game.rs`'s `feat_goal`),
    /// so unlike macro's root prior this carries no root-painting-mismatch
    /// risk: the four decomposed policy heads it reads are already
    /// behavior-cloned on macro-mcts's own committed picks (`brain.rs`).
    pub net_prior_w: f32,
}

impl Default for MicroParams {
    fn default() -> Self {
        Self { sims: 16, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0 }
    }
}

/// EXP_ELO_119: default sims budget once micro-mcts became the default
/// (previously opt-in only, off everywhere). Chosen from a throughput/
/// override-rate sweep on the canonical watch seed: sims=8 costs ~2.2x
/// baseline self-play throughput (21.7 vs 46.85 moves/sec) vs. sims=64's
/// ~6x (7.7 moves/sec), with no measured improvement in override rate
/// (10-15% at every tested budget on that one seed) -- Verdi's call,
/// given the flat curve. Not yet validated for win-rate impact, only
/// behavioral override rate -- see the EXP_ELO_119 ledger entry.
const MICRO_MCTS_DEFAULT_SIMS: usize = 8;

/// Real-trajectory-only search (never inside macro-mcts's own rollouts --
/// see the module doc). ON BY DEFAULT as of EXP_ELO_119 at
/// `MICRO_MCTS_DEFAULT_SIMS`; `POLYFISH_MICRO_MCTS_SIMS` overrides the
/// budget, and `POLYFISH_MICRO_MCTS_SIMS=0` is the escape hatch that
/// disables it entirely (replacing "just leave it unset", the old
/// opt-in's default state).
pub fn micro_mcts_params() -> Option<MicroParams> {
    static PARAMS: std::sync::OnceLock<Option<MicroParams>> = std::sync::OnceLock::new();
    *PARAMS.get_or_init(|| {
        let sims: usize = std::env::var("POLYFISH_MICRO_MCTS_SIMS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(MICRO_MCTS_DEFAULT_SIMS);
        if sims == 0 {
            return None;
        }
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
        let net_prior_w: f32 = std::env::var("POLYFISH_MICRO_MCTS_NET_PRIOR_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        Some(MicroParams { sims, depth, k, c_puct, net_prior_w })
    })
}

/// Diagnostic (temporary, not a standing feature): how often the tree's
/// argmax-visits pick actually disagrees with `rank_view`'s own top-ranked
/// candidate (index 0). If this stays at 0 across real games, the search is
/// not influencing move selection at all in practice, regardless of sims.
pub static MICRO_MCTS_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MICRO_MCTS_OVERRIDES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// EXP_ELO_079 diagnostic: emergent search depth actually reached per real
/// ply, not assumed from the unrelated old-GumbelMctsAgent depth/sims curve
/// cited in this module's own doc comment above. `DEPTH_SUM` / `CALLS` give
/// the mean max-depth-reached-by-any-sim per `micro_search_pick` call;
/// `MAX_DEPTH_SEEN` is the single deepest line found across the whole run.
pub static MICRO_MCTS_DEPTH_SUM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MICRO_MCTS_DEPTH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MICRO_MCTS_MAX_DEPTH_SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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

/// Floor on the population std used to normalize `softmax_priors`' input --
/// prevents a near-equal score cluster (std -> 0) from amplifying tiny raw
/// differences into an artificial spread. First-fit per this project's own
/// Q-gap dial convention (typically overshoots ~2x) -- untuned pending a
/// paired gauge (EXP_ELO_119).
const MICRO_SOFTMAX_STD_FLOOR: f32 = 1.0;

/// EXP_ELO_079/119: raw, un-normalized `score_move`/`rank_plies` scores
/// span wildly different scales depending on context (root candidates carry
/// full Φ pricing -- tens to low thousands; interior nodes use the raw
/// heuristic alone -- tens to low hundreds). Softmax at temperature 1 over
/// a RAW score gap of even ~20-30 points already collapses to a numerically
/// exact one-hot distribution (EXP_ELO_079 measured e^-110 ~ 1e-48 on a
/// real ply; EXP_ELO_119 independently reproduced prior=1.000000 on the
/// top candidate, 0.000000 on 7 others, on an unrelated ply), zeroing
/// PUCT's exploration term for every other candidate for the rest of the
/// search regardless of sims/k/depth -- the search can then never disagree
/// with its own root prior. Normalizing by the score list's OWN spread
/// (population std, floored) before the softmax keeps prior concentration
/// a function of RELATIVE preference within this specific candidate set,
/// not the arbitrary absolute unit scale that set happens to carry -- a
/// genuine outlier (many std-devs clear of the rest) still collapses close
/// to one-hot, which is correct; a modest real gap no longer
/// catastrophically zeroes every alternative.
fn softmax_priors(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    if scores.len() == 1 {
        return vec![1.0];
    }
    let mean = scores.iter().sum::<f32>() / scores.len() as f32;
    let var =
        scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
    let std = var.sqrt().max(MICRO_SOFTMAX_STD_FLOOR);
    let scaled: Vec<f32> = scores.iter().map(|s| (s - mean) / std).collect();
    let max = scaled.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|s| (s - max).exp()).collect();
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

/// Returns `(leaf value, absolute depth reached along this path)` — depth
/// is `params.depth - depth_remaining` at the point a leaf is hit, used by
/// `micro_search_pick` to measure the search's real emergent depth
/// (EXP_ELO_079) instead of assuming it from the unrelated GumbelMctsAgent
/// curve cited in this module's own doc comment.
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
) -> (f32, usize) {
    let current_depth = params.depth - depth_remaining;
    if node.is_terminal || depth_remaining == 0 {
        return (leaf_value(&node.game, pov, goal, evaluator), current_depth);
    }
    if node.children.is_empty() {
        let cands = cheap_candidates(&node.game, goal, star_gate, aux, params.k);
        if cands.len() == 1 && cands[0].0.move_type() == MoveType::EndTurn {
            // Turn is genuinely over: nothing left to explore, and we never
            // simulate past our own EndTurn into the opponent's turn.
            node.is_terminal = true;
            return (leaf_value(&node.game, pov, goal, evaluator), current_depth);
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

    let (value, depth) = if node.children[best_idx].node.is_none() {
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
        (v, current_depth + 1)
    } else {
        let child = node.children[best_idx].node.as_mut().unwrap();
        select_and_expand(child, pov, goal, star_gate, aux, evaluator, params, depth_remaining - 1)
    };
    node.visits += 1;
    node.value_sum += value;
    (value, depth)
}

/// Root children are `rank_view`'s own top candidates (already paid for,
/// full Δφ fidelity) -- only nodes below the root use `cheap_candidates`.
/// `carry`, if it matches one of this ply's candidates by move identity, is
/// spliced in as that child's already-explored subtree (root advancement --
/// a free warm start instead of discarding a ply's search every ply).
/// Returns `(index into `ranked` the search prefers, subtree to carry into
/// the NEXT ply if the caller ends up actually playing that pick, the
/// picked child's own backed-up Q)`. The index is `None` when there's
/// nothing to search (a lone EndTurn, or too few candidates); the carry and
/// Q are `None` whenever no search ran. The Q is `tree(V_net)` -- leaves are
/// scored by the trained value head (see `leaf_value`), so this is a
/// genuine per-ply self-distillation target, computed on nearly every real
/// ply already (search runs regardless; only the return value was new).
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
) -> (Option<usize>, Option<MicroTreeCarry>, Option<f32>) {
    if ranked.len() < 2 || ranked[0].1.move_type() == MoveType::EndTurn {
        return (None, None, None);
    }
    if carry.is_some() {
        MICRO_CARRY_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let top_n = ranked.len().min(params.k.max(4));
    let scores: Vec<f32> = ranked.iter().take(top_n).map(|(s, _)| *s).collect();
    let mut priors = softmax_priors(&scores);
    // EXP_ELO_124-family: blend in a net-derived prior over the same top_n
    // candidates. Goal painted here is the ply's own COMMITTED macro goal --
    // the exact convention real training rows already use (`game.rs`'s
    // `feat_goal`), so this carries none of macro's root-painting-mismatch
    // risk. The four decomposed heads this reads are already behavior-cloned
    // on macro-mcts's own committed picks (`brain.rs`).
    if params.net_prior_w > 0.0 {
        if let Some(raw) = crate::ai::features::state_to_cpu_features_goal(&view.state, pov, None, Some(goal))
            .ok()
            .and_then(|f| evaluator.evaluate(vec![f]).into_iter().next().map(|r| r.2))
        {
            let top_n_moves: Vec<Box<dyn Move>> = ranked
                .iter()
                .take(top_n)
                .map(|(_, mv)| dyn_clone::clone_box(mv.as_ref()))
                .collect();
            let map_size = view.state.settings.size as usize;
            let net_priors = crate::ai::search::policy_composer::compute_move_priors_raw(
                &raw,
                &top_n_moves,
                map_size,
                false,
            );
            let net_sum: f32 = net_priors.iter().sum();
            if net_sum > 0.0 && net_priors.len() == priors.len() {
                let w = params.net_prior_w;
                for (p, np) in priors.iter_mut().zip(net_priors.iter()) {
                    *p = (1.0 - w) * *p + w * (np / net_sum);
                }
                let renorm: f32 = priors.iter().sum();
                if renorm > 0.0 {
                    for p in priors.iter_mut() {
                        *p /= renorm;
                    }
                }
            }
        }
    }
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
    let mut max_depth_this_call: usize = 0;
    for _ in 0..params.sims {
        let (_, d) = select_and_expand(&mut root, pov, goal, star_gate, aux, evaluator, params, params.depth);
        max_depth_this_call = max_depth_this_call.max(d);
    }
    MICRO_MCTS_DEPTH_SUM.fetch_add(max_depth_this_call as u64, std::sync::atomic::Ordering::Relaxed);
    MICRO_MCTS_DEPTH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    MICRO_MCTS_MAX_DEPTH_SEEN.fetch_max(max_depth_this_call as u64, std::sync::atomic::Ordering::Relaxed);
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
    let picked_q = root.children[best_idx]
        .node
        .as_ref()
        .filter(|n| n.visits > 0)
        .map(|n| n.q().clamp(-1.0, 1.0));
    let mv_key = root.children[best_idx].mv.serialize();
    let grandchildren = root.children[best_idx].node.take().map(|node| node.children).unwrap_or_default();
    let next_carry =
        if grandchildren.is_empty() { None } else { Some(MicroTreeCarry { mv_key, children: grandchildren }) };
    (Some(best_idx), next_carry, picked_q)
}

#[cfg(test)]
mod tests {
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
        let params = MicroParams { sims: 64, depth: 64, k: 4, c_puct: 1.5, net_prior_w: 0.0 };

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
            micro_search_pick(&view, pov, &goal, &ranked, &aux, star_gate, &evaluator, &params, None);
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
}
