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
//!
//! Wave batching (`MicroParams::leaf_batch`, see its own doc comment):
//! `micro_search_pick` collects up to `leaf_batch` simulations' worth of
//! descents (`micro_collect_wave`) before making one batched
//! `evaluator.evaluate()` call and backing every leaf up
//! (`micro_resolve_wave`) — `leaf_batch == 1` is byte-identical to the old
//! strictly-sequential loop this replaced.

use crate::ai::eval_server::Evaluator;
use crate::ai::search::mcts_common::VIRTUAL_LOSS;
use crate::ai::oracle_macro::MacroGoal;
use crate::ai::search::goal_aux::GoalAux;
use crate::ai::search::macro_exec::gate_ok;
use crate::ai::scoring::score_move_cached;
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
    /// `rank_view`/Δφ-score softmax prior (0.0 = off, this struct's own
    /// default -- see `micro_mcts_params`'s entry-point default below).
    /// Mirrors `MacroParams::root_prior_w`'s convention. The goal painted for
    /// this eval call is the ply's own COMMITTED macro goal -- the same
    /// convention real training rows already use (`game.rs`'s `feat_goal`),
    /// so unlike macro's root prior this carries no root-painting-mismatch
    /// risk: the four decomposed policy heads it reads are already
    /// behavior-cloned on macro-mcts's own committed picks (`brain.rs`).
    ///
    /// A complete no-op whenever `micro_search_pick`'s own
    /// `root_already_net_ranked` is true (the `net_root` rework) -- `ranked`
    /// already reflects a net+heuristic blend in that case, so there is
    /// nothing left for this field's second forward pass to add.
    pub net_prior_w: f32,
    /// EXP_ELO_150: spend the first `min(sims, num_root_children)` sims
    /// guaranteeing every root child at least one real visit (in `idxs`
    /// order, i.e. heuristic-score order) before falling through to normal
    /// PUCT selection for whatever sims remain. At the tiny production
    /// `sims` budget (8, often against 4-6 children), a child that never
    /// gets picked by PUCT's own path-dependent early selection can end up
    /// with 0 visits all search — the FIRST sim's winner (decided by raw
    /// prior magnitude with every Q at FPU=0.0) then tends to keep
    /// accumulating visits regardless of whether it was actually good,
    /// since PUCT's exploration bonus for an unvisited sibling only grows
    /// as `sqrt(total_visits)`, which barely moves in a handful of sims.
    /// Measured directly on real games (EXP_ELO_150): 28/176 (15.9%) of
    /// Step moves by a unit with a live Expand goal picked a candidate the
    /// heuristic itself scored far worse than an available alternative
    /// (median gap in the hundreds of points) while moving away from or
    /// sideways to the goal. `false` (default) is a byte-identical no-op.
    pub forced_playouts: bool,
    /// EXP_ELO_153: weight on a root-only goal-alignment bonus folded into
    /// each candidate's score before `softmax_priors` -- NOT into `ranked`'s
    /// own stored score, so this only reshapes PUCT's exploration weighting
    /// for this search, nothing else that reads `ranked`. For each of the
    /// (up to `k`) root candidates, adds `goal_prior_w * (goal_potential(post)
    /// - goal_potential(pre))`, same telescoping-Δφ convention `macro_mcts.rs`'s
    /// `expand_execute` already uses (SAME `aux` for both terms -- reusing
    /// the caller's own `aux`, never recomputed post-move, so a directive
    /// switch can't mint reward). Verdi's diagnosis (Sep 13, 2026 session):
    /// `net_rank_root_candidates` (the real per-ply commit's shipped root
    /// source since EXP_ELO_151) carries no goal_potential term at all --
    /// `macro_exec::rank_plies`, the only place that pricing lives, is now a
    /// rare failure-fallback, not the live path. This is a narrower,
    /// differently-scoped bet than the macro-level directive-selection
    /// shaping this project tried and rejected three times (EXP_036b/037/038
    /// double-paid leaf-visible terms deciding WHICH plan to commit to) --
    /// this only asks whether an ALREADY-committed plan's own target gets
    /// pursued, the same shape as EXP_ELO_150's clean per-unit-goal fix.
    /// `0.0` (default) is a byte-identical no-op.
    pub goal_prior_w: f32,
    /// Leaves coalesced into one evaluator.evaluate() call per wave, mirroring
    /// `MacroParams::leaf_batch`'s convention exactly: `1` (default) is
    /// today's strictly-sequential sim loop, byte-identical -- every existing
    /// test constructs `MicroParams` with this at 1 and keeps passing
    /// unchanged. Values `>1` change move-selection behavior via virtual loss
    /// (see `select_child`'s doc comment and `collect_wave`), not just
    /// throughput: sims spent widening a wave's root-candidate coverage
    /// instead of deepening one line trade search depth for round-trip count.
    /// At the tiny production `sims` budget (8), a large `leaf_batch` can
    /// leave little room for the tree to differentiate at all past the first
    /// wave -- start small (2-4) when validating, not `sims` itself.
    pub leaf_batch: usize,
}

impl Default for MicroParams {
    fn default() -> Self {
        Self {
            sims: 16,
            depth: 64,
            k: 4,
            c_puct: 1.5,
            net_prior_w: 0.0,
            forced_playouts: false,
            goal_prior_w: 0.0,
            leaf_batch: 1,
        }
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
        // EXP_ELO_125 piece 1 (2026-09-05): on by default at 0.3 -- the one
        // weight actually smoke-tested (override rate 10.3% -> 21.1% vs the
        // Δφ-only prior, confirming the blend is genuinely behaviorally
        // active, not a no-op). Not yet arena-validated for win-rate impact
        // (see MICRO_MCTS_DEFAULT_SIMS's own doc comment above for the same
        // caveat pattern) -- Verdi's explicit "make it default ON" call.
        let net_prior_w: f32 = std::env::var("POLYFISH_MICRO_MCTS_NET_PRIOR_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.3);
        // EXP_ELO_150: opt-in, default off (presence-based, matching this
        // project's other boolean env flags e.g. WL_LABELS) -- see
        // `MicroParams::forced_playouts`'s own doc for the mechanism.
        let forced_playouts = std::env::var("POLYFISH_MICRO_FORCED_PLAYOUTS").is_ok();
        // EXP_ELO_153: opt-in, default off -- see `MicroParams::goal_prior_w`'s
        // own doc for the mechanism and why it isn't on by default yet.
        let goal_prior_w: f32 = std::env::var("POLYFISH_MICRO_MCTS_GOAL_PRIOR_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        // Inert by default (1 = sequential, byte-identical) -- see
        // `MicroParams::leaf_batch`'s own doc comment.
        let leaf_batch: usize = std::env::var("POLYFISH_MICRO_MCTS_LEAF_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        Some(MicroParams { sims, depth, k, c_puct, net_prior_w, forced_playouts, goal_prior_w, leaf_batch })
    })
}

/// EXP_ELO_136 (Experiment B): turn-conditional PUCT trust ramp. Value-head
/// discrimination is genuinely poor before turn ~10 (EXP_ELO_135: r2
/// 0.06-0.35 vs `final_outcome`, worse than the heuristic's own 0.14-0.40 in
/// the same bands) and only becomes load-bearing from turn ~10-20 onward
/// (r2 0.61-0.72). PUCT's `s = q + u` makes trust-weighting Q algebraically
/// equivalent to dividing `c_puct` by that same weight, so low trust is
/// implemented as a HIGHER `c_puct` (more exploration, less Q-reliance) --
/// `early` is the LARGER number, ramping DOWN to `late` as turn increases.
/// Returns `base` unchanged (today's exact behavior, byte-identical) unless
/// all four env vars parse; caller overrides its own `MicroParams.c_puct`
/// copy with the result right before calling `micro_search_pick`.
pub fn turn_conditional_c_puct(turn: i32, base: f32) -> f32 {
    static RAMP: std::sync::OnceLock<Option<(f32, f32, f32, f32)>> = std::sync::OnceLock::new();
    let ramp = RAMP.get_or_init(|| {
        let early: f32 = std::env::var("POLYFISH_MICRO_MCTS_CPUCT_EARLY").ok()?.parse().ok()?;
        let late: f32 = std::env::var("POLYFISH_MICRO_MCTS_CPUCT_LATE").ok()?.parse().ok()?;
        let turn_start: f32 = std::env::var("POLYFISH_MICRO_MCTS_CPUCT_TURN_START").ok()?.parse().ok()?;
        let turn_end: f32 = std::env::var("POLYFISH_MICRO_MCTS_CPUCT_TURN_END").ok()?.parse().ok()?;
        Some((early, late, turn_start, turn_end))
    });
    match *ramp {
        None => base,
        Some((early, late, turn_start, turn_end)) => {
            let t = turn as f32;
            if t <= turn_start {
                early
            } else if t >= turn_end {
                late
            } else {
                let frac = (t - turn_start) / (turn_end - turn_start);
                early + frac * (late - early)
            }
        }
    }
}

/// Diagnostic (temporary, not a standing feature): how often the tree's
/// argmax-visits pick actually disagrees with `rank_view`'s own top-ranked
/// candidate (index 0). If this stays at 0 across real games, the search is
/// not influencing move selection at all in practice, regardless of sims.
pub static MICRO_MCTS_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MICRO_MCTS_OVERRIDES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// `net_root` rework: how many `micro_search_pick` calls received an
/// already net-derived `ranked` (see `root_already_net_ranked`) -- cheap
/// live confirmation during a gauge run that the net path is actually being
/// exercised, not silently falling back to CPU `rank_plies`.
pub static MICRO_MCTS_NET_DERIVED_ROOTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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

/// EXP_ELO_126 diagnostic: the net's own prior used to be computed ONLY
/// over `rank_view`'s CPU-heuristic top-k, so a move the net actually
/// preferred could never enter the search if the heuristic ranked it lower
/// -- the heuristic silently gated what the net-driven layer was allowed to
/// consider. `WIDENED` counts plies where the net's own top-k candidates
/// (decoded over the FULL legal-move list) added at least one move outside
/// the heuristic's own top-k; `WON` counts plies where such a net-only
/// candidate was the one the search actually picked. Both are 0 whenever
/// `net_prior_w == 0.0` (the widening is fully gated off in that case).
pub static MICRO_MCTS_UNION_WIDENED: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub static MICRO_MCTS_UNION_WON: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    /// Wave-batching: phantom visit weight charged while a pending leaf
    /// beneath this edge hasn't backed up yet, removed once it does. Always
    /// 0.0 between waves (every wave collected is fully resolved before the
    /// next starts, or before `micro_search_pick` returns) -- so a MicroChild
    /// freshly built at the root, or a subtree carried in from `MicroTreeCarry`,
    /// is always born/received at 0.0. Zero everywhere makes `select_child`
    /// reduce byte-for-byte to the pre-wave-batching formula.
    virtual_loss: f32,
}

/// Debug-only snapshot of one root child, before vs. after the PUCT search:
/// the prior it was seeded with, and its post-search visit count/backed-up Q
/// (`None` if the search never visited it). Verdi, Sep 7 2026 -- added for
/// per-ply debugging traceability (root-seed vs. post-search), not read by
/// any production decision path.
#[derive(Clone, Debug)]
pub struct MicroChildTrace {
    pub mv: String,
    pub prior: f32,
    pub visits: u32,
    pub q: Option<f32>,
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
    let road_cache = crate::ai::movement::RoadReliefCache::default();
    let mut scored: Vec<(Box<dyn Move>, f32)> = moves
        .into_iter()
        .map(|m| {
            let s = score_move_cached(game, m.as_ref(), &road_cache);
            (m, s)
        })
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k.max(1));
    scored
}


/// Virtual-loss-aware PUCT selection over `node.children` (cold-start FPU =
/// 0.0, matching the pre-wave-batching formula exactly). Reduces byte-for-
/// byte to that formula whenever every child's `virtual_loss` is 0.0 --
/// always true between waves, so `leaf_batch == 1` search is unaffected.
fn select_child(node: &MicroNode, c_puct: f32) -> usize {
    let total_eff: f32 = node
        .children
        .iter()
        .map(|c| c.node.as_ref().map_or(0.0, |n| n.visits as f32) + c.virtual_loss)
        .sum();
    let mut best_idx = 0;
    let mut best_score = f32::MIN;
    for (i, c) in node.children.iter().enumerate() {
        let (value_sum, real_n) =
            c.node.as_ref().map_or((0.0, 0.0), |n| (n.value_sum, n.visits as f32));
        let n_eff = real_n + c.virtual_loss;
        // Each phantom visit is charged the pessimistic value -1 -- same
        // convention as `macro_mcts.rs`'s `edge_values[i] - edge_virtual_loss[i]`
        // (`VIRTUAL_LOSS == 1.0` makes "amount charged" and "phantom visit
        // count" the same number, so this subtraction and the `n_eff` in the
        // denominator stay consistent).
        let q = if n_eff > 0.0 { (value_sum - c.virtual_loss) / n_eff } else { 0.0 };
        let u = c_puct * c.prior * total_eff.max(1.0).sqrt() / (1.0 + n_eff);
        let s = q + u;
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }
    best_idx
}

/// Walks `path`'s child indices from `root`, returning the node reached.
/// Every element of a `micro_descend_once` path names an already-real node
/// to walk THROUGH except possibly the last, which may still be `None` (a
/// brand-new leaf) -- callers pass `&path[..path.len() - 1]` in that case.
fn node_at_path_mut<'a>(root: &'a mut MicroNode, path: &[usize]) -> &'a mut MicroNode {
    let mut cursor = root;
    for &idx in path {
        cursor = cursor.children[idx].node.as_mut().unwrap();
    }
    cursor
}

fn node_at_path<'a>(root: &'a MicroNode, path: &[usize]) -> &'a MicroNode {
    let mut cursor = root;
    for &idx in path {
        cursor = cursor.children[idx].node.as_ref().unwrap();
    }
    cursor
}

/// One root-to-frontier descent using `select_child` at every level, and
/// `path[i]` is the child index chosen at tree depth `i` -- walking it via
/// `node_at_path`/`node_at_path_mut` always reaches the same node. Charges
/// `VIRTUAL_LOSS` on every edge walked, unconditionally, mirroring
/// `macro_mcts.rs`'s own wave-batching descent: the charge does not depend
/// on how the descent ends, so `micro_apply_leaf`'s removal is always
/// exactly paired.
///
/// Stops at the first node needing a fresh `leaf_value` call: either a
/// brand-new child (`node: None`, returned as `Some((child_game, terminal))`
/// so `micro_apply_leaf` can materialize it once scored) or an ALREADY-real
/// node being re-visited because it's terminal or depth-capped (`None` --
/// mirrors the pre-wave-batching design's behavior of re-scoring such a node
/// on every visit without touching ITS OWN stored visits/value_sum, only its
/// ancestors').
fn micro_descend_once(
    root: &mut MicroNode,
    pov: PlayerId,
    goal: &MacroGoal,
    star_gate: bool,
    aux: &GoalAux,
    params: &MicroParams,
) -> (Vec<usize>, Option<(Game, bool)>) {
    let mut path: Vec<usize> = Vec::new();
    let mut cursor: &mut MicroNode = root;
    loop {
        if cursor.is_terminal || path.len() >= params.depth {
            return (path, None);
        }
        if cursor.children.is_empty() {
            let cands = cheap_candidates(&cursor.game, goal, star_gate, aux, params.k);
            if cands.len() == 1 && cands[0].0.move_type() == MoveType::EndTurn {
                // Turn is genuinely over: nothing left to explore, and we
                // never simulate past our own EndTurn into the opponent's.
                cursor.is_terminal = true;
                return (path, None);
            }
            let scores: Vec<f32> = cands.iter().map(|(_, s)| *s).collect();
            let priors = softmax_priors(&scores);
            cursor.children = cands
                .into_iter()
                .zip(priors)
                .map(|((mv, _), prior)| MicroChild { mv, prior, node: None, virtual_loss: 0.0 })
                .collect();
        }
        let best_idx = select_child(cursor, params.c_puct);
        cursor.children[best_idx].virtual_loss += VIRTUAL_LOSS;
        path.push(best_idx);
        if cursor.children[best_idx].node.is_none() {
            let mut child_game = cursor.game.clone();
            let ok = child_game.simulate_move(cursor.children[best_idx].mv.as_ref()).is_some();
            let terminal = !ok || child_game.state.settings.current_player_turn_id != pov;
            return (path, Some((child_game, terminal)));
        }
        cursor = cursor.children[best_idx].node.as_mut().unwrap();
    }
}

/// One in-flight leaf awaiting a batched eval call, mirroring
/// `macro_mcts.rs::PendingLeaf`. `new_leaf` is `Some` when the descent
/// stopped at a not-yet-created child (its simulated game + whether it's
/// already terminal, ready to materialize once scored); `None` when it
/// stopped at an already-real node being re-visited (see
/// `micro_descend_once`'s doc). `count`: a repeat pick of the exact same
/// path within one wave bumps this instead of adding a second entry, so its
/// one real result backs up multiple times.
struct MicroPendingLeaf {
    path: Vec<usize>,
    new_leaf: Option<(Game, bool)>,
    feat_offset: usize,
    count: u32,
}

/// Materializes a pending leaf (if new) and backs up `value` `count` times:
/// `visits += count, value_sum += value * count` on every node from `root`
/// down to (not including) the leaf itself, and removes the
/// `VIRTUAL_LOSS * count` charged along the full path during collection.
/// The leaf's OWN stored visits/value_sum are set once at creation and never
/// touched again on a later re-visit -- matches the pre-wave-batching
/// design, see `MicroPendingLeaf`'s doc.
fn micro_apply_leaf(root: &mut MicroNode, path: &[usize], new_leaf: Option<(Game, bool)>, value: f32, count: u32) {
    if path.is_empty() {
        // Only reachable with `params.depth == 0` (root itself can't be
        // `is_terminal` at construction) -- the pre-wave-batching top-level
        // call's own early return touched no node's counters at all in this
        // case; preserve that exactly.
        return;
    }
    if let Some((game, terminal)) = new_leaf {
        let parent = node_at_path_mut(root, &path[..path.len() - 1]);
        let last = *path.last().unwrap();
        parent.children[last].node = Some(MicroNode {
            game,
            visits: count,
            value_sum: value * count as f32,
            children: Vec::new(),
            is_terminal: terminal,
        });
    }
    let mut cursor = root;
    cursor.visits += count;
    cursor.value_sum += value * count as f32;
    for &idx in &path[..path.len() - 1] {
        cursor.children[idx].virtual_loss -= VIRTUAL_LOSS * count as f32;
        cursor = cursor.children[idx].node.as_mut().unwrap();
        cursor.visits += count;
        cursor.value_sum += value * count as f32;
    }
    let last = *path.last().unwrap();
    cursor.children[last].virtual_loss -= VIRTUAL_LOSS * count as f32;
}

/// Collects up to `budget` simulations' worth of descents in one wave:
/// repeated `micro_descend_once` calls, immediately backing up any leaf
/// whose features fail to build (mirrors `leaf_value`'s own `unwrap_or(0.0)`
/// fallback -- no eval call is possible for it either way), and
/// accumulating every other leaf (plus its feature row) for one batched
/// eval call. Returns the count of immediately-resolved sims plus whatever's
/// left pending for `micro_resolve_wave`.
#[allow(clippy::too_many_arguments)]
fn micro_collect_wave(
    root: &mut MicroNode,
    pov: PlayerId,
    goal: &MacroGoal,
    star_gate: bool,
    aux: &GoalAux,
    params: &MicroParams,
    budget: usize,
) -> (u32, Vec<MicroPendingLeaf>, Vec<crate::ai::features::RawFeatures>) {
    let mut immediate = 0u32;
    let mut pending: Vec<MicroPendingLeaf> = Vec::new();
    let mut features: Vec<crate::ai::features::RawFeatures> = Vec::new();
    let mut dedup: std::collections::HashMap<Vec<usize>, usize> = std::collections::HashMap::new();
    loop {
        let done = immediate as usize + pending.iter().map(|p| p.count as usize).sum::<usize>();
        if done >= budget {
            break;
        }
        let (path, new_leaf) = micro_descend_once(root, pov, goal, star_gate, aux, params);
        if let Some(&pi) = dedup.get(&path) {
            pending[pi].count += 1;
            continue;
        }
        let feats = match &new_leaf {
            Some((game, _)) => {
                crate::ai::features::state_to_cpu_features_goal(&game.state, pov, None, Some(goal)).ok()
            }
            None => {
                let n = node_at_path(root, &path);
                crate::ai::features::state_to_cpu_features_goal(&n.game.state, pov, None, Some(goal)).ok()
            }
        };
        match feats {
            Some(f) => {
                let feat_offset = features.len();
                features.push(f);
                dedup.insert(path.clone(), pending.len());
                pending.push(MicroPendingLeaf { path, new_leaf, feat_offset, count: 1 });
            }
            None => {
                micro_apply_leaf(root, &path, new_leaf, 0.0, 1);
                immediate += 1;
            }
        }
    }
    (immediate, pending, features)
}

/// The one batched `evaluator.evaluate()` call for a wave's pending leaves,
/// then applies each one's real result via `micro_apply_leaf` (`leaf.count`
/// times each, undoing exactly the virtual loss its `count` descents
/// charged).
fn micro_resolve_wave(
    root: &mut MicroNode,
    pending: Vec<MicroPendingLeaf>,
    features: Vec<crate::ai::features::RawFeatures>,
    evaluator: &Evaluator,
) {
    let results = evaluator.evaluate(features);
    for leaf in pending {
        let v = results.get(leaf.feat_offset).map(|r| r.0).unwrap_or(0.0);
        micro_apply_leaf(root, &leaf.path, leaf.new_leaf, v, leaf.count);
    }
}

/// Root children are `rank_view`'s own top candidates (already paid for,
/// full Δφ fidelity), widened by the net's own top-k when `net_prior_w >
/// 0.0` (EXP_ELO_126 -- see the union-building block below) -- only nodes
/// below the root use `cheap_candidates`. `carry`, if it matches one of
/// this ply's candidates by move identity, is spliced in as that child's
/// already-explored subtree (root advancement -- a free warm start instead
/// of discarding a ply's search every ply).
/// Returns `(index into `ranked` the search prefers, subtree to carry into
/// the NEXT ply if the caller ends up actually playing that pick, the
/// picked child's own backed-up Q, a per-root-child debug trace, the
/// post-search visit distribution over root children)`. The index is `None`
/// when there's nothing to search (a lone EndTurn, or too few candidates);
/// the carry and Q are `None` whenever no search ran. The Q is `tree(V_net)`
/// -- leaves are scored by the trained value head (see `leaf_value`), so
/// this is a genuine per-ply self-distillation target, computed on nearly
/// every real ply already (search runs regardless; only the return value
/// was new). The trace vec and the visit distribution are both empty
/// exactly when no search ran, one entry per root child otherwise -- cheap
/// to build (already-owned data, no new allocation of note next to the sims
/// loop itself), so both are unconditional rather than flag-gated; callers
/// that don't need them just drop them.
///
/// EXP_ELO_155: the visit distribution (`Vec<MoveVisit>`, real post-search
/// counts) exists specifically so a caller can distill the search's actual
/// preference instead of collapsing it to a one-hot label on the single
/// winner -- see `brain.rs`'s `SearchAgent::MacroMcts` arm of
/// `select_move_with_decomposed_visits`, the one caller that uses it this
/// way today.
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
    root_already_net_ranked: bool,
) -> (Option<usize>, Option<MicroTreeCarry>, Option<f32>, Vec<MicroChildTrace>, Vec<crate::ai::mcts_types::MoveVisit>) {
    if ranked.len() < 2 || ranked[0].1.move_type() == MoveType::EndTurn {
        return (None, None, None, Vec::new(), Vec::new());
    }
    if root_already_net_ranked {
        MICRO_MCTS_NET_DERIVED_ROOTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if carry.is_some() {
        MICRO_CARRY_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let heur_top = ranked.len().min(params.k.max(4));
    // `idxs`: the candidate set as indices into `ranked` -- the heuristic's
    // own top-k first, in order (so `idxs[i] == i` for `i < heur_top`,
    // which is exactly what makes the net_prior_w == 0.0 path below
    // byte-identical to this function's pre-EXP_ELO_126 behavior). Any
    // net-only candidates get appended after.
    let mut idxs: Vec<usize> = (0..heur_top).collect();
    // EXP_ELO_126: the net's own prior used to be computed ONLY over this
    // same heuristic top-k, so a move the net actually preferred could
    // never even enter the search if the CPU Δφ heuristic ranked it lower
    // -- the heuristic silently gated what the net-driven layer was
    // allowed to consider. Fix: decode the net's prior over the FULL
    // legal-move list instead (the network forward pass already ran once
    // regardless of candidate count -- `compute_move_priors_raw` is a
    // cheap per-move readout off that one already-computed tensor, not an
    // extra network call), take the net's own top-k, and union it into the
    // candidate set so a net-favored move can never be silently excluded.
    //
    // `net_root`-rework note: when `root_already_net_ranked` is true,
    // `ranked` was ALREADY built from one net forward pass + a heuristic
    // blend (see `net_root::net_rank_root_candidates`) -- this whole block
    // exists only to bridge a heuristic-derived `ranked` with the net's own
    // opinion, so there is nothing left for a second forward pass to add.
    // Skipping it here also removes a redundant forward pass that used to
    // run even when the ranker had already supplied `ranked`.
    let mut net_priors_full: Option<Vec<f32>> = None;
    if !root_already_net_ranked && params.net_prior_w > 0.0 {
        if let Some(raw) = crate::ai::features::state_to_cpu_features_goal(&view.state, pov, None, Some(goal))
            .ok()
            .and_then(|f| evaluator.evaluate(vec![f]).into_iter().next().map(|r| r.2))
        {
            let all_moves: Vec<Box<dyn Move>> =
                ranked.iter().map(|(_, mv)| dyn_clone::clone_box(mv.as_ref())).collect();
            let map_size = view.state.settings.size as usize;
            let np = crate::ai::search::policy_composer::compute_move_priors_raw(
                &raw, &all_moves, map_size, false,
            );
            if np.len() == ranked.len() {
                // Deterministic tie-break (lower original index wins) --
                // same-seed reproducibility is a resolved-bug invariant in
                // this repo (EXP_ELO_091); an unstable sort here would
                // regress it invisibly.
                let mut net_order: Vec<usize> = (0..np.len()).collect();
                net_order.sort_by(|&a, &b| {
                    np[b].partial_cmp(&np[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
                });
                let mut widened = false;
                for &i in net_order.iter().take(heur_top) {
                    if !idxs.contains(&i) {
                        idxs.push(i);
                        widened = true;
                    }
                }
                if widened {
                    MICRO_MCTS_UNION_WIDENED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                net_priors_full = Some(np);
            }
        }
    }
    let mut scores: Vec<f32> = idxs.iter().map(|&i| ranked[i].0).collect();
    // EXP_ELO_153: root-only goal-alignment bonus -- see
    // `MicroParams::goal_prior_w`'s own doc comment for the mechanism and
    // why this is scoped to `scores` (exploration weighting) rather than
    // `ranked`'s own stored score. `k` disposable clones, real-ply-only
    // (never inside a macro-mcts rollout), same cost class as the net
    // forward pass this function already does.
    if params.goal_prior_w != 0.0 {
        let phi_pre = crate::ai::reward::goal_potential(&view.state, pov, goal, Some(aux));
        for (score, &i) in scores.iter_mut().zip(idxs.iter()) {
            let mut probe = view.clone();
            if probe.simulate_move(ranked[i].1.as_ref()).is_some() {
                let phi_post = crate::ai::reward::goal_potential(&probe.state, pov, goal, Some(aux));
                *score += params.goal_prior_w * (phi_post - phi_pre);
            }
        }
    }
    let mut priors = softmax_priors(&scores);
    // EXP_ELO_124-family: blend in a net-derived prior over the (now
    // possibly widened) candidate set. Goal painted here is the ply's own
    // COMMITTED macro goal -- the exact convention real training rows
    // already use (`game.rs`'s `feat_goal`), so this carries none of
    // macro's root-painting-mismatch risk. The four decomposed heads this
    // reads are already behavior-cloned on macro-mcts's own committed
    // picks (`brain.rs`).
    if let Some(np) = &net_priors_full {
        let net_sub: Vec<f32> = idxs.iter().map(|&i| np[i]).collect();
        let net_sum: f32 = net_sub.iter().sum();
        if net_sum > 0.0 {
            let w = params.net_prior_w;
            for (p, snp) in priors.iter_mut().zip(net_sub.iter()) {
                *p = (1.0 - w) * *p + w * (snp / net_sum);
            }
            let renorm: f32 = priors.iter().sum();
            if renorm > 0.0 {
                for p in priors.iter_mut() {
                    *p /= renorm;
                }
            }
        }
    }
    // The carry's own `mv_key` is the move that was just played -- already
    // consumed, and can't reappear here. What can reappear (and is worth
    // matching) is the set of options already explored one ply below it.
    let mut carried_children = carry.map(|c| c.children).unwrap_or_default();
    let children: Vec<MicroChild> = idxs
        .iter()
        .zip(priors)
        .map(|(&orig_idx, prior)| {
            let mv = &ranked[orig_idx].1;
            let key = mv.serialize();
            let node = if let Some(pos) =
                carried_children.iter().position(|c| c.mv.serialize() == key)
            {
                MICRO_CARRY_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                carried_children.remove(pos).node
            } else {
                None
            };
            MicroChild { mv: dyn_clone::clone_box(mv.as_ref()), prior, node, virtual_loss: 0.0 }
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
    let mut spent_sims = 0usize;
    if params.forced_playouts {
        // One deterministic pass over every not-yet-visited root child (no
        // PUCT/virtual-loss dance needed -- each is targeted at most once by
        // construction), batched into a single eval call.
        let mut feats: Vec<crate::ai::features::RawFeatures> = Vec::new();
        let mut forced_pending: Vec<(usize, Game, bool)> = Vec::new();
        for i in 0..root.children.len() {
            if spent_sims >= params.sims {
                break;
            }
            if root.children[i].node.is_some() {
                continue; // already visited this ply, or carried over from last ply
            }
            let mut child_game = root.game.clone();
            let ok = child_game.simulate_move(root.children[i].mv.as_ref()).is_some();
            let terminal = !ok || child_game.state.settings.current_player_turn_id != pov;
            match crate::ai::features::state_to_cpu_features_goal(&child_game.state, pov, None, Some(goal)).ok() {
                Some(f) => {
                    feats.push(f);
                    forced_pending.push((i, child_game, terminal));
                }
                None => {
                    root.children[i].node = Some(MicroNode {
                        game: child_game,
                        visits: 1,
                        value_sum: 0.0,
                        children: Vec::new(),
                        is_terminal: terminal,
                    });
                    root.visits += 1;
                    max_depth_this_call = max_depth_this_call.max(1);
                }
            }
            spent_sims += 1;
        }
        if !forced_pending.is_empty() {
            let results = evaluator.evaluate(feats);
            for (offset, (i, child_game, terminal)) in forced_pending.into_iter().enumerate() {
                let v = results.get(offset).map(|r| r.0).unwrap_or(0.0);
                root.children[i].node = Some(MicroNode {
                    game: child_game,
                    visits: 1,
                    value_sum: v,
                    children: Vec::new(),
                    is_terminal: terminal,
                });
                root.visits += 1;
                root.value_sum += v;
                max_depth_this_call = max_depth_this_call.max(1);
            }
        }
    }
    let batch = params.leaf_batch.max(1);
    let mut done = spent_sims;
    while done < params.sims {
        let want = (params.sims - done).min(batch);
        let (immediate, pending, features) =
            micro_collect_wave(&mut root, pov, goal, star_gate, aux, params, want);
        done += immediate as usize;
        max_depth_this_call = max_depth_this_call.max(1);
        if !pending.is_empty() {
            done += pending.iter().map(|p| p.count as usize).sum::<usize>();
            let deepest = pending.iter().map(|p| p.path.len()).max().unwrap_or(0);
            max_depth_this_call = max_depth_this_call.max(deepest);
            micro_resolve_wave(&mut root, pending, features, evaluator);
        }
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
    // `idxs[best_idx]`, NOT `best_idx` itself, is the caller's contract:
    // the caller (macro_mcts.rs's `select_move`) indexes `ranked` with the
    // returned value directly (`ranked.swap(0, idx)`). Under the union
    // above, `root.children`'s position no longer equals `ranked`'s
    // position once a net-only candidate is appended -- returning the raw
    // `best_idx` here would silently swap the wrong move into position 0
    // and execute a move the search never actually picked. See
    // `union_pick_maps_back_to_the_original_ranked_index` for the pinned
    // regression test.
    let picked_orig_idx = idxs[best_idx];
    if picked_orig_idx >= heur_top {
        MICRO_MCTS_UNION_WON.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    let picked_q = root.children[best_idx]
        .node
        .as_ref()
        .filter(|n| n.visits > 0)
        .map(|n| n.q().clamp(-1.0, 1.0));
    // Snapshot every root child's prior/visits/Q BEFORE consuming
    // `root.children[best_idx].node` below -- this is root-seed-vs-post-search,
    // not just the winner, so it has to happen before the `.take()`.
    let child_trace: Vec<MicroChildTrace> = root
        .children
        .iter()
        .map(|c| MicroChildTrace {
            mv: c.mv.serialize().to_string(),
            prior: c.prior,
            visits: c.node.as_ref().map_or(0, |n| n.visits),
            q: c.node.as_ref().filter(|n| n.visits > 0).map(|n| n.q().clamp(-1.0, 1.0)),
        })
        .collect();
    // EXP_ELO_155: the same snapshot, shaped for training distillation --
    // real `Move` objects (not the trace's serialized strings, which can't
    // be turned back into one) paired with real post-search visit counts.
    // Every root child is included even at 0 visits (PUCT's own cold-start
    // ordering means a low-prior candidate can legitimately go unvisited at
    // a tiny `sims` budget) -- `decompose_visits` sums by weight, so a 0
    // entry simply contributes nothing.
    let visit_dist: Vec<crate::ai::mcts_types::MoveVisit> = root
        .children
        .iter()
        .map(|c| {
            crate::ai::mcts_types::MoveVisit::weighted(
                c.mv.as_ref(),
                c.node.as_ref().map_or(0, |n| n.visits) as f32,
            )
        })
        .collect();
    let mv_key = root.children[best_idx].mv.serialize();
    let grandchildren = root.children[best_idx].node.take().map(|node| node.children).unwrap_or_default();
    let next_carry =
        if grandchildren.is_empty() { None } else { Some(MicroTreeCarry { mv_key, children: grandchildren }) };
    (Some(picked_orig_idx), next_carry, picked_q, child_trace, visit_dist)
}


#[cfg(test)]
#[path = "micro_mcts_tests.rs"]
mod tests;
