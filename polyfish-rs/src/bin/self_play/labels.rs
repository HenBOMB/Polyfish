//! Training targets computed after a game ends: the TD(lambda) value
//! label and every auxiliary head's ground truth (SPT, per-city SPT, tech
//! multi-hot, fogged enemy occupancy, tile ownership, macro ballot).
//!
//! Pure functions over recorded snapshots -- nothing here touches a live
//! `Game`, which is what makes the whole module unit-testable.

use polyfish::ai::features;
use polyfish::ai::reward;
use polyfish::states::{GameState, PlayerId};
use crate::result::HistoryStep;
use std::collections::HashMap;
use strum::IntoEnumIterator;

// Absolute yardstick for the value target; 8K points is a good place to end a 30T game
pub(crate) const GOOD_BOT_FINAL_SCORE: f32 = 8000.0;
// How much to weight relative (vs opponent) vs absolute (vs yardstick) final outcome.
// 1.0 = pure relative (zero-sum). The value backup negates across every
// player-turn boundary (mcts_common.rs), which is only valid when
// v(mine) = -v(theirs); an absolute own-progress component is NOT
// antisymmetric — the opponent's progress isn't my loss — so any abs share
// gets systematically corrupted through EndTurn-crossing lines, worse as
// search deepens. The mirror-play "empty relative label" problem is fixed in
// the DATA instead: anchor games vs the heuristic backend (--anchor-frac)
// make passivity actually lose, giving the relative label real signal.
pub(crate) const FINAL_OUTCOME_REL_W: f32 = 1.0;

// Weight of the TD(lambda) delta vs the final-outcome tail.
pub(crate) const TD_W: f32 = 0.7;
// Bootstrap/Monte-Carlo blend: center of mass of the geometric weights is
// 1/(1-LAMBDA_RETURN) turns (~5 at 0.8). Chosen to reach the turns-away
// horizon a village approach needs credit across, without drifting back
// toward the high-variance near-pure-MC regime the TD project escaped.
pub(crate) const LAMBDA_RETURN: f32 = 0.8;

// Ramp (in iterations) for β on σ(Q) in the exported policy targets:
// β = min(1, iteration/20). Early on the value head's Q ordering is noise
// that min-max rescaling amplifies to full strength, so π' corrodes the
// prior; let search re-ranking into the targets only as the head matures.
pub(crate) const POLICY_TARGET_Q_RAMP_ITERS: f32 = 20.0;

/// The subset of `HistoryStep` the TD(lambda) label computation needs —
/// split out so `td_lambda_labels` is a pure, directly testable function
/// (no `GameFeatures`/policy tensors to fabricate in a unit test).
#[derive(Clone, Copy)]
pub(crate) struct LabelStep {
    pub(crate) player_id: PlayerId,
    pub(crate) turn: i32,
    pub(crate) my_score: f32,
    pub(crate) opp_score: f32,
    pub(crate) root_value: Option<f32>,
}

impl From<&HistoryStep> for LabelStep {
    fn from(s: &HistoryStep) -> Self {
        LabelStep {
            player_id: s.player_id,
            turn: s.turn,
            my_score: s.my_score,
            opp_score: s.opp_score,
            root_value: s.root_value,
        }
    }
}

/// One player's turn-boundary checkpoint: the first decision of that turn
/// with a recorded root value, else that turn's first decision (root_value
/// stays `None`, so a bootstrap through it contributes 0.0 — matches the
/// original 1-step fallback exactly, just per-horizon instead of once).
pub(crate) struct Checkpoint {
    pub(crate) turn: i32,
    pub(crate) my: f32,
    pub(crate) opp: f32,
    pub(crate) root_value: Option<f32>,
}

pub(crate) fn checkpoints_by_player(history: &[LabelStep]) -> HashMap<PlayerId, Vec<Checkpoint>> {
    let mut out: HashMap<PlayerId, Vec<Checkpoint>> = HashMap::new();
    for step in history {
        let list = out.entry(step.player_id).or_default();
        match list.last_mut() {
            Some(c) if c.turn == step.turn => {
                if c.root_value.is_none() && step.root_value.is_some() {
                    c.my = step.my_score;
                    c.opp = step.opp_score;
                    c.root_value = step.root_value;
                }
            }
            _ => list.push(Checkpoint {
                turn: step.turn,
                my: step.my_score,
                opp: step.opp_score,
                root_value: step.root_value,
            }),
        }
    }
    out
}

/// TD(lambda) forward-view value target for every step in `history`,
/// aligned 1:1. `label_rel_w` prices windows/terminal (EXP_ELO_006).
/// With `wl_z` (EXP_ELO_025): ±1 win/loss terminal, zero window reward,
/// undiscounted bootstrap through root values — a λ-blend of q-targets.
/// What an n-step return does when its checkpoint has no root value (forced
/// plies, or any backend that reports none).
#[derive(Copy, Clone, PartialEq, Eq, Debug, clap::ValueEnum)]
pub(crate) enum MissingBootstrap {
    /// Bootstrap with 0.0 — a truncated return. Legacy semantics.
    Zero,
    /// Skip the checkpoint and carry its weight forward, so the label falls
    /// back to the Monte-Carlo (λ=1) return over that region instead of
    /// being pulled toward zero.
    Mc,
}

pub(crate) fn td_lambda_labels(
    history: &[LabelStep],
    final_scores: &HashMap<i32, f32>,
    lambda: f32,
    label_rel_w: f32,
    wl_z: Option<&HashMap<i32, f32>>,
    missing: MissingBootstrap,
) -> Vec<f32> {
    let checkpoints = checkpoints_by_player(history);

    history
        .iter()
        .map(|step| {
            let terminal_return = match wl_z {
                Some(z) => z.get(&step.player_id).copied().unwrap_or(0.0),
                None => {
                    let my_final = final_scores.get(&step.player_id).copied().unwrap_or(0.0);
                    let opp_final = final_scores
                        .iter()
                        .filter(|(id, _)| **id != step.player_id)
                        .map(|(_, s)| *s)
                        .next()
                        .unwrap_or(0.0);
                    reward::normalized_reward_wf(
                        step.my_score,
                        step.opp_score,
                        my_final,
                        opp_final,
                        label_rel_w,
                    )
                }
            };

            let empty = Vec::new();
            let ahead = checkpoints.get(&step.player_id).unwrap_or(&empty);
            let start = ahead.partition_point(|c| c.turn <= step.turn);

            let mut acc = 0.0f32;
            let mut remaining_weight = 1.0f32;
            for cp in &ahead[start..] {
                if missing == MissingBootstrap::Mc && cp.root_value.is_none() {
                    continue; // weight carries forward to the terminal return
                }
                // Outcome space carries no per-window reward and no discount —
                // a γ<1 here would deflate early-game labels toward 0 by depth.
                let n_step_return = if wl_z.is_some() {
                    cp.root_value.unwrap_or(0.0)
                } else {
                    let r = reward::normalized_reward_wf(
                        step.my_score,
                        step.opp_score,
                        cp.my,
                        cp.opp,
                        label_rel_w,
                    );
                    let dt = (cp.turn - step.turn).max(0);
                    r + reward::GAMMA_TURN.powi(dt) * cp.root_value.unwrap_or(0.0)
                };

                let w = remaining_weight * (1.0 - lambda);
                acc += w * n_step_return;
                remaining_weight *= lambda;
            }
            acc += remaining_weight * terminal_return;

            acc.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Per-decision SPT snapshot for the aux_spt target. Parallel to `LabelStep`
/// rather than riding `Checkpoint`, whose within-turn replace rule is
/// root-value-driven and unrelated to SPT semantics.
#[derive(Clone, Copy)]
pub(crate) struct SptStep {
    pub(crate) player_id: PlayerId,
    pub(crate) turn: i32,
    pub(crate) my_spt: i32,
    pub(crate) opp_spt: i32,
}

/// First decision per (player, turn) — SPT at the start of that player's turn.
pub(crate) fn spt_checkpoints_by_player(steps: &[SptStep]) -> HashMap<PlayerId, Vec<SptStep>> {
    let mut out: HashMap<PlayerId, Vec<SptStep>> = HashMap::new();
    for s in steps {
        let list = out.entry(s.player_id).or_default();
        if list.last().map_or(true, |c| c.turn != s.turn) {
            list.push(*s);
        }
    }
    out
}

/// `[my, opp]` SPT at the first same-player turn >= turn+5, else the final
/// values (game ended inside the horizon).
pub(crate) fn spt_target(cps: Option<&Vec<SptStep>>, turn: i32, final_my: i32, final_opp: i32) -> (i32, i32) {
    cps.and_then(|c| {
        let i = c.partition_point(|s| s.turn < turn + 5);
        c.get(i).map(|s| (s.my_spt, s.opp_spt))
    })
    .unwrap_or((final_my, final_opp))
}

/// Per-decision territory snapshot for the aux_territory5 target. Same shape
/// as `SptStep`, deliberately a monotone "reached" quantity — tile count is
/// summed over cities currently held, so a captured-then-lost city simply
/// stops contributing rather than being penalized. This is the design choice
/// that keeps the head testing expansion TEMPO rather than assuming
/// possession (EXP_ELO_120: momentum/pressure, not possession, carries the
/// third-city win-rate effect).
#[derive(Clone, Copy)]
pub(crate) struct TerritoryStep {
    pub(crate) player_id: PlayerId,
    pub(crate) turn: i32,
    pub(crate) my_territory: i32,
    pub(crate) opp_territory: i32,
}

/// First decision per (player, turn) — territory at the start of that
/// player's turn. Mirrors `spt_checkpoints_by_player`.
pub(crate) fn territory_checkpoints_by_player(
    steps: &[TerritoryStep],
) -> HashMap<PlayerId, Vec<TerritoryStep>> {
    let mut out: HashMap<PlayerId, Vec<TerritoryStep>> = HashMap::new();
    for s in steps {
        let list = out.entry(s.player_id).or_default();
        if list.last().map_or(true, |c| c.turn != s.turn) {
            list.push(*s);
        }
    }
    out
}

/// `[my, opp]` territory tile count at the first same-player turn >= turn+5,
/// else the final values (game ended inside the horizon). Mirrors
/// `spt_target` exactly.
pub(crate) fn territory_target(
    cps: Option<&Vec<TerritoryStep>>,
    turn: i32,
    final_my: i32,
    final_opp: i32,
) -> (i32, i32) {
    cps.and_then(|c| {
        let i = c.partition_point(|s| s.turn < turn + 5);
        c.get(i).map(|s| (s.my_territory, s.opp_territory))
    })
    .unwrap_or((final_my, final_opp))
}

/// `[my, opp]` territory tile count at the first same-player turn >= turn+1,
/// else final values. Phase-2 spike (EXP_ELO_120): the turn-atomic horizon a
/// chainable transition prediction needs, vs. `territory_target`'s turn+5
/// representation-shaping horizon -- same checkpoints, different window.
pub(crate) fn territory_target_h1(
    cps: Option<&Vec<TerritoryStep>>,
    turn: i32,
    final_my: i32,
    final_opp: i32,
) -> (i32, i32) {
    cps.and_then(|c| {
        let i = c.partition_point(|s| s.turn < turn + 1);
        c.get(i).map(|s| (s.my_territory, s.opp_territory))
    })
    .unwrap_or((final_my, final_opp))
}

/// Per-decision army-value snapshot for the aux_army5 target (horizon-
/// compression Stage 3, EXP_ELO_120). Third copy of the SPT+5 template —
/// `evaluate_army` is already `[0,1]`-clamped, so unlike territory/eco
/// ceiling this needs no new normalization decision.
#[derive(Clone, Copy)]
pub(crate) struct ArmyStep {
    pub(crate) player_id: PlayerId,
    pub(crate) turn: i32,
    pub(crate) my_army: f32,
    pub(crate) opp_army: f32,
}

/// First decision per (player, turn). Mirrors `spt_checkpoints_by_player`.
pub(crate) fn army_checkpoints_by_player(steps: &[ArmyStep]) -> HashMap<PlayerId, Vec<ArmyStep>> {
    let mut out: HashMap<PlayerId, Vec<ArmyStep>> = HashMap::new();
    for s in steps {
        let list = out.entry(s.player_id).or_default();
        if list.last().map_or(true, |c| c.turn != s.turn) {
            list.push(*s);
        }
    }
    out
}

/// `[my, opp]` army value at the first same-player turn >= turn+5, else the
/// final values. Mirrors `spt_target` exactly.
pub(crate) fn army_target(
    cps: Option<&Vec<ArmyStep>>,
    turn: i32,
    final_my: f32,
    final_opp: f32,
) -> (f32, f32) {
    cps.and_then(|c| {
        let i = c.partition_point(|s| s.turn < turn + 5);
        c.get(i).map(|s| (s.my_army, s.opp_army))
    })
    .unwrap_or((final_my, final_opp))
}

/// EXP_ELO_120 (horizon-compression Stage 2): windowed-max, not a
/// single-point lookup like `spt_target` — "does a siege open on the
/// OPPONENT within (turn, turn+5]". Siege events are sparse (unlike SPT,
/// which is dense every turn), so a direct scan over the game's full event
/// list is cheap — no checkpoint/binary-search structure needed. No
/// final-game fallback: the event list is complete by construction, so "no
/// event in the window" honestly means 0.0, not a missing value.
pub(crate) fn siege_pressure_target(events: &[(i32, PlayerId)], turn: i32, opp_id: PlayerId) -> f32 {
    let in_window = events
        .iter()
        .any(|&(t, owner)| owner == opp_id && t > turn && t <= turn + 5);
    if in_window { 1.0 } else { 0.0 }
}

/// Per-decision per-city production snapshot, the spatial counterpart of
/// `SptStep`. Not `Copy` — one entry per city the POV holds.
#[derive(Clone)]
pub(crate) struct CitySptStep {
    pub(crate) player_id: PlayerId,
    pub(crate) turn: i32,
    pub(crate) cities: Vec<(i32, i32)>,
}

/// First decision per (player, turn), like `spt_checkpoints_by_player`.
pub(crate) fn city_spt_checkpoints(steps: &[CitySptStep]) -> HashMap<PlayerId, Vec<CitySptStep>> {
    let mut out: HashMap<PlayerId, Vec<CitySptStep>> = HashMap::new();
    for s in steps {
        let list = out.entry(s.player_id).or_default();
        if list.last().map_or(true, |c| c.turn != s.turn) {
            list.push(s.clone());
        }
    }
    out
}

/// Per-city production at the first same-player turn >= turn+5, painted onto a
/// board-sized grid at each city's own tile and normalized like `aux_spt`.
///
/// Cities the POV no longer holds at the horizon simply do not appear, so the
/// target says "nothing here" — which is true: a lost city yields you nothing.
/// When the game ends inside the horizon the last checkpoint stands in for it;
/// unlike `aux_spt` there is no separate final snapshot to fall back to, and
/// the last decision of the game is the closest honest answer.
pub(crate) fn city_spt_target(cps: Option<&Vec<CitySptStep>>, turn: i32, len: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; len];
    let at = cps.and_then(|c| {
        let i = c.partition_point(|s| s.turn < turn + 5);
        c.get(i).or_else(|| c.last())
    });
    if let Some(step) = at {
        for &(idx, prod) in &step.cities {
            if let Some(slot) = g.get_mut(idx as usize) {
                *slot = prod as f32 / 20.0;
            }
        }
    }
    g
}

/// Multi-hot over `TechnologyType::iter()` POSITION — discriminants are
/// sparse (-1..121), so raw ids would need a 123-slot vector.
pub(crate) fn tech_multihot(techs: &[polyfish::states::TechnologyState]) -> Vec<f32> {
    let order: Vec<polyfish::types::TechnologyType> =
        polyfish::types::TechnologyType::iter().collect();
    let mut v = vec![0.0f32; order.len()];
    for t in techs.iter().filter(|t| t.discovered) {
        if let Some(p) = order.iter().position(|x| *x == t.tech_type) {
            v[p] = 1.0;
        }
    }
    v
}

/// Ground-truth enemy-unit occupancy from the unfogged master state. Skips
/// `Invisible` units — the engine's single visibility rule.
pub(crate) fn enemy_unit_grid(state: &GameState, pov: PlayerId, len: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; len];
    for (id, t) in &state.tribes {
        if *id == pov {
            continue;
        }
        for u in &t.units {
            if u.effects.contains(&polyfish::types::UnitEffect::Invisible) {
                continue;
            }
            let i = u.coords.idx as usize;
            if i < len {
                g[i] = 1.0;
            }
        }
    }
    g
}

/// End-of-episode tile ownership mapped to the sample's POV: +1 mine,
/// -1 any opponent, 0 unowned.
pub(crate) fn ownership_from_pov(final_owner: &[i32], pov: PlayerId) -> Vec<f32> {
    final_owner
        .iter()
        .map(|&o| {
            if o == pov {
                1.0
            } else if o != 0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

/// EXP_ELO_061 (Stage 3b): marginalize a macro root ballot into the
/// (stance[4], order[3*H*W]) soft targets `network.rs`'s
/// pi_macro_stance/pi_macro_order heads are shaped for. Visit-mass
/// weighted, normalized by total visits — stance sums to 1 (softmax
/// target), order entries land in [0,1] per tile/kind (sigmoid target,
/// non-exclusive by construction: a tile can accumulate visit mass from
/// multiple candidates that share that (kind, target) pair).
pub(crate) fn macro_policy_targets(
    candidates: &[polyfish::ai::oracle_macro::MacroGoal],
    visits: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    use polyfish::ai::features::MAP_SIZE;
    let board = MAP_SIZE * MAP_SIZE;
    let mut stance = vec![0.0f32; 4];
    let mut order = vec![0.0f32; 3 * board];
    let total: f32 = visits.iter().sum();
    if total <= 0.0 {
        return (stance, order);
    }
    for (goal, &v) in candidates.iter().zip(visits.iter()) {
        stance[goal.stance as usize] += v;
        for &(kind, target) in &goal.orders {
            if let Some(idx) = usize::try_from(target).ok().filter(|&t| t < board) {
                order[kind as usize * board + idx] += v;
            }
        }
    }
    for s in stance.iter_mut() {
        *s /= total;
    }
    for o in order.iter_mut() {
        *o = (*o / total).min(1.0);
    }
    (stance, order)
}

/// Stage 3b dedup (bug fix): gate `HistoryStep::macro_ballot` capture to
/// once per (turn, pov), mirroring the `--dump-macro-policy` JSONL path's
/// own dedup (`last_macro_policy_key`) — the ballot is stable for every ply
/// within a turn (the macro agent only re-searches on a new (turn, pov)),
/// so capturing it on every ply just duplicates the (stance, order) target
/// across every ply's distinct feature vector. An empty-candidates ballot
/// (no search has run yet) collapses to `None` here too — passing it
/// through as `Some` would hit the wrong branch downstream
/// (self_play.rs:4551-4555 treats any `Some` as a real decision and sets
/// `macro_mask=1.0`, which is wrong for an all-zero target) — and does not
/// advance `last_key`, so the next ply retries instead of the empty ballot
/// permanently poisoning this turn's capture.
pub(crate) fn macro_ballot_for_history_step(
    key: (i32, PlayerId),
    last_key: &mut Option<(i32, PlayerId)>,
    ballot: Option<(Vec<polyfish::ai::oracle_macro::MacroGoal>, Vec<f32>)>,
) -> Option<(Vec<polyfish::ai::oracle_macro::MacroGoal>, Vec<f32>)> {
    if *last_key == Some(key) {
        return None;
    }
    let ballot = ballot.filter(|(c, _)| !c.is_empty())?;
    *last_key = Some(key);
    Some(ballot)
}

/// Horizon-compression Stage 1a (EXP_ELO_120): gates the (expensive-ish,
/// ~9-10ms/call per EXP_ELO_086) `ceiling_for_goal` computation to once per
/// (turn, pov) — the same dedup shape as `macro_ballot_for_history_step`,
/// since the empire ceiling is stable for every ply within a turn (nothing
/// about the reachable economy changes mid-turn except via the moves this
/// very search is choosing among). Row-masked downstream like `macro_mask`,
/// not the aux-head per-file convention — presence varies per-row.
pub(crate) fn eco_ceiling_for_history_step(
    key: (i32, PlayerId),
    last_key: &mut Option<(i32, PlayerId)>,
    ceiling: Option<[f32; 4]>,
) -> Option<[f32; 4]> {
    if *last_key == Some(key) {
        return None;
    }
    *last_key = Some(key);
    ceiling
}

/// End-of-episode ground truth for the aux heads: per-tile owner ids,
/// per-player SPT, per-player researched-tech multi-hot, and per-player
/// territory tile count (aux_territory5's horizon-past-game-end fallback).
/// Read off the final state before it is dropped.
pub(crate) fn final_ground_truth(
    state: &GameState,
) -> (
    Vec<i32>,
    HashMap<PlayerId, i32>,
    HashMap<PlayerId, Vec<f32>>,
    HashMap<PlayerId, i32>,
    HashMap<PlayerId, f32>,
) {
// Aux-head ground truth; the final state is dropped when this returns.
let n_tiles = features::MAP_SIZE * features::MAP_SIZE;
let mut final_owner = vec![0i32; n_tiles];
for (&idx, tile) in &state.tiles {
    let i = idx as usize;
    if i < n_tiles {
        final_owner[i] = tile.owner;
    }
}
let mut final_spt = HashMap::new();
let mut final_tech = HashMap::new();
let mut final_territory = HashMap::new();
let mut final_army = HashMap::new();
for (id, t) in &state.tribes {
    final_spt.insert(*id, polyfish::functions::get_tribe_spt(&state, t));
    final_tech.insert(*id, tech_multihot(&t.tech_vanilla));
    let territory: i32 = t
        .cities
        .iter()
        .map(|c| polyfish::rules::economy::territory_tiles(state, c).count() as i32)
        .sum();
    final_territory.insert(*id, territory);
    final_army.insert(*id, polyfish::ai::evaluator::army::evaluate_army(state, *id));
}
    (final_owner, final_spt, final_tech, final_territory, final_army)
}

#[cfg(test)]
#[path = "labels_tests.rs"]
mod tests;
