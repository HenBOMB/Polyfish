//! Training targets computed after a game ends: the TD(lambda) value
//! label and every auxiliary head's ground truth (SPT, per-city SPT, tech
//! multi-hot, fogged enemy occupancy, tile ownership, macro ballot).
//!
//! Pure functions over recorded snapshots -- nothing here touches a live
//! `Game`, which is what makes the whole module unit-testable.

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

#[cfg(test)]
#[path = "labels_tests.rs"]
mod tests;
