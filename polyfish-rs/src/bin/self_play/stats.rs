//! Seat classification and the net-only behaviour statistics derived from
//! a live `GameState`: SPT and army ratios at turn milestones, and
//! time-to-capture. Net-only is the point -- anchor and league-opponent
//! seats are excluded so mixed games report the training net alone.

use polyfish::game::STARTING_OWNER_ID;
use polyfish::states::PlayerId;
use std::collections::HashMap;

/// Up to 5 evenly spaced turn thresholds for periodic in-game progress.
pub(crate) fn turn_milestones(max_turns: i32) -> Vec<i32> {
    const MAX_REPORTS: usize = 5;
    if max_turns <= 0 {
        return Vec::new();
    }
    (1..=MAX_REPORTS)
        .map(|i| (max_turns * i as i32 + MAX_REPORTS as i32 - 1) / MAX_REPORTS as i32)
        .collect()
}

/// Game-count milestones at 20%, 40%, …, 100% for large runs.
pub(crate) fn finish_milestones(num_games: usize) -> Vec<usize> {
    (1..=5).map(|i| num_games * i / 5).collect()
}
pub(crate) const SPT_MILESTONES: [i32; 7] = [0, 5, 10, 15, 20, 25, 30];

/// True when `pid`'s seat is controlled by the training net ("model" /
/// "model_vs_anchor") — anchor (Greedy) and league-opponent seats are
/// excluded from the aggregate metrics so mixed games report the net only.
pub(crate) fn is_net_seat(seat_roles: [&'static str; 2], pid: PlayerId) -> bool {
    let i = (pid - 1) as usize;
    i < 2 && matches!(seat_roles[i], "model" | "model_vs_anchor")
}
/// Mean SPT over net-controlled tribes only (all tribes as a fallback if
/// none qualify — shouldn't happen with valid seat_roles).
pub(crate) fn mean_net_spt(state: &polyfish::states::GameState, seat_roles: [&'static str; 2]) -> f32 {
    let vals: Vec<f32> = state
        .tribes
        .iter()
        .filter(|(id, _)| is_net_seat(seat_roles, **id))
        .map(|(_, t)| polyfish::functions::get_tribe_spt(state, t) as f32)
        .collect();
    if vals.is_empty() {
        let n = state.tribes.len().max(1) as f32;
        return state
            .tribes
            .values()
            .map(|t| polyfish::functions::get_tribe_spt(state, t) as f32)
            .sum::<f32>()
            / n;
    }
    vals.iter().sum::<f32>() / vals.len() as f32
}

/// Mean over net seats of (Σ unit star cost ÷ unit count, Σ unit star cost ÷
/// city count). A seat with no units (or no cities) contributes 0 to that
/// component rather than being skipped, so the denominator stays the seat count.
pub(crate) fn mean_net_army_ratios(
    state: &polyfish::states::GameState,
    seat_roles: [&'static str; 2],
) -> (f32, f32) {
    let (mut worth, mut per_city, mut seats) = (0.0f32, 0.0f32, 0u32);
    for (_, t) in state
        .tribes
        .iter()
        .filter(|(id, _)| is_net_seat(seat_roles, **id))
    {
        let stars: i32 = t
            .units
            .iter()
            .map(polyfish::rules::combat::unit_worth)
            .sum();
        if !t.units.is_empty() {
            worth += stars as f32 / t.units.len() as f32;
        }
        if !t.cities.is_empty() {
            per_city += stars as f32 / t.cities.len() as f32;
        }
        seats += 1;
    }
    if seats == 0 {
        return (0.0, 0.0);
    }
    (worth / seats as f32, per_city / seats as f32)
}

pub(crate) fn record_spt_at_turn_start(
    state: &polyfish::states::GameState,
    spt_at_turn: &mut HashMap<i32, f32>,
    army_ratios_at_turn: &mut HashMap<i32, (f32, f32)>,
    next_idx: &mut usize,
    seat_roles: [&'static str; 2],
) {
    if state.settings.current_player_turn_id != STARTING_OWNER_ID {
        return;
    }
    while *next_idx < SPT_MILESTONES.len() {
        let milestone = SPT_MILESTONES[*next_idx];
        if state.settings.turn < milestone {
            break;
        }
        if state.settings.turn == milestone {
            spt_at_turn.insert(milestone, mean_net_spt(state, seat_roles));
            army_ratios_at_turn.insert(milestone, mean_net_army_ratios(state, seat_roles));
        }
        *next_idx += 1;
    }
}
/// Turn by which `frac` of `initial` capturables were taken, given the
/// chronological list of capture turns (`frac` 0.0 = the first capture).
/// `censor` (game length) when the game never reached that fraction or the
/// map had none to begin with.
pub(crate) fn t2c_turn(capture_turns: &[i32], initial: usize, frac: f64, censor: i32) -> f32 {
    if initial == 0 {
        return censor as f32;
    }
    let needed = ((initial as f64 * frac).ceil() as usize).max(1);
    capture_turns
        .get(needed - 1)
        .map(|&t| t as f32)
        .unwrap_or(censor as f32)
}
