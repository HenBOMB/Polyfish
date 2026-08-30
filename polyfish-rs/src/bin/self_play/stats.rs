//! Seat classification and the net-only behaviour statistics derived from
//! a live `GameState`: SPT and army ratios at turn milestones, and
//! time-to-capture. Net-only is the point -- anchor and league-opponent
//! seats are excluded so mixed games report the training net alone.

use polyfish::game::STARTING_OWNER_ID;
use polyfish::ai::reward;
use polyfish::states::{GameState, PlayerId};
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

/// Scores the adjacency hubs a net seat actually built.
///
/// Returns `(hub_levels, first_hub_rank)`: realized partner counts per
/// structure type, and how the first site of each type ranked against every
/// site that seat could legally have used. Attribution is by BUILDER, not by
/// end-of-game tile owner -- the latter would credit captured anchor hubs.
pub(crate) fn score_hubs(
    state: &GameState,
    built_hubs: &[(i32, polyfish::types::StructureType, PlayerId)],
    first_hub_sites: &HashMap<polyfish::types::StructureType, (i32, PlayerId, Vec<i32>)>,
) -> (
    HashMap<polyfish::types::StructureType, (u32, i64, u32, u32)>,
    HashMap<polyfish::types::StructureType, (i64, i64, u32, u32, i64, i64)>,
) {
// Realized level of the hubs the net BUILT (see `built_hubs`), scored at
// game end so a hub that grows as later partners go down is credited —
// partners are counted the way `build_structure` pays them, but against the
// BUILDER's ownership, so value lost with the territory reads as lost.
let mut hub_levels: HashMap<polyfish::types::StructureType, (u32, i64, u32, u32)> =
    HashMap::new();
for (idx, s_type, builder) in built_hubs {
    let settings = polyfish::settings::structures::get_structure_setting(*s_type);
    let still_held = state
        
        .tiles
        .get(idx)
        .is_some_and(|t| t.owner == *builder);
    let partners = polyfish::functions::get_adjacent_indices(&state, *idx, 1)
        .into_iter()
        .filter(|adj| {
            state.tiles.get(adj).is_some_and(|t| t.owner == *builder)
                && polyfish::functions::get_structure_at(&state, *adj)
                    .is_some_and(|p| settings.adjacent_types.contains(&p.structure_type))
        })
        .count() as i64;
    let e = hub_levels.entry(*s_type).or_insert((0, 0, 0, 0));
    e.0 += 1;
    e.1 += partners;
    e.2 += u32::from(partners <= 1);
    e.3 += u32::from(!still_held);
}

// Rank the tile the net actually used against every tile it could have used,
// both scored on partners standing at game end.
let mut first_hub_rank: HashMap<polyfish::types::StructureType, (i64, i64, u32, u32, i64, i64)> =
    HashMap::new();
for (s_type, (chosen, builder, cands)) in first_hub_sites {
    let settings = polyfish::settings::structures::get_structure_setting(*s_type);
    let partners_at = |idx: i32| -> i64 {
        polyfish::functions::get_adjacent_indices(&state, idx, 1)
            .into_iter()
            .filter(|adj| {
                state.tiles.get(adj).is_some_and(|t| t.owner == *builder)
                    && polyfish::functions::get_structure_at(&state, *adj)
                        .is_some_and(|p| settings.adjacent_types.contains(&p.structure_type))
            })
            .count() as i64
    };
    // TERRAIN ceiling: adjacent tiles that could ever host a partner, by
    // terrain + resource alone. Independent of what the net actually built,
    // so it does not inherit the hut-building policy the way `partners_at`
    // does — this is the site's potential, which is the real question.
    let ceiling_at = |idx: i32| -> i64 {
        polyfish::functions::get_adjacent_indices(&state, idx, 1)
            .into_iter()
            .filter(|&adj| {
                let Some(tile) = state.tiles.get(&adj) else { return false };
                settings.adjacent_types.iter().any(|p| {
                    let ps = polyfish::settings::structures::get_structure_setting(*p);
                    if !ps.terrain_types.contains(&tile.terrain_type) || tile.is_algae() {
                        return false;
                    }
                    match ps.resource_type {
                        Some(r) => state
                            
                            .resources
                            .get(&adj)
                            .and_then(|o| o.as_ref())
                            .is_some_and(|res| res.resource_type == r),
                        None => true,
                    }
                })
            })
            .count() as i64
    };
    let got = partners_at(*chosen);
    let best = cands.iter().map(|&c| partners_at(c)).max().unwrap_or(got).max(got);
    let n_better = cands.iter().filter(|&&c| partners_at(c) > got).count() as u32;
    let ceil_got = ceiling_at(*chosen);
    let ceil_best = cands.iter().map(|&c| ceiling_at(c)).max().unwrap_or(ceil_got).max(ceil_got);
    first_hub_rank.insert(
        *s_type,
        (got, best, n_better, cands.len() as u32, ceil_got, ceil_best),
    );
}
    (hub_levels, first_hub_rank)
}

/// How a finished game is scored and who won.
pub(crate) struct Adjudication {
    pub(crate) scores: HashMap<i32, i32>,
    pub(crate) final_potentials: HashMap<i32, f32>,
    pub(crate) winner_id: i32,
    pub(crate) winner_score: i32,
    /// True when the game ended by elimination rather than the turn cap.
    pub(crate) is_decisive: bool,
    pub(crate) alive_tribes: Vec<PlayerId>,
}

/// Adjudicates the final position: sole survivor wins, else highest score
/// at the turn cap. `final_potentials` is the shaped terminal snapshot the
/// TD labels use; it equals the raw score when shaping is off.
pub(crate) fn adjudicate(
    state: &GameState,
    shape_w_label: f32,
    pursuit_w_label: f32,
) -> Adjudication {
// Determine scores & winner
// In Domination, the winner is the last tribe alive.
// If the game timed out (safety cap), use score as tiebreaker.
let mut scores: HashMap<i32, i32> = HashMap::new();
let mut final_potentials: HashMap<i32, f32> = HashMap::new();
let mut alive: HashMap<i32, bool> = HashMap::new();
for (id, t) in &state.tribes {
    scores.insert(*id, t.score);
    alive.insert(*id, t.killed_turn <= 0 && t.resigned_turn <= 0);
}
for id in scores.keys() {
    let mut phi = 0.0;
    if shape_w_label != 0.0 {
        phi += shape_w_label * reward::dev_potential(&state, *id);
    }
    if pursuit_w_label != 0.0 {
        phi += pursuit_w_label * reward::pursuit_potential(&state, *id);
    }
    final_potentials.insert(*id, scores[id] as f32 + phi);
}

// Domination winner: the sole survivor, or highest score if timeout
let alive_tribes: Vec<i32> = alive
    .iter()
    .filter(|(_, is_alive)| **is_alive)
    .map(|(id, _)| *id)
    .collect();

let (winner_id, winner_score) = if alive_tribes.len() == 1 {
    let wid = alive_tribes[0];
    (wid, *scores.get(&wid).unwrap_or(&0))
} else {
    // Timeout: use score tiebreaker
    scores
        .iter()
        .max_by_key(|&(_, score)| score)
        .map(|(&id, &score)| (id, score))
        .unwrap_or((0, 0))
};

let is_decisive = alive_tribes.len() == 1;
    Adjudication { scores, final_potentials, winner_id, winner_score,
                   is_decisive, alive_tribes }
}
