//! Per-player development-tempo sampling and unit-count accounting.
//! `TempoTrack` is the per-game accumulator `GameResult` carries; the
//! counters come from post-move `unit_tally` diffs rather than hooks into
//! the actions layer.

use polyfish::states::{GameState, PlayerId};
use std::collections::HashMap;

/// One per-player development-tempo sample, taken at the start of that
/// player's turn (before any of their moves).
#[derive(Clone)]
pub(crate) struct TempoSample {
    pub(crate) turn: i32,
    pub(crate) cities: i32,
    pub(crate) city_levels: i32,
    pub(crate) spt: i32,
    pub(crate) units: i32,
    /// Σ star-cost of living units — army size weighted by quality.
    pub(crate) army_stars: i32,
    pub(crate) revealed: i32,
    pub(crate) techs: i32,
    /// Enemy units destroyed so far, read straight off `TribeState::kills`
    /// (engine-maintained, undo-safe). Conversions are not kills.
    pub(crate) kills: i32,
    /// Cumulative counters through this sample (mirrors of the TempoTrack
    /// counters, snapshotted so both curve and totals are per-turn/per-role).
    pub(crate) trained_cum: i32,
    pub(crate) lost_cum: i32,
    /// Σ star-cost of units lost so far — a dead giant costs 10, not 1.
    pub(crate) stars_lost_cum: i32,
}

/// One player's tempo curve plus event-accounted unit counters for the game.
/// Counters come from per-move unit-count diffs, so ruin grants, level-up
/// giants, conversions, and retaliation deaths are all captured without
/// hooking the actions layer (a conversion counts as lost+granted).
#[derive(Default, Clone)]
pub(crate) struct TempoTrack {
    pub(crate) samples: Vec<TempoSample>,
    /// Units gained by a Summon move — star-spent production only.
    pub(crate) units_trained: i32,
    /// Units gained any other way (ruins, conversion, level-up rewards).
    pub(crate) units_granted: i32,
    pub(crate) units_lost: i32,
    pub(crate) giants_made: i32,
    /// Σ star-cost of lost units (army VALUE destroyed, not just count).
    pub(crate) army_stars_lost: i32,
}

pub(crate) fn tempo_sample(state: &GameState, pov: PlayerId) -> Option<TempoSample> {
    let tribe = state.tribes.get(&pov)?;
    let army_stars: i32 = tribe
        .units
        .iter()
        .map(polyfish::rules::combat::unit_worth)
        .sum();
    Some(TempoSample {
        turn: state.settings.turn,
        cities: tribe.cities.len() as i32,
        city_levels: tribe.cities.iter().map(|c| c.level).sum(),
        spt: polyfish::functions::get_tribe_spt(state, tribe),
        units: tribe.units.len() as i32,
        army_stars,
        revealed: state
            .tiles
            .values()
            .filter(|t| t.explorers.contains(&pov))
            .count() as i32,
        techs: tribe.tech_vanilla.len() as i32,
        kills: tribe.kills,
        // Attached from the TempoTrack counters at the push site.
        trained_cum: 0,
        lost_cum: 0,
        stars_lost_cum: 0,
    })
}

/// `(unit_count, giant_count, army_star_cost)` per player, for post-move
/// diff accounting.
pub(crate) fn unit_tally(state: &GameState) -> HashMap<PlayerId, (i32, i32, i32)> {
    state
        .tribes
        .iter()
        .map(|(id, t)| {
            // Per-tribe super unit, not just Giant — Polaris/Aquarion/Elyrion/
            // Cymanti super units were invisible to this metric.
            let super_unit = polyfish::settings::units::get_super_unit(t.tribe_type);
            let giants = t
                .units
                .iter()
                .filter(|u| u.unit_type == super_unit)
                .count() as i32;
            let stars: i32 = t
                .units
                .iter()
                .map(polyfish::rules::combat::unit_worth)
                .sum();
            (*id, (t.units.len() as i32, giants, stars))
        })
        .collect()
}

