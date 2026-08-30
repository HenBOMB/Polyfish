//! Per-turn arena diagnostics: the --dump-turn-states row and the
//! --dump-stats-dir sample. Arrays index [config1, config2], never by
//! seat, so a swapped pair stays comparable.

use std::io::Write;

/// Append one start-of-turn ground-truth snapshot to <dir>/game<idx>.jsonl for
/// the vs-Greedy 3rd-city pursuit analysis. Both players' cities/units come
/// from ground truth every turn; villages are ground-truth neutral villages
/// (`neutral_villages`) plus the model player's FOW view
/// (`model_visible_villages`). Row-major 11x11 tile indices. One file per game,
/// so concurrent match workers never share a handle.
pub(crate) fn dump_turn_state(
    file: &mut std::fs::File,
    game_idx: usize,
    state: &polyfish::states::GameState,
    model_player: polyfish::states::PlayerId,
    greedy_player: polyfish::states::PlayerId,
) {
        // Currently-uncaptured (neutral) villages: owner only ever transitions
    // 0 -> nonzero via capture, so `owner == 0` is exactly self_play's
    // incremental open_villages set without intercepting the move loop.
    let neutral_villages: Vec<i32> = state
        .structures
        .iter()
        .filter_map(|(&idx, s)| {
            let s = s.as_ref()?;
            let neutral = s.structure_type == polyfish::types::StructureType::Village
                && state.tiles.get(&idx).map_or(false, |t| t.owner == 0);
            neutral.then_some(idx)
        })
        .collect();
    let model_visible_villages: Vec<i32> = neutral_villages
        .iter()
        .copied()
        .filter(|idx| {
            state
                .tiles
                .get(idx)
                .map_or(false, |t| t.explorers.contains(&model_player))
        })
        .collect();
    let cities_of = |pid: polyfish::states::PlayerId| -> Vec<i32> {
        state
            .tribes
            .get(&pid)
            .map(|t| t.cities.iter().map(|c| c.idx).collect())
            .unwrap_or_default()
    };
    let units_of = |pid: polyfish::states::PlayerId| -> Vec<i32> {
        state
            .tribes
            .get(&pid)
            .map(|t| t.units.iter().map(|u| u.coords.idx).collect())
            .unwrap_or_default()
    };
    let rec = serde_json::json!({
        "game": game_idx,
        "turn": state.settings.turn,
        "acting_player": state.settings.current_player_turn_id,
        "model_player": model_player,
        "model_cities": cities_of(model_player),
        "model_units": units_of(model_player),
        "model_visible_villages": model_visible_villages,
        "greedy_cities": cities_of(greedy_player),
        "greedy_units": units_of(greedy_player),
        "neutral_villages": neutral_villages,
    });
    if let Ok(s) = serde_json::to_string(&rec) {
        let _ = writeln!(file, "{s}");
    }
}

/// One per-turn sample for --dump-stats-dir; arrays index [config1, config2].
#[derive(serde::Serialize)]
pub(crate) struct TurnSample {
    pub(crate) turn: i32,
    pub(crate) score: [i32; 2],
    pub(crate) spt: [i32; 2],
    pub(crate) stars: [i32; 2],
    pub(crate) cities: [usize; 2],
    pub(crate) units: [usize; 2],
    pub(crate) unit_cost: [i32; 2],
    /// Super units alive, per tribe's own type — Giant for Imperius, Gaami for
    /// Polaris. "Did we make giants, and how many by turn N" is a headline
    /// behaviour question and unit COUNT alone cannot answer it.
    pub(crate) super_units: [usize; 2],
    pub(crate) techs: [usize; 2],
}

pub(crate) fn sample_turn(state: &polyfish::states::GameState, swap: bool) -> TurnSample {
    let mut s = TurnSample {
        turn: state.settings.turn,
        score: [0; 2],
        spt: [0; 2],
        stars: [0; 2],
        cities: [0; 2],
        units: [0; 2],
        unit_cost: [0; 2],
        super_units: [0; 2],
        techs: [0; 2],
    };
    for c in 0..2 {
        // Config 1 sits in the P1 seat unless swapped.
        let pid: polyfish::states::PlayerId = if (c == 0) != swap { 1 } else { 2 };
        if let Some(t) = state.tribes.get(&pid) {
            s.score[c] = t.score;
            s.spt[c] = polyfish::functions::get_tribe_spt(state, t);
            s.stars[c] = t.stars;
            s.cities[c] = t.cities.len();
            s.units[c] = t.units.len();
            s.unit_cost[c] = t
                .units
                .iter()
                .map(polyfish::rules::combat::unit_worth)
                .sum();
            let super_type = polyfish::settings::units::get_super_unit(t.tribe_type);
            s.super_units[c] = t.units.iter().filter(|u| u.unit_type == super_type).count();
            s.techs[c] = t.tech_vanilla.len();
        }
    }
    s
}

