//! Live per-state economic ceiling: "what's the best achievable economy from
//! here" for a real `GameState`, not a from-scratch empire design. Uniform-
//! lane sweep only (the CLI's `--no-mix` path — EXP_ELO_086 measured
//! ~9-10ms/`enumerate_empire` call there, "individually reasonable"; Gate
//! 0.2 of the horizon-compression program confirmed the same figure off a
//! real mid-game state, not synthetic). Horizon-compression Stage 1a
//! (EXP_ELO_120) needs this at self-play generation cadence, which the
//! CLI's full mixed-lane search (~1764x more `enumerate_empire` calls,
//! EXP_ELO_086) cannot support — this is the cheap path that can.

use super::*;
use crate::states::{GameState, PlayerId};
use crate::types::TechnologyType;
use std::collections::HashSet;

/// The `Goal`-conditioned ceiling for a real position: enumerate every
/// uniform-lane scenario's frontier, pool into one Pareto frontier, and pick
/// the best plan for `goal`. `None` when `cities` is empty (nothing to plan)
/// or every scenario is infeasible on this terrain (no frontier survives).
///
/// `owned` techs are read from the tribe's OWN researched state (`pov`'s
/// `tech_vanilla`, discovered only) — unlike the CLI, which defaults to a
/// manual `--techs` override, this must reflect what's actually been
/// researched for the "ceiling from here" question to be honest.
pub fn ceiling_for_goal(
    state: &GameState,
    pov: PlayerId,
    cities: &[i32],
    goal: Goal,
) -> Option<EmpirePlan> {
    if cities.is_empty() {
        return None;
    }
    let owned: HashSet<TechnologyType> = state
        .tribes
        .get(&pov)
        .map(|t| {
            t.tech_vanilla
                .iter()
                .filter(|ts| ts.discovered)
                .map(|ts| ts.tech_type)
                .collect()
        })
        .unwrap_or_default();
    // Turn-0-honest default, matching the CLI's own documented convention:
    // a tribe holds no monument at turn 0, and monument-funded plans are an
    // opt-in question this ceiling doesn't need to answer.
    const MONUMENTS: i32 = 0;
    const WITH_MARKETS: bool = true;
    let top_k = shortlist(8, cities.len(), SCENARIOS.len());

    let mut all_plans: Vec<EmpirePlan> = Vec::new();
    for sc in SCENARIOS {
        if !cities
            .iter()
            .any(|&c| lane_can_place_hub(state, &city_square(state, c), sc.lane))
        {
            continue;
        }
        let scs = uniform(sc, cities.len());
        let terr = allocate_value(state, cities, &scs, MONUMENTS);
        all_plans.extend(enumerate_empire(
            state,
            cities,
            &terr,
            &scs,
            &owned,
            MONUMENTS,
            top_k,
            WITH_MARKETS,
        ));
    }
    let front = pareto(&all_plans);
    pick_for_goal(&front, goal).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::get_adjacent_indices;
    use crate::states::{CityState, TechnologyState, TileState, TribeState};
    use crate::types::TerrainType;

    #[test]
    fn empty_cities_is_none() {
        let state = GameState::default();
        assert!(ceiling_for_goal(&state, 1, &[], Goal::Balanced).is_none());
    }

    #[test]
    fn a_real_minimal_city_yields_a_sane_plan_using_owned_techs() {
        let mut state = GameState::default();
        state.settings.size = 11;
        state.settings.current_player_turn_id = 1;
        let center = 5 * 11 + 5;
        let inner: Vec<i32> =
            get_adjacent_indices(&state, center, 1).into_iter().chain([center]).collect();
        for &idx in &inner {
            let mut t = TileState::default();
            // Forest tiles make the Forest lane (Sawmill) feasible for
            // lane_can_place_hub -- an all-Field territory has no lane any
            // scenario can place a hub on, and the ceiling would come back
            // None for the wrong reason (no feasible lane, not "broken").
            t.terrain_type = if idx == center { TerrainType::Field } else { TerrainType::Forest };
            t.owner = 1;
            t.ruling_city_coords = Some(crate::coords::Coords::from_index(center, 11));
            state.tiles.insert(idx, t);
        }
        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.stars = 50;
        // Owned techs must actually be read from the real state, not
        // defaulted -- Organization is required for LumberHut/harvest buys
        // to even be legal, so a plan with zero owned techs would be
        // near-empty and this test would silently pass for the wrong
        // reason (an empty-but-non-None plan) if the read were broken.
        tribe.tech_vanilla = vec![TechnologyState {
            tech_type: crate::types::TechnologyType::Organization,
            discovered: true,
            discovered_turn: 0,
        }];
        let mut city = CityState { idx: center, owner: 1, ..Default::default() };
        city._territory = inner.clone();
        tribe.cities.push(city);
        state.tribes.insert(1, tribe);

        let plan = ceiling_for_goal(&state, 1, &[center], Goal::Balanced)
            .expect("a real city on real terrain must yield a plan");
        assert!(plan.spt >= 0);
        assert!(plan.pop > 0, "Organization-owned city must plan SOME population");
        assert!(plan.stars <= 50, "plan must not spend more stars than the tribe holds");
    }
}
