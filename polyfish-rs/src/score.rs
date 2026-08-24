use crate::states::{CityState, GameState, StructureState, UnitState};
use crate::types::{CityRewardType, StructureType};

pub const CITY_BASE_SCORE: i32 = 100;
pub const CITY_LEVEL_UP_SCORE: i32 = 50;
pub const CITY_POPULATION_SCORE: i32 = 5;
pub const CITY_TERRITORY_SCORE: i32 = 20;

pub const CITY_PARK_SCORE: i32 = 250;
pub const TEMPLE_LEVEL_SCORE: i32 = 100;
pub const UNIT_COST_SCORE: i32 = 5;

pub fn get_city_score(city: &CityState) -> i32 {
    CITY_BASE_SCORE
        + (city.level - 1) * CITY_LEVEL_UP_SCORE
        + city.population * CITY_POPULATION_SCORE
}

/// Score a structure contributes to the owner of the city holding its tile:
/// temples scale with level, everything else pays its flat `reward_score`.
pub fn get_structure_score(structure: &StructureState) -> i32 {
    match structure.structure_type {
        StructureType::Temple
        | StructureType::WaterTemple
        | StructureType::ForestTemple
        | StructureType::MountainTemple
        | StructureType::IceTemple => structure.level * TEMPLE_LEVEL_SCORE,
        other => crate::settings::structures::get_structure_setting(other).reward_score,
    }
}

/// The score a city carries with it: the city itself plus its parks. Territory
/// and the structures standing on it are scored by tile ownership instead, so
/// they move with `claim_territory` rather than with the city (#40).
pub fn get_city_transfer_score(city: &CityState) -> i32 {
    let parks = city
        .rewards
        .iter()
        .filter(|r| **r == CityRewardType::Park)
        .count() as i32;
    get_city_score(city) + parks * CITY_PARK_SCORE
}

/// Territory value of one tile to its owner: the tile itself plus any
/// structure standing on it. Ownership changes must move exactly this.
pub fn get_tile_score(state: &GameState, idx: i32) -> i32 {
    CITY_TERRITORY_SCORE
        + crate::functions::get_structure_at(state, idx)
            .map(get_structure_score)
            .unwrap_or(0)
}

/// A unit's contribution to its owner's score: 5 per star spent on it and on
/// any passenger it carries. Converted units are worth nothing to their new
/// owner. Anything that retypes a unit in place must settle the difference.
pub fn get_unit_score(unit: &UnitState) -> i32 {
    if unit.converted {
        return 0;
    }
    let cost = crate::settings::units::get_unit_setting(unit.unit_type).cost
        + unit
            .passenger_type
            .map(|p| crate::settings::units::get_unit_setting(p).cost)
            .unwrap_or(0);
    cost * UNIT_COST_SCORE
}

/// Per-tribe `tribe.score − calculate_detailed_tribe_score`, for every tribe
/// whose incrementally maintained score disagrees with the canonical
/// recompute. Empty means the reward/value currency is self-consistent (#40).
pub fn score_drift(state: &GameState) -> Vec<(i32, i32)> {
    let mut out: Vec<(i32, i32)> = state
        .tribes
        .iter()
        .filter_map(|(&id, tribe)| {
            let delta = tribe.score - crate::functions::calculate_detailed_tribe_score(state, id);
            (delta != 0).then_some((id, delta))
        })
        .collect();
    out.sort_unstable();
    out
}

/// Largest absolute per-tribe score drift in `state`; 0 when consistent.
pub fn max_abs_score_drift(state: &GameState) -> i32 {
    score_drift(state)
        .into_iter()
        .map(|(_, d)| d.abs())
        .max()
        .unwrap_or(0)
}
