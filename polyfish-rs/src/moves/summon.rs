//! Summon move (Train Unit)
//!
//! Train a new unit at a city or specific tile.

use crate::actions::units::summon_unit;
use crate::functions;
use crate::moves::{Move, MoveResult};
use crate::settings::get_unit_setting;
use crate::settings::technology;
use crate::states::GameState;
use crate::types::{MoveType, SkillType, TerrainType, UnitType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummonMove {
    pub tile_index: i32,
    #[serde(rename = "type")]
    pub unit_type: UnitType,
}

impl SummonMove {
    pub fn new(tile_index: i32, unit_type: UnitType) -> Self {
        Self {
            tile_index,
            unit_type,
        }
    }
}

impl Move for SummonMove {
    fn move_type(&self) -> MoveType {
        MoveType::Summon
    }

    fn execute(&self, state: &mut GameState) -> MoveResult {
        // Costs are handled inside summon_unit if costs=true
        match summon_unit(state, self.unit_type, self.tile_index, true, false) {
            Ok(res) => res,
            Err(e) => {
                // Should not happen if generated correctly
                // Return empty result or panic?
                // Let's print error and return no-op
                eprintln!("Summon execution failed: {}", e);
                MoveResult {
                    undo: Box::new(|_| {}),
                    rewards: None,
                }
            }
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Train {:?} at {}", self.unit_type, self.tile_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Generate summon moves (train units)
pub fn generate_summon_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    // Find all spawnable unit types unlocked by researched techs
    let mut spawnables = Vec::new();
    for tech_state in &tribe.tech_vanilla {
        if !tech_state.discovered {
            continue;
        }

        if let Some(u_type) = get_tech_unit_type(tribe, tech_state.tech_type) {
            let settings = get_unit_setting(u_type);
            if settings.cost >= 1 && tribe.stars >= settings.cost && settings.upgrade_from.is_none()
            {
                spawnables.push(u_type);
            }
        }
    }

    if spawnables.is_empty() {
        return;
    }

    for city in &tribe.cities {
        let target_idx = city.tile_index;

        // Polytopia/TS Rule: City unit count vs level
        // and cannot spawn on occupied tile.
        if get_city_unit_count(state, city) > city.level || is_tile_occupied(state, target_idx) {
            continue;
        }

        for &u_type in &spawnables {
            let settings = get_unit_setting(u_type);

            // Navigate check: typically cannot move onto land except capture.
            // TS logic: allow spawning if unit has at least 1 adjacent water tile.
            if settings.skills.contains(&SkillType::Navigate) {
                let has_water = functions::get_adjacent_indices(state, target_idx, 1)
                    .iter()
                    .any(|&idx| {
                        is_water_terrain(
                            state
                                .tiles
                                .get(&idx)
                                .map(|t| t.terrain_type)
                                .unwrap_or(TerrainType::Field),
                        )
                    });

                if !has_water {
                    continue;
                }
            }

            moves.push(Box::new(SummonMove::new(target_idx, u_type)));
        }
    }
}

fn get_tech_unit_type(
    _tribe: &crate::states::TribeState,
    tech: crate::types::TechnologyType,
) -> Option<UnitType> {
    // In TS: getReplacedOrTechSettings(tribe, tech).unlocksUnit
    // We can simulate this by checking our technology settings

    // Actually, tribe.tech_vanilla usually contains the SPECIFIC tech the tribe has.
    // E.g. if Polaris, they HAVE Frostwork, not Fishing.
    // So get_technology_setting(tech) is already the "replaced" one.
    technology::get_technology_setting(tech).unlocks_unit
}

fn get_city_unit_count(_state: &GameState, city: &crate::states::CityState) -> i32 {
    let mut count = 0;
    for tribe in _state.tribes.values() {
        for unit in &tribe.units {
            if let Some(home) = &unit.home_coords {
                if home.idx == city.tile_index {
                    count += 1;
                }
            }
        }
    }
    count
}

fn is_tile_occupied(state: &GameState, idx: i32) -> bool {
    functions::get_unit_at(state, idx).is_some()
}

fn is_water_terrain(terrain: TerrainType) -> bool {
    matches!(terrain, TerrainType::Water | TerrainType::Ocean)
}
