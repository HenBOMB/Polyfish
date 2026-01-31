//! Summon move (Train Unit)
//!
//! Train a new unit at a city or specific tile.

use crate::actions::units::summon_unit;
use crate::functions::get_city_unit_count;
use crate::functions::get_tech_unit_type;
use crate::functions::{self, is_tile_occupied, is_water_terrain};
use crate::moves::{Move, MoveResult};
use crate::settings::get_unit_setting;
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

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        // Validation check for cost
        let settings = get_unit_setting(self.unit_type);
        let cost = settings.cost;
        let pov_id = state.settings.current_player_turn_id;

        if let Some(tribe) = state.tribes.get(&pov_id) {
            if tribe.stars < cost {
                return Err(format!(
                    "Insufficient stars for summon: need {}, have {}",
                    cost, tribe.stars
                ));
            }
        }

        // Costs are handled inside summon_unit if costs=true
        match summon_unit(state, self.unit_type, self.tile_index, true, false) {
            Ok(res) => Ok(res),
            Err(e) => Err(e),
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

        if let Some(u_type) = get_tech_unit_type(tech_state.tech_type) {
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

            // Training restriction: Water/Navigate units require adjacent water,
            // UNLESS they are Amphibious (like the Boomchi).
            let is_naval = settings.skills.contains(&SkillType::Navigate)
                || settings.skills.contains(&SkillType::Water);
            let is_amphibious = settings.skills.contains(&SkillType::Amphibious);

            if is_naval && !is_amphibious {
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
