//! Summon move (Train Unit)
//!
//! Train a new unit at a city or specific tile.

use crate::actions::units::summon_unit;
use crate::functions::get_city_unit_count;
use crate::functions::{self, is_tile_occupied};
use crate::moves::{Move, MoveResult};
use crate::settings::get_unit_setting;
use crate::states::GameState;
use crate::types::{MoveType, SkillType, UnitType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SummonMove {
    pub src_index: i32,
    #[serde(rename = "type")]
    pub unit_type: UnitType,
}

impl SummonMove {
    pub fn new(src_index: i32, unit_type: UnitType) -> Self {
        Self {
            src_index,
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
                    "Insufficient stars to summon {:?} at {}: need {}, have {}",
                    self.unit_type, self.src_index, cost, tribe.stars
                ));
            }
        }

        // Costs are handled inside summon_unit if costs=true
        match summon_unit(state, self.unit_type, self.src_index, true, false) {
            Ok(res) => Ok(res),
            Err(e) => Err(e),
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Train {:?} at {}", self.unit_type, self.src_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "src": self.src_index,
            "type": self.unit_type,
        })
    }

    fn cost(&self, _state: &GameState) -> Option<i32> {
        Some(get_unit_setting(self.unit_type).cost)
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn unit_type(&self) -> Result<UnitType, String> {
        Ok(self.unit_type)
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

    // Always include Warrior (base unit) if affordable
    let warrior_settings = get_unit_setting(UnitType::Warrior);
    if tribe.stars >= warrior_settings.cost {
        spawnables.push(UnitType::Warrior);
    } else {
        // Debug: print if Warrior is not affordable but we are generating moves
        // println!("🐛 Warrior not affordable: have {}, need {}", tribe.stars, warrior_settings.cost);
    }

    for tech_state in &tribe.tech_vanilla {
        // We use all researched techs for summoning, even if not "discovered" (simulation)
        for u_type in
            crate::settings::technology::get_unlocked_units(tech_state.tech_type, tribe.tribe_type)
        {
            // Avoid duplicates (Warrior)
            if u_type == UnitType::Warrior {
                continue;
            }

            let settings = get_unit_setting(u_type);
            if settings.cost >= 1 && settings.upgrade_from.is_none() {
                if tribe.stars >= settings.cost {
                    spawnables.push(u_type);
                }
            }
        }
    }

    if spawnables.is_empty() {
        return;
    }

    for city in &tribe.cities {
        let target_idx = city.idx;

        // Polytopia/TS Rule: City unit count vs level
        // and cannot spawn on occupied tile.
        let count = get_city_unit_count(state, city);
        if count > city.level || is_tile_occupied(state, target_idx) {
            // if tribe.stars >= warrior_settings.cost && target_idx == 163 {
            //     println!(
            //         "🐛 city 163 blocked: count={}, level={}, occupied={}",
            //         count,
            //         city.level,
            //         is_tile_occupied(state, target_idx)
            //     );
            // }
            continue;
        }

        // if target_idx == 163 {
        //     println!(
        //         "🐛 city 163 OK: count={}, level={}, afford={}",
        //         count, city.level, tribe.stars
        //     );
        // }

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
                        state
                            .tiles
                            .get(&idx)
                            .map(|t| t.is_water_terrain())
                            .unwrap_or(false)
                    });

                if !has_water {
                    continue;
                }
            }

            moves.push(Box::new(SummonMove::new(target_idx, u_type)));
        }
    }
}
