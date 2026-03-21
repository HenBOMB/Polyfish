use crate::actions::{chain_undos, modify_terrain, units::end_unit_turn};
use crate::functions::{get_adjacent_indices, has_skill};
use crate::moves::{Move, MoveResult};
use crate::states::{GameState, UnitState};
use crate::types::{AbilityType, MoveType, SkillType, TerrainType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BreakIceMove {
    pub target_index: i32, // The ice tile index
}

impl BreakIceMove {
    pub fn new(target_index: i32) -> Self {
        Self { target_index }
    }
}

impl Move for BreakIceMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();
        let ice_idx = self.target_index;
        let pov_id = state.settings.current_player_turn_id;

        // 1. Identify valid actor
        // We find the first adjacent unit of the current player that can actually perform this action.
        let adj = get_adjacent_indices(state, ice_idx, 1);
        let actor_idx = adj.iter().find_map(|&u_pos| {
            if let Some(tribe) = state.tribes.get(&pov_id) {
                tribe.units.iter().position(|u| {
                    if u.coords.idx != u_pos {
                        return false;
                    }

                    // Rules:
                    // - Unused: Yes
                    // - Moved but not attacked: Yes if has Dash
                    // - Attacked: Yes if has DoubleAttack
                    if !u.moved && !u.attacked {
                        true
                    } else if u.moved && !u.attacked {
                        has_skill(u, SkillType::Dash)
                    } else if u.attacked {
                        has_skill(u, SkillType::DoubleAttack)
                    } else {
                        false
                    }
                })
            } else {
                None
            }
        });

        if let Some(u_idx) = actor_idx {
            // 2. Modify Terrain (Ice -> Water)
            // Polytopia uses the original terrain (Water vs Ocean), but default to Water if unknown.
            undos.push(modify_terrain(state, ice_idx, TerrainType::Water));

            // 3. End unit turn
            undos.push(end_unit_turn(state, pov_id, u_idx));
        } else {
            return Err("No valid unit found adjacent to ice to perform Break Ice".to_string());
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Break ice at {}", self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "src": self.target_index, // Protocol expects 'src' for the ice tile
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::BreakIce)
    }
}

pub fn generate_break_ice_moves_for_unit(
    state: &GameState,
    unit: &UnitState,
    moves: &mut Vec<Box<dyn Move>>,
) {
    // let pov_id = state.settings.current_player_turn_id;
    let idx = unit.coords.idx;

    let can_act = if !unit.moved && !unit.attacked {
        true
    } else if unit.moved && !unit.attacked {
        has_skill(unit, SkillType::Dash)
    } else if unit.attacked {
        has_skill(unit, SkillType::DoubleAttack)
    } else {
        false
    };

    if !can_act {
        return;
    }

    let adj = get_adjacent_indices(state, idx, 1);
    for &adj_idx in &adj {
        if let Some(tile) = state.tiles.get(&adj_idx) {
            if tile.terrain_type == TerrainType::Ice {
                moves.push(Box::new(BreakIceMove::new(adj_idx)));
            }
        }
    }
}
