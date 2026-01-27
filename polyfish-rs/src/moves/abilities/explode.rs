use crate::actions::chain_undos;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExplodeMove {
    pub unit_idx: i32,
}

impl ExplodeMove {
    pub fn new(unit_idx: i32) -> Self {
        Self { unit_idx }
    }
}

impl Move for ExplodeMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> MoveResult {
        let owner = state
            .tiles
            .get(&self.unit_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let unit_idx_in_tribe = if let Some(t) = state.tribes.get(&owner) {
            t.units.iter().position(|u| u.coords.idx == self.unit_idx)
        } else {
            None
        };

        if let Some(u_idx) = unit_idx_in_tribe {
            let mut undos = Vec::new();

            // Collect all units in the chain to explode (this unit + all children)
            let mut units_to_explode = Vec::new();
            units_to_explode.push((u_idx, self.unit_idx));

            // Traverse chain to find children
            let mut current_idx = u_idx;
            loop {
                // We need to re-borrow tribe to check children to avoid holding borrow
                let child_info = if let Some(t) = state.tribes.get(&owner) {
                    if let Some(u) = t.units.get(current_idx) {
                        u.child_unit_idx.map(|c_idx| {
                            let c_coords = t.units.get(c_idx).map(|c| c.coords.idx).unwrap_or(-1);
                            (c_idx, c_coords)
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((c_idx, c_coords)) = child_info {
                    units_to_explode.push((c_idx, c_coords));
                    current_idx = c_idx;
                } else {
                    break;
                }
            }

            // Explode each unit in the chain
            // We iterate in reverse order (tail to head) so indices remain valid if we were removing simple vector items,
            // but remove_unit handles swap_remove or similar, so indices might shift.
            // Actually remove_unit uses indices. If we remove multiple units, indices shift!
            // Wait, remove_unit in actions/units.rs: "if unit_idx < tribe.units.len() { tribe.units.remove(unit_idx); }"
            // Vec::remove shifts elements. So removing index 0 invalidates all other indices.
            // We MUST sort indices descending and remove from highest to lowest.

            // Sort by unit index descending
            units_to_explode.sort_by(|a, b| b.0.cmp(&a.0));

            for (explode_u_idx, explode_tile_idx) in units_to_explode {
                let atk = 4.0; // Segment/Centipede attack is 4

                // 1. Deal AOE Damage
                let adj = crate::functions::get_adjacent_indices(state, explode_tile_idx, 1);
                for t_idx in adj {
                    if let Some(enemy_unit) = crate::functions::get_enemy_at(state, t_idx, owner) {
                        let enemy_owner = enemy_unit.owner;
                        if let Some(e_tribe) = state.tribes.get(&enemy_owner) {
                            if let Some(e_pos) =
                                e_tribe.units.iter().position(|u| u.coords.idx == t_idx)
                            {
                                let damage_val = (atk * 0.5f32 * 10.0).round() as i32; // 2 damage
                                undos.push(crate::actions::units::deal_damage(
                                    state,
                                    enemy_owner,
                                    e_pos,
                                    damage_val,
                                    Some(owner),
                                ));

                                // Poison enemy
                                undos.push(crate::actions::units::poison_unit(
                                    state,
                                    enemy_owner,
                                    e_pos,
                                ));
                            }
                        }
                    }
                }

                // 2. Create Spores (if land/ice and no structure)
                if let Some(tile) = state.tiles.get(&explode_tile_idx) {
                    let is_water = tile.terrain_type == crate::types::TerrainType::Water;
                    let has_structure =
                        crate::functions::get_structure_at(state, explode_tile_idx).is_some();

                    if !is_water && !has_structure {
                        undos.push(crate::actions::structure::create_structure(
                            state,
                            explode_tile_idx,
                            crate::types::StructureType::Spores,
                            owner,
                        ));
                    }
                }

                // 3. Remove the unit
                undos.push(crate::actions::units::remove_unit(
                    state,
                    owner,
                    explode_u_idx,
                    None,
                    None,
                ));
            }

            MoveResult {
                undo: chain_undos(undos),
                rewards: None,
            }
        } else {
            MoveResult {
                undo: Box::new(|_| {}),
                rewards: None,
            }
        }
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Explode unit at {}", self.unit_idx)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::Explode,
            "unitIdx": self.unit_idx
        })
    }
}
