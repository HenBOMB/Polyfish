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
            let atk = {
                let u = state.tribes.get(&owner).unwrap().units.get(u_idx).unwrap();
                crate::functions::get_unit_attack(u)
            };

            undos.push(crate::actions::units::remove_unit(
                state, owner, u_idx, None, None,
            ));

            let adj = crate::functions::get_adjacent_indices(state, self.unit_idx, 1);
            for t_idx in adj {
                if let Some(enemy_unit) = crate::functions::get_enemy_at(state, t_idx, owner) {
                    let enemy_owner = enemy_unit.owner;
                    if let Some(e_tribe) = state.tribes.get(&enemy_owner) {
                        if let Some(e_pos) =
                            e_tribe.units.iter().position(|u| u.coords.idx == t_idx)
                        {
                            let damage_val = (atk * 0.5 * 10.0).round() as i32;
                            undos.push(crate::actions::units::deal_damage(
                                state,
                                enemy_owner,
                                e_pos,
                                damage_val,
                                Some(owner),
                            ));
                        }
                    }
                }
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
