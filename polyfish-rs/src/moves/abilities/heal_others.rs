use crate::actions::chain_undos;
use crate::actions::units::{end_unit_turn, heal_unit};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealOthersMove {
    pub unit_idx: i32,
}

impl HealOthersMove {
    pub fn new(unit_idx: i32) -> Self {
        Self { unit_idx }
    }
}

impl Move for HealOthersMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> MoveResult {
        let actor_owner = state
            .tiles
            .get(&self.unit_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let actor_idx = if let Some(tribe) = state.tribes.get(&actor_owner) {
            tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.unit_idx)
        } else {
            None
        };

        if let Some(a_idx) = actor_idx {
            let mut undos = Vec::new();

            // Heal all adjacent allies
            let adj = crate::functions::get_adjacent_indices(state, self.unit_idx, 1);
            for target_idx in adj {
                let target_owner = state
                    .tiles
                    .get(&target_idx)
                    .and_then(|t| t._unit_owner_id)
                    .unwrap_or(0);
                if target_owner == actor_owner {
                    if let Some(tribe) = state.tribes.get(&target_owner) {
                        if let Some(pos) =
                            tribe.units.iter().position(|u| u.coords.idx == target_idx)
                        {
                            undos.push(heal_unit(state, target_owner, pos, 4));
                        }
                    }
                }
            }

            undos.push(end_unit_turn(state, actor_owner, a_idx));
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
        format!("Heal allies around {}", self.unit_idx)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::HealOthers,
            "unitIdx": self.unit_idx
        })
    }
}
