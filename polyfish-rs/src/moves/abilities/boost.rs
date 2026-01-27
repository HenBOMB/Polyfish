use crate::actions::chain_undos;
use crate::actions::units::end_unit_turn;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoostMove {
    pub unit_idx: i32,
}

impl BoostMove {
    pub fn new(unit_idx: i32) -> Self {
        Self { unit_idx }
    }
}

impl Move for BoostMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> MoveResult {
        let owner = state
            .tiles
            .get(&self.unit_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let actor_idx = if let Some(t) = state.tribes.get(&owner) {
            t.units.iter().position(|u| u.coords.idx == self.unit_idx)
        } else {
            None
        };

        if let Some(a_idx) = actor_idx {
            let mut undos = Vec::new();
            let adj = crate::functions::get_adjacent_indices(state, self.unit_idx, 1);
            for t_idx in adj {
                let target_owner = state
                    .tiles
                    .get(&t_idx)
                    .and_then(|t| t._unit_owner_id)
                    .unwrap_or(0);
                if target_owner == owner {
                    if let Some(t) = state.tribes.get(&target_owner) {
                        if let Some(pos) = t.units.iter().position(|u| u.coords.idx == t_idx) {
                            undos.push(crate::actions::try_add_effect(
                                state,
                                target_owner,
                                pos,
                                crate::types::EffectType::Boost,
                            ));
                        }
                    }
                }
            }
            undos.push(end_unit_turn(state, owner, a_idx));
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
        format!("Boost allies around {}", self.unit_idx)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::Boost,
            "unitIdx": self.unit_idx
        })
    }
}
