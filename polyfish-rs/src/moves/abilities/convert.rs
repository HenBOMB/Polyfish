use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertMove {
    pub unit_idx: i32,
    pub target_idx: i32,
}

impl ConvertMove {
    pub fn new(unit_idx: i32, target_idx: i32) -> Self {
        Self {
            unit_idx,
            target_idx,
        }
    }
}

impl Move for ConvertMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let owner = state
            .tiles
            .get(&self.unit_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        match crate::actions::units::convert_unit(state, self.unit_idx, self.target_idx) {
            Ok(undo) => {
                if let Some(tribe) = state.tribes.get_mut(&owner) {
                    tribe.attacked_this_turn = true;
                    tribe.conversions += 1; // Track for Converter task
                }
                Ok(MoveResult {
                    undo,
                    rewards: None,
                })
            }
            Err(e) => Err(e),
        }
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Convert unit at {}", self.target_idx)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::Convert,
            "unitIdx": self.unit_idx,
            "targetIdx": self.target_idx
        })
    }
}
