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
        match crate::actions::units::convert_unit(state, self.unit_idx, self.target_idx) {
            Ok(undo) => Ok(MoveResult {
                undo,
                rewards: None,
            }),
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
