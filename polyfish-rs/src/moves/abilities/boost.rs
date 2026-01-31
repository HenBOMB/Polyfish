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
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let undo = crate::actions::units::boost_unit(state, self.unit_idx);
        Ok(MoveResult {
            undo,
            rewards: None,
        })
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
