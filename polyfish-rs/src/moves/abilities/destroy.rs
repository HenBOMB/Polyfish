use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DestroyMove {
    pub tile_index: i32,
}

impl DestroyMove {
    pub fn new(tile_index: i32) -> Self {
        Self { tile_index }
    }
}

impl Move for DestroyMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> MoveResult {
        let undo = crate::actions::structure::destroy_structure(state, self.tile_index);
        MoveResult {
            undo,
            rewards: None,
        }
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Destroy structure at {}", self.tile_index)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::Destroy,
            "target": self.tile_index
        })
    }
}
