use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;

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
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let undo = crate::actions::structure::destroy_structure(state, self.tile_index);
        Ok(MoveResult {
            undo,
            rewards: None,
        })
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Destroy structure at {}", self.tile_index)
    }
    fn serialize(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let Some(obj) = value.as_object_mut() {
            obj.insert("moveType".to_string(), serde_json::json!(MoveType::Ability));
        }
        value
    }

    #[inline]
    fn action_coords(&self) -> (Option<i32>, Option<i32>) {
        (Some(self.tile_index), Some(self.tile_index))
    }
}
