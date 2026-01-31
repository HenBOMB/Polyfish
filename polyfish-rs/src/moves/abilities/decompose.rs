use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DecomposeMove {
    pub tile_index: i32,
}

impl DecomposeMove {
    pub fn new(tile_index: i32) -> Self {
        Self { tile_index }
    }
}

impl Move for DecomposeMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        state
            ._end_of_turn_queue
            .push(crate::states::EndOfTurnAction::Decompose {
                tile_index: self.tile_index,
                owner_id: pov_id,
            });

        Ok(MoveResult {
            undo: Box::new(move |s: &mut GameState| {
                s._end_of_turn_queue.pop();
            }),
            rewards: None,
        })
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Decompose structure at {}", self.tile_index)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::Decompose,
            "target": self.tile_index
        })
    }
}
