use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertMove {
    pub src_index: i32,
    pub target_index: i32,
}

impl ConvertMove {
    pub fn new(src_index: i32, target_index: i32) -> Self {
        Self {
            src_index,
            target_index,
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
            .get(&self.src_index)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        match crate::actions::units::convert_unit(state, self.src_index, self.target_index) {
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
        format!("Convert unit at {}", self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "src": self.src_index,
            "target": self.target_index,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::Convert)
    }
}
