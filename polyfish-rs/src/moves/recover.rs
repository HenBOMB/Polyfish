//! Recover move
//!
//! Unit recovers health and ends turn.

use crate::actions::chain_undos;
use crate::actions::units::{end_unit_turn, heal_unit};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecoverMove {
    pub src_index: i32,
}

impl RecoverMove {
    pub fn new(src_index: i32) -> Self {
        Self { src_index }
    }
}

impl Move for RecoverMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();

        let unit_owner = state
            .map
            .tiles
            .get(&self.src_index)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        if unit_owner == 0 {
            return Err("No unit at target".to_string());
        }

        let unit_idx = if let Some(tribe) = state.tribes.get(&unit_owner) {
            tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.src_index)
        } else {
            None
        };

        if let Some(idx) = unit_idx {
            // Calculate heal amount
            let in_territory =
                crate::functions::is_in_own_territory(state, self.src_index, unit_owner);
            let amount = if in_territory { 40 } else { 20 };

            undos.push(heal_unit(state, unit_owner, idx, amount));
            undos.push(end_unit_turn(state, unit_owner, idx));

            Ok(MoveResult {
                undo: chain_undos(undos),
                rewards: None,
            })
        } else {
            Err("Unit not found".to_string())
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Recover at {}", self.src_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": AbilityType::Recover,
            "src": self.src_index,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::Recover)
    }
}
