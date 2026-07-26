//! Disband move
//!
//! Disband a unit to regain some stars.

use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisbandMove {
    pub src_index: i32,
}

impl DisbandMove {
    pub fn new(src_index: i32) -> Self {
        Self { src_index }
    }
}

impl Move for DisbandMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let target = self.src_index;
        // Find unit
        let unit_owner = state.tiles
            .get(&target)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        let (unit_idx, unit_type) = if let Some(tribe) = state.tribes.get(&unit_owner) {
            match tribe
                .units
                .iter()
                .enumerate()
                .find(|(_, u)| u.coords.idx == target)
            {
                Some((idx, u)) => (Some(idx), Some(u.unit_type)),
                None => (None, None),
            }
        } else {
            (None, None)
        };

        if let (Some(idx), Some(_u_type)) = (unit_idx, unit_type) {
            let undo = crate::actions::units::disband_unit(state, unit_owner, idx)?;

            Ok(MoveResult {
                undo,
                rewards: None,
            })
        } else {
            Err(format!(
                "Unit not found for Disband at {}",
                self.src_index
            ))
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Disband unit at {}", self.src_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "src": self.src_index,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::Disband)
    }
}
