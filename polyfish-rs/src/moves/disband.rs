//! Disband move
//!
//! Disband a unit to regain some stars.

use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisbandMove {
    pub target: i32,
}

impl DisbandMove {
    pub fn new(target: i32) -> Self {
        Self { target }
    }
}

impl Move for DisbandMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let target = self.target;
        // Find unit
        let unit_owner = state
            .tiles
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
            Err("Unit not found".to_string())
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Disband unit at {}", self.target)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "abilityType": crate::types::AbilityType::Disband,
            "target": self.target,
        })
    }
}
