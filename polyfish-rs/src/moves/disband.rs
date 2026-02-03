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
            eprintln!(
                "Error: Unit not found for DisbandMove at target {}",
                self.target
            );
            eprintln!("Tile owner: {}", unit_owner);
            if let Some(tribe) = state.tribes.get(&unit_owner) {
                eprintln!(
                    "Tribe units: {:?}",
                    tribe.units.iter().map(|u| u.coords.idx).collect::<Vec<_>>()
                );
            } else {
                eprintln!("Tribe {} not found!", unit_owner);
            }
            Err("Unit not found".to_string())
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Disband unit at {}", self.target)
    }

    fn serialize(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let Some(obj) = value.as_object_mut() {
            obj.insert("moveType".to_string(), serde_json::json!(MoveType::Ability));
            obj.insert(
                "abilityType".to_string(),
                serde_json::json!(crate::types::AbilityType::Disband),
            );
        }
        value
    }

    #[inline]
    fn action_coords(&self) -> (Option<i32>, Option<i32>) {
        (Some(self.target), Some(self.target))
    }
}
