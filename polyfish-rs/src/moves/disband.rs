//! Disband move
//!
//! Disband a unit to regain some stars.

use crate::actions::chain_undos;
use crate::actions::gain_stars;
use crate::actions::units::remove_unit;
use crate::moves::{Move, MoveResult};
use crate::settings::get_unit_setting;
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
        MoveType::Disband
    }

    fn execute(&self, state: &mut GameState) -> MoveResult {
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

        if let (Some(idx), Some(u_type)) = (unit_idx, unit_type) {
            let settings = get_unit_setting(u_type);
            let refund = (settings.cost as f32 * 0.5).floor() as i32;

            let mut undos = Vec::new();
            if refund > 0 {
                undos.push(gain_stars(state, refund));
            }
            undos.push(remove_unit(state, unit_owner, idx, None, None));

            MoveResult {
                undo: chain_undos(undos),
                rewards: None,
            }
        } else {
            MoveResult {
                undo: Box::new(|_| {}),
                rewards: None,
            }
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Disband unit at {}", self.target)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Disband,
            "target": self.target,
        })
    }
}
