//! Recover move
//!
//! Unit recovers health and ends turn.

use crate::actions::chain_undos;
use crate::actions::units::{end_unit_turn, heal_unit};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecoverMove {
    pub target: i32,
}

impl RecoverMove {
    pub fn new(target: i32) -> Self {
        Self { target }
    }
}

impl Move for RecoverMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();

        let unit_owner = state
            .tiles
            .get(&self.target)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        if unit_owner == 0 {
            return Err("No unit at target".to_string());
        }

        let unit_idx = if let Some(tribe) = state.tribes.get(&unit_owner) {
            tribe.units.iter().position(|u| u.coords.idx == self.target)
        } else {
            None
        };

        if let Some(idx) = unit_idx {
            // Calculate heal amount
            let in_territory =
                crate::functions::is_in_own_territory(state, self.target, unit_owner);
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
        format!("Recover at {}", self.target)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "abilityType": crate::types::AbilityType::Recover,
            "target": self.target,
        })
    }
}
