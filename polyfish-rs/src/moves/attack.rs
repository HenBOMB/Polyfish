//! Attack move implementation

use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;

/// An attack move - a unit attacking another unit
#[derive(Debug, Clone)]
pub struct AttackMove {
    /// Attacker's tile index
    pub src: i32,
    /// Target tile index (where the defender is)
    pub target: i32,
}

impl AttackMove {
    pub fn new(src: i32, target: i32) -> Self {
        Self { src, target }
    }
}

impl Move for AttackMove {
    fn move_type(&self) -> MoveType {
        MoveType::Attack
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        // Find attacker
        let (attacker_owner, attacker_idx) = {
            let tile = state.tiles.get(&self.src).ok_or("Source tile not found")?;
            let owner = tile._unit_owner_id.ok_or("No unit at source")?;
            let tribe = state.tribes.get(&owner).ok_or("Tribe not found")?;
            let idx = tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.src)
                .ok_or("Unit not found in tribe")?;
            (owner, idx)
        };

        // Find defender
        let (defender_owner, defender_idx) = {
            let tile = state
                .tiles
                .get(&self.target)
                .ok_or("Target tile not found")?;
            let owner = tile._unit_owner_id.ok_or("No unit at target")?;
            let tribe = state.tribes.get(&owner).ok_or("Tribe not found")?;
            let idx = tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.target)
                .ok_or("Unit not found in tribe")?;
            (owner, idx)
        };

        Ok(MoveResult {
            undo: crate::actions::units::attack_unit(
                state,
                attacker_owner,
                attacker_idx,
                defender_owner,
                defender_idx,
            ),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Attack: {} -> {}", self.src, self.target)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Attack,
            "src": self.src,
            "target": self.target,
        })
    }
}
