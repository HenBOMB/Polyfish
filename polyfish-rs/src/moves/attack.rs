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

        // Find defender (Unit or City for Infiltration)
        let (defender_owner, defender_idx) = {
            let tile = state
                .tiles
                .get(&self.target)
                .ok_or("Target tile not found")?;

            if let Some(owner) = tile._unit_owner_id {
                let tribe = state.tribes.get(&owner).ok_or("Tribe not found")?;
                let idx = tribe
                    .units
                    .iter()
                    .position(|u| u.coords.idx == self.target)
                    .ok_or("Unit not found in tribe")?;
                (owner, Some(idx))
            } else {
                // Check if targeting a city for infiltration
                // Attacker must have Infiltrate
                if let Some(att_tribe) = state.tribes.get(&attacker_owner) {
                    if let Some(att_unit) = att_tribe.units.get(attacker_idx) {
                        let settings = crate::settings::units::get_unit_setting(att_unit.unit_type);
                        if settings
                            .skills
                            .contains(&crate::types::SkillType::Infiltrate)
                        {
                            if crate::functions::is_enemy_city(state, self.target, attacker_owner) {
                                // Return city owner and None for unit index
                                (tile.owner, None)
                            } else {
                                return Err(
                                    "No unit at target and not an enemy city for infiltration"
                                        .to_string(),
                                );
                            }
                        } else {
                            return Err("No unit at target".to_string());
                        }
                    } else {
                        return Err("Attacker unit not found".to_string());
                    }
                } else {
                    return Err("Attacker tribe not found".to_string());
                }
            }
        };

        // Mark the tribe as having attacked this turn for pacifist task
        if let Some(tribe) = state.tribes.get_mut(&attacker_owner) {
            tribe.attacked_this_turn = true;
        }

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
