//! Attack move implementation

use crate::actions::units::attack_unit;
use crate::functions::{get_enemy_at, get_unit_at};
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

    fn execute(&self, state: &mut GameState) -> MoveResult {
        let pov_id = state.settings.current_player_turn_id;

        // Find attacker
        let (attacker_owner, attacker_idx) = {
            let mut found = None;
            if let Some(tribe) = state.tribes.get(&pov_id) {
                for (idx, unit) in tribe.units.iter().enumerate() {
                    if unit.coords.idx == self.src {
                        found = Some((pov_id, idx));
                        break;
                    }
                }
            }
            match found {
                Some(f) => f,
                None => {
                    return MoveResult {
                        undo: Box::new(|_| {}),
                        rewards: None,
                    }
                }
            }
        };

        // Infiltrate check
        let attacker_unit = &state.tribes[&attacker_owner].units[attacker_idx];
        if crate::functions::has_skill(attacker_unit, crate::types::SkillType::Infiltrate) {
            return MoveResult {
                undo: crate::actions::units::infiltrate_city(
                    state,
                    attacker_owner,
                    attacker_idx,
                    self.target,
                ),
                rewards: None,
            };
        }

        // Find defender
        let (defender_owner, defender_idx) = {
            let mut found = None;
            for (tribe_id, tribe) in &state.tribes {
                if *tribe_id == pov_id {
                    continue; // Skip our own units
                }
                for (idx, unit) in tribe.units.iter().enumerate() {
                    if unit.coords.idx == self.target {
                        found = Some((*tribe_id, idx));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            match found {
                Some(f) => f,
                None => {
                    return MoveResult {
                        undo: Box::new(|_| {}),
                        rewards: None,
                    }
                }
            }
        };

        // Check for peace treaty
        if crate::functions::is_at_peace(state, pov_id, defender_owner) {
            return MoveResult {
                undo: Box::new(|_| {}),
                rewards: None,
            };
        }

        let undo = attack_unit(
            state,
            attacker_owner,
            attacker_idx,
            defender_owner,
            defender_idx,
        );

        MoveResult {
            undo,
            rewards: None,
        }
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
