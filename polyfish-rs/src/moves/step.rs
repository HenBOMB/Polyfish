//! Step move implementation

use crate::actions::units::step_unit;
use crate::functions::get_true_unit_at;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;

/// A step move - moving a unit from one tile to another
#[derive(Debug, Clone)]
pub struct StepMove {
    /// Source tile index
    pub src: i32,
    /// Target tile index  
    pub target: i32,
}

impl StepMove {
    pub fn new(src: i32, target: i32) -> Self {
        Self { src, target }
    }
}

impl Move for StepMove {
    fn move_type(&self) -> MoveType {
        MoveType::Step
    }

    fn execute(&self, state: &mut GameState) -> MoveResult {
        let pov_id = state.settings.current_player_turn_id;

        // Find the unit at src
        let (unit_owner, unit_idx) = {
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

        // Collision detection for invisible units
        if let Some(other_unit) = get_true_unit_at(state, self.target) {
            if other_unit.owner != unit_owner
                && other_unit
                    .effects
                    .contains(&crate::types::EffectType::Invisible)
            {
                // Reveal the cloak
                let other_owner = other_unit.owner;
                let other_pos = state.tribes[&other_owner]
                    .units
                    .iter()
                    .position(|u| u.coords.idx == self.target)
                    .unwrap();

                let undo_reveal = crate::actions::try_remove_effect(
                    state,
                    other_owner,
                    other_pos,
                    crate::types::EffectType::Invisible,
                );

                return MoveResult {
                    undo: undo_reveal,
                    rewards: None,
                };
            }
        }

        let undo = step_unit(state, unit_owner, unit_idx, self.target, false);

        MoveResult {
            undo,
            rewards: None,
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Step: {} -> {}", self.src, self.target)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Step,
            "src": self.src,
            "target": self.target,
        })
    }
}
