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
    pub src_index: i32,
    /// Target tile index  
    pub target_index: i32,
}

impl StepMove {
    pub fn new(src_index: i32, target_index: i32) -> Self {
        Self {
            src_index,
            target_index,
        }
    }
}

impl Move for StepMove {
    fn move_type(&self) -> MoveType {
        MoveType::Step
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;

        // Find the unit at src
        let (unit_owner, unit_idx) = {
            let mut found = None;
            if let Some(tribe) = state.tribes.get(&pov_id) {
                for (idx, unit) in tribe.units.iter().enumerate() {
                    if unit.coords.idx == self.src_index {
                        found = Some((pov_id, idx));
                        break;
                    }
                }
            }
            match found {
                Some(f) => f,
                // A silent Ok here would record a training sample for a move that
                // changed nothing; Err routes it into self_play's aborted_games.
                None => return Err(format!("No unit at source tile {}", self.src_index)),
            }
        };

        // Collision detection for invisible units
        if let Some(other_unit) = get_true_unit_at(state, self.target_index) {
            if other_unit.owner != unit_owner
                && other_unit
                    .effects
                    .contains(&crate::types::UnitEffect::Invisible)
            {
                // Reveal the cloak
                let other_owner = other_unit.owner;
                let other_pos = state.tribes[&other_owner]
                    .units
                    .iter()
                    .position(|u| u.coords.idx == self.target_index)
                    .unwrap();

                let undo_reveal = crate::actions::try_remove_effect(
                    state,
                    other_owner,
                    other_pos,
                    crate::types::UnitEffect::Invisible,
                );

                let undo_exhaust = crate::actions::units::end_unit_turn(state, unit_owner, unit_idx);

                return Ok(MoveResult {
                    undo: crate::actions::chain_undos(vec![undo_reveal, undo_exhaust]),
                    rewards: None,
                });
            }
        }

        let undo = step_unit(state, unit_owner, unit_idx, self.target_index, false);

        Ok(MoveResult {
            undo,
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Step: {} -> {}", self.src_index, self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "src": self.src_index,
            "target": self.target_index,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }
}
