//! Promote unit ability (Veteran)

use crate::actions::{chain_undos, UndoCallback};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromoteMove {
    pub unit_idx: i32, // tile index
}

impl PromoteMove {
    pub fn new(unit_idx: i32) -> Self {
        Self { unit_idx }
    }
}

impl Move for PromoteMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability // Or new implementation MoveType::Promote? No, TS uses Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos: Vec<UndoCallback> = Vec::new();

        let unit_owner = state
            .tiles
            .get(&self.unit_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        let unit_idx_in_tribe = if let Some(tribe) = state.tribes.get(&unit_owner) {
            tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.unit_idx)
        } else {
            None
        };

        if let Some(idx) = unit_idx_in_tribe {
            if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
                if let Some(unit) = tribe.units.get_mut(idx) {
                    let old_hp = unit.health;
                    let old_veteran = unit.veteran;

                    unit.veteran = true;
                    // Max HP logic: base * 10
                    unit.health = crate::functions::get_max_health(unit);

                    undos.push(Box::new(move |s: &mut GameState| {
                        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
                            if let Some(unit) = tribe.units.get_mut(idx) {
                                unit.veteran = old_veteran;
                                unit.health = old_hp;
                            }
                        }
                    }));
                }
            }

            Ok(MoveResult {
                undo: chain_undos(undos),
                rewards: None,
            })
        } else {
            Err("Unit not found".to_string())
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Promote unit at {}", self.unit_idx)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

pub fn generate_promote_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&pov_id) {
        for unit in &tribe.units {
            // Check promotion criteria
            if unit.kills >= 3
                && !unit.veteran
                && !crate::functions::has_skill(unit, crate::types::SkillType::Static)
            {
                // Polytopia: Can promote even if moved? Yes.
                moves.push(Box::new(PromoteMove::new(unit.coords.idx)));
            }
        }
    }
}
