//! Promote unit ability (Veteran)

use crate::actions::{UndoCallback, chain_undos};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromoteMove {
    pub src_index: i32, // tile index
}

impl PromoteMove {
    pub fn new(src_index: i32) -> Self {
        Self { src_index }
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
            .get(&self.src_index)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        let unit_idx_in_tribe = if let Some(tribe) = state.tribes.get(&unit_owner) {
            tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.src_index)
        } else {
            None
        };

        if let Some(idx) = unit_idx_in_tribe {
            if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
                if let Some(unit) = tribe.units.get_mut(idx) {
                    let old_hp = unit.health;
                    let old_veteran = unit.veteran;

                    unit.veteran = true;
                    unit.max_health = crate::functions::get_unit_max_health(unit);
                    unit.health = unit.max_health;

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
        format!("Promote unit at {}", self.src_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "src": self.src_index,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::Promote)
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
