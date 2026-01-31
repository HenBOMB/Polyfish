//! Break Peace move implementation

use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::states::PlayerId;
use crate::types::MoveType;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BreakPeaceMove {
    pub target_tribe_id: PlayerId,
}

impl BreakPeaceMove {
    pub fn new(target_tribe_id: PlayerId) -> Self {
        Self { target_tribe_id }
    }
}

impl Move for BreakPeaceMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        let target_id = self.target_tribe_id;

        let mut undos = Vec::new();

        // Break peace for both
        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            if let Some(relation) = tribe.relations.get_mut(&target_id) {
                let old_state = relation.state;
                relation.state = 0; // Neutral/War
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&pov_id) {
                        if let Some(r) = t.relations.get_mut(&target_id) {
                            r.state = old_state;
                        }
                    }
                }) as crate::actions::UndoCallback);
            }
        }

        if let Some(tribe) = state.tribes.get_mut(&target_id) {
            if let Some(relation) = tribe.relations.get_mut(&pov_id) {
                let old_state = relation.state;
                relation.state = 0;
                undos.push(Box::new(move |s: &mut GameState| {
                    if let Some(t) = s.tribes.get_mut(&target_id) {
                        if let Some(r) = t.relations.get_mut(&pov_id) {
                            r.state = old_state;
                        }
                    }
                }) as crate::actions::UndoCallback);
            }
        }

        Ok(MoveResult {
            undo: crate::actions::chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Break peace with tribe {}", self.target_tribe_id)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": crate::types::AbilityType::BreakPeace,
            "target": self.target_tribe_id,
        })
    }
}

pub fn generate_break_peace_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&pov_id) {
        for (&other_id, relation) in &tribe.relations {
            if relation.state == 1 {
                // At peace
                moves.push(Box::new(BreakPeaceMove::new(other_id)));
            }
        }
    }
}
