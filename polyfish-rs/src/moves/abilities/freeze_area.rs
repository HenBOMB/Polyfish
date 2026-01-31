use crate::actions::chain_undos;
use crate::actions::units::end_unit_turn;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FreezeAreaMove {
    pub unit_idx: i32,
}

impl FreezeAreaMove {
    pub fn new(unit_idx: i32) -> Self {
        Self { unit_idx }
    }
}

impl Move for FreezeAreaMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let owner = state
            .tiles
            .get(&self.unit_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let actor_idx = if let Some(t) = state.tribes.get(&owner) {
            t.units.iter().position(|u| u.coords.idx == self.unit_idx)
        } else {
            None
        };

        if let Some(a_idx) = actor_idx {
            let mut undos = Vec::new();
            undos.push(crate::actions::freeze_area(state, owner, self.unit_idx));
            undos.push(end_unit_turn(state, owner, a_idx));
            Ok(MoveResult {
                undo: chain_undos(undos),
                rewards: None,
            })
        } else {
            Err("Unit not found".to_string())
        }
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Freeze area around {}", self.unit_idx)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::FreezeArea,
            "unitIdx": self.unit_idx
        })
    }
}
