use crate::actions::chain_undos;
use crate::actions::units::end_unit_turn;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FreezeAreaMove {
    pub src_index: i32,
}

impl FreezeAreaMove {
    pub fn new(src_index: i32) -> Self {
        Self { src_index }
    }
}

impl Move for FreezeAreaMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let owner = state
            .map.tiles
            .get(&self.src_index)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let actor_idx = if let Some(t) = state.tribes.get(&owner) {
            t.units.iter().position(|u| u.coords.idx == self.src_index)
        } else {
            None
        };

        if let Some(a_idx) = actor_idx {
            let mut undos = Vec::new();
            undos.push(crate::actions::freeze_area(state, owner, self.src_index));
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
        format!("Freeze area around {}", self.src_index)
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
        Ok(AbilityType::FreezeArea)
    }
}
