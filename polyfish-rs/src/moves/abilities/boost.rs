use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoostMove {
    pub src_index: i32,
}

impl BoostMove {
    pub fn new(src_index: i32) -> Self {
        Self { src_index }
    }
}

impl Move for BoostMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let (unit_owner, unit_idx_in_tribe) = {
            let mut found = None;
            for (tribe_id, tribe) in &state.tribes {
                if let Some(pos) = tribe.units.iter().position(|u| u.coords.idx == self.src_index) {
                    found = Some((*tribe_id, pos));
                    break;
                }
            }
            match found {
                Some(f) => f,
                None => return Err("Unit not found".to_string()),
            }
        };

        let undo_boost = crate::actions::units::boost_unit(state, self.src_index);
        let undo_exhaust = crate::actions::units::end_unit_turn(state, unit_owner, unit_idx_in_tribe);

        Ok(MoveResult {
            undo: crate::actions::chain_undos(vec![undo_boost, undo_exhaust]),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Boost allies around {}", self.src_index)
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
        Ok(AbilityType::Swarm)
    }
}
