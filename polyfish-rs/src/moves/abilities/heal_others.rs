use crate::actions::chain_undos;
use crate::actions::units::{end_unit_turn, heal_unit};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HealOthersMove {
    pub src_index: i32,
}

impl HealOthersMove {
    pub fn new(src_index: i32) -> Self {
        Self { src_index }
    }
}

impl Move for HealOthersMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let actor_owner = state
            .tiles
            .get(&self.src_index)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let actor_idx = if let Some(tribe) = state.tribes.get(&actor_owner) {
            tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.src_index)
        } else {
            None
        };

        if let Some(a_idx) = actor_idx {
            let mut undos = Vec::new();

            // Heal all adjacent allies
            let adj = crate::functions::get_adjacent_indices(state, self.src_index, 1);
            for target_idx in adj {
                let target_owner = state
                    .tiles
                    .get(&target_idx)
                    .and_then(|t| t._unit_owner_id)
                    .unwrap_or(0);
                if target_owner == actor_owner {
                    if let Some(tribe) = state.tribes.get(&target_owner) {
                        if let Some(pos) =
                            tribe.units.iter().position(|u| u.coords.idx == target_idx)
                        {
                            undos.push(heal_unit(state, target_owner, pos, 4));
                        }
                    }
                }
            }

            undos.push(end_unit_turn(state, actor_owner, a_idx));
            Ok(MoveResult {
                undo: chain_undos(undos),
                rewards: None,
            })
        } else {
            Err("Unit not found".to_string())
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Heal allies around {}", self.src_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type(),
            "src": self.src_index,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::HealOthers)
    }
}
