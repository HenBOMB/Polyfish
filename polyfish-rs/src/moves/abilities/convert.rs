use crate::actions::chain_undos;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConvertMove {
    pub unit_idx: i32,
    pub target_idx: i32,
}

impl ConvertMove {
    pub fn new(unit_idx: i32, target_idx: i32) -> Self {
        Self {
            unit_idx,
            target_idx,
        }
    }
}

impl Move for ConvertMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> MoveResult {
        let target_owner = state
            .tiles
            .get(&self.target_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);
        let actor_owner = state
            .tiles
            .get(&self.unit_idx)
            .and_then(|t| t._unit_owner_id)
            .unwrap_or(0);

        if target_owner == actor_owner || target_owner == 0 {
            return MoveResult {
                undo: Box::new(|_| {}),
                rewards: None,
            };
        }

        let mut target_unit: Option<crate::states::UnitState> = None;
        if let Some(tribe) = state.tribes.get_mut(&target_owner) {
            if let Some(pos) = tribe
                .units
                .iter()
                .position(|u| u.coords.idx == self.target_idx)
            {
                target_unit = Some(tribe.units.remove(pos));
            }
        }

        let mut undos: Vec<crate::actions::UndoCallback> = Vec::new();

        if let Some(mut unit) = target_unit {
            unit.owner = actor_owner;
            unit.converted = true;
            unit.attacked = true;
            unit.moved = true;

            if let Some(tribe) = state.tribes.get_mut(&actor_owner) {
                tribe.units.push(unit.clone());
            }
            if let Some(tile) = state.tiles.get_mut(&self.target_idx) {
                tile._unit_owner_id = Some(actor_owner);
            }

            let u_clone = unit.clone();
            undos.push(Box::new(move |s| {
                if let Some(nt) = s.tribes.get_mut(&actor_owner) {
                    if let Some(p) = nt
                        .units
                        .iter()
                        .position(|u| u.coords.idx == u_clone.coords.idx)
                    {
                        nt.units.remove(p);
                    }
                }
                if let Some(ot) = s.tribes.get_mut(&target_owner) {
                    let mut restored = u_clone.clone();
                    restored.owner = target_owner;
                    restored.converted = false;
                    ot.units.push(restored);
                }
                if let Some(tile) = s.tiles.get_mut(&u_clone.coords.idx) {
                    tile._unit_owner_id = Some(target_owner);
                }
            }));
        }

        MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        }
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Convert unit at {}", self.target_idx)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::Convert,
            "unitIdx": self.unit_idx,
            "targetIdx": self.target_idx
        })
    }
}
