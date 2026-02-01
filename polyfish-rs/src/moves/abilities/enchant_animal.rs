use crate::actions::chain_undos;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnchantAnimalMove {
    pub tile_index: i32,
}

impl EnchantAnimalMove {
    pub fn new(tile_index: i32) -> Self {
        Self { tile_index }
    }
}

impl Move for EnchantAnimalMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();
        let tile_idx = self.tile_index;
        let pov_id = state.settings.current_player_turn_id;

        // 1. Consume Resource
        undos.push(crate::actions::resource::consume_resource(
            state,
            tile_idx,
            Some(crate::types::ResourceType::Game),
        ));

        // 2. Spend Stars (Costs three stars)
        undos.push(crate::actions::spend_stars(state, 3));

        // 3. Handle Unit on tile (Polypush)
        let push_result = crate::actions::units::push_unit(state, tile_idx);

        match push_result {
            Ok(res) => {
                undos.push(res.undo);
            }
            Err(_) => {}
        }

        // 4. Create Polytaur
        let unit_type = crate::types::UnitType::Polytaur;
        undos.push(crate::actions::units::spawn_unit(
            state, pov_id, unit_type, tile_idx, false,
        ));

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Enchant Animal at {}", self.tile_index)
    }
    fn serialize(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let Some(obj) = value.as_object_mut() {
            obj.insert("moveType".to_string(), serde_json::json!(MoveType::Ability));
            obj.insert(
                "ability".to_string(),
                serde_json::json!(AbilityType::EnchantAnimal),
            );
            obj.insert("target".to_string(), serde_json::json!(self.tile_index));
        }
        value
    }
}
