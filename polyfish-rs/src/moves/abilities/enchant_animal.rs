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
    fn execute(&self, state: &mut GameState) -> MoveResult {
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

        MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        }
    }
    fn describe(&self, _state: &GameState) -> String {
        format!("Enchant Animal at {}", self.tile_index)
    }
    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Ability,
            "ability": AbilityType::EnchantAnimal,
            "target": self.tile_index
        })
    }
}
