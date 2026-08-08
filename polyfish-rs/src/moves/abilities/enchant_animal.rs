use crate::actions::chain_undos;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnchantAnimalMove {
    pub target_index: i32,
}

impl EnchantAnimalMove {
    pub fn new(target_index: i32) -> Self {
        Self { target_index }
    }
}

impl Move for EnchantAnimalMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();
        let tile_idx = self.target_index;
        let pov_id = state.settings.current_player_turn_id;

        // 1. Consume Resource
        undos.push(crate::actions::resource::consume_resource(
            state,
            tile_idx,
            Some(crate::types::ResourceType::Game),
        ));

        // 2. Spend Stars (Costs three stars)
        undos.push(crate::actions::spend_stars(
            state,
            crate::version_sync::get_polytaur_cost(state),
        ));

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
        format!("Enchant Animal at {}", self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type().unwrap(),
            "target": self.target_index,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    fn cost(&self, state: &GameState) -> Option<i32> {
        Some(crate::version_sync::get_polytaur_cost(state))
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::EnchantAnimal)
    }
}
