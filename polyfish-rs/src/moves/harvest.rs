//! Harvest move implementation

use crate::actions::resource::harvest_resource;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;

/// A harvest move - gathering a resource
#[derive(Debug, Clone)]
pub struct HarvestMove {
    /// Target tile index
    pub target: i32,
}

impl HarvestMove {
    pub fn new(target: i32) -> Self {
        Self { target }
    }
}

impl Move for HarvestMove {
    fn move_type(&self) -> MoveType {
        MoveType::Harvest
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        use crate::settings::resources::get_resource_setting;

        let pov = state.settings.current_player_turn_id;

        // Check validation
        let cost = if let Some(Some(res)) = state.resources.get(&self.target).as_ref() {
            let settings = get_resource_setting(res.resource_type);
            settings.cost.unwrap_or(0)
        } else {
            return Err("No resource at tile".to_string());
        };

        if let Some(tribe) = state.tribes.get(&pov) {
            if tribe.stars < cost {
                return Err(format!(
                    "Insufficient stars for harvest: need {}, have {}",
                    cost, tribe.stars
                ));
            }
        }

        // Delegate to action
        let undo = harvest_resource(state, self.target)?;
        Ok(MoveResult {
            undo,
            rewards: None,
        })
    }

    fn describe(&self, state: &GameState) -> String {
        let resource = state
            .resources
            .get(&self.target)
            .and_then(|r| r.as_ref())
            .map(|r| format!("{:?}", r.resource_type))
            .unwrap_or_else(|| "Unknown".to_string());
        format!("Harvest {} at {}", resource, self.target)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Harvest,
            "target": self.target,
        })
    }
}
