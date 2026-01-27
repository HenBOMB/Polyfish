//! Harvest move implementation

use crate::actions::resource::harvest_resource;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::MoveType;
use crate::functions::get_city_owning_tile;

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
    
    fn execute(&self, state: &mut GameState) -> MoveResult {
        let undo = harvest_resource(state, self.target);
        
        MoveResult {
            undo,
            rewards: None, // Population growth rewards handled by add_population internally usually? 
                           // In TS: popBranch returns rewards (like leveling up).
                           // My add_population implementation doesn't return rewards yet (just score).
                           // TODO: Implement level-up rewards properly.
        }
    }
    
    fn describe(&self, state: &GameState) -> String {
        let resource = state.resources.get(&self.target)
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
