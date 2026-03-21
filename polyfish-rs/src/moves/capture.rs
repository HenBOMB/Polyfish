//! Capture move implementation

use crate::actions::resource::consume_resource;
use crate::actions::{chain_undos, end_unit_turn, gain_stars};
use crate::functions::get_unit_at;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{MoveType, ResourceType, StructureType, TechnologyType};

/// A capture move - taking control of a village, city, or ruins
#[derive(Debug, Clone)]
pub struct CaptureMove {
    /// Tile index to capture
    pub src_index: i32,
}

impl CaptureMove {
    pub fn new(src_index: i32) -> Self {
        Self { src_index }
    }
}

impl Move for CaptureMove {
    fn move_type(&self) -> MoveType {
        MoveType::Capture
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let _pov_id = state.settings.current_player_turn_id;
        let capturer_idx = if let Some(unit) = get_unit_at(state, self.src_index) {
            state
                .tribes
                .get(&unit.owner)
                .and_then(|t| t.units.iter().position(|u| u.coords.idx == self.src_index))
                .unwrap()
        } else {
            return Err("No unit at capture site".to_string());
        };

        let unit_owner = get_unit_at(state, self.src_index).unwrap().owner;

        let mut undos = Vec::new();

        // End unit turn
        undos.push(end_unit_turn(state, unit_owner, capturer_idx));

        // Check structure type
        let struct_type = state
            .structures
            .get(&self.src_index)
            .and_then(|s| s.as_ref())
            .map(|s| s.structure_type);

        match struct_type {
            Some(StructureType::Village) => {
                let capture_undo = crate::actions::city::capture_city(state, self.src_index)?;
                undos.push(capture_undo);

                // Update capturer's home
                let map_size = state.settings.size;
                if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
                    if let Some(unit) = tribe.units.get_mut(capturer_idx) {
                        unit.home_coords =
                            Some(crate::coords::Coords::from_index(self.src_index, map_size));
                    }
                }
            }
            Some(StructureType::Ruin) => {
                undos.push(crate::actions::structure::capture_ruin(
                    state,
                    self.src_index,
                ));
            }
            _ => {
                // Check Starfish (collectible resource)
                if let Some(Some(resource)) = state.resources.get(&self.src_index) {
                    if resource.resource_type == crate::types::ResourceType::Starfish {
                        // Anti-cheat: verify Navigation tech
                        if let Some(tribe) = state.tribes.get(&unit_owner) {
                            if !crate::settings::technology::has_technology(
                                &tribe.tech_vanilla,
                                crate::types::TechnologyType::Navigation,
                            ) {
                                return Err("Cannot gather starfish without Navigation technology"
                                    .to_string());
                            }
                        }

                        let settings = crate::settings::resources::get_resource_setting(resource.resource_type);
                        undos.push(consume_resource(state, self.src_index, None));
                        undos.push(gain_stars(state, settings.reward_stars));
                    }
                }
            }
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Capture at {}", self.src_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "src": self.src_index,
        })
    }

    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Ok(self.src_index as usize)
    }
}
