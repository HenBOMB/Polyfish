//! Capture move implementation

use crate::actions::resource::consume_resource;
use crate::actions::{chain_undos, end_unit_turn, gain_stars};
use crate::functions::get_unit_at;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{MoveType, StructureType};

/// A capture move - taking control of a village, city, or ruins
#[derive(Debug, Clone)]
pub struct CaptureMove {
    /// Tile index to capture
    pub src: i32,
}

impl CaptureMove {
    pub fn new(src: i32) -> Self {
        Self { src }
    }
}

impl Move for CaptureMove {
    fn move_type(&self) -> MoveType {
        MoveType::Capture
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let _pov_id = state.settings.current_player_turn_id;
        let capturer_idx = if let Some(unit) = get_unit_at(state, self.src) {
            state
                .tribes
                .get(&unit.owner)
                .and_then(|t| t.units.iter().position(|u| u.coords.idx == self.src))
                .unwrap()
        } else {
            return Err("No unit at capture site".to_string());
        };

        let unit_owner = get_unit_at(state, self.src).unwrap().owner;

        let mut undos = Vec::new();

        // End unit turn
        undos.push(end_unit_turn(state, unit_owner, capturer_idx));

        // Check structure type
        let struct_type = state
            .structures
            .get(&self.src)
            .and_then(|s| s.as_ref())
            .map(|s| s.structure_type);

        match struct_type {
            Some(StructureType::Village) => {
                let capture_undo = crate::actions::city::capture_city(state, self.src)?;
                undos.push(capture_undo);

                // Update capturer's home
                let map_size = state.settings.size;
                if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
                    if let Some(unit) = tribe.units.get_mut(capturer_idx) {
                        unit.home_coords =
                            Some(crate::coords::Coords::from_index(self.src, map_size));
                        unit.city_id = self.src;
                    }
                }
            }
            Some(StructureType::Ruin) => {
                undos.push(crate::actions::structure::capture_ruin(state, self.src));
            }
            _ => {
                undos.push(consume_resource(state, self.src, None));
                undos.push(gain_stars(state, 8));
            }
        }

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Capture at {}", self.src)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": MoveType::Capture,
            "src": self.src,
        })
    }
}
