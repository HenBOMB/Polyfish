//! Forest related abilities

use crate::actions::resource::consume_resource;
use crate::actions::{chain_undos, gain_stars, modify_terrain, spend_stars};
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{AbilityType, MoveType, ResourceType, TerrainType};

// --- ClearForest ---
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClearForestMove {
    pub target_index: i32,
}
impl ClearForestMove {
    pub fn new(target_index: i32) -> Self {
        Self { target_index }
    }
}

impl Move for ClearForestMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();
        // Clear Forest -> Field + Gain 1 Star
        undos.push(modify_terrain(state, self.target_index, TerrainType::Field));
        undos.push(gain_stars(state, 1));

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Clear Forest at {}", self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type(),
            "target": self.target_index,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::ClearForest)
    }
}

// --- GrowForest ---
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GrowForestMove {
    pub target_index: i32,
}
impl GrowForestMove {
    pub fn new(target_index: i32) -> Self {
        Self { target_index }
    }
}

impl Move for GrowForestMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();
        // Spiritualism: Field -> Forest (Cost 5)
        undos.push(modify_terrain(
            state,
            self.target_index,
            TerrainType::Forest,
        ));
        undos.push(spend_stars(state, 5));

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Grow Forest at {}", self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type(),
            "target": self.target_index,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::GrowForest)
    }
}

// --- BurnForest ---
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BurnForestMove {
    pub target_index: i32,
}
impl BurnForestMove {
    pub fn new(target_index: i32) -> Self {
        Self { target_index }
    }
}

impl Move for BurnForestMove {
    fn move_type(&self) -> MoveType {
        MoveType::Ability
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let mut undos = Vec::new();
        // Construction: Forest -> Field + Add Crop (Cost 2)
        undos.push(modify_terrain(state, self.target_index, TerrainType::Field));
        undos.push(spend_stars(state, 2));
        undos.push(consume_resource(
            state,
            self.target_index,
            Some(ResourceType::Crop),
        ));

        Ok(MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Burn Forest at {}", self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "type": self.ability_type(),
            "target": self.target_index,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Ok(AbilityType::BurnForest)
    }
}

pub fn generate_forest_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    use crate::settings::technology::has_technology;
    use crate::types::TechnologyType;

    let pov_id = state.settings.current_player_turn_id;

    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    let is_elyrion = tribe.tribe_type == crate::types::TribeType::Elyrion;

    let has_clear = !is_elyrion && has_technology(&tribe.tech_vanilla, TechnologyType::Forestry);
    let has_grow = has_technology(&tribe.tech_vanilla, TechnologyType::Spiritualism);
    let has_burn = !is_elyrion
        && tribe.tribe_type != crate::types::TribeType::Cymanti
        && has_technology(&tribe.tech_vanilla, TechnologyType::Construction);

    if !has_clear && !has_grow && !has_burn {
        return;
    }

    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            // Check enemy
            if crate::functions::get_enemy_at(state, tile_idx, pov_id).is_some() {
                continue;
            }
            // Check structure
            if let Some(structure) = crate::functions::get_structure_at(state, tile_idx) {
                if structure.structure_type != crate::types::StructureType::Road {
                    continue;
                }
            }

            if let Some(tile) = state.tiles.get(&tile_idx) {
                // Clear / Burn: Needs Forest
                if tile.terrain_type == TerrainType::Forest {
                    if has_clear {
                        moves.push(Box::new(ClearForestMove::new(tile_idx)));
                    }
                    if has_burn && tribe.stars >= 3 {
                        moves.push(Box::new(BurnForestMove::new(tile_idx)));
                    }
                }
                // Grow: Needs Field
                if tile.terrain_type == TerrainType::Field || tile.terrain_type == TerrainType::None
                {
                    if has_grow && tribe.stars >= 5 {
                        moves.push(Box::new(GrowForestMove::new(tile_idx)));
                    }
                }
            }
        }
    }
}
