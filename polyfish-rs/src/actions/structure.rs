//! Structure-related actions (create, destroy)

use crate::actions::city::add_population;
use crate::actions::{chain_undos, UndoCallback};
use crate::functions::get_city_owning_tile;
use crate::settings::structures::get_structure_setting;
use crate::states::{GameState, StructureState};
use crate::types::StructureType;

/// Create a structure at a tile
pub fn create_structure(state: &mut GameState, idx: i32, structure_type: StructureType, level: i32) -> UndoCallback {
    let old_struct = state.structures.get(&idx).cloned();
    
    let structure = StructureState {
        structure_type,
        level,
        founded: state.settings.turn,
        tile_index: idx,
        score: 0,
    };
    
    state.structures.insert(idx, Some(structure));
    
    // Valid references for move closure
    let pov_id = state.settings.current_player_turn_id;
    let old_has_road = state.tiles.get(&idx).map(|t| t.has_road).unwrap_or(false);
    
    // Sync has_road
    if structure_type == StructureType::Road {
         if let Some(tile) = state.tiles.get_mut(&idx) {
             tile.has_road = true;
         }
    }
    
    let mut undos: Vec<UndoCallback> = Vec::new();
    
    // Update connections if Road or Port
    if structure_type == StructureType::Road || structure_type == StructureType::Port {
        undos.push(crate::actions::connection::update_capital_connections(state, pov_id));
    }
    
    undos.push(Box::new(move |s| {
        // Undo connection logic handled by its own closure in chain
        
        // Restore has_road
        if structure_type == StructureType::Road {
            if let Some(tile) = s.tiles.get_mut(&idx) {
                tile.has_road = old_has_road;
            }
        }
        
        if let Some(old) = old_struct {
            s.structures.insert(idx, old);
        } else {
            s.structures.remove(&idx);
        }
    }));
    
    chain_undos(undos)
}

/// Destroy a structure at a tile
pub fn destroy_structure(state: &mut GameState, idx: i32) -> UndoCallback {
    let structure = match state.structures.get(&idx).cloned().flatten() {
        Some(s) => s,
        None => return Box::new(|_| {}),
    };
    
    // Remove structure
    state.structures.remove(&idx);
    
    let mut undos: Vec<UndoCallback> = Vec::new();
    
    // Restore structure on undo
    let undo_structure = structure.clone();
    undos.push(Box::new(move |s: &mut GameState| {
        s.structures.insert(idx, Some(undo_structure.clone()));
    }));
    
    // Handle population reduction for city structures
    if structure.structure_type != StructureType::Ruin {
        if let Some(city) = get_city_owning_tile(state, idx) {
            let city_tile_idx = city.tile_index;
            let settings = get_structure_setting(structure.structure_type);
            
            if settings.reward_pop > 0 {
                // Reduce population (negative add)
                undos.push(add_population(state, city_tile_idx, -settings.reward_pop));
            }
        }
    }
    
    // Valid stats for undo
    let pov_id = state.settings.current_player_turn_id;
    let old_has_road = state.tiles.get(&idx).map(|t| t.has_road).unwrap_or(false);
    
    // Sync has_road
    if structure.structure_type == StructureType::Road {
         if let Some(tile) = state.tiles.get_mut(&idx) {
             tile.has_road = false;
         }
    }
    
    // Update connections if Road or Port
    if structure.structure_type == StructureType::Road || structure.structure_type == StructureType::Port {
        undos.push(crate::actions::connection::update_capital_connections(state, pov_id));
    }

    undos.push(Box::new(move |s: &mut GameState| {
        // Restore has_road
        if structure.structure_type == StructureType::Road {
             if let Some(tile) = s.tiles.get_mut(&idx) {
                 tile.has_road = old_has_road;
             }
        }
    }));
    
    chain_undos(undos)
}
