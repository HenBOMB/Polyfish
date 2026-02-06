//! Resource-related actions (consume, harvest)

use crate::actions::city::add_population;
use crate::actions::{UndoCallback, spend_stars};
use crate::functions::get_city_owning_tile;
use crate::settings::resources::get_resource_setting;
use crate::states::GameState;
use crate::types::ResourceType;

/// Consume a resource at a tile (remove or replace it)
pub fn consume_resource(
    state: &mut GameState,
    tile_idx: i32,
    replace_type: Option<ResourceType>,
) -> UndoCallback {
    let old_resource = match state.resources.get(&tile_idx) {
        Some(r) => r.clone(),
        None => {
            // If replacing but none existed? TS logic creates new if replacing.
            if let Some(rt) = replace_type {
                // Insert new
                state.resources.insert(
                    tile_idx,
                    Some(crate::states::ResourceState {
                        resource_type: rt,
                        tile_index: tile_idx,
                    }),
                );
                // Undo removes
                return Box::new(move |s| {
                    s.resources.shift_remove(&tile_idx);
                });
            }
            return Box::new(|_| {}); // Nothing to consume
        }
    };

    // Remove or replace
    if let Some(rt) = replace_type {
        if rt == ResourceType::None {
            state.resources.shift_remove(&tile_idx);
        } else {
            if let Some(Some(res)) = state.resources.get_mut(&tile_idx) {
                res.resource_type = rt;
            }
        }
    } else {
        state.resources.shift_remove(&tile_idx);
    }

    Box::new(move |s| {
        // Restore old resource
        s.resources.insert(tile_idx, old_resource);
    })
}

/// Harvest a resource
///
/// This handles:
/// - Cost checking/spending
/// - Removing the resource
/// - Adding population to the ruling city
pub fn harvest_resource(state: &mut GameState, tile_idx: i32) -> Result<UndoCallback, String> {
    // Get resource info
    let resource_type = match state.resources.get(&tile_idx) {
        Some(Some(r)) => r.resource_type,
        _ => return Err("No resource at tile".to_string()),
    };

    let settings = get_resource_setting(resource_type);

    // Find ruling city
    let city_tile_idx = match get_city_owning_tile(state, tile_idx) {
        Some(city) => city.tile_index,
        None => return Err("Resource not within city borders".to_string()),
    };

    let mut undos: Vec<UndoCallback> = Vec::new();

    // Spend stars
    if let Some(cost) = settings.cost {
        undos.push(spend_stars(state, cost));
    }

    // Consume resource
    undos.push(consume_resource(state, tile_idx, None));

    // Add population
    if settings.reward_pop > 0 {
        undos.push(add_population(state, city_tile_idx, settings.reward_pop));
    }

    Ok(Box::new(move |s| {
        for undo in undos.into_iter().rev() {
            undo(s);
        }
    }))
}
