//! Build move
//!
//! Build a structure on a tile.

use crate::actions::structure::create_structure;
use crate::actions::UndoCallback;
use crate::moves::{Move, MoveResult};
use crate::states::GameState;
use crate::types::{MoveType, StructureType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildMove {
    pub tile_index: i32,
    #[serde(rename = "structure")]
    pub structure_type: StructureType,
}

impl BuildMove {
    pub fn new(tile_index: i32, structure_type: StructureType) -> Self {
        Self {
            tile_index,
            structure_type,
        }
    }
}

// ... (struct remains)

impl Move for BuildMove {
    fn move_type(&self) -> MoveType {
        MoveType::Build
    }

    fn execute(&self, state: &mut GameState) -> MoveResult {
        use crate::actions::city::add_population;
        use crate::actions::{chain_undos, spend_stars};
        use crate::functions::{get_adjacent_indices, get_city_owning_tile, get_structure_at};
        use crate::settings::structures::get_structure_setting;

        let mut undos = Vec::new();
        let settings = get_structure_setting(self.structure_type);

        // 1. Spend stars
        if let Some(cost) = settings.cost {
            undos.push(spend_stars(state, cost));
        }

        // 2. Create structure
        undos.push(create_structure(
            state,
            self.tile_index,
            self.structure_type,
            1,
        ));

        // 3. Add population
        if let Some(city) = get_city_owning_tile(state, self.tile_index) {
            let city_tile_idx = city.tile_index;
            let mut reward_pop = settings.reward_pop;

            // Handle adjacent multipliers (Windmill, Sawmill, Forge)
            if !settings.adjacent_types.is_empty() {
                let adj = get_adjacent_indices(state, self.tile_index, 1);
                let adj_count = adj
                    .iter()
                    .filter(|&&adj_idx| {
                        if let Some(s) = get_structure_at(state, adj_idx) {
                            settings.adjacent_types.contains(&s.structure_type)
                        } else {
                            false
                        }
                    })
                    .count() as i32;
                reward_pop *= adj_count;
            }

            if reward_pop > 0 {
                undos.push(add_population(state, city_tile_idx, reward_pop));
            }
        }

        MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        }
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Build {:?} at {}", self.structure_type, self.tile_index)
    }

    fn serialize(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        if let Some(obj) = value.as_object_mut() {
            obj.insert("moveType".to_string(), serde_json::json!(MoveType::Build));
        }
        value
    }
}

/// Generate build moves
pub fn generate_build_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&pov_id) {
        use crate::functions::{get_resource_at, get_structure_at};
        use crate::settings::resources::get_resource_setting;
        use crate::settings::structures::{get_structure_setting, PLACEABLE_STRUCTURES};
        use crate::types::ResourceType;

        for city in &tribe.cities {
            for &idx in &city._territory {
                // Check enemy
                if crate::functions::get_enemy_at(state, idx, pov_id).is_some() {
                    continue;
                }

                // Check if tile is empty of structures
                // (Except Roads can coexist? For simplicity assume 1 struct per tile for now or check override)
                if get_structure_at(state, idx).is_some() {
                    // TODO: Handle Road + Structure combinations
                    continue;
                }

                // 1. Resource-based structures
                if let Some(res_type) = get_resource_at(state, idx) {
                    let res_settings = get_resource_setting(res_type);
                    if let Some(struct_type) = res_settings.struct_type {
                        // Check tech requirement for structure (or resource tech implies it?)
                        // Usually structure has no tech req if resource requires it.
                        // But we check structure settings.
                        let struct_settings = get_structure_setting(struct_type);

                        // Check if we have the tech for the resource itself (harvest tech usually unlocks build)
                        // Actually e.g. Farming (tech) unlocks Crop (resource) AND Farm (structure).

                        // Check cost
                        let cost = struct_settings.cost.unwrap_or(0);

                        // Check tribe stars
                        if tribe.stars >= cost {
                            // Check tech (if structure has specific tech req not covered by resource visible req)
                            // Usually resource visibility implies we can build?
                            // No, we need to have researched the tech.
                            let tech_req = res_settings.tech_required;
                            let has_tech = match tech_req {
                                crate::types::TechnologyType::Unrequired => true,
                                t => tribe.tech_vanilla.iter().any(|tech| tech.tech_type == t),
                            };

                            if has_tech {
                                moves.push(Box::new(BuildMove::new(idx, struct_type)));
                            }
                        }
                    }
                }

                // 2. Free placement structures (Road, Port, etc.)
                let tile = match state.tiles.get(&idx) {
                    Some(t) => t,
                    None => continue,
                };

                for &struct_type in PLACEABLE_STRUCTURES {
                    // Roads can be built on top of structures (and structures on roads).
                    // If building a Road, we ignore existing structure check (but check if road exists on tile).
                    // If building a Structure, we check if there is an existing Structure (but ignore Road representation if it's separate).

                    if struct_type != StructureType::Road {
                        if get_structure_at(state, idx).is_some() {
                            continue;
                        }
                    } else {
                        // Check if already has road
                        if tile.has_road {
                            continue;
                        }
                    }

                    let settings = get_structure_setting(struct_type);

                    // Check terrain validity
                    if !settings.terrain_types.contains(&tile.terrain_type) {
                        continue;
                    }

                    // Check cost
                    let cost = settings.cost.unwrap_or(0);
                    if tribe.stars < cost {
                        continue;
                    }

                    // Check tech requirement
                    if is_structure_unlocked(tribe, struct_type) {
                        // Tribe-specific filters
                        if let Some(s_tribe) = settings.tribe_type {
                            if s_tribe != tribe.tribe_type {
                                continue;
                            }
                        }

                        // Special check for Market
                        if struct_type == StructureType::Market {
                            let adj = crate::functions::get_adjacent_indices(state, idx, 1);
                            let has_prod_building = adj.iter().any(|&n_idx| {
                                if let Some(s) = crate::functions::get_structure_at(state, n_idx) {
                                    matches!(
                                        s.structure_type,
                                        StructureType::Sawmill
                                            | StructureType::Windmill
                                            | StructureType::Forge
                                    )
                                } else {
                                    false
                                }
                            });

                            if !has_prod_building {
                                continue;
                            }
                        }

                        // Special check for Mycelium: one per city
                        if struct_type == StructureType::Mycelium {
                            let already_has_mycelium = city._territory.iter().any(|&t_idx| {
                                if let Some(s) = crate::functions::get_structure_at(state, t_idx) {
                                    s.structure_type == StructureType::Mycelium
                                } else {
                                    false
                                }
                            });
                            if already_has_mycelium {
                                continue;
                            }
                        }

                        // Polaris disabled (except if tech allows, but explicit exclude in logic was requested or just placeholder?)
                        // User said "im not sure what you mean", so I'll relax this check and rely on tech tree which replaces standard buildings.
                        // Standard buildings (Port, etc) should be filtered by tech.
                        // However, PLACEABLE_STRUCTURES might have standard ones.
                        // Since `is_structure_unlocked` checks tech, and Polaris doesn't have standard techs (Fishing/Sailing/etc replaced),
                        // this should be fine. I'll remove the explicit block.

                        moves.push(Box::new(BuildMove::new(idx, struct_type)));
                    }
                }

                // 3. Monuments (Tasks)
                {
                    use crate::settings::tasks::{check_task, get_task_setting};
                    use crate::types::TaskType;

                    // Iterate all tasks
                    let tasks = [
                        TaskType::Pacifist,
                        TaskType::Genius,
                        TaskType::Wealth,
                        TaskType::Explorer,
                        TaskType::Killer,
                        TaskType::Network,
                        TaskType::Metropolis,
                    ];

                    // Check if tile is empty (no structure)
                    // Already checked above: `if get_structure_at(state, idx).is_some() { continue; }`
                    // But we are inside the tile loop.

                    // Monuments can be placed on any empty tile owned by the player?
                    // TS check: `if (!settings.terrainType?.has(tile.type)) { ... }`
                    // Monuments usually have specific terrain? Or any land?
                    // StructureSettings for monuments usually allow any land.
                    // We need to check if the structure is already built?
                    // "Unique Improvements" - only 1 per tribe?
                    // TS: `if (!pov.builtUniqueImprovements.has(settings.structureType))`
                    // We need to track `builtUniqueImprovements` or check all structures.
                    // For now, let's iterate all tribe structures to check uniqueness if expensive, or assume we have a set.
                    // TribeState doesn't have `builtUniqueImprovements` set in Rust yet?
                    // Let's sweep all cities/structures to check if we already have it.

                    // Optimization: Build a set of existing structures once per tribe loop?
                    // Currently `generate_build_moves` iterates cities.
                    // Let's just check uniqueness.

                    for task in tasks {
                        let setting = get_task_setting(task);

                        // Check if we have the tech (if required)
                        if let Some(tech) = setting.tech_type {
                            if !crate::settings::technology::has_technology(
                                &tribe.tech_vanilla,
                                tech,
                            ) {
                                continue;
                            }
                        }

                        // Check if we already built it
                        let already_built = tribe.cities.iter().any(|c| {
                            c._territory.iter().any(|&t_idx| {
                                if let Some(s) = crate::functions::get_structure_at(state, t_idx) {
                                    s.structure_type == setting.structure_type
                                } else {
                                    false
                                }
                            })
                        });

                        if already_built {
                            continue;
                        }

                        // Check task completion
                        if check_task(state, task) {
                            // Valid placement?
                            // Monuments usually restricted to land?
                            // Let's assume standard terrain rules apply (defined in StructureSettings).
                            // If `StructureSettings` says "No terrain restriction" or matches tile.
                            // We can use `get_structure_setting` to verify terrain.
                            let struct_settings = get_structure_setting(setting.structure_type);
                            if !struct_settings.terrain_types.contains(&tile.terrain_type) {
                                continue;
                            }

                            // Cost? Monuments usually free? Or cost?
                            // Eye of God etc are usually free if unlocked? Or cost stars?
                            // TS code implies cost checks: `if (cost > pov.stars)`.
                            // If cost is 0, it passes.
                            let cost = struct_settings.cost.unwrap_or(0);
                            if tribe.stars >= cost {
                                moves.push(Box::new(BuildMove::new(idx, setting.structure_type)));
                            }
                        }
                    }
                }
            }
        }
    }
}

fn is_structure_unlocked(tribe: &crate::states::TribeState, struct_type: StructureType) -> bool {
    use crate::settings::technology::get_technology_setting;
    for tech in &tribe.tech_vanilla {
        let settings = get_technology_setting(tech.tech_type);
        if settings.unlocks_structure == Some(struct_type) {
            return true;
        }
        if settings.unlocks_special_structures.contains(&struct_type) {
            return true;
        }
    }
    false
}
