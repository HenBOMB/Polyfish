//! Build move
//!
//! Build a structure on a tile.

use crate::functions::{get_resource_at, get_structure_at};
use crate::moves::{Move, MoveResult};
use crate::settings::get_resource_setting;
use crate::states::GameState;
use crate::types::{MoveType, StructureType, TerrainType};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BuildMove {
    pub target_index: i32,
    #[serde(rename = "structure")]
    pub structure_type: StructureType,
}

impl BuildMove {
    pub fn new(target_index: i32, structure_type: StructureType) -> Self {
        Self {
            target_index,
            structure_type,
        }
    }
}

// ... (struct remains)

impl Move for BuildMove {
    fn move_type(&self) -> MoveType {
        MoveType::Build
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        use crate::settings::structures::get_structure_setting;

        let pov_id = state.settings.current_player_turn_id;
        let settings = get_structure_setting(self.structure_type);

        // 0. Validation
        if self.structure_type == StructureType::Road
            && crate::functions::is_city(state, self.target_index)
        {
            return Err("Cannot build road on a city".to_string());
        }
        if let Some(cost) = settings.cost {
            if let Some(tribe) = state.tribes.get(&pov_id) {
                if tribe.stars < cost {
                    return Err(format!(
                        "Insufficient stars for build: need {} (Stars: {}), have {}",
                        cost, tribe.stars, tribe.stars
                    ));
                }
                // Check if already built unique structure
                if settings.reward_score > 0
                    && tribe
                        .built_unique_improvements
                        .contains(&self.structure_type)
                {
                    return Err(format!(
                        "Unique structure {:?} already built",
                        self.structure_type
                    ));
                }
            } else {
                return Err("Tribe not found".to_string());
            }
        } else if settings.reward_score > 0 {
            // Logic for free structures (monuments)
            if let Some(tribe) = state.tribes.get(&pov_id) {
                if tribe
                    .built_unique_improvements
                    .contains(&self.structure_type)
                {
                    return Err(format!(
                        "Unique structure {:?} already built",
                        self.structure_type
                    ));
                }
            }
        }

        use crate::actions::structure::build_structure;

        Ok(MoveResult {
            undo: build_structure(state, self.target_index, self.structure_type),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        format!("Build {:?} at {}", self.structure_type, self.target_index)
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({
            "moveType": self.move_type(),
            "target": self.target_index,
            "type": self.structure_type,
        })
    }

    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Ok(self.target_index as usize)
    }

    #[inline]
    fn structure_type(&self) -> Result<StructureType, String> {
        Ok(self.structure_type)
    }
}

/// Generate build / structure moves
pub fn generate_build_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    if let Some(tribe) = state.tribes.get(&pov_id) {
        use crate::settings::structures::get_structure_setting;
        use crate::settings::tasks::{check_task, get_task_setting};
        use strum::IntoEnumIterator;

        // --- OPTIMIZATION: Pre-calculate tribe-wide data ---

        let mut unlocked_structures = Vec::new();
        for struct_type in crate::types::StructureType::iter() {
            let settings = get_structure_setting(struct_type);

            if settings.cost.is_none() || settings.resource_type.is_some() {
                continue;
            }

            if tribe.stars < settings.cost.unwrap_or(0) {
                continue;
            }

            // Tribe-specific structure
            if let Some(s_tribe) = settings.tribe_type {
                if s_tribe != tribe.tribe_type {
                    continue;
                }
            }

            if settings.limited_per_tribe {
                let already_built = tribe.cities.iter().any(|c| {
                    c._territory.iter().any(|&t_idx| {
                        if let Some(s) = crate::functions::get_structure_at(state, t_idx) {
                            s.structure_type == struct_type
                        } else {
                            false
                        }
                    })
                });
                if already_built {
                    continue;
                }
            }

            if is_structure_unlocked(tribe, struct_type) {
                unlocked_structures.push((struct_type, settings));
            }
        }

        // Structures rewarded from tasks
        let mut pending_monument_structures = Vec::new();
        for task in crate::types::TaskType::iter() {
            let setting = get_task_setting(task);

            if let Some(tech) = setting.tech_type {
                if !crate::settings::technology::has_technology(&tribe.tech_vanilla, tech) {
                    continue;
                }
            }

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

            if check_task(state, task) {
                let struct_settings = get_structure_setting(setting.structure_type);
                if tribe.stars >= struct_settings.cost.unwrap_or(0) {
                    pending_monument_structures.push((setting.structure_type, struct_settings));
                }
            }
        }

        // Standard structures
        for city in &tribe.cities {
            for &idx in &city._territory {
                if crate::functions::get_enemy_at(state, idx, pov_id).is_some() {
                    continue;
                }

                let existing_struct = get_structure_at(state, idx);
                let tile = match state.tiles.get(&idx) {
                    Some(t) => t,
                    None => continue,
                };

                // 1. Resource-based structures
                if existing_struct.is_none() {
                    if let Some(res_type) = get_resource_at(state, idx) {
                        let res_settings = get_resource_setting(res_type);
                        if let Some(struct_type) = res_settings.struct_required {
                            let struct_settings = get_structure_setting(struct_type);
                            let cost = struct_settings.cost.unwrap_or(0);

                            if tribe.stars >= cost {
                                let tech_req = res_settings.tech_required;
                                let has_tech = match tech_req {
                                    crate::types::TechnologyType::Basic => true,
                                    t => tribe.tech_vanilla.iter().any(|tech| tech.tech_type == t),
                                };

                                if has_tech {
                                    moves.push(Box::new(BuildMove::new(idx, struct_type)));
                                }
                            }
                        }
                    }
                }

                // 2. Terrain-based structures
                for &(struct_type, ref settings) in &unlocked_structures {
                    if struct_type != StructureType::Road {
                        if existing_struct.is_some() {
                            continue;
                        }
                    } else if tile.has_road || crate::functions::is_city(state, idx) {
                        continue;
                    }

                    let terrain_valid = settings.terrain_types.contains(&tile.terrain_type) && !tile.is_algae();
                    if !terrain_valid {
                        continue;
                    }

                    if settings.limited_per_city {
                        let already_has = city._territory.iter().any(|&t_idx| {
                            crate::functions::get_structure_type_at(state, t_idx) == Some(struct_type)
                        });
                        if already_has {
                            continue;
                        }
                    }

                    if !settings.adjacent_types.is_empty() {
                        let adj = crate::functions::get_adjacent_indices(state, idx, 1);
                        let has_required_adj = adj.iter().any(|&n_idx| {
                            if let Some(s) = crate::functions::get_structure_at(state, n_idx)
                                && state.tiles.get(&n_idx).unwrap().owner == pov_id
                            {
                                settings.adjacent_types.contains(&s.structure_type)
                            } else {
                                false
                            }
                        });
                        if !has_required_adj {
                            continue;
                        }
                    }

                    if struct_type == StructureType::Bridge {
                        let size = state.settings.size;
                        let (x, y) = (idx % size, idx / size);

                        // Vertical check
                        let v_valid = if y > 0 && y < size - 1 {
                            let terrain_up = crate::fow::get_terrain_at(state, idx - size, pov_id);
                            let terrain_down = crate::fow::get_terrain_at(state, idx + size, pov_id);
                            // Land is everything non-water/non-ocean
                            !matches!(terrain_up, TerrainType::Water | TerrainType::Ocean)
                                && !matches!(terrain_down, TerrainType::Water | TerrainType::Ocean)
                        } else {
                            false
                        };

                        // Horizontal check
                        let h_valid = if x > 0 && x < size - 1 {
                            let terrain_left = crate::fow::get_terrain_at(state, idx - 1, pov_id);
                            let terrain_right = crate::fow::get_terrain_at(state, idx + 1, pov_id);
                            !matches!(terrain_left, TerrainType::Water | TerrainType::Ocean)
                                && !matches!(terrain_right, TerrainType::Water | TerrainType::Ocean)
                        } else {
                            false
                        };

                        if !v_valid && !h_valid {
                            continue;
                        }
                    }

                    if struct_type == StructureType::Clathrus {
                        let adj = crate::functions::get_adjacent_indices(state, idx, 1);
                        let has_adj_algae = adj.iter().any(|&n_idx| {
                            state.tiles.get(&n_idx).map_or(false, |t| {
                                t.is_algae() && state.tiles.get(&n_idx).unwrap().owner == pov_id
                            })
                        });
                        if !has_adj_algae {
                            continue;
                        }
                    }


                    moves.push(Box::new(BuildMove::new(idx, struct_type)));
                }

                // 3. Monuments (Tasks)
                if existing_struct.is_none() {
                    for &(struct_type, ref settings) in &pending_monument_structures {
                        if settings.terrain_types.contains(&tile.terrain_type) {
                            if tribe.built_unique_improvements.contains(&struct_type) {
                                continue;
                            }
                            // This if statement is dev specific behavior. Node: with god mode there is no need for this monument
                            if struct_type != StructureType::EyeOfGod || state.settings._fow {
                                moves.push(Box::new(BuildMove::new(idx, struct_type)));
                            }
                        }
                    }
                }
            }
        }

        // 4. Roads in neutral territory
        for (&idx, tile) in &state.tiles {
            if tile.owner == 0 && crate::functions::is_tile_explored(state, idx, pov_id) && !tile.has_road {
                // Cannot build if an enemy unit is on the tile
                if crate::functions::get_enemy_at(state, idx, pov_id).is_some() {
                    continue;
                }

                for &(struct_type, ref settings) in &unlocked_structures {
                    if struct_type == StructureType::Road {
                        if settings.terrain_types.contains(&tile.terrain_type) && !tile.is_algae() {
                            moves.push(Box::new(BuildMove::new(idx, struct_type)));
                        }
                        break;
                    }
                }
            }
        }
    }
}

fn is_structure_unlocked(tribe: &crate::states::TribeState, struct_type: StructureType) -> bool {
    use crate::settings::structures::get_structure_setting;
    use crate::settings::technology::get_technology_setting;
    use crate::types::TribeType;

    // Check if the structure itself is tribe-restricted
    let struct_setting = get_structure_setting(struct_type);
    if let Some(required_tribe) = struct_setting.tribe_type {
        if required_tribe != tribe.tribe_type {
            return false;
        }
    }

    let target = if tribe.tribe_type == TribeType::Polaris && struct_type == StructureType::IceBank {
        StructureType::Market
    } else {
        struct_type
    };

    if tribe.tribe_type == TribeType::Polaris && struct_type == StructureType::Market {
        return false;
    }

    for tech in &tribe.tech_vanilla {
        if !tech.discovered {
            continue;
        }
        let settings = get_technology_setting(tech.tech_type);
        if settings.unlocks_structure == Some(target) {
            return true;
        }
        if settings.unlocks_special_structures.contains(&target) {
            return true;
        }
    }
    false
}
