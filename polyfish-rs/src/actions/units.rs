//! Unit-specific action functions
//!
//! These are the core actions for manipulating units in the game.

use crate::actions::UndoCallback;
use crate::coords::Coords;
use crate::functions::*;
use crate::settings::{get_unit_setting, has_skill};
use crate::states::*;
use crate::types::*;

/// Remove a unit from the game
///
/// This handles:
/// - Removing from tribe's unit list
/// - Updating tile unit owner tracking
/// - Kill/casualty tracking
/// - Score adjustments
pub fn remove_unit(
    state: &mut GameState,
    unit_owner: PlayerId,
    unit_idx: usize,
    killer_owner: Option<PlayerId>,
    killer_idx: Option<usize>,
) -> UndoCallback {
    // Get the unit to remove
    let (removed_unit, tile_idx, unit_type) = {
        let tribe = match state.tribes.get(&unit_owner) {
            Some(t) => t,
            None => return Box::new(|_| {}),
        };
        let unit = match tribe.units.get(unit_idx) {
            Some(u) => u,
            None => return Box::new(|_| {}),
        };
        (unit.clone(), unit.coords.idx, unit.unit_type)
    };

    let mut undos: Vec<UndoCallback> = Vec::new();

    // 0. Drop Spores/Algae if Poisoned
    if removed_unit.effects.contains(&EffectType::Poison) {
        if let Some(tile) = state.tiles.get(&tile_idx) {
            let is_water_like = crate::functions::is_water_terrain(tile.terrain_type)
                || tile.terrain_type == TerrainType::Algae
                || tile.flooded;

            let has_struct = crate::functions::get_structure_at(state, tile_idx).is_some();

            if is_water_like {
                if tile.terrain_type != TerrainType::Algae && !has_struct {
                    // Drop Algae (convert Water/Ocean to Algae)
                    if let Some(t) = state.tiles.get_mut(&tile_idx) {
                        let old_terrain = t.terrain_type;
                        t.terrain_type = TerrainType::Algae;
                        undos.push(Box::new(move |s| {
                            if let Some(t) = s.tiles.get_mut(&tile_idx) {
                                t.terrain_type = old_terrain;
                            }
                        }));
                    }
                }
            } else {
                // Drop Spores Resource (if no structure)
                if !has_struct {
                    let resource = crate::states::ResourceState {
                        resource_type: crate::types::ResourceType::Spores,
                        tile_index: tile_idx,
                    };
                    let old_res_opt = state.resources.insert(tile_idx, Some(resource));

                    undos.push(Box::new(move |s| match old_res_opt {
                        Some(r) => {
                            s.resources.insert(tile_idx, r);
                        }
                        None => {
                            s.resources.remove(&tile_idx);
                        }
                    }));
                }
            }
        }
    }

    // Get unit cost for score
    let settings = get_unit_setting(unit_type);
    let cost = if settings.is_super { 10 } else { settings.cost };
    let score_deduction = 5 * cost;

    // Clear tile unit owner
    if let Some(tile) = state.tiles.get_mut(&tile_idx) {
        tile._unit_owner_id = None;
    }

    // Centipede head replacement logic
    // If unit has a child segment, promote it to head
    if let Some(child_idx) = removed_unit.child_unit_idx {
        if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
            // Adjust index if child is after removed unit
            let adj_child_idx = if child_idx > unit_idx {
                child_idx - 1
            } else {
                child_idx
            };

            if let Some(child) = tribe.units.get_mut(adj_child_idx) {
                // Promote segment to Centipede
                if child.unit_type == crate::types::UnitType::Segment {
                    child.unit_type = crate::types::UnitType::Centipede;
                }
                // Clear parent link since head is gone
                child.parent_unit_idx = None;
            }
        }
    }

    // Remove from tribe and update stats
    if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
        if unit_idx < tribe.units.len() {
            tribe.units.remove(unit_idx);
        }
        if !removed_unit.converted {
            tribe.score -= score_deduction;
        }
        tribe.casualties += 1;
    }

    // Update killer stats
    if let (Some(k_owner), Some(k_idx)) = (killer_owner, killer_idx) {
        if let Some(killer_tribe) = state.tribes.get_mut(&k_owner) {
            killer_tribe.kills += 1;
            if let Some(killer_unit) = killer_tribe.units.get_mut(k_idx) {
                killer_unit.kills += 1;
            }
        }
    }

    undos.push(Box::new(move |s| {
        // Undo killer stats
        if let (Some(k_owner), Some(k_idx)) = (killer_owner, killer_idx) {
            if let Some(killer_tribe) = s.tribes.get_mut(&k_owner) {
                killer_tribe.kills -= 1;
                if let Some(killer_unit) = killer_tribe.units.get_mut(k_idx) {
                    killer_unit.kills -= 1;
                }
            }
        }

        // Restore to tribe
        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
            if !removed_unit.converted {
                tribe.score += score_deduction;
            }
            tribe.casualties -= 1;
            tribe.units.insert(unit_idx, removed_unit.clone());
        }

        // Restore tile unit owner
        if let Some(tile) = s.tiles.get_mut(&tile_idx) {
            tile._unit_owner_id = Some(unit_owner);
        }
    }));

    crate::actions::chain_undos(undos)
}

/// Step a unit to a new tile
///
/// This handles:
/// - Moving the unit
/// - Embark/disembark logic
/// - Skill activations (Dash, Hide, Stomp, AutoFreeze)
/// - Ending the unit's turn
pub fn step_unit(
    state: &mut GameState,
    unit_owner: PlayerId,
    unit_idx: usize,
    to_tile_idx: i32,
    involuntary: bool,
) -> UndoCallback {
    let map_size = state.settings.size;

    // Get current unit state
    let (old_tile_idx, old_moved, old_attacked, old_type, old_passenger) = {
        let tribe = match state.tribes.get(&unit_owner) {
            Some(t) => t,
            None => return Box::new(|_| {}),
        };
        let unit = match tribe.units.get(unit_idx) {
            Some(u) => u,
            None => return Box::new(|_| {}),
        };
        (
            unit.coords.idx,
            unit.moved,
            unit.attacked,
            unit.unit_type,
            unit.passenger_type,
        )
    };

    // Clear old tile unit owner
    if let Some(tile) = state.tiles.get_mut(&old_tile_idx) {
        tile._unit_owner_id = None;
    }

    // Move the unit
    if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            unit.prev_coords.copy_from(&unit.coords);
            unit.coords.set_at(to_tile_idx, map_size);
            unit.moved = true;

            // End turn unless involuntary and has Skate
            if !involuntary || !has_skill(unit.unit_type, SkillType::Skate) {
                unit.attacked = true;
            }
        }
    }

    // Set new tile unit owner
    if let Some(tile) = state.tiles.get_mut(&to_tile_idx) {
        tile._unit_owner_id = Some(unit_owner);
    }

    let mut undos = Vec::new();
    let _rewards: Option<()> = None; // Stars, task progress

    let tiles_to_reveal = if let Some(tribe) = state.tribes.get(&unit_owner) {
        if let Some(unit) = tribe.units.get(unit_idx) {
            let range = if state.tiles.get(&to_tile_idx).map_or(false, |t| {
                t.terrain_type == crate::types::TerrainType::Mountain
            }) || has_skill(unit.unit_type, SkillType::Scout)
            {
                2
            } else {
                1
            };
            let mut adj = crate::functions::get_adjacent_indices(state, to_tile_idx, range);
            adj.push(to_tile_idx);
            Some(adj)
        } else {
            None
        }
    } else {
        None
    };

    if let Some(indices) = tiles_to_reveal {
        undos.push(crate::actions::discovery::discover_tiles(
            state,
            None,
            Some(indices),
        ));
    }

    // Algae: Auto-spawn algae on water/ocean tiles
    if has_skill(old_type, SkillType::Algae) {
        let tile = state.tiles.get(&to_tile_idx);
        let is_water_like = tile.map_or(false, |t| {
            matches!(
                t.terrain_type,
                crate::types::TerrainType::Water | crate::types::TerrainType::Ocean
            )
        });
        let has_algae = tile.map_or(false, |t| t.terrain_type == TerrainType::Algae);

        if is_water_like && !has_algae {
            // Convert Water/Ocean to Algae
            if let Some(t) = state.tiles.get_mut(&to_tile_idx) {
                let old_terrain = t.terrain_type;
                t.terrain_type = TerrainType::Algae;

                undos.push(Box::new(move |s| {
                    if let Some(t) = s.tiles.get_mut(&to_tile_idx) {
                        t.terrain_type = old_terrain;
                    }
                }));
            }

            // Spawn Fruit (according to rework spec, but LivingIsland spawns fruitless algae)
            if old_type != UnitType::LivingIsland {
                let resource = crate::states::ResourceState {
                    resource_type: crate::types::ResourceType::Fruit,
                    tile_index: to_tile_idx,
                };
                let old_resource = state.resources.get(&to_tile_idx).cloned().flatten();
                state.resources.insert(to_tile_idx, Some(resource));

                undos.push(Box::new(move |s| {
                    if let Some(old) = old_resource {
                        s.resources.insert(to_tile_idx, Some(old));
                    } else {
                        s.resources.remove(&to_tile_idx);
                    }
                }));
            }

            // Update connections (Algae acts as road for Cymanti)
            undos.push(crate::actions::connection::update_capital_connections(
                state, unit_owner,
            ));
        }
    }

    // Stomp: Deal 4 damage to adjacent enemies after moving
    if has_skill(old_type, SkillType::Stomp) {
        let adjacent_tiles = crate::functions::get_adjacent_indices(state, to_tile_idx, 1);
        let stomp_damage = 4;
        let mut stomp_targets = Vec::new();

        // 1. Identification Phase
        for adj_idx in adjacent_tiles {
            if let Some(adj_enemy) = crate::functions::get_enemy_at(state, adj_idx, unit_owner) {
                let adj_owner = adj_enemy.owner;
                if let Some(adj_tribe) = state.tribes.get(&adj_owner) {
                    if let Some((adj_unit_idx, _)) = adj_tribe
                        .units
                        .iter()
                        .enumerate()
                        .find(|(_, u)| u.coords.idx == adj_idx)
                    {
                        stomp_targets.push((adj_owner, adj_unit_idx));
                    }
                }
            }
        }

        // 2. Application Phase
        for (adj_owner, adj_unit_idx) in stomp_targets {
            // Apply Damage
            let mut unit_died = false;
            if let Some(tribe) = state.tribes.get_mut(&adj_owner) {
                if let Some(unit) = tribe.units.get_mut(adj_unit_idx) {
                    unit.health -= stomp_damage;
                    if unit.health <= 0 {
                        unit_died = true;
                    }

                    // Undo for this damage
                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&adj_owner) {
                            if let Some(u) = t.units.get_mut(adj_unit_idx) {
                                u.health += stomp_damage;
                            }
                        }
                    }));
                }
            }

            // Apply Poison (now safe)
            if has_skill(old_type, SkillType::Poison) {
                undos.push(crate::actions::try_add_effect(
                    state,
                    adj_owner,
                    adj_unit_idx,
                    EffectType::Poison,
                ));
            }

            // Remove if dead (now safe)
            if unit_died {
                undos.push(remove_unit(
                    state,
                    adj_owner,
                    adj_unit_idx,
                    Some(unit_owner),
                    Some(unit_idx),
                ));
            }
        }
    }

    // AutoFlood
    if has_skill(old_type, SkillType::AutoFlood) {
        if let Some(tile) = state.tiles.get_mut(&to_tile_idx) {
            if !tile.flooded {
                tile.flooded = true;
                undos.push(Box::new(move |s| {
                    if let Some(t) = s.tiles.get_mut(&to_tile_idx) {
                        t.flooded = false;
                    }
                }));
            }
        }
    }

    // Clathrus: Poisons enemy units that move onto it
    if let Some(dest_tile) = state.tiles.get(&to_tile_idx) {
        if let Some(dest_struct) = get_structure_type_at(state, to_tile_idx) {
            if dest_struct == StructureType::Clathrus {
                // If the Clathrus belongs to an enemy
                if dest_tile.owner != 0 && dest_tile.owner != unit_owner {
                    undos.push(crate::actions::try_add_effect(
                        state,
                        unit_owner,
                        unit_idx,
                        EffectType::Poison,
                    ));
                }
            }
        }
    }

    // Check embark/disembark
    let new_terrain = state.tiles.get(&to_tile_idx).map(|t| t.terrain_type);
    let struct_at_dest = get_structure_type_at(state, to_tile_idx);
    let is_port = struct_at_dest == Some(StructureType::Port);

    // Embark logic
    if is_port
        && !crate::functions::has_skill(
            {
                let tribe = state.tribes.get(&unit_owner).unwrap();
                tribe.units.get(unit_idx).unwrap()
            },
            SkillType::Float,
        )
        && !crate::functions::has_skill(
            {
                let tribe = state.tribes.get(&unit_owner).unwrap();
                tribe.units.get(unit_idx).unwrap()
            },
            SkillType::Fly,
        )
    {
        if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                match old_type {
                    UnitType::Cloak => unit.unit_type = UnitType::Dinghy,
                    UnitType::Dagger => unit.unit_type = UnitType::Pirate,
                    UnitType::Giant => unit.unit_type = UnitType::Juggernaut,
                    _ => {
                        unit.unit_type = UnitType::Raft;
                        unit.passenger_type = Some(old_type);
                    }
                }
            }
        }
    }
    // Disembark logic
    else if crate::functions::has_skill(
        {
            let tribe = state.tribes.get(&unit_owner).unwrap();
            tribe.units.get(unit_idx).unwrap()
        },
        SkillType::Carry,
    ) && !is_water_terrain_type(new_terrain.unwrap_or(TerrainType::Field))
    {
        if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                let new_type = match old_type {
                    UnitType::Dinghy => UnitType::Cloak,
                    UnitType::Pirate => UnitType::Dagger,
                    UnitType::Juggernaut => UnitType::Giant,
                    _ => old_passenger.unwrap_or(old_type),
                };
                unit.unit_type = new_type;
                unit.passenger_type = None;
                unit.attacked = true; // Ends the unit's turn
            }
        }
    }
    // Carry disembark: Naval units with passengers moving to land transform and spawn passenger
    else if has_skill(old_type, SkillType::Carry) && old_passenger.is_some() {
        let tile = state.tiles.get(&to_tile_idx);
        let is_water = tile.map_or(false, |t| {
            t.terrain_type == crate::types::TerrainType::Water
        });

        if !is_water {
            // Determine land unit type based on carrier type
            let land_unit_type = match old_type {
                crate::types::UnitType::Dinghy => crate::types::UnitType::Cloak,
                crate::types::UnitType::Pirate => crate::types::UnitType::Dagger,
                crate::types::UnitType::Juggernaut => crate::types::UnitType::Giant,
                crate::types::UnitType::Raft | crate::types::UnitType::Scout => {
                    // For Raft/Scout with passenger, transform to the passenger type
                    old_passenger.unwrap_or(crate::types::UnitType::Warrior)
                }
                _ => old_passenger.unwrap_or(crate::types::UnitType::Warrior),
            };

            // Transform carrier to land unit
            if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
                if let Some(unit) = tribe.units.get_mut(unit_idx) {
                    let old_unit_type = unit.unit_type;
                    unit.unit_type = land_unit_type;
                    unit.passenger_type = None;

                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&unit_owner) {
                            if let Some(u) = t.units.get_mut(unit_idx) {
                                u.unit_type = old_unit_type;
                                u.passenger_type = old_passenger;
                            }
                        }
                    }));
                }
            }
        }
    }
    // Hide logic
    else if crate::functions::has_skill(
        {
            let tribe = state.tribes.get(&unit_owner).unwrap();
            tribe.units.get(unit_idx).unwrap()
        },
        SkillType::Hide,
    ) && !crate::functions::has_effect(
        {
            let tribe = state.tribes.get(&unit_owner).unwrap();
            tribe.units.get(unit_idx).unwrap()
        },
        EffectType::Invisible,
    ) {
        if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.effects.insert(EffectType::Invisible);
            }
        }
    }
    // Dash logic (must be before ending turn/attacked status update if we want to allow move-then-attack)
    // Wait, the logic for dash is usually: if unit has dash, it doesn't lose 'attacked' status when moving?
    // Or it resets it?
    // In current implementation: unit.attacked = true is set in step_unit (line 146).
    // If unit has dash, we might want to reset it.
    if !involuntary
        && !old_moved
        && crate::functions::has_skill(
            {
                let tribe = state.tribes.get(&unit_owner).unwrap();
                tribe.units.get(unit_idx).unwrap()
            },
            SkillType::Dash,
        )
        && has_enemies_in_range(state, unit_owner, to_tile_idx, 1)
    {
        // Dash allows attacking after moving
        // Prohibited for Skate units on land
        let on_ice = state.tiles.get(&to_tile_idx).map_or(false, |t| t.frozen);
        let can_dash = !crate::functions::has_skill(
            {
                let tribe = state.tribes.get(&unit_owner).unwrap();
                tribe.units.get(unit_idx).unwrap()
            },
            SkillType::Skate,
        ) || on_ice;

        if can_dash {
            if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
                if let Some(unit) = tribe.units.get_mut(unit_idx) {
                    unit.attacked = false;
                }
            }
        }
    }

    undos.push(Box::new(move |s| {
        // Undo Hide
        if has_skill(old_type, SkillType::Hide) {
            if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
                if let Some(unit) = tribe.units.get_mut(unit_idx) {
                    unit.effects.remove(&EffectType::Invisible);
                }
            }
        }

        // Clear new tile
        if let Some(tile) = s.tiles.get_mut(&to_tile_idx) {
            tile._unit_owner_id = None;
        }

        // Restore unit state
        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.coords.set_at(old_tile_idx, map_size);
                unit.unit_type = old_type;
                unit.passenger_type = old_passenger;
                unit.moved = old_moved;
                unit.attacked = old_attacked;
            }
        }

        // Restore old tile
        if let Some(tile) = s.tiles.get_mut(&old_tile_idx) {
            tile._unit_owner_id = Some(unit_owner);
        }
    }));

    // Segment chain following: Move child to parent's old position
    if let Some(child_idx) = {
        state
            .tribes
            .get(&unit_owner)
            .and_then(|t| t.units.get(unit_idx))
            .and_then(|u| u.child_unit_idx)
    } {
        // Move child to this unit's previous position
        undos.push(step_unit(state, unit_owner, child_idx, old_tile_idx, true));
    }

    crate::actions::chain_undos(undos)
}

/// Calculate combat damage between attacker and defender
pub fn calculate_combat(
    attacker_attack: f32,
    attacker_health: i32,
    attacker_max_health: i32,
    defender_defense: f32,
    defender_health: i32,
    defender_max_health: i32,
    defense_bonus: f32,
) -> CombatResult {
    // Polytopia official damage formula:
    // damage = (attack_force / (attack_force + defense_force)) * attacker_attack * 4.5
    // attack_force = attacker_attack * (attacker_health / attacker_max_health)
    // defense_force = (defender_defense * defense_bonus) * (defender_health / defender_max_health)

    let attack_force = attacker_attack * (attacker_health as f32 / attacker_max_health as f32);
    let defense_force =
        (defender_defense * defense_bonus) * (defender_health as f32 / defender_max_health as f32);

    let total_force = attack_force + defense_force;
    let attack_result = (attack_force / total_force * attacker_attack * 4.5).round();

    // Retaliation damage (if defender survives)
    let new_defender_health = defender_health as f32 - attack_result;
    let defense_damage = if new_defender_health > 0.0 {
        let ret_force =
            (defender_defense * defense_bonus) * (new_defender_health / defender_max_health as f32);
        let ret_total = attacker_attack + ret_force;
        (ret_force / ret_total * (defender_defense * defense_bonus) * 4.5).round()
    } else {
        0.0
    };

    CombatResult {
        attack_damage: attack_result,
        defense_damage,
        splash_damage: 0.0,
    }
}

/// Attack action - one unit attacks another
///
/// This handles:
/// - Damage calculation and application
/// - Splash damage for units with Splash skill
/// - Unit death and removal
/// - Retaliation damage
/// - Moving to defender's tile if killed (for melee)
pub fn attack_unit(
    state: &mut GameState,
    attacker_owner: PlayerId,
    attacker_idx: usize,
    defender_owner: PlayerId,
    defender_idx: Option<usize>,
) -> UndoCallback {
    let mut undos: Vec<UndoCallback> = Vec::new();

    // Get attacker stats
    let (atk_atk, atk_health, atk_max_health, atk_skills, atk_type, atk_coords) = {
        let tribe = state.tribes.get(&attacker_owner).unwrap();
        let unit = tribe.units.get(attacker_idx).unwrap();
        (
            get_unit_attack(unit),
            unit.health,
            get_max_health(unit),
            get_unit_setting(unit.unit_type).skills.clone(),
            unit.unit_type,
            unit.coords.idx,
        )
    };

    // If defender_idx is None, this is an Infiltration attack on a city
    if defender_idx.is_none() {
        if atk_skills.contains(&SkillType::Infiltrate) {
            // Determine spawn type (Moth -> InsectEgg, Cloak -> Daggers)
            let is_moth = atk_type == UnitType::Moth;

            // Logic for Infiltration
            // 1. Defenses broken / Poisoned? (Moth poisons defenses)
            // 2. Spawn units
            // 3. Remove infiltrator

            // Assume target is where the attack was directed
            // Wait, we don't have target coord here if we don't look it up or pass it.
            // AttackMove passed defender_owner based on City Owner. We don't have city coord directly from `defender_idx: None`.
            // BUT `AttackMove` passes `self.target` which is the tile index. We usually pass indices to actions.
            // Problem: `attack_unit` takes defender_idx (unit index). It doesn't take tile index.
            // We need to find the city tile.
            // Since we know defender_owner, iterating their cities to find one adjacent to attacker?
            // Attack range is 1 for Infiltrate.

            let target_city_idx = if let Some(def_tribe) = state.tribes.get(&defender_owner) {
                def_tribe.cities.iter().find_map(|c| {
                    if crate::functions::get_adjacent_indices(state, atk_coords, 1)
                        .contains(&c.tile_index)
                    {
                        Some(c.tile_index)
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            if let Some(city_tile_idx) = target_city_idx {
                // 1. Cause Riot
                if let Some(def_tribe) = state.tribes.get_mut(&defender_owner) {
                    if let Some(c) = def_tribe
                        .cities
                        .iter_mut()
                        .find(|c| c.tile_index == city_tile_idx)
                    {
                        c._riot = true;
                        // Add undo
                        undos.push(Box::new(move |s| {
                            if let Some(t) = s.tribes.get_mut(&defender_owner) {
                                if let Some(c) =
                                    t.cities.iter_mut().find(|c| c.tile_index == city_tile_idx)
                                {
                                    c._riot = false;
                                }
                            }
                        }));
                    }
                }

                // 2. Spawn Logic
                if is_moth {
                    // Moth: Poison adjacent units and spawn Eggs
                    let adj = crate::functions::get_adjacent_indices(state, city_tile_idx, 1);
                    for t_idx in &adj {
                        if let Some(enemy) =
                            crate::functions::get_enemy_at(state, *t_idx, attacker_owner)
                        {
                            // Poison enemy
                            let enemy_owner = enemy.owner;
                            if let Some(e_tribe) = state.tribes.get(&enemy_owner) {
                                if let Some(e_idx) =
                                    e_tribe.units.iter().position(|u| u.coords.idx == *t_idx)
                                {
                                    undos.push(crate::actions::try_add_effect(
                                        state,
                                        enemy_owner,
                                        e_idx,
                                        EffectType::Poison,
                                    ));
                                }
                            }
                        }
                    }

                    // Spawn Eggs
                    let mut spawn_count = 0;
                    for t_idx in &adj {
                        if spawn_count >= 3 {
                            break;
                        }
                        if crate::functions::get_unit_at(state, *t_idx).is_none() {
                            undos.push(spawn_unit(
                                state,
                                attacker_owner,
                                UnitType::InsectEgg,
                                *t_idx,
                                false,
                            ));
                            spawn_count += 1;
                        }
                    }
                } else {
                    // Cloak: Spawn Daggers
                    let adj = crate::functions::get_adjacent_indices(state, city_tile_idx, 1);
                    let mut spawn_count = 0;
                    for t_idx in &adj {
                        if spawn_count >= 3 {
                            break;
                        }
                        if crate::functions::get_unit_at(state, *t_idx).is_none() {
                            undos.push(spawn_unit(
                                state,
                                attacker_owner,
                                UnitType::Dagger,
                                *t_idx,
                                false,
                            ));
                            spawn_count += 1;
                        }
                    }
                }

                // 3. Remove Infiltrator
                undos.push(remove_unit(state, attacker_owner, attacker_idx, None, None));
            }

            return crate::actions::chain_undos(undos);
        } else {
            return Box::new(|_| {}); // Should not happen given AttackMove checks
        }
    }

    let defender_idx = defender_idx.unwrap(); // fast fail if logic error

    let (def_def, def_health, def_max_health, defense_bonus, def_coords) = {
        let tribe = state.tribes.get(&defender_owner).unwrap();
        let unit = tribe.units.get(defender_idx).unwrap();
        (
            get_unit_defense(unit),
            unit.health,
            get_max_health(unit),
            get_defense_bonus(state, unit),
            unit.coords.idx,
        )
    };

    // Calculate combat result
    let result = calculate_combat(
        atk_atk,
        atk_health,
        atk_max_health,
        def_def,
        def_health,
        def_max_health,
        defense_bonus,
    );

    // Apply damage to defender
    let def_damage = result.attack_damage as i32;
    if let Some(tribe) = state.tribes.get_mut(&defender_owner) {
        if let Some(unit) = tribe.units.get_mut(defender_idx) {
            unit.health -= def_damage;

            // Boost effect is lost when attacked
            if unit.effects.contains(&EffectType::Boost) {
                unit.effects.remove(&EffectType::Boost);

                undos.push(Box::new(move |s| {
                    if let Some(t) = s.tribes.get_mut(&defender_owner) {
                        if let Some(u) = t.units.get_mut(defender_idx) {
                            u.effects.insert(EffectType::Boost);
                        }
                    }
                }));
            }
        }
    }
    undos.push(Box::new(move |s| {
        if let Some(tribe) = s.tribes.get_mut(&defender_owner) {
            if let Some(unit) = tribe.units.get_mut(defender_idx) {
                unit.health += def_damage;
            }
        }
    }));

    // Apply splash damage to adjacent enemies if attacker has Splash skill
    if atk_skills.contains(&SkillType::Splash) {
        let splash_damage = def_damage / 2; // 50% of primary damage, rounded down
        let adjacent_tiles = crate::functions::get_adjacent_indices(state, def_coords, 1);
        let mut splash_targets = Vec::new();

        // 1. Identification Phase
        for adj_idx in adjacent_tiles {
            if let Some(adj_enemy) = crate::functions::get_enemy_at(state, adj_idx, attacker_owner)
            {
                let adj_owner = adj_enemy.owner;
                if let Some(adj_tribe) = state.tribes.get(&adj_owner) {
                    if let Some((adj_unit_idx, _)) = adj_tribe
                        .units
                        .iter()
                        .enumerate()
                        .find(|(_, u)| u.coords.idx == adj_idx)
                    {
                        splash_targets.push((adj_owner, adj_unit_idx));
                    }
                }
            }
        }

        // 2. Application Phase
        for (adj_owner, adj_unit_idx) in splash_targets {
            let mut unit_died = false;
            // Apply Damage
            if let Some(tribe) = state.tribes.get_mut(&adj_owner) {
                if let Some(unit) = tribe.units.get_mut(adj_unit_idx) {
                    unit.health -= splash_damage;
                    if unit.health <= 0 {
                        unit_died = true;
                    }

                    // Undo for this splash damage
                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&adj_owner) {
                            if let Some(u) = t.units.get_mut(adj_unit_idx) {
                                u.health += splash_damage;
                            }
                        }
                    }));
                }
            }

            // Apply Poison (if attacker has it)
            if atk_skills.contains(&SkillType::Poison) {
                undos.push(crate::actions::try_add_effect(
                    state,
                    adj_owner,
                    adj_unit_idx,
                    EffectType::Poison,
                ));
            }

            // Check if splashed unit died
            if unit_died {
                undos.push(remove_unit(
                    state,
                    adj_owner,
                    adj_unit_idx,
                    Some(attacker_owner),
                    Some(attacker_idx),
                ));
            }
        }
    }

    // Check if defender died
    let defender_health_after = {
        state
            .tribes
            .get(&defender_owner)
            .and_then(|t| t.units.get(defender_idx))
            .map(|u| u.health)
            .unwrap_or(0)
    };

    if defender_health_after <= 0 {
        // Remove defender
        undos.push(remove_unit(
            state,
            defender_owner,
            defender_idx,
            Some(attacker_owner),
            Some(attacker_idx),
        ));

        // Eat: Spawn segment when Centipede kills via attack (not retaliation)
        if atk_skills.contains(&SkillType::Eat) {
            let atk_prev_idx = {
                state
                    .tribes
                    .get(&attacker_owner)
                    .and_then(|t| t.units.get(attacker_idx))
                    .map(|u| u.prev_coords.idx)
                    .unwrap_or(-1)
            };

            if atk_prev_idx >= 0 && state.tiles.contains_key(&atk_prev_idx) {
                // Check if tile is unoccupied
                let tile_occupied = crate::functions::get_unit_at(state, atk_prev_idx).is_some();

                if !tile_occupied {
                    // Spawn segment at previous position
                    let new_segment_idx = state
                        .tribes
                        .get(&attacker_owner)
                        .map(|t| t.units.len())
                        .unwrap_or(0);

                    undos.push(spawn_unit(
                        state,
                        attacker_owner,
                        crate::types::UnitType::Segment,
                        atk_prev_idx,
                        false,
                    ));

                    // Link segment to parent
                    if let Some(tribe) = state.tribes.get_mut(&attacker_owner) {
                        let old_child_idx_value =
                            tribe.units.get(attacker_idx).and_then(|u| u.child_unit_idx);

                        // Set segment's parent to be the attacker
                        if let Some(segment) = tribe.units.get_mut(new_segment_idx) {
                            segment.parent_unit_idx = Some(attacker_idx);
                            segment.child_unit_idx = old_child_idx_value;
                        }
                    }

                    // Update parent-child chain
                    if let Some(tribe) = state.tribes.get_mut(&attacker_owner) {
                        let old_child_idx_value =
                            tribe.units.get(attacker_idx).and_then(|u| u.child_unit_idx);

                        // Update old child's parent if exists
                        if let Some(old_child_idx) = old_child_idx_value {
                            if let Some(old_child) = tribe.units.get_mut(old_child_idx) {
                                old_child.parent_unit_idx = Some(new_segment_idx);
                            }
                        }

                        // Set attacker's child to new segment
                        if let Some(attacker) = tribe.units.get_mut(attacker_idx) {
                            attacker.child_unit_idx = Some(new_segment_idx);
                        }
                    }
                }
            }
        }

        // Move attacker to defender's position if melee
        let atk_settings = get_unit_setting({
            state
                .tribes
                .get(&attacker_owner)
                .and_then(|t| t.units.get(attacker_idx))
                .map(|u| u.unit_type)
                .unwrap_or(UnitType::None)
        });

        if atk_settings.range < 2 {
            undos.push(step_unit(
                state,
                attacker_owner,
                attacker_idx,
                def_coords,
                true,
            ));
        }
    } else {
        // Apply retaliation damage to attacker (skip if defender has Stiff)
        let def_skills = get_unit_setting({
            state
                .tribes
                .get(&defender_owner)
                .and_then(|t| t.units.get(defender_idx))
                .map(|u| u.unit_type)
                .unwrap_or(UnitType::Warrior)
        })
        .skills
        .clone();

        let can_retaliate = !def_skills.contains(&SkillType::Stiff);
        let atk_damage = result.defense_damage as i32;
        if atk_damage > 0 && can_retaliate {
            if let Some(tribe) = state.tribes.get_mut(&attacker_owner) {
                if let Some(unit) = tribe.units.get_mut(attacker_idx) {
                    unit.health -= atk_damage;
                }
            }
            undos.push(Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&attacker_owner) {
                    if let Some(unit) = tribe.units.get_mut(attacker_idx) {
                        unit.health += atk_damage;
                    }
                }
            }));

            // Check if attacker died from retaliation
            let attacker_health_after = {
                state
                    .tribes
                    .get(&attacker_owner)
                    .and_then(|t| t.units.get(attacker_idx))
                    .map(|u| u.health)
                    .unwrap_or(0)
            };

            if attacker_health_after <= 0 {
                undos.push(remove_unit(
                    state,
                    attacker_owner,
                    attacker_idx,
                    Some(defender_owner),
                    Some(defender_idx),
                ));
            }
        }

        // Apply freeze effect if attacker has Freeze skill
        if atk_skills.contains(&SkillType::Freeze) {
            if let Some(tribe) = state.tribes.get_mut(&defender_owner) {
                if let Some(unit) = tribe.units.get_mut(defender_idx) {
                    unit.effects.insert(EffectType::Frozen);
                }
            }
            undos.push(Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&defender_owner) {
                    if let Some(unit) = tribe.units.get_mut(defender_idx) {
                        unit.effects.remove(&EffectType::Frozen);
                    }
                }
            }));
        }
    }

    // End attacker's turn (unless Persist allows chain attacks or DoubleAttack allows second attack)
    if let Some(tribe) = state.tribes.get_mut(&attacker_owner) {
        if let Some(unit) = tribe.units.get_mut(attacker_idx) {
            let old_attacked = unit.attacked;
            let old_moved = unit.moved;
            let old_attacks_performed = unit.attacks_performed;

            // Persist: If attacker has Persist skill and killed the defender, don't set attacked=true
            let killed_defender = defender_health_after <= 0;
            let has_persist = crate::functions::has_skill(unit, SkillType::Persist);
            let has_double_attack = crate::functions::has_skill(unit, SkillType::DoubleAttack);

            // Increment attack counter for DoubleAttack tracking
            unit.attacks_performed += 1;

            // Set attacked flag based on skill interactions
            if !(killed_defender && has_persist) {
                // DoubleAttack allows 2 attacks, so only set attacked=true after 2nd attack
                if has_double_attack && unit.attacks_performed < 2 {
                    // Don't set attacked yet, allow second attack
                } else {
                    unit.attacked = true;
                }
            }

            // Escape allows moving after attacking
            // Prohibited for Skate units on land
            let on_ice = state
                .tiles
                .get(&unit.coords.idx)
                .map_or(false, |t| t.frozen);
            let can_escape = crate::functions::has_skill(unit, SkillType::Escape)
                && (!crate::functions::has_skill(unit, SkillType::Skate) || on_ice);

            if !can_escape {
                unit.moved = true;
            }

            undos.push(Box::new(move |s| {
                if let Some(tribe) = s.tribes.get_mut(&attacker_owner) {
                    if let Some(u) = tribe.units.get_mut(attacker_idx) {
                        u.attacked = old_attacked;
                        u.moved = old_moved;
                        u.attacks_performed = old_attacks_performed;
                    }
                }
            }));
        }
    }
    Box::new(move |s| {
        for undo in undos.into_iter().rev() {
            undo(s);
        }
    })
}

/// Heal a unit
pub fn heal_unit(
    state: &mut GameState,
    unit_owner: PlayerId,
    unit_idx: usize,
    amount: i32,
) -> UndoCallback {
    let old_health = {
        state
            .tribes
            .get(&unit_owner)
            .and_then(|t| t.units.get(unit_idx))
            .map(|u| u.health)
            .unwrap_or(0)
    };

    if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            let max_hp = get_max_health(unit);
            unit.health = (unit.health + amount).min(max_hp);
        }
    }

    Box::new(move |s| {
        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.health = old_health;
            }
        }
    })
}

// Helper functions

/// Check if terrain is water
fn is_water_terrain_type(terrain: TerrainType) -> bool {
    terrain == TerrainType::Water || terrain == TerrainType::Ocean
}

/// Check if there are enemies in range from a tile
fn has_enemies_in_range(state: &GameState, owner: PlayerId, from_idx: i32, range: i32) -> bool {
    let adjacent = get_adjacent_indices(state, from_idx, range);
    for idx in adjacent {
        if let Some(_enemy) = get_enemy_at(state, idx, owner) {
            return true;
        }
    }
    false
}

/// Push a unit to a valid adjacent tile
pub fn push_unit(state: &mut GameState, tile_idx: i32) -> Result<crate::moves::MoveResult, String> {
    use crate::functions::{calculate_pushable_position, get_true_unit_at};
    use crate::moves::MoveResult;

    // Find unit to push
    let (unit_owner, unit_idx) = {
        let unit = match get_true_unit_at(state, tile_idx) {
            Some(u) => u,
            None => {
                return Ok(MoveResult {
                    undo: Box::new(|_| {}),
                    rewards: None,
                })
            }
        };

        let tribe = state.tribes.get(&unit.owner).ok_or("Tribe not found")?;
        let idx = tribe
            .units
            .iter()
            .position(|u| u.coords.idx == tile_idx)
            .ok_or("Unit idx not found")?;
        (unit.owner, idx)
    };

    let unit = get_true_unit_at(state, tile_idx).unwrap();
    let old_moved = unit.moved;
    let old_attacked = unit.attacked;

    // Calculate destination
    let moved_to = calculate_pushable_position(state, unit);

    let undo_push: UndoCallback;
    let rewards = None;

    if let Some(dest_idx) = moved_to {
        if get_true_unit_at(state, dest_idx).is_some() {
            return Err("Push target occupied".to_string());
        }

        // Push is a forced step
        let undo = step_unit(state, unit_owner, unit_idx, dest_idx, true);
        undo_push = undo;
    } else {
        // If no valid position, unit dies (is squashed)
        undo_push = remove_unit(state, unit_owner, unit_idx, None, None);
    }

    // Restore unit state on undo
    let final_undo = Box::new(move |s: &mut GameState| {
        undo_push(s);
        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.moved = old_moved;
                unit.attacked = old_attacked;
            }
        }
    });

    Ok(MoveResult {
        undo: final_undo,
        rewards,
    })
}

/// Internal helper to spawn a unit into the state.
/// This handles UnitState initialization, tribe unit list, score, and tile owner tracking.
pub fn spawn_unit(
    state: &mut GameState,
    owner: PlayerId,
    unit_type: UnitType,
    tile_idx: i32,
    force_independent: bool,
) -> UndoCallback {
    let settings = get_unit_setting(unit_type);
    let independent = force_independent || settings.skills.contains(&SkillType::Independent);
    let map_size = state.settings.size;

    let new_unit = UnitState {
        owner,
        unit_type,
        health: settings.health * crate::states::HEALTH_SCALE,
        prev_coords: Coords::invalid(),
        direction: 0,
        flipped: false,
        veteran: false,
        kills: 0,
        created_turn: state.settings.turn,
        home_coords: if independent {
            None
        } else {
            Some(Coords::from_index(tile_idx, map_size))
        },
        coords: Coords::from_index(tile_idx, map_size),
        moved: true,
        attacked: true,
        effects: std::collections::HashSet::new(),
        passenger_type: None,
        converted: false,
        attacks_performed: 0,
        parent_unit_idx: None,
        child_unit_idx: None,
    };

    let old_unit_owner: Option<PlayerId> =
        state.tiles.get(&tile_idx).and_then(|t| t._unit_owner_id);
    let mut undos: Vec<UndoCallback> = Vec::new();

    // 1. Add to tribe units and update score
    if let Some(tribe) = state.tribes.get_mut(&owner) {
        tribe.units.push(new_unit);
        let score_gain = 5 * if settings.is_super { 10 } else { settings.cost };
        tribe.score += score_gain;

        undos.push(Box::new(move |s: &mut GameState| {
            if let Some(t) = s.tribes.get_mut(&owner) {
                t.score -= score_gain;
                t.units.pop();
            }
        }) as UndoCallback);
    }

    // 2. Set tile owner
    if let Some(tile) = state.tiles.get_mut(&tile_idx) {
        tile._unit_owner_id = Some(owner);
        undos.push(Box::new(move |s: &mut GameState| {
            if let Some(t) = s.tiles.get_mut(&tile_idx) {
                t._unit_owner_id = old_unit_owner;
            }
        }) as UndoCallback);
    }

    crate::actions::chain_undos(undos)
}

/// Summon a unit at a specific tile
pub fn summon_unit(
    state: &mut GameState,
    unit_type: UnitType,
    spawn_tile_idx: i32,
    costs: bool,
    force_independent: bool,
) -> Result<crate::moves::MoveResult, String> {
    use crate::actions::discovery::discover_tiles;
    use crate::actions::{freeze_area, spend_stars};
    use crate::moves::MoveResult;

    let pov_id = state.settings.current_player_turn_id;
    let settings = get_unit_setting(unit_type);

    // Push occupied unit away if any
    let push_result = push_unit(state, spawn_tile_idx)?;
    let mut undos = Vec::new();

    undos.push(push_result.undo);

    // Spend stars
    if costs {
        undos.push(spend_stars(state, settings.cost));
    }

    // Spawn unit
    undos.push(spawn_unit(
        state,
        pov_id,
        unit_type,
        spawn_tile_idx,
        force_independent,
    ));

    // Discover tiles around unit
    let unit_copy = state
        .tribes
        .get(&pov_id)
        .and_then(|t| t.units.last())
        .cloned();
    let discover_undo = discover_tiles(state, unit_copy.as_ref(), None);
    undos.push(discover_undo);

    // AutoFreeze
    if has_skill(unit_type, SkillType::AutoFreeze) || has_skill(unit_type, SkillType::FreezeArea) {
        undos.push(freeze_area(state, pov_id, spawn_tile_idx));
    }

    Ok(MoveResult {
        undo: crate::actions::chain_undos(undos),
        rewards: None,
    })
}

/// End a unit's turn (mark as moved/attacked)
pub fn end_unit_turn(state: &mut GameState, unit_owner: PlayerId, unit_idx: usize) -> UndoCallback {
    let (old_moved, old_attacked) = {
        let tribe = match state.tribes.get(&unit_owner) {
            Some(t) => t,
            None => return Box::new(|_| {}),
        };
        let unit = match tribe.units.get(unit_idx) {
            Some(u) => u,
            None => return Box::new(|_| {}),
        };
        (unit.moved, unit.attacked)
    };

    if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            unit.moved = true;
            unit.attacked = true;
        }
    }

    Box::new(move |s| {
        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.moved = old_moved;
                unit.attacked = old_attacked;
            }
        }
    })
}

/// Spawn a unit at a city (e.g. for rewards)
pub fn spawn_unit_at_city(
    state: &mut GameState,
    city_tile_idx: i32,
    unit_type: UnitType,
) -> UndoCallback {
    match summon_unit(state, unit_type, city_tile_idx, false, false) {
        Ok(result) => result.undo,
        Err(_) => Box::new(|_| {}),
    }
}

/// Deal raw damage to a unit (without an attacker unit)
pub fn deal_damage(
    state: &mut GameState,
    owner: PlayerId,
    unit_idx: usize,
    damage: i32,
    killer_owner: Option<PlayerId>,
) -> UndoCallback {
    let mut undos = Vec::new();

    if let Some(tribe) = state.tribes.get_mut(&owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            let old_hp = unit.health;
            unit.health -= damage;

            undos.push(Box::new(move |s: &mut GameState| {
                if let Some(t) = s.tribes.get_mut(&owner) {
                    if let Some(u) = t.units.get_mut(unit_idx) {
                        u.health = old_hp;
                    }
                }
            }) as UndoCallback);

            if unit.health <= 0 {
                undos.push(remove_unit(state, owner, unit_idx, killer_owner, None));
            }
        }
    }

    crate::actions::chain_undos(undos)
}

/// Infiltrate a city with a Cloak
pub fn infiltrate_city(
    state: &mut GameState,
    unit_owner: PlayerId,
    unit_idx: usize,
    target_city_idx: i32,
) -> UndoCallback {
    let mut undos: Vec<UndoCallback> = Vec::new();

    // 1. Consume the Cloak
    undos.push(remove_unit(state, unit_owner, unit_idx, None, None));

    // 2. Damage enemy in city
    if let Some(enemy_unit) = crate::functions::get_unit_at(state, target_city_idx) {
        if enemy_unit.owner != unit_owner {
            let enemy_owner = enemy_unit.owner;
            // Find unit index
            if let Some(tribe) = state.tribes.get(&enemy_owner) {
                if let Some(pos) = tribe
                    .units
                    .iter()
                    .position(|u| u.coords.idx == target_city_idx)
                {
                    // Apply 2 damage
                    if let Some(tribe_mut) = state.tribes.get_mut(&enemy_owner) {
                        if let Some(u) = tribe_mut.units.get_mut(pos) {
                            u.health -= 2;
                        }
                    }

                    let health_after = state.tribes[&enemy_owner].units[pos].health;

                    undos.push(Box::new(move |s| {
                        if let Some(t) = s.tribes.get_mut(&enemy_owner) {
                            if let Some(u) = t.units.get_mut(pos) {
                                u.health += 2;
                            }
                        }
                    }));

                    if health_after <= 0 {
                        // Kill unit
                        undos.push(remove_unit(state, enemy_owner, pos, Some(unit_owner), None));
                    }
                }
            }
        }
    }

    // 3. Identify spawn tiles
    let (mut def_tiles, mut water_tiles, mut other_tiles, city_income) = {
        let city_tile = state.tiles.get(&target_city_idx).unwrap();
        let city_owner_id = city_tile.owner;
        let city = state.tribes[&city_owner_id]
            .cities
            .iter()
            .find(|c| c.tile_index == target_city_idx)
            .unwrap();
        let income = std::cmp::min(5, crate::functions::get_city_production(state, city));

        let pov = &state.tribes[&unit_owner];
        let has_climbing = crate::settings::technology::has_technology(
            &pov.tech_vanilla,
            TechnologyType::Climbing,
        );
        let has_archery =
            crate::settings::technology::has_technology(&pov.tech_vanilla, TechnologyType::Archery);
        let has_sailing =
            crate::settings::technology::has_technology(&pov.tech_vanilla, TechnologyType::Sailing);

        let mut def = Vec::new();
        let mut wat = Vec::new();
        let mut oth = Vec::new();

        for &idx in &city._territory {
            if idx == target_city_idx {
                continue;
            }
            if crate::functions::get_unit_at(state, idx).is_some() {
                continue;
            }

            if let Some(tile) = state.tiles.get(&idx) {
                match tile.terrain_type {
                    TerrainType::Mountain => {
                        if has_climbing {
                            def.push(idx);
                        }
                    }
                    TerrainType::Forest => {
                        if has_archery {
                            def.push(idx);
                        }
                    }
                    TerrainType::Water | TerrainType::Ocean => {
                        let is_cymanti = pov.tribe_type == TribeType::Cymanti;
                        // Cymanti Explorer can now move on water like other tribes (requires Pascetism/Sailing equivalent)
                        let has_pascetism = crate::settings::technology::has_technology(
                            &pov.tech_vanilla,
                            TechnologyType::Pascetism,
                        );

                        if has_sailing || (is_cymanti && has_pascetism) {
                            wat.push(idx);
                        }
                    } // End match TerrainType::Water
                    _ => oth.push(idx),
                }
            }
        }
        (def, wat, oth, income)
    };

    // Prioritize city tile if empty
    if crate::functions::get_unit_at(state, target_city_idx).is_none() {
        other_tiles.insert(0, target_city_idx);
    }

    // 4. Spawn units
    for _ in 0..city_income {
        let (tile_idx, unit_type) = if let Some(idx) = def_tiles.pop().or_else(|| other_tiles.pop())
        {
            (idx, UnitType::Dagger)
        } else if let Some(idx) = water_tiles.pop() {
            (idx, UnitType::Pirate)
        } else {
            break;
        };

        if let Ok(res) = summon_unit(state, unit_type, tile_idx, false, false) {
            undos.push(res.undo);
            if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
                if let Some(unit) = tribe.units.last_mut() {
                    unit.moved = true;
                    unit.attacked = true;
                }
            }
        }
    }

    // 5. Gain stars
    undos.push(crate::actions::spend_stars(state, -city_income));

    // 6. Riot
    let city_owner = state.tiles[&target_city_idx].owner;
    if let Some(city) = state.tribes.get_mut(&city_owner).and_then(|t| {
        t.cities
            .iter_mut()
            .find(|c| c.tile_index == target_city_idx)
    }) {
        city._riot = true;
        undos.push(Box::new(move |s| {
            if let Some(city) = s.tribes.get_mut(&city_owner).and_then(|t| {
                t.cities
                    .iter_mut()
                    .find(|c| c.tile_index == target_city_idx)
            }) {
                city._riot = false;
            }
        }));
    }

    crate::actions::chain_undos(undos)
}

/// Poison a unit
pub fn poison_unit(state: &mut GameState, unit_owner: PlayerId, unit_idx: usize) -> UndoCallback {
    let old_has_poison = if let Some(tribe) = state.tribes.get(&unit_owner) {
        if let Some(unit) = tribe.units.get(unit_idx) {
            unit.effects.contains(&crate::types::EffectType::Poison)
        } else {
            return Box::new(|_| {});
        }
    } else {
        return Box::new(|_| {});
    };

    if old_has_poison {
        return Box::new(|_| {});
    }

    if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            unit.effects.insert(crate::types::EffectType::Poison);
        }
    }

    Box::new(move |s| {
        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.effects.remove(&crate::types::EffectType::Poison);
            }
        }
    })
}

// Boost a unit
pub fn boost_unit(state: &mut GameState, unit_idx: i32) -> UndoCallback {
    let mut undos: Vec<UndoCallback> = Vec::new();
    let unit_owner = if let Some(tile) = state.tiles.get(&unit_idx) {
        tile._unit_owner_id.unwrap_or(0)
    } else {
        0
    };

    if unit_owner == 0 {
        return Box::new(|_| {});
    }

    // Find adjacent friendly units
    let adj = crate::functions::get_adjacent_indices(state, unit_idx, 1);
    for adj_idx in adj {
        if let Some(_target) = crate::functions::get_unit_at(state, adj_idx) {
            if _target.owner == unit_owner {
                // Apply boost
                // Need to find unit index in tribe.
                let target_owner = _target.owner;
                if let Some(tribe) = state.tribes.get(&target_owner) {
                    if let Some(idx) = tribe.units.iter().position(|u| u.coords.idx == adj_idx) {
                        if !tribe.units[idx]
                            .effects
                            .contains(&crate::types::EffectType::Boost)
                        {
                            if let Some(tribe_mut) = state.tribes.get_mut(&target_owner) {
                                tribe_mut.units[idx]
                                    .effects
                                    .insert(crate::types::EffectType::Boost);
                            }
                            undos.push(Box::new(move |s| {
                                if let Some(t) = s.tribes.get_mut(&target_owner) {
                                    if let Some(u) = t.units.get_mut(idx) {
                                        u.effects.remove(&crate::types::EffectType::Boost);
                                    }
                                }
                            }));
                        }
                    }
                }
            }
        }
    }
    crate::actions::chain_undos(undos)
}

pub fn convert_unit(
    state: &mut GameState,
    converter_idx: i32,
    target_idx: i32,
) -> Result<UndoCallback, String> {
    // 1. Validate
    let converter_owner = state
        .tiles
        .get(&converter_idx)
        .and_then(|t| t._unit_owner_id)
        .ok_or("No converter")?;
    let target_owner = state
        .tiles
        .get(&target_idx)
        .and_then(|t| t._unit_owner_id)
        .ok_or("No target")?;

    // 2. Perform conversion (change owner)
    // Find unit in old tribe
    let (mut unit, _old_idx) = {
        let tribe = state
            .tribes
            .get_mut(&target_owner)
            .ok_or("Target tribe not found")?;
        let idx = tribe
            .units
            .iter()
            .position(|u| u.coords.idx == target_idx)
            .ok_or("Target unit not found")?;
        let unit = tribe.units.remove(idx); // Take it out
        (unit, idx)
    };

    // Capture ORIGINAL state for undo
    let original_unit = unit.clone();

    // Insert into new tribe
    let new_idx = {
        let tribe = state
            .tribes
            .get_mut(&converter_owner)
            .ok_or("Converter tribe not found")?;
        // Update unit owner field
        unit.owner = converter_owner;
        unit.converted = true; // Mark as converted
        unit.attacked = true;
        unit.moved = true;
        unit.effects.clear(); // Converted units lose effects

        tribe.units.push(unit); // Move the modified unit
        tribe.units.len() - 1
    };

    // Update tile owner
    if let Some(tile) = state.tiles.get_mut(&target_idx) {
        tile._unit_owner_id = Some(converter_owner);
    }

    // Undo
    Ok(Box::new(move |s| {
        // Restore to old tribe
        // Remove from new tribe
        if let Some(tribe) = s.tribes.get_mut(&converter_owner) {
            if new_idx < tribe.units.len() {
                tribe.units.remove(new_idx);
            }
        }
        // Add back to old tribe
        if let Some(tribe) = s.tribes.get_mut(&target_owner) {
            // Restore exact original state
            tribe.units.push(original_unit.clone());
        }
        // Restore tile
        if let Some(tile) = s.tiles.get_mut(&target_idx) {
            tile._unit_owner_id = Some(target_owner);
        }
    }))
}

/// Disband a unit and partial refund
pub fn disband_unit(
    state: &mut GameState,
    unit_owner: i32,
    unit_idx: usize,
) -> Result<UndoCallback, String> {
    use crate::actions::{chain_undos, gain_stars};
    use crate::settings::get_unit_setting;

    let unit_type = if let Some(tribe) = state.tribes.get(&unit_owner) {
        if let Some(u) = tribe.units.get(unit_idx) {
            u.unit_type
        } else {
            return Err("Unit not found".to_string());
        }
    } else {
        return Err("Tribe not found".to_string());
    };

    let settings = get_unit_setting(unit_type);
    let refund = (settings.cost as f32 * 0.5).floor() as i32;

    let mut undos = Vec::new();
    if refund > 0 {
        undos.push(gain_stars(state, refund));
    }
    undos.push(remove_unit(state, unit_owner, unit_idx, None, None));

    Ok(chain_undos(undos))
}

/// Upgrade a unit (e.g. Boat -> Ship)
pub fn upgrade_unit(
    state: &mut GameState,
    tile_idx: i32,
    target_type: crate::types::UnitType,
) -> Result<UndoCallback, String> {
    use crate::actions::spend_stars;
    use crate::settings::get_unit_setting;

    let settings = get_unit_setting(target_type);
    let mut undos: Vec<UndoCallback> = Vec::new();

    // 1. Spend stars
    if settings.cost > 0 {
        undos.push(spend_stars(state, settings.cost));
    }

    // 2. Find and update unit
    let unit_owner = state
        .tiles
        .get(&tile_idx)
        .and_then(|t| t._unit_owner_id)
        .ok_or("No unit at tile")?;

    if unit_owner == 0 {
        return Err("No unit at tile".to_string());
    }

    let unit_idx = if let Some(tribe) = state.tribes.get(&unit_owner) {
        tribe
            .units
            .iter()
            .position(|u| u.coords.idx == tile_idx)
            .ok_or("Unit not found in tribe")?
    } else {
        return Err("Tribe not found".to_string());
    };

    let old_type = if let Some(tribe) = state.tribes.get(&unit_owner) {
        tribe.units[unit_idx].unit_type
    } else {
        return Err("Tribe not found".to_string());
    };

    // Update unit
    if let Some(tribe) = state.tribes.get_mut(&unit_owner) {
        if let Some(unit) = tribe.units.get_mut(unit_idx) {
            unit.unit_type = target_type;
        }
    }

    // Undo unit change
    undos.push(Box::new(move |s| {
        if let Some(tribe) = s.tribes.get_mut(&unit_owner) {
            if let Some(unit) = tribe.units.get_mut(unit_idx) {
                unit.unit_type = old_type;
            }
        }
    }));

    Ok(crate::actions::chain_undos(undos))
}
