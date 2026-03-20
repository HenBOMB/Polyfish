use crate::functions::{self, get_adjacent_indices, get_unit_max_health, has_effect};
use crate::moves::Move;
use crate::moves::{DisbandMove, PromoteMove, RecoverMove};
use crate::settings::technology;
use crate::states::{GameState, TribeState, UnitState};
use crate::types::{SkillType, StructureType, TechnologyType, UnitEffect};

use super::*;

pub fn generate_unit_action_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    for unit in &tribe.units {
        generate_unit_action_moves_for_unit(state, tribe, unit, moves);
    }
}

pub fn generate_unit_action_moves_for_unit(
    state: &GameState,
    tribe: &TribeState,
    unit: &UnitState,
    moves: &mut Vec<Box<dyn Move>>,
) {
    let pov_id = state.settings.current_player_turn_id;
    let idx = unit.coords.idx;

    // Promote
    if !unit.veteran && unit.kills >= 3 && !functions::has_skill(unit, SkillType::Static) {
        moves.push(Box::new(PromoteMove::new(idx)));
    }

    // Explode
    // new change: Explode units can explode even after attacking
    if functions::has_skill(unit, SkillType::Explode) {
        let enemies_around = get_adjacent_indices(state, idx, 1)
            .iter()
            .any(|&adj| functions::get_enemy_at(state, adj, pov_id).is_some());
        if enemies_around {
            moves.push(Box::new(ExplodeMove::new(idx)));
        }
    }

    if unit.moved || unit.attacked {
        return;
    }

    // Disband
    if technology::has_technology(&tribe.tech_vanilla, TechnologyType::FreeSpirit) {
        moves.push(Box::new(DisbandMove::new(idx)));
    }

    // Recover
    if unit.health < get_unit_max_health(unit) {
        moves.push(Box::new(RecoverMove::new(idx)));
    }

    // Ability Skills
    if functions::has_skill(unit, SkillType::HealOthers) {
        let damaged_around = get_adjacent_indices(state, idx, 1).iter().any(|&adj| {
            if let Some(target) = functions::get_unit_at(state, adj) {
                return target.owner == pov_id && target.health < get_unit_max_health(target);
            }
            false
        });
        if damaged_around {
            moves.push(Box::new(HealOthersMove::new(idx)));
        }
    } else if functions::has_skill(unit, SkillType::Swarm) {
        let unboosted_around = get_adjacent_indices(state, idx, 1).iter().any(|&adj| {
            if let Some(target) = functions::get_unit_at(state, adj) {
                return target.owner == pov_id && !has_effect(target, UnitEffect::Boosted);
            }
            false
        });
        if unboosted_around {
            moves.push(Box::new(BoostMove::new(idx)));
        }
    } else if functions::has_skill(unit, SkillType::FreezeArea) {
        let stuff_around = get_adjacent_indices(state, idx, 1).iter().any(|&adj| {
            if let Some(enemy) = functions::get_enemy_at(state, adj, pov_id) {
                if !has_effect(enemy, UnitEffect::Frozen) || !functions::is_tile_frozen(state, adj)
                {
                    return true;
                }
            }
            false
        });
        if stuff_around {
            moves.push(Box::new(FreezeAreaMove::new(idx)));
        }
    }
}

pub fn generate_economic_ability_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    let has_destroy = technology::has_technology(&tribe.tech_vanilla, TechnologyType::FreeSpirit);
    let has_decompose =
        technology::has_technology(&tribe.tech_vanilla, TechnologyType::Construction); // Recycling replaces Construction
    let is_cymanti = tribe.tribe_type == crate::types::TribeType::Cymanti;

    if has_destroy || (is_cymanti && has_decompose) {
        for city in &tribe.cities {
            for &tile_idx in &city._territory {
                // Check enemy
                if functions::get_enemy_at(state, tile_idx, pov_id).is_some() {
                    continue;
                }

                if let Some(st_type) = functions::get_structure_type_at(state, tile_idx) {
                    if st_type != StructureType::Village && st_type != StructureType::Ruin {
                        if is_cymanti && has_decompose {
                            moves.push(Box::new(DecomposeMove::new(tile_idx)));
                        } else if has_destroy {
                            moves.push(Box::new(DestroyMove::new(tile_idx)));
                        }
                    }
                }
            }
        }
    }

    // Enchant Animal (Elyrion)
    // Needs ForestMagic tech
    if technology::has_technology(&tribe.tech_vanilla, TechnologyType::ForestMagic) {
        for city in &tribe.cities {
            for &tile_idx in &city._territory {
                // Check enemy
                if functions::get_enemy_at(state, tile_idx, pov_id).is_some() {
                    continue;
                }

                // Check for Game/Animal resource
                if let Some(Some(resource)) = state.resources.get(&tile_idx).as_ref() {
                    if resource.resource_type == crate::types::ResourceType::Game {
                        if tribe.stars >= 3 {
                            moves.push(Box::new(EnchantAnimalMove::new(tile_idx)));
                        }
                    }
                }
            }
        }
    }
}
