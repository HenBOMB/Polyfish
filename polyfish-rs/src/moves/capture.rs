//! Capture move implementation

use crate::actions::city::claim_territory;
use crate::actions::discovery::discover_tiles;
use crate::actions::resource::consume_resource;
use crate::actions::structure::{create_structure, destroy_structure};
use crate::actions::tech::unlock_tech;
use crate::actions::units::remove_unit;
use crate::actions::{chain_undos, end_unit_turn, gain_stars, spend_stars, UndoCallback};
use crate::functions::{
    get_adjacent_indices, get_capital_city, get_pov_tribe, get_structure_at, get_unit_at,
};
use crate::moves::{Move, MoveResult};
use crate::states::{CityState, GameState};
use crate::types::{MoveType, RewardType, StructureType, TechnologyType, TerrainType, TribeType};
use rand::Rng;

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

    fn execute(&self, state: &mut GameState) -> MoveResult {
        let pov_id = state.settings.current_player_turn_id;
        let capturer_idx = if let Some(unit) = get_unit_at(state, self.src) {
            state
                .tribes
                .get(&unit.owner)
                .and_then(|t| t.units.iter().position(|u| u.coords.idx == self.src))
                .unwrap() // Simplified
        } else {
            return MoveResult {
                undo: Box::new(|_| {}),
                rewards: None,
            };
        };

        let unit_owner = get_unit_at(state, self.src).unwrap().owner; // Should be pov_id usually

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
                // Capture village or city
                let tile_owner = state.tiles.get(&self.src).map(|t| t.owner).unwrap_or(0);

                if tile_owner > 0 && tile_owner != pov_id {
                    // Capture enemy City
                    let mut old_city: Option<CityState> = None;
                    let mut old_city_idx: Option<usize> = None;

                    if let Some(old_tribe) = state.tribes.get_mut(&tile_owner) {
                        if let Some(pos) = old_tribe
                            .cities
                            .iter()
                            .position(|c| c.tile_index == self.src)
                        {
                            old_city_idx = Some(pos);
                            old_city = Some(old_tribe.cities.remove(pos));
                        }
                    }

                    if let Some(mut city) = old_city {
                        let city_name_old = city.name.clone();
                        let old_capital_val = state
                            .tiles
                            .get(&self.src)
                            .map(|t| t.capital_of)
                            .unwrap_or(0);

                        // 1. Update city
                        city.owner = pov_id;
                        let tribe_type = state
                            .tribes
                            .get(&pov_id)
                            .map(|t| t.tribe_type)
                            .unwrap_or(TribeType::None);
                        city.name = format!(
                            "{:?} {}",
                            tribe_type,
                            if old_capital_val > 0 {
                                "Capital"
                            } else {
                                "City"
                            }
                        );

                        // 2. Update tiles
                        if let Some(tile) = state.tiles.get_mut(&self.src) {
                            tile.owner = pov_id;
                            if tile.capital_of > 0 {
                                tile.capital_of = pov_id;
                            }
                        }

                        // 3. Add to new owner
                        if let Some(new_tribe) = state.tribes.get_mut(&pov_id) {
                            new_tribe.cities.push(city.clone());
                        }

                        // 4. Tribe elimination check
                        let is_eliminated = state
                            .tribes
                            .get(&tile_owner)
                            .map(|t| t.cities.is_empty())
                            .unwrap_or(false);
                        if is_eliminated {
                            if let Some(old_tribe) = state.tribes.get_mut(&tile_owner) {
                                old_tribe.killed_turn = state.settings.turn;
                                old_tribe.killer_id = pov_id;

                                // Remove all units
                                let unit_count = old_tribe.units.len();
                                for i in (0..unit_count).rev() {
                                    undos.push(remove_unit(
                                        state,
                                        tile_owner,
                                        i,
                                        Some(pov_id),
                                        None,
                                    ));
                                }
                            }
                        }

                        // Undo logic
                        let c_clone = city.clone();
                        undos.push(Box::new(move |s| {
                            // Restore tribe status
                            if let Some(ot) = s.tribes.get_mut(&tile_owner) {
                                if ot.killer_id == pov_id && ot.killed_turn == s.settings.turn {
                                    ot.killer_id = 0;
                                    ot.killed_turn = 0;
                                }
                            }
                            // Remove from new
                            if let Some(nt) = s.tribes.get_mut(&pov_id) {
                                if let Some(p) = nt
                                    .cities
                                    .iter()
                                    .position(|c| c.tile_index == c_clone.tile_index)
                                {
                                    nt.cities.remove(p);
                                }
                            }
                            // Restore tiles
                            if let Some(tile) = s.tiles.get_mut(&c_clone.tile_index) {
                                tile.owner = tile_owner;
                                if old_capital_val > 0 {
                                    tile.capital_of = tile_owner;
                                }
                            }
                            // Add back to old
                            if let Some(ot) = s.tribes.get_mut(&tile_owner) {
                                let mut restored = c_clone.clone();
                                restored.owner = tile_owner;
                                restored.name = city_name_old;
                                if let Some(pos) = old_city_idx {
                                    ot.cities.insert(pos, restored);
                                } else {
                                    ot.cities.push(restored);
                                }
                            }
                        }));

                        // Claim territory updates owners of surrounding tiles
                        undos.push(claim_territory(state, &city._territory, self.src, false));
                    }
                } else {
                    // Capture Village (new city)
                    let territory = get_adjacent_indices(state, self.src, 1);
                    let tribe_type = state
                        .tribes
                        .get(&pov_id)
                        .map(|t| t.tribe_type)
                        .unwrap_or(TribeType::None);

                    let created_city = CityState {
                        name: format!("{:?} City", tribe_type),
                        population: 0,
                        progress: 0,
                        border_size: 1,
                        connected_to_capital: false,
                        level: 1,
                        production: 1,
                        owner: pov_id,
                        tile_index: self.src,
                        rewards: std::collections::HashSet::new(),
                        _territory: territory.clone(),
                        _riot: false,
                        _walls: false,
                    };

                    if let Some(tribe) = state.tribes.get_mut(&pov_id) {
                        tribe.cities.push(created_city.clone());
                    }

                    undos.push(Box::new(move |s| {
                        if let Some(tribe) = s.tribes.get_mut(&pov_id) {
                            tribe.cities.pop();
                        }
                    }));

                    undos.push(claim_territory(state, &territory, self.src, false));
                }
            }
            Some(StructureType::Ruin) => {
                // Capture Ruins
                // Destroy ruins
                undos.push(destroy_structure(state, self.src));

                let mut possible_rewards: Vec<Box<dyn FnOnce(&mut GameState) -> UndoCallback>> =
                    Vec::new();

                // 1. Stars: 5 stars
                possible_rewards.push(Box::new(|s: &mut GameState| gain_stars(s, 5)));

                // 2. Tech: random unlockable
                // Collect all unlockable techs first (tier > 0)
                let tribe = state.tribes.get(&pov_id).unwrap().clone();
                let mut unlockable_cand = Vec::new();
                for t_val in 1..=24 {
                    // Standard tech tree range
                    if t_val == 11 {
                        continue;
                    }
                    let t_type: TechnologyType = unsafe { std::mem::transmute(t_val as i8) };
                    if !crate::settings::technology::has_technology(&tribe.tech_vanilla, t_type) {
                        unlockable_cand.push(t_type);
                    }
                }

                if !unlockable_cand.is_empty() {
                    possible_rewards.push(Box::new(move |s: &mut GameState| {
                        let mut rng = rand::thread_rng();
                        let picked = unlockable_cand[rng.gen_range(0..unlockable_cand.len())];
                        unlock_tech(s, picked, true)
                    }));
                }

                // 3. Pop growth: 3 to capital
                if let Some(cap) = get_capital_city(state, pov_id) {
                    let cap_tile_idx = cap.tile_index;
                    possible_rewards.push(Box::new(move |s: &mut GameState| {
                        crate::actions::city::add_population(s, cap_tile_idx, 3)
                    }));
                }

                // 4. Explorer: if nearby is fog
                let mut fog_nearby = false;
                let around = get_adjacent_indices(state, self.src, 2);
                for &idx in &around {
                    if !state._visible_tiles.contains_key(&idx) {
                        fog_nearby = true;
                        break;
                    }
                }
                if fog_nearby {
                    let src_idx = self.src;
                    possible_rewards.push(Box::new(move |s: &mut GameState| {
                        let revealed = crate::actions::discovery::predict_explorer(s, src_idx);
                        discover_tiles(s, None, Some(revealed))
                    }));
                }

                /*
                // 5. Veteran Swordsman or Rammer (if on ocean)
                // (This is disabled in TS via possibleRewards.pop() but kept here for parity)
                let is_ocean = state.tiles.get(&self.src).map(|t| t.terrain_type == crate::types::TerrainType::Ocean).unwrap_or(false);
                let src_idx = self.src;
                possible_rewards.push(Box::new(move |s: &mut GameState| {
                    let u_type = if is_ocean { crate::types::UnitType::Rammer } else { crate::types::UnitType::Swordsman };
                    let summon_undo = match crate::actions::units::summon_unit(s, u_type, src_idx, false, true) {
                        Ok(res) => res.undo,
                        Err(_) => Box::new(|_| {}),
                    };

                    // Set veteran status
                    if let Some(tribe) = s.tribes.get_mut(&pov_id) {
                        if let Some(u) = tribe.units.last_mut() {
                            u.veteran = true;
                            u.kills = 3;
                        }
                    }
                    summon_undo
                }));
                */

                // Pick one
                if !possible_rewards.is_empty() {
                    let mut rng = rand::thread_rng();
                    let reward_fn =
                        possible_rewards.remove(rng.gen_range(0..possible_rewards.len()));
                    undos.push(reward_fn(state));
                }
            }
            _ => {
                // Starfish is handled if no structure
                // TS logic: consume resource + gain 8 stars
                undos.push(consume_resource(state, self.src, None));
                undos.push(gain_stars(state, 8));
            }
        }

        MoveResult {
            undo: chain_undos(undos),
            rewards: None,
        }
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
