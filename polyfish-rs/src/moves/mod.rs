//! Move generation and move types
//!
//! This module contains the Move trait and all move type implementations.

pub mod abilities;
pub mod attack;
pub mod build;
pub mod capture;
pub mod disband;
pub mod harvest;
pub mod recover;
pub mod research;
pub mod reward;
pub mod step;
pub mod summon;
pub mod upgrade;

pub use abilities::*;
pub use attack::AttackMove;
pub use build::BuildMove;
pub use capture::CaptureMove;
pub use disband::DisbandMove;
pub use harvest::HarvestMove;
pub use recover::RecoverMove;
pub use research::ResearchMove;
pub use reward::RewardMove;
pub use step::StepMove;
pub use summon::SummonMove;
pub use upgrade::UpgradeMove;

use crate::actions::UndoCallback;
use crate::functions::{
    get_adjacent_indices, get_enemy_at, get_structure_at, get_structure_type_at, get_unit_at,
};
use crate::settings::resources::get_resource_setting;
use crate::settings::{get_unit_setting, has_skill};
use crate::states::*;
use crate::types::*;

/// Result of executing a move
pub struct MoveResult {
    /// Undo callback to reverse the move
    pub undo: UndoCallback,
    /// Optional reward moves generated
    pub rewards: Option<Vec<Box<dyn Move>>>,
}

/// A game move that can be executed and undone
pub trait Move: std::fmt::Debug + Send + Sync {
    /// Get the move type
    fn move_type(&self) -> MoveType;

    /// Execute the move on the game state
    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String>;

    /// Human-readable description
    fn describe(&self, state: &GameState) -> String;

    /// Serialize to JSON for network/storage
    fn serialize(&self) -> serde_json::Value;

    // === Policy Composition Helpers ===
    // These methods extract components needed for decomposed policy heads

    /// Get source tile index (for moves with 'from'concept)
    #[inline]
    fn source_idx(&self) -> Result<usize, String> {
        Err(format!("Move {:?} requires source index", self.move_type()))
    }

    /// Get target tile index (for moves with 'to' concept)
    #[inline]
    fn target_idx(&self) -> Result<usize, String> {
        Err(format!("Move {:?} requires target index", self.move_type()))
    }

    /// Get structure type (for Build moves)
    #[inline]
    fn structure_type(&self) -> Result<StructureType, String> {
        Err(format!(
            "Move {:?} requires structure type",
            self.move_type()
        ))
    }

    /// Get unit type (for Summon moves)
    #[inline]
    fn unit_type(&self) -> Result<UnitType, String> {
        Err(format!("Move {:?} requires unit type", self.move_type()))
    }

    /// Get tech type (for Research moves)
    #[inline]
    fn tech_type(&self) -> Result<TechnologyType, String> {
        Err(format!("Move {:?} requires tech type", self.move_type()))
    }

    /// Get ability type (for Ability moves)
    #[inline]
    fn ability_type(&self) -> Result<AbilityType, String> {
        Err(format!("Move {:?} requires ability type", self.move_type()))
    }

    /// Get reward type (for Reward moves)
    #[inline]
    fn reward_type(&self) -> Result<CityRewardType, String> {
        Err(format!("Move {:?} requires reward type", self.move_type()))
    }
}

/// End turn move
#[derive(Debug, Clone)]
pub struct EndTurnMove;

impl Move for EndTurnMove {
    fn move_type(&self) -> MoveType {
        MoveType::EndTurn
    }

    fn execute(&self, _state: &mut GameState) -> Result<MoveResult, String> {
        // EndTurn is handled specially in Game::play_move
        Ok(MoveResult {
            undo: Box::new(|_| {}),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        "End Turn".to_string()
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({ "moveType": MoveType::EndTurn })
    }
}

/// Resign move
#[derive(Debug, Clone)]
pub struct ResignMove;

impl Move for ResignMove {
    fn move_type(&self) -> MoveType {
        MoveType::Resign
    }

    fn execute(&self, state: &mut GameState) -> Result<MoveResult, String> {
        let pov_id = state.settings.current_player_turn_id;
        let turn = state.settings.turn;
        let mut undos = Vec::new();

        let old_resigned = state
            .tribes
            .get(&pov_id)
            .map(|t| t.resigned_turn)
            .unwrap_or(0);

        if let Some(tribe) = state.tribes.get_mut(&pov_id) {
            tribe.resigned_turn = turn;
        }

        // Remove all units - done after dropping the mutable tribe borrow to avoid conflict
        let unit_count = state
            .tribes
            .get(&pov_id)
            .map(|t| t.units.len())
            .unwrap_or(0);
        for i in (0..unit_count).rev() {
            undos.push(crate::actions::units::remove_unit(
                state, pov_id, i, None, None,
            ));
        }

        undos.push(Box::new(move |s| {
            if let Some(t) = s.tribes.get_mut(&pov_id) {
                t.resigned_turn = old_resigned;
            }
        }));

        Ok(MoveResult {
            undo: crate::actions::chain_undos(undos),
            rewards: None,
        })
    }

    fn describe(&self, _state: &GameState) -> String {
        "Resign".to_string()
    }

    fn serialize(&self) -> serde_json::Value {
        serde_json::json!({ "moveType": MoveType::Resign })
    }
}

/// Generate all legal moves for the current player
pub fn generate_legal_moves(state: &GameState) -> Vec<Box<dyn Move>> {
    let mut moves: Vec<Box<dyn Move>> = Vec::new();

    // let pov_id = state.settings.current_player_turn_id;
    // if let Some(tribe) = state.tribes.get(&pov_id) {
    //     println!("🐛 Player {} has {} stars", pov_id, tribe.stars);
    // }
    crate::moves::reward::generate_reward_moves(state, &mut moves);
    if !moves.is_empty() {
        return moves;
    }

    // EndTurn is always available
    moves.push(Box::new(EndTurnMove));

    // 2. Army Moves (Units, Summons, Upgrades)
    generate_unit_moves(state, &mut moves);
    crate::moves::summon::generate_summon_moves(state, &mut moves);
    crate::moves::upgrade::generate_upgrade_moves(state, &mut moves);
    crate::moves::abilities::unit_actions::generate_unit_action_moves(state, &mut moves);

    // 3. Econ Moves (Research, Build, Harvest, Econ Abilities)
    generate_econ_moves(state, &mut moves);
    crate::moves::abilities::unit_actions::generate_economic_ability_moves(state, &mut moves);
    crate::moves::abilities::diplomacy::generate_diplomacy_moves(state, &mut moves);

    moves
}

fn generate_unit_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;

    // Get current player's tribe
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    for unit in &tribe.units {
        // PER UNIT ORDER: captures, actions, attacks, steps

        // 1. Capture
        if !unit.moved && !unit.attacked {
            generate_capture_moves(state, unit, moves);
        }

        // 2. Actions (Promote, Disband, Recover, Ability skills)
        crate::moves::abilities::unit_actions::generate_unit_action_moves_for_unit(
            state, tribe, unit, moves,
        );

        // 3. Attack (Requires positive health)
        if !unit.attacked && unit.health > 0.0 {
            generate_attack_moves(state, unit, unit.coords.idx, moves);
        }

        // 4. Steps (Escape units will have moved=false after attacking, so this covers them)
        if !unit.moved {
            generate_step_moves(state, unit, moves);
        }
    }
}

/// Generate capture moves
fn generate_capture_moves(state: &GameState, unit: &UnitState, moves: &mut Vec<Box<dyn Move>>) {
    let idx = unit.coords.idx;
    if let Some(structure) = get_structure_at(state, idx) {
        match structure.structure_type {
            StructureType::Village => {
                let tile = state.tiles.get(&idx);
                let tile_owner = tile.map(|t| t.owner);
                // Village is capturable if it has no owner or is owned by an enemy
                if tile_owner.is_none() || tile_owner != Some(unit.owner) {
                    moves.push(Box::new(CaptureMove::new(idx)));
                }
            }
            StructureType::Ruin => {
                moves.push(Box::new(CaptureMove::new(idx)));
            }
            _ => {}
        }
    }

    // Check Starfish (resource)
    if let Some(Some(resource)) = state.resources.get(&idx).as_ref() {
        if resource.resource_type == ResourceType::Starfish {
            let tribe = &state.tribes[&unit.owner];
            if crate::settings::technology::has_technology(
                &tribe.tech_vanilla,
                TechnologyType::Navigation,
            ) {
                moves.push(Box::new(CaptureMove::new(idx)));
            }
        }
    }
}

/// Generate economy moves
fn generate_econ_moves(state: &GameState, moves: &mut Vec<Box<dyn Move>>) {
    let pov_id = state.settings.current_player_turn_id;

    // Get current player's tribe
    let tribe = match state.tribes.get(&pov_id) {
        Some(t) => t,
        None => return,
    };

    // Iterate cities for territory-based econ moves
    for city in &tribe.cities {
        for &tile_idx in &city._territory {
            // Check for enemy unit
            if get_enemy_at(state, tile_idx, pov_id).is_some() {
                continue;
            }

            // Check for harvesting
            if let Some(Some(resource)) = state.resources.get(&tile_idx).as_ref() {
                // Must not have a structure
                if get_structure_at(state, tile_idx).is_some() {
                    continue;
                }

                let settings = get_resource_setting(resource.resource_type);
                let tech_ok = match settings.tech_required {
                    TechnologyType::Basic => true,
                    tech => crate::settings::technology::has_technology(&tribe.tech_vanilla, tech),
                };

                // Cannot harvest if structure is required or if it requires capture
                if settings.struct_required.is_none()
                    && !settings.requires_capture
                    && tech_ok
                    && tribe.stars >= settings.cost.unwrap_or(0)
                {
                    moves.push(Box::new(HarvestMove::new(tile_idx)));
                }
            }
        }
    }

    // Generate research moves
    crate::moves::research::generate_research_moves(state, moves);

    // Generate build moves
    crate::moves::build::generate_build_moves(state, moves);

    // Generate forest moves (Clear/Grow/Burn)
    crate::moves::abilities::forest::generate_forest_moves(state, moves);
}

#[derive(Clone, Copy, PartialEq)]
struct ReachableNode {
    index: i32,
    cost: f32,
    terminal: bool,
}

impl Eq for ReachableNode {}

impl std::cmp::Ord for ReachableNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::cmp::PartialOrd for ReachableNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Generate step moves for a unit using pathfinding
fn generate_step_moves(state: &GameState, unit: &UnitState, moves: &mut Vec<Box<dyn Move>>) {
    if unit.moved {
        return;
    }

    let reachable = compute_reachable_tiles(state, unit);

    for (&tile_index, _) in &reachable {
        if unit.coords.idx == tile_index {
            continue;
        }

        // Cannot end turn on a tile occupied by another unit (own or enemy)
        // Friendly units can be passed through but not occupied.
        if crate::functions::get_true_unit_at(state, tile_index).is_some() {
            continue;
        }

        moves.push(Box::new(StepMove::new(unit.coords.idx, tile_index)));
    }
}

fn compute_reachable_tiles(
    state: &GameState,
    unit: &UnitState,
) -> std::collections::HashMap<i32, f32> {
    let mut effective_movement = crate::functions::get_unit_movement(state, unit) as f32;
    // Cap movement at 1 if unit has segments attached
    if unit.child_unit_idx.is_some() {
        effective_movement = 1.0;
    }
    let mut reachable = std::collections::HashMap::new();
    let mut open_list = std::collections::BinaryHeap::new();

    open_list.push(ReachableNode {
        index: unit.coords.idx,
        cost: 0.0,
        terminal: false,
    });
    reachable.insert(unit.coords.idx, 0.0);

    while let Some(current) = open_list.pop() {
        // If current path already exists and is better, skip
        if let Some(&best_cost) = reachable.get(&current.index) {
            if current.cost > best_cost {
                continue;
            }
        }

        if current.terminal {
            continue;
        }

        // If we have exhausted movement, we cannot expand further.
        // But we allow the node itself to be reachable (moved into).
        // The "Round Up" rule allows moving INTO a tile as long as we had > 0 movement remaining.
        // So checking >= effective_movement means we have 0 remaining capacity to START a move.
        // Exception: if cost is exactly 0.0 (unlikely for steps), infinite loop?
        // Game costs are > 0.
        if current.cost >= effective_movement {
            continue;
        }

        for n_idx in get_adjacent_indices(state, current.index, 1) {
            if n_idx == unit.coords.idx {
                continue;
            }

            if !is_steppable(state, unit, n_idx, false) {
                continue;
            }

            let move_cost = compute_movement_cost(state, unit, current.index, n_idx);
            if move_cost < 0.0 {
                continue;
            }

            let new_cost = current.cost + move_cost;

            // "Rounding up" rule: We allow the move even if new_cost > effective_movement,
            // provided current.cost < effective_movement (checked above).

            let terminal = is_terminal(state, unit, n_idx);

            let existing_cost = reachable.get(&n_idx);
            if existing_cost.is_none() || new_cost < *existing_cost.unwrap() {
                reachable.insert(n_idx, new_cost);
                open_list.push(ReachableNode {
                    index: n_idx,
                    cost: new_cost,
                    terminal,
                });
            }
        }
    }

    reachable
}

pub fn compute_shortest_path(
    state: &GameState,
    unit: &UnitState,
    target_idx: i32,
) -> Option<Vec<i32>> {
    let mut effective_movement = crate::functions::get_unit_movement(state, unit) as f32;
    // Cap movement at 1 if unit has segments attached
    if unit.child_unit_idx.is_some() {
        effective_movement = 1.0;
    }
    let mut reachable = std::collections::HashMap::new();
    let mut parent = std::collections::HashMap::new();
    let mut open_list = std::collections::BinaryHeap::new();

    open_list.push(ReachableNode {
        index: unit.coords.idx,
        cost: 0.0,
        terminal: false,
    });
    reachable.insert(unit.coords.idx, 0.0);

    while let Some(current) = open_list.pop() {
        if current.index == target_idx {
            let mut path = Vec::new();
            let mut curr = target_idx;
            while curr != unit.coords.idx {
                path.push(curr);
                // In case parent is missing due to disconnected components (shouldn't happen here)
                if let Some(&p) = parent.get(&curr) {
                    curr = p;
                } else {
                    break;
                }
            }
            path.reverse();
            return Some(path);
        }

        // If current path already exists and is better, skip
        if let Some(&best_cost) = reachable.get(&current.index) {
            if current.cost > best_cost {
                continue;
            }
        }

        if current.terminal {
            continue;
        }

        if current.cost >= effective_movement {
            continue;
        }

        for n_idx in get_adjacent_indices(state, current.index, 1) {
            if n_idx == unit.coords.idx {
                continue;
            }

            if !is_steppable(state, unit, n_idx, true) {
                continue;
            }

            let move_cost = compute_movement_cost(state, unit, current.index, n_idx);
            if move_cost < 0.0 {
                continue;
            }

            let new_cost = current.cost + move_cost;

            let terminal = is_terminal(state, unit, n_idx);

            let existing_cost = reachable.get(&n_idx);
            if existing_cost.is_none() || new_cost < *existing_cost.unwrap() {
                reachable.insert(n_idx, new_cost);
                parent.insert(n_idx, current.index);
                open_list.push(ReachableNode {
                    index: n_idx,
                    cost: new_cost,
                    terminal,
                });
            }
        }
    }

    None
}

fn compute_movement_cost(state: &GameState, unit: &UnitState, from_idx: i32, to_idx: i32) -> f32 {
    let cost: f32 = 1.0;

    let settings = get_unit_setting(unit.unit_type);

    // Flying and creeping disable road bonuses
    if settings.skills.contains(&SkillType::Fly) || settings.skills.contains(&SkillType::Creep) {
        return cost;
    }

    // Road bonus
    if is_roadpath_and_usable(state, unit, from_idx) && is_roadpath_and_usable(state, unit, to_idx)
    {
        return 0.5;
    }

    // Terrain specific costs
    if let Some(tile) = state.tiles.get(&to_idx) {
        // Glide/Skate doubles movement on frozen tiles (halves cost)
        if tile.is_frozen() {
            if settings.skills.contains(&SkillType::Skate) {
                // Skate uses standard cost (1.0), but gets +1 movement range (handled in get_unit_movement)
            }

            // Polarism bonus: "gain an extra move when entering ice for the first time"
            // We implement this as reduced cost (0.5) for the first step onto ice if the unit doesn't have Skate.
            if unit.coords.idx == from_idx {
                if let Some(tribe) = state.tribes.get(&unit.owner) {
                    if crate::settings::technology::has_technology(
                        &tribe.tech_vanilla,
                        TechnologyType::Polarism,
                    ) {
                        return 0.5;
                    }
                }
            }
        }

        // Amphibious units moving on land are slowed (Cost = 2.0)
        // "slows movement on land". Land is any non-water, non-flooded tile.
        if crate::functions::is_amphibious(unit) {
            if !tile.is_water_terrain() {
                return 2.0;
            }
        }
    }

    cost
}

fn is_terminal(state: &GameState, unit: &UnitState, tile_idx: i32) -> bool {
    let settings = get_unit_setting(unit.unit_type);
    if settings.skills.contains(&SkillType::Fly) {
        return false;
    }

    let tile = state.tiles.get(&tile_idx).unwrap();

    // Skate units are restricted to 1 movement on land (non-ice tiles)
    if settings.skills.contains(&SkillType::Skate) && !tile.is_frozen() {
        return true;
    }

    // Embark (Land unit to Port)
    let is_port = get_structure_type_at(state, tile_idx) == Some(StructureType::Port)
        && tile.owner == unit.owner;

    // "Most land units moving into a Port... unable to perform actions... applied to land units"
    // Exceptions: Water, Amphibious, Flying.
    let is_aquatic = settings.skills.contains(&SkillType::Float);
    let is_amphibious = crate::functions::is_amphibious(unit);

    if is_port && !is_aquatic && !is_amphibious {
        return true;
    }

    // Zone of Control (ZOC)
    if is_adjacent_to_enemy(state, tile_idx, unit.owner) {
        let ignores_zoc = settings.skills.contains(&SkillType::Infiltrate)
            || settings.skills.contains(&SkillType::Creep)
            || settings.skills.contains(&SkillType::Sneak)
            || unit.effects.contains(&UnitEffect::Invisible);
        if !ignores_zoc {
            return true;
        }
    }

    if tile.is_water_terrain() {
        // Navigate on to village
        if settings.skills.contains(&SkillType::Navigate)
            && get_structure_type_at(state, tile_idx) == Some(StructureType::Village)
        {
            return true;
        }
    } else {
        // Disembark (Water unit to Land)
        // If Amphibious, land is not terminal (just slow, handled in cost).
        if is_naval_unit(unit.unit_type) && !is_amphibious {
            return true;
        }
    }

    // blocking terrain (Rough Terrain)
    // "Units cannot move through them (they can move onto them, but no further in the same turn)"
    // Flooded rough terrain usually acts like water (not terminal for aquatic)
    // Algae is Rough Terrain (stops movement)
    if tile.is_algae()
        && !settings.skills.contains(&SkillType::Creep)
        && !settings.skills.contains(&SkillType::Fly)
    // Respect "stops the movement of units without the creep or fly unit skills"
    {
        return true;
    }

    if !tile.is_flooded() {
        match tile.terrain_type {
            TerrainType::Forest => {
                let has_creep = settings.skills.contains(&SkillType::Creep);
                let has_usable_road = is_roadpath_and_usable(state, unit, tile_idx);
                // Forest blocks unless Creep or Road
                if !has_creep && !has_usable_road {
                    return true;
                }
            }
            TerrainType::Mountain => {
                // Mountain blocks movement (Creep no longer overcomes this penalty)
                return true;
            }
            _ => {}
        }
    }

    false
}

fn is_roadpath_and_usable(state: &GameState, unit: &UnitState, idx: i32) -> bool {
    let tile = match state.tiles.get(&idx) {
        Some(t) => t,
        None => return false,
    };

    // Bridges also act as roads!
    let structure = get_structure_at(state, idx);
    let has_road = tile.has_road
        || crate::functions::get_city_at(state, idx).is_some()
        || (structure.is_some() && structure.unwrap().structure_type == StructureType::Bridge)
        // Also acts like a city
        || (structure.is_some() && structure.unwrap().structure_type == StructureType::Village);
    if !has_road {
        return false;
    }

    // Usable if friendly or neutral or peace treaty or Infiltrate
    tile.owner == unit.owner
        || tile.owner == 0
        || crate::functions::is_at_peace(state, unit.owner, tile.owner)
        || get_unit_setting(unit.unit_type)
            .skills
            // TODO not sure infiltrate influences movement range..?
            .contains(&SkillType::Infiltrate)
}

fn is_naval_unit(unit_type: UnitType) -> bool {
    crate::settings::units::get_unit_setting(unit_type)
        .skills
        .contains(&SkillType::Carry)
}

fn is_adjacent_to_enemy(state: &GameState, idx: i32, owner: PlayerId) -> bool {
    for n_idx in get_adjacent_indices(state, idx, 1) {
        if let Some(enemy) = get_enemy_at(state, n_idx, owner) {
            if !enemy.effects.contains(&UnitEffect::Invisible) {
                return true;
            }
        }
    }
    false
}

/// Generate attack moves for a unit
fn generate_attack_moves(
    state: &GameState,
    unit: &UnitState,
    from_idx: i32,
    moves: &mut Vec<Box<dyn Move>>,
) {
    // If unit has moved, it needs Dash to attack
    // Skate units on land cannot use Dash
    let on_ice = crate::functions::is_tile_frozen(state, from_idx);
    let can_dash = crate::functions::has_skill(unit, SkillType::Dash)
        && (!crate::functions::has_skill(unit, SkillType::Skate) || on_ice);

    if unit.moved && !can_dash {
        return;
    }

    if crate::functions::has_skill(unit, SkillType::Infiltrate) {
        for target_idx in get_adjacent_indices(state, from_idx, 1) {
            if crate::functions::is_enemy_city(state, target_idx, unit.owner)
                && !crate::functions::is_under_siege(state, target_idx)
            {
                moves.push(Box::new(AttackMove::new(from_idx, target_idx)));
            }
        }
    } else {
        if crate::functions::get_unit_attack(state, unit) < 0.0 {
            return;
        }

        let range = get_unit_setting(unit.unit_type).range;

        // Check all tiles in range
        for target_idx in get_tiles_in_range(state, from_idx, range) {
            if let Some(enemy) = get_enemy_at(state, target_idx, unit.owner) {
                if !crate::functions::is_at_peace(state, unit.owner, enemy.owner) {
                    moves.push(Box::new(AttackMove::new(from_idx, target_idx)));
                }
            }
        }
    }
}

/// Get all tile indices within range (for ranged units)
fn get_tiles_in_range(state: &GameState, from_idx: i32, range: i32) -> Vec<i32> {
    let map_size = state.settings.size;
    let fx = from_idx % map_size;
    let fy = from_idx / map_size;

    let mut tiles = Vec::new();

    for dy in -range..=range {
        for dx in -range..=range {
            if dx == 0 && dy == 0 {
                continue;
            }

            let nx = fx + dx;
            let ny = fy + dy;

            if nx >= 0 && nx < map_size && ny >= 0 && ny < map_size {
                if dx.abs().max(dy.abs()) <= range {
                    tiles.push(ny * map_size + nx);
                }
            }
        }
    }

    tiles
}

fn is_steppable(state: &GameState, unit: &UnitState, idx: i32, strict: bool) -> bool {
    // Can only step on explored tiles, regardless F
    let is_explored = state
        .tiles
        .get(&idx)
        .map(|t| t.explorers.contains(&unit.owner))
        .unwrap_or(false);

    if !is_explored {
        return false;
    }

    let settings = get_unit_setting(unit.unit_type);

    let tribe = state.tribes.get(&unit.owner).unwrap();
    let tile = state.tiles.get(&idx).unwrap();

    // Use FOW-safe terrain access for MCTS anti-cheat
    let terrain = crate::fow::get_terrain_at(state, idx, unit.owner);

    // tech check using FOW-safe terrain
    if !is_navigationable_terrain(state, tribe, unit.unit_type, terrain, tile) {
        return false;
    }

    // SkillType::Water units can only move on water/ocean/flooded tiles
    if settings.skills.contains(&SkillType::Water) {
        let is_water_like =
            matches!(terrain, TerrainType::Water | TerrainType::Ocean) || tile.is_flooded();
        if !is_water_like {
            return false;
        }
    }

    // cannot step if the tile is occupied by an enemy (or any unit if strict)
    if let Some(other) = get_unit_at(state, idx) {
        if strict || other.owner != unit.owner {
            return false;
        }
    }

    let is_aquatic = settings.skills.contains(&SkillType::Float)
        || settings.skills.contains(&SkillType::Fly)
        || settings.skills.contains(&SkillType::Water);

    if !is_aquatic {
        // Algae and Bridges act like a bridge for land units
        let has_bridge = get_structure_type_at(state, idx) == Some(StructureType::Bridge);
        if tile.is_algae() || has_bridge {
            // Pass
        } else if let Some(structure) = get_structure_at(state, idx) {
            if structure.structure_type == StructureType::Port {
                return tile.owner == unit.owner;
            }
        }
    }

    if settings.skills.contains(&SkillType::Navigate) {
        let is_water = matches!(terrain, TerrainType::Water | TerrainType::Ocean);
        if !is_water {
            if let Some(structure) = get_structure_at(state, idx) {
                match structure.structure_type {
                    StructureType::Village | StructureType::Ruin => return true,
                    _ => return false,
                }
            } else {
                return false;
            }
        }
    } else if is_aquatic {
        // Aquatic units can step on any water/ocean
        let is_water = matches!(terrain, TerrainType::Water | TerrainType::Ocean);
        if is_water {
            return true;
        }
    } else {
        // Fallback to non-water units, unsteppable
        let is_water = matches!(terrain, TerrainType::Water | TerrainType::Ocean);
        if is_water {
            // algae and bridges are steppable water
            let has_bridge = get_structure_type_at(state, idx) == Some(StructureType::Bridge);
            return tile.is_algae() || has_bridge;
        }
    }

    true
}

/// FOW-safe version that takes terrain as a parameter (may be predicted during MCTS)
fn is_navigationable_terrain(
    state: &GameState,
    tribe: &TribeState,
    unit_type: UnitType,
    terrain: TerrainType,
    tile: &TileState,
) -> bool {
    if has_skill(unit_type, SkillType::Fly) || has_skill(unit_type, SkillType::Navigate) {
        return true;
    }

    if has_skill(unit_type, SkillType::Water) {
        // Restricted to water/ocean/algae/ice/flooded land
        let is_water_like = matches!(terrain, TerrainType::Water | TerrainType::Ocean)
            || tile.is_algae()
            || tile.is_frozen()
            || tile.is_flooded();
        return is_water_like;
    }

    let has_bridge = get_structure_type_at(state, tile.coords.idx) == Some(StructureType::Bridge);
    if tile.is_frozen() || tile.is_algae() || has_bridge {
        return true;
    }

    match terrain {
        TerrainType::Water => tribe.tech_vanilla.iter().any(|t| {
            let s = crate::settings::technology::get_technology_setting(t.tech_type);
            s.unlocks_terrain == Some(TerrainType::Water)
        }),
        TerrainType::Ocean => tribe.tech_vanilla.iter().any(|t| {
            let s = crate::settings::technology::get_technology_setting(t.tech_type);
            s.unlocks_terrain == Some(TerrainType::Ocean)
        }),
        TerrainType::Mountain => tribe.tech_vanilla.iter().any(|t| {
            let s = crate::settings::technology::get_technology_setting(t.tech_type);
            s.unlocks_terrain == Some(TerrainType::Mountain)
        }),
        _ => {
            // If not water/mountain, it's walkable land unless it's flooded.
            // Flooded land is treated like water.
            if tile.is_flooded() {
                return tribe.tech_vanilla.iter().any(|t| {
                    let s = crate::settings::technology::get_technology_setting(t.tech_type);
                    s.unlocks_terrain == Some(TerrainType::Water)
                });
            }
            true
        }
    }
}
