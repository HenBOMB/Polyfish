use polyfish_rs::states::*;
use polyfish_rs::types::*;
use polyfish_rs::functions::*;
use polyfish_rs::actions::units::{push_unit, spawn_unit};
use polyfish_rs::moves::Move;

fn setup_game(size: i32) -> GameState {
    let mut state = GameState::default();
    state.settings.size = size;
    state.settings.current_player_turn_id = 1;
    
    // Add two tribes
    state.tribes.insert(1, TribeState::new(1, TribeType::XinXi));
    state.tribes.insert(2, TribeState::new(2, TribeType::Imperius));
    
    // Fill with basic tiles
    for y in 0..size {
        for x in 0..size {
            let idx = y * size + x;
            state.tiles.insert(idx, TileState {
                coords: Coords::from_index(idx, size),
                terrain_type: TerrainType::Field,
                owner: 0,
                ..Default::default()
            });
        }
    }
    state
}

#[test]
fn test_push_friendly_move() {
    let mut state = setup_game(11);
    let size = 11;
    let center_idx = 11 * 5 + 5; // (5,5)
    
    // Add friendly unit at (5,5) that moved from (5,4) - moved South
    let mut unit = UnitState::new(1, UnitType::Warrior, Coords::from_index(center_idx, size));
    unit.prev_coords = Coords::from_index(11 * 4 + 5, size); // (5,4)
    unit.moved = true;
    state.tribes.get_mut(&1).unwrap().units.push(unit);
    state.tiles.get_mut(&center_idx).unwrap()._unit_owner_id = Some(1);
    
    // Calculate pushable position
    let unit_ref = &state.tribes[&1].units[0];
    let dest = calculate_pushable_position(&state, unit_ref);
    
    // Should be pushed South (same direction) to (5,6)
    assert_eq!(dest, Some(11 * 6 + 5));
}

#[test]
fn test_push_enemy_move() {
    let mut state = setup_game(11);
    let size = 11;
    let center_idx = 11 * 5 + 5; // (5,5)
    
    // Add enemy unit at (5,5) that moved from (5,4) - moved South
    let mut unit = UnitState::new(2, UnitType::Warrior, Coords::from_index(center_idx, size));
    unit.prev_coords = Coords::from_index(11 * 4 + 5, size); // (5,4)
    unit.moved = true;
    state.tribes.get_mut(&2).unwrap().units.push(unit);
    state.tiles.get_mut(&center_idx).unwrap()._unit_owner_id = Some(2);
    
    // Current player is 1
    state.settings.current_player_turn_id = 1;
    
    // Calculate pushable position
    let unit_ref = &state.tribes[&2].units[0];
    let dest = calculate_pushable_position(&state, unit_ref);
    
    // Should be pushed North (opposite direction) to (5,4)
    assert_eq!(dest, Some(11 * 4 + 5));
}

#[test]
fn test_push_ranged_attack() {
    let mut state = setup_game(11);
    let size = 11;
    let center_idx = 11 * 5 + 5; // (5,5)
    
    // Add friendly archer at (5,5) that attacked (5,4) - attacked North
    let mut unit = UnitState::new(1, UnitType::Archer, Coords::from_index(center_idx, size));
    unit.last_attack_coords = Some(Coords::from_index(11 * 4 + 5, size));
    unit.attacked = true;
    unit.moved = false; // Priority: Attack
    state.tribes.get_mut(&1).unwrap().units.push(unit);
    state.tiles.get_mut(&center_idx).unwrap()._unit_owner_id = Some(1);
    
    // Calculate pushable position
    let unit_ref = &state.tribes[&1].units[0];
    let dest = calculate_pushable_position(&state, unit_ref);
    
    // Should be pushed North (direction of attack) to (5,4)
    assert_eq!(dest, Some(11 * 4 + 5));
}

#[test]
fn test_push_towards_center() {
    let mut state = setup_game(11);
    let size = 11;
    let idx = 0; // (0,0) NW corner
    
    // Add idle unit
    let unit = UnitState::new(1, UnitType::Warrior, Coords::from_index(idx, size));
    state.tribes.get_mut(&1).unwrap().units.push(unit);
    state.tiles.get_mut(&idx).unwrap()._unit_owner_id = Some(1);
    
    // Calculate pushable position
    let unit_ref = &state.tribes[&1].units[0];
    let dest = calculate_pushable_position(&state, unit_ref);
    
    // Should be pushed towards center (5,5) -> SE direction (1,1)
    assert_eq!(dest, Some(11 * 1 + 1));
}

#[test]
fn test_push_center_spawn() {
    let mut state = setup_game(11);
    let size = 11;
    let center_idx = 11 * 5 + 5; // (5,5) exact center
    
    // Add idle unit at center
    let unit = UnitState::new(1, UnitType::Warrior, Coords::from_index(center_idx, size));
    state.tribes.get_mut(&1).unwrap().units.push(unit);
    state.tiles.get_mut(&center_idx).unwrap()._unit_owner_id = Some(1);
    
    // Calculate pushable position
    let unit_ref = &state.tribes[&1].units[0];
    let dest = calculate_pushable_position(&state, unit_ref);
    
    // Should be pushed South by default (5,6)
    assert_eq!(dest, Some(11 * 6 + 5));
}

#[test]
fn test_push_alternating_search() {
    let mut state = setup_game(11);
    let size = 11;
    let center_idx = 11 * 5 + 5; // (5,5)
    
    // Unit at (5,5) moved South from (5,4)
    let mut unit = UnitState::new(1, UnitType::Warrior, Coords::from_index(center_idx, size));
    unit.prev_coords = Coords::from_index(11 * 4 + 5, size);
    unit.moved = true;
    state.tribes.get_mut(&1).unwrap().units.push(unit);
    state.tiles.get_mut(&center_idx).unwrap()._unit_owner_id = Some(1);
    
    // Block South (5,6)
    let s_idx = 11 * 6 + 5;
    state.tribes.get_mut(&2).unwrap().units.push(UnitState::new(2, UnitType::Warrior, Coords::from_index(s_idx, size)));
    state.tiles.get_mut(&s_idx).unwrap()._unit_owner_id = Some(2);
    
    // Order: Target (S), CCW1 (SW), CW1 (SE), ...
    // CCW1: (4,6) -> idx 11*6 + 4 = 70.
    
    let unit_ref = &state.tribes[&1].units[0];
    let dest = calculate_pushable_position(&state, unit_ref);
    
    assert_eq!(dest, Some(70));
}

#[test]
fn test_push_mountain_climbing() {
    let mut state = setup_game(11);
    let size = 11;
    let center_idx = 11 * 5 + 5;
    
    // Warrior moved South to (5,5), target (5,6) is Mountain
    let mut unit = UnitState::new(1, UnitType::Warrior, Coords::from_index(center_idx, size));
    unit.prev_coords = Coords::from_index(11 * 4 + 5, size);
    unit.moved = true;
    state.tribes.get_mut(&1).unwrap().units.push(unit);
    
    let m_idx = 11 * 6 + 5;
    state.tiles.get_mut(&m_idx).unwrap().terrain_type = TerrainType::Mountain;
    
    let unit_ref = &state.tribes[&1].units[0];
    let dest = calculate_pushable_position(&state, unit_ref);
    
    // Warrior doesn't have Climbing, so it should skip (5,6) and try CCW1 (SW) (4,6)
    assert_eq!(dest, Some(70));
    
    // Now give it Climbing (via tech/skill)
    // Actually Warrior settings don't have it, but we can simulate it by giving them the skill
    state.tribes.get_mut(&1).unwrap().tech_vanilla.push(DiscoveryTech { tech_type: TechnologyType::Climbing, discovered: true });
    
    // calculate_pushable_position calls has_skill(unit, Climb)
    // We need to ensure the skill check correctly pulls from tech.
    // In our codebase, has_skill(unit, skill) often checks tech too.
    
    // Wait, let's just make it a unit type with Climbing like a Cymanti Phychi or something?
    // Phychi is Fly.
    // Let's just check the logic in functions.rs: `has_skill(unit, SkillType::Climb)`
}
