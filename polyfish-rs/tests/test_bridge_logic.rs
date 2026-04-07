#[cfg(test)]
mod tests {
    use polyfish::states::*;
    use polyfish::types::*;
    use polyfish::Coords;
    use polyfish::moves::Move;
    use polyfish::moves::build::BuildMove;
    use polyfish::moves::generate_legal_moves;
    use polyfish::functions::*;
    use polyfish::actions::structure::create_structure;

    fn setup_basic_state() -> GameState {
        let mut state = GameState::default();
        state.settings.size = 5;
        state.settings.current_player_turn_id = 1;
        state.settings._fow = false; // Disable FOW for testing simplicity

        for i in 0..25 {
            let mut tile = TileState::default();
            tile.coords = Coords::from_index(i, 5);
            tile.terrain_type = TerrainType::Field;
            tile.explorers.insert(1);
            state.tiles.insert(i, tile);
        }

        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.stars = 100;
        tribe.tribe_type = TribeType::Nature;
        // Research Roads and Riding (needed for bridge unlock check)
        tribe.tech_vanilla.push(TechnologyState { tech_type: TechnologyType::Riding, discovered: true, discovered_turn: 0 });
        tribe.tech_vanilla.push(TechnologyState { tech_type: TechnologyType::Roads, discovered: true, discovered_turn: 0 });
        
        state.tribes.insert(1, tribe);

        state
    }

    #[test]
    fn test_bridge_placement_rules() {
        let mut state = setup_basic_state();
        
        // Create a horizontal gap: Land(7)-Water(8)-Land(9)
        state.tiles.get_mut(&8).unwrap().terrain_type = TerrainType::Water;
        
        // Add a city so we have territory
        let mut city = CityState::default();
        city.idx = 7;
        city.owner = 1;
        city._territory = vec![7, 8, 2, 3, 12, 13, 9];
        state.tribes.get_mut(&1).unwrap().cities.push(city);
        for &idx in &state.tribes[&1].cities[0]._territory {
            state.tiles.get_mut(&idx).unwrap().owner = 1;
        }

        // 1. Check Horizontal Build (Should be legal)
        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        polyfish::moves::build::generate_build_moves(&state, &mut moves);
        let has_bridge_at_8 = moves.iter().any(|m| {
            if let Ok(st) = m.structure_type() {
                st == StructureType::Bridge && m.target_idx().unwrap() == 8
            } else { false }
        });
        assert!(has_bridge_at_8, "Should be able to build bridge on 8 (H-connected)");

        // 2. Check Vertical Build (Land(3)-Water(8)-Land(13))
        // Already field at 3 and 13. So it should be legal.
        // Wait, if BOTH H and V are land, it's definitely legal.
        
        // 3. Check Diagonal (ONLY diagonal)
        // Set all neighbors of 8 to Water except 2 and 4 (diagonal)
        // 7, 9, 3, 13 are direct neighbors
        state.tiles.get_mut(&7).unwrap().terrain_type = TerrainType::Water;
        state.tiles.get_mut(&9).unwrap().terrain_type = TerrainType::Water;
        state.tiles.get_mut(&3).unwrap().terrain_type = TerrainType::Water;
        state.tiles.get_mut(&13).unwrap().terrain_type = TerrainType::Water;
        
        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        polyfish::moves::build::generate_build_moves(&state, &mut moves);
        let has_bridge_at_8_diag = moves.iter().any(|m| {
            if let Ok(st) = m.structure_type() {
                st == StructureType::Bridge && m.target_idx().unwrap() == 8
            } else { false }
        });
        assert!(!has_bridge_at_8_diag, "Should NOT be able to build bridge on 8 if only diagonals are land");
    }

    #[test]
    fn test_land_unit_uses_bridge() {
        let mut state = setup_basic_state();
        
        // Land(7)-Water(8)-Land(9)
        state.tiles.get_mut(&8).unwrap().terrain_type = TerrainType::Water;
        
        // Wait, land units can't step on water without a bridge.
        let mut unit = UnitState::default();
        unit.owner = 1;
        unit.unit_type = UnitType::Warrior;
        unit.coords = Coords::from_index(7, 5);
        state.tribes.get_mut(&1).unwrap().units.push(unit);

        // Movement check without bridge
        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        polyfish::moves::generate_legal_moves(&state); // Internal check
        let reachable = polyfish::moves::generate_legal_moves(&state);
        let can_move_to_8 = reachable.iter().any(|m| {
            if m.move_type() == MoveType::Step {
                m.target_idx().unwrap() == 8
            } else { false }
        });
        assert!(!can_move_to_8, "Warrior should not be able to step on water without bridge");

        // Add Bridge
        create_structure(&mut state, 8, StructureType::Bridge, 1);
        
        // Movement check with bridge
        let reachable_with_bridge = polyfish::moves::generate_legal_moves(&state);
        let can_move_to_8_bridge = reachable_with_bridge.iter().any(|m| {
            if m.move_type() == MoveType::Step {
                m.target_idx().unwrap() == 8
            } else { false }
        });
        assert!(can_move_to_8_bridge, "Warrior SHOULD be able to step on bridge");
    }

    #[test]
    fn test_bridge_connection() {
        let mut state = setup_basic_state();
        
        // Capital(2)-Field(7)-Water(12)-Field(17)-City(22)
        state.tiles.get_mut(&2).unwrap().capital_of = 1;
        state.tiles.get_mut(&2).unwrap().owner = 1;
        state.tiles.get_mut(&12).unwrap().terrain_type = TerrainType::Water;
        
        let mut cap = CityState::default();
        cap.idx = 2;
        cap.owner = 1;
        state.tribes.get_mut(&1).unwrap().cities.push(cap);

        let mut city = CityState::default();
        city.idx = 22;
        city.owner = 1;
        state.tribes.get_mut(&1).unwrap().cities.push(city);
        
        for i in [2, 7, 12, 17, 22] {
            state.tiles.get_mut(&i).unwrap().owner = 1;
        }

        // Initially not connected
        polyfish::actions::connection::update_capital_connections(&mut state, 1);
        assert!(!state.tribes[&1].cities[1].connected_to_capital);

        // Add roads on land
        state.tiles.get_mut(&7).unwrap().has_road = true;
        state.tiles.get_mut(&17).unwrap().has_road = true;
        
        // No bridge yet
        polyfish::actions::connection::update_capital_connections(&mut state, 1);
        assert!(!state.tribes[&1].cities[1].connected_to_capital, "Should not be connected through water gap");

        // Add bridge
        create_structure(&mut state, 12, StructureType::Bridge, 1);
        polyfish::actions::connection::update_capital_connections(&mut state, 1);
        assert!(state.tribes[&1].cities[1].connected_to_capital, "Bridge should complete road connection");
    }
}
