#[cfg(test)]
mod tests {
    use polyfish::states::*;
    use polyfish::types::*;
    use polyfish::Coords;
    use polyfish::moves::Move;
    use polyfish::moves::build::BuildMove;
    use polyfish::functions::*;

    fn setup_cymanti_state() -> GameState {
        let mut state = GameState::default();
        state.settings.size = 5;
        state.settings.current_player_turn_id = 1;
        state.settings._fow = false;

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
        tribe.tribe_type = TribeType::Cymanti;
        // Basic tech is always there
        tribe.tech_vanilla.push(TechnologyState { tech_type: TechnologyType::Basic, discovered: true, discovered_turn: 0 });
        
        state.tribes.insert(1, tribe);

        // Add a city
        let mut city = CityState::default();
        city.idx = 12; // Center
        city.owner = 1;
        city._territory = (0..25).collect();
        state.tribes.get_mut(&1).unwrap().cities.push(city);
        for i in 0..25 {
            state.tiles.get_mut(&i).unwrap().owner = 1;
            state.tiles.get_mut(&i).unwrap().ruling_city_coords = Some(Coords::from_index(12, 5));
        }

        state
    }

    #[test]
    fn test_algae_build_logic() {
        let mut state = setup_cymanti_state();
        
        // 1. Set some water tiles
        state.tiles.get_mut(&7).unwrap().terrain_type = TerrainType::Water;
        state.tiles.get_mut(&13).unwrap().terrain_type = TerrainType::Ocean;

        // 2. Initially, Algae should not be buildable (no Hydrology)
        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        polyfish::moves::build::generate_build_moves(&state, &mut moves);
        let has_algae = moves.iter().any(|m| {
            m.structure_type().map_or(false, |st| st == StructureType::Algae)
        });
        assert!(!has_algae, "Should not be able to build Algae without Hydrology");

        // 3. Research Hydrology
        state.tribes.get_mut(&1).unwrap().tech_vanilla.push(TechnologyState { 
            tech_type: TechnologyType::Hydrology, 
            discovered: true,
            discovered_turn: 0
        });

        // 4. Now Algae should be buildable on Water and Ocean
        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        polyfish::moves::build::generate_build_moves(&state, &mut moves);
        
        let algae_at_7 = moves.iter().any(|m| {
            m.structure_type().map_or(false, |st| st == StructureType::Algae) && m.target_idx().unwrap() == 7
        });
        let algae_at_13 = moves.iter().any(|m| {
            m.structure_type().map_or(false, |st| st == StructureType::Algae) && m.target_idx().unwrap() == 13
        });
        
        assert!(algae_at_7, "Should be able to build Algae on Water at 7");
        assert!(algae_at_13, "Should be able to build Algae on Ocean at 13");

        // 5. Execute BuildMove at 7
        let build_move = BuildMove::new(7, StructureType::Algae);
        let initial_stars = state.tribes[&1].stars;
        let initial_pop = state.tribes[&1].cities[0].population;

        let res = build_move.execute(&mut state).expect("Build Algae should succeed");
        
        assert_eq!(state.tribes[&1].stars, initial_stars - 5, "Should cost 5 stars");
        assert_eq!(state.tribes[&1].cities[0].population, initial_pop + 1, "Should reward 1 population");
        assert!(state.tiles[&7].effects.contains(&TileEffect::Algae), "Tile should have Algae effect");
        assert_eq!(get_structure_type_at(&state, 7), Some(StructureType::Algae), "Tile should have Algae structure");

        // 6. Undo
        (res.undo)(&mut state);
        assert_eq!(state.tribes[&1].stars, initial_stars, "Stars should be restored on undo");
        assert_eq!(state.tribes[&1].cities[0].population, initial_pop, "Population should be restored on undo");
        assert!(!state.tiles[&7].effects.contains(&TileEffect::Algae), "Algae effect should be removed on undo");
        assert_eq!(get_structure_type_at(&state, 7), None, "Algae structure should be removed on undo");
    }
}
