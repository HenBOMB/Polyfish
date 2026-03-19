#[cfg(test)]
mod tests {
    use polyfish::moves::Move;
    use polyfish::states::{GameState, StructureState, TechnologyState, TileState, TribeState};
    use polyfish::types::{StructureType, TechnologyType, TerrainType, TribeType};

    #[test]
    fn test_forest_moves_blocked_by_structure() {
        let mut state = GameState::default();
        state.settings.current_player_turn_id = 1;
        state.settings.size = 11;

        // Setup tile with Forest and Structure
        let idx = 0;
        state.map.tiles.insert(
            idx,
            TileState {
                coords: polyfish::coords::Coords { x: 0, y: 0, idx },
                terrain_type: TerrainType::Forest,
                owner: 1,
                ..Default::default()
            },
        );

        // Add Structure (EyeOfGod)
        state.structures.insert(
            idx,
            Some(StructureState {
                structure_type: StructureType::EyeOfGod,
                tile_index: idx,
                ..Default::default()
            }),
        );

        // Setup Tribe with Forestry tech (needed for Clear Forest)
        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.tribe_type = TribeType::Imperius;
        tribe.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Forestry,
            discovered: true,
        });

        // Add city with territory
        let mut city = polyfish::states::CityState::default();
        city.tile_index = 10; // capital elsewhere
        city._territory.push(idx);
        tribe.cities.push(city);

        state.tribes.insert(1, tribe);

        // Generate moves
        let mut moves: Vec<Box<dyn Move>> = Vec::new();
        polyfish::moves::abilities::forest::generate_forest_moves(&state, &mut moves);

        // Assert no forest moves generated
        assert!(
            moves.is_empty(),
            "Forest moves should be blocked by structure"
        );
    }

    #[test]
    fn test_harvest_moves_blocked_by_structure() {
        use polyfish::states::ResourceState;
        use polyfish::types::ResourceType;

        let mut state = GameState::default();
        state.settings.current_player_turn_id = 1;

        let idx = 1;
        state.map.tiles.insert(
            idx,
            TileState {
                coords: polyfish::coords::Coords { x: 0, y: 1, idx },
                terrain_type: TerrainType::Field,
                owner: 1,
                ..Default::default()
            },
        );

        // Add Resource (Fruit)
        state.resources.insert(
            idx,
            Some(ResourceState {
                resource_type: ResourceType::Fruit,
                tile_index: idx,
                ..Default::default()
            }),
        );

        // Add Structure (EyeOfGod)
        state.structures.insert(
            idx,
            Some(StructureState {
                structure_type: StructureType::EyeOfGod,
                tile_index: idx,
                ..Default::default()
            }),
        );

        // Setup Tribe with Organization (unlocks harvest fruit)
        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.tribe_type = TribeType::Imperius;
        tribe.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Organization,
            discovered: true,
        });
        tribe.stars = 10;

        // Territory
        let mut city = polyfish::states::CityState::default();
        city._territory.push(idx);
        tribe.cities.push(city);
        state.tribes.insert(1, tribe);

        // Generate generic moves (which calls econ moves -> harvest)
        let moves = polyfish::moves::generate_legal_moves(&state);

        // Filter for HarvestMove on idx
        let harvest_moves: Vec<_> = moves
            .iter()
            .filter(|m| m.move_type() == polyfish::types::MoveType::Harvest)
            .collect();

        assert!(
            harvest_moves.is_empty(),
            "Harvest moves should be blocked by structure"
        );
    }
}
