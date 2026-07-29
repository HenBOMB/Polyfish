#[cfg(test)]
mod tests {
    use polyfish::states::*;
    use polyfish::types::*;
    use polyfish::Coords;
    use polyfish::actions::structure::{build_structure, destroy_structure};
    use polyfish::actions::city::claim_territory;
    use polyfish::functions::get_city_production;

    fn setup_state_with_city() -> GameState {
        let mut state = GameState::default();
        state.settings.size = 5;
        state.settings.current_player_turn_id = 1;

        for i in 0..25 {
            let mut tile = TileState::default();
            tile.coords = Coords::from_index(i, 5);
            tile.terrain_type = TerrainType::Field;
            tile.owner = 0;
            state.tiles.insert(i, tile);
        }

        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.stars = 100;
        tribe.tribe_type = TribeType::Imperius;
        tribe.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Farming,
            discovered: true,
            discovered_turn: 0,
        });
        tribe.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Construction,
            discovered: true,
            discovered_turn: 0,
        });
        state.tribes.insert(1, tribe);

        let mut city = CityState::default();
        city.idx = 12; // Center
        city.owner = 1;
        city.level = 1;
        city.population = 0;
        city.progress = 0;
        state.tribes.get_mut(&1).unwrap().cities.push(city);
        state.tiles.get_mut(&12).unwrap().capital_of = 1;

        let territory: Vec<i32> = (0..25).collect();
        let _ = claim_territory(&mut state, &territory, 12, true); // Don't invoke the undo callback!

        state
    }

    #[test]
    fn test_destroy_building_negative_progress() {
        let mut state = setup_state_with_city();

        // Build a Farm (+2 pop) at tile 13 (adjacent to city at 12)
        // Level 1 city needs 2 progress to level up.
        let _undo = build_structure(&mut state, 13, StructureType::Farm);

        let city = &state.tribes.get(&1).unwrap().cities[0];
        assert_eq!(city.level, 2, "City should level up to level 2");
        assert_eq!(city.population, 2, "Total population should be 2");
        assert_eq!(city.progress, 0, "Progress resets to 0 after level up (2 - 2 = 0)");

        // Now destroy the Farm (-2 pop)
        let undo_destroy = destroy_structure(&mut state, 13);

        let city_after = &state.tribes.get(&1).unwrap().cities[0];
        assert_eq!(city_after.level, 1, "City level drops back to 1 when progress goes negative");
        assert_eq!(city_after.population, 0, "Total population is 0 (2 - 2 = 0)");
        assert_eq!(city_after.progress, 0, "Progress returns to 0 after leveling down from level 2");

        // Verify undo restores the level 2 city
        undo_destroy(&mut state);
        let city_undo = &state.tribes.get(&1).unwrap().cities[0];
        assert_eq!(city_undo.level, 2, "City level restored to 2 on undo");
        assert_eq!(city_undo.population, 2, "Population restored on undo");
        assert_eq!(city_undo.progress, 0, "Progress restored on undo");
    }

    #[test]
    fn test_destroy_building_negative_population_and_progress() {
        let mut state = setup_state_with_city();

        // Suppose we have a structure on tile 13 that was placed without population (or we destroy when pop=0)
        let mut struct_state = StructureState::default();
        struct_state.structure_type = StructureType::Farm; // Farm has reward_pop = 2
        state.structures.insert(13, Some(struct_state));

        // Currently city population is 0, progress is 0, level is 1
        let _undo_destroy = destroy_structure(&mut state, 13);

        let city_after = &state.tribes.get(&1).unwrap().cities[0];
        assert_eq!(city_after.population, -2, "Total population becomes negative (-2)");
        assert_eq!(city_after.progress, -2, "Progress becomes negative (-2)");
        assert_eq!(city_after.level, 1, "Level remains 1");
    }

    #[test]
    fn test_negative_population_income_penalty() {
        let mut state = setup_state_with_city();

        // Set city production to 3 (e.g., Level 3 city or base + bonuses)
        {
            let city = &mut state.tribes.get_mut(&1).unwrap().cities[0];
            city.production = 3;
            city.progress = 0;
        }

        let city = state.tribes.get(&1).unwrap().cities[0].clone();
        assert_eq!(get_city_production(&state, &city), 4, "Base 3 + 1 capital bonus = 4 SPT");

        // Now set progress to -1 -> income should decrease by 1
        {
            let city = &mut state.tribes.get_mut(&1).unwrap().cities[0];
            city.progress = -1;
        }
        let city = state.tribes.get(&1).unwrap().cities[0].clone();
        assert_eq!(get_city_production(&state, &city), 3, "4 - 1 penalty = 3 SPT");

        // Now set progress to -2 -> income should decrease by 2
        {
            let city = &mut state.tribes.get_mut(&1).unwrap().cities[0];
            city.progress = -2;
        }
        let city = state.tribes.get(&1).unwrap().cities[0].clone();
        assert_eq!(get_city_production(&state, &city), 2, "4 - 2 penalty = 2 SPT");

        // Now set progress to -10 -> income should be clamped at 0
        {
            let city = &mut state.tribes.get_mut(&1).unwrap().cities[0];
            city.progress = -10;
        }
        let city = state.tribes.get(&1).unwrap().cities[0].clone();
        assert_eq!(get_city_production(&state, &city), 0, "4 - 10 penalty clamped to 0 SPT");
    }
}
