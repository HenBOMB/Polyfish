//! Guards on `execute()`: a move that cannot land must `Err`, never silently no-op.

#[cfg(test)]
mod tests {
    use polyfish::Coords;
    use polyfish::moves::Move;
    use polyfish::moves::abilities::forest::GrowForestMove;
    use polyfish::moves::step::StepMove;
    use polyfish::states::*;
    use polyfish::types::*;

    fn setup_state(stars: i32) -> GameState {
        let mut state = GameState::default();
        state.settings.size = 5;
        state.settings.current_player_turn_id = 1;

        for i in 0..25 {
            let mut tile = TileState::default();
            tile.coords = Coords::from_index(i, 5);
            tile.terrain_type = TerrainType::Field;
            tile.owner = 1;
            state.tiles.insert(i, tile);
        }

        let mut tribe = TribeState::default();
        tribe.id = 1;
        tribe.stars = stars;
        tribe.tribe_type = TribeType::Imperius;
        state.tribes.insert(1, tribe);

        state
    }

    #[test]
    fn step_with_no_unit_at_source_errs() {
        let mut state = setup_state(10);

        let result = StepMove::new(12, 13).execute(&mut state);

        assert!(
            result.is_err(),
            "StepMove with no unit at src must Err so self_play counts an aborted game, \
             not record a policy target for a move that changed nothing"
        );
    }

    #[test]
    fn grow_forest_without_stars_errs_and_leaves_terrain_alone() {
        let mut state = setup_state(2);

        let result = GrowForestMove::new(12).execute(&mut state);

        assert!(result.is_err(), "unaffordable GrowForest must Err");
        assert_eq!(
            state.tiles[&12].terrain_type,
            TerrainType::Field,
            "affordability must be checked before any state mutation"
        );
        assert_eq!(state.tribes[&1].stars, 2, "stars must not go negative");
    }

    #[test]
    fn grow_forest_with_stars_succeeds() {
        let mut state = setup_state(5);

        let result = GrowForestMove::new(12).execute(&mut state);

        assert!(result.is_ok());
        assert_eq!(state.tiles[&12].terrain_type, TerrainType::Forest);
        assert_eq!(state.tribes[&1].stars, 0);
    }
}
