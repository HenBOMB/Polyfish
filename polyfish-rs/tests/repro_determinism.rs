use polyfish::actions::structure::capture_ruin;
use polyfish::actions::tech::unlock_tech;
use polyfish::coords::Coords;
use polyfish::game::Game;
use polyfish::states::{GameState, StructureState, TribeState, UnitState};
use polyfish::types::{StructureType, TechnologyType, TribeType, UnitType};

#[test]
fn test_capture_ruin_determinism() {
    // Setup a base state
    let mut base_state = GameState::default();
    base_state.settings.seed = 123456789;
    base_state.settings.current_player_turn_id = 1;

    // Add a tribe - Xin-Xi (ID 2 in log?)
    // Log said "Tribe: 2". In polyfish usually ID matches index or something.
    // Let's assume ID 2.
    let mut tribe = TribeState::default();
    tribe.id = 2;
    tribe.tribe_type = TribeType::XinXi;
    tribe.stars = 20;
    base_state.tribes.insert(2, tribe);
    base_state.settings.current_player_turn_id = 2;

    // Add a ruin at (0,0) -> index 0
    let ruin = StructureState {
        structure_type: StructureType::Ruin,
        ..Default::default()
    };
    base_state.structures.insert(0, Some(ruin));

    // Add a unit on the ruin
    let unit = UnitState {
        owner: 2,
        unit_type: UnitType::Warrior,
        coords: Coords::from_index(0, 11),
        ..Default::default()
    };

    // Initialize Game
    let mut base_game = Game { state: base_state };
    base_game.post_load();

    // Add unit to tribe (after post_load to avoid index issues if any?)
    if let Some(t) = base_game.state.tribes.get_mut(&1) {
        t.units.push(unit);
    }

    // Setup iterations
    for i in 0..100 {
        let seed = 50000 + i as u64;
        base_game.state.settings.seed = seed as i64;

        // Clone two identical games using production clone (JSON cycle)
        let mut game_a = base_game.clone();
        let mut game_b = base_game.clone();

        // --- Step 1: Research Tech ---
        let _ = unlock_tech(&mut game_a.state, TechnologyType::Organization, false);
        let _ = unlock_tech(&mut game_b.state, TechnologyType::Organization, false);

        // --- Step 2: Capture Ruin ---
        let _undo_a = capture_ruin(&mut game_a.state, 0, None, None, None);
        let _undo_b = capture_ruin(&mut game_b.state, 0, None, None, None);

        // Compare explicit outcomes
        let stars_a = game_a.state.tribes.get(&1).unwrap().stars;
        let stars_b = game_b.state.tribes.get(&1).unwrap().stars;

        // Check exact match of stars
        assert_eq!(
            stars_a, stars_b,
            "Stars diverged at iter {}: A={} B={}",
            i, stars_a, stars_b
        );

        // Check messages (indicates what reward was picked: Tech vs Stars vs Pop)
        let msgs_a = &game_a.state._messages;
        let msgs_b = &game_b.state._messages;
        assert_eq!(
            msgs_a, msgs_b,
            "Messages diverged at iter {}: \nA={:?}\nB={:?}",
            i, msgs_a, msgs_b
        );
    }
}
