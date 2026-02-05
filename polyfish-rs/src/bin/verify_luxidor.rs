fn main() {
    use polyfish::functions;
    use polyfish::game::Game;
    use polyfish::states::{CityState, TileState, TribeState, UnitState};
    use polyfish::types::{TerrainType, TribeType, UnitType};
    use std::collections::HashSet;

    use polyfish::mapgen::{MapGenSettings, generate};
    use polyfish::types::MapSize;

    let mut settings = MapGenSettings::default();
    settings.size = MapSize::Tiny; // Small map for speed
    settings.tribes = vec![TribeType::Luxidoor, TribeType::Imperius];
    settings.seed = 42; // Fixed seed

    // Generate game state
    let state = generate(settings);
    let mut game = Game { state };
    game.post_load(); // Ensures scores are synced

    let player_id = 1; // Luxidoor should be player 1 (index 0 in vec, but IDs usually 1-based?)
    // Need to find which ID is Luxidoor
    let lux_id = game
        .state
        .tribes
        .iter()
        .find(|(_, t)| t.tribe_type == TribeType::Luxidoor)
        .map(|(&id, _)| id)
        .unwrap();

    let score = game.state.tribes.get(&lux_id).unwrap().score;
    let pop = game.state.tribes.get(&lux_id).unwrap().cities[0].population;
    let level = game.state.tribes.get(&lux_id).unwrap().cities[0].level;

    println!("Luxidoor ID: {}", lux_id);
    println!("Score: {}", score);
    println!("City Level: {}", level);
    println!("City Pop: {}", pop);

    // Run calculation
    functions::sync_scores(&mut game.state);
    let score = game.state.tribes.get(&lux_id).unwrap().score;

    println!("Luxidoor Score: {}", score);
}
