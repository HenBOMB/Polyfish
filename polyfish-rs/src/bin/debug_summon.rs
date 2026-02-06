use polyfish::functions;
use polyfish::game::Game;
use polyfish::mapgen::{self, MapGenSettings};
use polyfish::types::MoveType;

fn main() {
    println!("Starting Summon check...");

    let settings = MapGenSettings::default();
    let state = mapgen::generate(settings);
    let mut game = Game { state };
    game.post_load();

    let pid = game.state.settings.current_player_turn_id;
    if let Some(tribe) = game.state.tribes.get_mut(&pid) {
        tribe.stars = 10;
        if let Some(city) = tribe.cities.first_mut() {
            city.level = 5;
            println!("City Found at {}, Level {}", city.tile_index, city.level);
        } else {
            println!("No cities found!");
        }
        tribe.units.clear();
        println!("Units cleared. Count: {}", tribe.units.len());
    }

    // Manual checks mimick generate_summon_moves
    if let Some(tribe) = game.state.tribes.get(&pid) {
        println!("Stars: {}", tribe.stars);
        for city in &tribe.cities {
            let idx = city.tile_index;
            let count = functions::get_city_unit_count(&game.state, city);
            let occupied = functions::is_tile_occupied(&game.state, idx);
            println!(
                "City @ {}: Count={}, Level={}, Occupied={}",
                idx, count, city.level, occupied
            );
        }
    }

    let moves = game.legal_moves();
    let summon_moves = moves
        .iter()
        .filter(|m| m.move_type() == MoveType::Summon)
        .count();
    println!("Found {} summon moves.", summon_moves);
}
