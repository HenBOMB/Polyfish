use polyfish::TribeType;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType};

fn main() {
    println!("=== POLYTOPIA BRANCHING FACTOR TEST ===");
    println!("Game: 1v1 Small Map, Perfection Mode, 10 turns\n");
    println!(
        "{:<6} {:<8} {:<12} {:<15}",
        "Turn", "Player", "Legal Moves", "Cumulative"
    );

    let mut turn_count = 0;
    let mut total_total_moves = 0;
    let mut min_moves = 1;
    let mut max_moves = 0;

    for _ in 0..100 {
        let gen_settings = MapGenSettings {
            size: MapSize::Small,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Imperius],
            seed: 12345,
            ..Default::default()
        };

        let mut total_moves = 0;
        let mut game = Game::new();
        game.state = generate(gen_settings);
        game.state.settings.mode = ModeType::Perfection;
        game.state.settings.max_turns = 10;
        game.post_load();

        while !polyfish::functions::is_game_over(&game.state) && turn_count < 1000000 {
            let legal_moves = game.legal_moves();
            let current_player = game.state.settings.current_player_turn_id;
            let current_turn = game.state.settings.turn;
            let move_count = legal_moves.len();

            total_moves += move_count;

            if min_moves.min(move_count) < min_moves {
                min_moves = min_moves.min(move_count);
            }

            if max_moves.max(move_count) > max_moves {
                max_moves = max_moves.max(move_count);
            }

            println!(
                "{:<6} {:<8} {:<12} {:<15}",
                current_turn, current_player, move_count, total_moves
            );

            use rand::seq::SliceRandom;
            if let Some(m) = legal_moves.choose(&mut rand::thread_rng()) {
                let _ = game.play_move(m.as_ref());
                turn_count += 1;
            }
        }
        total_total_moves += total_moves;
    }

    println!("\n=== STATISTICS ===");
    println!("Total turns played: {}", turn_count);
    println!(
        "Average legal moves per turn: {:.1}",
        total_total_moves as f64 / turn_count as f64 / 100 as f64
    );
    println!("Min legal moves: {}", min_moves);
    println!("Max legal moves: {}", max_moves);
    println!("\nThis branching factor determines MCTS tree complexity!");
}
