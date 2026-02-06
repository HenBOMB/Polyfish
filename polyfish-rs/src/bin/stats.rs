use clap::Parser;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, MoveType, TribeType};
use rand::seq::SliceRandom;

/// Stats: Calculate average moves available and played (Random Agent)
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to play
    #[arg(long, default_value_t = 10000)]
    games: usize,

    /// Max turns per game
    #[arg(long, default_value_t = 30)]
    max_turns: i32,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!(
        "Starting Stats Collection (Random Agent): {} games, max {} turns...",
        args.games, args.max_turns
    );

    let mut total_turns_played = 0;
    // Total moves actually played by actions (excluding end turn)
    let mut total_moves_played = 0;
    let mut total_available_sum = 0;
    let mut total_steps_count = 0; // Opportunities to move

    // Per-turn stats
    let mut moves_per_turn_sum = 0;
    let mut current_turn_moves = 0;
    let mut current_turn_id = 0;
    let mut max_available_moves = 0;

    // Turn -> (Sum, Count, Max)
    // We use a tuple (Sum, Count, Max)
    let mut turn_branching_stats: std::collections::BTreeMap<i32, (u64, u64, usize)> =
        std::collections::BTreeMap::new();

    let mut rng = rand::thread_rng();

    for _i in 0..args.games {
        let gen_settings = MapGenSettings {
            size: MapSize::Tiny, // Simple 1v1
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Bardur, TribeType::Imperius],
            seed: rand::random(),
            ..Default::default()
        };

        let mut game = Game::new();
        game.state = generate(gen_settings);
        game.state.settings.mode = ModeType::Perfection;
        game.state.settings.max_turns = args.max_turns;
        game.post_load();

        // Disable console logs for speed if possible, but we can't easily here.

        current_turn_id = game.state.settings.turn;
        current_turn_moves = 0;

        let mut game_steps = 0;

        while !polyfish::functions::is_game_over(&game.state) && game.state.settings.turn <= 1000000
        {
            // Check turn change
            if game.state.settings.turn != current_turn_id {
                moves_per_turn_sum += current_turn_moves;
                total_turns_played += 1;

                current_turn_id = game.state.settings.turn;
                current_turn_moves = 0;
            }

            let legal_moves = game.legal_moves();
            if legal_moves.is_empty() {
                break;
            }

            // Record stats
            let available_count = legal_moves.len();
            total_available_sum += available_count;
            total_steps_count += 1;
            if available_count > max_available_moves {
                max_available_moves = available_count;
            }

            // Per-turn stats
            // We use a tuple (Sum, Count, Max)
            let entry = turn_branching_stats
                .entry(game.state.settings.turn)
                .or_insert((0, 0, 0));
            entry.0 += available_count as u64;
            entry.1 += 1;
            if available_count > entry.2 {
                entry.2 = available_count;
            }

            // Pick RANDOM move
            let random_move = legal_moves.choose(&mut rng).unwrap();
            let is_end_turn = random_move.move_type() == MoveType::EndTurn;

            game.play_move(random_move.as_ref());
            total_moves_played += 1;

            if !is_end_turn {
                current_turn_moves += 1;
            }

            game_steps += 1;
            // Safety break to prevent infinite loops in weird states
            if game_steps > 3000 {
                break;
            }
        }
        // Count the last partial turn if game ended
        if current_turn_moves > 0 {
            moves_per_turn_sum += current_turn_moves;
            total_turns_played += 1;
        }
    }

    println!("\n=== RANDOM AGENT STATISTICS ===");
    println!("Games Played:             {}", args.games);
    println!("Total Turns Tracked:      {}", total_turns_played);
    println!("Total Steps (Decisions):  {}", total_steps_count);
    println!("Total Moves Played:       {}", total_moves_played);
    println!("-------------------------");

    if total_steps_count > 0 {
        let avg_available = total_available_sum as f64 / total_steps_count as f64;
        println!(
            "Avg Moves AVAILABLE:      {:.2} (per decision step)",
            avg_available
        );
        println!(
            "Max Moves AVAILABLE:      {} (Max Branching Factor)",
            max_available_moves
        );
    }

    if total_turns_played > 0 {
        let avg_played = moves_per_turn_sum as f64 / total_turns_played as f64;
        println!("Avg Moves PLAYED:         {:.2} (per turn)", avg_played);
    }

    println!("\n=== BRANCHING FACTOR PER TURN (Min / Avg / Max) ===");
    println!("{:<6} | {:<20} | {:<10}", "Turn", "Avg", "Max");
    println!("{:-<6}-|-{:-<10}-|-{:-<20}-|-{:-<10}", "", "", "", "");

    for (turn, (sum, count, max)) in turn_branching_stats {
        if count > 0 && turn <= 30 {
            // Limit output to first 30 turns
            let avg = sum as f64 / count as f64;
            println!("{:<6} | {:<20.2} | {:<10}", turn, avg, max);
        }
    }

    Ok(())
}
