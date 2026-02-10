use clap::Parser;
use polyfish::ai::evaluator;
use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::moves::Move;
use polyfish::recorder::GameRecorder;
use polyfish::types::{MapSize, MapType, MoveType, TribeType};
use std::io::{self, Write};

/// Interactive Trainer: Play against AI and correct its moves
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// MCTS Iterations
    #[arg(long, default_value_t = 1000)]
    mcts: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let recorder = GameRecorder::new();

    println!("=== Polyfish Interactive Trainer ===");
    println!("You are commanding the Imperius (P1). AI is Imperius (P2).");
    println!("Goal: Play the game. When AI moves, you can CORRECT it.");

    // Setup Game
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.post_load();

    // Heuristic Agent for P2
    let agent = HeuristicMctsAgent::new(args.mcts);

    while !game.state.settings._game_over {
        let pid = game.state.settings.current_player_turn_id;
        print_game_status(&game);

        if pid == 1 {
            // Human Turn
            handle_human_turn(&mut game, &recorder)?;
        } else {
            // AI Turn
            handle_ai_turn(&mut game, &agent, &recorder)?;
        }

        // Save periodically?
        // Let's save on exit or game over
    }

    println!("Game Over!");
    recorder.save()?;
    Ok(())
}

fn print_game_status(game: &Game) {
    let pid = game.state.settings.current_player_turn_id;
    let turn = game.state.settings.turn;
    let tribe = game.state.tribes.get(&pid).unwrap();
    println!(
        "\n--- Turn {} | Player {} (Stars: {}) ---",
        turn, pid, tribe.stars
    );
}

fn handle_human_turn(game: &mut Game, recorder: &GameRecorder) -> anyhow::Result<()> {
    // 1. List Legal Moves
    let moves = game.legal_moves();
    if moves.is_empty() {
        println!("No legal moves! Skipping turn (should auto-end).");
        return Ok(());
    }

    // 2. Display Moves (Simplified)
    for (i, m) in moves.iter().enumerate() {
        // Try to get src/target from JSON for display
        // Ideally we'd have a nice Display trait
        let json = m.serialize();
        let name = m.move_type();
        // let src = json["src"].as_i64().unwrap_or(-1);
        println!("{}: {:?} {:?}", i, name, json);
    }

    // 3. Prompt
    loop {
        print!("> Select move (index): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if let Ok(idx) = input.trim().parse::<usize>() {
            if idx < moves.len() {
                let m = &moves[idx];

                // Record
                let eco = evaluator::economy::evaluate_economy(&game.state, 1);
                let mil = evaluator::army::evaluate_army(&game.state, 1);
                recorder.record_step(&game.state, m.as_ref(), eco, mil);

                println!("Executing: {:?}", m.move_type());
                game.play_move(m.as_ref());
                return Ok(());
            }
        }
        println!("Invalid index.");
    }
}

fn handle_ai_turn(
    game: &mut Game,
    agent: &HeuristicMctsAgent,
    recorder: &GameRecorder,
) -> anyhow::Result<()> {
    println!("AI is thinking...");
    let (best_move, _) = agent.select_move_with_analysis(game);

    let chosen_move = match best_move {
        Some(m) => m,
        None => {
            // Should not happen unless no moves
            return Ok(());
        }
    };

    println!(
        "\nAI Proposes: {:?} {:?}",
        chosen_move.move_type(),
        chosen_move.serialize()
    );

    // Correction Loop
    loop {
        print!("Accept this move? (y/n): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let choice = input.trim().to_lowercase();

        if choice == "y" || choice == "" {
            // Accept
            let eco = evaluator::economy::evaluate_economy(&game.state, 2);
            // Note: recording from P2's perspective
            let mil = evaluator::army::evaluate_army(&game.state, 2);
            recorder.record_step(&game.state, chosen_move.as_ref(), eco, mil);

            game.play_move(chosen_move.as_ref());
            break;
        } else if choice == "n" {
            // Reject - Ask Human to select better move for AI
            println!("Override mode: Select the BEST move for AI:");
            let moves = game.legal_moves();
            for (i, m) in moves.iter().enumerate() {
                println!("{}: {:?} {:?}", i, m.move_type(), m.serialize());
            }

            print!("> Select CORRECT move (index): ");
            io::stdout().flush()?;
            let mut corr_input = String::new();
            io::stdin().read_line(&mut corr_input)?;

            if let Ok(idx) = corr_input.trim().parse::<usize>() {
                if idx < moves.len() {
                    let m = &moves[idx];
                    // Record Correction
                    let eco = evaluator::economy::evaluate_economy(&game.state, 2);
                    let mil = evaluator::army::evaluate_army(&game.state, 2);
                    recorder.record_step(&game.state, m.as_ref(), eco, mil);

                    println!("Executing Correction: {:?}", m.move_type());
                    game.play_move(m.as_ref());
                    break;
                }
            }
            println!("Invalid index, try again.");
        }
    }

    Ok(())
}
