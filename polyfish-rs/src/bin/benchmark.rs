use clap::Parser;
use polyfish::ai::genes::AIGenes;
use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, TribeType};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the candidate genes JSON
    #[arg(long)]
    genes: String,

    /// Number of matches to play
    #[arg(long, default_value_t = 20)]
    matches: usize,

    /// MCTS iterations per move
    #[arg(long, default_value_t = 50)]
    mcts: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=== Polyfish AI Benchmark ===");
    println!("Candidate: {}", args.genes);
    println!("Matches: {}, MCTS Iters: {}", args.matches, args.mcts);

    let candidate_genes = AIGenes::load(&args.genes)?;
    let baseline_genes = AIGenes::default();

    let mut candidate_wins = 0;
    let mut baseline_wins = 0;
    let mut draws = 0;
    let mut candidate_total_score = 0;
    let mut baseline_total_score = 0;

    for i in 0..args.matches {
        // Swap sides every match
        let (p1_genes, p2_genes) = if i % 2 == 0 {
            (&candidate_genes, &baseline_genes)
        } else {
            (&baseline_genes, &candidate_genes)
        };

        let seed = 1000 + i as u64;
        let (s1, s2) = play_match(p1_genes, p2_genes, args.mcts, seed);

        let (c_score, b_score) = if i % 2 == 0 { (s1, s2) } else { (s2, s1) };
        candidate_total_score += c_score;
        baseline_total_score += b_score;

        if c_score > b_score {
            candidate_wins += 1;
            print!("W");
        } else if b_score > c_score {
            baseline_wins += 1;
            print!("L");
        } else {
            draws += 1;
            print!("D");
        }

        use std::io::{self, Write};
        io::stdout().flush().unwrap();
    }

    println!("\n\n--- Results ---");
    println!(
        "Candidate Wins: {} ({:.1}%)",
        candidate_wins,
        (candidate_wins as f32 / args.matches as f32) * 100.0
    );
    println!(
        "Baseline Wins:  {} ({:.1}%)",
        baseline_wins,
        (baseline_wins as f32 / args.matches as f32) * 100.0
    );
    println!("Draws:          {}", draws);
    println!(
        "Avg Score (Candidate): {:.0}",
        candidate_total_score as f32 / args.matches as f32
    );
    println!(
        "Avg Score (Baseline):  {:.0}",
        baseline_total_score as f32 / args.matches as f32
    );

    let improvement = (candidate_total_score as f32 / baseline_total_score as f32 - 1.0) * 100.0;
    println!("Raw Score Improvement: {:.2}%", improvement);

    Ok(())
}

fn play_match(genes1: &AIGenes, genes2: &AIGenes, mcts_iters: usize, seed: u64) -> (i32, i32) {
    let settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: seed,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(settings);
    game.post_load();

    let agent1 = HeuristicMctsAgent::with_genes(mcts_iters, genes1.clone());
    let agent2 = HeuristicMctsAgent::with_genes(mcts_iters, genes2.clone());

    let mut turn_limit = 200;

    while !game.state.settings._game_over && turn_limit > 0 {
        let pid = game.state.settings.current_player_turn_id;
        let agent = if pid == 1 { &agent1 } else { &agent2 };

        if let (Some(m), _) = agent.select_move_with_analysis(&mut game) {
            game.play_move(m.as_ref());
        } else {
            break;
        }

        turn_limit -= 1;
    }

    let s1 = game.state.tribes.get(&1).map(|t| t.score).unwrap_or(0);
    let s2 = game.state.tribes.get(&2).map(|t| t.score).unwrap_or(0);

    (s1, s2)
}
