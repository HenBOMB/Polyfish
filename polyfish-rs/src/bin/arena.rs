use clap::Parser;
use polyfish::ai::genes::AIGenes;
use polyfish::ai::heuristic_mcts::HeuristicMctsAgent;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, TribeType};
use rand::Rng;
use rayon::prelude::*;
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Population size
    #[arg(long, default_value_t = 20)]
    pop_size: usize,

    /// Number of generations
    #[arg(long, default_value_t = 10)]
    gens: usize,

    /// MCTS iterations per move
    #[arg(long, default_value_t = 100)]
    mcts: usize,

    /// Matches per pair in tournament
    #[arg(long, default_value_t = 2)]
    matches: usize,

    /// Mutation rate (0.0 to 1.0)
    #[arg(long, default_value_t = 0.1)]
    mutation_rate: f32,

    /// Elite count (keep top N winners)
    #[arg(long, default_value_t = 4)]
    elites: usize,

    /// Output directory for genes
    #[arg(long, default_value = "evolution")]
    output: String,

    /// Load existing genes from file to start population
    #[arg(long)]
    load: Option<String>,
}

#[derive(Clone)]
struct Candidate {
    id: usize,
    genes: AIGenes,
    fitness: f32,
    wins: usize,
    total_score: f32,
    games_played: usize,
}

impl Candidate {
    fn new(id: usize, genes: AIGenes) -> Self {
        Self {
            id,
            genes,
            fitness: 0.0,
            wins: 0,
            total_score: 0.0,
            games_played: 0,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output)?;

    println!("=== Polyfish Evolutionary Arena ===");
    println!("Population: {}, Generations: {}", args.pop_size, args.gens);
    println!("MCTS Iters: {}, Elites: {}", args.mcts, args.elites);
    if let Some(path) = &args.load {
        println!("Resuming from: {}", path);
    }

    // 1. Initialize Population
    let mut population: Vec<Candidate> = Vec::new();

    // Load base genes and determine start generation
    let mut start_gen = 0;
    let base_genes = if let Some(path) = &args.load {
        // Try to parse generation number from filename "gen_X..."
        if let Some(file_name) = Path::new(path).file_name().and_then(|n| n.to_str()) {
            if file_name.starts_with("gen_") {
                if let Some(idx_str) = file_name.split('_').nth(1) {
                    start_gen = idx_str.parse::<usize>().unwrap_or(0) + 1;
                }
            }
        }
        AIGenes::load(path)
            .map_err(|e| anyhow::anyhow!("Failed to load genes from {}: {}", path, e))?
    } else {
        AIGenes::default()
    };

    // Start with base genes as the first candidate
    population.push(Candidate::new(0, base_genes.clone()));

    // Fill the rest with mutated base genes
    let initial_mutation = if args.load.is_some() { 0.2 } else { 0.5 };
    for i in 1..args.pop_size {
        let genes = base_genes.mutate(initial_mutation);
        population.push(Candidate::new(i, genes));
    }

    let mut last_best_fitness = 0.0;
    let mut global_stagnation = 0;

    for current_gen_offset in 0..args.gens {
        let gen_idx = start_gen + current_gen_offset;
        println!("\n--- Generation {} ---", gen_idx);

        // 2. Run Tournament
        run_tournament(&mut population, &args);

        // Sort by fitness
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Report Best
        let best = &population[0];
        println!(
            "Best Candidate: ID={}, Fitness={:.4}, Wins={}, AvgScore={:.2}",
            best.id,
            best.fitness,
            best.wins,
            best.total_score / best.games_played as f32
        );

        // Stagnation Detection
        if best.fitness <= last_best_fitness + 0.0001 {
            global_stagnation += 1;
            println!(
                "⚠️ Stagnation detected! ({} generations)",
                global_stagnation
            );
        } else {
            global_stagnation = 0;
            last_best_fitness = best.fitness;
        }

        // Save Best Genes (including fitness in filename for easy sorting/cleanup)
        let best_path =
            Path::new(&args.output).join(format!("gen_{}_fit_{:.4}.json", gen_idx, best.fitness));
        let json = serde_json::to_string_pretty(&best.genes)?;
        fs::write(&best_path, json)?;

        // Cleanup: Only keep top 5% + latest
        if let Err(e) = cleanup_results(&args.output, args.gens, gen_idx) {
            eprintln!("Warning: Failed to cleanup old results: {}", e);
        }

        if current_gen_offset == args.gens - 1 {
            break;
        }

        // 3. Evolve
        let mut next_population = Vec::new();

        // Elitism
        for i in 0..args.elites {
            next_population.push(Candidate::new(i, population[i].genes.clone()));
        }

        // Fill remaining with Crossover and Mutation
        // More gradual spike: starts after 5 gens of stagnation, maxes at 3x
        let mutation_spike = if global_stagnation > 5 {
            (1.0 + ((global_stagnation - 5) as f32 * 0.2)).min(3.0)
        } else {
            1.0
        };

        let current_mutation_rate = args.mutation_rate * mutation_spike;
        if mutation_spike > 1.0 {
            println!(
                "🔥 Spiking mutation to {:.2}% (Stagnation: {})",
                current_mutation_rate * 100.0,
                global_stagnation
            );
        }

        while next_population.len() < args.pop_size {
            // Stranger Injection: 5% chance to just add a fresh random candidate
            use rand::Rng;
            if next_population.len() < args.pop_size - 1 && rand::thread_rng().gen_bool(0.05) {
                let genes = AIGenes::default().mutate(0.8);
                next_population.push(Candidate::new(next_population.len(), genes));
                continue;
            }

            let p1 = select_parent(&population);
            let p2 = select_parent(&population);

            let child_genes =
                AIGenes::crossover(&p1.genes, &p2.genes).mutate(current_mutation_rate);

            next_population.push(Candidate::new(next_population.len(), child_genes));
        }

        population = next_population;
    }

    println!("\nEvolution Complete!");
    Ok(())
}

fn select_parent(population: &[Candidate]) -> &Candidate {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let i = rng.gen_range(0..population.len());
    let j = rng.gen_range(0..population.len());
    if population[i].fitness > population[j].fitness {
        &population[i]
    } else {
        &population[j]
    }
}

fn run_tournament(population: &mut [Candidate], args: &Args) {
    let pop_len = population.len();

    let matches_to_run: Vec<(usize, usize)> = (0..pop_len)
        .flat_map(|i| (i + 1..pop_len).map(move |j| (i, j)))
        .collect();

    println!("Running {} match pairings...", matches_to_run.len());

    // Use a base seed that changes every generation to prevent over-fitting
    let mut rng = rand::thread_rng();
    let gen_seed_base = rng.gen_range(0..1000) * 100;

    let results: Vec<(usize, usize, i32, i32)> = matches_to_run
        .into_par_iter()
        .flat_map(|(i, j)| {
            let mut pairing_results = Vec::new();
            for m in 0..args.matches {
                // Alternate who starts
                let (p1_idx, p2_idx) = if m % 2 == 0 { (i, j) } else { (j, i) };

                let g1 = population[p1_idx].genes.clone();
                let g2 = population[p2_idx].genes.clone();

                // Seed is generation-based
                let (score1, score2) = play_match(&g1, &g2, args.mcts, gen_seed_base + m as u64);

                if m % 2 == 0 {
                    pairing_results.push((i, j, score1, score2));
                } else {
                    pairing_results.push((i, j, score2, score1));
                }
            }
            pairing_results
        })
        .collect();

    // Reset scores
    for p in population.iter_mut() {
        p.wins = 0;
        p.total_score = 0.0;
        p.games_played = 0;
    }

    // Accumulate results
    for (i, j, score_i, score_j) in results {
        population[i].total_score += score_i as f32;
        population[j].total_score += score_j as f32;
        population[i].games_played += 1;
        population[j].games_played += 1;

        if score_i > score_j {
            population[i].wins += 1;
        } else if score_j > score_i {
            population[j].wins += 1;
        }
    }

    // Calculate fitness
    for p in population.iter_mut() {
        let win_rate = p.wins as f32 / p.games_played as f32;
        let avg_score = p.total_score / p.games_played as f32;

        // If they are scoring high (stagnated), prioritize win rate and efficient city leveling
        if avg_score > 2500.0 {
            p.fitness = (win_rate * 2.0) + (avg_score / 15000.0).clamp(0.0, 1.0);
        } else {
            p.fitness = win_rate + (avg_score / 8000.0).clamp(0.0, 1.0);
        }
    }
}

fn play_match(
    genes1: &AIGenes,
    genes2: &AIGenes,
    mcts_iters: usize,
    seed_offset: u64,
) -> (i32, i32) {
    let settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed: 42 + seed_offset,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(settings);
    game.post_load();

    let agent1 = HeuristicMctsAgent::with_genes(mcts_iters, genes1.clone());
    let agent2 = HeuristicMctsAgent::with_genes(mcts_iters, genes2.clone());

    let mut turn_limit = 1000; // Increased limit

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

fn cleanup_results(dir: &str, total_gens: usize, latest_idx: usize) -> anyhow::Result<()> {
    let limit = (total_gens as f32 * 0.05).ceil() as usize;
    let limit = limit.max(2); // Keep at least 2 (latest + best)

    let mut entries = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().into_string().unwrap();
        if name.starts_with("gen_") && name.contains("_fit_") {
            let parts: Vec<&str> = name.split('_').collect();
            if parts.len() >= 4 {
                let idx: usize = parts[1].parse().unwrap_or(0);
                let fit_str = parts[3].trim_end_matches(".json");
                let fit: f32 = fit_str.parse().unwrap_or(0.0);
                entries.push((idx, fit, entry.path()));
            }
        }
    }

    if entries.len() <= limit {
        return Ok(());
    }

    // Sort by fitness descending
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Determine which indices to keep:
    // 1. The latest one (for resuming)
    // 2. The top (limit-1) ones by fitness
    let mut to_keep = std::collections::HashSet::new();
    to_keep.insert(latest_idx);

    for i in 0..entries.len() {
        if to_keep.len() < limit {
            to_keep.insert(entries[i].0);
        }
    }

    // Delete files not in to_keep
    for (idx, _, path) in entries {
        if !to_keep.contains(&idx) {
            fs::remove_file(path)?;
        }
    }

    Ok(())
}
