//! Measures how well the scoreboard predicts the Domination winner (audit A2b).
//!
//! `ai/reward.rs` builds every TD value label from `TribeState::score`, but
//! training plays Domination, where the winner is the last tribe alive. This
//! tool quantifies how good a proxy score actually is, and compares it against
//! alternative signals that a win-relevant label could use instead.
//!
//! Games decided on score (timeouts) are excluded from the accuracy figures —
//! score trivially predicts a winner it defined.

use clap::Parser;
use polyfish::ai::heuristic_mcts::GreedyHeuristicAgent;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType, TribeType};
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of games to play.
    #[arg(long, default_value_t = 100)]
    games: usize,

    /// Turn cap, matching the self-play curriculum's late stage.
    #[arg(long, default_value_t = 45)]
    max_turns: i32,

    /// Base seed; game i uses seed + i. Fixed by default so runs are comparable.
    #[arg(long, default_value_t = 90_210)]
    seed: i64,

    /// 2 = Domination (what run_training_loop.sh trains on).
    #[arg(long, default_value_t = 2)]
    gamemode: u8,

    /// Hard move cap per game, mirroring arena's.
    #[arg(long, default_value_t = 500)]
    max_moves: usize,
}

/// One per-turn observation of both seats.
#[derive(Clone, Copy)]
struct Snapshot {
    turn: i32,
    score: [i32; 2],
    cities: [i32; 2],
    units: [i32; 2],
}

struct GameOutcome {
    /// Seat that won: 1 or 2. `None` if the game was a draw.
    winner: Option<i32>,
    /// True when the game ended by elimination rather than the turn cap.
    decisive: bool,
    snapshots: Vec<Snapshot>,
}

fn snapshot(game: &Game) -> Snapshot {
    let get = |pid: i32| {
        game.state
            .tribes
            .get(&pid)
            .map(|t| (t.score, t.cities.len() as i32, t.units.len() as i32))
            .unwrap_or((0, 0, 0))
    };
    let (s1, c1, u1) = get(1);
    let (s2, c2, u2) = get(2);
    Snapshot {
        turn: game.state.settings.turn,
        score: [s1, s2],
        cities: [c1, c2],
        units: [u1, u2],
    }
}

fn play_game(seed: i64, max_turns: i32, gamemode: u8, max_moves: usize) -> GameOutcome {
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    game.state.settings.mode = ModeType::from_repr(gamemode).unwrap_or(ModeType::Domination);
    game.state.settings.max_turns = max_turns;
    game.post_load();

    let agent = GreedyHeuristicAgent::new();
    let mut snapshots = vec![snapshot(&game)];
    let mut last_turn = game.state.settings.turn;
    let mut moves = 0usize;

    while !polyfish::functions::is_game_over(&game.state) && moves < max_moves {
        let Some(mv) = agent.select_move(&mut game) else {
            break;
        };
        game.play_move(mv.as_ref());
        if game.state.settings.turn != last_turn {
            last_turn = game.state.settings.turn;
            snapshots.push(snapshot(&game));
        }
        moves += 1;
    }
    snapshots.push(snapshot(&game));

    let alive = |pid: i32| {
        game.state
            .tribes
            .get(&pid)
            .map(|t| t.killed_turn <= 0 && t.resigned_turn <= 0)
            .unwrap_or(false)
    };
    let (a1, a2) = (alive(1), alive(2));
    let last = *snapshots.last().unwrap();

    // Elimination beats score adjudication, mirroring self_play's Domination rule.
    let decisive = a1 != a2;
    let winner = if decisive {
        Some(if a1 { 1 } else { 2 })
    } else if last.score[0] > last.score[1] {
        Some(1)
    } else if last.score[1] > last.score[0] {
        Some(2)
    } else {
        None
    };

    GameOutcome {
        winner,
        decisive,
        snapshots,
    }
}

/// Accuracy of "seat with the larger value at `turn` wins", over `games`.
/// Returns `(accuracy, n_compared)`; ties count as half credit.
fn accuracy_at(games: &[&GameOutcome], turn: i32, pick: fn(&Snapshot) -> [i32; 2]) -> (f64, usize) {
    let mut correct = 0.0;
    let mut n = 0usize;
    for g in games {
        let Some(winner) = g.winner else { continue };
        let Some(snap) = g.snapshots.iter().rev().find(|s| s.turn <= turn) else {
            continue;
        };
        if snap.turn < turn && g.snapshots.iter().all(|s| s.turn < turn) {
            continue; // game ended before this turn — nothing to predict from
        }
        let v = pick(snap);
        n += 1;
        correct += if v[0] == v[1] {
            0.5
        } else {
            let predicted = if v[0] > v[1] { 1 } else { 2 };
            if predicted == winner { 1.0 } else { 0.0 }
        };
    }
    if n == 0 { (f64::NAN, 0) } else { (correct / n as f64, n) }
}

fn main() {
    let args = Args::parse();

    println!(
        "Playing {} greedy-vs-greedy games (mode {}, max_turns {}, seeds {}..{})",
        args.games,
        args.gamemode,
        args.max_turns,
        args.seed,
        args.seed + args.games as i64 - 1
    );

    let outcomes: Vec<GameOutcome> = (0..args.games)
        .into_par_iter()
        .map(|i| {
            play_game(
                args.seed + i as i64,
                args.max_turns,
                args.gamemode,
                args.max_moves,
            )
        })
        .collect();

    let decisive: Vec<&GameOutcome> = outcomes.iter().filter(|g| g.decisive).collect();
    let timeouts = outcomes.len() - decisive.len();
    let draws = outcomes.iter().filter(|g| g.winner.is_none()).count();

    println!(
        "\n{} games: {} decided by elimination, {} by turn cap, {} draws",
        outcomes.len(),
        decisive.len(),
        timeouts,
        draws
    );

    if decisive.is_empty() {
        println!(
            "\nNo game ended by elimination, so score-vs-outcome cannot be measured:\n\
             every winner here was *defined* by score. That is itself the finding —\n\
             at these settings the Domination win condition is never reached, so the\n\
             value label and the win condition are the same quantity by default.\n\
             Re-run with a larger --max-turns or --max-moves."
        );
        return;
    }

    println!(
        "\nAccuracy of 'leader at turn N wins', over the {} eliminations only.\n\
         Score is what ai/reward.rs feeds the TD label; the others are candidate\n\
         replacements. 0.50 = coin flip.\n",
        decisive.len()
    );
    println!("{:>5}  {:>8}  {:>8}  {:>8}  {:>6}", "turn", "score", "cities", "units", "n");

    let max_turn = decisive
        .iter()
        .flat_map(|g| g.snapshots.iter().map(|s| s.turn))
        .max()
        .unwrap_or(0);
    for turn in (0..=max_turn).step_by(3) {
        let (s, n) = accuracy_at(&decisive, turn, |s| s.score);
        let (c, _) = accuracy_at(&decisive, turn, |s| s.cities);
        let (u, _) = accuracy_at(&decisive, turn, |s| s.units);
        if n == 0 {
            continue;
        }
        println!("{turn:>5}  {s:>8.3}  {c:>8.3}  {u:>8.3}  {n:>6}");
    }

    println!(
        "\nRead: if the score column sits near 0.50 through the turns the agent is\n\
         actually deciding in, then the quantity every value label is built from\n\
         carries little information about winning, and a label built on a better\n\
         column would carry more."
    );
}
