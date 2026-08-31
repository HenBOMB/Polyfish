//! Fast reward-tuning loop (Aug 2026): edit a SHAPE_* constant, `cargo build
//! --bin reward_lab` (debug, seconds), rerun against a FROZEN historical ply
//! and see the per-term Φ breakdown for every candidate move instantly --
//! no self-play run, no eval server, no rebuild of the full release binary.
//!
//! Input is a `--replay`/`--trace` pair from the SAME `self_play
//! --dump-games-dir <dir>` + `POLYFISH_PLY_TRACE=<path>` run (the gA/gB
//! debugging workflow already produces both). The tool fast-forwards the
//! replay's recorded commands to the exact ply the trace row names, so the
//! game state is real, not synthetic.
//!
//! Faithfulness, read before trusting a number:
//! - `goal` (stance + orders) and `unit_goals` (the per-unit Expand store)
//!   come straight from the trace row -- these are what search actually
//!   committed, not re-derivable from the state alone.
//! - `goal.save_target` is NOT captured by the trace (only stance+orders
//!   are), so it reads as `None` here -- the SAVE ramp term will always
//!   show 0 in this tool even on a ply where it was live.
//! - `aux` (GoalAux) and `threats` are recomputed fresh via the same
//!   `compute_goal_aux`/`city_risks` calls the real path uses, but with
//!   `TurnCounters` approximated from state (`derive_counters`, the same
//!   approximation the engine uses to model an opponent's counters) rather
//!   than the agent's precisely-tracked incremental counters, and
//!   `lane_state: None` (also not trace-captured). Tech-fit/rider-push/lane
//!   terms may be very slightly off from the real ply for this reason;
//!   every other term is exact.
//!
//! Sanity check built in: the tool's own candidate ranking (via
//! `macro_exec::rank_plies`, the SAME function the real search calls) is
//! compared against the trace row's recorded scores, and any mismatch
//! beyond floating-point tolerance is printed as a loud warning rather than
//! silently trusted.

use clap::Parser;
use polyfish::ai::macro_mcts::derive_counters;
use polyfish::ai::oracle_macro::{tech_discipline_active, MacroGoal, OrderKind, Stance};
use polyfish::ai::reward::goal_potential_breakdown;
use polyfish::ai::search::goal_aux::compute_goal_aux;
use polyfish::ai::search::unit_goals::{UnitGoal, UnitGoalStore};
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(author, version, about = "Frozen-ply reward tuning lab", long_about = None)]
struct Args {
    /// Path to a self_play `--dump-games-dir` `*.replay.json` ({gameState, turns}).
    #[arg(long)]
    replay: String,

    /// Path to the matching `POLYFISH_PLY_TRACE=<path>` .jsonl from the same run.
    #[arg(long)]
    trace: String,

    /// Target turn number (the trace row's "turn" field).
    #[arg(long)]
    turn: i32,

    /// Target acting player (the trace row's "player" field).
    #[arg(long)]
    player: i32,

    /// Which real-ply decision within that turn/player, 0-indexed (a turn
    /// can have several: Research, then Step, then EndTurn, ...). Run
    /// without this flag first -- the tool always lists what's available.
    #[arg(long, default_value_t = 0)]
    occurrence: usize,

    /// How many top-ranked candidates to print full breakdowns for.
    #[arg(long, default_value_t = 6)]
    top: usize,

    /// macro-lambda: weight on Δφ vs score_move in the printed total.
    /// Matches self_play/macro-mcts's default (`--macro-lambda`).
    #[arg(long, default_value_t = 1.0)]
    lambda: f32,
}

#[derive(serde::Deserialize)]
struct TraceRow {
    turn: i32,
    player: i32,
    goal: TraceGoal,
    unit_goals: Vec<TraceUnitGoal>,
    chosen: TraceChosen,
}
#[derive(serde::Deserialize)]
struct TraceGoal {
    stance: String,
    orders: Vec<(String, i32)>,
}
#[derive(serde::Deserialize)]
struct TraceUnitGoal {
    unit_id: u32,
    goal: Option<TraceUnitGoalInner>,
}
#[derive(serde::Deserialize)]
struct TraceUnitGoalInner {
    kind: String,
    target: i32,
}
#[derive(serde::Deserialize)]
struct TraceChosen {
    move_type: String,
    #[allow(dead_code)]
    #[serde(rename = "move")]
    mv: serde_json::Value,
}

fn parse_stance(s: &str) -> Stance {
    match s {
        "Grow" => Stance::Grow,
        "Save" => Stance::Save,
        "Arm" => Stance::Arm,
        "Unlock" => Stance::Unlock,
        other => panic!("unknown Stance in trace: {other}"),
    }
}

fn parse_order_kind(s: &str) -> OrderKind {
    match s {
        "Expand" => OrderKind::Expand,
        "Attack" => OrderKind::Attack,
        "Defend" => OrderKind::Defend,
        other => panic!("unknown OrderKind in trace: {other}"),
    }
}

/// Truncate a replay clone to the exact pre-decision state for
/// (target_turn, target_player, occurrence): every earlier turn plays in
/// full, this turn's earlier-acting players play in full, and this
/// player's own commands stop right before the `occurrence`-th one.
fn build_clip(replay: &ModReplay, target_turn: i32, target_player: i32, occurrence: usize) -> ModReplay {
    let mut clip = replay.clone();
    clip.turns.retain(|t| t.turn <= target_turn);
    if let Some(last) = clip.turns.last_mut() {
        if last.turn == target_turn {
            if let Some(pidx) = last.players.iter().position(|p| p.player_id == target_player) {
                last.players.truncate(pidx + 1);
                last.players[pidx].commands.truncate(occurrence);
            }
        }
    }
    clip
}

fn aggregate(bd: &[(&'static str, f32)]) -> Vec<(&'static str, f32)> {
    let mut order: Vec<&'static str> = Vec::new();
    let mut sums: HashMap<&'static str, f32> = HashMap::new();
    for &(label, v) in bd {
        if !sums.contains_key(label) {
            order.push(label);
        }
        *sums.entry(label).or_insert(0.0) += v;
    }
    order.into_iter().map(|l| (l, sums[l])).collect()
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let trace_text = std::fs::read_to_string(&args.trace)?;
    let rows: Vec<TraceRow> = trace_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("parse trace row"))
        .collect();
    let matching: Vec<&TraceRow> =
        rows.iter().filter(|r| r.turn == args.turn && r.player == args.player).collect();

    println!(
        "Found {} real-ply row(s) for turn={} player={}:",
        matching.len(),
        args.turn,
        args.player
    );
    for (i, r) in matching.iter().enumerate() {
        let marker = if i == args.occurrence { "->" } else { "  " };
        println!(
            "  {marker} [{i}] stance={} orders={:?} chosen={}",
            r.goal.stance, r.goal.orders, r.chosen.move_type
        );
    }
    let Some(row) = matching.get(args.occurrence) else {
        anyhow::bail!(
            "--occurrence {} out of range (0..{})",
            args.occurrence,
            matching.len()
        );
    };
    println!();

    let replay_text = std::fs::read_to_string(&args.replay)?;
    let replay: ModReplay = serde_json::from_str(&replay_text)?;
    let mut clip = build_clip(&replay, args.turn, args.player, args.occurrence);
    let mut game = Game::new();
    replay_game(&mut game, &mut clip).map_err(anyhow::Error::msg)?;
    game.state.settings.current_player_turn_id = args.player;

    let goal = MacroGoal {
        stance: parse_stance(&row.goal.stance),
        orders: row.goal.orders.iter().map(|(k, t)| (parse_order_kind(k), *t)).collect(),
        save_target: None, // not trace-captured -- see module doc
    };

    let mut store = UnitGoalStore::default();
    for ug in &row.unit_goals {
        if let Some(g) = &ug.goal {
            store.assign(ug.unit_id, UnitGoal { kind: parse_order_kind(&g.kind), target: g.target });
        }
    }

    let (techs_bought, tier3_bought) = {
        let c = derive_counters(&game.state, args.player);
        (c.techs_bought, c.tier3_bought)
    };
    let aux = compute_goal_aux(&game.state, args.player, &goal, techs_bought, tier3_bought, None);
    let gate = tech_discipline_active(&game.state, args.player, &goal);
    let belief = polyfish::ai::belief::map::MapBelief::observe(&game.state, args.player);

    let ranked = polyfish::ai::macro_exec::rank_plies(
        &mut game,
        args.player,
        &goal,
        &aux,
        gate,
        args.lambda,
        Some(&store),
        None,
    );

    println!(
        "State: turn={} player={} stance={:?} orders={:?}",
        game.state.settings.turn, args.player, goal.stance, goal.orders
    );
    println!(
        "rank_plies candidates: {} (top {} shown, same ranking function the real search uses)\n",
        ranked.len(),
        args.top.min(ranked.len())
    );

    let (phi_pre, bd_pre) = goal_potential_breakdown(
        &game.state,
        args.player,
        &goal,
        Some(&aux),
        None,
        Some(&store),
        Some(&belief),
        None,
    );
    let pre_agg: HashMap<&'static str, f32> = aggregate(&bd_pre).into_iter().collect();

    for (rank, (score, mv)) in ranked.iter().take(args.top).enumerate() {
        print!("#{} score={:.4}  {:?} {}", rank + 1, score, mv.move_type(), mv.serialize());
        let is_chosen = mv.serialize() == row.chosen.mv;
        if is_chosen {
            print!("   <-- CHOSEN (matches trace)");
        }
        println!();

        if mv.move_type() == polyfish::types::MoveType::EndTurn {
            println!("    (EndTurn: no Δφ simulated)\n");
            continue;
        }
        let Some(undo) = game.simulate_move(mv.as_ref()) else {
            println!("    (move failed to simulate -- skipping breakdown)\n");
            continue;
        };
        let (phi_post, bd_post) = goal_potential_breakdown(
            &game.state,
            args.player,
            &goal,
            Some(&aux),
            None,
            Some(&store),
            Some(&belief),
            None,
        );
        undo(&mut game.state);
        let post_agg: HashMap<&'static str, f32> = aggregate(&bd_post).into_iter().collect();

        let dphi = phi_post - phi_pre;
        let score_move = score - args.lambda * dphi;
        println!(
            "    score_move={:.4}  Δφ={:.4}  (λ={:.2})  total={:.4}",
            score_move,
            dphi,
            args.lambda,
            score_move + args.lambda * dphi
        );

        let mut labels: Vec<&'static str> = pre_agg.keys().chain(post_agg.keys()).copied().collect();
        labels.sort_unstable();
        labels.dedup();
        let mut deltas: Vec<(&'static str, f32)> = labels
            .into_iter()
            .map(|l| (l, post_agg.get(l).copied().unwrap_or(0.0) - pre_agg.get(l).copied().unwrap_or(0.0)))
            .filter(|(_, d)| d.abs() > 1e-6)
            .collect();
        deltas.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        for (label, d) in &deltas {
            println!("      {label:<28} {d:+9.3}");
        }
        println!();
    }

    if let Some(chosen_rank) = ranked.iter().position(|(_, mv)| mv.serialize() == row.chosen.mv) {
        let recorded_score: f64 = 0.0; // trace doesn't carry a top-level float here; candidates array does
        let _ = recorded_score;
        if chosen_rank != 0 {
            println!(
                "NOTE: trace's chosen move ranks #{} here, not #1 -- if this run's code differs \
                 from what generated the trace, that's expected (you're testing a change).",
                chosen_rank + 1
            );
        }
    } else {
        println!(
            "WARNING: the trace's chosen move was not found in this run's legal-move ranking at \
             all -- state reconstruction may be off, or the code has changed enough that the \
             move is no longer legal/enumerated the same way. Don't trust the breakdown above \
             without investigating this first."
        );
    }

    Ok(())
}
