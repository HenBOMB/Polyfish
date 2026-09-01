//! EXP_ELO_119: does micro-mcts's own within-turn lookahead already catch
//! the turn6/idx74 "researched Hunting while city41 sat open" decision --
//! and if not, is it EXP_ELO_079's already-diagnosed-but-unfixed
//! `softmax_priors` collapse (raw, un-temperatured `score_move` softmax
//! goes near one-hot whenever one candidate's score dominates, so PUCT's
//! exploration term can never pull visits toward any other child)?
//! Same faithful-executor pattern as `micro_depth_probe.rs`
//! (commit_macro_goal/StanceCommit/LaneState tracked turn-by-turn, not a
//! fresh recompute), applied to THIS game/ply instead of EXP_074's.
//! `Evaluator::Dummy` (constant leaf value) -- mechanics-only measurement
//! of the search's own structure, not a production-faithful decision read.

use polyfish::ai::eval_server::{DummyEvalHandle, Evaluator, InlineEvalHandle};
use polyfish::ai::network::PolyZeroNet;
use polyfish::ai::macro_exec::{rank_plies, TurnCounters};
use polyfish::ai::oracle_macro::{commit_macro_goal, tech_discipline_active, StanceCommit};
use polyfish::ai::search::goal_aux::compute_goal_aux;
use polyfish::ai::search::lane::{observe_lane_state, select_lane, LaneState};
use polyfish::ai::search::micro_mcts::{
    micro_search_pick, MicroParams, MICRO_MCTS_DEPTH_CALLS, MICRO_MCTS_DEPTH_SUM,
    MICRO_MCTS_MAX_DEPTH_SEEN,
};
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::types::TechnologyType;
use std::sync::atomic::Ordering;

const POV: i32 = 1;

fn state_at_p1_turn_start(full: &ModReplay, turn: i32) -> Game {
    let mut mr = full.clone();
    mr.turns.retain(|t| t.turn < turn);
    let mut g = Game::new();
    replay_game(&mut g, &mut mr).expect("deterministic replay failed");
    g
}

fn counters_before(full: &ModReplay, turn: i32) -> TurnCounters {
    let mut c = TurnCounters::default();
    for t in &full.turns {
        if t.turn >= turn {
            break;
        }
        for p in &t.players {
            if p.player_id != POV {
                continue;
            }
            for cmd in &p.commands {
                if cmd.get("moveType").and_then(|v| v.as_i64()) != Some(7) {
                    continue;
                }
                c.techs_bought += 1;
                if let Some(tv) = cmd.get("type") {
                    if let Ok(tech) = serde_json::from_value::<TechnologyType>(tv.clone()) {
                        if polyfish::settings::technology::get_technology_setting(tech).tier
                            == Some(3)
                        {
                            c.tier3_bought += 1;
                        }
                    }
                }
            }
        }
    }
    c
}

fn state_at_step(full: &ModReplay, target_idx: usize) -> Game {
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                if idx == target_idx {
                    return game;
                }
                let legal = game.legal_moves();
                let m = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} move not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    panic!("target_idx {target_idx} beyond game length {idx}");
}

/// Mirrors `micro_mcts::softmax_priors` exactly (private in that module) so
/// this probe can show what priors the real search actually computes for
/// the flagged ply's real root candidates.
fn softmax_priors(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        let n = scores.len() as f32;
        return vec![1.0 / n; scores.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

fn main() {
    let raw = std::fs::read_to_string(
        "replays/watch/game_iter100_game0_seed1787500020.replay.json",
    )
    .expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let target_idx = 74usize;
    let target_turn = 6i32;

    let mut sc = StanceCommit::default();
    let mut lane = LaneState::default();
    for turn in 0..target_turn {
        let g = state_at_p1_turn_start(&full, turn);
        let view0 = g.clone_for_mcts(POV);
        let counters = counters_before(&full, turn);
        let base = commit_macro_goal(&view0.state, POV, &mut sc, counters.tier3_bought);
        observe_lane_state(&view0.state, POV, &mut lane);
        select_lane(&view0.state, POV, &mut lane, None);
        let _ = base;
    }

    let counters = counters_before(&full, target_turn);
    let true_game = state_at_step(&full, target_idx);
    let mut view = true_game.clone_for_mcts(POV);
    observe_lane_state(&view.state, POV, &mut lane);
    select_lane(&view.state, POV, &mut lane, None);
    let goal = commit_macro_goal(&view.state, POV, &mut sc, counters.tier3_bought);
    let aux = compute_goal_aux(
        &view.state,
        POV,
        &goal,
        counters.techs_bought,
        counters.tier3_bought,
        Some(&lane),
    );
    let gate = tech_discipline_active(&view.state, POV, &goal);
    let mut eco_plan = polyfish::ai::eco_plan_commit::EcoPlanCommit::default();
    eco_plan.update(&view.state, POV);
    let ranked = rank_plies(&mut view, POV, &goal, &aux, gate, 1.0, None, Some(&eco_plan));

    println!("goal.orders = {:?}", goal.orders);
    println!("goal.stance = {:?}", goal.stance);
    println!("goal.save_target = {:?}", goal.save_target);
    println!(
        "=== idx={target_idx} turn={target_turn} stance={:?} candidates={} ===",
        goal.stance,
        ranked.len()
    );
    for (s, m) in ranked.iter().take(8) {
        println!("    {s:9.3}  {:?} {}", m.move_type(), m.serialize());
    }

    // Decompose the Summon@41 candidate's dphi by term (advisor's suspects:
    // unit_train_opportunity_cost, an eco_plan-driven Research boost).
    println!("\n=== dphi breakdown: Summon@41 vs the executed Research(Hunting) ===");
    for m in ranked.iter().map(|(_, m)| m.as_ref()).filter(|m| {
        m.serialize() == serde_json::json!({"moveType":4,"src":41,"type":2})
            || m.serialize() == serde_json::json!({"moveType":7,"type":15})
    }) {
        let (phi_pre, bd_pre) = polyfish::ai::reward::goal_potential_breakdown(
            &view.state, POV, &goal, Some(&aux), None, None, None, None,
        );
        let mut post = view.clone();
        if post.simulate_move(m).is_none() {
            println!("  {} -> could not simulate", m.serialize());
            continue;
        }
        let (phi_post, bd_post) = polyfish::ai::reward::goal_potential_breakdown(
            &post.state, POV, &goal, Some(&aux), None, None, None, None,
        );
        println!("  --- {} (dphi={:.3}) ---", m.serialize(), phi_post - phi_pre);
        use std::collections::HashMap;
        let mut pre_by: HashMap<&str, f32> = HashMap::new();
        for (l, v) in &bd_pre {
            *pre_by.entry(l).or_insert(0.0) += v;
        }
        let mut post_by: HashMap<&str, f32> = HashMap::new();
        for (l, v) in &bd_post {
            *post_by.entry(l).or_insert(0.0) += v;
        }
        let mut labels: Vec<&str> = pre_by.keys().chain(post_by.keys()).copied().collect();
        labels.sort();
        labels.dedup();
        for l in labels {
            let pre = *pre_by.get(l).unwrap_or(&0.0);
            let post = *post_by.get(l).unwrap_or(&0.0);
            if (post - pre).abs() > 0.01 {
                println!("      {l:30} pre={pre:10.3} post={post:10.3} delta={:10.3}", post - pre);
            }
        }
    }

    let top_scores: Vec<f32> = ranked.iter().take(8).map(|(s, _)| *s).collect();
    let priors = softmax_priors(&top_scores);
    println!("\n=== softmax_priors over the real top-8 (as micro-mcts would compute them) ===");
    for ((s, m), p) in ranked.iter().take(8).zip(priors.iter()) {
        println!("    score={s:9.3} prior={p:.6}  {}", m.serialize());
    }

    if ranked.len() < 2 || ranked[0].1.move_type() == polyfish::types::MoveType::EndTurn {
        println!("(< 2 candidates or lone EndTurn -- micro_search_pick would no-op here)");
        return;
    }

    let dummy = Evaluator::Dummy(DummyEvalHandle::new());
    let device = candle_core::Device::Cpu;
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&["model.safetensors"], candle_core::DType::F32, &device)
            .expect("load model.safetensors")
    };
    let network = std::sync::Arc::new(PolyZeroNet::new(vs).expect("construct PolyZeroNet"));
    let real = Evaluator::Inline(InlineEvalHandle::new(network));

    for (label, evaluator) in [("DUMMY", &dummy), ("REAL-NET", &real)] {
        for (sims, k) in [(16usize, 4usize), (64, 4), (64, 8)] {
            MICRO_MCTS_DEPTH_CALLS.store(0, Ordering::Relaxed);
            MICRO_MCTS_DEPTH_SUM.store(0, Ordering::Relaxed);
            MICRO_MCTS_MAX_DEPTH_SEEN.store(0, Ordering::Relaxed);
            let params = MicroParams { sims, depth: 64, k, c_puct: 1.5 };
            let (pick, _carry) =
                micro_search_pick(&view, POV, &goal, &ranked, &aux, gate, evaluator, &params, None);
            let calls = MICRO_MCTS_DEPTH_CALLS.load(Ordering::Relaxed);
            let sum = MICRO_MCTS_DEPTH_SUM.load(Ordering::Relaxed);
            let max_seen = MICRO_MCTS_MAX_DEPTH_SEEN.load(Ordering::Relaxed);
            println!(
                "\n[{label}] sims={sims} k={k}: pick={pick:?} (Some(0)=no override) mean_max_depth={:.2} deepest_seen={max_seen}",
                sum as f64 / calls.max(1) as f64
            );
            if let Some(i) = pick {
                println!("    picked: {}", ranked[i].1.serialize());
            }
        }
    }
}
