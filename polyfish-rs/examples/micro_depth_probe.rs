//! EXP_ELO_079: measure micro-mcts's own emergent search depth at the exact
//! GARRISON_49 idx177/178 plies (the real, busy turn-11 state where the
//! star-sequencing bug happened) instead of a fresh, action-starved
//! mapgen'd turn. Uses `Evaluator::Dummy` (constant leaf value) for speed --
//! a mechanics-only, conservative-lower-bound measurement of depth, not a
//! production-faithful decision-quality read. Read-only on search/engine
//! code.

use polyfish::ai::eval_server::{DummyEvalHandle, Evaluator};
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

const REPLAY: &str =
    "replays/exp074_seed0_watch/game_iter51_game0_seed1787500020.replay.json";
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

fn probe_depth_at(full: &ModReplay, target_idx: usize, target_turn: i32, evaluator: &Evaluator, params: &MicroParams) {
    let mut sc = StanceCommit::default();
    let mut lane = LaneState::default();
    for turn in 0..target_turn {
        let g = state_at_p1_turn_start(full, turn);
        let view0 = g.clone_for_mcts(POV);
        let counters = counters_before(full, turn);
        let base = commit_macro_goal(&view0.state, POV, &mut sc, counters.tier3_bought);
        observe_lane_state(&view0.state, POV, &mut lane);
        select_lane(&view0.state, POV, &mut lane, None);
        let _ = base;
    }

    let counters = counters_before(full, target_turn);
    let true_game = state_at_step(full, target_idx);
    let mut view = true_game.clone_for_mcts(POV);
    observe_lane_state(&view.state, POV, &mut lane);
    select_lane(&view.state, POV, &mut lane, None);
    let goal = commit_macro_goal(&view.state, POV, &mut sc, counters.tier3_bought);
    let aux = compute_goal_aux(&view.state, POV, &goal, counters.techs_bought, counters.tier3_bought, Some(&lane));
    let gate = tech_discipline_active(&view.state, POV, &goal);
    let ranked = rank_plies(&mut view, POV, &goal, &aux, gate, 1.0, None);

    println!("=== idx={target_idx} turn={target_turn} stance={:?} candidates={} ===", goal.stance, ranked.len());
    for (s, m) in ranked.iter().take(6) {
        println!("    {s:9.3}  {:?} {}", m.move_type(), m.serialize());
    }
    if ranked.len() < 2 || ranked[0].1.move_type() == polyfish::types::MoveType::EndTurn {
        println!("    (< 2 candidates or lone EndTurn -- micro_search_pick would no-op here)");
        return;
    }

    let (pick, _carry) = micro_search_pick(&view, POV, &goal, &ranked, &aux, gate, evaluator, params, None);
    let after_calls = MICRO_MCTS_DEPTH_CALLS.load(Ordering::Relaxed);
    let sum = MICRO_MCTS_DEPTH_SUM.load(Ordering::Relaxed);
    let max_seen = MICRO_MCTS_MAX_DEPTH_SEEN.load(Ordering::Relaxed);
    println!(
        "    micro_search_pick: pick={pick:?} (Some(0) means no override of rank_plies' own top choice)"
    );
    println!(
        "    running: calls={after_calls} mean_max_depth={:.2} deepest_line_seen={max_seen}",
        sum as f64 / after_calls.max(1) as f64
    );
}

fn main() {
    let raw = std::fs::read_to_string(REPLAY).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let evaluator = Evaluator::Dummy(DummyEvalHandle::new());

    println!("\n#### production params: sims=64, k=4 ####");
    let params = MicroParams { sims: 64, depth: 64, k: 4, c_puct: 1.5 };
    probe_depth_at(&full, 177, 11, &evaluator, &params);
    probe_depth_at(&full, 178, 11, &evaluator, &params);
    probe_depth_at(&full, 179, 11, &evaluator, &params);

    for (sims, k) in [(64usize, 4usize), (128, 4), (256, 4), (64, 2), (64, 3)] {
        MICRO_MCTS_DEPTH_CALLS.store(0, Ordering::Relaxed);
        MICRO_MCTS_DEPTH_SUM.store(0, Ordering::Relaxed);
        MICRO_MCTS_MAX_DEPTH_SEEN.store(0, Ordering::Relaxed);
        println!("\n#### sweep: sims={sims}, k={k} (idx177 only) ####");
        let params = MicroParams { sims, depth: 64, k, c_puct: 1.5 };
        probe_depth_at(&full, 177, 11, &evaluator, &params);
    }
}
