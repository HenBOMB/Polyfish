//! Parametrized: decompose why one candidate move outscored another at a
//! real ply, reproducing rank_plies' exact score = score_move + λ·Δφ
//! computation (aux/threats/belief computed the same way), plus the full
//! Φ term breakdown for both moves. Usage: cargo run --example
//! attack_pricing_probe2 -- <replay.json> <pov> <target_turn> <target_idx>
//! <move1_json> <move2_json>
//! Move JSON e.g.: '{"moveType":2,"src":68,"target":79}' (Attack) or
//! '{"moveType":1,"src":68,"target":67}' (Step).
use polyfish::ai::macro_exec::TurnCounters;
use polyfish::ai::oracle_macro::{commit_macro_goal, compute_goal_aux, StanceCommit};
use polyfish::ai::search::lane::{observe_lane_state, select_lane, LaneState};
use polyfish::ai::{combat, reward, scoring};
use polyfish::ai::belief::map::MapBelief;
use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::types::TechnologyType;
use std::collections::HashMap;

const LAMBDA: f32 = 1.0;

fn state_at_p1_turn_start(full: &ModReplay, turn: i32, pov: i32) -> Game {
    let mut mr = full.clone();
    mr.turns.retain(|t| t.turn < turn);
    let mut g = Game::new();
    replay_game(&mut g, &mut mr).expect("deterministic replay failed");
    let _ = pov;
    g
}

fn counters_before(full: &ModReplay, turn: i32, pov: i32) -> TurnCounters {
    let mut c = TurnCounters::default();
    for t in &full.turns {
        if t.turn >= turn {
            break;
        }
        for p in &t.players {
            if p.player_id != pov {
                continue;
            }
            for cmd in &p.commands {
                if cmd.get("moveType").and_then(|v| v.as_i64()) != Some(7) {
                    continue;
                }
                c.techs_bought += 1;
                if let Some(tv) = cmd.get("type") {
                    if let Ok(tech) = serde_json::from_value::<TechnologyType>(tv.clone()) {
                        if polyfish::settings::technology::get_technology_setting(tech).tier == Some(3) {
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
                let m = legal.iter().find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} move not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    panic!("target_idx {target_idx} beyond game length {idx}");
}

fn sum_by_label(bd: &[(&'static str, f32)]) -> HashMap<&'static str, f32> {
    let mut m = HashMap::new();
    for (k, v) in bd {
        *m.entry(*k).or_insert(0.0) += v;
    }
    m
}

fn evaluate(
    label: &str,
    true_game: &Game,
    view: &Game,
    goal: &polyfish::ai::oracle_macro::MacroGoal,
    aux: &polyfish::ai::oracle_macro::GoalAux,
    threats: &[(polyfish::states::UnitState, f32)],
    belief: &MapBelief,
    pov: i32,
    move_json: &serde_json::Value,
) {
    let legal = view.legal_moves();
    let Some(m) = legal.iter().find(|m| &m.serialize() == move_json) else {
        println!("\n--- {label}: {move_json} NOT LEGAL at this ply ---");
        return;
    };
    let base = scoring::score_move_with_unit_goals(true_game, m.as_ref(), None, None);
    let (phi_pre, bd_pre) = reward::goal_potential_breakdown(
        &view.state, pov, goal, Some(aux), Some(threats), None, Some(belief), None,
    );
    let mut probe = Game { state: view.state.clone() };
    let undo = probe.simulate_move(m.as_ref());
    let (phi_post, bd_post) = reward::goal_potential_breakdown(
        &probe.state, pov, goal, Some(aux), Some(threats), None, Some(belief), None,
    );
    if let Some(undo) = undo {
        undo(&mut probe.state);
    }
    let dphi = phi_post - phi_pre;
    let total = base + LAMBDA * dphi;
    println!(
        "\n--- {label}: {} :: {move_json}\n    base={base:.3} dphi={dphi:.3} (phi {phi_pre:.3} -> {phi_post:.3}) total={total:.3}",
        m.serialize()
    );
    let pre = sum_by_label(&bd_pre);
    let post = sum_by_label(&bd_post);
    let mut labels: Vec<&&str> = pre.keys().chain(post.keys()).collect();
    labels.sort();
    labels.dedup();
    for label in labels {
        let p = pre.get(*label).copied().unwrap_or(0.0);
        let q = post.get(*label).copied().unwrap_or(0.0);
        let d = q - p;
        if d.abs() > 0.01 {
            println!("    {label:30} {p:10.3} -> {q:10.3}  (Δ {d:+.3})");
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let replay_path = &args[1];
    let pov: i32 = args[2].parse().unwrap();
    let target_turn: i32 = args[3].parse().unwrap();
    let target_idx: usize = args[4].parse().unwrap();
    let move1: serde_json::Value = serde_json::from_str(&args[5]).unwrap();
    let move2: serde_json::Value = serde_json::from_str(&args[6]).unwrap();

    let raw = std::fs::read_to_string(replay_path).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");

    let mut sc = StanceCommit::default();
    let mut lane = LaneState::default();
    for turn in 0..target_turn {
        let g = state_at_p1_turn_start(&full, turn, pov);
        let view0 = g.clone_for_mcts(pov);
        let counters = counters_before(&full, turn, pov);
        let _ = commit_macro_goal(&view0.state, pov, &mut sc, counters.tier3_bought);
        observe_lane_state(&view0.state, pov, &mut lane);
        select_lane(&view0.state, pov, &mut lane, None);
    }

    let counters = counters_before(&full, target_turn, pov);
    let true_game = state_at_step(&full, target_idx);
    let view = true_game.clone_for_mcts(pov);
    observe_lane_state(&view.state, pov, &mut lane);
    select_lane(&view.state, pov, &mut lane, None);
    let goal = commit_macro_goal(&view.state, pov, &mut sc, counters.tier3_bought);
    println!(
        "=== goal @ idx{target_idx} turn{target_turn}: stance={:?} orders={:?} ===",
        goal.stance, goal.orders
    );

    let aux = compute_goal_aux(&view.state, pov, &goal, counters.techs_bought, counters.tier3_bought, Some(&lane));
    let threats = combat::threat_units(&view.state, pov);
    let belief = MapBelief::observe(&view.state, pov);

    evaluate("move1", &true_game, &view, &goal, &aux, &threats, &belief, pov, &move1);
    evaluate("move2", &true_game, &view, &goal, &aux, &threats, &belief, pov, &move2);
}
