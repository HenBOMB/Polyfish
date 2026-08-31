//! Ground-truth Φ breakdown for one real ply, seeded from POLYFISH_PLY_TRACE
//! (goal + unit_goals dumped straight off the real trajectory) instead of
//! reconstructing the macro-mcts ballot pick and per-ply UnitGoalStore by
//! hand — probe2's reconstruction silently used the wrong (ephemeral-match)
//! unit_goals path and got a materially different score. This one reproduces
//! `rank_view`'s real total (score_move_with_unit_goals + λ·Δφ) exactly,
//! verified by matching the recorded `score` field bit-for-bit before
//! trusting the per-term breakdown.
//!
//! Usage: cargo run --example attack_pricing_probe3 -- <replay.json>
//!   <ply_trace.jsonl> <ply_trace_line_idx> <pov> <target_turn> <target_idx>
//!   <move1_json> <move2_json>
use polyfish::ai::search::goal_aux::compute_goal_aux;
use polyfish::ai::search::lane::{observe_lane_state, select_lane, LaneState};
use polyfish::ai::search::unit_goals::{UnitGoal, UnitGoalStore};
use polyfish::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
use polyfish::ai::{combat, reward, scoring};
use polyfish::ai::belief::map::MapBelief;
use polyfish::game::Game;
use polyfish::moves::Move;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::types::{MoveType, TechnologyType};
use std::collections::HashMap;

const LAMBDA: f32 = 1.0;
// EXP_ELO_111: mirrors macro_exec.rs's STEP_LETHAL_ENTRY_PENALTY exactly --
// keep these two constants in sync by hand, same as LAMBDA above.
const STEP_LETHAL_ENTRY_PENALTY: f32 = 3.0;

fn parse_order_kind(s: &str) -> OrderKind {
    match s {
        "Expand" => OrderKind::Expand,
        "Attack" => OrderKind::Attack,
        "Defend" => OrderKind::Defend,
        other => panic!("unknown OrderKind {other}"),
    }
}

fn parse_stance(s: &str) -> Stance {
    match s {
        "Grow" => Stance::Grow,
        "Arm" => Stance::Arm,
        "Unlock" => Stance::Unlock,
        "Save" => Stance::Save,
        other => panic!("unknown Stance {other}"),
    }
}

fn goal_from_json(v: &serde_json::Value) -> MacroGoal {
    let stance = parse_stance(v["stance"].as_str().unwrap());
    let orders = v["orders"]
        .as_array()
        .unwrap()
        .iter()
        .map(|pair| {
            let arr = pair.as_array().unwrap();
            (parse_order_kind(arr[0].as_str().unwrap()), arr[1].as_i64().unwrap() as i32)
        })
        .collect();
    MacroGoal { orders, stance, save_target: None }
}

fn unit_goals_from_json(v: &serde_json::Value) -> UnitGoalStore {
    let mut store = UnitGoalStore::default();
    for row in v.as_array().unwrap() {
        let Some(g) = row.get("goal").and_then(|g| if g.is_null() { None } else { Some(g) }) else {
            continue;
        };
        let id = row["unit_id"].as_u64().unwrap() as u32;
        let kind = parse_order_kind(g["kind"].as_str().unwrap());
        let target = g["target"].as_i64().unwrap() as i32;
        store.assign(id, UnitGoal { kind, target });
    }
    store
}

fn state_at_p1_turn_start(full: &ModReplay, turn: i32) -> Game {
    let mut mr = full.clone();
    mr.turns.retain(|t| t.turn < turn);
    let mut g = Game::new();
    replay_game(&mut g, &mut mr).expect("deterministic replay failed");
    g
}

fn counters_before(full: &ModReplay, turn: i32, pov: i32) -> polyfish::ai::macro_exec::TurnCounters {
    let mut c = polyfish::ai::macro_exec::TurnCounters::default();
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

#[allow(clippy::too_many_arguments)]
fn evaluate(
    label: &str,
    expected_score: Option<f64>,
    true_game: &Game,
    view: &Game,
    goal: &MacroGoal,
    aux: &polyfish::ai::oracle_macro::GoalAux,
    unit_goals: &UnitGoalStore,
    threats: &[(polyfish::states::UnitState, f32)],
    belief: &MapBelief,
    pre_health: &rustc_hash::FxHashMap<u32, f32>,
    pre_lethal: &rustc_hash::FxHashMap<u32, bool>,
    pov: i32,
    move_json: &serde_json::Value,
) {
    let legal = view.legal_moves();
    let Some(m) = legal.iter().find(|m| &m.serialize() == move_json) else {
        println!("\n--- {label}: {move_json} NOT LEGAL at this ply ---");
        return;
    };
    let base = scoring::score_move_with_unit_goals(true_game, m.as_ref(), Some(unit_goals), None);
    // EXP_ELO_111: resolve the acting unit's id from its PRE-move coords
    // before simulate_move relocates it -- Step-only, mirrors rank_plies.
    let step_unit_id = if m.move_type() == MoveType::Step {
        m.source_idx().ok().and_then(|src| {
            view.state
                .tribes
                .get(&pov)
                .and_then(|t| t.units.iter().find(|u| u.coords.idx == src as i32).map(|u| u.id))
        })
    } else {
        None
    };
    let (phi_pre, bd_pre) = reward::goal_potential_breakdown(
        &view.state, pov, goal, Some(aux), Some(threats), Some(unit_goals), Some(belief), Some(pre_health),
    );
    let mut probe = Game { state: view.state.clone() };
    let undo = probe.simulate_move(m.as_ref());
    let (phi_post, bd_post) = reward::goal_potential_breakdown(
        &probe.state, pov, goal, Some(aux), Some(threats), Some(unit_goals), Some(belief), Some(pre_health),
    );
    let mut step_penalty = 0.0f32;
    if let Some(uid) = step_unit_id {
        if !pre_lethal.get(&uid).copied().unwrap_or(false) {
            if let Some(u) = probe.state.tribes.get(&pov).and_then(|t| t.units.iter().find(|u| u.id == uid)) {
                let post_w = combat::lethal_threat_weight(&probe.state, u, threats);
                if post_w > 0.0 {
                    step_penalty = STEP_LETHAL_ENTRY_PENALTY * post_w;
                }
            }
        }
    }
    if let Some(undo) = undo {
        undo(&mut probe.state);
    }
    let dphi = phi_post - phi_pre;
    let total = base - step_penalty + LAMBDA * dphi;
    let match_str = match expected_score {
        Some(e) if (e - total as f64).abs() < 0.05 => "MATCH".to_string(),
        Some(e) => format!("MISMATCH (expected {e:.3})"),
        None => "no recorded score".to_string(),
    };
    println!(
        "\n--- {label}: {} :: {move_json}\n    base={base:.3} step_penalty={step_penalty:.3} dphi={dphi:.3} (phi {phi_pre:.3} -> {phi_post:.3}) total={total:.3}  [{match_str}]",
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
    let trace_path = &args[2];
    let trace_line_idx: usize = args[3].parse().unwrap();
    let pov: i32 = args[4].parse().unwrap();
    let target_turn: i32 = args[5].parse().unwrap();
    let target_idx: usize = args[6].parse().unwrap();
    let move1: serde_json::Value = serde_json::from_str(&args[7]).unwrap();
    let move2: serde_json::Value = serde_json::from_str(&args[8]).unwrap();

    let raw = std::fs::read_to_string(replay_path).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");

    let trace_raw = std::fs::read_to_string(trace_path).expect("read ply_trace");
    let trace_line = trace_raw.lines().nth(trace_line_idx).expect("trace line idx out of range");
    let trace_row: serde_json::Value = serde_json::from_str(trace_line).expect("parse trace row");

    assert_eq!(trace_row["turn"].as_i64().unwrap() as i32, target_turn, "trace row turn mismatch");
    assert_eq!(trace_row["player"].as_i64().unwrap() as i32, pov, "trace row player mismatch");

    let goal = goal_from_json(&trace_row["goal"]);
    let unit_goals = unit_goals_from_json(&trace_row["unit_goals"]);
    println!("=== ground-truth goal @ idx{target_idx} turn{target_turn}: stance={:?} orders={:?} ===", goal.stance, goal.orders);
    println!("=== unit_goals from trace: {} ===", serde_json::to_string(&trace_row["unit_goals"]).unwrap());

    let mut lane = LaneState::default();
    for turn in 0..target_turn {
        let g = state_at_p1_turn_start(&full, turn);
        let view0 = g.clone_for_mcts(pov);
        observe_lane_state(&view0.state, pov, &mut lane);
        select_lane(&view0.state, pov, &mut lane, None);
    }

    let counters = counters_before(&full, target_turn, pov);
    let true_game = state_at_step(&full, target_idx);
    let view = true_game.clone_for_mcts(pov);
    observe_lane_state(&view.state, pov, &mut lane);
    select_lane(&view.state, pov, &mut lane, None);

    let aux = compute_goal_aux(&view.state, pov, &goal, counters.techs_bought, counters.tier3_bought, Some(&lane));
    let threats = combat::threat_units(&view.state, pov);
    let belief = MapBelief::observe(&view.state, pov);
    // EXP_ELO_110: mirror rank_plies's pre_health map (unit id -> health at
    // ply start), so the Defend waterfall's self-wound floor stays in
    // parity here too -- see macro_exec.rs's rank_plies for the canonical
    // implementation this must stay in parity with.
    let pre_health: rustc_hash::FxHashMap<u32, f32> = view
        .state
        .tribes
        .get(&pov)
        .map(|t| t.units.iter().map(|u| (u.id, u.health)).collect())
        .unwrap_or_default();
    // EXP_ELO_111: mirror rank_plies's per-ply pre-lethal cache (unit id ->
    // was this unit lethally exposed at its PRE-move coords/health).
    let pre_lethal: rustc_hash::FxHashMap<u32, bool> = view
        .state
        .tribes
        .get(&pov)
        .map(|t| {
            t.units
                .iter()
                .map(|u| (u.id, combat::lethal_threat_weight(&view.state, u, &threats) > 0.0))
                .collect()
        })
        .unwrap_or_default();

    let expected: HashMap<String, f64> = trace_row["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (serde_json::to_string(&c["move"]).unwrap(), c["score"].as_f64().unwrap()))
        .collect();
    let e1 = expected.get(&serde_json::to_string(&move1).unwrap()).copied();
    let e2 = expected.get(&serde_json::to_string(&move2).unwrap()).copied();

    evaluate("move1", e1, &true_game, &view, &goal, &aux, &unit_goals, &threats, &belief, &pre_health, &pre_lethal, pov, &move1);
    evaluate("move2", e2, &true_game, &view, &goal, &aux, &unit_goals, &threats, &belief, &pre_health, &pre_lethal, pov, &move2);
}
