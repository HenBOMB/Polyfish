//! Faithful re-run of the two contested plies (idx122 reward choice, idx179
//! garrison departure) through the REAL executor path: fogged view
//! (clone_for_mcts), threaded LaneState/TurnCounters, real GoalAux, real
//! star_gate, and rank_plies itself -- not a hand-assembled score_move +
//! goal_potential approximation. Mirrors macro_root_t5.rs's Pass-1 pattern.
//! Read-only on search/engine code.

use polyfish::ai::macro_exec::{rank_plies, TurnCounters};
use polyfish::ai::oracle_macro::{commit_macro_goal, tech_discipline_active, MacroGoal, StanceCommit};
use polyfish::ai::search::goal_aux::compute_goal_aux;
use polyfish::ai::search::lane::{observe_lane_state, select_lane, LaneState};
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::types::TechnologyType;

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

/// Replay every move up to (not including) global step `target_idx`
/// (0-based, matching main.rs's flatten_turns / the web replay viewer).
/// Returns the true (omniscient) Game at that exact ply.
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

fn faithful_rank(full: &ModReplay, target_idx: usize, target_turn: i32, goal: &MacroGoal) {
    // Pass 1: thread StanceCommit + LaneState from turn 0 up to target_turn
    // (exclusive of the goal-commit at target_turn itself -- we override
    // with the REAL picked goal from game0.jsonl instead of recomputing,
    // since MCTS may have picked a non-base candidate).
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
    let aux = compute_goal_aux(&view.state, POV, goal, counters.techs_bought, counters.tier3_bought, Some(&lane));
    let gate = tech_discipline_active(&view.state, POV, goal);

    println!(
        "=== faithful rank @ idx={target_idx} turn={target_turn} gate={gate} aux.recommended_techs={:?} aux.water_dead={} ===",
        aux.recommended_techs, aux.water_dead
    );

    for lambda in [0.0f32, 1.0f32] {
        let ranked = rank_plies(&mut view, POV, goal, &aux, gate, lambda, None);
        println!("  --- lambda={lambda} top 12 of {} ---", ranked.len());
        for (s, m) in ranked.iter().take(12) {
            println!("    {s:9.3}  {:?} {}", m.move_type(), m.serialize());
        }
        if let Some(pos) = ranked.iter().position(|(_, m)| m.move_type() == polyfish::types::MoveType::EndTurn) {
            println!("    (EndTurn survived gating, ranked #{} of {})", pos + 1, ranked.len());
        } else {
            println!("    (EndTurn gated out entirely)");
        }
    }
}

fn main() {
    let raw = std::fs::read_to_string(REPLAY).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");

    use polyfish::ai::oracle_macro::{OrderKind, Stance};

    // idx122, turn 9: real picked goal from game0.jsonl.
    let goal9 = MacroGoal {
        orders: vec![
            (OrderKind::Expand, 21),
            (OrderKind::Expand, 43),
            (OrderKind::Expand, 55),
            (OrderKind::Expand, 79),
            (OrderKind::Defend, 41),
            (OrderKind::Defend, 49),
            (OrderKind::Defend, 84),
        ],
        stance: Stance::Arm,
        save_target: None,
    };
    faithful_rank(&full, 122, 9, &goal9);

    // idx179, turn 11: real picked goal from game0.jsonl.
    let goal11 = MacroGoal {
        orders: vec![
            (OrderKind::Expand, 55),
            (OrderKind::Defend, 41),
            (OrderKind::Defend, 49),
            (OrderKind::Defend, 84),
        ],
        stance: Stance::Save,
        save_target: None,
    };
    faithful_rank(&full, 179, 11, &goal11);
}
