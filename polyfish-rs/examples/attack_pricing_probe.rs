//! EXP_ELO_094: decompose the garrison-preserving Attack's Φ collapse at the
//! flagged idx179 ply. EXP_ELO_075 measured Attack 49->39 at -558.240 (base
//! 45.000, dphi -600.0000) -- exactly SHAPE_GOAL_DEFEND_COVER (600.0) --
//! suggesting the garrison's own defend_cover credit vanishes post-Attack
//! even though the garrison never moves. Dumps city_risks/defend_plan
//! before and after simulating the Attack to confirm the mechanism
//! directly instead of guessing from the round number. Read-only.

use polyfish::ai::macro_exec::TurnCounters;
use polyfish::ai::oracle_macro::{commit_macro_goal, StanceCommit};
use polyfish::ai::search::lane::{observe_lane_state, select_lane, LaneState};
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::types::{MoveType, TechnologyType};

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

fn dump_risk_and_plan(game: &Game, label: &str, attack_targets: &[i32]) {
    let risks = polyfish::ai::combat::city_risks(&game.state, POV);
    for r in &risks {
        println!(
            "  [{label}] risk city={} risk={:.4} at_risk={} sieged={} open={} attackers={:?} need_damage={:.3}",
            r.city, r.risk, r.at_risk, r.sieged, r.open, r.attackers, r.need_damage
        );
        let plan = polyfish::ai::combat::defend_plan(&game.state, POV, r, attack_targets);
        const RISK_GARRISON_FALLS: f32 = 0.35; // combat.rs's own value, pub(crate) only
        let urgency = (r.risk / RISK_GARRISON_FALLS)
            .tanh()
            .max(if r.at_risk { 1.0 } else { 0.0 });
        println!(
            "  [{label}] defend_plan city={} hold_needed={} shortfall={:.3} assigned={:?} urgency={:.4} cover_credit={:.1}",
            r.city, plan.hold_needed, plan.shortfall, plan.assigned, urgency,
            plan.assigned.iter().map(|(_, sat)| 600.0 * urgency * sat).sum::<f32>()
                + if plan.hold_needed { 400.0 * urgency } else { 0.0 }
        );
    }
    if !risks.iter().any(|r| r.city == 49) {
        println!("  [{label}] city 49 NOT in city_risks() output at all -- Defend order for it would be a stale no-op");
    }
}

fn main() {
    let raw = std::fs::read_to_string(REPLAY).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let target_turn = 11i32;
    let target_idx = 179usize;

    let mut sc = StanceCommit::default();
    let mut lane = LaneState::default();
    for turn in 0..target_turn {
        let g = state_at_p1_turn_start(&full, turn);
        let view0 = g.clone_for_mcts(POV);
        let counters = counters_before(&full, turn);
        let _ = commit_macro_goal(&view0.state, POV, &mut sc, counters.tier3_bought);
        observe_lane_state(&view0.state, POV, &mut lane);
        select_lane(&view0.state, POV, &mut lane, None);
    }

    let counters = counters_before(&full, target_turn);
    let true_game = state_at_step(&full, target_idx);
    let view = true_game.clone_for_mcts(POV);
    observe_lane_state(&view.state, POV, &mut lane);
    select_lane(&view.state, POV, &mut lane, None);
    let goal = commit_macro_goal(&view.state, POV, &mut sc, counters.tier3_bought);
    println!("=== goal @ idx{target_idx} turn{target_turn}: stance={:?} orders={:?} ===", goal.stance, goal.orders);

    let attack_targets: Vec<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == polyfish::ai::oracle_macro::OrderKind::Attack)
        .map(|(_, i)| *i)
        .collect();

    println!("\n--- BEFORE Attack ---");
    dump_risk_and_plan(&view, "before", &attack_targets);

    let legal = view.legal_moves();
    let attack = legal
        .iter()
        .find(|m| {
            m.move_type() == MoveType::Attack
                && m.serialize().get("src").and_then(|v| v.as_i64()) == Some(49)
        })
        .expect("Attack 49->? must be legal here");
    println!("\nSimulating: {:?} {}", attack.move_type(), attack.serialize());

    let mut probe = Game { state: view.state.clone() };
    let ok = probe.play_move(attack.as_ref());
    println!("play_move result: {:?}", ok.map(|_| "ok"));

    println!("\n--- AFTER Attack ---");
    dump_risk_and_plan(&probe, "after", &attack_targets);

    // Direct Φ decomposition via the accumulator dump, if available.
    let phi_pre = polyfish::ai::reward::goal_potential(&view.state, POV, &goal, None);
    let phi_post = polyfish::ai::reward::goal_potential(&probe.state, POV, &goal, None);
    println!("\nphi_pre={phi_pre:.4} phi_post={phi_post:.4} dphi={:.4}", phi_post - phi_pre);
}
