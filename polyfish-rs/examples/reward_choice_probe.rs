//! Ad-hoc: decompose the turn-3 Explorer-vs-Workshop reward pick at the
//! capital (idx29 of seed0) via the REAL goal_potential_breakdown
//! accumulator, with belief properly threaded (unlike attack_pricing_probe,
//! which passed None throughout) -- this decision specifically depends on
//! MapBelief's frontier signal and the tribe.cities.len()<=1 first-city
//! discount, both of which need a faithful reconstruction to see. Read-only.

use polyfish::ai::belief::map::MapBelief;
use polyfish::ai::macro_exec::TurnCounters;
use polyfish::ai::oracle_macro::{commit_macro_goal, StanceCommit};
use polyfish::ai::search::lane::{observe_lane_state, select_lane, LaneState};
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::types::{CityRewardType, MoveType};

const REPLAY: &str =
    "replays/exp096_seed0_watch/game_iter1_game0_seed1787500020.replay.json";
const POV: i32 = 1;

fn state_at_p1_turn_start(full: &ModReplay, turn: i32) -> Game {
    let mut mr = full.clone();
    mr.turns.retain(|t| t.turn < turn);
    let mut g = Game::new();
    replay_game(&mut g, &mut mr).expect("deterministic replay failed");
    g
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

fn main() {
    let raw = std::fs::read_to_string(REPLAY).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let target_turn = 3i32;
    let target_idx = 29usize; // the Reward decision

    let mut sc = StanceCommit::default();
    let mut lane = LaneState::default();
    for turn in 0..target_turn {
        let g = state_at_p1_turn_start(&full, turn);
        let view0 = g.clone_for_mcts(POV);
        let _ = commit_macro_goal(&view0.state, POV, &mut sc, 0);
        observe_lane_state(&view0.state, POV, &mut lane);
        select_lane(&view0.state, POV, &mut lane, None);
    }

    let true_game = state_at_step(&full, target_idx);
    let view = true_game.clone_for_mcts(POV);
    observe_lane_state(&view.state, POV, &mut lane);
    select_lane(&view.state, POV, &mut lane, None);
    let goal = commit_macro_goal(&view.state, POV, &mut sc, 0);
    println!("=== goal @ idx{target_idx} turn{target_turn}: stance={:?} orders={:?} ===", goal.stance, goal.orders);

    let tribe = view.state.tribes.get(&POV).unwrap();
    println!(
        "player cities: {:?} (len={})",
        tribe.cities.iter().map(|c| c.idx).collect::<Vec<_>>(),
        tribe.cities.len()
    );

    let belief = MapBelief::observe(&view.state, POV);

    let legal = view.legal_moves();
    for (label, want_type) in [("Explorer", CityRewardType::Explorer), ("Workshop", CityRewardType::Workshop)] {
        let mv = legal
            .iter()
            .find(|m| m.move_type() == MoveType::Reward && m.reward_type().ok() == Some(want_type))
            .unwrap_or_else(|| panic!("{label} reward not legal here"));

        let heuristic = polyfish::ai::scoring::score_move_with_unit_goals(&true_game, mv.as_ref(), None);

        let mut probe = Game { state: view.state.clone() };
        let (phi_pre, _) = polyfish::ai::reward::goal_potential_breakdown(
            &probe.state, POV, &goal, None, None, None, Some(&belief),
        );
        probe.play_move(mv.as_ref());
        let (phi_post, bd_post) = polyfish::ai::reward::goal_potential_breakdown(
            &probe.state, POV, &goal, None, None, None, Some(&belief),
        );
        println!("\n--- {label}: heuristic={heuristic:.2} dphi={:.3} (phi {phi_pre:.3} -> {phi_post:.3}) ---", phi_post - phi_pre);
        let mut terms: Vec<_> = bd_post.iter().filter(|(k, _)| *k == "explorer").collect();
        terms.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        for (k, v) in &terms {
            println!("  {k}: {v:.3}");
        }
    }

    // Direct read of the frontier/discount inputs for the capital.
    let city = tribe.cities.iter().find(|c| c.idx == 84).unwrap();
    println!("\ncapital city 84: rewards={:?} progress={} level={}", city.rewards, city.progress, city.level);
    let revealed = view.state.tiles.values().filter(|t| t.explorers.contains(&POV)).count() as f32;
    let width = view.state.settings.size as f32;
    println!("hidden_frac = {:.4} ({} revealed / {} total tiles)", (1.0 - revealed / (width * width)).max(0.0), revealed as i32, (width*width) as i32);
}
