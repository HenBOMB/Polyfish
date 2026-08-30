//! Parametrized version of reward_choice_probe.rs: decompose any
//! Explorer-vs-Workshop (or Explorer-vs-other) reward pick via the REAL
//! goal_potential_breakdown accumulator, with belief properly threaded.
//! Usage: cargo run --example reward_choice_probe2 -- <replay.json> <pov>
//! <target_turn> <target_idx> <city_idx>

use polyfish::ai::belief::map::MapBelief;
use polyfish::ai::oracle_macro::{commit_macro_goal, StanceCommit};
use polyfish::ai::search::lane::{observe_lane_state, select_lane, LaneState};
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::types::{CityRewardType, MoveType};

fn state_at_p1_turn_start(full: &ModReplay, turn: i32, pov: i32) -> Game {
    let mut mr = full.clone();
    mr.turns.retain(|t| t.turn < turn);
    let mut g = Game::new();
    replay_game(&mut g, &mut mr).expect("deterministic replay failed");
    let _ = pov;
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
    let args: Vec<String> = std::env::args().collect();
    let replay_path = &args[1];
    let pov: i32 = args[2].parse().unwrap();
    let target_turn: i32 = args[3].parse().unwrap();
    let target_idx: usize = args[4].parse().unwrap();
    let city: i32 = args[5].parse().unwrap();

    let raw = std::fs::read_to_string(replay_path).expect("read replay");
    let full: ModReplay = serde_json::from_str(&raw).expect("parse replay");

    let mut sc = StanceCommit::default();
    let mut lane = LaneState::default();
    for turn in 0..target_turn {
        let g = state_at_p1_turn_start(&full, turn, pov);
        let view0 = g.clone_for_mcts(pov);
        let _ = commit_macro_goal(&view0.state, pov, &mut sc, 0);
        observe_lane_state(&view0.state, pov, &mut lane);
        select_lane(&view0.state, pov, &mut lane, None);
    }

    let true_game = state_at_step(&full, target_idx);
    let view = true_game.clone_for_mcts(pov);
    observe_lane_state(&view.state, pov, &mut lane);
    select_lane(&view.state, pov, &mut lane, None);
    let goal = commit_macro_goal(&view.state, pov, &mut sc, 0);
    println!(
        "=== goal @ idx{target_idx} turn{target_turn} city{city}: stance={:?} orders={:?} ===",
        goal.stance, goal.orders
    );

    let tribe = view.state.tribes.get(&pov).unwrap();
    println!(
        "player cities: {:?} (len={})",
        tribe.cities.iter().map(|c| c.idx).collect::<Vec<_>>(),
        tribe.cities.len()
    );
    let is_capital = view
        .state
        .tiles
        .get(&city)
        .map_or(false, |t| t.capital_of == pov);
    println!("city {city} is_capital={is_capital}");

    let belief = MapBelief::observe(&view.state, pov);
    let avg = polyfish::ai::reward::goal_shape_consts::avg_frontier_in_reach(
        &view.state,
        &belief,
        city,
        polyfish::ai::reward::goal_shape_consts::EXPLORER_WALK_RANGE,
    );
    println!(
        "avg_frontier_in_reach(city={city}, range={}) = {avg:.4} (FOG baseline=1.0)",
        polyfish::ai::reward::goal_shape_consts::EXPLORER_WALK_RANGE
    );
    let corners = polyfish::coords::map_corners(view.state.settings.size);
    for k in corners {
        let d = polyfish::functions::get_chebyshev_distance(city, k, view.state.settings.size);
        let dark = !view
            .state
            .tiles
            .get(&k)
            .map_or(false, |t| t.explorers.contains(&pov));
        let w = polyfish::ai::reward::goal_shape_consts::walkable_weight(&view.state, tribe, city, k);
        println!("  corner {k}: dist={d} dark={dark} walkable_weight={w:.4}");
    }

    let legal = view.legal_moves();
    for (label, want_type) in [
        ("Explorer", CityRewardType::Explorer),
        ("Workshop", CityRewardType::Workshop),
    ] {
        let Some(mv) = legal.iter().find(|m| {
            m.move_type() == MoveType::Reward
                && m.reward_type().ok() == Some(want_type)
                && m.target_idx().ok() == Some(city as usize)
        }) else {
            println!("\n--- {label}: not a legal reward at city {city} here ---");
            continue;
        };

        let heuristic = polyfish::ai::scoring::score_move_with_unit_goals(&true_game, mv.as_ref(), None);

        let mut probe = Game { state: view.state.clone() };
        let (phi_pre, bd_pre) = polyfish::ai::reward::goal_potential_breakdown(
            &probe.state, pov, &goal, None, None, None, Some(&belief),
        );
        probe.play_move(mv.as_ref());
        let (phi_post, bd_post) = polyfish::ai::reward::goal_potential_breakdown(
            &probe.state, pov, &goal, None, None, None, Some(&belief),
        );
        println!(
            "\n--- {label}: heuristic={heuristic:.2} dphi={:.3} (phi {phi_pre:.3} -> {phi_post:.3}) ---",
            phi_post - phi_pre
        );
        use std::collections::HashMap;
        let sum_by_label = |bd: &[(&'static str, f32)]| -> HashMap<&'static str, f32> {
            let mut m = HashMap::new();
            for (k, v) in bd {
                *m.entry(*k).or_insert(0.0) += v;
            }
            m
        };
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
                println!("  {label:30} {p:10.3} -> {q:10.3}  (Δ {d:+.3})");
            }
        }
    }
}
