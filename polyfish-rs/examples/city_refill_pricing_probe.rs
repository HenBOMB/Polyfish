//! EXP_ELO_117 (in progress): why does stepping a unit BACK onto a just-
//! vacated own city score deeply negative instead of near-zero? Reconstructs
//! state at `target_idx`, computes a fresh `compute_macro_goal` for `player`,
//! and prints the `goal_potential_breakdown` delta (phi_post - phi_pre) by
//! term for a specific Step candidate, alongside its base `score_move`.
//! Diagnostic-only (None sinks for aux/threats/unit_goals/belief/pre_health
//! -- not byte-identical to the live ply's score, but enough to identify
//! which term dominates).
//! Usage: cargo run --example city_refill_pricing_probe -- <replay.json> <target_idx> <src> <target> [goal_at_idx]
//! `goal_at_idx` (optional, defaults to target_idx) lets the caller price the
//! move against a goal frozen at an EARLIER state -- e.g. the real turn-start
//! commit point -- instead of a fresh recompute at the move's own state.
use polyfish::game::Game;
use polyfish::moves::step::StepMove;
use polyfish::replayer::ModReplay;

fn state_at(full: &ModReplay, target_idx: usize) -> Game {
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    'outer: for t in &full.turns {
        let mut players: Vec<_> = t.players.iter().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            for cmd in &pl.commands {
                if idx == target_idx {
                    break 'outer;
                }
                let legal = game.legal_moves();
                let m = legal
                    .iter()
                    .find(|m| &m.serialize() == cmd)
                    .unwrap_or_else(|| panic!("idx={idx} not legal: {cmd}"));
                game.play_move(m.as_ref());
                idx += 1;
            }
        }
    }
    game
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let target_idx: usize = args[2].parse().unwrap();
    let src: i32 = args[3].parse().unwrap();
    let target: i32 = args[4].parse().unwrap();
    let goal_at_idx: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(target_idx);

    let game = state_at(&full, target_idx);
    let goal_state = state_at(&full, goal_at_idx);

    let player = game.state.settings.current_player_turn_id;
    let goal = polyfish::ai::oracle_macro::compute_macro_goal(&goal_state.state, player, 0);
    println!("goal.orders = {:?}", goal.orders);
    println!("goal.stance = {:?}", goal.stance);
    println!("goal.prepare = {:?}", goal.prepare);

    let mv = StepMove::new(src, target);
    let base = polyfish::ai::scoring::score_move_with_unit_goals(&game, &mv, None, None);
    println!("score_move (base heuristic) = {base:.3}");

    let (phi_pre, bd_pre) =
        polyfish::ai::reward::goal_potential_breakdown(&game.state, player, &goal, None, None, None, None, None);
    let mut post_game = game.clone();
    post_game
        .simulate_move(&mv)
        .expect("move should be legal/simulatable");
    let (phi_post, bd_post) = polyfish::ai::reward::goal_potential_breakdown(
        &post_game.state,
        player,
        &goal,
        None,
        None,
        None,
        None,
        None,
    );
    println!("phi_pre = {phi_pre:.3}  phi_post = {phi_post:.3}  dphi = {:.3}", phi_post - phi_pre);

    use std::collections::HashMap;
    let mut pre_by_label: HashMap<&str, f32> = HashMap::new();
    for (l, v) in &bd_pre {
        *pre_by_label.entry(l).or_insert(0.0) += v;
    }
    let mut post_by_label: HashMap<&str, f32> = HashMap::new();
    for (l, v) in &bd_post {
        *post_by_label.entry(l).or_insert(0.0) += v;
    }
    let mut labels: Vec<&str> =
        pre_by_label.keys().chain(post_by_label.keys()).copied().collect();
    labels.sort();
    labels.dedup();
    let mut rows: Vec<(&str, f32, f32, f32)> = labels
        .into_iter()
        .map(|l| {
            let pre = *pre_by_label.get(l).unwrap_or(&0.0);
            let post = *post_by_label.get(l).unwrap_or(&0.0);
            (l, pre, post, post - pre)
        })
        .collect();
    rows.sort_by(|a, b| a.3.abs().partial_cmp(&b.3.abs()).unwrap());
    for (label, pre, post, delta) in rows {
        if delta.abs() > 0.01 {
            println!("  {label:30} pre={pre:10.3} post={post:10.3} delta={delta:10.3}");
        }
    }
}
