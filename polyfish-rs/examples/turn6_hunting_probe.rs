//! Diagnostic for Verdi's report: at move_count=74 (turn 6, player 1) the
//! watch-replay chose "Research Hunting" instead of addressing an undefended
//! city with an enemy nearby. Reconstructs the state right before that move,
//! lists legal candidates (Research + any Train/city-defense options), and
//! prices the chosen move against the best defense alternative using the
//! same score_move + goal_potential machinery rank_plies uses. Diagnostic-
//! only (aux/threats/unit_goals/belief/pre_health all None -- not byte-
//! identical to the live search's score, enough to identify which term
//! dominates), same convention as city_refill_pricing_probe.rs.
//! Usage: cargo run --example turn6_hunting_probe -- <replay.json> <target_idx>
use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use polyfish::types::MoveType;

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

/// Traces `watch_city`'s garrison-occupancy TRANSITIONS (empty<->occupied)
/// from game start through `target_idx`, so "when did it go empty" is
/// answered by direct replay rather than assumed from a single snapshot.
fn trace_garrison_transitions(full: &ModReplay, target_idx: usize, watch_city: i32) {
    let mut game = Game::new();
    game.state = full.game_state.clone();
    game.post_load();
    let mut idx = 0usize;
    println!("\n=== city{watch_city} garrison transitions up to idx={target_idx} ===");
    let start_occ = polyfish::functions::get_unit_at(&game.state, watch_city).is_some();
    println!("  idx=0 (game start) occupied={start_occ}");
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
                let occ_before = polyfish::functions::get_unit_at(&game.state, watch_city).is_some();
                game.play_move(m.as_ref());
                let occ_after = polyfish::functions::get_unit_at(&game.state, watch_city).is_some();
                if occ_before != occ_after {
                    println!(
                        "  idx={idx} turn={} player={} move={} -> occupied {occ_before}->{occ_after}",
                        game.state.settings.turn,
                        pl.player_id,
                        m.serialize(),
                    );
                }
                idx += 1;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read_to_string(&args[1]).unwrap();
    let full: ModReplay = serde_json::from_str(&raw).unwrap();
    let target_idx: usize = args[2].parse().unwrap();

    let mut game = state_at(&full, target_idx);
    let player = game.state.settings.current_player_turn_id;
    let turn = game.state.settings.turn;
    let stars = game.state.tribes.get(&player).map(|t| t.stars).unwrap_or(0);
    println!("=== state before idx={target_idx} ===");
    println!("turn={turn} player={player} stars={stars}");

    let goal = polyfish::ai::oracle_macro::compute_macro_goal(&game.state, player, 0);
    println!("goal.orders = {:?}", goal.orders);
    println!("goal.stance = {:?}", goal.stance);

    println!("\n=== threat_units (what player {player} can actually see/remember) ===");
    for (u, trust) in polyfish::ai::combat::threat_units(&game.state, player) {
        let d = polyfish::functions::get_chebyshev_distance(
            u.coords.idx,
            41,
            game.state.settings.size as i32,
        );
        println!(
            "  owner={} type={:?} idx={} dist_to_city41={d} trust={trust:.3}",
            u.owner, u.unit_type, u.coords.idx
        );
    }
    // Ground truth (omniscient, ignores FOW): is the TRUE nearest enemy's
    // own tile currently in player 1's explorer set (i.e. actually visible),
    // or only known via the true/omniscient state this probe otherwise uses?
    for (opp_id, opp_tribe) in &game.state.tribes {
        if *opp_id == player {
            continue;
        }
        for u in &opp_tribe.units {
            let d = polyfish::functions::get_chebyshev_distance(
                u.coords.idx,
                41,
                game.state.settings.size as i32,
            );
            if d <= 4 {
                let visible = game
                    .state
                    .tiles
                    .get(&u.coords.idx)
                    .map_or(false, |t| t.explorers.contains(&player));
                println!(
                    "  TRUE enemy near city41: owner={opp_id} type={:?} idx={} dist={d} visible_to_p{player}={visible}",
                    u.unit_type, u.coords.idx
                );
            }
        }
    }

    println!("\n=== turns_to_reach ground truth (real engine pathfinding, horizon=10) ===");
    for (u, _) in polyfish::ai::combat::threat_units(&game.state, player) {
        let d = polyfish::functions::get_chebyshev_distance(u.coords.idx, 41, game.state.settings.size as i32);
        let mv = polyfish::functions::get_unit_movement(&game.state, &u);
        let turns = polyfish::ai::combat::turns_to_reach_debug(&game.state, &u, 41, 10);
        println!(
            "  {:?} at idx={} (chebyshev_dist={d}, movement={mv}) -> turns_to_reach(city41)={turns:?}",
            u.unit_type, u.coords.idx
        );
    }

    println!("\n=== city_risks (drives Defend-order assignment via needs_order()) ===");
    for r in polyfish::ai::combat::city_risks(&game.state, player) {
        println!(
            "  city={} sieged={} open={} arrives_next_turn={} at_risk={} risk={:.3} needs_order={} attackers={:?}",
            r.city,
            r.sieged,
            r.open,
            r.arrives_next_turn,
            r.at_risk,
            r.risk,
            r.needs_order(),
            r.attackers.iter().map(|(u, w)| (u.id, *w)).collect::<Vec<_>>(),
        );
    }

    trace_garrison_transitions(&full, target_idx, 41);

    // Cities owned by this player: level, garrison occupant, nearest enemy unit.
    println!("\n=== cities ===");
    if let Some(tribe) = game.state.tribes.get(&player) {
        for city in &tribe.cities {
            let city_idx = &city.idx;
            let occ = polyfish::functions::get_unit_at(&game.state, *city_idx);
            let size = game.state.settings.size as i32;
            let mut nearest_enemy: Option<(i32, i32)> = None;
            for (opp_id, opp_tribe) in &game.state.tribes {
                if *opp_id == player {
                    continue;
                }
                for u in &opp_tribe.units {
                    let d = polyfish::functions::get_chebyshev_distance(u.coords.idx, *city_idx, size);
                    if nearest_enemy.map(|(_, nd)| d < nd).unwrap_or(true) {
                        nearest_enemy = Some((u.coords.idx, d));
                    }
                }
            }
            println!(
                "  city={city_idx} level={} garrison={} nearest_enemy={:?}",
                city.level,
                occ.map(|u| format!("id={} type={:?} owner={}", u.id, u.unit_type, u.owner))
                    .unwrap_or_else(|| "EMPTY".into()),
                nearest_enemy,
            );
        }
    }

    // Legal candidates grouped by move type.
    println!("\n=== legal candidates ===");
    let legal = game.legal_moves();
    let mut by_type: std::collections::HashMap<MoveType, usize> = std::collections::HashMap::new();
    for m in &legal {
        *by_type.entry(m.move_type()).or_insert(0) += 1;
    }
    let mut types: Vec<_> = by_type.into_iter().collect();
    types.sort_by_key(|(t, _)| format!("{t:?}"));
    for (t, n) in &types {
        println!("  {t:?}: {n}");
    }

    println!("\n=== Research candidates ===");
    for m in legal.iter().filter(|m| m.move_type() == MoveType::Research) {
        let s = polyfish::ai::scoring::score_move_with_unit_goals(&game, m.as_ref(), None, None);
        println!("  {} -> score_move={s:.3}", m.serialize());
    }

    println!("\n=== Summon/Build candidates (structure/city builds) ===");
    for m in legal
        .iter()
        .filter(|m| matches!(m.move_type(), MoveType::Summon | MoveType::Build))
    {
        let s = polyfish::ai::scoring::score_move_with_unit_goals(&game, m.as_ref(), None, None);
        println!("  {} -> score_move={s:.3}", m.serialize());
    }

    // Price the chosen move (Research Hunting, assumed) with a phi breakdown.
    println!("\n=== phi breakdown for each Research candidate ===");
    for m in legal.iter().filter(|m| m.move_type() == MoveType::Research) {
        let (phi_pre, _) = polyfish::ai::reward::goal_potential_breakdown(
            &game.state, player, &goal, None, None, None, None, None,
        );
        let mut post = game.clone();
        if post.simulate_move(m.as_ref()).is_none() {
            continue;
        }
        let (phi_post, bd_post) = polyfish::ai::reward::goal_potential_breakdown(
            &post.state, player, &goal, None, None, None, None, None,
        );
        println!(
            "  {} dphi={:.3} (phi_pre={phi_pre:.3} phi_post={phi_post:.3})",
            m.serialize(),
            phi_post - phi_pre
        );
        let _ = bd_post;
    }

    // EVERY legal candidate, with score_move -- Verdi's flag is city49, not
    // city41 (both attackers sit ADJACENT to 49, not 41), so this is a full,
    // unfiltered listing rather than the earlier Research/Summon/Build-only
    // slices, to find anything that reinforces or otherwise helps 49.
    println!("\n=== ALL legal candidates, sorted by score_move ===");
    let mut all_scored: Vec<(f32, String, i32)> = legal
        .iter()
        .map(|m| {
            let s = polyfish::ai::scoring::score_move_with_unit_goals(&game, m.as_ref(), None, None);
            (s, m.serialize().to_string(), m.source_idx().ok().map(|v| v as i32).unwrap_or(-1))
        })
        .collect();
    all_scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (s, mv, src) in &all_scored {
        let touches_49 = mv.contains("\"src\":49") || mv.contains("\"target\":49");
        let flag = if touches_49 { "  <-- touches city49" } else { "" };
        println!("  {s:9.3}  {mv}  (src={src}){flag}");
    }

    // Full dphi breakdown for every candidate that touches city49 at all,
    // plus the executed Research(Hunting), for direct side-by-side pricing.
    println!("\n=== dphi breakdown: candidates touching city49 vs Research(Hunting) ===");
    for m in legal.iter().filter(|m| {
        let s = m.serialize().to_string();
        s.contains("\"src\":49") || s.contains("\"target\":49") || m.move_type() == MoveType::Research
    }) {
        let (phi_pre, bd_pre) = polyfish::ai::reward::goal_potential_breakdown(
            &game.state, player, &goal, None, None, None, None, None,
        );
        let mut post = game.clone();
        if post.simulate_move(m.as_ref()).is_none() {
            println!("  {} -> could not simulate", m.serialize());
            continue;
        }
        let (phi_post, bd_post) = polyfish::ai::reward::goal_potential_breakdown(
            &post.state, player, &goal, None, None, None, None, None,
        );
        let s = polyfish::ai::scoring::score_move_with_unit_goals(&game, m.as_ref(), None, None);
        println!(
            "  --- {} (score_move={s:.3} dphi={:.3}) ---",
            m.serialize(),
            phi_post - phi_pre
        );
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
}
