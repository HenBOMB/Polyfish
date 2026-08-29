//! Throwaway diagnostic: replay the seed[0] watch game move-by-move through
//! the real engine (exactly matching main.rs's flatten_turns step indexing),
//! and dump rich state context around the 4 steps Verdi flagged from the
//! replay viewer: 92 (Research Organization), 122 (Reward PopGrowth vs
//! BorderGrowth for city 84), 147 (Build Forge at 48 vs proposed 61), and
//! 177-180 (Harvest x2 then Step off tile 49, possible undefended-city bug).
//! Read-only on search/engine code.

use polyfish::functions::{get_adjacent_indices, get_city_at, get_structure_at};
use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use polyfish::types::{MoveType, ResourceType, TerrainType};
use std::collections::BTreeSet;

const REPLAY: &str =
    "replays/exp074_seed0_watch/game_iter51_game0_seed1787500020.replay.json";

/// EXACT match to scoring.rs's Build adj_count for Forge: owned tiles with an
/// already-BUILT Mine structure. Raw Mountain terrain does NOT count.
fn built_mine_partners(state: &polyfish::states::GameState, tile: i32, player: i32) -> usize {
    get_adjacent_indices(state, tile, 1)
        .iter()
        .filter(|&&i| {
            state.tiles.get(&i).map(|t| t.owner) == Some(player)
                && get_structure_at(state, i).map(|s| s.structure_type)
                    == Some(polyfish::types::StructureType::Mine)
        })
        .count()
}

fn mountain_terrain_adjacent(state: &polyfish::states::GameState, tile: i32) -> usize {
    get_adjacent_indices(state, tile, 1)
        .iter()
        .filter(|&&i| {
            state.tiles.get(&i).map(|t| t.terrain_type).unwrap_or(TerrainType::Ocean)
                == TerrainType::Mountain
        })
        .count()
}

fn legal_peek(game: &Game, move_type: i64, target: i64, reward_or_type: i64) -> Option<Box<dyn polyfish::moves::Move>> {
    game.legal_moves().into_iter().find(|m| {
        let j = m.serialize();
        j.get("moveType").and_then(|v| v.as_i64()) == Some(move_type)
            && j.get("target").and_then(|v| v.as_i64()) == Some(target)
            && j.get("type").and_then(|v| v.as_i64()) == Some(reward_or_type)
    })
}

fn dump_city(state: &polyfish::states::GameState, city_tile: i32, label: &str) {
    if let Some(c) = get_city_at(state, city_tile) {
        println!(
            "  [{label}] city@{} name={} lvl={} pop/prog={}/{} border_size={} territory_len={} rewards={:?}",
            c.idx, c.name, c.level, c.population, c.progress, c.border_size, c._territory.len(), c.rewards
        );
        let terr: BTreeSet<i32> = c._territory.iter().copied().collect();
        println!("      territory tiles: {:?}", terr);
    } else {
        println!("  [{label}] NO CITY at tile {city_tile}");
    }
}

fn dump_risks(state: &polyfish::states::GameState, pov: i32, label: &str) {
    for r in polyfish::ai::combat::city_risks(state, pov) {
        println!(
            "  [{label}] risk city={} risk={:.3} at_risk={} needs_order={} sieged={} open={} attackers={:?}",
            r.city, r.risk, r.at_risk, r.needs_order(), r.sieged, r.open, r.attackers
        );
    }
}

fn main() {
    let raw = std::fs::read_to_string(REPLAY).expect("read replay");
    let mut replay: ModReplay = serde_json::from_str(&raw).expect("parse replay");
    let pov = 1;

    let mut game = Game::new();
    game.state = replay.game_state.clone();
    game.post_load();

    // Watch windows: (start, end, tag) inclusive step indices (0-based, matching REPLAY_STEP_INDEX).
    let windows: Vec<(usize, usize, &str)> = vec![
        (88, 96, "ORG_TECH"),
        (118, 126, "REWARD_84"),
        (143, 151, "FORGE_SITE"),
        (173, 183, "GARRISON_49"),
    ];

    let mut idx = 0usize;
    for turn_data in &mut replay.turns {
        let turn_no = turn_data.turn;
        let mut players: Vec<_> = turn_data.players.iter_mut().collect();
        players.sort_by_key(|p| p.player_id);
        for pl in players {
            let pid = pl.player_id;
            for cmd_json in &pl.commands {
                let in_window = windows.iter().find(|(s, e, _)| idx >= *s && idx <= *e);
                if let Some((_, _, tag)) = in_window {
                    let cur_stars = game.state.tribes.get(&pid).map(|t| t.stars).unwrap_or(-1);
                    let move_type = cmd_json
                        .get("moveType")
                        .and_then(|v| v.as_i64())
                        .map(|v| format!("{:?}", MoveType::from(v as i32)))
                        .unwrap_or_else(|| "?".into());
                    println!(
                        "idx={idx:3} turn={turn_no:2} p{pid} stars={cur_stars:3} [{tag}] BEFORE move={move_type} raw={cmd_json}"
                    );
                    if tag == &"REWARD_84" {
                        dump_city(&game.state, 84, "city84-before");
                        if idx == 122 {
                            use polyfish::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
                            let goal = MacroGoal {
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
                            let phi_pre = polyfish::ai::reward::goal_potential(&game.state, pov, &goal, None);
                            for (rt, name) in [(8i64, "PopGrowth"), (5i64, "BorderGrowth")] {
                                if let Some(mv) = legal_peek(&game, 9, 84, rt) {
                                    let mut probe = Game { state: game.state.clone() };
                                    let base = polyfish::ai::scoring::score_move(&game, mv.as_ref());
                                    probe.play_move(mv.as_ref());
                                    let phi_post = polyfish::ai::reward::goal_potential(&probe.state, pov, &goal, None);
                                    println!(
                                        "      [reward candidate {name}] base={base:.3} dphi={:.4} total(lambda=1)={:.3}",
                                        phi_post - phi_pre, base + (phi_post - phi_pre)
                                    );
                                } else {
                                    println!("      [reward candidate {name}] NOT LEGAL at idx122");
                                }
                            }
                        }
                    }
                    if tag == &"FORGE_SITE" {
                        for t in [48, 61] {
                            let terr = state_terrain(&game.state, t);
                            let built = built_mine_partners(&game.state, t, pid);
                            let mtn = mountain_terrain_adjacent(&game.state, t);
                            let owner = game.state.tiles.get(&t).map(|x| x.owner).unwrap_or(-1);
                            println!(
                                "      tile {t}: terrain={terr:?} owner={owner} built_mine_adj={built} mountain_terrain_adj={mtn} structure={:?}",
                                get_structure_at(&game.state, t).map(|s| s.structure_type)
                            );
                            for adj in get_adjacent_indices(&game.state, t, 1) {
                                let at = state_terrain(&game.state, adj);
                                let ast = get_structure_at(&game.state, adj).map(|s| s.structure_type);
                                println!("        adj {adj}: terrain={at:?} structure={ast:?}");
                            }
                        }
                        dump_city(&game.state, 84, "city84-at-build");
                        // Also check which city (if any) owns tile 48 / 61.
                        for t in [48, 61] {
                            if let Some(owning) = polyfish::functions::get_city_owning_tile(&game.state, t) {
                                println!("      tile {t} owned by city@{}", owning.idx);
                            } else {
                                println!("      tile {t} not owned by any city");
                            }
                        }
                    }
                    if tag == &"GARRISON_49" {
                        dump_risks(&game.state, pov, "risk-before");
                        for r in polyfish::ai::combat::city_risks(&game.state, pov) {
                            if r.city == 49 {
                                let plan = polyfish::ai::combat::defend_plan(&game.state, pov, &r, &[]);
                                println!(
                                    "      [defend_plan-before city49] hold_needed={} shortfall={:.3} assigned={:?}",
                                    plan.hold_needed, plan.shortfall, plan.assigned
                                );
                            }
                        }
                        if let Some(c) = get_city_at(&game.state, 49) {
                            println!("      tile49 IS a city (name={} lvl={})", c.name, c.level);
                        } else {
                            println!("      tile49 is NOT a city");
                        }
                        let units_on_49: Vec<_> = game
                            .state
                            .tribes
                            .get(&pid)
                            .map(|t| {
                                t.units
                                    .iter()
                                    .filter(|u| u.coords.idx == 49)
                                    .map(|u| (u.unit_type, u.coords.idx))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        println!("      p{pid} units on tile49: {:?}", units_on_49);
                    }
                }

                let legal = game.legal_moves();
                if idx == 177 || idx == 178 {
                    let mut scored: Vec<(f32, String)> = legal
                        .iter()
                        .map(|m| (polyfish::ai::scoring::score_move(&game, m.as_ref()), format!("{:?} {}", m.move_type(), m.serialize())))
                        .collect();
                    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
                    println!("      [full ranked legal moves @ idx{idx}, top 15 of {}]", scored.len());
                    for (s, desc) in scored.iter().take(15) {
                        println!("        {s:8.3}  {desc}");
                    }
                    let summon_49: Vec<_> = scored.iter().filter(|(_, d)| d.starts_with("Summon") && d.contains("\"src\":49")).collect();
                    if summon_49.is_empty() {
                        println!("        (no legal Summon at city 49)");
                    } else {
                        for (s, d) in &summon_49 {
                            println!("        [SUMMON@49 candidate] {s:8.3}  {d}");
                        }
                    }
                    if let Some(pos) = scored.iter().position(|(_, d)| d.starts_with("Harvest")) {
                        println!("        (top Harvest is ranked #{} of {}: {})", pos + 1, scored.len(), scored[pos].1);
                    }
                }
                if idx == 179 {
                    use polyfish::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
                    let goal = MacroGoal {
                        orders: vec![
                            (OrderKind::Expand, 55),
                            (OrderKind::Defend, 41),
                            (OrderKind::Defend, 49),
                            (OrderKind::Defend, 84),
                        ],
                        stance: Stance::Save,
                        save_target: None,
                    };
                    let phi_pre = polyfish::ai::reward::goal_potential(&game.state, pov, &goal, None);
                    println!("      [phi_pre @ idx179] = {phi_pre:.4}");
                    for target in [48, 60, 37] {
                        if let Some(mv) = legal.iter().find(|m| {
                            m.move_type() == MoveType::Step
                                && m.serialize().get("src").and_then(|v| v.as_i64()) == Some(49)
                                && m.serialize().get("target").and_then(|v| v.as_i64()) == Some(target)
                        }) {
                            let mut probe = Game { state: game.state.clone() };
                            probe.play_move(mv.as_ref());
                            let phi_post = polyfish::ai::reward::goal_potential(&probe.state, pov, &goal, None);
                            let base = polyfish::ai::scoring::score_move(&game, mv.as_ref());
                            println!(
                                "      [candidate Step 49->{target}] base={base:.3} phi_pre={phi_pre:.4} phi_post={phi_post:.4} dphi={:.4} total(lambda=1)={:.3}",
                                phi_post - phi_pre, base + (phi_post - phi_pre)
                            );
                        }
                    }
                    if let Some(mv) = legal.iter().find(|m| {
                        m.move_type() == MoveType::Attack
                            && m.serialize().get("src").and_then(|v| v.as_i64()) == Some(49)
                    }) {
                        let mut probe = Game { state: game.state.clone() };
                        probe.play_move(mv.as_ref());
                        let phi_post = polyfish::ai::reward::goal_potential(&probe.state, pov, &goal, None);
                        let base = polyfish::ai::scoring::score_move(&game, mv.as_ref());
                        println!(
                            "      [candidate Attack 49->?] base={base:.3} phi_pre={phi_pre:.4} phi_post={phi_post:.4} dphi={:.4} total(lambda=1)={:.3}",
                            phi_post - phi_pre, base + (phi_post - phi_pre)
                        );
                    }
                }
                if idx == 179 {
                    let mut scored: Vec<(f32, String)> = legal
                        .iter()
                        .map(|m| (polyfish::ai::scoring::score_move(&game, m.as_ref()), format!("{:?} {}", m.move_type(), m.serialize())))
                        .collect();
                    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
                    println!("      [full ranked legal moves @ idx179, top 15 of {}]", scored.len());
                    for (s, desc) in scored.iter().take(15) {
                        println!("        {s:8.3}  {desc}");
                    }
                    if let Some(pos) = scored.iter().position(|(_, d)| d.contains("\"src\":49")) {
                        println!("        (Step from 49 is ranked #{} of {})", pos + 1, scored.len());
                    }
                }
                let matched = legal.iter().find(|m| &m.serialize() == cmd_json);
                match matched {
                    Some(m) => {
                        if game.play_move(m.as_ref()).is_none() {
                            println!("  !! EXEC FAILED idx={idx}");
                        }
                    }
                    None => {
                        println!("  !! idx={idx} RECORDED MOVE NOT IN LEGAL SET: {cmd_json}");
                    }
                }

                if let Some((_, _, tag)) = in_window {
                    if tag == &"GARRISON_49" {
                        dump_risks(&game.state, pov, "risk-after");
                        for r in polyfish::ai::combat::city_risks(&game.state, pov) {
                            if r.city == 49 {
                                let plan = polyfish::ai::combat::defend_plan(&game.state, pov, &r, &[]);
                                println!(
                                    "      [defend_plan-after city49] hold_needed={} shortfall={:.3} assigned={:?}",
                                    plan.hold_needed, plan.shortfall, plan.assigned
                                );
                            }
                        }
                    }
                }
                idx += 1;
            }
        }
    }

    println!("\n=== FINAL turn={} ===", game.state.settings.turn);
    dump_risks(&game.state, pov, "risk-final");
    // Trace tile 49's owner across the whole rest of the game by re-scanning
    // is expensive to add post-hoc; the per-window risk-after dumps above
    // plus a final snapshot should be enough to confirm siege/loss.
    if let Some(c) = get_city_at(&game.state, 49) {
        println!("tile49 final: city name={} owner={} lvl={}", c.name, c.owner, c.level);
    } else {
        println!("tile49 final: not a city (never was, or was razed)");
    }
}

fn state_terrain(state: &polyfish::states::GameState, idx: i32) -> TerrainType {
    state.tiles.get(&idx).map(|t| t.terrain_type).unwrap_or(TerrainType::Ocean)
}

#[allow(dead_code)]
fn unused_resource_ref() -> ResourceType {
    ResourceType::Metal
}
