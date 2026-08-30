//! One head-to-head game. `swap` puts config2 in the P1 seat, so a
//! seed's two halves cover both seats and positional advantage cancels.
//! 
//! (`match` is a keyword, hence the file name.)

use polyfish::TribeType;
use polyfish::ai::brain::{SearchAgent, SearchBackend, make_search_agent};
use polyfish::ai::eval_server::Evaluator;
use polyfish::ai::macro_agent::MacroParams;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, ModeType};
use std::time::Instant;
use crate::dumps::{TurnSample, dump_turn_state, sample_turn};
use crate::{MID_DUMP_TURN, seat_tribes};
use crate::siege::SiegeTracker;

pub(crate) struct MatchResult {
    pub(crate) winner_config: u8,
    /// true = config 2 sat in the P1 seat this game.
    pub(crate) swap: bool,
    pub(crate) score_config1: i32,
    pub(crate) score_config2: i32,
    /// EXP_ELO_041 per config: (sieges suffered, unsieged, cities lost).
    pub(crate) siege_config1: (u32, u32, u32),
    pub(crate) siege_config2: (u32, u32, u32),
    pub(crate) ns_config1: u64,
    pub(crate) moves_config1: u64,
    pub(crate) ns_config2: u64,
    pub(crate) moves_config2: u64,
    /// Search telemetry for config1:
    /// (depth_sum, depth_count, depth_max, horizon_hits, agree, decisions).
    pub(crate) depth_config1: Option<(u64, u64, u32, u64, u64, u64)>,
    /// EXP_ELO_032, config1 macro-lookahead only: (divergent, planned) turns.
    pub(crate) macro_divergence: Option<(u32, u32)>,
    /// EXP_ELO_035, config1 belief-enabled macro-mcts only:
    /// (capital-materialized turns, units materialized, planned turns).
    pub(crate) belief_mat: Option<(u32, u32, u32)>,
    /// EXP_ELO_036/038, config1 macro-mcts: winning-candidate class counts
    /// (base/stance/real/attackCapital/claim/contest/continuation/
    /// attackWeakest/defendUrgent), belief-target re-picks, mid-turn
    /// fog-order strips.
    pub(crate) belief_gen: Option<(
        [u32; polyfish::ai::search::macro_agent::CANDIDATE_CLASSES],
        u32,
        u32,
    )>,
}

/// Play one game. `swap` puts config2 in the P1 seat and config1 in P2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn play_match(
    eval1: &Evaluator,
    eval2: &Evaluator,
    mcts1: usize,
    mcts2: usize,
    backend1: SearchBackend,
    backend2: SearchBackend,
    leaf_batch: Option<usize>,
    seed: i64,
    swap: bool,
    max_turns: i32,
    gamemode: u8,
    dump_stats_dir: Option<&str>,
    game_idx: usize,
    dump_turn_states: Option<&str>,
    macro_commit: bool,
    macro_star_gate: bool,
    goal_script: bool,
    goal_w_tree: f32,
    macro_params1: MacroParams,
    macro_params2: MacroParams,
    tribe1: TribeType,
    tribe2: TribeType,
    trace_tech: Option<polyfish::types::TechnologyType>,
    belief_calib: bool,
) -> MatchResult {
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        // Seat-indexed (P1, P2), NOT config-indexed -- see `seat_tribes`.
        // `swap` (below) only decides which config sits in which seat; it
        // must never touch this vec, or the free config/tribe fairness the
        // caller relies on across a seed's swapped pair breaks.
        tribes: seat_tribes(tribe1, tribe2),
        seed,
        ..Default::default()
    };

    let mut game = Game::new();
    game.state = generate(gen_settings);
    // generate() replaces the whole state (Game::new()'s own initial_seed
    // assignment doesn't survive it) and never sets initial_seed itself, so
    // every self-play/arena game was seeing initial_seed=0 regardless of
    // the real map seed -- fixed here rather than in mapgen so replay-load
    // (main.rs) keeps setting it explicitly from the recorded value.
    game.state.initial_seed = game.state.settings.seed;
    game.state.settings.mode =
        ModeType::from_repr(gamemode).unwrap_or(ModeType::Perfection);
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // p1_config / p2_config map each seat to its configuration so timing and
    // scores attribute to the right config when sides are swapped.
    let (mut agent_p1, p1_config, mut agent_p2, p2_config) = if swap {
        (
            make_search_agent(backend2, eval2, mcts2, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params2)),
            2u8,
            make_search_agent(backend1, eval1, mcts1, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params1)),
            1u8,
        )
    } else {
        (
            make_search_agent(backend1, eval1, mcts1, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params1)),
            1u8,
            make_search_agent(backend2, eval2, mcts2, leaf_batch, None, None, None, None, None, None, Some(false), Some(macro_params2)),
            2u8,
        )
    };

    // Config1 (gumbel/model) sits in the P1 seat unless swapped; the model_*
    // dump fields always describe this player, greedy_* the other, regardless
    // of who is acting.
    let model_player: polyfish::states::PlayerId = if swap { 2 } else { 1 };
    let greedy_player: polyfish::states::PlayerId = if swap { 1 } else { 2 };
    let mut turn_dump_file: Option<std::fs::File> = None;
    if let Some(dir) = dump_turn_states {
        match std::fs::File::create(std::path::Path::new(dir).join(format!("game{game_idx}.jsonl")))
        {
            Ok(f) => turn_dump_file = Some(f),
            Err(e) => eprintln!("[dump-turn-states] failed to open game{game_idx} file: {e}"),
        }
    }
    let mut last_dump_key: Option<(i32, polyfish::states::PlayerId)> = None;
    let mut mid_dumped = false;

    let mut moves = 0;
    let mut ns_config1: u64 = 0;
    let mut moves_config1: u64 = 0;
    let mut ns_config2: u64 = 0;
    let mut moves_config2: u64 = 0;
    let mut samples: Vec<TurnSample> = Vec::new();
    let mut last_sampled_turn = i32::MIN;
    // Hub-placement optimality: for every hub the model builds, the ceiling of
    // the tile it chose against the best ceiling legally available to it at
    // that instant. Measured AT THE DECISION, which end-state ceilings cannot
    // see -- by the last turn every buildable tile is built and chosen and best
    // collapse onto each other.
    let mut placements: Vec<serde_json::Value> = Vec::new();
    // EXP_ELO_026: config1's sticky expansion commitment (None = retired or
    // no capturable village visible). Tracked even in a gate-only arm, since
    // the gate is defined as active "while committed".
    let mut commitment: Option<i32> = None;
    // v2.3 tech-cap counters for the model seat (goal_script only).
    let mut techs_bought = 0u32;
    let mut tier3_bought = 0u32;
    // v3 lane doctrine state for the model seat.
    let mut lane_state = polyfish::ai::oracle_macro::LaneState::default();
    // v7: standing macro commitment for the model seat (mirrors self_play).
    let mut stance_commit = polyfish::ai::oracle_macro::StanceCommit::default();
    // One row per model ply: the goal the script set, and the move that
    // followed it. Separates "the plan was dropped" from "the plan was held
    // and ignored" -- the flip counters alone cannot tell those apart.
    /// Root traces are ~40 candidates plus every halving round each; cap them
    /// so one game's dump stays readable.
    const TRACE_CAP: usize = 12;
    let mut tech_traces: Vec<serde_json::Value> = Vec::new();
    let mut goal_trace: Vec<serde_json::Value> = Vec::new();
    let mut pending_goal: Option<serde_json::Value> = None;
    // EXP_ELO_034/035/036: the belief feed. The harness reads true state
    // solely to stream each observer its legal observables (and, for
    // --belief-calib, to log truth rows). Belief-enabled macro seats consume
    // clones per turn, per their MacroParams::belief_mode.
    let feed_on = |p: &MacroParams| p.belief_mode != polyfish::ai::macro_agent::BeliefMode::Off;
    let mut calib: Option<polyfish::ai::belief::CalibHarness> =
        if belief_calib || feed_on(&macro_params1) || feed_on(&macro_params2) {
            Some(polyfish::ai::belief::CalibHarness::new(&game.state))
        } else {
            None
        };
    let mut last_calib_key: Option<(i32, polyfish::states::PlayerId)> = None;
    // Which SEAT consumes belief (params are seat-swapped like configs).
    let mb_p1 = feed_on(if swap { &macro_params2 } else { &macro_params1 });
    let mb_p2 = feed_on(if swap { &macro_params1 } else { &macro_params2 });

    let mut siege_tracker = SiegeTracker::new(dump_stats_dir.is_some());
    while !polyfish::functions::is_game_over(&game.state) && moves < 500 {
        if dump_stats_dir.is_some() && game.state.settings.turn != last_sampled_turn {
            samples.push(sample_turn(&game.state, swap));
            last_sampled_turn = game.state.settings.turn;
        }
        let current_pid = game.state.settings.current_player_turn_id;

        // EXP_ELO_026: refresh the commitment before each of config1's
        // decisions — stars and cities change within a turn, and the target
        // may have been captured since the last ply.
        if (macro_commit || macro_star_gate) && current_pid == model_player {
            commitment =
                polyfish::ai::oracle_macro::update_commitment(&game.state, model_player, commitment);
            let model_agent = if swap { &mut agent_p2 } else { &mut agent_p1 };
            if let SearchAgent::Gumbel(a) = model_agent {
                a.pursuit_focus = if macro_commit { commitment } else { None };
                a.star_gate = macro_star_gate && commitment.is_some();
            }
        }

        // EXP_ELO_028: scripted goal channels for config1.
        if goal_script && current_pid == model_player {
            let goal = polyfish::ai::oracle_macro::commit_macro_goal(
                &game.state,
                model_player,
                &mut stance_commit,
                tier3_bought,
            );
            let gate =
                polyfish::ai::oracle_macro::tech_discipline_active(&game.state, model_player, &goal);
            polyfish::ai::oracle_macro::update_lane_state(&game.state, model_player, &mut lane_state);
            let aux = polyfish::ai::oracle_macro::compute_goal_aux(
                &game.state,
                model_player,
                &goal,
                techs_bought,
                tier3_bought,
                Some(&lane_state),
            );
            if dump_stats_dir.is_some() {
                // The uncommitted goal too: `commit_macro_goal` returns the stance
                // after hysteresis, so a script that wants to switch and a
                // script that is content look identical in the result alone.
                let fresh = polyfish::ai::oracle_macro::compute_macro_goal(
                    &game.state,
                    model_player,
                    tier3_bought,
                );
                let tribe = game.state.tribes.get(&model_player);
                pending_goal = Some(serde_json::json!({
                    "turn": game.state.settings.turn,
                    "stance": format!("{:?}", goal.stance),
                    "stance_fresh": format!("{:?}", fresh.stance),
                    "save_target": goal.save_target.as_ref().map(|l| l.cost),
                    "save_lane": goal.save_target.as_ref()
                        .map(|l| format!("{:?}+{:?}", l.tech, l.structure)),
                    "save_target_fresh": fresh.save_target.as_ref().map(|l| l.cost),
                    "orders": goal.orders.iter()
                        .map(|(k, i)| serde_json::json!([format!("{k:?}"), i]))
                        .collect::<Vec<_>>(),
                    "star_gate": gate,
                    "stars": tribe.map(|t| t.stars),
                    "spt": tribe.map(|t| polyfish::functions::get_tribe_spt(&game.state, t)),
                    "cities": tribe.map(|t| t.cities.len()),
                    "techs_bought": techs_bought,
                    "tier3_bought": tier3_bought,
                }));
            }
            let model_agent = if swap { &mut agent_p2 } else { &mut agent_p1 };
            if let SearchAgent::Gumbel(a) = model_agent {
                a.star_gate = gate;
                a.macro_goal = Some(goal);
                a.goal_shape_w = goal_w_tree;
                a.goal_aux = Some(aux);
            }
        }

        // A board frozen while the build-out is still live. Hubs are committed
        // turns 10-17, so the final board -- every buildable tile already
        // built -- is the least informative position to plan from.
        if let Some(dir) = dump_stats_dir {
            if !mid_dumped
                && game.state.settings.turn >= MID_DUMP_TURN
                && current_pid == model_player
            {
                mid_dumped = true;
                let p = std::path::Path::new(dir).join(format!(
                    "mid_{}_{}.json",
                    seed,
                    if swap { "b" } else { "a" }
                ));
                if let Ok(j) = serde_json::to_string(&game.state) {
                    let _ = std::fs::write(&p, j);
                }
            }
        }

        // Dump the start-of-turn ground-truth snapshot once per (turn, acting
        // player), before any move that turn mutates the state.
        if let Some(f) = turn_dump_file.as_mut() {
            let key = (game.state.settings.turn, current_pid);
            if last_dump_key != Some(key) {
                dump_turn_state(f, game_idx, &game.state, model_player, greedy_player);
                last_dump_key = Some(key);
            }
        }

        // EXP_ELO_034: belief-vs-truth row at the start of the acting
        // player's turn — the moment a planner would consume the belief.
        if let Some(c) = calib.as_mut() {
            if belief_calib {
                let key = (game.state.settings.turn, current_pid);
                if last_calib_key != Some(key) {
                    c.turn_row(&game.state, current_pid);
                    last_calib_key = Some(key);
                }
            }
        }

        // EXP_ELO_035: hand the acting belief-enabled macro seat its current
        // belief before it plans this ply's move.
        if (current_pid == 1 && mb_p1) || (current_pid == 2 && mb_p2) {
            if let Some(b) = calib.as_ref().and_then(|c| c.belief_for(current_pid)) {
                let agent = if current_pid == 1 { &mut agent_p1 } else { &mut agent_p2 };
                if let SearchAgent::MacroMcts(a) = agent {
                    a.set_belief(b.clone());
                }
            }
        }

        // Arm the root trace only on plies where `trace_tech` is a live
        // choice: prerequisite owned, tech not yet bought, cost affordable.
        let mut armed_ctx: Option<serde_json::Value> = None;
        if let Some(tech) = trace_tech {
            if current_pid == model_player && tech_traces.len() < TRACE_CAP {
                if let Some(t) = game.state.tribes.get(&model_player) {
                    use polyfish::settings::technology as tech_mod;
                    let owned = tech_mod::has_technology(&t.tech_vanilla, tech);
                    let prereq = tech_mod::get_technology_setting(tech)
                        .requires
                        .map_or(true, |r| tech_mod::has_technology(&t.tech_vanilla, r));
                    let cost = tech_mod::get_tech_cost(
                        t.cities.len() as i32,
                        tech_mod::tech_tier(tech),
                        tech_mod::has_technology(
                            &t.tech_vanilla,
                            polyfish::types::TechnologyType::Philosophy,
                        ),
                    );
                    if !owned && prereq && t.stars >= cost {
                        armed_ctx = Some(serde_json::json!({
                            "turn": game.state.settings.turn,
                            "stars": t.stars,
                            "cost": cost,
                            "cities": t.cities.len(),
                            "spt": polyfish::functions::get_tribe_spt(&game.state, t),
                        }));
                        let a = if swap { &mut agent_p2 } else { &mut agent_p1 };
                        if let SearchAgent::Gumbel(g) = a {
                            g.arm_trace();
                        }
                    }
                }
            }
        }

        let t0 = Instant::now();
        // Search on a clone: MCTS execute/undo must never touch the scored
        // state (Brain::think_decomposed clones for the same reason).
        let best_move = if current_pid == 1 {
            agent_p1.select_move(&mut game.clone())
        } else {
            agent_p2.select_move(&mut game.clone())
        };
        let dt = t0.elapsed().as_nanos() as u64;

        // Stage 4, the macro path's own trace: `ply <- order <- playstyle`.
        // The scripted emitter above is gated on `--goal-script`, which arena
        // refuses to combine with a non-Gumbel backend — so it never fires
        // here. Same row schema, so downstream analysis stays one parser;
        // filled AFTER the search because the macro agent commits its lane and
        // directive during it.
        if dump_stats_dir.is_some() && current_pid == model_player {
            let model_agent = if swap { &agent_p2 } else { &agent_p1 };
            if let Some(ps) = model_agent.macro_playstyle() {
                let goal = model_agent.macro_committed_goal();
                let tribe = game.state.tribes.get(&model_player);
                pending_goal = Some(serde_json::json!({
                    "turn": game.state.settings.turn,
                    "source": "macro",
                    "playstyle": ps.lane.map(|a| format!("{a:?}")),
                    "playstyle_committed_turn": ps.committed_turn,
                    "playstyle_pivots_used": ps.pivots_used,
                    "lane_blocked_turns": ps.lane_blocked_turns,
                    // oracle_macro::LANE_ORDER: RiderRoads, ArcherLine, SpamGiants.
                    "playstyle_scores": ps.last_scores,
                    "stance": goal.map(|g| format!("{:?}", g.stance)),
                    "save_target": goal.and_then(|g| g.save_target.as_ref().map(|l| l.cost)),
                    "save_lane": goal.and_then(|g| {
                        g.save_target.as_ref().map(|l| format!("{:?}+{:?}", l.tech, l.structure))
                    }),
                    "orders": goal.map(|g| {
                        g.orders
                            .iter()
                            .map(|(k, i)| serde_json::json!([format!("{k:?}"), i]))
                            .collect::<Vec<_>>()
                    }),
                    "stars": tribe.map(|t| t.stars),
                    "spt": tribe.map(|t| polyfish::functions::get_tribe_spt(&game.state, t)),
                    "cities": tribe.map(|t| t.cities.len()),
                    "techs_bought": techs_bought,
                    "tier3_bought": tier3_bought,
                }));
            }
        }

        if let Some(mut ctx) = armed_ctx {
            let a = if swap { &mut agent_p2 } else { &mut agent_p1 };
            if let SearchAgent::Gumbel(g) = a {
                if let Some(tr) = g.take_trace() {
                    ctx["stance"] = pending_goal
                        .as_ref()
                        .and_then(|p| p.get("stance").cloned())
                        .unwrap_or(serde_json::Value::Null);
                    ctx["save_target"] = pending_goal
                        .as_ref()
                        .and_then(|p| p.get("save_target").cloned())
                        .unwrap_or(serde_json::Value::Null);
                    ctx["trace"] = serde_json::to_value(&tr).unwrap_or_default();
                    tech_traces.push(ctx);
                }
            }
        }

        let cfg = if current_pid == 1 { p1_config } else { p2_config };
        if cfg == 1 {
            ns_config1 += dt;
            moves_config1 += 1;
        } else {
            ns_config2 += dt;
            moves_config2 += 1;
        }

        if let Some(mut row) = pending_goal.take() {
            let (kind, desc) = match &best_move {
                Some(m) => (
                    format!("{:?}", m.move_type()),
                    m.describe(&game.state),
                ),
                None => ("None".to_string(), String::new()),
            };
            row["move_type"] = serde_json::json!(kind);
            row["move"] = serde_json::json!(desc);
            row["tech"] = match best_move.as_ref().and_then(|m| m.tech_type().ok()) {
                Some(t) => serde_json::json!(format!("{t:?}")),
                None => serde_json::Value::Null,
            };
            row["structure"] = match best_move.as_ref().and_then(|m| m.structure_type().ok()) {
                Some(s) => serde_json::json!(format!("{s:?}")),
                None => serde_json::Value::Null,
            };
            row["unit"] = match best_move.as_ref().and_then(|m| m.unit_type().ok()) {
                Some(u) => serde_json::json!(format!("{u:?}")),
                None => serde_json::Value::Null,
            };
            goal_trace.push(row);
        }

        if let Some(m) = best_move {
            if current_pid == model_player
                && m.move_type() == polyfish::types::MoveType::Build
            {
                if let (Ok(kind), Ok(tile)) = (m.structure_type(), m.target_idx()) {
                    let setting =
                        polyfish::settings::structures::get_structure_setting(kind);
                    if setting.reward_pop > 0 && !setting.adjacent_types.is_empty() {
                        let chosen = tile as i32;
                        let chosen_ceiling = polyfish::rules::economy::partner_ceiling(
                            &game.state, chosen, kind, model_player,
                        );
                        // Every tile this same hub could legally go on right now.
                        let mut alts: Vec<(i32, i32)> = Vec::new();
                        for cand in polyfish::moves::generate_legal_moves(&game.state) {
                            if cand.move_type() != polyfish::types::MoveType::Build {
                                continue;
                            }
                            if cand.structure_type().ok() != Some(kind) {
                                continue;
                            }
                            if let Ok(t) = cand.target_idx() {
                                let t = t as i32;
                                // limited_per_city: only this city's tiles are
                                // alternatives to this city's placement.
                                if !polyfish::rules::economy::same_city(
                                    &game.state, t, chosen,
                                ) {
                                    continue;
                                }
                                alts.push((
                                    t,
                                    polyfish::rules::economy::partner_ceiling(
                                        &game.state, t, kind, model_player,
                                    ),
                                ));
                            }
                        }
                        let best = alts.iter().map(|&(_, c)| c).max().unwrap_or(chosen_ceiling);
                        let best_tile = alts
                            .iter()
                            .filter(|&&(_, c)| c == best)
                            .map(|&(t, _)| t)
                            .min()
                            .unwrap_or(chosen);
                        // What the model traded away. A tile can be a poor hub
                        // site and still be worth keeping -- it may carry a
                        // resource the hub would crush, or be a partner slot.
                        let describe = |t: i32| {
                            let res = game
                                .state
                                .resources
                                .get(&t)
                                .and_then(|r| r.as_ref())
                                .map(|r| format!("{:?}", r.resource_type));
                            let terr = game
                                .state
                                .tiles
                                .get(&t)
                                .map(|x| format!("{:?}", x.terrain_type));
                            let city = polyfish::functions::get_city_owning_tile(&game.state, t)
                                .map(|c| c.idx);
                            serde_json::json!({
                                "tile": t,
                                "terrain": terr,
                                "resource": res,
                                "city": city,
                                "dist_to_city": city.map(|c| {
                                    polyfish::functions::get_chebyshev_distance(
                                        t, c, game.state.settings.size)
                                }),
                            })
                        };
                        placements.push(serde_json::json!({
                            "turn": game.state.settings.turn,
                            "kind": format!("{kind:?}"),
                            "tile": chosen,
                            "chosen_ceiling": chosen_ceiling,
                            "best_ceiling": best,
                            "n_options": alts.len(),
                            "chosen_detail": describe(chosen),
                            "best_detail": describe(best_tile),
                            "stars": game.state.tribes.get(&model_player).map(|t| t.stars),
                        }));
                    }
                }
            }
            if goal_script
                && current_pid == model_player
                && m.move_type() == polyfish::types::MoveType::Research
            {
                techs_bought += 1;
                if let Ok(tech) = m.tech_type() {
                    if polyfish::settings::technology::get_technology_setting(tech).tier
                        == Some(3)
                    {
                        tier3_bought += 1;
                    }
                }
            }
            game.play_move(m.as_ref());
            {
                // The committed directive of each seat, so an episode records
                // whether Tier 2 had even named this city as one to defend.
                let g1 = agent_p1.macro_committed_goal();
                let g2 = agent_p2.macro_committed_goal();
                siege_tracker.scan(&game.state, [g1, g2]);
            }
            if let Some(c) = calib.as_mut() {
                c.after_move(&game.state, current_pid, m.as_ref());
            }
        } else {
            break;
        }
        moves += 1;
    }

    let p1_score = game.state.tribes.get(&1).map(|t| t.score).unwrap_or(0);
    let p2_score = game.state.tribes.get(&2).map(|t| t.score).unwrap_or(0);

    let (score_config1, score_config2) = if swap {
        (p2_score, p1_score)
    } else {
        (p1_score, p2_score)
    };

    let seat_siege = |i: usize| {
        (
            siege_tracker.sieges[i],
            siege_tracker.unsieged[i],
            siege_tracker.lost[i],
        )
    };
    let (siege_config1, siege_config2) = if swap {
        (seat_siege(1), seat_siege(0))
    } else {
        (seat_siege(0), seat_siege(1))
    };

    let winner_config = if score_config1 > score_config2 {
        1
    } else if score_config2 > score_config1 {
        2
    } else {
        0
    };

    // EXP_ELO_032: how often lookahead overrode the scripted base directive.
    // A flat lookahead-vs-script result is uninterpretable without this.
    let macro_divergence = {
        let model_agent = if swap { &agent_p2 } else { &agent_p1 };
        match model_agent {
            SearchAgent::MacroLookahead(a) => Some((a.divergent_turns, a.planned_turns)),
            SearchAgent::MacroMcts(a) => Some((a.divergent_turns, a.planned_turns)),
            _ => None,
        }
    };
    // EXP_ELO_035: how often/how much materialization actually ran for
    // config1 — a flat belief-vs-baseline result is uninterpretable without
    // it (the posterior confirms fast, so the window may be turns 0-10 only).
    let belief_mat = {
        let model_agent = if swap { &agent_p2 } else { &agent_p1 };
        match model_agent {
            SearchAgent::MacroMcts(a) if a.belief.is_some() => {
                Some((a.mat_capital_turns, a.mat_units, a.planned_turns))
            }
            _ => None,
        }
    };
    // EXP_ELO_036: which candidate class won each planned turn for config1,
    // plus consecutive-turn re-picks of the same belief fog target.
    let belief_gen = {
        let model_agent = if swap { &agent_p2 } else { &agent_p1 };
        match model_agent {
            SearchAgent::MacroMcts(a) => {
                Some((a.class_picks, a.belief_repicks, a.intra_strips))
            }
            _ => None,
        }
    };

    if let Some(dir) = dump_stats_dir {
        samples.push(sample_turn(&game.state, swap)); // final post-game state
        // End-state build-out for the model seat: what it actually put on the
        // board and at what level, so a game can be held against eco_plan's
        // frontier instead of inferred from SPT alone.
        let model_pid: polyfish::states::PlayerId = if swap { 2 } else { 1 };
        let model_tribe = game.state.tribes.get(&model_pid);
        let mut territory: Vec<i32> = model_tribe
            .map(|t| t.cities.iter().flat_map(|c| c._territory.iter().copied()).collect())
            .unwrap_or_default();
        territory.sort_unstable();
        territory.dedup();
        let model_structures: Vec<serde_json::Value> = territory
            .iter()
            .filter_map(|&idx| {
                let st = polyfish::functions::get_structure_at(&game.state, idx)?;
                let setting =
                    polyfish::settings::structures::get_structure_setting(st.structure_type);
                // For an adjacency hub: what the site could ever collect vs what
                // it did. Separates a bad tile (low ceiling) from an unfinished
                // one (high ceiling, few partners built).
                let ceiling = if setting.adjacent_types.is_empty() {
                    -1
                } else {
                    polyfish::rules::economy::partner_ceiling(
                        &game.state,
                        idx,
                        st.structure_type,
                        model_pid,
                    )
                };
                let realized = if setting.adjacent_types.is_empty() {
                    -1
                } else {
                    polyfish::rules::economy::partner_count(
                        &game.state,
                        idx,
                        st.structure_type,
                        model_pid,
                    )
                };
                Some(serde_json::json!({
                    "idx": idx,
                    "type": format!("{:?}", st.structure_type),
                    "level": st.level,
                    "ceiling": ceiling,
                    "realized": realized,
                }))
            })
            .collect();
        let model_techs: Vec<String> = model_tribe
            .map(|t| {
                t.tech_vanilla
                    .iter()
                    .filter(|x| x.discovered)
                    .map(|x| format!("{:?}", x.tech_type))
                    .collect()
            })
            .unwrap_or_default();
        let model_city_levels: Vec<serde_json::Value> = model_tribe
            .map(|t| {
                t.cities
                    .iter()
                    .map(|c| serde_json::json!({
                        "idx": c.idx,
                        "level": c.level,
                        "pop": c.population,
                        // Which side of each level's fork the city took. The
                        // level-3 slot is PopGrowth vs BorderGrowth, and border
                        // is what grows the territory a hub's ceiling lives in.
                        "rewards": c.rewards.iter().map(|r| format!("{r:?}")).collect::<Vec<_>>(),
                    }))
                    .collect()
            })
            .unwrap_or_default();
        // config1 sits in the P2 seat when sides are swapped.
        let macro_playstyle =
            if swap { agent_p2.macro_playstyle() } else { agent_p1.macro_playstyle() }.cloned();
        let dump = serde_json::json!({
            "seed": seed,
            "swap": swap,
            "winner_config": winner_config,
            "score_config1": score_config1,
            "score_config2": score_config2,
            "sieges_config1": siege_config1.0,
            "unsieged_config1": siege_config1.1,
            "cities_lost_config1": siege_config1.2,
            "sieges_config2": siege_config2.0,
            "unsieged_config2": siege_config2.1,
            "cities_lost_config2": siege_config2.2,
            "macro_commit": macro_commit,
            "macro_star_gate": macro_star_gate,
            "samples": samples,
            "model_structures": model_structures,
            "model_techs": model_techs,
            "model_city_levels": model_city_levels,
            "placements": placements,
            "goal_trace": goal_trace,
            // EXP_ELO_049: one record per siege episode — the facts at the
            // moment the attacker stepped on, and how it ended.
            "siege_episodes": siege_tracker.episodes,
            "tech_traces": tech_traces,
            "stance_flips": stance_commit.stance_flips,
            "order_flips": stance_commit.order_flips,
            "goal_turns_seen": stance_commit.turns_seen,
            // EXP_ELO_045a: Tier-1 telemetry for config1's macro seat — the
            // committed lane and how stable it was.
            "playstyle": macro_playstyle
                .as_ref()
                .and_then(|p| p.lane.map(|a| format!("{a:?}"))),
            "playstyle_pivots_used": macro_playstyle.as_ref().map(|p| p.pivots_used),
            "playstyle_committed_turn": macro_playstyle.as_ref().and_then(|p| p.committed_turn),
            "playstyle_scores": macro_playstyle.as_ref().map(|p| p.last_scores.to_vec()),
            "macro_divergent_turns": macro_divergence.map(|(d, _)| d),
            "macro_planned_turns": macro_divergence.map(|(_, p)| p),
            "belief_calib": calib.as_ref().map(|c| c.rows.clone()),
            "mat_capital_turns": belief_mat.map(|(c, _, _)| c),
            "mat_units": belief_mat.map(|(_, u, _)| u),
            "class_picks": belief_gen.map(|(c, _, _)| c.to_vec()),
            "belief_repicks": belief_gen.map(|(_, r, _)| r),
            "intra_strips": belief_gen.map(|(_, _, s)| s),
        });
        // Drop the whole final board next to the summary. The server loads a
        // bare GameState, so the partner count around a hub can be counted off
        // the map rather than trusted. Unconditional: gating this on the model
        // holding a hub conditioned the board sample on the very thing an
        // economy audit is trying to measure.
        let sp = std::path::Path::new(dir)
            .join(format!("state_{}_{}.json", seed, if swap { "b" } else { "a" }));
        if let Ok(j) = serde_json::to_string(&game.state) {
            let _ = std::fs::write(&sp, j);
        }
        let name = format!("game_{}_{}.json", seed, if swap { "b" } else { "a" });
        let path = std::path::Path::new(dir).join(name);
        match serde_json::to_vec_pretty(&dump) {
            Ok(bytes) => {
                if let Err(e) = std::fs::write(&path, bytes) {
                    eprintln!("[dump-stats] failed to write {}: {e}", path.display());
                }
            }
            Err(e) => eprintln!("[dump-stats] failed to serialize seed {seed}: {e}"),
        }
    }

    // config1 sits in the P2 seat when sides are swapped.
    let depth_config1 = if swap {
        agent_p2.depth_stats()
    } else {
        agent_p1.depth_stats()
    };

    MatchResult {
        winner_config,
        swap,
        score_config1,
        score_config2,
        siege_config1,
        siege_config2,
        ns_config1,
        moves_config1,
        ns_config2,
        moves_config2,
        depth_config1,
        macro_divergence,
        belief_mat,
        belief_gen,
    }
}
