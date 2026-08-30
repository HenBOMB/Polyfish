//! Per-turn JSONL diagnostics: the macro goal-setter snapshot, the macro
//! root ballot, and the EXPAND-plan tripwire that records whether each painted
//! plan was achieved, contested, or dropped.

use polyfish::ai::brain::Brain;
use polyfish::states::{GameState, PlayerId};
use serde_json::json;
use std::fs::File;
use std::io::Write;

/// Append one start-of-player-turn snapshot to <dir>/game<idx>.jsonl for the
/// multi-turn 3rd-city pursuit analysis: the acting player's owned cities,
/// FOW-visible uncaptured villages (open_villages seen by pov — same set the
/// ThirdCity trace uses), and unit tiles. Row-major 11x11 tile indices.
/// v7 belief tripwire: what actually happened to each painted EXPAND plan.
///
/// This is the discriminator for whether a belief state is the binding
/// constraint. If plans mostly die to enemies that were NOT visible when we
/// committed, the missing machinery is probabilistic opponent modelling. If
/// they mostly die to our own goal churn, the missing machinery is commitment,
/// and belief can wait.
#[derive(Default)]
pub(crate) struct PlanTracker {
    /// target tile -> (turn first painted, enemy already visible near it then)
    pub(crate) open: std::collections::HashMap<i32, (i32, bool)>,
    pub(crate) achieved: u32,
    pub(crate) contested_known: u32,
    pub(crate) contested_surprise: u32,
    pub(crate) dropped: u32,
}

/// A living enemy unit within `r` of `idx` that this seat can actually see.
pub(crate) fn enemy_visible_near(state: &GameState, pov: PlayerId, idx: i32, r: i32) -> bool {
    let w = state.settings.size as i32;
    if w == 0 {
        return false;
    }
    let (bx, by) = (idx % w, idx / w);
    state
        .tribes
        .iter()
        .filter(|(id, _)| **id != pov)
        .flat_map(|(_, t)| t.units.iter())
        .any(|u| {
            let ui = u.coords.idx;
            let (ax, ay) = (ui % w, ui / w);
            (ax - bx).abs().max((ay - by).abs()) <= r
                && state.tiles.get(&ui).map_or(false, |t| t.explorers.contains(&pov))
        })
}

/// Opens a record for every newly painted EXPAND target and resolves the ones
/// that left the goal since the last ply.
pub(crate) fn update_plans(
    state: &GameState,
    pov: PlayerId,
    goal: &polyfish::ai::oracle_macro::MacroGoal,
    pt: &mut PlanTracker,
) {
    use polyfish::ai::oracle_macro::OrderKind;
    let now: std::collections::HashSet<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Expand)
        .map(|(_, i)| *i)
        .collect();
    let turn = state.settings.turn;
    for &t in &now {
        pt.open.entry(t).or_insert_with(|| (turn, enemy_visible_near(state, pov, t, 3)));
    }
    let gone: Vec<i32> = pt.open.keys().copied().filter(|t| !now.contains(t)).collect();
    for t in gone {
        let Some((_, enemy_at_commit)) = pt.open.remove(&t) else { continue };
        if state.tiles.get(&t).map_or(false, |ti| ti.owner == pov) {
            pt.achieved += 1;
        } else if enemy_visible_near(state, pov, t, 2) {
            if enemy_at_commit {
                pt.contested_known += 1;
            } else {
                pt.contested_surprise += 1;
            }
        } else {
            pt.dropped += 1;
        }
    }
}

pub(crate) fn dump_turn_state(
    file: &mut File,
    game_idx: usize,
    state: &GameState,
    pov: PlayerId,
    open_villages: &std::collections::HashSet<i32>,
    lane_state: &polyfish::ai::oracle_macro::LaneState,
    // The macro agent's OWN Tier-1 state when this seat searches with
    // macro-mcts — a different `LaneState` than the script path's, and
    // the one that drove the ply, so it wins when present.
    macro_lane_state: Option<&polyfish::ai::oracle_macro::LaneState>,
    goal: Option<&polyfish::ai::oracle_macro::MacroGoal>,
    commit: &polyfish::ai::oracle_macro::StanceCommit,
    plans: &PlanTracker,
    tier3_bought: u32,
) {
    let Some(tribe) = state.tribes.get(&pov) else {
        return;
    };
    let cities: Vec<i32> = tribe.cities.iter().map(|c| c.idx).collect();
    let mut visible_villages: Vec<i32> = open_villages
        .iter()
        .copied()
        .filter(|idx| {
            state
                .tiles
                .get(idx)
                .map_or(false, |t| t.explorers.contains(&pov))
        })
        .collect();
    // Sorted: `open_villages` is a std HashSet, so its iteration order is
    // randomized per process and would otherwise leak into this dump.
    visible_villages.sort_unstable();
    let units: Vec<i32> = tribe.units.iter().map(|u| u.coords.idx).collect();
    let city_detail: Vec<serde_json::Value> = tribe
        .cities
        .iter()
        .map(|c| {
            json!({
                "idx": c.idx,
                "level": c.level,
                "progress": c.progress,
                "production": polyfish::functions::get_city_production(state, c),
                "connected": c.connected_to_capital,
            })
        })
        .collect();
    // Hub census: the multiplier-tier structures and how many partners each
    // actually pays on. `partner_count` is the hub's LEVEL in the sense that
    // matters — a Forge with one mine is a different building from a Forge
    // with four.
    let hubs: Vec<serde_json::Value> = {
        use polyfish::types::StructureType as S;
        const HUBS: [S; 4] = [S::Forge, S::Windmill, S::Sawmill, S::Market];
        tribe
            .cities
            .iter()
            .flat_map(|c| c._territory.iter().copied())
            .filter_map(|idx| {
                let s = polyfish::functions::get_structure_at(state, idx)?;
                if !HUBS.contains(&s.structure_type) {
                    return None;
                }
                Some(json!({
                    "idx": idx,
                    "type": format!("{:?}", s.structure_type),
                    "partners": polyfish::rules::economy::partner_count(
                        state, idx, s.structure_type, pov),
                }))
            })
            .collect()
    };
    // Stage 4 attribution: `ply <- order <- playstyle`. The lane is the root
    // cause, the orders are the middle tier, and both are recorded from the
    // state that actually drove this ply (dumped post-search, pre-move).
    let ps = macro_lane_state.unwrap_or(lane_state);
    let rec = json!({
        "game": game_idx,
        "turn": state.settings.turn,
        "player": pov,
        "playstyle": ps.lane.map(|a| format!("{a:?}")),
        "playstyle_source": if macro_lane_state.is_some() { "macro" } else { "script" },
        "playstyle_committed_turn": ps.committed_turn,
        "playstyle_pivots_used": ps.pivots_used,
        "lane_blocked_turns": ps.lane_blocked_turns,
        // In `oracle_macro::LANE_ORDER` order: RiderRoads, ArcherLine, SpamGiants.
        "playstyle_scores": ps.last_scores,
        "orders": goal.map(|g| {
            g.orders
                .iter()
                .map(|(kind, t)| json!({"kind": format!("{kind:?}"), "target": t}))
                .collect::<Vec<_>>()
        }),
        "cities": cities,
        "city_count": cities.len(),
        "city_detail": city_detail,
        "hubs": hubs,
        "connected_cities": tribe.cities.iter().filter(|c| c.connected_to_capital).count(),
        "visible_villages": visible_villages,
        "units": units,
        "seen_squishy": lane_state.seen_squishy,
        "seen_heavy": lane_state.seen_heavy,
        "seen_cavalry": lane_state.seen_cavalry,
        "knight_commit": lane_state.overlays.knight_commit,
        // v7 commitment + plan outcomes.
        "stance": goal.map(|g| format!("{:?}", g.stance)),
        "save_target": goal.and_then(|g| g.save_target.as_ref().map(|l| l.cost)),
        "save_lane": goal.and_then(|g| {
            g.save_target.as_ref().map(|l| format!("{:?}+{:?}", l.tech, l.structure))
        }),
        // Raw batch cost regardless of the SAVE gate: separates "no batch was
        // ever placeable" (the tier-3 tech wall) from "a batch existed but the
        // reachability gate rejected it". Without this a dead SAVE stance is
        // indistinguishable from a correctly quiet one.
        "save_batch": polyfish::ai::oracle_macro::pick_save_lane(state, pov, tier3_bought)
            .map(|l| l.cost),
        "stance_flips": commit.stance_flips,
        "order_flips": commit.order_flips,
        "turns_seen": commit.turns_seen,
        "plan_achieved": plans.achieved,
        "plan_contested_known": plans.contested_known,
        "plan_contested_surprise": plans.contested_surprise,
        "plan_dropped": plans.dropped,
    });
    if let Ok(s) = serde_json::to_string(&rec) {
        let _ = writeln!(file, "{s}");
    }
}

/// Stage 3b (macro policy head, first step): one JSON record per macro root
/// decision — the candidate ballot the tree searched and its own post-search
/// visit count per candidate, raw. `candidates`/`visits` are parallel arrays
/// (same indexing); no (stance/order/target) encoding decided yet — that
/// waits until there's real data to design the head shape against.
pub(crate) fn dump_macro_policy_row(
    file: &mut File,
    turn: i32,
    pov: PlayerId,
    candidates: &[polyfish::ai::oracle_macro::MacroGoal],
    visits: &[f32],
) {
    let cand_json: Vec<serde_json::Value> = candidates
        .iter()
        .map(|g| {
            json!({
                "stance": format!("{:?}", g.stance),
                "orders": g.orders.iter()
                    .map(|(kind, t)| json!([format!("{kind:?}"), t]))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let rec = json!({
        "turn": turn,
        "pov": pov,
        "candidates": cand_json,
        "visits": visits,
    });
    if let Ok(s) = serde_json::to_string(&rec) {
        let _ = writeln!(file, "{s}");
    }
}

/// Opens `<dir>/game<idx>.jsonl`, truncating. Distinct `game_idx` per actor
/// means no cross-thread contention on the handle. A tag names the flag in
/// the warning so a failed dump is attributable; failures are non-fatal --
/// diagnostics never abort a training run.
pub(crate) fn open_game_jsonl(dir: Option<&str>, tag: &str, game_idx: usize) -> Option<File> {
    let dir = dir?;
    let path = std::path::Path::new(dir);
    if let Err(e) = std::fs::create_dir_all(path) {
        eprintln!("[{tag}] failed to create {}: {e}", path.display());
        return None;
    }
    match File::create(path.join(format!("game{game_idx}.jsonl"))) {
        Ok(f) => Some(f),
        Err(e) => {
            eprintln!("[{tag}] failed to open game file: {e}");
            None
        }
    }
}

/// Writes the reward-choice and pop-spend candidate dumps for this ply.
///
/// Both consume `take_trace()`, so they only ever produce rows under a
/// backend that builds one (Gumbel); with a zero-search backend they are
/// silently inert.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_choice_dumps(
    state: &GameState,
    pov: PlayerId,
    agent: &mut Brain<'_>,
    reward_choice_ply: bool,
    pop_spend_ply: bool,
    reward_choice_file: &mut Option<File>,
    pop_spend_file: &mut Option<File>,
    pop_spend_dumped: &mut usize,
    last_pop_spend_turn: &mut i32,
) {
    if reward_choice_ply {
        if let Some(trace) = agent.take_trace() {
            if let Some(f) = reward_choice_file.as_mut() {
                let width = state.settings.size as usize;
                let revealed = state
                    
                    .tiles
                    .values()
                    .filter(|t| t.explorers.contains(&pov))
                    .count() as f32;
                let hidden_frac = (1.0 - revealed / (width * width) as f32).max(0.0);
                let cands: Vec<serde_json::Value> = trace
                    .candidates
                    .iter()
                    .map(|c| {
                        json!({
                            "desc": c.description,
                            "q": c.q_value,
                            "visits": c.visits,
                            "own_value": c.own_value,
                            "edge_reward": c.edge_reward,
                            "prior": c.search_prior_prob,
                            "raw_prob": c.raw_net_prob,
                        })
                    })
                    .collect();
                let row = json!({
                    "turn": state.settings.turn,
                    "player_id": pov,
                    "hidden_frac": hidden_frac,
                    "chosen_idx": trace.chosen_candidate_idx,
                    "root_value": trace.root_search_value,
                    "candidates": cands,
                });
                let _ = writeln!(f, "{row}");
            }
        }
    }

    if pop_spend_ply {
        if let Some(trace) = agent.take_trace() {
            if let Some(f) = pop_spend_file.as_mut() {
                const ECON_TYPES: [&str; 5] =
                    ["Harvest", "Build", "Summon", "Research", "EndTurn"];
                let cands: Vec<serde_json::Value> = trace
                    .candidates
                    .iter()
                    .filter(|c| ECON_TYPES.contains(&c.move_type.as_str()))
                    .map(|c| {
                        json!({
                            "desc": c.description,
                            "move_type": c.move_type,
                            "q": c.q_value,
                            "visits": c.visits,
                            "prior": c.search_prior_prob,
                            "own_value": c.own_value,
                        })
                    })
                    .collect();
                if !cands.is_empty() {
                    let chosen = trace
                        .candidates
                        .get(trace.chosen_candidate_idx)
                        .map(|c| c.description.clone())
                        .unwrap_or_default();
                    let tribe = state.tribes.get(&pov);
                    let cities: Vec<serde_json::Value> = tribe
                        .map(|t| {
                            t.cities
                                .iter()
                                .map(|c| json!({
                                    "idx": c.idx,
                                    "level": c.level,
                                    "progress": c.progress,
                                }))
                                .collect()
                        })
                        .unwrap_or_default();
                    let row = json!({
                        "turn": state.settings.turn,
                        "player_id": pov,
                        "stars": tribe.map(|t| t.stars).unwrap_or(0),
                        "units": tribe.map(|t| t.units.len()).unwrap_or(0),
                        "cities": cities,
                        "chosen": chosen,
                        "candidates": cands,
                    });
                    let _ = writeln!(f, "{row}");
                    *pop_spend_dumped += 1;
                    *last_pop_spend_turn = state.settings.turn;
                }
            }
        }
    }
}

/// Writes the post-move star-spend and level-completion rows.
///
/// Both are pre/post pairs around `play_move`: the caller snapshots the
/// pre-move state (`*_pre`) and this writes the row afterwards, so each row
/// reports the delta the move actually caused.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_spend_dumps(
    state: &GameState,
    pov: PlayerId,
    m_type: polyfish::types::MoveType,
    m: &dyn polyfish::moves::Move,
    game_idx: usize,
    star_spend_pre: Option<(i32, i32)>,
    level_completion_pre: Option<(i32, i32, i32, i32, i32, bool)>,
    star_spend_file: &mut Option<File>,
    level_completion_file: &mut Option<File>,
) {
    if let (Some((turn, stars_before)), Some(f)) =
        (star_spend_pre, star_spend_file.as_mut())
    {
        let stars_after = state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
        let line = json!({
            "game_idx": game_idx,
            "turn": turn,
            "player_id": pov,
            "move_type": format!("{:?}", m_type),
            "ability": (m_type == polyfish::types::MoveType::Ability)
                .then(|| m.ability_type().ok())
                .flatten()
                .map(|a| format!("{:?}", a)),
            // v7: tech identity + tier, so an audit can check the
            // economy-first tier-3 ordering directly. Its absence was
            // a standing gap (EXP_ELO_028 Phase 0 flagged it) that
            // forced the Chivalry-crowds-out-Construction read to lean
            // on best-games replays.
            "tech": (m_type == polyfish::types::MoveType::Research)
                .then(|| m.tech_type().ok())
                .flatten()
                .map(|t| format!("{:?}", t)),
            "tech_tier": (m_type == polyfish::types::MoveType::Research)
                .then(|| m.tech_type().ok())
                .flatten()
                .and_then(|t| {
                    polyfish::settings::technology::get_technology_setting(t).tier
                }),
            "tech_eco3": (m_type == polyfish::types::MoveType::Research)
                .then(|| m.tech_type().ok())
                .flatten()
                .map(polyfish::settings::technology::is_eco_tier3),
            "stars_spent": (stars_before - stars_after).max(0),
        });
        if let Ok(s) = serde_json::to_string(&line) {
                                let _ = writeln!(f, "{s}");
        }
    }
    if let (
        Some((target, city_idx, level_b, progress_b, stars_b, threatened)),
        Some(f),
    ) = (level_completion_pre, level_completion_file.as_mut())
    {
        let city = state
            
            .tribes
            .get(&pov)
            .and_then(|t| t.cities.iter().find(|c| c.idx == city_idx));
        if let Some(c) = city {
            let stars_after =
                state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
            let line = json!({
                "game_idx": game_idx,
                "turn": state.settings.turn,
                "player_id": pov,
                "move_type": format!("{:?}", m_type),
                "structure": (m_type == polyfish::types::MoveType::Build)
                    .then(|| m.structure_type().ok())
                    .flatten()
                    .map(|s| format!("{:?}", s)),
                "target": target,
                "city_idx": city_idx,
                "level_before": level_b,
                "level_after": c.level,
                "progress_before": progress_b,
                "progress_after": c.progress,
                "stars_before": stars_b,
                "stars_after": stars_after,
                "completes": c.level > level_b,
                "threatened": threatened,
            });
            if let Ok(s) = serde_json::to_string(&line) {
                                        let _ = writeln!(f, "{s}");
            }
        }
    }
}
