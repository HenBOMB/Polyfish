//! One self-play game, start to finish: build the map and the two agents,
//! run the ply loop, and hand back a `GameResult`.
//!
//! The per-ply diagnostics this drives live in `dumps` and `traces`; the
//! trackers it threads live in `stats` and `tempo`. Labels are NOT computed
//! here -- `GameResult.history` goes back raw and `dataset` turns it into
//! training targets.

use candle_core::Device;
use polyfish::TribeType;
use polyfish::ai::brain::{Brain, SearchBackend};
use polyfish::ai::eval_backend::{self, EvalBackendKind};
use polyfish::ai::eval_server::Evaluator;
use polyfish::ai::features;
use polyfish::ai::macro_agent::MacroParams;
use polyfish::ai::network::PolyZeroNet;
use polyfish::ai::reward;
use polyfish::game::Game;
use polyfish::replayer::ModReplay;
use polyfish::states::PlayerId;
use polyfish::types::MapSize;
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::crutches::{HEURISTIC_PRIOR_DECAY, HEURISTIC_PRIOR_W0, decay_crutch};
use crate::dumps::{PlanTracker, dump_macro_policy_row, dump_turn_state, update_plans};
use crate::labels::{POLICY_TARGET_Q_RAMP_ITERS, macro_ballot_for_history_step,
                    enemy_unit_grid, tech_multihot};
use crate::result::{DecomposedPolicyData, GameResult, HistoryStep, decompose_visits,
                    group_recap};
use crate::stats::{is_net_seat, record_spt_at_turn_start, t2c_turn, turn_milestones};
use crate::tempo::{TempoTrack, tempo_sample, unit_tally};
use crate::traces::{TraceTrigger, TracedDecision, dump_failed_game, find_harvest_trigger,
                    find_village_pursuit_trigger, find_village_trigger, find_wander_trigger,
                    write_decision_trace};
use crate::ProgressMode;

/// Load the main network (and opponent network, defaulting to the main one)
/// onto the given device from `model.safetensors`.
///
/// When `eval_backend_kind` is `Candle` and a distinct opponent is given, the
/// opponent network is loaded on its own freshly-obtained device rather than
/// `device`: under Candle, player 1 and player 2 each get an independent
/// `EvalServer` thread, and candle's Metal backend corrupts if two threads
/// encode ops (e.g. `forward_t`) against the same `Device` (see
/// `eval_backend.rs`'s device-isolation contract). tch/metal shards load
/// their own weights on the eval-server thread and never touch this candle
/// device for inference, so sharing is harmless for them.
pub(crate) fn load_networks(
    device: &Device,
    opponent: Option<&str>,
    eval_backend_kind: EvalBackendKind,
) -> anyhow::Result<(Arc<PolyZeroNet>, Arc<PolyZeroNet>)> {
    let model_path = "model.safetensors";
    if !std::path::Path::new(model_path).exists() {
        anyhow::bail!(
            "Model file {} not found! Please run init_model.py first.",
            model_path
        );
    }
    // Inference-only load: `VarBuilder::from_mmaped_safetensors` loads by key from file;
    // VarMap::load fills only pre-registered vars.
    let vs1 = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[model_path],
            candle_core::DType::F32,
            device,
        )?
    };
    let network1 = Arc::new(PolyZeroNet::new(vs1)?);

    let network2 = if let Some(opp_path) = opponent {
        let device2 = if eval_backend_kind == EvalBackendKind::Candle {
            eval_backend::select_device()?
        } else {
            device.clone()
        };
        let vs2 = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[opp_path],
                candle_core::DType::F32,
                &device2,
            )?
        };
        Arc::new(PolyZeroNet::new(vs2)?)
    } else {
        network1.clone()
    };
    Ok((network1, network2))
}

/// Play a single game and return the result
#[allow(clippy::too_many_arguments)]
pub(crate) fn play_single_game(
    network1: &PolyZeroNet,
    network2: &PolyZeroNet, // Added network2
    eval1: &Evaluator,
    eval2: &Evaluator,
    mcts_iters: usize,
    game_idx: usize,
    seed: i64,
    tribes: Vec<TribeType>,
    iteration: usize,
    decay_last_iter: usize,
    force_zero_crutches: bool,
    gamemode: u8,
    backend1: SearchBackend,
    backend2: SearchBackend,
    value_trust: Option<f32>,
    leaf_batch: Option<usize>,
    progress: ProgressMode,
    trace_villages: bool,
    trace_trigger: TraceTrigger,
    trace_max: usize,
    trace_counter: &AtomicUsize,
    dump_failed_dir: Option<&str>,
    dump_games_dir: Option<&str>,
    dump_turn_states: Option<&str>,
    dump_city_rewards: Option<&str>,
    dump_star_spend: Option<&str>,
    dump_reward_choices: Option<&str>,
    dump_level_completion: Option<&str>,
    dump_pop_spend_choices: Option<&str>,
    dump_macro_policy: Option<&str>,
    seat_roles: [&'static str; 2],
    shape_w_label: f32,
    shape_w_tree: f32,
    pursuit_w_label: f32,
    pursuit_w_tree: f32,
    unfreeze_opponent: bool,
    dagger_alpha: f32,
    goal_channels: bool,
    goal_w_tree: f32,
    macro_params: MacroParams,
    max_turns: i32,
) -> Option<GameResult> {
    // Verdi: drop the turn-count ramp — Tiny maps, flat 50-turn cap
    // regardless of iteration (was 10/15/20/30 ramping by iteration; games
    // this short couldn't mature a hub economy or a giants push). max_turns
    // is now a CLI override (default 50, unchanged) for throughput
    // experiments that don't need full-maturity games — see EXP_ELO_061's
    // throughput investigation.
    let map_size = MapSize::Tiny;

    // Init Game using MapGen
    let gen_settings = polyfish::mapgen::MapGenSettings {
        size: map_size,
        map_type: polyfish::types::MapType::Drylands,
        tribes: tribes.clone(),
        seed,
        ..Default::default()
    };
    if progress == ProgressMode::Full {
        eprintln!(
            "[Game {}] Started with seed: {} Tribes: {:?} (Curriculum: {:?}, max_turns: {})",
            game_idx, seed, gen_settings.tribes, map_size, max_turns
        );
    }

    let mut game = Game::new();
    game.state = polyfish::mapgen::generate(gen_settings);
    // generate() replaces the whole state (Game::new()'s own initial_seed
    // assignment doesn't survive it) and never sets initial_seed itself, so
    // every self-play/arena game was seeing initial_seed=0 regardless of
    // the real map seed -- fixed here rather than in mapgen so replay-load
    // (main.rs) keeps setting it explicitly from the recorded value.
    game.state.initial_seed = game.state.settings.seed;
    game.state.settings.mode = polyfish::types::ModeType::from_repr(gamemode)
        .unwrap_or(polyfish::types::ModeType::Perfection);
    game.state.settings.max_turns = max_turns;
    game.post_load();

    // Time-to-capture tracking: snapshot the map's initial open villages and
    // ruins, then record the turn each one is taken (by either player).
    let mut open_villages: std::collections::HashSet<i32> = std::collections::HashSet::new();
    let mut open_ruins: std::collections::HashSet<i32> = std::collections::HashSet::new();
    for (&idx, s) in game.state.structures.iter() {
        let Some(s) = s else { continue };
        match s.structure_type {
            polyfish::types::StructureType::Village
                if game.state.tiles.get(&idx).map_or(false, |t| t.owner == 0) =>
            {
                open_villages.insert(idx);
            }
            polyfish::types::StructureType::Ruin => {
                open_ruins.insert(idx);
            }
            _ => {}
        }
    }
    let initial_villages = open_villages.len();
    let initial_ruins = open_ruins.len();
    let mut village_capture_turns: Vec<i32> = Vec::new();
    let mut ruin_capture_turns: Vec<i32> = Vec::new();
    // Turn of each net seat's OWN first village capture. The pooled
    // `village_capture_turns` above cannot answer this: a mirror game puts two
    // net seats in one list, an anchor game one, so any rate built from it is a
    // blend of two different per-seat probabilities.
    let mut first_village_turn: HashMap<PlayerId, i32> = HashMap::new();

    // --dump-failed-dir: trace every decision; the log is written out only
    // if the game ends with zero village captures.
    let trace_all = dump_failed_dir.is_some() || dump_games_dir.is_some();
    let mut decision_log: Vec<TracedDecision> = Vec::new();

    // --dump-turn-states: one JSONL file per game, one record per player-turn
    // (written post-search, pre-move — see the Stage 4 dump below). Distinct
    // game_idx => no cross-actor contention; created/truncated once here.
    let mut turn_dump_file: Option<File> = None;
    if let Some(dir) = dump_turn_states {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-turn-states] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => turn_dump_file = Some(f),
                Err(e) => eprintln!("[dump-turn-states] failed to open game file: {e}"),
            }
        }
    }
    let mut last_dump_key: Option<(i32, PlayerId)> = None;

    // --dump-macro-policy: one JSONL file per game, one record per macro
    // root decision (Stage 3b first step — see the Stage 4 dump below for
    // the write, same once-per-(turn,pov) dedup as turn_dump_file).
    let mut macro_policy_file: Option<File> = None;
    if let Some(dir) = dump_macro_policy {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-macro-policy] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => macro_policy_file = Some(f),
                Err(e) => eprintln!("[dump-macro-policy] failed to open game file: {e}"),
            }
        }
    }
    let mut last_macro_policy_key: Option<(i32, PlayerId)> = None;
    // Separate from `last_macro_policy_key`: that tracker only advances
    // inside the `--dump-macro-policy` branch, which is `None` (off) during
    // real training runs -- reusing it here would silently disable this
    // dedup whenever the diagnostic dump isn't also requested.
    let mut last_macro_ballot_key: Option<(i32, PlayerId)> = None;

    // --dump-city-rewards: one JSONL file per game, one record per city
    // level-up reward choice — (turn, player, city level pre-choice, tribe
    // stars at time of choice, reward type chosen). Reward moves are always
    // forced (generate_reward_moves preempts everything else when a choice
    // is pending — moves/mod.rs), so this is a clean, uncontested read of
    // what the policy actually wants at each level, no Step competition.
    let mut city_reward_file: Option<File> = None;
    if let Some(dir) = dump_city_rewards {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-city-rewards] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => city_reward_file = Some(f),
                Err(e) => eprintln!("[dump-city-rewards] failed to open game file: {e}"),
            }
        }
    }

    // --dump-star-spend: one JSONL record per Research/Harvest/Build/Summon
    // move actually executed — (turn, player, move type, stars spent =
    // tribe.stars before minus after). Reads the real star delta off game
    // state rather than re-deriving cost formulas, so it's exact even with
    // discounts (e.g. Philosophy's tech discount).
    // Adjacency hubs a NET seat built itself, recorded at build time. Counting
    // by end-of-game tile ownership instead would credit the net for hubs it
    // captured from the anchor — with --anchor-frac 1.0 that is most of them.
    let mut built_hubs: Vec<(i32, polyfish::types::StructureType, PlayerId)> = Vec::new();
    // Per hub type: (chosen tile, builder, every tile it could legally have used).
    let mut first_hub_sites: HashMap<polyfish::types::StructureType, (i32, PlayerId, Vec<i32>)> =
        HashMap::new();
    let mut star_spend_file: Option<File> = None;
    if let Some(dir) = dump_star_spend {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-star-spend] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => star_spend_file = Some(f),
                Err(e) => eprintln!("[dump-star-spend] failed to open game file: {e}"),
            }
        }
    }
    // --dump-reward-choices: one JSONL record per city-reward choice ply with
    // the full search trace of the (modal) candidate pair — per-candidate
    // post-search Q, visits, prior, edge reward — for Q-gap sizing of the
    // reward-choice pricing terms. Not combinable with --dump-failed-dir
    // (that path consumes the trace first).
    let mut reward_choice_file: Option<File> = None;
    if let Some(dir) = dump_reward_choices {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-reward-choices] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => reward_choice_file = Some(f),
                Err(e) => eprintln!("[dump-reward-choices] failed to open game file: {e}"),
            }
        }
    }

    // --dump-level-completion: one JSONL record per executed Harvest/Build
    // with owning-city level/progress and stars before/after.
    let mut level_completion_file: Option<File> = None;
    if let Some(dir) = dump_level_completion {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-level-completion] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => level_completion_file = Some(f),
                Err(e) => eprintln!("[dump-level-completion] failed to open game file: {e}"),
            }
        }
    }

    // --dump-pop-spend-choices: sampled early-economy ply traces for Q-gap
    // sizing of the completion-discipline and body-count terms.
    let mut pop_spend_file: Option<File> = None;
    if let Some(dir) = dump_pop_spend_choices {
        let path = std::path::Path::new(dir);
        if let Err(e) = std::fs::create_dir_all(path) {
            eprintln!("[dump-pop-spend-choices] failed to create {}: {e}", path.display());
        } else {
            match File::create(path.join(format!("game{game_idx}.jsonl"))) {
                Ok(f) => pop_spend_file = Some(f),
                Err(e) => eprintln!("[dump-pop-spend-choices] failed to open game file: {e}"),
            }
        }
    }
    let mut pop_spend_dumped: usize = 0;
    let mut last_pop_spend_turn: i32 = -1;

    const STAR_SPEND_TYPES: [polyfish::types::MoveType; 5] = [
        polyfish::types::MoveType::Research,
        polyfish::types::MoveType::Harvest,
        polyfish::types::MoveType::Build,
        polyfish::types::MoveType::Summon,
        polyfish::types::MoveType::Ability,
    ];

    let prior_w = decay_crutch(
        HEURISTIC_PRIOR_W0,
        HEURISTIC_PRIOR_DECAY,
        iteration,
        decay_last_iter,
        force_zero_crutches,
    );
    // One trust scalar drives β on σ(Q) in both the exported targets and the
    // search tree itself. --value-trust overrides the iteration ramp, which
    // saturates immediately on ITER_OFFSET-shifted runs.
    let q_target_w =
        value_trust.unwrap_or_else(|| (iteration as f32 / POLICY_TARGET_Q_RAMP_ITERS).min(1.0));

    // Create two agents (they might share the same network, or be different)
    // macro_params reach BOTH agents: agent2 carries the --opponent
    // evaluator, so without it a league iteration under macro-mcts never
    // consults the loaded checkpoint (a heuristic mirror wearing its name).
    let mut agent1 = Brain::with_backend(eval1, mcts_iters, backend1)
        .with_prior_heuristic_weight(prior_w)
        .with_policy_target_q_weight(q_target_w)
        .with_tree_q_weight(q_target_w)
        .with_reward_shape_w(shape_w_tree)
        .with_pursuit_shape_w(pursuit_w_tree)
        .with_goal_shape_w(goal_w_tree)
        .with_unfreeze_opponent(unfreeze_opponent)
        .with_macro_params(macro_params);
    let mut agent2 = Brain::with_backend(eval2, mcts_iters, backend2)
        .with_prior_heuristic_weight(prior_w)
        .with_policy_target_q_weight(q_target_w)
        .with_tree_q_weight(q_target_w)
        .with_reward_shape_w(shape_w_tree)
        .with_pursuit_shape_w(pursuit_w_tree)
        .with_goal_shape_w(goal_w_tree)
        .with_unfreeze_opponent(unfreeze_opponent)
        .with_macro_params(macro_params);

    if let Some(b) = leaf_batch {
        agent1 = agent1.with_leaf_batch(b);
        agent2 = agent2.with_leaf_batch(b);
    }

    let initial_state = game.state.clone();
    let mut flat_recap: Vec<(i32, i32, serde_json::Value)> = Vec::new();

    let mut cap_ruins = 0;
    let mut cap_villages = 0;
    let mut cap_cities = 0;
    let mut cap_capitals = 0;

    // Game Loop
    let mut game_history: Vec<HistoryStep> = Vec::new();
    let mut action_counts: HashMap<polyfish::types::MoveType, usize> = HashMap::new();
    let mut moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>> = HashMap::new();

    let current_scores: Vec<(PlayerId, i32)> = game
        .state
        .tribes
        .iter()
        .map(|(id, t)| (*id, t.score))
        .collect();

    if progress == ProgressMode::Full {
        eprintln!(
            "[Game {}]: Turn: {} Scores: {:?}",
            game_idx, game.state.settings.turn, current_scores
        );
    }

    let milestones = if progress == ProgressMode::Periodic {
        turn_milestones(max_turns)
    } else {
        Vec::new()
    };
    let mut next_milestone = 0usize;

    let mut spt_at_turn: HashMap<i32, f32> = HashMap::new();
    let mut army_ratios_at_turn: HashMap<i32, (f32, f32)> = HashMap::new();
    let mut next_spt_milestone = 0usize;

    // Per-player tempo tracking: turn-start samples + move-diff unit counters.
    let mut tempo: HashMap<PlayerId, TempoTrack> = HashMap::new();
    let mut last_tempo_key: Option<(i32, PlayerId)> = None;
    let mut prev_tally = unit_tally(&game.state);

    let mut move_count = 0;
    let mut net_moves = 0; // net-seat plies only (excludes Greedy/opponent seats)
    // Up to 3 traces per game, spaced >= 3 game-turns apart, to sample several
    // mid-game stalled decisions rather than only the turn-15 entry ply.
    let mut traces_this_game = 0usize;
    let mut last_trace_turn = -100i32;
    // HarvestReady window state: fires once per game, then captures every
    // ply belonging to the triggering player for the trigger turn + the
    // next 3 turns (not just the first ply of each turn — Step and
    // Harvest/Build aren't exclusive within a turn; the real question is
    // whether Harvest/Build gets picked up once Step options run dry).
    let mut harvest_trigger_turn: Option<i32> = None;
    let mut harvest_trigger_tile: i32 = -1;
    let mut harvest_trigger_pov: Option<PlayerId> = None;
    // VillagePursuit window state: same shape as HarvestReady's — fires once
    // per game, then captures every ply belonging to the triggering player
    // for the trigger turn + the next 3 turns.
    let mut pursuit_trigger_turn: Option<i32> = None;
    let mut pursuit_trigger_village: i32 = -1;
    let mut pursuit_trigger_pov: Option<PlayerId> = None;
    // Wander window state: same shape as VillagePursuit's.
    let mut wander_trigger_turn: Option<i32> = None;
    let mut wander_trigger_unit: i32 = -1;
    let mut wander_trigger_pov: Option<PlayerId> = None;
    // v2.3 tech-cap counters: Research moves executed per seat (ruin-granted
    // techs never pass through a Research move, so they don't count).
    let mut techs_bought = [0u32; 2];
    let mut tier3_bought = [0u32; 2];
    // v3 lane doctrine state per seat (peak enemy sightings, sticky
    // doctrine choice, overlays) — persists across plies like the counters.
    let mut lane_states: [polyfish::ai::oracle_macro::LaneState; 2] =
        Default::default();
    // v7: standing macro commitment per seat — the goal-setter's memory.
    let mut stance_commits: [polyfish::ai::oracle_macro::StanceCommit; 2] = Default::default();
    // v7 belief tripwire: per-seat EXPAND plan outcomes.
    let mut plan_trackers: [PlanTracker; 2] = Default::default();
    while !polyfish::functions::is_game_over(&game.state) {
        record_spt_at_turn_start(
            &game.state,
            &mut spt_at_turn,
            &mut army_ratios_at_turn,
            &mut next_spt_milestone,
            seat_roles,
        );

        if move_count > 50000 {
            // Reduced for safety
            eprintln!(
                "[Game {}] Move count exceeded 50000 (Safety Break)",
                game_idx
            );
            break;
        }

        let pov = game.state.settings.current_player_turn_id;

        // EXP_ELO_028 Stage 1: scripted macro goal for net seats. The SAME
        // goal must appear in the recorded features and in every encode the
        // agent performs, or training data and search would disagree.
        // v7: resolved through the standing commitment, not recomputed cold,
        // and taken before the turn dump so the snapshot sees this ply's goal.
        let macro_goal = if goal_channels && is_net_seat(seat_roles, pov) {
            let seat = ((pov - 1) as usize).min(1);
            let g = polyfish::ai::oracle_macro::commit_macro_goal(
                &game.state,
                pov,
                &mut stance_commits[seat],
                tier3_bought[seat],
            );
            update_plans(&game.state, pov, &g, &mut plan_trackers[seat]);
            Some(g)
        } else {
            None
        };


        // Tempo curve: sample the acting player once per (turn, pov), pre-move.
        let tempo_key = (game.state.settings.turn, pov);
        if last_tempo_key != Some(tempo_key) {
            if let Some(mut s) = tempo_sample(&game.state, pov) {
                let track = tempo.entry(pov).or_default();
                s.trained_cum = track.units_trained;
                s.lost_cum = track.units_lost;
                s.stars_lost_cum = track.army_stars_lost;
                track.samples.push(s);
            }
            last_tempo_key = Some(tempo_key);
        }


        let current_network = if pov == 1 { network1 } else { network2 };
        let device = current_network.device();

        // MCTS Search - use the correct agent
        let current_agent = if pov == 1 { &mut agent1 } else { &mut agent2 };
        let star_gate = macro_goal.as_ref().map_or(false, |g| {
            polyfish::ai::oracle_macro::tech_discipline_active(&game.state, pov, g)
        });
        let seat = ((pov - 1) as usize).min(1);
        let goal_aux = macro_goal.as_ref().map(|g| {
            polyfish::ai::oracle_macro::update_lane_state(&game.state, pov, &mut lane_states[seat]);
            polyfish::ai::oracle_macro::compute_goal_aux(
                &game.state,
                pov,
                g,
                techs_bought[seat],
                tier3_bought[seat],
                Some(&lane_states[seat]),
            )
        });
        current_agent.set_macro_goal(macro_goal.clone(), star_gate);
        current_agent.set_goal_aux(goal_aux);

        let trigger_info = if trace_villages
            && !trace_all
            && !matches!(
                trace_trigger,
                TraceTrigger::HarvestReady | TraceTrigger::VillagePursuit | TraceTrigger::Wander
            )
            && traces_this_game < 3
            && game.state.settings.turn >= last_trace_turn + 3
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            find_village_trigger(&game.state, pov, &open_villages, trace_trigger)
        } else {
            None
        };

        // HarvestReady: (trigger_tile, turns_since_trigger) for THIS ply, if
        // it belongs to the triggering player and falls inside the
        // [trigger_turn, trigger_turn+3] window. Captures EVERY such ply
        // (not just the first per turn) so post-hoc analysis can bucket by
        // how many Step candidates were still live at capture time.
        let harvest_capture = if trace_villages
            && !trace_all
            && trace_trigger == TraceTrigger::HarvestReady
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            if harvest_trigger_turn.is_none() {
                if let Some(tile) = find_harvest_trigger(&game.state, pov) {
                    harvest_trigger_turn = Some(game.state.settings.turn);
                    harvest_trigger_tile = tile;
                    harvest_trigger_pov = Some(pov);
                }
            }
            harvest_trigger_turn.and_then(|start| {
                let turn = game.state.settings.turn;
                (harvest_trigger_pov == Some(pov) && turn <= start + 3)
                    .then_some((harvest_trigger_tile, turn - start))
            })
        } else {
            None
        };

        // VillagePursuit: (trigger_village, turns_since_trigger) for THIS
        // ply, same window/every-ply shape as HarvestReady above.
        let pursuit_capture = if trace_villages
            && !trace_all
            && trace_trigger == TraceTrigger::VillagePursuit
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            if pursuit_trigger_turn.is_none() {
                if let Some(village) =
                    find_village_pursuit_trigger(&game.state, pov, &open_villages)
                {
                    pursuit_trigger_turn = Some(game.state.settings.turn);
                    pursuit_trigger_village = village;
                    pursuit_trigger_pov = Some(pov);
                }
            }
            pursuit_trigger_turn.and_then(|start| {
                let turn = game.state.settings.turn;
                (pursuit_trigger_pov == Some(pov) && turn <= start + 3)
                    .then_some((pursuit_trigger_village, turn - start))
            })
        } else {
            None
        };

        // Wander: (trigger_unit, turns_since_trigger) for THIS ply, same
        // window/every-ply shape as VillagePursuit above.
        let wander_capture = if trace_villages
            && !trace_all
            && trace_trigger == TraceTrigger::Wander
            && trace_counter.load(Ordering::Relaxed) < trace_max
        {
            if wander_trigger_turn.is_none() {
                if let Some(unit) = find_wander_trigger(&game.state, pov, &open_villages) {
                    wander_trigger_turn = Some(game.state.settings.turn);
                    wander_trigger_unit = unit;
                    wander_trigger_pov = Some(pov);
                }
            }
            wander_trigger_turn.and_then(|start| {
                let turn = game.state.settings.turn;
                (wander_trigger_pov == Some(pov) && turn <= start + 3)
                    .then_some((wander_trigger_unit, turn - start))
            })
        } else {
            None
        };

        // Reward-choice ply? (modal — generate_reward_moves preempts all
        // other moves, so the root is exactly the pending choice pair(s))
        let reward_choice_ply = reward_choice_file.is_some() && {
            let mut rm: Vec<Box<dyn polyfish::moves::Move>> = Vec::new();
            polyfish::moves::reward::generate_reward_moves(&game.state, &mut rm);
            !rm.is_empty()
        };

        // Sampled early-economy ply for the pop-spend Q-gap dump.
        let pop_spend_ply = pop_spend_file.is_some()
            && !reward_choice_ply
            && game.state.settings.turn <= 15
            && pop_spend_dumped < 12
            && game.state.settings.turn > last_pop_spend_turn
            && game.state.tribes.get(&pov).map_or(false, |t| t.stars >= 2);

        if trace_all
            || reward_choice_ply
            || pop_spend_ply
            || trigger_info.is_some()
            || harvest_capture.is_some()
            || pursuit_capture.is_some()
            || wander_capture.is_some()
        {
            current_agent.request_trace();
        }

        let (best_move, move_visits) = current_agent.think_decomposed(&mut game, move_count);
        // The search that just ran was for the CURRENT (pre-move) state, so
        // this is that state's own root value — the TD bootstrap target for
        // whichever earlier step's label lands here as its "next decision".
        let root_value = current_agent.last_root_value();
        let root_own_value = current_agent.last_root_own_value();

        // State tensor for the training sample. Encoded AFTER the search:
        // a macro agent commits its directive DURING think, and the recorded
        // features must carry the goal that actually drove this ply (the
        // committed directive when macro, the scripted goal otherwise —
        // search itself never mutates `game`, so the state is still
        // pre-move). Gated on goal_channels like every other paint.
        let feat_goal = if goal_channels {
            current_agent.macro_committed_goal().or_else(|| macro_goal.clone())
        } else {
            None
        };
        let state_t = features::state_to_cpu_features_goal(
            &game.state,
            pov,
            None,
            feat_goal.as_ref(),
        )
        .and_then(|r| r.into_game_features(&device))
        .expect("BUG: Failed to create state tensor - game state is invalid");
        let step_trace = if trace_all {
            current_agent.take_trace()
        } else {
            None
        };

        // Stage 4: one snapshot per (turn, pov), taken AFTER the search and
        // before the move is applied — the only point where the state is
        // still pre-move but the lane and directive that drove the ply are
        // both committed. A turn-start dump would report last turn's lane.
        if let Some(f) = turn_dump_file.as_mut() {
            let key = (game.state.settings.turn, pov);
            if last_dump_key != Some(key) {
                dump_turn_state(
                    f,
                    game_idx,
                    &game.state,
                    pov,
                    &open_villages,
                    &lane_states[seat],
                    current_agent.macro_committed_playstyle(),
                    feat_goal.as_ref(),
                    &stance_commits[seat],
                    &plan_trackers[seat],
                    tier3_bought[seat],
                );
                last_dump_key = Some(key);
            }
        }

        // Stage 3b (macro policy head, first step): same once-per-(turn,pov)
        // point as the Stage 4 dump above — the ballot is stable for every
        // ply within a turn (the macro agent only re-searches on a new
        // (turn, pov)), so writing on every ply would just repeat the row.
        if let Some(f) = macro_policy_file.as_mut() {
            let key = (game.state.settings.turn, pov);
            if last_macro_policy_key != Some(key) {
                if let Some((candidates, visits)) = current_agent.macro_root_ballot() {
                    if !candidates.is_empty() {
                        dump_macro_policy_row(f, game.state.settings.turn, pov, &candidates, &visits);
                        last_macro_policy_key = Some(key);
                    }
                }
            }
        }

        if reward_choice_ply {
            if let Some(trace) = current_agent.take_trace() {
                if let Some(f) = reward_choice_file.as_mut() {
                    let width = game.state.settings.size as usize;
                    let revealed = game
                        .state
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
                        "turn": game.state.settings.turn,
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
            if let Some(trace) = current_agent.take_trace() {
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
                        let tribe = game.state.tribes.get(&pov);
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
                            "turn": game.state.settings.turn,
                            "player_id": pov,
                            "stars": tribe.map(|t| t.stars).unwrap_or(0),
                            "units": tribe.map(|t| t.units.len()).unwrap_or(0),
                            "cities": cities,
                            "chosen": chosen,
                            "candidates": cands,
                        });
                        let _ = writeln!(f, "{row}");
                        pop_spend_dumped += 1;
                        last_pop_spend_turn = game.state.settings.turn;
                    }
                }
            }
        }

        if let Some((trigger_unit_idx, trigger_village_idx)) = trigger_info {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    let mut visible_villages: Vec<i32> = open_villages
                        .iter()
                        .copied()
                        .filter(|idx| {
                            game.state
                                .tiles
                                .get(idx)
                                .map_or(false, |t| t.explorers.contains(&pov))
                        })
                        .collect();
                    // Sorted: `open_villages` is a std HashSet, so its iteration order is
                    // randomized per process and would otherwise leak into this dump.
                    visible_villages.sort_unstable();
                    write_decision_trace(
                        "decision_traces",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        trigger_unit_idx,
                        trigger_village_idx,
                        &visible_villages,
                        None,
                    );
                }
                traces_this_game += 1;
                last_trace_turn = game.state.settings.turn;
            }
        }

        if let Some((trigger_tile, turns_since)) = harvest_capture {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    write_decision_trace(
                        "decision_traces_harvest",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        trigger_tile,
                        -1,
                        &[],
                        Some(turns_since),
                    );
                }
            }
        }

        if let Some((trigger_village, turns_since)) = pursuit_capture {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    write_decision_trace(
                        "decision_traces_pursuit",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        -1,
                        trigger_village,
                        &[],
                        Some(turns_since),
                    );
                }
            }
        }

        if let Some((trigger_unit, turns_since)) = wander_capture {
            if let Some(trace) = current_agent.take_trace() {
                if trace_counter.fetch_add(1, Ordering::Relaxed) < trace_max {
                    let mut visible_villages: Vec<i32> = open_villages
                        .iter()
                        .copied()
                        .filter(|idx| {
                            game.state
                                .tiles
                                .get(idx)
                                .map_or(false, |t| t.explorers.contains(&pov))
                        })
                        .collect();
                    // Sorted: `open_villages` is a std HashSet, so its iteration order is
                    // randomized per process and would otherwise leak into this dump.
                    visible_villages.sort_unstable();
                    write_decision_trace(
                        "decision_traces_wander",
                        &trace,
                        iteration,
                        game_idx,
                        game.state.settings.turn,
                        move_count,
                        pov,
                        trigger_unit,
                        -1,
                        &visible_villages,
                        Some(turns_since),
                    );
                }
            }
        }

        let map_size = game.state.settings.size as usize;

        let (mut p_action, mut p_source, mut p_target, mut p_option) =
            decompose_visits(&move_visits, map_size);

        // EXP_ELO_020: DAgger — blend Greedy's move-ranking AT THE MODEL'S OWN
        // state into the policy target, so the expert corrects the policy
        // exactly where the net actually plays (on-distribution). Net seats
        // only — the Greedy anchor seat already emits Greedy's ranking as its
        // target. This attacks the collapsed capture prior at the model's forks
        // without needing the value head; unlike BC (EXP_ELO_007) it labels the
        // learner's states, not the expert's, which is the erosion fix.
        if dagger_alpha > 0.0 && is_net_seat(seat_roles, pov) {
            let greedy_visits = polyfish::ai::heuristic_mcts::GreedyHeuristicAgent
                .select_move_with_decomposed_visits(&mut game, move_count)
                .1;
            let (ga, gs, gt, go) = decompose_visits(&greedy_visits, map_size);
            // Normalization-preserving blend: mix the target's DIRECTION toward
            // Greedy but keep each head's original total mass, so forced/partial
            // states (empty or sub-normal MCTS heads) aren't distorted — a head
            // with zero MCTS mass stays zero (nothing to correct), a full head
            // stays full.
            let blend = |p: &mut [f32], g: &[f32]| {
                let orig: f32 = p.iter().sum();
                for (x, &gv) in p.iter_mut().zip(g.iter()) {
                    *x = (1.0 - dagger_alpha) * *x + dagger_alpha * gv;
                }
                let mixed: f32 = p.iter().sum();
                if mixed > 0.0 {
                    let k = orig / mixed;
                    for x in p.iter_mut() {
                        *x *= k;
                    }
                }
            };
            blend(&mut p_action, &ga);
            blend(&mut p_source, &gs);
            blend(&mut p_target, &gt);
            blend(&mut p_option, &go);
        }

        let policy_data = DecomposedPolicyData {
            action_type: p_action,
            source_spatial: p_source,
            target_spatial: p_target,
            move_option: p_option,
        };

        if let Some(m) = best_move {
            if trace_all {
                decision_log.push(TracedDecision {
                    turn: game.state.settings.turn,
                    move_count,
                    player_id: pov,
                    chosen: m.describe(&game.state),
                    root_value,
                    trace: step_trace,
                });
            }
            let m_type = m.move_type();
            // Move-mix and capture metrics count NET-controlled seats only —
            // in anchor/league games the Greedy/opponent seat's moves would
            // otherwise blend into "the model's" behavior charts.
            let net_move = is_net_seat(seat_roles, pov);
            if net_move {
                net_moves += 1;
                *action_counts.entry(m_type).or_insert(0) += 1;
                *moves_by_turn
                    .entry(game.state.settings.turn)
                    .or_default()
                    .entry(m_type)
                    .or_insert(0) += 1;
            }

            if m_type == polyfish::types::MoveType::Reward {
                if let Some(f) = city_reward_file.as_mut() {
                    if let (Ok(target), Ok(reward_type)) = (m.target_idx(), m.reward_type()) {
                        let target = target as i32;
                        let city = game
                            .state
                            .tribes
                            .get(&pov)
                            .and_then(|t| t.cities.iter().find(|c| c.idx == target));
                        if let Some(city) = city {
                            let stars = game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
                            let line = json!({
                                "game_idx": game_idx,
                                "turn": game.state.settings.turn,
                                "player_id": pov,
                                "city_idx": target,
                                "city_level": city.level,
                                "city_population": city.population,
                                "stars": stars,
                                "reward": format!("{:?}", reward_type),
                            });
                            if let Ok(s) = serde_json::to_string(&line) {
                                use std::io::Write;
                                let _ = writeln!(f, "{s}");
                            }
                        }
                    }
                }
            }

            if m_type == polyfish::types::MoveType::Capture {
                if let Ok(src) = m.source_idx() {
                    let idx = src as i32;

                    let struct_opt = game.state.structures.get(&idx).and_then(|s| s.as_ref());
                    let is_ruin = struct_opt.map(|s| s.structure_type)
                        == Some(polyfish::types::StructureType::Ruin);
                    let is_village = struct_opt.map(|s| s.structure_type)
                        == Some(polyfish::types::StructureType::Village);
                    let owner = game.state.tiles.get(&idx).map(|t| t.owner).unwrap_or(0);
                    let is_capital = game
                        .state
                        .tiles
                        .get(&idx)
                        .map(|t| t.capital_of > 0)
                        .unwrap_or(false);

                    if net_move {
                        if is_ruin {
                            cap_ruins += 1;
                        } else if is_village {
                            if owner == 0 {
                                cap_villages += 1;
                            } else if owner != pov {
                                if is_capital {
                                    cap_capitals += 1;
                                } else {
                                    cap_cities += 1;
                                }
                            }
                        }
                    }

                    // The open-set must shrink regardless of who captured
                    // (it also feeds FOW-visible tracking), but t2c turns
                    // record net captures only — a Greedy grab is the net
                    // LOSING the race, not capturing.
                    if open_villages.remove(&idx) {
                        if net_move {
                            village_capture_turns.push(game.state.settings.turn);
                            first_village_turn
                                .entry(pov)
                                .or_insert(game.state.settings.turn);
                        }
                    } else if open_ruins.remove(&idx) && net_move {
                        ruin_capture_turns.push(game.state.settings.turn);
                    }
                }
            }

            flat_recap.push((
                game.state.settings.turn,
                game.state.settings.current_player_turn_id,
                m.serialize(),
            ));
            // Snapshot (possibly Φ-shaped) scores at this moment (pre-move)
            // for the TD label.
            let (my_score_now, opp_score_now) =
                reward::shaped_snapshot(&game.state, pov, shape_w_label, pursuit_w_label);
            let opp_id = game
                .state
                .tribes
                .keys()
                .copied()
                .find(|id| *id != pov)
                .unwrap_or(pov);
            let spt_of = |id: PlayerId| {
                game.state
                    .tribes
                    .get(&id)
                    .map(|t| polyfish::functions::get_tribe_spt(&game.state, t))
                    .unwrap_or(0)
            };
            game_history.push(HistoryStep {
                features: state_t,
                policy: policy_data,
                player_id: pov,
                my_score: my_score_now,
                opp_score: opp_score_now,
                turn: game.state.settings.turn,
                root_value,
                root_own_value,
                macro_ballot: macro_ballot_for_history_step(
                    (game.state.settings.turn, pov),
                    &mut last_macro_ballot_key,
                    current_agent.macro_root_ballot(),
                ),
                enemy_units: enemy_unit_grid(&game.state, pov, features::MAP_SIZE * features::MAP_SIZE),
                my_spt: spt_of(pov),
                opp_spt: spt_of(opp_id),
                city_spt: game
                    .state
                    .tribes
                    .get(&pov)
                    .map(|t| {
                        t.cities
                            .iter()
                            .map(|c| {
                                (c.idx, polyfish::functions::get_city_production(&game.state, c))
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                pursuit: reward::pursuit_potential(&game.state, pov)
                    / reward::SHAPE_PURSUIT_PER_TILE
                    / reward::SHAPE_PROX_CAP as f32,
            });
            if progress == ProgressMode::Full && move_count > 0 && move_count % 10 == 0 {
                eprintln!(
                    "[Game {}]: Turn: {} Player: {} Move: {}",
                    game_idx,
                    game.state.settings.turn,
                    pov,
                    m.describe(&game.state),
                );
            }
            let star_spend_pre = star_spend_file.as_ref().and_then(|_| {
                STAR_SPEND_TYPES
                    .contains(&m_type)
                    .then(|| game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0))
                    .map(|stars_before| (game.state.settings.turn, stars_before))
            });
            // --dump-level-completion: snapshot the owning city pre-move.
            let level_completion_pre = level_completion_file.as_ref().and_then(|_| {
                if m_type != polyfish::types::MoveType::Harvest
                    && m_type != polyfish::types::MoveType::Build
                {
                    return None;
                }
                let target = m.target_idx().ok()? as i32;
                let city = polyfish::functions::get_city_owning_tile(&game.state, target)?;
                if city.owner != pov {
                    return None;
                }
                let stars = game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
                let threatened =
                    polyfish::ai::reward::city_threatened(&game.state, pov, city.idx);
                Some((target, city.idx, city.level, city.progress, stars, threatened))
            });
            // First hub of each type the net builds: snapshot every tile the hub
            // could legally have gone to at THIS ply (mirrors moves/build.rs —
            // own-city territory, empty, terrain-legal, non-algae, and only
            // cities without one since the tier is limited_per_city). Scored on
            // FINAL adjacency at game end, so it asks the long-term question:
            // of the sites available then, did it pick one that grew partners?
            if m_type == polyfish::types::MoveType::Build && is_net_seat(seat_roles, pov) {
                if let (Ok(target), Ok(s_type)) = (m.target_idx(), m.structure_type()) {
                    let st = polyfish::settings::structures::get_structure_setting(s_type);
                    if !st.adjacent_types.is_empty()
                        && st.reward_pop > 0
                        && !first_hub_sites.contains_key(&s_type)
                    {
                        let mut cands: Vec<i32> = Vec::new();
                        if let Some(tribe) = game.state.tribes.get(&pov) {
                            for city in &tribe.cities {
                                let taken = city._territory.iter().any(|&t| {
                                    polyfish::functions::get_structure_at(&game.state, t)
                                        .is_some_and(|s| s.structure_type == s_type)
                                });
                                if taken {
                                    continue;
                                }
                                for &t in &city._territory {
                                    let Some(tile) = game.state.tiles.get(&t) else { continue };
                                    if polyfish::functions::get_structure_at(&game.state, t).is_some()
                                        || tile.is_algae()
                                        || !st.terrain_types.contains(&tile.terrain_type)
                                    {
                                        continue;
                                    }
                                    cands.push(t);
                                }
                            }
                        }
                        if !cands.is_empty() {
                            first_hub_sites.insert(s_type, (target as i32, pov, cands));
                        }
                    }
                }
            }
            let _ = game.play_move(m.as_ref());
            if m_type == polyfish::types::MoveType::Build && is_net_seat(seat_roles, pov) {
                if let (Ok(target), Ok(s_type)) = (m.target_idx(), m.structure_type()) {
                    let st = polyfish::settings::structures::get_structure_setting(s_type);
                    if !st.adjacent_types.is_empty() && st.reward_pop > 0 {
                        built_hubs.push((target as i32, s_type, pov));
                    }
                }
            }
            if m_type == polyfish::types::MoveType::Research {
                let seat = ((pov - 1) as usize).min(1);
                techs_bought[seat] += 1;
                if let Ok(tech) = m.tech_type() {
                    if polyfish::settings::technology::get_technology_setting(tech).tier
                        == Some(3)
                    {
                        tier3_bought[seat] += 1;
                    }
                }
            }
            if let (Some((turn, stars_before)), Some(f)) =
                (star_spend_pre, star_spend_file.as_mut())
            {
                let stars_after = game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
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
                    use std::io::Write;
                    let _ = writeln!(f, "{s}");
                }
            }
            if let (
                Some((target, city_idx, level_b, progress_b, stars_b, threatened)),
                Some(f),
            ) = (level_completion_pre, level_completion_file.as_mut())
            {
                let city = game
                    .state
                    .tribes
                    .get(&pov)
                    .and_then(|t| t.cities.iter().find(|c| c.idx == city_idx));
                if let Some(c) = city {
                    let stars_after =
                        game.state.tribes.get(&pov).map(|t| t.stars).unwrap_or(0);
                    let line = json!({
                        "game_idx": game_idx,
                        "turn": game.state.settings.turn,
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
                        use std::io::Write;
                        let _ = writeln!(f, "{s}");
                    }
                }
            }

            // Unit-accounting diff: attribute gains to Summon (trained) vs
            // anything else (granted), and any decrease as a loss — both in
            // count and in star value (a dead giant is 10, not 1).
            let new_tally = unit_tally(&game.state);
            for (&pid, &(n_units, n_giants, n_stars)) in &new_tally {
                let (p_units, p_giants, p_stars) =
                    prev_tally.get(&pid).copied().unwrap_or((0, 0, 0));
                let track = tempo.entry(pid).or_default();
                if n_units > p_units {
                    if m_type == polyfish::types::MoveType::Summon && pid == pov {
                        track.units_trained += n_units - p_units;
                    } else {
                        track.units_granted += n_units - p_units;
                    }
                } else if n_units < p_units {
                    track.units_lost += p_units - n_units;
                }
                if n_stars < p_stars {
                    track.army_stars_lost += p_stars - n_stars;
                }
                if n_giants > p_giants {
                    track.giants_made += n_giants - p_giants;
                }
            }
            prev_tally = new_tally;

            if progress == ProgressMode::Periodic {
                while next_milestone < milestones.len()
                    && game.state.settings.turn >= milestones[next_milestone]
                {
                    eprintln!(
                        "[Game {}] turn {} reached (move {})",
                        game_idx,
                        milestones[next_milestone],
                        move_count + 1,
                    );
                    next_milestone += 1;
                }
            }
        } else {
            break;
        }
        move_count += 1;
    }

    // Final tempo sample per player from the end state (a capture on the last
    // turn would otherwise be invisible); replaces a same-turn start sample.
    let tempo_pids: Vec<PlayerId> = game.state.tribes.keys().copied().collect();
    for pid in tempo_pids {
        if let Some(mut s) = tempo_sample(&game.state, pid) {
            let track = tempo.entry(pid).or_default();
            s.trained_cum = track.units_trained;
            s.lost_cum = track.units_lost;
            s.stars_lost_cum = track.army_stars_lost;
            match track.samples.last_mut() {
                Some(last) if last.turn == s.turn => *last = s,
                _ => track.samples.push(s),
            }
        }
    }

    // Determine scores & winner
    // In Domination, the winner is the last tribe alive.
    // If the game timed out (safety cap), use score as tiebreaker.
    let mut scores: HashMap<i32, i32> = HashMap::new();
    let mut final_potentials: HashMap<i32, f32> = HashMap::new();
    let mut alive: HashMap<i32, bool> = HashMap::new();
    for (id, t) in &game.state.tribes {
        scores.insert(*id, t.score);
        alive.insert(*id, t.killed_turn <= 0 && t.resigned_turn <= 0);
    }
    for id in scores.keys() {
        let mut phi = 0.0;
        if shape_w_label != 0.0 {
            phi += shape_w_label * reward::dev_potential(&game.state, *id);
        }
        if pursuit_w_label != 0.0 {
            phi += pursuit_w_label * reward::pursuit_potential(&game.state, *id);
        }
        final_potentials.insert(*id, scores[id] as f32 + phi);
    }

    // Domination winner: the sole survivor, or highest score if timeout
    let alive_tribes: Vec<i32> = alive
        .iter()
        .filter(|(_, is_alive)| **is_alive)
        .map(|(id, _)| *id)
        .collect();

    let (winner_id, winner_score) = if alive_tribes.len() == 1 {
        let wid = alive_tribes[0];
        (wid, *scores.get(&wid).unwrap_or(&0))
    } else {
        // Timeout: use score tiebreaker
        scores
            .iter()
            .max_by_key(|&(_, score)| score)
            .map(|(&id, &score)| (id, score))
            .unwrap_or((0, 0))
    };

    let is_decisive = alive_tribes.len() == 1;
    if progress == ProgressMode::Full {
        eprintln!(
            "[Game {}] Finished. Moves: {} | Winner: {} (Score: {}) | Decisive: {}",
            game_idx, move_count, winner_id, winner_score, is_decisive
        );
    }

    // Net-seat-only territory/exploration (anchor/opponent seats excluded).
    let captured_tiles = game
        .state
        .tiles
        .values()
        .filter(|t| t.owner != 0 && is_net_seat(seat_roles, t.owner))
        .count() as i32;
    let revealed_tiles: i32 = game
        .state
        .tribes
        .keys()
        .filter(|&&pid| is_net_seat(seat_roles, pid))
        .map(|&pid| {
            game.state
                .tiles
                .values()
                .filter(|t| t.explorers.contains(&pid))
                .count() as i32
        })
        .sum();

    // Realized level of the hubs the net BUILT (see `built_hubs`), scored at
    // game end so a hub that grows as later partners go down is credited —
    // partners are counted the way `build_structure` pays them, but against the
    // BUILDER's ownership, so value lost with the territory reads as lost.
    let mut hub_levels: HashMap<polyfish::types::StructureType, (u32, i64, u32, u32)> =
        HashMap::new();
    for (idx, s_type, builder) in &built_hubs {
        let settings = polyfish::settings::structures::get_structure_setting(*s_type);
        let still_held = game
            .state
            .tiles
            .get(idx)
            .is_some_and(|t| t.owner == *builder);
        let partners = polyfish::functions::get_adjacent_indices(&game.state, *idx, 1)
            .into_iter()
            .filter(|adj| {
                game.state.tiles.get(adj).is_some_and(|t| t.owner == *builder)
                    && polyfish::functions::get_structure_at(&game.state, *adj)
                        .is_some_and(|p| settings.adjacent_types.contains(&p.structure_type))
            })
            .count() as i64;
        let e = hub_levels.entry(*s_type).or_insert((0, 0, 0, 0));
        e.0 += 1;
        e.1 += partners;
        e.2 += u32::from(partners <= 1);
        e.3 += u32::from(!still_held);
    }

    // Rank the tile the net actually used against every tile it could have used,
    // both scored on partners standing at game end.
    let mut first_hub_rank: HashMap<polyfish::types::StructureType, (i64, i64, u32, u32, i64, i64)> =
        HashMap::new();
    for (s_type, (chosen, builder, cands)) in &first_hub_sites {
        let settings = polyfish::settings::structures::get_structure_setting(*s_type);
        let partners_at = |idx: i32| -> i64 {
            polyfish::functions::get_adjacent_indices(&game.state, idx, 1)
                .into_iter()
                .filter(|adj| {
                    game.state.tiles.get(adj).is_some_and(|t| t.owner == *builder)
                        && polyfish::functions::get_structure_at(&game.state, *adj)
                            .is_some_and(|p| settings.adjacent_types.contains(&p.structure_type))
                })
                .count() as i64
        };
        // TERRAIN ceiling: adjacent tiles that could ever host a partner, by
        // terrain + resource alone. Independent of what the net actually built,
        // so it does not inherit the hut-building policy the way `partners_at`
        // does — this is the site's potential, which is the real question.
        let ceiling_at = |idx: i32| -> i64 {
            polyfish::functions::get_adjacent_indices(&game.state, idx, 1)
                .into_iter()
                .filter(|&adj| {
                    let Some(tile) = game.state.tiles.get(&adj) else { return false };
                    settings.adjacent_types.iter().any(|p| {
                        let ps = polyfish::settings::structures::get_structure_setting(*p);
                        if !ps.terrain_types.contains(&tile.terrain_type) || tile.is_algae() {
                            return false;
                        }
                        match ps.resource_type {
                            Some(r) => game
                                .state
                                .resources
                                .get(&adj)
                                .and_then(|o| o.as_ref())
                                .is_some_and(|res| res.resource_type == r),
                            None => true,
                        }
                    })
                })
                .count() as i64
        };
        let got = partners_at(*chosen);
        let best = cands.iter().map(|&c| partners_at(c)).max().unwrap_or(got).max(got);
        let n_better = cands.iter().filter(|&&c| partners_at(c) > got).count() as u32;
        let ceil_got = ceiling_at(*chosen);
        let ceil_best = cands.iter().map(|&c| ceiling_at(c)).max().unwrap_or(ceil_got).max(ceil_got);
        first_hub_rank.insert(
            *s_type,
            (got, best, n_better, cands.len() as u32, ceil_got, ceil_best),
        );
    }

    let mut final_cities = HashMap::new();
    let mut total_cities = 0;
    for (id, t) in &game.state.tribes {
        final_cities.insert(*id, t.cities.len() as i32);
        total_cities += t.cities.len() as i32;
    }

    // Aux-head ground truth; the final state is dropped when this returns.
    let n_tiles = features::MAP_SIZE * features::MAP_SIZE;
    let mut final_owner = vec![0i32; n_tiles];
    for (&idx, tile) in &game.state.tiles {
        let i = idx as usize;
        if i < n_tiles {
            final_owner[i] = tile.owner;
        }
    }
    let mut final_spt = HashMap::new();
    let mut final_tech = HashMap::new();
    for (id, t) in &game.state.tribes {
        final_spt.insert(*id, polyfish::functions::get_tribe_spt(&game.state, t));
        final_tech.insert(*id, tech_multihot(&t.tech_vanilla));
    }

    let recap = ModReplay {
        game_state: initial_state,
        turns: group_recap(flat_recap),
    };
    if let Some(dir) = dump_games_dir {
        dump_failed_game(
            dir,
            "game",
            iteration,
            game_idx,
            seed,
            &tribes,
            backend1,
            backend2,
            max_turns,
            &scores,
            &recap,
            &decision_log,
        );
    }
    if let Some(dir) = dump_failed_dir {
        if village_capture_turns.is_empty() {
            dump_failed_game(
                dir,
                "failed",
                iteration,
                game_idx,
                seed,
                &tribes,
                backend1,
                backend2,
                max_turns,
                &scores,
                &recap,
                &decision_log,
            );
        }
    }

    let net_seats: Vec<PlayerId> = game
        .state
        .tribes
        .keys()
        .copied()
        .filter(|pid| is_net_seat(seat_roles, *pid))
        .collect();
    let (mut vf_captured, mut vf_turn_sum, mut vf_censored_sum) = (0u32, 0.0f64, 0.0f64);
    for pid in &net_seats {
        match first_village_turn.get(pid) {
            Some(&t) => {
                vf_captured += 1;
                vf_turn_sum += f64::from(t);
                vf_censored_sum += f64::from(t);
            }
            None => vf_censored_sum += f64::from(max_turns),
        }
    }

    Some(GameResult {
        history: game_history,
        scores,
        final_potentials,
        final_cities,
        total_cities,
        moves: move_count,
        net_moves,
        revealed_tiles,
        captured_tiles,
        hub_levels,
        first_hub_rank,
        villages_t2c_p50: t2c_turn(&village_capture_turns, initial_villages, 0.5, max_turns),
        villages_t2c_p80: t2c_turn(&village_capture_turns, initial_villages, 0.8, max_turns),
        villages_t2c_all: t2c_turn(&village_capture_turns, initial_villages, 1.0, max_turns),
        villages_first_seats: net_seats.len() as u32,
        villages_first_captured: vf_captured,
        villages_first_turn_sum: vf_turn_sum,
        villages_first_censored_sum: vf_censored_sum,
        ruins_t2c_p50: t2c_turn(&ruin_capture_turns, initial_ruins, 0.5, max_turns),
        ruins_t2c_p80: t2c_turn(&ruin_capture_turns, initial_ruins, 0.8, max_turns),
        ruins_t2c_all: t2c_turn(&ruin_capture_turns, initial_ruins, 1.0, max_turns),
        spt_at_turn,
        army_ratios_at_turn,
        final_owner,
        final_spt,
        final_tech,
        tempo,
        roles: seat_roles,
        winner_score,
        winner_id,
        recap,
        cap_ruins,
        cap_villages,
        cap_cities,
        cap_capitals,
        action_counts,
        moves_by_turn,
    })
}
