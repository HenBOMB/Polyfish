// The METRICS json! literal outgrew serde_json's default macro recursion.
#![recursion_limit = "256"]

use candle_core::{Device, Tensor};
use polyfish::TribeType;
use polyfish::ai::brain::{Brain, SearchBackend, SearchBackendArg};
use polyfish::ai::macro_agent::{MacroLeaf, MacroParams};
use polyfish::ai::eval_backend::{self, EvalBackendKind, PlayerBackend};
use polyfish::ai::eval_server::{EvalServerConfig, EvalServerStats, Evaluator};
use polyfish::ai::features;
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
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use strum::IntoEnumIterator;

mod crutches;
use crutches::{ANCHOR_FRAC_DECAY, HEURISTIC_PRIOR_DECAY, HEURISTIC_PRIOR_W0, decay_crutch};
mod result;
mod labels;
use labels::{CitySptStep, LabelStep, SptStep, city_spt_checkpoints, city_spt_target,
            enemy_unit_grid, macro_ballot_for_history_step, macro_policy_targets,
            ownership_from_pov, spt_checkpoints_by_player, spt_target, td_lambda_labels,
            tech_multihot, GOOD_BOT_FINAL_SCORE, FINAL_OUTCOME_REL_W,
            POLICY_TARGET_Q_RAMP_ITERS};
use result::{DecomposedPolicyData, GameResult, HistoryStep, decompose_visits, group_recap};
mod stats;
mod tempo;
use stats::{finish_milestones, is_net_seat, record_spt_at_turn_start, t2c_turn,
            turn_milestones};
use tempo::{TempoTrack, tempo_sample, unit_tally};
mod traces;
use traces::{TraceTrigger, TracedDecision, dump_failed_game, find_harvest_trigger,
            find_village_pursuit_trigger, find_village_trigger, find_wander_trigger,
            write_decision_trace};
mod shard;
use shard::{SHARD_GAMES, flush_shard};
mod cli;
use cli::Args;
mod dumps;
use polyfish::eval_seeds::{CORE_TRIBES, SeedEntry, load_seed_file, parse_tribe,
                            resolve_tribes, seed_for_game, tribes_for_game};
use dumps::{PlanTracker, dump_macro_policy_row, dump_turn_state, update_plans};

/// Console verbosity for long self-play runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProgressMode {
    /// Full play-by-play (move every 10 steps, start/finish per game).
    Full,
    /// No move-by-move noise; up to 5 turn-milestone lines per game.
    Periodic,
    /// Silent during games; caller reports ~every 20% on game finish.
    SampledFinish,
}

impl ProgressMode {
    fn from_num_games(num_games: usize) -> Self {
        if num_games >= 64 {
            Self::SampledFinish
        } else if num_games > 32 {
            Self::Periodic
        } else {
            Self::Full
        }
    }
}









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
fn load_networks(
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
fn play_single_game(
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



fn main() -> anyhow::Result<()> {
    use clap::Parser;

    let start_time = Instant::now();

    let mut args = Args::parse();

    // --dump-games-dir is the "give me everything for this game" flag, but
    // its own dump (.replay.json/.decisions.json) doesn't carry the macro
    // ballot or per-ply candidate scoring for macro-mcts games -- those are
    // separate opt-in mechanisms (--dump-macro-policy, POLYFISH_PLY_TRACE)
    // that silently stay off unless remembered explicitly, leaving
    // .decisions.json's `trace` field null with no companion data anywhere.
    // Default both to the same directory whenever --dump-games-dir is set
    // and they weren't already pointed elsewhere.
    if let Some(dir) = &args.dump_games_dir {
        if args.dump_macro_policy.is_none() {
            args.dump_macro_policy = Some(dir.clone());
        }
        if std::env::var("POLYFISH_PLY_TRACE").is_err() {
            // Safety: still single-threaded here, before any actor threads
            // or the OnceLock in `ply_trace_path()` are touched.
            unsafe {
                std::env::set_var("POLYFISH_PLY_TRACE", format!("{dir}/ply_trace.jsonl"));
            }
        }
    }

    if args.anchor_frac > 0.0 && args.opponent.is_some() {
        anyhow::bail!("--anchor-frac and --opponent are mutually exclusive");
    }
    if !(0.0..=1.0).contains(&args.anchor_frac) {
        anyhow::bail!("--anchor-frac must be in [0, 1]");
    }
    if let Some(t) = args.value_trust {
        if !(0.0..=1.0).contains(&t) {
            anyhow::bail!("--value-trust must be in [0, 1]");
        }
    }
    if !(0.0..=1.0).contains(&args.td_lambda) {
        anyhow::bail!("--td-lambda must be in [0, 1]");
    }
    if args.goal_w_tree != 0.0 && !args.goal_channels {
        anyhow::bail!("--goal-w-tree requires --goal-channels (no goal is set without them)");
    }
    let is_macro_backend = matches!(args.search_backend, SearchBackendArg::MacroMcts);
    // The macro tree commits a directive during think; without goal channels
    // the recorded features carry ZERO goal planes, so the data says nothing
    // about what the teacher was pursuing. Silent before this guard.
    if is_macro_backend && !args.goal_channels {
        anyhow::bail!(
            "--search-backend macro-mcts requires --goal-channels (the tree's committed \
             directive would otherwise be dropped from the recorded features)"
        );
    }
    if !is_macro_backend
        && (args.macro_leaf != MacroLeaf::Heuristic
            || args.macro_sims != 32
            || args.macro_k != 4
            || args.macro_lambda != 1.0
            || args.macro_rollout_lambda.is_some()
            || args.macro_shape_w != 0.0
            || args.macro_root_prior_w != 0.0)
    {
        anyhow::bail!("--macro-* flags require --search-backend macro-mcts");
    }
    let macro_params = MacroParams {
        k: args.macro_k,
        leaf: args.macro_leaf,
        lambda: args.macro_lambda,
        rollout_lambda: args.macro_rollout_lambda.unwrap_or(args.macro_lambda),
        sims: args.macro_sims,
        shape_w: args.macro_shape_w,
        root_prior_w: args.macro_root_prior_w,
        ..MacroParams::default()
    };

    // Default Metal op-flush cadence to 1000 for better GPU efficiency on Metal
    if std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").is_err() {
        // This is safe because it runs at the top of `main`, so no concurrent writes.
        unsafe {
            std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "1000");
        }
    }
    let metal_compute_per_buffer =
        std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").unwrap_or_else(|_| "1000".to_string());

    let backend = match args.search_backend {
        SearchBackendArg::Zero => SearchBackend::Zero,
        SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: args.gumbel_k },
        SearchBackendArg::Heuristic => SearchBackend::Heuristic,
        SearchBackendArg::Greedy => SearchBackend::Greedy,
        // Stage 3: macro-mcts generates training games (behavior-cloning
        // policy targets + on-distribution value labels for the macro leaf).
        SearchBackendArg::MacroMcts => SearchBackend::MacroMcts,
        // EXP_ELO_032: arena-only bootstrap backends.
        SearchBackendArg::MacroScript | SearchBackendArg::MacroLookahead => {
            anyhow::bail!("macro-script/lookahead are arena-only (EXP_ELO_032)")
        }
    };

    let device = eval_backend::select_device()?;

    // Resolve the eval backend up front (explicit --eval-backend, else auto:
    // metal when compiled in, else tch when compiled in, else candle) — the
    // network load below needs it to decide whether player 2 gets an
    // isolated device (see `load_networks`'s doc comment).
    let eval_backend_kind = eval_backend::resolve_eval_backend_kind(&args.eval_backend)?;
    let eval_servers = eval_backend::resolve_eval_servers(eval_backend_kind, args.eval_servers)?;

    // Load models (P1, and P2 defaulting to P1 when no opponent is given)
    let (network1, network2) = load_networks(&device, args.opponent.as_deref(), eval_backend_kind)?;

    let base_seed = if args.base_seed != 0 {
        args.base_seed
    } else {
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
    };

    let seed_entries: Option<Vec<SeedEntry>> = args
        .seed_file
        .as_ref()
        .map(|path| load_seed_file(path, args.num_games, parse_tribe))
        .transpose()?;
    let seed_list: Option<Vec<i64>> = seed_entries
        .as_ref()
        .map(|entries| entries.iter().map(|e| e.seed).collect());

    // Pool of tribes to draw from when tribe1/tribe2 aren't pinned via CLI args
    // or a --seed-file entry. Each game in this run independently samples its
    // own pair from this pool (see `pick_tribes`/`resolve_tribes`), rather
    // than the whole run sharing one fixed pair.
    // The v1 training pool; special tribes are deliberately excluded.
    let all_tribes = CORE_TRIBES.to_vec();

    // Game generation: a pool of actor threads pulls game indices off a
    // shared counter. Each actor blocks (parks, no CPU) while awaiting an
    // eval-server reply, so oversubscribing actors past core count is fine —
    // RAM (a Game clone + MCTS tree per actor) is the real ceiling, not CPU.
    // The eval server owns the sole network/device and coalesces requests
    // from every actor into batched forward_t calls (see ai/eval_server.rs
    // for the Metal cross-thread-tensor invariant this design preserves).
    let games_start = Instant::now();

    // Each shard sees ~1/N of the working set (hash-routed), so dividing the
    // per-shard cache by N keeps total resident cache ~constant while
    // preserving the hit rate (cache / working-set ratio is unchanged).
    let per_shard_cache = eval_backend::split_cache_capacity(args.cache_cap, eval_servers);
    let eval_config = EvalServerConfig {
        max_batch: args.max_batch,
        coalesce_timeout: std::time::Duration::from_micros(args.coalesce_timeout_us),
        cache_capacity: per_shard_cache,
        pipeline_workers: args.eval_workers,
    };
    let p1_path = "model.safetensors";
    let p2_path = args.opponent.as_deref().unwrap_or("model.safetensors");
    let has_opponent = args.opponent.is_some();

    // Spawn the shards. Each EvalServer owns its inference thread + device
    // context; the handles are collected into a ShardedEvalHandle that
    // routes leaves by hash so each shard owns its own LRU cache. No
    // opponent => player 2 shares player 1's shard set (one set of
    // inference threads for the same weights).
    let (p1_servers, p2_servers, eval1, eval2) = eval_backend::build_two_player_evaluators(
        eval_backend_kind,
        eval_servers,
        eval_config,
        PlayerBackend {
            model_path: p1_path,
            candle_net: &network1,
        },
        has_opponent.then(|| PlayerBackend {
            model_path: p2_path,
            candle_net: &network2,
        }),
    );

    let num_actors = if args.actors > 0 {
        args.actors
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    };
    let progress_mode = ProgressMode::from_num_games(args.num_games);

    let tribe_label = match (&args.tribe1, &args.tribe2) {
        (Some(t1), Some(t2)) => format!("{t1} vs {t2}"),
        (Some(t1), None) => format!("{t1} vs random"),
        (None, Some(t2)) => format!("random vs {t2}"),
        (None, None) => "random".to_string(),
    };
    let match_label = match &args.opponent {
        Some(opp) => format!("league vs {opp}"),
        None if args.anchor_frac > 0.0 => {
            format!(
                "self-play + up to {:.0}% heuristic-anchor games (decaying)",
                args.anchor_frac * 100.0
            )
        }
        None => "self-play".to_string(),
    };
    let backend_label = match eval_backend_kind {
        EvalBackendKind::Tch => "tch (libtorch/MPS)",
        EvalBackendKind::Metal => "metal (MPSGraph)",
        EvalBackendKind::Candle => "candle",
    };
    let search_label = match backend {
        SearchBackend::Zero => "Zero MCTS".to_string(),
        SearchBackend::Gumbel { k } => format!("Gumbel k={k}"),
        SearchBackend::Heuristic => "Heuristic MCTS (no NN)".to_string(),
        SearchBackend::Greedy => "Greedy heuristic (no NN, no search)".to_string(),
        SearchBackend::MacroScript => "Macro script (EXP_ELO_032)".to_string(),
        SearchBackend::MacroLookahead => "Macro lookahead (EXP_ELO_032)".to_string(),
        SearchBackend::MacroMcts => "Macro MCTS (EXP_ELO_033)".to_string(),
    };
    println!(
        "[selfplay] {match_label}: {} games, {} mcts-iters, {search_label}, tribes {tribe_label} | eval {backend_label} | {eval_servers} shard(s) cache={per_shard_cache:?} workers={} | {num_actors} actors max_batch={} coalesce_us={} leaf_batch={:?} | device {:?} (CANDLE_METAL_COMPUTE_PER_BUFFER={metal_compute_per_buffer})",
        args.num_games,
        args.mcts_iters,
        args.eval_workers,
        args.max_batch,
        args.coalesce_timeout_us,
        args.leaf_batch,
        device,
    );

    let job_counter = Arc::new(AtomicUsize::new(0));
    let games_completed = Arc::new(AtomicUsize::new(0));
    let trace_counter = Arc::new(AtomicUsize::new(0));
    let finish_milestones = finish_milestones(args.num_games);
    let results_mutex: Arc<std::sync::Mutex<Vec<GameResult>>> =
        Arc::new(std::sync::Mutex::new(Vec::with_capacity(args.num_games)));

    std::thread::scope(|scope| {
        for _ in 0..num_actors {
            let job_counter = job_counter.clone();
            let results_mutex = results_mutex.clone();
            let games_completed = games_completed.clone();
            let trace_counter = trace_counter.clone();
            let finish_milestones = finish_milestones.clone();
            let network1 = &network1;
            let network2 = &network2;
            let eval1 = &eval1;
            let eval2 = &eval2;
            let args = &args;
            let all_tribes = &all_tribes;
            let seed_list = &seed_list;
            let seed_entries = &seed_entries;
            scope.spawn(move || {
                loop {
                    let i = job_counter.fetch_add(1, Ordering::Relaxed);
                    if i >= args.num_games {
                        break;
                    }

                    let seed = seed_for_game(i, base_seed, seed_list.as_deref());
                    let swap_players = i % 2 == 1; // Swap every other game
                    let (p1_net, p2_net, p1_eval, p2_eval) = if swap_players {
                        (&**network2, &**network1, eval2, eval1)
                    } else {
                        (&**network1, &**network2, eval1, eval2)
                    };

                    // Anchor games: evenly spread across the run at rate
                    // anchor_frac (decayed from its starting value the same
                    // way prior_heuristic_weight is — see decay_crutch);
                    // the anchor's seat alternates by anchor ordinal (game
                    // parity alone would pin it to one seat at e.g. frac
                    // 0.25, where anchor games are all odd-i).
                    let anchor_frac = decay_crutch(
                        args.anchor_frac,
                        ANCHOR_FRAC_DECAY,
                        args.iteration.saturating_sub(args.anchor_decay_start),
                        args.decay_last_iter,
                        args.force_zero_crutches,
                    );
                    let anchor_ordinal = (((i + 1) as f32) * anchor_frac).floor() as usize;
                    let is_anchor = anchor_frac > 0.0
                        && anchor_ordinal > ((i as f32) * anchor_frac).floor() as usize;
                    // Greedy (score_move argmax), not the rollout Heuristic MCTS:
                    // measured first-village capture 1.00/t6.5 vs 0.94/t8.9 — the
                    // rollout noise drowned the ordering gradient. Greedy is also
                    // the exact distribution blend_heuristic_prior injects into the
                    // net's root, so anchor data and search priors agree.
                    // --anchor-seat pins the Greedy seat; otherwise it
                    // alternates so neither seat accumulates a side bias.
                    let anchor_first = match args.anchor_seat {
                        Some(1) => true,
                        Some(2) => false,
                        _ => anchor_ordinal % 2 == 0,
                    };
                    let (backend_seat1, backend_seat2) = if is_anchor {
                        if anchor_first {
                            (SearchBackend::Greedy, backend)
                        } else {
                            (backend, SearchBackend::Greedy)
                        }
                    } else {
                        (backend, backend)
                    };

                    // Seat roles for tempo aggregation: "model" (mirror seat),
                    // "model_vs_anchor" (net seat racing the anchor — the
                    // contested population), "anchor" (Greedy reference
                    // curve), "opponent" (league checkpoint seat).
                    let seat_roles: [&'static str; 2] = if is_anchor {
                        if anchor_first {
                            ["anchor", "model_vs_anchor"]
                        } else {
                            ["model_vs_anchor", "anchor"]
                        }
                    } else if has_opponent {
                        if swap_players {
                            ["opponent", "model"]
                        } else {
                            ["model", "opponent"]
                        }
                    } else {
                        ["model", "model"]
                    };

                    // Sample this game's own tribe pair, seeded off its game
                    // seed so runs stay reproducible while each game gets a
                    // distinct matchup. See `resolve_tribes` for the
                    // CLI > seed-file > random precedence.
                    use rand::SeedableRng;
                    let mut tribe_rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
                    let seed_file_tribes = tribes_for_game(i, seed_entries.as_deref());
                    let (t1, t2) = resolve_tribes(
                        &mut tribe_rng,
                        all_tribes,
                        &args.tribe1,
                        &args.tribe2,
                        seed_file_tribes,
                    );
                    let game_tribes = vec![t1, t2];

                    // Ensure panicking game doesnt kill the whole run
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        play_single_game(
                            p1_net,
                            p2_net,
                            p1_eval,
                            p2_eval,
                            args.mcts_iters,
                            i,
                            seed,
                            game_tribes,
                            args.iteration,
                            args.decay_last_iter,
                            args.force_zero_crutches,
                            args.gamemode,
                            backend_seat1,
                            backend_seat2,
                            args.value_trust,
                            args.leaf_batch,
                            progress_mode,
                            args.trace_villages,
                            args.trace_trigger,
                            args.trace_max,
                            &trace_counter,
                            args.dump_failed_dir.as_deref(),
                            args.dump_games_dir.as_deref(),
                            args.dump_turn_states.as_deref(),
                            args.dump_city_rewards.as_deref(),
                            args.dump_star_spend.as_deref(),
                            args.dump_reward_choices.as_deref(),
                            args.dump_level_completion.as_deref(),
                            args.dump_pop_spend_choices.as_deref(),
                            args.dump_macro_policy.as_deref(),
                            seat_roles,
                            args.shape_w_label,
                            args.shape_w_tree,
                            args.pursuit_w_label,
                            args.pursuit_w_tree,
                            args.unfreeze_opponent,
                            args.dagger_alpha,
                            args.goal_channels,
                            args.goal_w_tree,
                            macro_params,
                            args.max_turns,
                        )
                    }))
                    .unwrap_or_else(|_| {
                        eprintln!("[ERROR] Game {i} (seed {seed}) panicked — discarding its data");
                        None
                    });

                    if let Some(result) = result {
                        if progress_mode == ProgressMode::SampledFinish {
                            let done = games_completed.fetch_add(1, Ordering::Relaxed) + 1;
                            if finish_milestones.contains(&done) {
                                eprintln!(
                                    "[Progress] {}/{} games complete (game {} — {} moves, winner score {})",
                                    done,
                                    args.num_games,
                                    i,
                                    result.moves,
                                    result.winner_score,
                                );
                            }
                        }
                        results_mutex.lock().unwrap().push(result);
                    }
                }
            });
        }
    });

    let results: Vec<GameResult> = match Arc::try_unwrap(results_mutex) {
        Ok(mutex) => mutex.into_inner().unwrap(),
        Err(_) => {
            panic!("BUG: actor threads still hold a results_mutex reference after scope exit")
        }
    };

    let games_duration = games_start.elapsed();
    println!(
        "Game generation completed in: {:.2}s ({} games)",
        games_duration.as_secs_f32(),
        results.len()
    );
    println!(
        "  Average: {:.2}s per game",
        games_duration.as_secs_f32() / results.len().max(1) as f32
    );
    let total_moves_now: usize = results.iter().map(|r| r.moves).sum();
    let moves_per_sec = total_moves_now as f64 / games_duration.as_secs_f64().max(1e-9);
    println!(
        "  Throughput: {:.2} moves/sec ({} moves over {:.2}s)",
        moves_per_sec,
        total_moves_now,
        games_duration.as_secs_f32()
    );

    // Eval-server stats: aggregate across all shards (the number to compare
    // against the single-server baseline).
    let mut all_shard_stats: Vec<(&str, &EvalServerStats)> = Vec::new();
    for s in p1_servers.iter() {
        all_shard_stats.push(("p1", s.stats()));
    }
    if let Some(p2) = p2_servers.as_ref() {
        for s in p2 {
            all_shard_stats.push(("p2", s.stats()));
        }
    }

    let wall_s = games_duration.as_secs_f64().max(1e-9);

    // Aggregate across shards.
    let (mut agg_forwards, mut agg_rows, mut agg_max_batch, mut agg_busy_us) =
        (0u64, 0u64, 0u64, 0u64);
    let (mut agg_hits, mut agg_misses) = (0u64, 0u64);
    let (mut agg_compiles, mut agg_compile_us) = (0u64, 0u64);
    let (mut agg_prep_us, mut agg_wait_us, mut agg_post_us) = (0u64, 0u64, 0u64);
    for (_, s) in &all_shard_stats {
        agg_forwards += s.forwards.load(Ordering::Relaxed);
        agg_rows += s.rows.load(Ordering::Relaxed);
        agg_max_batch = agg_max_batch.max(s.max_batch.load(Ordering::Relaxed));
        agg_busy_us += s.busy_us.load(Ordering::Relaxed);
        agg_hits += s.cache_hits.load(Ordering::Relaxed);
        agg_misses += s.cache_misses.load(Ordering::Relaxed);
        agg_compiles += s.compiles.load(Ordering::Relaxed);
        agg_compile_us += s.compile_us.load(Ordering::Relaxed);
        agg_prep_us += s.prep_us.load(Ordering::Relaxed);
        agg_wait_us += s.wait_us.load(Ordering::Relaxed);
        agg_post_us += s.post_us.load(Ordering::Relaxed);
    }
    let agg_busy_s = agg_busy_us as f64 / 1e6;
    let agg_compile_s = agg_compile_us as f64 / 1e6;
    let agg_avg_batch = if agg_forwards > 0 {
        agg_rows as f64 / agg_forwards as f64
    } else {
        0.0
    };
    let agg_cache_total = agg_hits + agg_misses;
    let agg_cache_hit_rate = if agg_cache_total > 0 {
        agg_hits as f64 / agg_cache_total as f64
    } else {
        0.0
    };
    println!(
        "EVAL_SERVER_STATS_AGG: {{\"shards\": {}, \"forwards\": {}, \"rows\": {}, \"avg_batch\": {:.2}, \"max_batch\": {}, \"busy_s\": {:.2}, \"busy_frac\": {:.3}, \"prep_s\": {:.2}, \"wait_s\": {:.2}, \"post_s\": {:.2}, \"cache_hits\": {}, \"cache_misses\": {}, \"cache_hit_rate\": {:.3}, \"compiles\": {}, \"compile_s\": {:.3}, \"compile_frac_wall\": {:.4}, \"compile_frac_busy\": {:.4}}}",
        all_shard_stats.len(),
        agg_forwards,
        agg_rows,
        agg_avg_batch,
        agg_max_batch,
        agg_busy_s,
        agg_busy_s / wall_s,
        agg_prep_us as f64 / 1e6,
        agg_wait_us as f64 / 1e6,
        agg_post_us as f64 / 1e6,
        agg_hits,
        agg_misses,
        agg_cache_hit_rate,
        agg_compiles,
        agg_compile_s,
        agg_compile_s / wall_s,
        if agg_busy_s > 0.0 {
            agg_compile_s / agg_busy_s
        } else {
            0.0
        }
    );

    // Aggregate results
    let mut collected_spatial_maps: Vec<Tensor> = Vec::new();
    let mut collected_player_states: Vec<Tensor> = Vec::new();

    // Decomposed policy targets (7 heads)
    let mut collected_action_type: Vec<Vec<f32>> = Vec::new();
    let mut collected_source_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_target_spatial: Vec<Vec<f32>> = Vec::new();
    let mut collected_option: Vec<Vec<f32>> = Vec::new();

    let mut collected_values: Vec<f32> = Vec::new();
    let mut collected_progress: Vec<f32> = Vec::new();

    // Aux-head targets (see the aux_* helpers above GameResult).
    let num_techs = polyfish::types::TechnologyType::iter().count();
    let mut collected_aux_own: Vec<Vec<f32>> = Vec::new();
    let mut collected_aux_fog: Vec<Vec<f32>> = Vec::new();
    let mut collected_aux_spt: Vec<f32> = Vec::new(); // flat, 2 per step
    let mut collected_aux_tech: Vec<Vec<f32>> = Vec::new();
    let mut collected_aux_pursuit: Vec<f32> = Vec::new(); // scalar per step
    let mut collected_aux_city_spt: Vec<Vec<f32>> = Vec::new(); // board-sized per step

    // EXP_ELO_061 (Stage 3b): macro policy targets. Per-ROW mask, not just
    // per-file — even a macro-mcts-heavy run has steps with no ballot (the
    // opponent seat, an anchor game). Zero-filled + mask=0 there, matching
    // the aux-head-per-key-mask lesson: never let an absent target train
    // toward a fake zero.
    let mut collected_macro_stance: Vec<Vec<f32>> = Vec::new();
    let mut collected_macro_order: Vec<Vec<f32>> = Vec::new();
    let mut collected_macro_mask: Vec<f32> = Vec::new();

    let mut max_score = 0;
    let mut best_recap: Option<ModReplay> = None;
    let mut total_moves = 0; // both seats — throughput + sim-ratio denominators
    let mut total_net_moves = 0; // net-seat plies — the avg_moves behavior chart

    let mut p1_total = 0;
    let mut p2_total = 0;
    let mut p1_count = 0;
    let mut p2_count = 0;

    let mut total_captures = 0;
    let mut total_cap_ruins = 0;
    let mut total_cap_villages = 0;
    let mut total_cap_cities = 0;
    let mut total_cap_capitals = 0;
    let mut total_harvests = 0;
    let mut total_builds = 0;
    let mut total_research = 0;
    let mut total_attacks = 0;
    let mut total_abilities = 0;
    let mut total_revealed_tiles: i64 = 0;
    let mut total_captured_tiles: i64 = 0;
    let mut hub_totals: HashMap<polyfish::types::StructureType, (u32, i64, u32, u32)> =
        HashMap::new();
    // per type: (games, chosen_sum, best_sum, optimal_games, rank_pct_sum, cand_sum)
    let mut site_totals: HashMap<polyfish::types::StructureType, (u32, i64, i64, u32, f64, u64, i64, i64, u32)> =
        HashMap::new();
    let mut total_t2c = [0.0f64; 6]; // villages p50/p80/all, ruins p50/p80/all
    let (mut first_cap_seats, mut first_cap_captured) = (0u32, 0u32);
    let mut first_cap_turn_sum = 0.0f64;
    let mut first_cap_censored_sum = 0.0f64;
    // Contested anchor games: an embedded per-iteration strength peek vs the
    // Greedy anchor (n is small — ~anchor_frac * num_games — so ±1/sqrt(n)).
    let (mut anchor_games, mut anchor_net_wins) = (0u32, 0u32);
    let mut spt_sums: HashMap<i32, f64> = HashMap::new();
    let mut spt_counts: HashMap<i32, u32> = HashMap::new();
    let mut worth_sums: HashMap<i32, f64> = HashMap::new();
    let mut army_per_city_sums: HashMap<i32, f64> = HashMap::new();

    let mut total_moves_by_turn: HashMap<i32, HashMap<polyfish::types::MoveType, usize>> =
        HashMap::new();

    /// Per-role tempo accumulator across all player-games.
    #[derive(Default)]
    struct TempoAgg {
        /// turn -> ([cities, city_levels, spt, units, army_stars, revealed,
        /// techs, kills, trained_cum, lost_cum, stars_lost_cum] sums, sample count)
        by_turn: HashMap<i32, ([f64; 11], u32)>,
        trained: i64,
        granted: i64,
        lost: i64,
        giants: i64,
        stars_lost: i64,
        kills: i64,
        /// Σ star-cost of units still alive at game end — the "held" counterpart
        /// to `stars_lost`, on the same end-of-game time base.
        army_stars_end: i64,
        player_games: u32,
        /// cities >= 2/3/4: (reached count, turn sum over reached)
        reach: [(u32, f64); 3],
    }
    let mut tempo_aggs: HashMap<&'static str, TempoAgg> = HashMap::new();

    // Value-head calibration dump (one JSON line per net-seat step).
    let mut value_calib_file = args
        .dump_value_calib
        .as_ref()
        .and_then(|p| File::create(p).ok());

    let run_ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    // Trace runs are diagnostics, not training data: quarantine their games
    // under a prefix the training loop's games_* glob won't match.
    let shard_prefix = if args.trace_villages {
        "trace_games"
    } else {
        "games"
    };
    let mut shard_files: Vec<String> = Vec::new();
    let mut games_in_shard = 0usize;

    for result in results {
        total_moves += result.moves;
        total_net_moves += result.net_moves;
        total_revealed_tiles += result.revealed_tiles as i64;
        total_captured_tiles += result.captured_tiles as i64;
        for (k, (got, best, n_better, n_cands, cg, cb)) in &result.first_hub_rank {
            let e = site_totals.entry(*k).or_insert((0, 0, 0, 0, 0.0, 0, 0, 0, 0));
            e.0 += 1;
            e.1 += got;
            e.2 += best;
            e.3 += u32::from(got >= best);
            // 1.0 = no legal site would have ended with more partners.
            e.4 += 1.0 - f64::from(*n_better) / f64::from((*n_cands).max(1));
            e.5 += u64::from(*n_cands);
            e.6 += cg;
            e.7 += cb;
            e.8 += u32::from(cg >= cb);
        }
        for (k, (n, sum, starved, lost)) in &result.hub_levels {
            let e = hub_totals.entry(*k).or_insert((0, 0, 0, 0));
            e.0 += n;
            e.1 += sum;
            e.2 += starved;
            e.3 += lost;
        }
        for (&turn, &spt) in &result.spt_at_turn {
            *spt_sums.entry(turn).or_default() += spt as f64;
            *spt_counts.entry(turn).or_default() += 1;
        }
        // Shares spt_counts as its denominator — both are written by the same
        // milestone recorder, so a turn present in one is present in the other.
        for (&turn, &(worth, per_city)) in &result.army_ratios_at_turn {
            *worth_sums.entry(turn).or_default() += worth as f64;
            *army_per_city_sums.entry(turn).or_default() += per_city as f64;
        }
        for (acc, v) in total_t2c.iter_mut().zip([
            result.villages_t2c_p50,
            result.villages_t2c_p80,
            result.villages_t2c_all,
            result.ruins_t2c_p50,
            result.ruins_t2c_p80,
            result.ruins_t2c_all,
        ]) {
            *acc += v as f64;
        }
        first_cap_seats += result.villages_first_seats;
        first_cap_captured += result.villages_first_captured;
        first_cap_turn_sum += result.villages_first_turn_sum;
        first_cap_censored_sum += result.villages_first_censored_sum;
        if result.roles.contains(&"anchor") {
            anchor_games += 1;
            let winner_seat = (result.winner_id - 1) as usize;
            if winner_seat < 2 && result.roles[winner_seat] == "model_vs_anchor" {
                anchor_net_wins += 1;
            }
        }
        // Net-only: mirror games count both seats (both are net); anchor/league
        // games exclude the non-net (Greedy/opponent) seat, so the score metrics
        // reflect the net's play, not the opponent's.
        let mut game_net_max = 0;
        for (id, score) in &result.scores {
            if !is_net_seat(result.roles, *id) {
                continue;
            }
            game_net_max = game_net_max.max(*score);
            if *id == 1 {
                p1_total += score;
                p1_count += 1;
            } else if *id == 2 {
                p2_total += score;
                p2_count += 1;
            }
        }
        // Best net seat rather than `winner_score`: an anchor/league opponent
        // win would otherwise set the reported max and get its replay saved.
        if game_net_max > max_score {
            max_score = game_net_max;
            best_recap = Some(result.recap.clone());
        }

        total_captures += result
            .action_counts
            .get(&polyfish::types::MoveType::Capture)
            .copied()
            .unwrap_or(0);
        total_cap_ruins += result.cap_ruins;
        total_cap_villages += result.cap_villages;
        total_cap_cities += result.cap_cities;
        total_cap_capitals += result.cap_capitals;
        total_harvests += result
            .action_counts
            .get(&polyfish::types::MoveType::Harvest)
            .copied()
            .unwrap_or(0);
        total_builds += result
            .action_counts
            .get(&polyfish::types::MoveType::Build)
            .copied()
            .unwrap_or(0);
        total_research += result
            .action_counts
            .get(&polyfish::types::MoveType::Research)
            .copied()
            .unwrap_or(0);
        total_attacks += result
            .action_counts
            .get(&polyfish::types::MoveType::Attack)
            .copied()
            .unwrap_or(0);
        total_abilities += result
            .action_counts
            .get(&polyfish::types::MoveType::Ability)
            .copied()
            .unwrap_or(0);

        for (turn, counts) in &result.moves_by_turn {
            let entry = total_moves_by_turn.entry(*turn).or_default();
            for (mt, c) in counts {
                *entry.entry(*mt).or_insert(0) += c;
            }
        }

        for (&pid, track) in &result.tempo {
            let seat = (pid - 1) as usize;
            if seat >= 2 {
                continue;
            }
            let agg = tempo_aggs.entry(result.roles[seat]).or_default();
            agg.player_games += 1;
            agg.trained += track.units_trained as i64;
            agg.granted += track.units_granted as i64;
            agg.lost += track.units_lost as i64;
            agg.giants += track.giants_made as i64;
            agg.stars_lost += track.army_stars_lost as i64;
            // End-of-game state comes from the final forced sample, so games
            // that ended early are still counted at their true final turn.
            if let Some(last) = track.samples.last() {
                agg.kills += last.kills as i64;
                agg.army_stars_end += last.army_stars as i64;
            }
            for s in &track.samples {
                let (sums, n) = agg.by_turn.entry(s.turn).or_default();
                for (acc, v) in sums.iter_mut().zip([
                    s.cities,
                    s.city_levels,
                    s.spt,
                    s.units,
                    s.army_stars,
                    s.revealed,
                    s.techs,
                    s.kills,
                    s.trained_cum,
                    s.lost_cum,
                    s.stars_lost_cum,
                ]) {
                    *acc += v as f64;
                }
                *n += 1;
            }
            for (slot, target) in agg.reach.iter_mut().zip([2, 3, 4]) {
                if let Some(s) = track.samples.iter().find(|s| s.cities >= target) {
                    slot.0 += 1;
                    slot.1 += s.turn as f64;
                }
            }
        }

        // Backpropagate value
        // Domination: Win/Loss is the primary signal.
        // The winner gets +1.0, loser gets -1.0.
        // If timeout, use score differential as a softer signal.
        let final_scores = &result.scores;

        let label_steps: Vec<LabelStep> = result.history.iter().map(LabelStep::from).collect();
        // EXP_ELO_025: outcome-space labels — z anchors the TD tail too.
        let wl_z: Option<HashMap<i32, f32>> = if args.wl_labels {
            Some(
                result
                    .scores
                    .keys()
                    .map(|&id| (id, if id == result.winner_id { 1.0 } else { -1.0 }))
                    .collect(),
            )
        } else {
            None
        };
        let td_deltas = td_lambda_labels(
            &label_steps,
            &result.final_potentials,
            args.td_lambda,
            args.label_rel_w,
            wl_z.as_ref(),
            args.td_missing_bootstrap,
        );

        let spt_steps: Vec<SptStep> = result
            .history
            .iter()
            .map(|s| SptStep {
                player_id: s.player_id,
                turn: s.turn,
                my_spt: s.my_spt,
                opp_spt: s.opp_spt,
            })
            .collect();
        let spt_cp = spt_checkpoints_by_player(&spt_steps);
        let city_spt_steps: Vec<CitySptStep> = result
            .history
            .iter()
            .map(|s| CitySptStep {
                player_id: s.player_id,
                turn: s.turn,
                cities: s.city_spt.clone(),
            })
            .collect();
        let city_spt_cp = city_spt_checkpoints(&city_spt_steps);

        let game_winner_id = result.winner_id;

        for (step_idx, step) in result.history.into_iter().enumerate() {
            let HistoryStep {
                features,
                policy: policy_data,
                player_id: p_id,
                turn,
                enemy_units,
                pursuit,
                my_score: step_my_score,
                opp_score: step_opp_score,
                root_value: step_root_value,
                root_own_value: step_root_own_value,
                macro_ballot,
                ..
            } = step;
            let flat_map = features
                .spatial_map
                .flatten_all()
                .expect("BUG: Failed to flatten spatial map tensor");
            collected_spatial_maps.push(flat_map);

            let flat_player = features
                .player_state
                .flatten_all()
                .expect("BUG: Failed to flatten player state tensor");
            collected_player_states.push(flat_player);

            collected_action_type.push(policy_data.action_type);
            collected_source_spatial.push(policy_data.source_spatial);
            collected_target_spatial.push(policy_data.target_spatial);
            collected_option.push(policy_data.move_option);

            // Perfection: Score-based value target
            // Every game produces a meaningful score — use normalized differential
            let my_final = final_scores.get(&p_id).copied().unwrap_or(0) as f32;
            let opp_final = final_scores
                .iter()
                .filter(|(id, _)| **id != p_id)
                .map(|(_, score)| *score as f32)
                .next()
                .unwrap_or(0.0);

            // Asymmetric Reward Shaping to fix P1 advantage
            let (mut my_adjusted, mut opp_adjusted) = (my_final, opp_final);
            if !args.no_reward_shaping {
                let penalty = 0.05; // 5% adjustment
                if p_id == 1 {
                    my_adjusted = my_final * (1.0 - penalty);
                    opp_adjusted = opp_final * (1.0 + penalty);
                } else if p_id == 2 {
                    my_adjusted = my_final * (1.0 + penalty);
                    opp_adjusted = opp_final * (1.0 - penalty);
                }
            }

            // Normalize by combined economic activity with scaling multiplier
            let combined_score = my_adjusted + opp_adjusted;
            // to spread distribution into useful training range
            let scaling_factor = args.outcome_scale;
            let relative_outcome = if combined_score > 0.0 {
                let ratio = (my_adjusted - opp_adjusted) / combined_score;
                (ratio * scaling_factor).clamp(-1.0, 1.0)
            } else {
                0.0 // Both players scored 0 - treat as draw
            };

            // Absolute value: final score vs fixed yardstick, not current scoreboard.
            let abs_outcome = (my_final / GOOD_BOT_FINAL_SCORE).clamp(0.0, 1.0) * 2.0 - 1.0;
            let final_outcome = if args.wl_labels {
                // EXP_ELO_011: the score ratio under-punishes close losses
                // (4257 vs 4676 reads -0.14); win/loss makes them -1.
                if p_id == game_winner_id {
                    1.0
                } else {
                    -1.0
                }
            } else {
                (FINAL_OUTCOME_REL_W * relative_outcome
                    + (1.0 - FINAL_OUTCOME_REL_W) * abs_outcome)
                    .clamp(-1.0, 1.0)
            };

            let value = if !args.no_reward_shaping {
                // TD delta carries per-action credit; the final-outcome tail
                // carries the long-horizon signal.
                (args.td_w * td_deltas[step_idx] + (1.0 - args.td_w) * final_outcome)
                    .clamp(-1.0, 1.0)
            } else {
                final_outcome.clamp(-1.0, 1.0)
            };

            collected_values.push(value);

            // Value-head calibration: NN prediction vs current-score-ratio vs
            // the actual final outcome, for net seats that ran a real search.
            if let (Some(f), Some(rv)) = (value_calib_file.as_mut(), step_root_value) {
                if is_net_seat(result.roles, p_id) {
                    let raw = step_root_own_value.unwrap_or(rv);
                    let _ = writeln!(
                        f,
                        "{{\"turn\":{turn},\"my\":{step_my_score},\"opp\":{step_opp_score},\"root_value\":{rv},\"raw_value\":{raw},\"final_outcome\":{final_outcome},\"value_target\":{value}}}"
                    );
                }
            }

            let my_final_cities = result.final_cities.get(&p_id).copied().unwrap_or(0) as f32;
            let total_cities = result.total_cities as f32;
            let progress_target = if total_cities > 0.0 {
                (my_final_cities / total_cities).clamp(0.0, 1.0) * 2.0 - 1.0
            } else {
                -1.0
            };
            collected_progress.push(progress_target);

            let opp_id = final_scores
                .keys()
                .copied()
                .find(|id| *id != p_id)
                .unwrap_or(p_id);
            collected_aux_own.push(ownership_from_pov(&result.final_owner, p_id));
            collected_aux_fog.push(enemy_units);
            let (spt_my, spt_opp) = spt_target(
                spt_cp.get(&p_id),
                turn,
                result.final_spt.get(&p_id).copied().unwrap_or(0),
                result.final_spt.get(&opp_id).copied().unwrap_or(0),
            );
            collected_aux_spt.push(spt_my as f32 / 20.0);
            collected_aux_spt.push(spt_opp as f32 / 20.0);
            collected_aux_pursuit.push(pursuit);
            if let Some((candidates, visits)) = &macro_ballot {
                let (stance, order) = macro_policy_targets(candidates, visits);
                collected_macro_stance.push(stance);
                collected_macro_order.push(order);
                collected_macro_mask.push(1.0);
            } else {
                collected_macro_stance.push(vec![0.0; 4]);
                collected_macro_order.push(vec![0.0; 3 * features::MAP_SIZE * features::MAP_SIZE]);
                collected_macro_mask.push(0.0);
            }
            collected_aux_city_spt.push(city_spt_target(
                city_spt_cp.get(&p_id),
                turn,
                features::MAP_SIZE * features::MAP_SIZE,
            ));
            collected_aux_tech.push(
                result
                    .final_tech
                    .get(&opp_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; num_techs]),
            );
        }

        games_in_shard += 1;
        if games_in_shard >= SHARD_GAMES && !collected_spatial_maps.is_empty() {
            let path = format!(
                "{shard_prefix}_{run_ts}_p{}.safetensors",
                shard_files.len()
            );
            flush_shard(
                std::mem::take(&mut collected_spatial_maps),
                std::mem::take(&mut collected_player_states),
                std::mem::take(&mut collected_action_type),
                std::mem::take(&mut collected_source_spatial),
                std::mem::take(&mut collected_target_spatial),
                std::mem::take(&mut collected_option),
                std::mem::take(&mut collected_values),
                std::mem::take(&mut collected_progress),
                std::mem::take(&mut collected_aux_own),
                std::mem::take(&mut collected_aux_fog),
                std::mem::take(&mut collected_aux_spt),
                std::mem::take(&mut collected_aux_pursuit),
                std::mem::take(&mut collected_aux_city_spt),
                std::mem::take(&mut collected_aux_tech),
                num_techs,
                std::mem::take(&mut collected_macro_stance),
                std::mem::take(&mut collected_macro_order),
                std::mem::take(&mut collected_macro_mask),
                &device,
                &path,
            )?;
            shard_files.push(path);
            games_in_shard = 0;
        }
    }

    let mut net_games = 0u32;
    let (mut net_trained, mut net_granted, mut net_lost, mut net_giants) = (0i64, 0i64, 0i64, 0i64);
    let mut net_kills = 0i64;
    let mut net_reach = [(0u32, 0.0f64); 3];
    for role in ["model", "model_vs_anchor"] {
        if let Some(a) = tempo_aggs.get(role) {
            net_games += a.player_games;
            net_trained += a.trained;
            net_granted += a.granted;
            net_lost += a.lost;
            net_giants += a.giants;
            net_kills += a.kills;
            for (dst, src) in net_reach.iter_mut().zip(a.reach.iter()) {
                dst.0 += src.0;
                dst.1 += src.1;
            }
        }
    }
    let per_net_game = |x: i64| {
        if net_games > 0 {
            x as f64 / f64::from(net_games)
        } else {
            0.0
        }
    };

    // Print Average Metrics. avg_score is net-only (see the score loop): the
    // mean score over net seats across games, so anchor/league games don't
    // blend the opponent's score into the net's performance chart.
    let net_score_count = p1_count + p2_count;
    let avg_score = if net_score_count > 0 {
        (p1_total + p2_total) as f32 / net_score_count as f32
    } else {
        0.0
    };
    let avr_moves = per_net_game(total_net_moves as i64) as f32;
    let p1_avg = if p1_count > 0 {
        p1_total as f32 / p1_count as f32
    } else {
        0.0
    };
    let p2_avg = if p2_count > 0 {
        p2_total as f32 / p2_count as f32
    } else {
        0.0
    };

    // Per net PLAYER-GAME, not per game: these counters only accrue on net
    // seats, and a mirror game supplies two of them to an anchor game's one.
    // Dividing by games made the whole family drift as anchor_frac decayed.
    let avg_captures = per_net_game(total_captures as i64) as f32;
    let avg_cap_ruins = per_net_game(total_cap_ruins as i64) as f32;
    let avg_cap_villages = per_net_game(total_cap_villages as i64) as f32;
    let avg_cap_cities = per_net_game(total_cap_cities as i64) as f32;
    let avg_cap_capitals = per_net_game(total_cap_capitals as i64) as f32;
    let avg_harvests = per_net_game(total_harvests as i64) as f32;
    let avg_builds = per_net_game(total_builds as i64) as f32;
    let avg_research = per_net_game(total_research as i64) as f32;
    let avg_attacks = per_net_game(total_attacks as i64) as f32;
    let avg_abilities = per_net_game(total_abilities as i64) as f32;
    let avg_revealed_tiles = per_net_game(total_revealed_tiles) as f32;
    let avg_captured_tiles = per_net_game(total_captured_tiles) as f32;

    // -1.0 when the net built no hubs at all: 0.0 is a legal level.
    let (hub_n, hub_sum, hub_starved, hub_lost) = hub_totals.values().fold(
        (0u32, 0i64, 0u32, 0u32),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3),
    );
    let avg_hub_level = if hub_n > 0 {
        hub_sum as f32 / hub_n as f32
    } else {
        -1.0
    };
    let hub_starved_frac = if hub_n > 0 {
        hub_starved as f32 / hub_n as f32
    } else {
        -1.0
    };
    let first_hub_site: serde_json::Value = site_totals
        .iter()
        .map(|(k, (games, got, best, optimal, rank_pct, cands, cg, cb, ceil_opt))| {
            let g = f64::from(*games).max(1.0);
            (
                format!("{k:?}"),
                serde_json::json!({
                    "games": games,
                    "chosen_partners": *got as f64 / g,
                    "best_available_partners": *best as f64 / g,
                    "optimal_frac": f64::from(*optimal) / g,
                    "mean_rank_pct": rank_pct / g,
                    "sites_available": *cands as f64 / g,
                    "ceiling_chosen": *cg as f64 / g,
                    "ceiling_best_available": *cb as f64 / g,
                    "ceiling_optimal_frac": f64::from(*ceil_opt) / g,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();

    let hub_lost_frac = if hub_n > 0 {
        hub_lost as f32 / hub_n as f32
    } else {
        -1.0
    };
    let avg_hubs_built = per_net_game(i64::from(hub_n)) as f32;
    let hub_levels_by_type: serde_json::Value = hub_totals
        .iter()
        .map(|(k, (n, sum, starved, lost))| {
            (
                format!("{k:?}"),
                serde_json::json!({
                    "built": n,
                    "mean_level": *sum as f32 / *n as f32,
                    "starved_frac": *starved as f32 / *n as f32,
                    "lost_frac": *lost as f32 / *n as f32,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>()
        .into();
    let avg_kills = per_net_game(net_kills) as f32;

    let avg_spt_at = |turn: i32| -> f32 {
        let c = spt_counts.get(&turn).copied().unwrap_or(0);
        if c == 0 {
            0.0
        } else {
            (spt_sums.get(&turn).copied().unwrap_or(0.0) / c as f64) as f32
        }
    };
    // -1.0 (not 0.0) when a turn was never reached: 0 is a legal value for both
    // ratios, so a sentinel is the only way to distinguish "no army" from "no
    // data" downstream.
    let avg_ratio_at = |sums: &HashMap<i32, f64>, turn: i32| -> f32 {
        let c = spt_counts.get(&turn).copied().unwrap_or(0);
        if c == 0 {
            -1.0
        } else {
            (sums.get(&turn).copied().unwrap_or(0.0) / c as f64) as f32
        }
    };

    // "typical move by turn N" chart data: {"<turn>": {"<MoveType>": count, ...}, ...}
    let moves_by_turn = {
        let mut turns_sorted: Vec<&i32> = total_moves_by_turn.keys().collect();
        turns_sorted.sort();
        let mut turn_map = serde_json::Map::new();
        for turn in turns_sorted {
            let mut counts_map = serde_json::Map::new();
            for (mt, c) in &total_moves_by_turn[turn] {
                counts_map.insert(format!("{mt:?}"), serde_json::Value::from(*c));
            }
            turn_map.insert(turn.to_string(), serde_json::Value::Object(counts_map));
        }
        serde_json::Value::Object(turn_map)
    };

    // Final partial shard.
    if !collected_spatial_maps.is_empty() {
        let path = format!(
            "{shard_prefix}_{run_ts}_p{}.safetensors",
            shard_files.len()
        );
        flush_shard(
            std::mem::take(&mut collected_spatial_maps),
            std::mem::take(&mut collected_player_states),
            std::mem::take(&mut collected_action_type),
            std::mem::take(&mut collected_source_spatial),
            std::mem::take(&mut collected_target_spatial),
            std::mem::take(&mut collected_option),
            std::mem::take(&mut collected_values),
            std::mem::take(&mut collected_progress),
            std::mem::take(&mut collected_aux_own),
            std::mem::take(&mut collected_aux_fog),
            std::mem::take(&mut collected_aux_spt),
            std::mem::take(&mut collected_aux_pursuit),
            std::mem::take(&mut collected_aux_city_spt),
            std::mem::take(&mut collected_aux_tech),
            num_techs,
            std::mem::take(&mut collected_macro_stance),
            std::mem::take(&mut collected_macro_order),
            std::mem::take(&mut collected_macro_mask),
            &device,
            &path,
        )?;
        shard_files.push(path);
    }
    // METRICS carries the first shard (the value-distribution reader wants a
    // ~64-game sample, not every file); everything else globs the _p* stem.
    let games_file = shard_files.first().cloned().unwrap_or_default();

    // Save BEST game as replay
    if let Some(recap) = best_recap {
        let replay_filename = format!(
            "replays/high_scores/best_game_score_{}_{}.json",
            max_score, run_ts
        );
        if let Ok(json) = serde_json::to_string_pretty(&recap) {
            if let Ok(mut file) = File::create(&replay_filename) {
                let _ = file.write_all(json.as_bytes());
                println!(
                    "🏆 Highest score game ({}) saved to {}",
                    max_score, replay_filename
                );
            }
        }
    }

    // Tempo curves per role + net-seat scalar aggregates ("model" mirror
    // seats + "model_vs_anchor" contested seats combined; "anchor" is the
    // Greedy reference curve and stays out of the scalars).
    let tempo_by_turn = {
        let mut roles_map = serde_json::Map::new();
        for (role, agg) in &tempo_aggs {
            let mut turn_map = serde_json::Map::new();
            let mut turns: Vec<i32> = agg.by_turn.keys().copied().collect();
            turns.sort_unstable();
            for t in turns {
                let (sums, n) = &agg.by_turn[&t];
                let nf = f64::from(*n).max(1.0);
                let mut o = serde_json::Map::new();
                for (name, v) in [
                    "cities",
                    "city_levels",
                    "spt",
                    "units",
                    "army_stars",
                    "revealed",
                    "techs",
                    "kills",
                    "trained_cum",
                    "lost_cum",
                    "stars_lost_cum",
                ]
                .iter()
                .zip(sums.iter())
                {
                    o.insert((*name).to_string(), serde_json::Value::from(v / nf));
                }
                o.insert("n".to_string(), serde_json::Value::from(*n));
                turn_map.insert(t.to_string(), serde_json::Value::Object(o));
            }
            // Unbiased per-player-game totals (the last-turn-key cums under-
            // count games that ended early). "_totals" is non-numeric, so
            // turn-key consumers must filter it.
            let pg = f64::from(agg.player_games).max(1.0);
            let mut totals = serde_json::Map::new();
            for (name, v) in [
                ("trained", agg.trained),
                ("granted", agg.granted),
                ("lost", agg.lost),
                ("giants", agg.giants),
                ("stars_lost", agg.stars_lost),
                ("kills", agg.kills),
                ("army_stars_end", agg.army_stars_end),
            ] {
                totals.insert(name.to_string(), serde_json::Value::from(v as f64 / pg));
            }
            totals.insert(
                "n_games".to_string(),
                serde_json::Value::from(agg.player_games),
            );
            turn_map.insert("_totals".to_string(), serde_json::Value::Object(totals));
            roles_map.insert((*role).to_string(), serde_json::Value::Object(turn_map));
        }
        serde_json::Value::Object(roles_map)
    };
    let reach_rate = |i: usize| {
        if net_games > 0 {
            f64::from(net_reach[i].0) / f64::from(net_games)
        } else {
            0.0
        }
    };
    let reach_turn = |i: usize| {
        if net_reach[i].0 > 0 {
            net_reach[i].1 / f64::from(net_reach[i].0)
        } else {
            -1.0
        }
    };

    let metrics = json!({
        "num_games": args.num_games,
        "avg_score": avg_score,
        "max_score": max_score,
        "avg_moves": avr_moves,
        "p1_avg": p1_avg,
        "p2_avg": p2_avg,
        "avg_captures": avg_captures,
        "avg_cap_ruins": avg_cap_ruins,
        "avg_cap_villages": avg_cap_villages,
        "avg_cap_cities": avg_cap_cities,
        "avg_cap_capitals": avg_cap_capitals,
        "avg_harvests": avg_harvests,
        "avg_builds": avg_builds,
        "avg_research": avg_research,
        "avg_attacks": avg_attacks,
        "avg_abilities": avg_abilities,
        "avg_kills": avg_kills,
        "avg_revealed_tiles": avg_revealed_tiles,
        "avg_captured_tiles": avg_captured_tiles,
        "first_hub_site": first_hub_site,
        "avg_hub_level": avg_hub_level,
        "avg_hubs_built": avg_hubs_built,
        "hub_starved_frac": hub_starved_frac,
        "hub_lost_frac": hub_lost_frac,
        "hub_levels_by_type": hub_levels_by_type,
        "avg_spt_t0": avg_spt_at(0),
        "avg_spt_t5": avg_spt_at(5),
        "avg_spt_t10": avg_spt_at(10),
        "avg_spt_t15": avg_spt_at(15),
        "avg_spt_t20": avg_spt_at(20),
        "avg_spt_t25": avg_spt_at(25),
        "avg_spt_t30": avg_spt_at(30),
        "unit_worth_t15": avg_ratio_at(&worth_sums, 15),
        "unit_worth_t25": avg_ratio_at(&worth_sums, 25),
        "army_stars_per_city_t15": avg_ratio_at(&army_per_city_sums, 15),
        "army_stars_per_city_t25": avg_ratio_at(&army_per_city_sums, 25),
        // Per NET SEAT, not per game — a mirror game contributes two seats and
        // an anchor game one, so a games denominator blended two different
        // per-seat probabilities and drifted with anchor_frac.
        "villages_t2c_first": if first_cap_seats > 0 {
            (first_cap_censored_sum / f64::from(first_cap_seats)) as f32
        } else {
            -1.0
        },
        "villages_first_rate": if first_cap_seats > 0 {
            (f64::from(first_cap_captured) / f64::from(first_cap_seats)) as f32
        } else {
            0.0
        },
        "villages_t2c_first_cond": if first_cap_captured > 0 {
            (first_cap_turn_sum / f64::from(first_cap_captured)) as f32
        } else {
            -1.0
        },
        "tribes": format!(
            "{}+{}",
            args.tribe1.as_deref().unwrap_or("random"),
            args.tribe2.as_deref().unwrap_or("random")
        ),
        "villages_t2c_p50": (total_t2c[0] / args.num_games as f64) as f32,
        "villages_t2c_p80": (total_t2c[1] / args.num_games as f64) as f32,
        "villages_t2c_all": (total_t2c[2] / args.num_games as f64) as f32,
        "ruins_t2c_p50": (total_t2c[3] / args.num_games as f64) as f32,
        "ruins_t2c_p80": (total_t2c[4] / args.num_games as f64) as f32,
        "ruins_t2c_all": (total_t2c[5] / args.num_games as f64) as f32,
        "games_file": games_file,
        "moves_by_turn": moves_by_turn,
        "avg_units_spawned": per_net_game(net_trained),
        "avg_units_granted": per_net_game(net_granted),
        "avg_units_lost": per_net_game(net_lost),
        "avg_giants_made": per_net_game(net_giants),
        "t2c_2nd_rate": reach_rate(0),
        "t2c_2nd_turn": reach_turn(0),
        "t2c_3rd_rate": reach_rate(1),
        "t2c_3rd_turn": reach_turn(1),
        "t2c_4th_rate": reach_rate(2),
        "t2c_4th_turn": reach_turn(2),
        "anchor_games": anchor_games,
        "anchor_net_wr": if anchor_games > 0 {
            f64::from(anchor_net_wins) / f64::from(anchor_games)
        } else {
            -1.0
        },
        "tempo_by_turn": tempo_by_turn,
        "gate_blocks": polyfish::ai::gumbel_mcts::gate_stats::snapshot(),
    });
    std::fs::write(
        ".last_self_play_metrics.json",
        serde_json::to_string(&metrics)?,
    )?;

    let total_duration = start_time.elapsed();
    println!("\n=== Self-Play Complete ===");
    println!("Total time: {:.2}s", total_duration.as_secs_f32());
    println!("Breakdown:");
    println!(
        "  - Game generation: {:.2}s ({:.1}%)",
        games_duration.as_secs_f32(),
        100.0 * games_duration.as_secs_f32() / total_duration.as_secs_f32()
    );
    let final_moves_per_sec = total_moves as f64 / games_duration.as_secs_f64().max(1e-9);
    println!(
        "  - Throughput: {:.2} moves/sec ({} moves)",
        final_moves_per_sec, total_moves
    );
    // How often search crossed a turn boundary in-tree (simulated EndTurn
    // edges only; real played moves don't count). ~0/move decision means the
    // tree essentially never sees beyond the current turn.
    let sim_end_turns =
        polyfish::game::SIM_END_TURN_EDGES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - Sim EndTurn edges: {} total ({:.2} per move decision)",
        sim_end_turns,
        sim_end_turns as f64 / (total_moves as f64).max(1.0)
    );
    // How often a simulated move failed to execute against the replayed state
    // (tree-reuse staleness in Gumbel MCTS — see SIM_MOVE_FAILURES doc comment
    // in game.rs). Set POLYFISH_VERBOSE_SIM_FAILURES=1 for illegal_moves/*.json dumps.
    let sim_move_failures =
        polyfish::game::SIM_MOVE_FAILURES.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - Sim move failures: {} total ({:.2} per move decision)",
        sim_move_failures,
        sim_move_failures as f64 / (total_moves as f64).max(1.0)
    );
    // Ply-distillation throughput envelope input (EXP_ELO_061 GPU-ply-work
    // plan, Phase 0): how many rank_plies calls (rollout + real-commit)
    // and candidate moves per real move decision under macro-mcts. Zero
    // under gumbel (rank_plies is macro-mcts-only).
    let rank_plies_calls =
        polyfish::ai::search::macro_exec::RANK_PLIES_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let rank_plies_candidates = polyfish::ai::search::macro_exec::RANK_PLIES_CANDIDATES
        .load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - rank_plies calls: {} total ({:.2} per move decision), {} candidates total ({:.1} per call)",
        rank_plies_calls,
        rank_plies_calls as f64 / (total_moves as f64).max(1.0),
        rank_plies_candidates,
        rank_plies_candidates as f64 / (rank_plies_calls as f64).max(1.0)
    );
    println!(
        "  - EXP_ELO_083 tech-limit no-recommendation rejections (diagnostic, temporary): {} candidates",
        polyfish::ai::search::goal_aux::TECH_LIMIT_REJECTIONS.load(std::sync::atomic::Ordering::Relaxed)
    );
    if let Ok(m) = polyfish::ai::search::goal_aux::TECH_LIMIT_REJECTIONS_BY_TECH.lock() {
        let mut by_tech: Vec<(&polyfish::types::TechnologyType, &u64)> = m.iter().collect();
        by_tech.sort_by(|a, b| b.1.cmp(a.1));
        let top: Vec<String> =
            by_tech.iter().take(12).map(|(t, c)| format!("{t:?}:{c}")).collect();
        println!("  - EXP_ELO_088 rejections by tech (diagnostic, top 12): {}", top.join(", "));
    }
    {
        let eligible = polyfish::ai::search::macro_exec::ENDTURN_ELIGIBLE_PLIES
            .load(std::sync::atomic::Ordering::Relaxed);
        let chosen = polyfish::ai::search::macro_exec::ENDTURN_CHOSEN_WITH_ALTERNATIVES
            .load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "  - EXP_ELO_085 EndTurn chosen despite alternatives (diagnostic, temporary): {chosen}/{eligible} ({:.3}%)",
            100.0 * chosen as f64 / (eligible as f64).max(1.0)
        );
    }
    println!(
        "  - EXP_ELO_095 shared-attacker partial weights (diagnostic, temporary): {} entries",
        polyfish::ai::combat::SHARED_ATTACKER_PARTIAL_WEIGHTS.load(std::sync::atomic::Ordering::Relaxed)
    );
    {
        let cover_total =
            polyfish::ai::combat::DEFEND_CREDIT_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let cover_partial =
            polyfish::ai::combat::DEFEND_CREDIT_PARTIAL.load(std::sync::atomic::Ordering::Relaxed);
        let hold_total =
            polyfish::ai::combat::DEFEND_HOLD_TOTAL.load(std::sync::atomic::Ordering::Relaxed);
        let hold_partial =
            polyfish::ai::combat::DEFEND_HOLD_PARTIAL.load(std::sync::atomic::Ordering::Relaxed);
        println!(
            "  - EXP_ELO_096 defend_cover fractional credit (diagnostic, temporary): {cover_partial}/{cover_total} ({:.3}%) assignments partial",
            100.0 * cover_partial as f64 / (cover_total as f64).max(1.0)
        );
        println!(
            "  - EXP_ELO_096 defend_hold fractional margin (diagnostic, temporary): {hold_partial}/{hold_total} ({:.3}%) evaluations partial",
            100.0 * hold_partial as f64 / (hold_total as f64).max(1.0)
        );
    }
    // Micro-mcts Phase 0 (throughput/cache-hit probe, POLYFISH_MICRO_PROBE_SIMS):
    // zero unless that env var is set. Note the rank_plies numbers above also
    // inflate while this probe is active -- its own continuation walk calls
    // rank_view/rank_plies, so that's the probe's real CPU cost showing up in
    // already-existing instrumentation, not contamination to filter out.
    let micro_probe_evals =
        polyfish::ai::search::macro_mcts::MICRO_PROBE_EVALS.load(std::sync::atomic::Ordering::Relaxed);
    let micro_probe_failures = polyfish::ai::search::macro_mcts::MICRO_PROBE_SIM_FAILURES
        .load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - micro-probe evals: {} total ({:.2} per move decision), {} sim failures",
        micro_probe_evals,
        micro_probe_evals as f64 / (total_moves as f64).max(1.0),
        micro_probe_failures
    );
    let micro_mcts_calls =
        polyfish::ai::search::micro_mcts::MICRO_MCTS_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    let micro_mcts_overrides = polyfish::ai::search::micro_mcts::MICRO_MCTS_OVERRIDES
        .load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - micro-mcts calls: {} total, {} overrode rank_view's top pick ({:.1}%)",
        micro_mcts_calls,
        micro_mcts_overrides,
        micro_mcts_overrides as f64 / (micro_mcts_calls as f64).max(1.0) * 100.0
    );
    let micro_carry_attempts = polyfish::ai::search::micro_mcts::MICRO_CARRY_ATTEMPTS
        .load(std::sync::atomic::Ordering::Relaxed);
    let micro_carry_hits =
        polyfish::ai::search::micro_mcts::MICRO_CARRY_HITS.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  - micro-mcts root-advancement: {} carries offered, {} candidate children spliced in ({:.2} avg per carry)",
        micro_carry_attempts,
        micro_carry_hits,
        micro_carry_hits as f64 / (micro_carry_attempts as f64).max(1.0)
    );
    polyfish::ai::search::macro_exec::dphi_probe_flush();

    // Deterministic teardown. Drop the evaluator handles first — these hold the
    // only remaining request-channel senders, so dropping them makes each eval
    // thread's `recv` error out and return, which drops its inference backend
    // (and any MPS/device tensors). Then join the threads so that drop finishes
    // *before* the process starts static/atexit teardown. Without this the
    // detached eval thread races libtorch's atexit mutex destruction and the
    // process aborts with "recursive_mutex lock failed: Invalid argument".
    drop(eval1);
    drop(eval2);
    for server in p1_servers {
        server.shutdown();
    }
    if let Some(p2) = p2_servers {
        for server in p2 {
            server.shutdown();
        }
    }

    Ok(())
}

