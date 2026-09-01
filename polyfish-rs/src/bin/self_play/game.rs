//! One self-play game, start to finish: build the map and the two agents,
//! run the ply loop, and hand back a `GameResult`.
//!
//! The per-ply diagnostics this drives live in `dumps` and `traces`; the
//! trackers it threads live in `stats` and `tempo`. Labels are NOT computed
//! here -- `GameResult.history` goes back raw and `dataset` turns it into
//! training targets.

use polyfish::TribeType;
use polyfish::ai::brain::{Brain, SearchBackend};
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
use std::io::Write;
use std::sync::atomic::AtomicUsize;

use crate::crutches::{HEURISTIC_PRIOR_DECAY, HEURISTIC_PRIOR_W0, decay_crutch};
use crate::dumps::{PlanTracker, open_game_jsonl, write_choice_dumps, write_spend_dumps, dump_macro_policy_row, dump_turn_state, update_plans};
use crate::labels::{POLICY_TARGET_Q_RAMP_ITERS, final_ground_truth,
                    macro_ballot_for_history_step, eco_ceiling_for_history_step, enemy_unit_grid};
use crate::result::{DecomposedPolicyData, GameResult, HistoryStep, decompose_visits,
                    group_recap};
use crate::stats::{Adjudication, adjudicate, is_net_seat, score_hubs, record_spt_at_turn_start, t2c_turn, turn_milestones};
use crate::tempo::{TempoTrack, tempo_sample, unit_tally};
use crate::traces::{TraceTrigger, TraceWindows, TracedDecision, dump_game_artifacts,
                    write_ply_traces};
use crate::ProgressMode;


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
    // Horizon-compression Stage 2 (EXP_ELO_120): siege-open events for the
    // aux_pressure target, using the arena's own siege definition
    // (polyfish::ai::siege — cross-checked against arena's SiegeTracker).
    // Sparse by construction (siege events are rare), so a global event
    // list scanned per-row in dataset.rs is cheap -- no checkpoint
    // structure needed the way SPT+5/territory+5 need one.
    let mut siege_active: std::collections::HashSet<(i32, i32)> = std::collections::HashSet::new();
    let mut siege_opens: Vec<(i32, PlayerId)> = Vec::new();
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
    let mut turn_dump_file = open_game_jsonl(dump_turn_states, "dump-turn-states", game_idx);
    let mut last_dump_key: Option<(i32, PlayerId)> = None;

    // --dump-macro-policy: one JSONL file per game, one record per macro
    // root decision (Stage 3b first step — see the Stage 4 dump below for
    // the write, same once-per-(turn,pov) dedup as turn_dump_file).
    let mut macro_policy_file = open_game_jsonl(dump_macro_policy, "dump-macro-policy", game_idx);
    let mut last_macro_policy_key: Option<(i32, PlayerId)> = None;
    // Separate from `last_macro_policy_key`: that tracker only advances
    // inside the `--dump-macro-policy` branch, which is `None` (off) during
    // real training runs -- reusing it here would silently disable this
    // dedup whenever the diagnostic dump isn't also requested.
    let mut last_macro_ballot_key: Option<(i32, PlayerId)> = None;
    // Horizon-compression Stage 1a (EXP_ELO_120): same shape, gates
    // ceiling_for_goal to once per (turn, pov).
    let mut last_eco_ceiling_key: Option<(i32, PlayerId)> = None;

    // --dump-city-rewards: one JSONL file per game, one record per city
    // level-up reward choice — (turn, player, city level pre-choice, tribe
    // stars at time of choice, reward type chosen). Reward moves are always
    // forced (generate_reward_moves preempts everything else when a choice
    // is pending — moves/mod.rs), so this is a clean, uncontested read of
    // what the policy actually wants at each level, no Step competition.
    let mut city_reward_file = open_game_jsonl(dump_city_rewards, "dump-city-rewards", game_idx);

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
    let mut star_spend_file = open_game_jsonl(dump_star_spend, "dump-star-spend", game_idx);
    // --dump-reward-choices: one JSONL record per city-reward choice ply with
    // the full search trace of the (modal) candidate pair — per-candidate
    // post-search Q, visits, prior, edge reward — for Q-gap sizing of the
    // reward-choice pricing terms. Not combinable with --dump-failed-dir
    // (that path consumes the trace first).
    let mut reward_choice_file = open_game_jsonl(dump_reward_choices, "dump-reward-choices", game_idx);

    // --dump-level-completion: one JSONL record per executed Harvest/Build
    // with owning-city level/progress and stars before/after.
    let mut level_completion_file = open_game_jsonl(dump_level_completion, "dump-level-completion", game_idx);

    // --dump-pop-spend-choices: sampled early-economy ply traces for Q-gap
    // sizing of the completion-discipline and body-count terms.
    let mut pop_spend_file = open_game_jsonl(dump_pop_spend_choices, "dump-pop-spend-choices", game_idx);
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
    // Trace-window state for every trigger kind (see traces::TraceWindows).
    let mut windows = TraceWindows::new();
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

        let (trigger_info, harvest_capture, pursuit_capture, wander_capture) =
            windows.for_ply(
                &game.state, pov, trace_villages, trace_all, trace_trigger,
                trace_counter, trace_max, &open_villages,
            );

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

        write_choice_dumps(
            &game.state, pov, current_agent, reward_choice_ply, pop_spend_ply,
            &mut reward_choice_file, &mut pop_spend_file, &mut pop_spend_dumped,
            &mut last_pop_spend_turn,
        );

        write_ply_traces(
            &game.state, pov, current_agent, &open_villages, trace_counter,
            trace_max, iteration, game_idx, move_count, trigger_info,
            harvest_capture, pursuit_capture, wander_capture,
            &mut windows.traces_this_game, &mut windows.last_trace_turn,
        );

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
            // EXP_ELO_120 / horizon-compression Stage 1b: territory tile
            // count, summed over cities currently held. Monotone "reached"
            // by construction — a captured-then-lost city just stops
            // contributing, it isn't penalized.
            let territory_of = |id: PlayerId| {
                game.state
                    .tribes
                    .get(&id)
                    .map(|t| {
                        t.cities
                            .iter()
                            .map(|c| {
                                polyfish::rules::economy::territory_tiles(&game.state, c).count()
                                    as i32
                            })
                            .sum()
                    })
                    .unwrap_or(0)
            };
            // EXP_ELO_120 / horizon-compression Stage 3: army value
            // differential, already [0,1]-clamped -- simplest of the three
            // heads, no new normalization decision needed.
            let army_of = |id: PlayerId| {
                polyfish::ai::evaluator::army::evaluate_army(&game.state, id)
            };
            // EXP_ELO_120 / horizon-compression Stage 1a: eco_plan's
            // Balanced-goal ceiling from this state, gated to once per
            // (turn, pov) since it's stable all turn and each call costs
            // ~9-10ms (EXP_ELO_086). Own normalization per field -- spt,
            // pop, giants, and monuments_used are on different scales, none
            // of them SPT's own /20.0.
            let eco_ceiling = eco_ceiling_for_history_step(
                (game.state.settings.turn, pov),
                &mut last_eco_ceiling_key,
                game.state.tribes.get(&pov).map(|t| {
                    t.cities.iter().map(|c| c.idx).collect::<Vec<i32>>()
                }).filter(|cities| !cities.is_empty()).and_then(|cities| {
                    polyfish::rules::eco_plan::ceiling_for_goal(
                        &game.state,
                        pov,
                        &cities,
                        polyfish::rules::eco_plan::Goal::Balanced,
                    )
                }).map(|plan| {
                    [
                        plan.spt as f32 / 20.0,
                        plan.pop as f32 / 100.0,
                        plan.giants as f32 / 10.0,
                        plan.monuments_used() as f32 / 5.0,
                    ]
                }),
            );
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
                eco_ceiling,
                enemy_units: enemy_unit_grid(&game.state, pov, features::MAP_SIZE * features::MAP_SIZE),
                my_spt: spt_of(pov),
                opp_spt: spt_of(opp_id),
                my_territory: territory_of(pov),
                opp_territory: territory_of(opp_id),
                my_army: army_of(pov),
                opp_army: army_of(opp_id),
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
            let siege_t = polyfish::ai::siege::scan_siege_transitions(&game.state, &mut siege_active);
            for (owner, _city) in siege_t.opened {
                siege_opens.push((game.state.settings.turn, owner));
            }
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
            write_spend_dumps(
                &game.state, pov, m_type, m.as_ref(), game_idx,
                star_spend_pre, level_completion_pre,
                &mut star_spend_file, &mut level_completion_file,
            );

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

    let Adjudication { scores, final_potentials, winner_id, winner_score,
                       is_decisive, alive_tribes: _ } =
        adjudicate(&game.state, shape_w_label, pursuit_w_label);
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

    let (hub_levels, first_hub_rank) =
        score_hubs(&game.state, &built_hubs, &first_hub_sites);

    let mut final_cities = HashMap::new();
    let mut total_cities = 0;
    for (id, t) in &game.state.tribes {
        final_cities.insert(*id, t.cities.len() as i32);
        total_cities += t.cities.len() as i32;
    }

    let (final_owner, final_spt, final_tech, final_territory, final_army) = final_ground_truth(&game.state);

    let recap = ModReplay {
        game_state: initial_state,
        turns: group_recap(flat_recap),
    };
    dump_game_artifacts(
        dump_games_dir, dump_failed_dir, iteration, game_idx, seed, &tribes,
        backend1, backend2, max_turns, &scores, &recap, &decision_log,
        &village_capture_turns, &ruin_capture_turns,
    );

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
        final_territory,
        final_army,
        siege_opens,
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
