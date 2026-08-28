//! Throwaway diagnostic: reconstruct macro-mcts's TURN-LEVEL root search for
//! XinXi (player 1) at turn 4 of the v2 showcase replay (seed 1787500020),
//! and dump the real k=6 candidate ballot + post-search visits/root_q/
//! max_depth, plus per-city risk numbers at every turn boundary 0..=4.
//! Read-only on search code.

use polyfish::ai::eval_backend::{self, PlayerBackend};
use polyfish::ai::eval_server::EvalServerConfig;
use polyfish::ai::macro_agent::{
    enumerate_candidates_with_belief, CandidateClass, MacroLeaf, MacroParams,
};
use polyfish::ai::macro_exec::TurnCounters;
use polyfish::ai::macro_mcts::{derive_counters, MacroMctsSearch};
use polyfish::ai::network::PolyZeroNet;
use polyfish::ai::oracle_macro::{
    commit_macro_goal, expand_target_valid, observe_lane_state, select_lane, MacroGoal, OrderKind,
    StanceCommit,
};
use polyfish::ai::search::lane::LaneState;
use polyfish::game::Game;
use polyfish::replayer::{replay_game, ModReplay};
use polyfish::states::GameState;
use polyfish::types::TechnologyType;
use std::sync::Arc;

const REPLAY: &str =
    "replays/eval_seeds_showcase_v2/game_iter1_game0_seed1787500020.replay.json";
const POV: i32 = 1;
const TARGET_TURN: i32 = 4;

/// Deterministic replay of the recorded commands up to (but not including)
/// player 1's `turn` block. Truncating the turn list and reusing `replay_game`
/// verbatim keeps the reward/capture hint plumbing byte-identical.
fn state_at_p1_turn_start(full: &ModReplay, turn: i32) -> Game {
    let mut mr = full.clone();
    mr.turns.retain(|t| t.turn < turn);
    let mut g = Game::new();
    replay_game(&mut g, &mut mr).expect("deterministic replay failed");
    g
}

/// `TurnCounters` as the live agent would hold them: it counts only its OWN
/// executed Research moves (`TurnCounters::count`), so count them off the
/// recorded command stream rather than inferring from tech state.
fn counters_before(full: &ModReplay, turn: i32) -> TurnCounters {
    let mut c = TurnCounters::default();
    for t in &full.turns {
        if t.turn >= turn {
            break;
        }
        for p in &t.players {
            if p.player_id != POV {
                continue;
            }
            for cmd in &p.commands {
                if cmd.get("moveType").and_then(|v| v.as_i64()) != Some(7) {
                    continue;
                }
                c.techs_bought += 1;
                if let Some(tv) = cmd.get("type") {
                    if let Ok(tech) = serde_json::from_value::<TechnologyType>(tv.clone()) {
                        if polyfish::settings::technology::get_technology_setting(tech).tier
                            == Some(3)
                        {
                            c.tier3_bought += 1;
                        }
                    }
                }
            }
        }
    }
    c
}

/// Mirror of macro_mcts.rs's private `fog_order_dead` (continuation stripping).
fn fog_order_dead(state: &GameState, t: i32, pov: i32) -> bool {
    let Some(tile) = state.tiles.get(&t) else {
        return true;
    };
    if !tile.explorers.contains(&pov) {
        return false;
    }
    if tile.owner == pov && polyfish::functions::get_city_at(state, t).is_some() {
        return false;
    }
    !expand_target_valid(state, t, pov)
}

/// The ballot `MacroMctsAgent::select_move` would hand to the root search:
/// `enumerate_candidates_with_belief` (truncated to k) then the deduped,
/// dead-Expand-stripped continuation entries appended AFTER the truncation.
fn build_ballot(
    state: &GameState,
    base: &MacroGoal,
    counters: TurnCounters,
    k: usize,
    recent: &[MacroGoal],
) -> Vec<(MacroGoal, CandidateClass)> {
    let mut tagged =
        enumerate_candidates_with_belief(state, POV, base.clone(), counters, k, None);
    for g in recent.iter().rev() {
        let mut cand = g.clone();
        cand.orders
            .retain(|(kind, t)| *kind != OrderKind::Expand || !fog_order_dead(state, *t, POV));
        cand.orders.sort();
        if !tagged.iter().any(|(x, _)| *x == cand) {
            tagged.push((cand, CandidateClass::Continuation));
        }
    }
    tagged
}

fn fmt_goal(g: &MacroGoal) -> String {
    let orders: Vec<String> = g
        .orders
        .iter()
        .map(|(k, t)| format!("{k:?}@{t}"))
        .collect();
    let save = match &g.save_target {
        Some(s) => format!(" save={:?}/{:?} cost={}", s.tech, s.structure, s.cost),
        None => String::new(),
    };
    format!("stance={:?} orders=[{}]{}", g.stance, orders.join(", "), save)
}

fn main() -> anyhow::Result<()> {
    let raw = std::fs::read_to_string(REPLAY)?;
    let full: ModReplay = serde_json::from_str(&raw)?;

    // ---- Pass 1 (no network): thread the goal-setter's own memory through
    // every player-1 turn boundary 0..=5 and audit every base goal for Defend.
    let mut sc = StanceCommit::default();
    let mut lane = LaneState::default();
    let mut bases: Vec<MacroGoal> = Vec::new();
    let mut t_target: Option<(Game, TurnCounters, MacroGoal, LaneState)> = None;

    println!("=== PASS 1: deterministic base goals, player {POV}, turns 0..={TARGET_TURN} ===");
    for turn in 0..=TARGET_TURN {
        let game = state_at_p1_turn_start(&full, turn);
        assert_eq!(
            game.state.settings.turn, turn,
            "replay landed on the wrong turn"
        );
        assert_eq!(game.state.settings.current_player_turn_id, POV);
        let view0 = game.clone_for_mcts(POV);
        let counters = counters_before(&full, turn);
        let base = commit_macro_goal(&view0.state, POV, &mut sc, counters.tier3_bought);
        observe_lane_state(&view0.state, POV, &mut lane);
        select_lane(&view0.state, POV, &mut lane, None);

        let has_defend = base.orders.iter().any(|(k, _)| *k == OrderKind::Defend);
        println!(
            "turn {turn}: {}  | counters={:?} (derive_counters={:?}) lane={:?} | DEFEND_ORDER={}",
            fmt_goal(&base),
            counters,
            derive_counters(&view0.state, POV),
            lane.lane,
            has_defend
        );
        for r in polyfish::ai::combat::city_risks(&view0.state, POV) {
            println!(
                "    city {:>3} risk={:.4} at_risk={} needs_order={} sieged={} open={} attackers={:?}",
                r.city, r.risk, r.at_risk, r.needs_order(), r.sieged, r.open, r.attackers
            );
        }
        bases.push(base.clone());
        if turn == TARGET_TURN {
            t_target = Some((game, counters, base, lane.clone()));
        }
    }

    let (game5, counters5, base5, lane5) = t_target.expect("target turn missing");
    let view5 = game5.clone_for_mcts(POV);

    // recent_goals holds the last RECENT_GOALS=3 PICKED directives (the 3
    // turns before TARGET_TURN). The picks are search-dependent and
    // unrecoverable, so run both the empty-history and the pick==base
    // (tie-breaks-toward-base) case.
    let recent_start = (TARGET_TURN - 3).max(0) as usize;
    let recent_base: Vec<MacroGoal> = bases[recent_start..TARGET_TURN as usize].to_vec();
    let variants: Vec<(&str, Vec<MacroGoal>)> = vec![
        ("A: continuations = none", Vec::new()),
        ("B: continuations = bases of turns 2,3,4", recent_base),
    ];

    // ---- Evaluator, mirroring self_play's construction exactly.
    let device = eval_backend::select_device()?;
    let kind = eval_backend::resolve_eval_backend_kind("")?;
    let shards = eval_backend::resolve_eval_servers(kind, 0)?;
    println!("\n[eval] backend={kind:?} shards={shards} device={device:?}");
    let vs = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &["model.safetensors"],
            candle_core::DType::F32,
            &device,
        )?
    };
    let net = Arc::new(PolyZeroNet::new(vs)?);
    let config = EvalServerConfig {
        max_batch: 256,
        coalesce_timeout: std::time::Duration::from_micros(1000),
        cache_capacity: eval_backend::split_cache_capacity(524288, shards),
        pipeline_workers: 2,
    };
    let (p1_servers, p2_servers, eval1, eval2) = eval_backend::build_two_player_evaluators(
        kind,
        shards,
        config,
        PlayerBackend {
            model_path: "model.safetensors",
            candle_net: &net,
        },
        None,
    );

    let params = MacroParams {
        k: 8,
        horizon: 2,
        leaf: MacroLeaf::NetAsym,
        lambda: 1.0,
        rollout_lambda: 1.0,
        sims: std::env::var("SIMS").ok().and_then(|s| s.parse().ok()).unwrap_or(64),
        belief_mode: polyfish::ai::macro_agent::BeliefMode::Off,
        shape_w: 0.0,
        root_prior_w: 0.0,
    };
    println!("[params] {params:?}");

    let repeats: usize = std::env::var("REPEATS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    for (label, recent) in &variants {
        let tagged = build_ballot(&view5.state, &base5, counters5, params.k, recent);
        let cands: Vec<MacroGoal> = tagged.iter().map(|(g, _)| g.clone()).collect();
        println!("\n================ TURN {TARGET_TURN} ROOT BALLOT — {label} ================");
        println!("ballot size = {} (k={})", cands.len(), params.k);
        for (i, (g, c)) in tagged.iter().enumerate() {
            let defend = g.orders.iter().any(|(k, _)| *k == OrderKind::Defend);
            println!(
                "  [{i}] class={c:?}  {}{}",
                fmt_goal(g),
                if defend { "   <-- HAS DEFEND" } else { "" }
            );
        }
        println!("--- {repeats} repeated searches (identical reconstructed state) ---");
        for rep in 0..repeats {
            let t0 = std::time::Instant::now();
            let (pick, stats) = MacroMctsSearch::run(
                &view5,
                POV,
                cands.clone(),
                counters5,
                &lane5,
                &params,
                &eval1,
            );
            let total: f32 = stats.root_visits.iter().sum();
            let shares: Vec<String> = stats
                .root_visits
                .iter()
                .map(|v| format!("{:.0}({:.1}%)", v, 100.0 * v / total.max(1.0)))
                .collect();
            println!(
                "rep{rep}: pick=#{pick}  visits=[{}]  root_q={:?} root_q_spread={:?} nodes={} max_depth={} max_share={:.3}  [{:.1}s]",
                shares.join(" "),
                stats.root_q,
                stats.root_q_spread,
                stats.nodes,
                stats.max_depth,
                stats.root_visit_max_share,
                t0.elapsed().as_secs_f32()
            );
        }
    }

    drop(eval1);
    drop(eval2);
    for s in p1_servers {
        s.shutdown();
    }
    if let Some(servers) = p2_servers {
        for s in servers {
            s.shutdown();
        }
    }
    Ok(())
}
