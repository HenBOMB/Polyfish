//! Root-candidate generation directly from one NN forward pass over the
//! cheap legal-move list, replacing `rank_plies`'s expensive per-candidate
//! simulate/undo pipeline at its two real call sites: the real per-ply
//! trajectory (`rank_view_net_or_cpu`, consumed by `MacroMctsAgent::
//! select_move`) and macro-mcts's own root-candidate-turn rollouts
//! (`execute_turn_net_greedy`, consumed by `expand()`).
//!
//! Mirrors `gumbel_mcts::root::build_fresh_root`'s already-shipping pattern
//! (cheap legal-move enumeration -> one net call -> blend in a heuristic
//! prior -> rank) rather than reinventing it: EXP_ELO_131's first cut
//! (`net_rank_plies`, now removed from `macro_exec.rs`) used an ad-hoc
//! uncentered-log-prior formula and measured strictly worse than the
//! already-shipped `--macro-rollout-lambda 0.0` lever on both throughput and
//! quality (see `hypothesis_driven_improvements.md`'s "EXP_ELO_131 Phase 1"
//! section). This module is a plumbing rework, not a quality fix.

use crate::ai::eco_plan_commit::EcoPlanCommit;
use crate::ai::eval_server::Evaluator;
use crate::ai::features::RawFeatures;
use crate::ai::network::RawPolicyOutput;
use crate::ai::oracle_macro::{
    LaneState, MacroGoal, compute_goal_aux, observe_lane_state, tech_discipline_active,
};
use crate::ai::ply_ranker::PlyRanker;
use crate::ai::scoring;
use crate::ai::search::goal_aux::GoalAux;
use crate::ai::search::macro_exec::{self, MAX_EXEC_PLIES, TurnCounters, gate_ok};
use crate::ai::search::policy_composer::{blend_heuristic_into_logits, compute_move_log_probs_raw};
use crate::ai::search::unit_goals::UnitGoalStore;
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::states::PlayerId;
use crate::types::MoveType;
use std::sync::Arc;

/// Where `net_rank_root_candidates` gets its forward pass from. `MainNet`
/// exists for Part B's B1 experiment (compare the dedicated `PlyRanker`
/// against the existing production policy heads) -- not used by anything in
/// this module's own Part-A callers.
pub enum RootRankerSource<'a> {
    Ply(&'a PlyRanker),
    MainNet(&'a Evaluator),
}

impl RootRankerSource<'_> {
    fn forward(&self, feats: RawFeatures) -> Option<Arc<RawPolicyOutput>> {
        match self {
            RootRankerSource::Ply(ranker) => ranker.forward_raw(&feats).ok().map(Arc::new),
            RootRankerSource::MainNet(evaluator) => {
                evaluator.evaluate(vec![feats]).into_iter().next().map(|r| r.2)
            }
        }
    }
}

/// Steering knob: weight on the cheap `score_move` heuristic blended into
/// the net's own logits before ranking (see `blend_heuristic_into_logits`'s
/// `p' = (1-w)*p_net + w*p_heur`). Same starting point as Gumbel's own
/// `HEURISTIC_PRIOR_W0`. Unlike Gumbel, no iteration-based decay yet -- that
/// schedule only means something inside a training loop with an `iteration`
/// counter, and there is no retraining loop yet for whichever source Part B
/// ends up validating.
const NET_ROOT_HEURISTIC_BLEND_W_DEFAULT: f32 = 0.5;

fn net_root_heuristic_blend_w() -> f32 {
    static W: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("POLYFISH_NET_ROOT_HEURISTIC_W")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(NET_ROOT_HEURISTIC_BLEND_W_DEFAULT)
    })
}

/// EXP_ELO_133 (Part B, B1): which net feeds the real-per-ply root-candidate
/// builder. `POLYFISH_NET_ROOT_SOURCE=main_net` switches `rank_view_net_or_cpu`
/// from the default dedicated `PlyRanker` to the existing production
/// `Evaluator` policy heads -- zero new training, since `model.safetensors`
/// is already trained. Unrelated to `execute_turn_net_greedy`, which (Sep 7
/// 2026) unconditionally uses the main net for macro-mcts's own rollouts
/// regardless of this setting. Any other value of this env var (including
/// unset) keeps A6's `ply_ranker`-or-CPU-fallback behavior for the real-ply
/// path only.
fn net_root_source_is_main_net() -> bool {
    static MAIN_NET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MAIN_NET.get_or_init(|| std::env::var("POLYFISH_NET_ROOT_SOURCE").as_deref() == Ok("main_net"))
}

/// Diagnostic: how many times this module's builder ran and how many
/// candidates it scored in total -- the net-path counterpart to
/// `RANK_PLIES_CALLS`/`RANK_PLIES_CANDIDATES`, necessary since the net path
/// bypasses `rank_plies` for real plies and that older pair alone would
/// silently under-count real per-ply decisions.
pub static NET_ROOT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static NET_ROOT_CANDIDATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Build root candidates directly from one NN forward pass over the cheap
/// legal-move list: enumerate + gate (cheap) -> one net call -> log-domain
/// composition -> blend in the cheap `score_move` heuristic -> rank.
///
/// `game` is read-only (`&Game`, not `&mut`) -- unlike `rank_plies`, no
/// simulate/undo happens here at all. Returns `None` (never panics) on any
/// feature/forward-pass/shape failure so callers can fall back to the CPU
/// `rank_plies` path -- a ranker hiccup must never crash a real turn.
#[allow(clippy::too_many_arguments)]
pub fn net_rank_root_candidates(
    game: &Game,
    player: PlayerId,
    goal: &MacroGoal,
    aux: &GoalAux,
    star_gate: bool,
    source: RootRankerSource,
    unit_goals: Option<&UnitGoalStore>,
    eco_plan: Option<&EcoPlanCommit>,
) -> Option<Vec<(f32, Box<dyn Move>)>> {
    NET_ROOT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut moves = game.legal_moves();
    moves.retain(|m| gate_ok(&game.state, m.as_ref(), star_gate, Some(goal.stance), Some(aux)));
    let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
    if has_other {
        moves.retain(|m| m.move_type() != MoveType::EndTurn);
    }
    if moves.is_empty() {
        return Some(vec![(0.0, Box::new(EndTurnMove) as Box<dyn Move>)]);
    }
    NET_ROOT_CANDIDATES.fetch_add(moves.len() as u64, std::sync::atomic::Ordering::Relaxed);

    let feats =
        crate::ai::features::state_to_cpu_features_goal(&game.state, player, None, Some(goal)).ok()?;
    let raw = source.forward(feats)?;
    let map_size = game.state.settings.size as usize;
    let mut logits = compute_move_log_probs_raw(&raw, &moves, map_size);
    if logits.len() != moves.len() {
        return None;
    }
    let heur_scores: Vec<f32> = moves
        .iter()
        .map(|m| scoring::score_move_with_unit_goals(game, m.as_ref(), unit_goals, eco_plan))
        .collect();
    blend_heuristic_into_logits(&mut logits, &heur_scores, net_root_heuristic_blend_w());

    let mut scored: Vec<(f32, Box<dyn Move>)> =
        moves.into_iter().zip(logits).map(|(m, s)| (s, m)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    // Not `revive_endturn_if_worse_than_floor` -- its -700 floor is
    // calibrated to score_move+λΔφ's scale (EXP_ELO_077/082/102), a
    // different quantity from this blended logit. Lone-doomed-unit revival
    // is type-based, not score-scale-dependent, so it applies unchanged.
    Some(macro_exec::revive_endturn_for_lone_doomed_unit(
        scored,
        has_other,
        true,
        &game.state,
    ))
}

/// Real-per-ply-trajectory candidate builder: net-direct when a `PlyRanker`
/// is loaded (or, under EXP_ELO_133's `POLYFISH_NET_ROOT_SOURCE=main_net`,
/// via the existing production policy heads instead), verbatim `rank_plies`
/// otherwise. The returned `bool` is `true` iff the candidates came from the
/// net path -- threaded into `micro_search_pick` so its own second-forward-
/// pass widening logic knows it has nothing left to add (see
/// `MicroParams::net_prior_w`'s doc comment).
#[allow(clippy::too_many_arguments)]
pub fn rank_view_net_or_cpu(
    view: &mut Game,
    pov: PlayerId,
    goal: &MacroGoal,
    lane_state: &mut LaneState,
    counters: &mut TurnCounters,
    lambda: f32,
    unit_goals: Option<&UnitGoalStore>,
    eco_plan: Option<&EcoPlanCommit>,
    evaluator: &Evaluator,
) -> (Vec<(f32, Box<dyn Move>)>, bool) {
    observe_lane_state(&view.state, pov, lane_state);
    let aux = compute_goal_aux(
        &view.state,
        pov,
        goal,
        counters.techs_bought,
        counters.tier3_bought,
        Some(lane_state),
    );
    let gate = tech_discipline_active(&view.state, pov, goal);

    let source = if net_root_source_is_main_net() {
        Some(RootRankerSource::MainNet(evaluator))
    } else {
        crate::ai::ply_ranker::ply_ranker().map(RootRankerSource::Ply)
    };
    if let Some(source) = source {
        if let Some(v) =
            net_rank_root_candidates(view, pov, goal, &aux, gate, source, unit_goals, eco_plan)
        {
            return (v, true);
        }
    }
    (
        macro_exec::rank_plies(view, pov, goal, &aux, gate, lambda, unit_goals, eco_plan),
        false,
    )
}

/// Greedy net-driven playout of one macro-mcts root-candidate turn -- one
/// main-net forward pass per ply (`net_rank_root_candidates`), no search, no
/// `rank_plies` in any form (not even as a failure fallback: a feature/
/// forward-pass hiccup stops the rollout where it is, the same "anomaly
/// leaves the state where it stopped" convention `execute_turn` itself
/// already uses, rather than reaching for the CPU path). Verdi, Sep 7 2026:
/// rank_plies is gone from macro-mcts's own root-candidate rollouts,
/// unconditionally -- no env-var, no A/B toggle.
pub fn execute_turn_net_greedy(
    game: &mut Game,
    player: PlayerId,
    goal: &MacroGoal,
    lane_state: &mut LaneState,
    counters: &mut TurnCounters,
    evaluator: &Evaluator,
) -> bool {
    for _ in 0..MAX_EXEC_PLIES {
        if game.state.settings._game_over || game.state.settings.current_player_turn_id != player {
            return true;
        }
        observe_lane_state(&game.state, player, lane_state);
        let aux = compute_goal_aux(
            &game.state,
            player,
            goal,
            counters.techs_bought,
            counters.tier3_bought,
            Some(lane_state),
        );
        let gate = tech_discipline_active(&game.state, player, goal);
        // Rollouts never see the real trajectory's UnitGoalStore/EcoPlanCommit
        // -- same convention `execute_turn_recorded` already uses.
        let Some(ranked) = net_rank_root_candidates(
            game,
            player,
            goal,
            &aux,
            gate,
            RootRankerSource::MainNet(evaluator),
            None,
            None,
        ) else {
            return false;
        };
        let Some((_, best)) = ranked.into_iter().next() else {
            break;
        };
        if best.move_type() == MoveType::EndTurn {
            break;
        }
        if game.simulate_move(best.as_ref()).is_none() {
            return false;
        }
        counters.count(best.as_ref());
    }
    let _ = game.simulate_single_end_turn();
    true
}
