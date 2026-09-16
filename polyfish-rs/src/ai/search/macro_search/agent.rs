use super::config::{BeliefMode, MacroLeaf, MacroParams};
use super::search::MacroMctsSearch;
use crate::ai::belief::BeliefState;
use crate::ai::eco_plan_commit::EcoPlanCommit;
use crate::ai::eval_server::Evaluator;
use crate::ai::macro_agent::CandidateClass;
use crate::ai::macro_exec::TurnCounters;
use crate::ai::oracle_macro::{
    LaneState, MacroGoal, OrderKind, Stance, StanceCommit, commit_macro_goal, compute_goal_aux,
    compute_macro_goal, observe_lane_state, pick_save_lane, tech_discipline_active,
};
use crate::ai::search::macro_mcts::MacroMctsStats;
use crate::game::Game;
use crate::moves::Move;
use crate::states::PlayerId;

pub(super) struct MacroMctsAgent<'a> {
    /// Leaf evaluator for macro search if network-based evaluation is enabled.
    evaluator: &'a crate::ai::eval_server::Evaluator,
    /// Macro search configuration parameters.
    params: MacroParams,
    /// Track committed stance directives for the turn.
    stance_commit: StanceCommit,
    /// Stores lane-level macro search state.
    lane_state: LaneState,
    /// Turn-level counters for the search episode.
    counters: TurnCounters,
    /// Committed eco plan for joint frontier/factory-mine matching.
    eco_plan_commit: crate::ai::eco_plan_commit::EcoPlanCommit,
    /// Cache key for plan tracking (usually (hash, player)).
    plan_key: Option<(i32, PlayerId)>,
    /// Turn-level macro-goal for current plan.
    turn_goal: Option<MacroGoal>,
    /// Number of consecutive turns where the plan diverged (recommitted).
    pub divergent_turns: u32,
    /// Number of planned turns by this agent instance.
    pub planned_turns: u32,
    /// Last completed macro search statistics.
    pub last_stats: MacroMctsStats,
    /// Latest input belief state (e.g. fog/hypothesis/conditioning).
    pub belief: Option<crate::ai::belief::BeliefState>,
    /// Number of turns since the player's capital was lost.
    pub mat_capital_turns: u32,
    /// Current count of player's material units.
    pub mat_units: u32,
    pub branch_revealed_tiles: u32,
    pub branch_resource_reveals: u32,
    pub branch_capital_refutations: u32,
    pub branch_capital_confirmations: u32,
    pub branch_village_confirmations: u32,
    pub branch_capital_materializations: u32,
    pub branch_village_materializations: u32,
    pub branch_unit_materializations: u32,
    pub branch_belief_candidate_nodes: u32,
    /// Number of belief-conditioned root candidate nodes in the current macro search branch.
    pub branch_belief_candidates: u32,
    /// Winning candidate class counts per planned turn, by CandidateClass index.
    pub class_picks: [u32; crate::ai::macro_agent::CANDIDATE_CLASSES],
    /// Number of times the same fog-target is immediately re-picked by belief on consecutive turns.
    pub belief_repicks: u32,
    /// Last chosen belief target, if any.
    last_belief_target: Option<i32>,
    /// Recently picked macro-goals, re-offered as Continuation candidates each plan.
    recent_goals: std::collections::VecDeque<MacroGoal>,
    /// Number of fog orders cut from the live plan mid-turn due to new vision.
    pub intra_strips: u32,
    /// Persistent per-unit macro goals for the planned real trajectory only.
    unit_goals: crate::ai::search::unit_goals::UnitGoalStore,
    /// Root micro-mcts tree carryover across plies within the same turn.
    micro_carry: Option<crate::ai::search::micro_mcts::MicroTreeCarry>,
    /// Backed-up value of the picked micro-mcts child at the root this ply, if computed.
    last_micro_root_q: Option<f32>,
    /// Post-search visit counts over root micro-mcts children for this ply.
    last_micro_visits: Vec<crate::ai::mcts_types::MoveVisit>,
    /// Raw main-net value-head output at this ply's root, if 'macro root own value' mode is enabled.
    last_root_own_value: Option<f32>,
}

impl<'a> MacroMctsAgent<'a> {
    // Constructor for the macro search agent.
    pub(super) fn new(
        evaluator: &'a crate::ai::eval_server::Evaluator,
        params: MacroParams,
    ) -> Self {
        Self {
            evaluator,
            params,
            stance_commit: StanceCommit::default(),
            lane_state: LaneState::default(),
            counters: TurnCounters::default(),
            eco_plan_commit: crate::ai::eco_plan_commit::EcoPlanCommit::default(),
            plan_key: None,
            turn_goal: None,
            divergent_turns: 0,
            planned_turns: 0,
            last_stats: MacroMctsStats::default(),
            belief: None,
            mat_capital_turns: 0,
            mat_units: 0,
            branch_revealed_tiles: 0,
            branch_resource_reveals: 0,
            branch_capital_refutations: 0,
            branch_capital_confirmations: 0,
            branch_village_confirmations: 0,
            branch_capital_materializations: 0,
            branch_village_materializations: 0,
            branch_unit_materializations: 0,
            branch_belief_candidate_nodes: 0,
            branch_belief_candidates: 0,
            class_picks: [0; crate::ai::macro_agent::CANDIDATE_CLASSES],
            belief_repicks: 0,
            last_belief_target: None,
            recent_goals: std::collections::VecDeque::new(),
            intra_strips: 0,
            unit_goals: crate::ai::search::unit_goals::UnitGoalStore::default(),
            micro_carry: None,
            last_micro_root_q: None,
            last_micro_visits: Vec::new(),
            last_root_own_value: None,
        }
    }

    // Set the belief state for the macro search agent.
    pub(super) fn set_belief(&mut self, belief: crate::ai::belief::BeliefState) {
        self.belief = Some(belief);
    }

    // Get the committed goal for the macro search agent.
    pub(super) fn committed_goal(&self) -> Option<&MacroGoal> {
        self.turn_goal.as_ref()
    }

    // Get the committed playstyle for the macro search agent.
    pub(super) fn committed_playstyle(&self) -> &LaneState {
        &self.lane_state
    }

    // Get the root value of the committed directive for the macro search agent.
    pub fn last_root_value(&self) -> Option<f32> {
        match self.params.leaf {
            MacroLeaf::Net | MacroLeaf::NetAsym | MacroLeaf::NetAsymPaint => self.last_stats.root_q,
            MacroLeaf::Heuristic | MacroLeaf::HeuristicV2 => None,
        }
    }

    /// `root_q` with no leaf gate — the tree's backed-up value of the winning
    /// root edge whatever scored the leaves. Telemetry only: `last_root_value`
    /// stays the TD-label path.
    pub fn last_root_q_raw(&self) -> Option<f32> {
        self.last_stats.root_q
    }

    /// This ply's micro-mcts root Q (`tree(V_net)`) -- see
    /// `last_micro_root_q`'s field doc. Fresh every ply (unlike
    /// `last_root_q_raw`, which only updates once per turn), and available
    /// under every macro leaf kind since it is computed independently of
    /// which evaluator scores the macro tree's own leaves.
    pub fn micro_root_q(&self) -> Option<f32> {
        self.last_micro_root_q
    }

    /// This ply's micro-mcts post-search visit distribution -- see
    /// `last_micro_visits`'s field doc. Empty exactly when micro-mcts
    /// didn't run this ply (mirrors `micro_root_q`'s `None` condition).
    pub fn last_micro_visits(&self) -> &[crate::ai::mcts_types::MoveVisit] {
        &self.last_micro_visits
    }

    /// This ply's RAW pre-search value-head output -- see
    /// `last_root_own_value`'s field doc. `None` unless
    /// `POLYFISH_MACRO_ROOT_OWN_VALUE=1`.
    pub fn last_root_own_value(&self) -> Option<f32> {
        self.last_root_own_value
    }

    /// Companion to `SearchAgent::clear_last_root_value` (Gumbel-only today):
    /// a forced-move ply (no search runs at all) must not let a stale value
    /// from the PREVIOUS ply leak through `game.rs`'s unconditional
    /// `last_root_own_value()` read.
    pub fn clear_last_root_own_value(&mut self) {
        self.last_root_own_value = None;
    }

    /// The current turn's root ballot: candidate directives and the tree's
    /// own post-search visit count per candidate, parallel arrays. Available
    /// under every leaf kind (unlike `last_root_value`) — the visit
    /// distribution is real search output regardless of what scored the
    /// leaves. Empty before the first search of the run.
    pub fn last_root_ballot(&self) -> (&[MacroGoal], &[f32]) {
        (
            &self.last_stats.root_candidates,
            &self.last_stats.root_visits,
        )
    }

    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        let pov = game.state.settings.current_player_turn_id;
        let key = (game.state.settings.turn, pov);
        if self.plan_key != Some(key) {
            // A new turn invalidates any micro-mcts subtree from the last
            // one — it was built off a different goal/lane/state entirely.
            self.micro_carry = None;
            let mut view0 = game.clone_for_mcts(pov);
            if self.params.tree_belief_materialization && self.belief.is_some() {
                macro_search::belief::BranchBelief::sanitize_fog_view(&mut view0.state, pov);
            }
            let use_world = matches!(
                self.params.belief_mode,
                BeliefMode::World | BeliefMode::Both
            );
            let use_cand = matches!(
                self.params.belief_mode,
                BeliefMode::Candidates | BeliefMode::Both
            );
            if self.params.tree_belief_materialization && !use_world {
                if let Some(belief) = &self.belief {
                    let _ = crate::ai::belief::materialize_branch_units(&mut view0, belief);
                }
            }
            if use_world {
                if let Some(b) = &self.belief {
                    let st = crate::ai::belief::materialize_into(&mut view0, b);
                    if st.capital {
                        self.mat_capital_turns += 1;
                    }
                    self.mat_units += st.ghost_units + st.residual_units;
                }
            }
            let base = commit_macro_goal(
                &view0.state,
                pov,
                &mut self.stance_commit,
                self.counters.tier3_bought,
            );
            // EXP_ELO_100: once per real turn, same gate as everything else
            // here — the Mine-lane scoring signal `rank_view` reads below
            // (real ply AND every simulated rollout that reuses this same
            // committed snapshot) must not go stale mid-turn or recompute
            // per candidate (`enumerate_empire` costs single-digit ms).
            self.eco_plan_commit.update(&view0.state, pov);
            // Tier 1, once per turn: score every lane and commit one. The
            // executor plies below only OBSERVE, so the lane stays the
            // turn's identity instead of drifting ply to ply. In-tree turns
            // inherit this lane rather than re-selecting (v1).
            crate::ai::oracle_macro::observe_lane_state(&view0.state, pov, &mut self.lane_state);
            crate::ai::oracle_macro::select_lane(&view0.state, pov, &mut self.lane_state, None);
            let mut tagged = crate::ai::macro_agent::enumerate_candidates_with_belief(
                &view0.state,
                pov,
                base.clone(),
                self.counters,
                self.params.k,
                if use_cand { self.belief.as_ref() } else { None },
            );
            // EXP_ELO_038 (Verdi spec): the strategist's last picked
            // directives join the ballot — continuity through informed
            // selection, never injection. Orders the evidence has since
            // killed are stripped before the offer; duplicates of candidates
            // already on the ballot vanish.
            for g in self.recent_goals.iter().rev() {
                let mut cand = g.clone();
                cand.orders.retain(|(kind, t)| {
                    *kind != crate::ai::oracle_macro::OrderKind::Expand
                        || !fog_order_dead(&view0.state, *t, pov)
                });
                cand.orders.sort();
                if !tagged.iter().any(|(x, _)| *x == cand) {
                    tagged.push((cand, CandidateClass::Continuation));
                }
            }
            let candidates: Vec<MacroGoal> = tagged.iter().map(|(g, _)| g.clone()).collect();
            let (pick, stats) = if self.params.tree_belief_materialization {
                self.belief.as_ref().map(|belief| {
                    MacroMctsSearch::run_with_belief_world(
                        &view0,
                        pov,
                        candidates.clone(),
                        self.counters,
                        &self.lane_state,
                        &self.params,
                        self.evaluator,
                        belief.clone(),
                    )
                })
            } else {
                None
            }
            .unwrap_or_else(|| {
                MacroMctsSearch::run(
                    &view0,
                    pov,
                    candidates.clone(),
                    self.counters,
                    &self.lane_state,
                    &self.params,
                    self.evaluator,
                )
            });
            self.last_stats = stats;
            self.branch_revealed_tiles += self.last_stats.branch_revealed_tiles;
            self.branch_resource_reveals += self.last_stats.branch_resource_reveals;
            self.branch_capital_refutations += self.last_stats.branch_capital_refutations;
            self.branch_capital_confirmations += self.last_stats.branch_capital_confirmations;
            self.branch_village_confirmations += self.last_stats.branch_village_confirmations;
            self.branch_capital_materializations += self.last_stats.branch_capital_materializations;
            self.branch_village_materializations += self.last_stats.branch_village_materializations;
            self.branch_unit_materializations += self.last_stats.branch_unit_materializations;
            self.branch_belief_candidate_nodes += self.last_stats.branch_belief_candidate_nodes;
            self.branch_belief_candidates += self.last_stats.branch_belief_candidates;
            self.planned_turns += 1;
            if pick != 0 {
                self.divergent_turns += 1;
            }
            if net_candidate_debug() {
                if let Some(idx) = self.last_stats.net_candidate_index {
                    let visits = self.last_stats.root_visits.get(idx).copied().unwrap_or(0.0);
                    eprintln!(
                        "NET_CANDIDATE_DEBUG turn={} pov={pov} offered=1 picked={} visits={visits}",
                        view0.state.settings.turn,
                        (pick == idx) as u8
                    );
                }
            }
            let picked_class = tagged.get(pick).map(|(_, c)| *c);
            if let Some(c) = picked_class {
                self.class_picks[c as usize] += 1;
            }
            // Plan-stability: the same belief fog-target winning consecutive
            // planned turns means units aren't being yanked mid-approach.
            let belief_target = match picked_class {
                Some(CandidateClass::ClaimSafe) | Some(CandidateClass::Contest) => {
                    tagged.get(pick).and_then(|(g, _)| {
                        g.orders
                            .iter()
                            .find(|o| !base.orders.contains(o))
                            .map(|(_, t)| *t)
                    })
                }
                _ => None,
            };
            if belief_target.is_some() && belief_target == self.last_belief_target {
                self.belief_repicks += 1;
            }
            self.last_belief_target = belief_target;
            // EXP_ELO_165: `pick` can index PAST `candidates`' own length
            // when it's the net-synthesized candidate (appended inside
            // `run_with`, never part of this locally-built `candidates`
            // vec) -- `self.last_stats.root_candidates` is the tree's own
            // final ballot (`run_with` copies `root.candidates` into it
            // verbatim, net candidate included), so it's always the correct
            // source of truth for `pick`, unlike the local `candidates` var.
            self.turn_goal = self.last_stats.root_candidates.get(pick).cloned();
            // EXP_ELO_119: a standalone probe's own `commit_macro_goal`
            // reconstruction (fresh StanceCommit/LaneState) does not
            // reliably reproduce this turn's REAL committed goal -- ballot
            // candidates 5-8 alone (EXP_ELO_038 continuations from
            // `self.recent_goals`) have no fresh-recompute equivalent. This
            // is the only ground truth for "what did the real search
            // actually commit to," one line, env-gated, no-op when unset.
            if turn_goal_debug() {
                eprintln!(
                    "TURN_GOAL_DEBUG turn={} pov={pov} pick={pick} goal={:?}",
                    view0.state.settings.turn, self.turn_goal
                );
            }
            paint_probe(
                self.evaluator,
                &view0.state,
                pov,
                &base,
                self.turn_goal.as_ref(),
                pick != 0,
                &self.last_stats,
            );
            if let Some(g) = self.turn_goal.as_ref() {
                tier_probe(
                    &view0,
                    pov,
                    &self.lane_state,
                    self.counters,
                    self.params.lambda,
                    &base,
                    g,
                    pick != 0,
                );
            }
            // EXP_ELO_038: remember what we chose — tomorrow's ballot
            // includes it.
            if let Some(g) = &self.turn_goal {
                self.recent_goals.push_back(g.clone());
                while self.recent_goals.len() > RECENT_GOALS {
                    self.recent_goals.pop_front();
                }
            }
            self.plan_key = Some(key);
        }
        let mut view = game.clone_for_mcts(pov);
        // EXP_ELO_037 rule 1: per-ply belief consumption — this ply's fresh
        // view may have disconfirmed a fog order mid-turn (the directive was
        // the only thing not consuming per-move belief updates). Strip dead
        // fog orders from the LIVE goal now, not at the next plan.
        if let Some(g) = self.turn_goal.as_mut() {
            let before = g.orders.len();
            g.orders.retain(|(kind, t)| {
                *kind != crate::ai::oracle_macro::OrderKind::Expand
                    || !fog_order_dead(&view.state, *t, pov)
            });
            let stripped = before - g.orders.len();
            if stripped > 0 {
                self.intra_strips += stripped as u32;
            }
        }
        let goal = self.turn_goal.clone().unwrap_or_default();
        self.last_root_own_value = macro_root_own_value_enabled()
            .then(|| {
                crate::ai::features::state_to_cpu_features_goal(&view.state, pov, None, Some(&goal))
                    .ok()
                    .and_then(|f| {
                        self.evaluator
                            .evaluate(vec![f])
                            .into_iter()
                            .next()
                            .map(|r| r.0)
                    })
            })
            .flatten();
        let unit_status = crate::ai::search::unit_goals::reconcile_unit_goals(
            &view.state,
            pov,
            &goal,
            &mut self.unit_goals,
        );
        let (mut ranked, net_derived) = crate::ai::search::net_root::rank_view_net_or_cpu(
            &mut view,
            pov,
            &goal,
            &mut self.lane_state,
            &mut self.counters,
            self.params.lambda,
            Some(&self.unit_goals),
            Some(&self.eco_plan_commit),
            self.evaluator,
        );
        let mut pending_micro_carry: Option<(
            serde_json::Value,
            crate::ai::search::micro_mcts::MicroTreeCarry,
        )> = None;
        self.last_micro_root_q = None;
        self.last_micro_visits.clear();
        // Debug traceability (Verdi, Sep 7 2026): root-seed prior vs.
        // post-search visits/Q per candidate, threaded into the
        // POLYFISH_PLY_TRACE dump below. Empty whenever micro-mcts didn't
        // run (params off, or nothing to search).
        let mut micro_child_trace: Vec<crate::ai::search::micro_mcts::MicroChildTrace> = Vec::new();
        if let Some(mut micro_params) = crate::ai::search::micro_mcts::micro_mcts_params() {
            micro_params.c_puct = crate::ai::search::micro_mcts::turn_conditional_c_puct(
                game.state.settings.turn,
                micro_params.c_puct,
            );
            let star_gate =
                crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
            let aux = crate::ai::search::goal_aux::compute_goal_aux(
                &view.state,
                pov,
                &goal,
                self.counters.techs_bought,
                self.counters.tier3_bought,
                Some(&self.lane_state),
            );
            let (pick, next_carry, picked_q, child_trace, visit_dist) =
                crate::ai::search::micro_mcts::micro_search_pick(
                    &view,
                    pov,
                    &goal,
                    &ranked,
                    &aux,
                    star_gate,
                    self.evaluator,
                    &micro_params,
                    self.micro_carry.take(),
                    net_derived,
                );
            self.last_micro_root_q = picked_q;
            self.last_micro_visits = visit_dist;
            micro_child_trace = child_trace;
            if let Some(idx) = pick {
                let predicted_key = ranked[idx].1.serialize();
                ranked.swap(0, idx);
                if let Some(carry) = next_carry {
                    pending_micro_carry = Some((predicted_key, carry));
                }
            }
        }
        run_micro_probe(
            self.evaluator,
            &view,
            pov,
            &goal,
            &self.lane_state,
            self.counters,
            self.params.lambda,
            &self.unit_goals,
            Some(&self.eco_plan_commit),
            &ranked,
        );
        if let Some(path) = ply_trace_path() {
            let turn = game.state.settings.turn;
            let candidates: Vec<serde_json::Value> = ranked
                .iter()
                .map(|(score, mv)| {
                    serde_json::json!({
                        "score": score,
                        "move_type": format!("{:?}", mv.move_type()),
                        "move": mv.serialize(),
                    })
                })
                .collect();
            let unit_goals_trace: Vec<serde_json::Value> = view
                .state
                .tribes
                .get(&pov)
                .map(|t| {
                    t.units
                        .iter()
                        .map(|u| {
                            let g = self.unit_goals.active(u.id);
                            serde_json::json!({
                                "unit_id": u.id,
                                "unit_type": format!("{:?}", u.unit_type),
                                "coords": u.coords.idx,
                                "goal": g.map(|g| serde_json::json!({
                                    "kind": format!("{:?}", g.kind),
                                    "target": g.target,
                                })),
                                "status": unit_status.get(&u.id).map(|s| format!("{s:?}")),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let m = crate::ai::macro_agent::first_true_legal(game, ranked);
            self.micro_carry = pending_micro_carry
                .filter(|(key, _)| *key == m.serialize())
                .map(|(_, carry)| carry);
            dump_ply_decision(
                path,
                turn,
                pov,
                &goal,
                candidates,
                unit_goals_trace,
                m.as_ref(),
                &micro_child_trace,
            );
            self.counters.count(m.as_ref());
            return Some(m);
        }
        let m = crate::ai::macro_agent::first_true_legal(game, ranked);
        self.micro_carry = pending_micro_carry
            .filter(|(key, _)| *key == m.serialize())
            .map(|(_, carry)| carry);
        self.counters.count(m.as_ref());
        Some(m)
    }
}
