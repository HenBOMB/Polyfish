//! EXP_ELO_033: adversarial turn-level MCTS over macro directives (Stage 2 of
//! the macro-search redesign). Nodes are turn boundaries, edges are
//! `MacroGoal` directives executed by the deterministic executor, and — the
//! upgrade over the Stage-1 lookahead — the OPPONENT's turns are searched
//! adversarially instead of ghost-scripted. Two-player only; negamax backup
//! over the antisymmetric heuristic `evaluate_state`.

use crate::ai::macro_agent::{MacroLeaf, MacroParams, enumerate_candidates};
use crate::ai::macro_exec::{self, TurnCounters};
use crate::ai::oracle_macro::{LaneState, MacroGoal, StanceCommit, compute_macro_goal, commit_macro_goal};
use crate::game::Game;
use crate::moves::Move;
use crate::states::{GameState, PlayerId};

/// Single-game deep inspection (not a standing feature, not sampled): when
/// `POLYFISH_PLY_TRACE=<path>` is set, `MacroMctsAgent::select_move` appends
/// one JSONL row per REAL ply decision — the turn's committed goal, every
/// legal candidate move `rank_view` scored (already-computed, no extra
/// search cost), and which one was actually chosen. Unlike the training
/// harvest probes elsewhere in this file, this fires once per real ply only
/// (not once per internal rollout `expand()` call), so a whole game is a few
/// hundred rows — no sampling needed, no write-storm risk.
fn ply_trace_path() -> Option<&'static str> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var("POLYFISH_PLY_TRACE").ok())
        .as_deref()
}

fn dump_ply_decision(
    path: &str,
    turn: i32,
    player: PlayerId,
    goal: &MacroGoal,
    candidates: Vec<serde_json::Value>,
    unit_goals: Vec<serde_json::Value>,
    chosen: &dyn Move,
) {
    let row = serde_json::json!({
        "turn": turn,
        "player": player,
        "goal": {
            "stance": format!("{:?}", goal.stance),
            "orders": goal.orders.iter()
                .map(|(kind, t)| serde_json::json!([format!("{kind:?}"), t]))
                .collect::<Vec<_>>(),
        },
        "candidates": candidates,
        // Per-unit-goal design (Aug 2026): what each unit is trying to do
        // right now, as a queryable fact -- the actual deliverable this
        // whole design exists for.
        "unit_goals": unit_goals,
        "chosen": {
            "move_type": format!("{:?}", chosen.move_type()),
            "move": chosen.serialize(),
        },
    });
    if let Ok(s) = serde_json::to_string(&row) {
        use std::io::Write;
        if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(fh, "{s}");
        }
    }
}

/// Single-search deep inspection (not a standing feature): when
/// `POLYFISH_MACRO_ROLLOUT_TRACE=<path>` is set, `MacroMctsSearch::expand`
/// appends one JSONL row per simulated node it creates — `parent` plus the
/// turn/mover/directive that produced it, and the ROOT PLAYER's own city
/// ownership at that simulated point. Built to answer a concrete question
/// the aggregate root_q/visits can't: walking down one specific root
/// candidate's simulated rollout, does the adversarial opponent ever
/// actually take one of the root player's cities, or does the simulated
/// future never reproduce the danger a real opponent found? `parent` alone
/// is enough to reconstruct any node's full ancestry (and therefore which
/// root edge it descends from) by walking the chain back to node 0 in
/// post-processing — no need to track that redundantly here.
fn macro_rollout_trace_path() -> Option<&'static str> {
    static PATH: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var("POLYFISH_MACRO_ROLLOUT_TRACE").ok())
        .as_deref()
}

fn dump_rollout_node(
    path: &str,
    node_idx: usize,
    parent: usize,
    turn: i32,
    mover: PlayerId,
    goal: &MacroGoal,
    pov: PlayerId,
    state: &crate::states::GameState,
) {
    let pov_cities: Vec<i32> = state
        .tribes
        .get(&pov)
        .map(|t| t.cities.iter().map(|c| c.idx).collect())
        .unwrap_or_default();
    let row = serde_json::json!({
        "node_idx": node_idx,
        "parent": parent,
        "turn": turn,
        "mover": mover,
        "directive": {
            "stance": format!("{:?}", goal.stance),
            "orders": goal.orders.iter()
                .map(|(kind, t)| serde_json::json!([format!("{kind:?}"), t]))
                .collect::<Vec<_>>(),
        },
        "pov_cities": pov_cities,
    });
    if let Ok(s) = serde_json::to_string(&row) {
        use std::io::Write;
        if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(fh, "{s}");
        }
    }
}

/// Micro-mcts Phase 0 (throughput/cache-hit probe, not a standing feature):
/// how many extra synthetic single-leaf `eval_server` requests this process
/// has issued, and how many of the probe's own hypothetical continuations
/// failed to simulate. Always-on atomics, mirroring `RANK_PLIES_CALLS`
/// (macro_exec.rs) -- zero cost when the probe's env var is unset, since
/// `run_micro_probe` returns before either is touched.
pub static MICRO_PROBE_EVALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static MICRO_PROBE_SIM_FAILURES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

fn micro_probe_sims() -> Option<usize> {
    static SIMS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *SIMS.get_or_init(|| std::env::var("POLYFISH_MICRO_PROBE_SIMS").ok().and_then(|s| s.parse().ok()))
}

fn micro_probe_depth() -> usize {
    static DEPTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEPTH.get_or_init(|| {
        std::env::var("POLYFISH_MICRO_PROBE_DEPTH").ok().and_then(|s| s.parse().ok()).unwrap_or(1)
    })
}

/// Micro-mcts Phase 0: measure the throughput/cache-hit cost of network-leaf
/// evals a bounded within-turn search would generate, BEFORE building any
/// real search tree. When `POLYFISH_MICRO_PROBE_SIMS=<n>` is set, issues `n`
/// extra synthetic `eval_server` calls per REAL ply against up-to-`depth`-ply
/// hypothetical continuations of the already-committed turn.
///
/// Real-trajectory-only and strictly read-only w.r.t. everything but its own
/// disposable clone: never mutates `ranked`, `view`, `game`, or any of the
/// agent's persistent state, and never changes which move is actually
/// played. Composed `simulate_move`/undo across a turn is NOT safe (see
/// `cross_end_turn`'s documented panic, gumbel_mcts/rounds.rs) -- this walks
/// forward on a disposable clone instead and never calls undo, the same
/// idiom `MacroMctsSearch::expand` already uses for rollout nodes.
#[allow(clippy::too_many_arguments)]
fn run_micro_probe(
    evaluator: &crate::ai::eval_server::Evaluator,
    view: &Game,
    pov: PlayerId,
    goal: &MacroGoal,
    lane_state: &LaneState,
    counters: TurnCounters,
    lambda: f32,
    unit_goals: &crate::ai::search::unit_goals::UnitGoalStore,
    ranked: &[(f32, Box<dyn Move>)],
) {
    let Some(sims) = micro_probe_sims() else { return };
    let top_n = ranked.len().min(4);
    if top_n < 2 || matches!(ranked[0].1.move_type(), crate::types::MoveType::EndTurn) {
        return;
    }
    let depth = micro_probe_depth().max(1);
    for i in 0..sims {
        // Vary (idx, walk_depth) jointly, not idx alone -- else most sims
        // converge on identical leaves via deterministic continuation,
        // self-inflating the eval cache's hit rate instead of measuring it.
        let idx = i % top_n;
        let walk_depth = 1 + (i / top_n) % depth;
        let forced = ranked[idx].1.as_ref();
        if matches!(forced.move_type(), crate::types::MoveType::EndTurn) {
            continue;
        }
        let mut probe_game = view.clone();
        if probe_game.simulate_move(forced).is_none() {
            MICRO_PROBE_SIM_FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let mut probe_lane = lane_state.clone();
        let mut probe_counters = counters;
        probe_counters.count(forced);
        let mut ok = true;
        for _ in 1..walk_depth {
            if probe_game.state.settings.current_player_turn_id != pov {
                break;
            }
            // rank_view, not raw rank_plies: byte-for-byte the same per-ply
            // sequence the real decision uses, for free.
            let mut next = crate::ai::macro_agent::rank_view(
                &mut probe_game,
                pov,
                goal,
                &mut probe_lane,
                &mut probe_counters,
                lambda,
                Some(unit_goals),
            );
            if next.is_empty() {
                break;
            }
            let (_, mv) = next.swap_remove(0);
            if matches!(mv.move_type(), crate::types::MoveType::EndTurn) {
                break;
            }
            if probe_game.simulate_move(mv.as_ref()).is_none() {
                ok = false;
                break;
            }
            probe_counters.count(mv.as_ref());
        }
        if !ok {
            continue;
        }
        if let Ok(feats) =
            crate::ai::features::state_to_cpu_features_goal(&probe_game.state, pov, None, Some(goal))
        {
            let _ = evaluator.evaluate(vec![feats]);
            MICRO_PROBE_EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // probe_game drops here -- no undo call anywhere in this function.
    }
}

/// UCT exploration on [0,1]-mapped values, dialed to the MEASURED root q01
/// spread between directives (0.01–0.06, smoke probe 2026-08-12): at c=0.6
/// (and even the HeuristicMctsAgent 0.6 precedent) the bonus is 6–30x the
/// signal and visits stay uniform. 0.05 makes a 0.03 q01 gap decisive within
/// ~10 visits while genuine ties still split evenly.
const EXPLORATION: f32 = 0.05;

/// Tree depth cap in game turns from the root — beyond it a node is scored by
/// the leaf evaluator instead of expanded.
const TURN_DEPTH_CAP: i32 = 8;

fn other(p: PlayerId) -> PlayerId {
    if p == 1 { 2 } else { 1 }
}

fn seat(p: PlayerId) -> usize {
    (p != 1) as usize
}

/// Approximate a seat's whole-game purchase counters from the (fogged) state:
/// techs discovered after turn 0 (starting techs — Basic + the tribe tech —
/// are stamped turn 0 by mapgen; ruin grants are counted too, an acceptable
/// over-count). Keeps the tech caps binding for the in-tree opponent instead
/// of letting it out-research reality.
pub fn derive_counters(state: &GameState, player: PlayerId) -> TurnCounters {
    let Some(tribe) = state.tribes.get(&player) else {
        return TurnCounters::default();
    };
    let bought: Vec<_> = tribe
        .tech_vanilla
        .iter()
        .filter(|t| t.discovered && t.discovered_turn > 0)
        .collect();
    let tier3 = bought
        .iter()
        .filter(|t| {
            crate::settings::technology::get_technology_setting(t.tech_type).tier == Some(3)
        })
        .count() as u32;
    TurnCounters { techs_bought: bought.len() as u32, tier3_bought: tier3 }
}

/// Terminal value from `perspective`'s side by score comparison — the same
/// convention the backup expects of `evaluate_state` at that node.
fn terminal_value(state: &GameState, perspective: PlayerId) -> f32 {
    let my = state.tribes.get(&perspective).map(|t| t.score).unwrap_or(0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != perspective)
        .map(|(_, t)| t.score)
        .max()
        .unwrap_or(0);
    if my > opp {
        1.0
    } else if my < opp {
        -1.0
    } else {
        0.0
    }
}

/// One tree node = one turn boundary. `player` is the acting player BY
/// ALTERNATION from the root (not read from the state — a game that ends
/// mid-edge leaves `current_player_turn_id` unreliable). Every value stored
/// or computed at a node is from `player`'s perspective; the backup negates
/// once per level (negamax).
struct Node {
    game: Game,
    player: PlayerId,
    counters: [TurnCounters; 2],
    lane_states: [LaneState; 2],
    candidates: Vec<MacroGoal>,
    children: Vec<Option<usize>>,
    edge_visits: Vec<f32>,
    edge_values: Vec<f32>,
    /// EXP_ELO_036b: potential-based edge reward w·(φ(s',g)−φ(s,g)) from the
    /// EDGE OWNER's perspective; nonzero only on the root player's edges.
    edge_shape: Vec<f32>,
    visits: f32,
    /// Terminal (game over) or depth-capped leaf value, from `player`'s
    /// perspective; computed once (the executor and evaluator are
    /// deterministic within a process).
    frozen_value: Option<f32>,
    /// The edge that produced this node: who executed the turn, and the
    /// directive they executed. `None` at the root. This is the one piece of
    /// committed-directive truth a leaf can have without searching.
    from: Option<(PlayerId, MacroGoal)>,
    /// Root-only PUCT-style prior over `candidates`, decoded from the macro
    /// policy head's (stance, order) prediction at this node's state and
    /// pre-scaled by `MacroParams::root_prior_w`. Empty for every non-root
    /// node and for the root when the weight is 0.0 (the default) — in both
    /// cases `select_edge` runs the original plain-UCT path unchanged.
    edge_prior: Vec<f32>,
}

impl Node {
    fn new(
        game: Game,
        player: PlayerId,
        counters: [TurnCounters; 2],
        mut lane_states: [LaneState; 2],
        root_turn: i32,
        k: usize,
        from: Option<(PlayerId, MacroGoal)>,
        leaf_fn: &dyn Fn(&crate::states::GameState, PlayerId, u32) -> f32,
    ) -> Self {
        let frozen_value = if game.state.settings._game_over {
            Some(terminal_value(&game.state, player))
        } else if game.state.settings.turn - root_turn >= TURN_DEPTH_CAP {
            Some(leaf_fn(
                &game.state,
                player,
                counters[seat(player)].tier3_bought,
            ))
        } else {
            None
        };
        let candidates = if frozen_value.is_some() {
            Vec::new()
        } else {
            let base = compute_macro_goal(&game.state, player, counters[seat(player)].tier3_bought);
            // A node IS a turn boundary, so the Tier-1 selector belongs here —
            // for BOTH seats. The executor plies below only observe, so
            // without this the simulated opponent would play laneless for the
            // whole rollout (no lane techs, no preferred-unit pricing) and
            // the tree would evaluate futures the real game never produces.
            let s = seat(player);
            crate::ai::oracle_macro::observe_lane_state(&game.state, player, &mut lane_states[s]);
            crate::ai::oracle_macro::select_lane(&game.state, player, &mut lane_states[s], None);
            enumerate_candidates(&game.state, player, base, counters[seat(player)], k)
        };
        let n = candidates.len();
        Node {
            game,
            player,
            counters,
            lane_states,
            candidates,
            children: vec![None; n],
            edge_visits: vec![0.0; n],
            edge_values: vec![0.0; n],
            edge_shape: vec![0.0; n],
            visits: 0.0,
            frozen_value,
            from,
            edge_prior: Vec::new(),
        }
    }

    /// UCT over edges on [0,1]-mapped Q; unvisited edges first, in candidate
    /// order (base first) — ALWAYS, regardless of `edge_prior`. A first
    /// version let the prior reorder cold-start too, but `argmax(w·p)` picks
    /// the same edge for every `w > 0` — that reordering was a hard on/off
    /// switch, not a dial, and with only `sims`=16-64 split across a handful
    /// of candidates, cold start alone can dominate the whole tree's visit
    /// budget. Measured: `root_prior_w=0.05` regressed nearly as hard as
    /// `1.0` (EXP_ELO_067 sweep, Aug 21) — the smoking gun for exactly this.
    /// Cold start now stays prior-agnostic on principle (byte-identical to
    /// plain UCT at this stage, prior or not); once every edge has a visit,
    /// the exploration score gets a PUCT-style `prior/(1+n)` bonus on top of
    /// the existing UCT term (added, not replacing it), which IS genuinely
    /// continuous in `root_prior_w` — this is the only place the prior acts.
    fn select_edge(&self) -> usize {
        if let Some(i) = self.edge_visits.iter().position(|&v| v == 0.0) {
            return i;
        }
        let ln_n = self.visits.max(1.0).ln();
        let sqrt_n = self.visits.max(1.0).sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..self.candidates.len() {
            let q01 = (self.edge_values[i] / self.edge_visits[i] + 1.0) / 2.0;
            let mut score = q01 + EXPLORATION * (ln_n / self.edge_visits[i]).sqrt();
            if let Some(&p) = self.edge_prior.get(i) {
                score += p * sqrt_n / (1.0 + self.edge_visits[i]);
            }
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }
}

/// Decode the macro policy head's marginalized (stance[4], order[3·H·W])
/// prediction into a per-candidate prior, mirroring `self_play.rs`'s
/// `macro_policy_targets` in reverse: that function sums visit-mass INTO
/// stance/order; this scores each candidate's (stance, orders) choice
/// AGAINST the predicted marginals. Orders were never trained as a joint
/// distribution (each (kind, target) slot is an independent sigmoid), so a
/// candidate with multiple orders is scored by their geometric mean, not
/// product — a plain product would shrink toward 0 with every extra order
/// regardless of how well each one matches, penalizing goals for having
/// more concurrent orders rather than for being a worse match. Returns a
/// distribution over `candidates` (sums to 1); falls back to uniform if the
/// decoded scores are degenerate (all ~0 or non-finite).
fn decode_macro_prior(
    stance_probs: &[f32],
    order_maps: &[f32],
    candidates: &[MacroGoal],
    map_size: usize,
) -> Vec<f32> {
    let board = map_size * map_size;
    let mut raw: Vec<f32> = candidates
        .iter()
        .map(|g| {
            let stance_p = stance_probs.get(g.stance as usize).copied().unwrap_or(0.0).max(1e-6);
            if g.orders.is_empty() {
                return stance_p;
            }
            let log_sum: f32 = g
                .orders
                .iter()
                .map(|&(kind, target)| {
                    let idx = kind as usize * board
                        + usize::try_from(target).unwrap_or(0).min(board.saturating_sub(1));
                    order_maps.get(idx).copied().unwrap_or(0.0).max(1e-6).ln()
                })
                .sum();
            let geo_mean = (log_sum / g.orders.len() as f32).exp();
            stance_p * geo_mean
        })
        .collect();
    let total: f32 = raw.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        let n = candidates.len().max(1);
        return vec![1.0 / n as f32; candidates.len()];
    }
    for r in raw.iter_mut() {
        *r /= total;
    }
    raw
}

/// Search telemetry from the last `run` call (smoke instrumentation).
#[derive(Clone, Debug, Default)]
pub struct MacroMctsStats {
    pub nodes: usize,
    pub max_depth: usize,
    pub root_visit_max_share: f32,
    /// Mean value of the WINNING root edge, from the root player's
    /// perspective — the turn-level analogue of Gumbel's root value, used as
    /// the TD bootstrap. `None` when the winning edge was never backed up.
    pub root_q: Option<f32>,
    /// Spread (max − min) of the backed-up edge means over visited root
    /// edges — the value difference the tree must resolve to prefer one
    /// directive over another. `None` when fewer than two edges were backed.
    pub root_q_spread: Option<f32>,
    /// First step toward a macro policy head (Stage 3b): the root's own
    /// candidate ballot and post-search visit counts, parallel arrays.
    /// Populated regardless of leaf kind — a heuristic-leaf tree's visit
    /// distribution is still real behavior-cloning supervision. Raw, not
    /// pre-encoded into any (stance/order/target) head shape yet: encoding
    /// decisions wait until there's real data to design against.
    pub root_candidates: Vec<MacroGoal>,
    pub root_visits: Vec<f32>,
}

pub struct MacroMctsSearch<'a> {
    nodes: Vec<Node>,
    /// Root player: the only player whose edges earn Δφ shaping rewards.
    pov: PlayerId,
    /// Stage 3: leaf scoring — heuristic `evaluate_state`, or the trained
    /// value head when `leaf == Net` (macro-distilled model, EXP_ELO_039).
    eval: &'a crate::ai::eval_server::Evaluator,
    leaf: crate::ai::macro_agent::MacroLeaf,
    pub stats: MacroMctsStats,
}

/// Leaf value from `player`'s perspective. Net mode paints the SCRIPTED base
/// goal for the leaf player (the committed directive is unknowable before
/// the choice — a registered approximation) and reads win_value only
/// (`.1` progress is stubbed 0.0 on tch/metal). Falls back to the heuristic
/// on any feature/eval failure.
fn leaf_value(
    eval: &crate::ai::eval_server::Evaluator,
    leaf: crate::ai::macro_agent::MacroLeaf,
    state: &crate::states::GameState,
    player: PlayerId,
    tier3: u32,
    // The edge that produced this state: (the player who executed the turn,
    // the directive they executed). `None` at the root. Only `NetAsymPaint`
    // reads it — for that player the committed directive is KNOWN, which is
    // the one painting inference can align with training for free.
    from: Option<(PlayerId, &MacroGoal)>,
) -> f32 {
    use crate::ai::macro_agent::MacroLeaf;
    let aligned = leaf == MacroLeaf::NetAsymPaint;
    let net = |p: PlayerId| -> Option<f32> {
        let scripted;
        let goal = match from {
            Some((mover, g)) if aligned && mover == p => g,
            _ => {
                scripted = compute_macro_goal(state, p, tier3);
                &scripted
            }
        };
        crate::ai::features::state_to_cpu_features_goal(state, p, None, Some(goal))
            .ok()
            .and_then(|f| eval.evaluate(vec![f]).first().map(|r| r.0))
    };
    match leaf {
        MacroLeaf::Net => {
            if let Some(v) = net(player) {
                return v;
            }
        }
        // Both perspectives, halved: makes the zero-sum identity the negamax
        // backup assumes hold BY CONSTRUCTION, at two forwards per leaf.
        MacroLeaf::NetAsym | MacroLeaf::NetAsymPaint => {
            let opp: PlayerId = if player == 1 { 2 } else { 1 };
            if let (Some(a), Some(b)) = (net(player), net(opp)) {
                return (a - b) / 2.0;
            }
        }
        MacroLeaf::Heuristic => {}
    }
    crate::ai::evaluate_state(state, player)
}

impl<'a> MacroMctsSearch<'a> {
    /// Run `sims` simulations from `root_game` (the acting player's fogged
    /// view) and return the winning root directive index. Root candidate 0
    /// must be the committed script base; ties break toward it.
    pub fn run(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
    ) -> (usize, MacroMctsStats) {
        Self::run_with(
            root_game,
            pov,
            root_candidates,
            own_counters,
            own_lane_state,
            params,
            evaluator,
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_with(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
        inspect: impl FnOnce(&MacroMctsSearch),
    ) -> (usize, MacroMctsStats) {
        debug_assert_eq!(root_game.state.tribes.len(), 2, "macro MCTS is 2-player only");
        let root_turn = root_game.state.settings.turn;
        let mut counters = [TurnCounters::default(); 2];
        counters[seat(pov)] = own_counters;
        counters[seat(other(pov))] = derive_counters(&root_game.state, other(pov));
        let mut lane_states: [LaneState; 2] = Default::default();
        lane_states[seat(pov)] = own_lane_state.clone();

        let leaf = params.leaf;
        // The root has no incoming edge, so no committed directive is known
        // for it — every leaf read there falls back to the scripted painting.
        let leaf_fn = move |s: &crate::states::GameState, p: PlayerId, t3: u32| {
            leaf_value(evaluator, leaf, s, p, t3, None)
        };
        let mut root =
            Node::new(root_game.clone(), pov, counters, lane_states, root_turn, params.k, None, &leaf_fn);
        root.candidates = root_candidates;
        let n = root.candidates.len();
        root.children = vec![None; n];
        root.edge_visits = vec![0.0; n];
        root.edge_values = vec![0.0; n];
        root.edge_shape = vec![0.0; n];

        // War-room item 3: inject the macro policy head as a PUCT-style
        // prior at the root only, one eval call per real turn decision (not
        // per rollout — cheap). EXP_ELO_066: the first version of this
        // painted the scripted base goal here, matching `leaf_value`'s
        // fallback convention — but that convention was WRONG for this
        // specific head. Every existing macro_stance/macro_order training
        // row is painted with the search's own COMMITTED (post-search,
        // already-chosen) goal, so the head partly learned to echo its own
        // input rather than predict blind, and painting anything at the
        // root (which by definition has no committed goal yet) fed it an
        // out-of-distribution input — measured as a real −18.75pp
        // regression, not a weak prior. The fix is training-side (repaint
        // those label rows `None` = goal-blind) and this call must paint
        // the SAME way for the two to agree once retrained. Off (0.0
        // weight) by default: skips the eval call entirely and leaves
        // `edge_prior` empty, so `select_edge` is byte-identical to plain
        // UCT unless explicitly turned on.
        if params.root_prior_w > 0.0 && n > 0 {
            if let Ok(feats) = crate::ai::features::state_to_cpu_features_goal(
                &root_game.state,
                pov,
                None,
                None,
            ) {
                if let Some(result) = evaluator.evaluate(vec![feats]).into_iter().next() {
                    if let (Some(stance), Some(order)) =
                        (&result.2.macro_stance, &result.2.macro_order)
                    {
                        let map_size = root_game.state.settings.size as usize;
                        let prior = decode_macro_prior(stance, order, &root.candidates, map_size);
                        root.edge_prior =
                            prior.iter().map(|p| p * params.root_prior_w).collect();
                    }
                }
            }
        }

        let mut search = MacroMctsSearch {
            nodes: vec![root],
            pov,
            eval: evaluator,
            leaf,
            stats: MacroMctsStats::default(),
        };
        for _ in 0..params.sims.max(1) {
            search.simulate(0, root_turn, params);
        }
        inspect(&search);

        let root = &search.nodes[0];
        let mut best = 0;
        for i in 1..root.edge_visits.len() {
            if root.edge_visits[i] > root.edge_visits[best] {
                best = i;
            }
        }
        search.stats.nodes = search.nodes.len();
        search.stats.root_visit_max_share = if root.visits > 0.0 {
            root.edge_visits.iter().cloned().fold(0.0, f32::max) / root.visits
        } else {
            0.0
        };
        search.stats.root_q = if root.edge_visits[best] > 0.0 {
            Some((root.edge_values[best] / root.edge_visits[best]).clamp(-1.0, 1.0))
        } else {
            None
        };
        let backed: Vec<f32> = (0..root.candidates.len())
            .filter(|&i| root.edge_visits[i] > 0.0)
            .map(|i| root.edge_values[i] / root.edge_visits[i])
            .collect();
        search.stats.root_q_spread = if backed.len() > 1 {
            let hi = backed.iter().cloned().fold(f32::MIN, f32::max);
            let lo = backed.iter().cloned().fold(f32::MAX, f32::min);
            Some(hi - lo)
        } else {
            None
        };
        search.stats.root_candidates = root.candidates.clone();
        search.stats.root_visits = root.edge_visits.clone();
        (best, search.stats)
    }

    /// `run` plus a per-edge root dump on stdout (smoke instrumentation only).
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn run_probed(
        root_game: &Game,
        pov: PlayerId,
        root_candidates: Vec<MacroGoal>,
        own_counters: TurnCounters,
        own_lane_state: &LaneState,
        params: &MacroParams,
        evaluator: &crate::ai::eval_server::Evaluator,
    ) -> (usize, MacroMctsStats) {
        let cands_dbg: Vec<String> = root_candidates
            .iter()
            .map(|c| format!("{:?}/{}ord", c.stance, c.orders.len()))
            .collect();
        let (best, stats) = Self::run_with(root_game, pov, root_candidates, own_counters, own_lane_state, params, evaluator, |s| {
            let root = &s.nodes[0];
            for i in 0..root.candidates.len() {
                let q = if root.edge_visits[i] > 0.0 {
                    root.edge_values[i] / root.edge_visits[i]
                } else {
                    f32::NAN
                };
                println!(
                    "    edge {i} [{}]: visits={} q={q:+.4} shape={:+.4}",
                    cands_dbg[i], root.edge_visits[i], root.edge_shape[i]
                );
            }
        });
        (best, stats)
    }

    /// One descent: walk UCT edges until an unexpanded edge, expand it (one
    /// `execute_turn` on a fresh clone), then back the child's value up the
    /// path with one negation per level.
    fn simulate(&mut self, root_idx: usize, root_turn: i32, params: &MacroParams) {
        let mut path: Vec<(usize, usize)> = Vec::new();
        let mut idx = root_idx;
        let mut value: f32;
        loop {
            if let Some(v) = self.nodes[idx].frozen_value {
                value = v;
                break;
            }
            if self.nodes[idx].candidates.is_empty() {
                let n = &self.nodes[idx];
                value = leaf_value(
                    self.eval,
                    self.leaf,
                    &n.game.state,
                    n.player,
                    n.counters[seat(n.player)].tier3_bought,
                    n.from.as_ref().map(|(p, g)| (*p, g)),
                );
                break;
            }
            let e = self.nodes[idx].select_edge();
            path.push((idx, e));
            if let Some(child) = self.nodes[idx].children[e] {
                idx = child;
                continue;
            }
            let child = self.expand(idx, e, root_turn, params);
            let cn = &self.nodes[child];
            value = cn.frozen_value.unwrap_or_else(|| {
                leaf_value(
                    self.eval,
                    self.leaf,
                    &cn.game.state,
                    cn.player,
                    cn.counters[seat(cn.player)].tier3_bought,
                    cn.from.as_ref().map(|(p, g)| (*p, g)),
                )
            });
            // The child's value is from the child's perspective; the edge we
            // just descended belongs to the parent, so negate once here and
            // once per level in the unwind below.
            break;
        }
        for &(pidx, e) in path.iter().rev() {
            // Negamax with edge rewards: the child's value arrives from the
            // child's perspective; the edge belongs to the parent, so
            // v(parent) = r(edge) − v(child). r is 0 on opponent edges,
            // reducing to the plain negation this replaced (EXP_ELO_036b).
            value = self.nodes[pidx].edge_shape[e] - value;
            let node = &mut self.nodes[pidx];
            node.visits += 1.0;
            node.edge_visits[e] += 1.0;
            node.edge_values[e] += value;
        }
        self.stats.max_depth = self.stats.max_depth.max(path.len());
    }

    fn expand(&mut self, parent: usize, edge: usize, root_turn: i32, params: &MacroParams) -> usize {
        let (mut game, player, mut counters, mut lane_states, goal) = {
            let p = &self.nodes[parent];
            (
                p.game.clone(),
                p.player,
                p.counters,
                p.lane_states.clone(),
                p.candidates[edge].clone(),
            )
        };
        let s = seat(player);
        // EXP_ELO_036b: pre-move potential of THIS edge's directive, with one
        // GoalAux for both sides of the difference (the executor's
        // edge_snapshot pattern) — mixing directives or auxes across the
        // difference would mint reward on switches, not approach.
        let shape_pre = if params.shape_w != 0.0 && player == self.pov {
            let aux = crate::ai::oracle_macro::compute_goal_aux(
                &game.state,
                player,
                &goal,
                counters[s].techs_bought,
                counters[s].tier3_bought,
                Some(&lane_states[s]),
            );
            Some((crate::ai::reward::goal_potential(&game.state, player, &goal, Some(&aux)), aux))
        } else {
            None
        };
        // An executor anomaly leaves the state where it stopped; the node is
        // still scoreable, so treat it like any other boundary.
        // EXP_ELO_061: rollout_lambda, not lambda -- this runs up to `sims`
        // times per real turn (once per node expansion), vs the real
        // per-ply commit's one call in select_move.
        let _ = macro_exec::execute_turn(
            &mut game,
            player,
            &goal,
            &mut lane_states[s],
            &mut counters[s],
            params.rollout_lambda,
        );
        let shape = match &shape_pre {
            Some((pre, aux)) => {
                let post =
                    crate::ai::reward::goal_potential(&game.state, player, &goal, Some(aux));
                params.shape_w * (post - pre)
            }
            None => 0.0,
        };
        let leaf = self.leaf;
        let eval = self.eval;
        // The child's depth-capped value (computed inside `Node::new`) gets the
        // same edge context every other leaf read of this node will get.
        let from = (player, goal.clone());
        let leaf_fn = move |s: &crate::states::GameState, p: PlayerId, t3: u32| {
            leaf_value(eval, leaf, s, p, t3, Some((from.0, &from.1)))
        };
        let child = Node::new(
            game,
            other(player),
            counters,
            lane_states,
            root_turn,
            params.k,
            Some((player, goal.clone())),
            &leaf_fn,
        );
        let child_idx = self.nodes.len();
        self.nodes.push(child);
        self.nodes[parent].children[edge] = Some(child_idx);
        self.nodes[parent].edge_shape[edge] = shape;
        if let Some(path) = macro_rollout_trace_path() {
            let child_state = &self.nodes[child_idx].game.state;
            dump_rollout_node(
                path,
                child_idx,
                parent,
                child_state.settings.turn,
                player,
                &goal,
                self.pov,
                child_state,
            );
        }
        child_idx
    }
}

/// Stage 2 agent: per-turn directive commit like the Stage-1 lookahead, but
/// the directive is chosen by the adversarial turn-level tree.
pub struct MacroMctsAgent<'a> {
    /// Stage 3: leaf evaluator when `params.leaf == Net` (idle otherwise).
    evaluator: &'a crate::ai::eval_server::Evaluator,
    params: MacroParams,
    stance_commit: StanceCommit,
    lane_state: LaneState,
    counters: TurnCounters,
    plan_key: Option<(i32, PlayerId)>,
    turn_goal: Option<MacroGoal>,
    pub divergent_turns: u32,
    pub planned_turns: u32,
    pub last_stats: MacroMctsStats,
    /// EXP_ELO_035: belief handed in by the harness each turn; consumed per
    /// `params.belief_mode` (World: materialize the plan view; Candidates:
    /// belief-conditioned root enumeration; Both; Off: ignored).
    pub belief: Option<crate::ai::belief::BeliefState>,
    pub mat_capital_turns: u32,
    pub mat_units: u32,
    /// EXP_ELO_036/038: winning candidate class per planned turn, indexed by
    /// `CandidateClass as usize` (base/stance/real/attack/claim/contest/
    /// continuation).
    pub class_picks: [u32; crate::ai::macro_agent::CANDIDATE_CLASSES],
    /// Same belief fog-target re-picked on the very next planned turn
    /// (fresh claim/contest picks only; sustained plays show as
    /// Continuation picks since 038).
    pub belief_repicks: u32,
    last_belief_target: Option<i32>,
    /// EXP_ELO_038 (Verdi spec): the strategist's last picked directives —
    /// re-offered each plan as Continuation candidates. Continuity through
    /// informed selection; the 036b/037 forced base-injection is gone.
    recent_goals: std::collections::VecDeque<MacroGoal>,
    /// EXP_ELO_037: fog orders stripped from the live goal MID-TURN when a
    /// ply's own vision disconfirmed them (per-ply belief consumption).
    pub intra_strips: u32,
    /// Per-unit-goal design (Aug 2026): real-trajectory-only persistent
    /// EXPAND assignment, reconciled once per real ply in `select_move`.
    /// Never threaded into rollouts (Fork 2) -- rollouts stay on the
    /// ephemeral `None` path, byte-identical to before this field existed.
    unit_goals: crate::ai::search::unit_goals::UnitGoalStore,
    /// Micro-mcts root-advancement warm start (see `micro_mcts::
    /// MicroTreeCarry`). Reset whenever a new turn is planned or the
    /// predicted move didn't end up executed; otherwise carried ply to ply
    /// within the same turn so a search's unused depth isn't discarded.
    micro_carry: Option<crate::ai::search::micro_mcts::MicroTreeCarry>,
}

/// EXP_ELO_038: how many recent picked directives stay on the ballot.
const RECENT_GOALS: usize = 3;

/// A fog-expansion order the observer's own vision has disconfirmed:
/// explored, not ours-with-city (achieved orders keep paying by 028's
/// achieved-holds-cap semantics), not capturable, not retakeable.
fn fog_order_dead(state: &crate::states::GameState, t: i32, pov: PlayerId) -> bool {
    let Some(tile) = state.tiles.get(&t) else { return true };
    if !tile.explorers.contains(&pov) {
        return false;
    }
    if tile.owner == pov && crate::functions::get_city_at(state, t).is_some() {
        return false;
    }
    !crate::ai::oracle_macro::expand_target_valid(state, t, pov)
}

/// EXP_ELO_047 Phase A. One JSONL row per planned root: the same state read
/// under the two paintings that training and inference disagree about, plus
/// the antisymmetry check the Phase B negation depends on. Env-gated
/// (`POLYFISH_PAINT_PROBE=<path>`); no cost and no rows when unset.
fn paint_probe(
    eval: &crate::ai::eval_server::Evaluator,
    state: &crate::states::GameState,
    pov: PlayerId,
    scripted: &MacroGoal,
    committed: Option<&MacroGoal>,
    diverged: bool,
    stats: &MacroMctsStats,
) {
    let Ok(path) = std::env::var("POLYFISH_PAINT_PROBE") else {
        return;
    };
    let v = |p: PlayerId, g: &MacroGoal| -> Option<f32> {
        crate::ai::features::state_to_cpu_features_goal(state, p, None, Some(g))
            .ok()
            .and_then(|f| eval.evaluate(vec![f]).first().map(|r| r.0))
    };
    let opp: PlayerId = if pov == 1 { 2 } else { 1 };
    let opp_scripted = compute_macro_goal(state, opp, 0);
    let (Some(v_scripted), Some(v_committed), Some(v_opp)) = (
        v(pov, scripted),
        v(pov, committed.unwrap_or(scripted)),
        v(opp, &opp_scripted),
    ) else {
        return;
    };
    // Control for P3: the heuristic on the SAME (fogged) states. Fog alone
    // makes a view non-zero-sum for any evaluator, so the net's asymmetry is
    // only the net's insofar as it exceeds this.
    let h_pov = crate::ai::evaluate_state(state, pov);
    let h_opp = crate::ai::evaluate_state(state, opp);
    let f = |x: Option<f32>| x.map(|n| format!("{n:.5}")).unwrap_or("null".into());
    let row = format!(
        "{{\"turn\":{},\"pov\":{},\"diverged\":{},\"v_scripted\":{:.5},\"v_committed\":{:.5},\"v_opp\":{:.5},\"h_pov\":{:.5},\"h_opp\":{:.5},\"q_spread\":{},\"q_best\":{}}}\n",
        state.settings.turn,
        pov,
        diverged,
        v_scripted,
        v_committed,
        v_opp,
        h_pov,
        h_opp,
        f(stats.root_q_spread),
        f(stats.root_q),
    );
    use std::io::Write;
    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = fh.write_all(row.as_bytes());
    }
}

/// EXP_ELO_048: does Tier 3 follow Tier 2? Re-executes this root's turn on
/// throwaway clones under three directives — the tree's pick, the scripted
/// base, and no directive at all — and reports how much the executed PLY
/// SEQUENCE actually changes. Env-gated (`POLYFISH_TIER_PROBE=<path>`).
fn tier_probe(
    view: &Game,
    pov: PlayerId,
    lane_state: &LaneState,
    counters: TurnCounters,
    lambda: f32,
    base: &MacroGoal,
    picked: &MacroGoal,
    diverged: bool,
) {
    let Ok(path) = std::env::var("POLYFISH_TIER_PROBE") else {
        return;
    };
    let run = |goal: &MacroGoal| -> (Vec<crate::ai::macro_exec::PlyRec>, Game) {
        let mut g = view.clone();
        let mut a = lane_state.clone();
        let mut c = counters;
        let mut rec = Vec::new();
        macro_exec::execute_turn_recorded(&mut g, pov, goal, &mut a, &mut c, lambda, Some(&mut rec));
        (rec, g)
    };
    let (a, after_pick) = run(picked);
    let (b, _) = run(base);
    let (c, after_none) = run(&MacroGoal::default());
    // Multiset overlap on serialized moves: order-insensitive, so a reordered
    // but identical set of plies still reads as "the directive changed
    // nothing", which is the conservative direction for this question.
    let overlap = |x: &[crate::ai::macro_exec::PlyRec], y: &[crate::ai::macro_exec::PlyRec]| -> f32 {
        let mut pool: Vec<&String> = y.iter().map(|p| &p.mv).collect();
        let mut hit = 0usize;
        for p in x {
            if let Some(i) = pool.iter().position(|q| **q == p.mv) {
                pool.remove(i);
                hit += 1;
            }
        }
        let denom = x.len().max(y.len());
        if denom == 0 { 1.0 } else { hit as f32 / denom as f32 }
    };
    // Star-spending plies only: Steps shuffle cheaply, but Research/Build/
    // Summon are where a turn's stars are actually committed — and the star
    // gate is the one intervention that ever moved wins (EXP_ELO_026).
    let spend = |v: &[crate::ai::macro_exec::PlyRec]| -> Vec<crate::ai::macro_exec::PlyRec> {
        v.iter()
            .filter(|p| matches!(p.kind.as_str(), "Research" | "Build" | "Summon"))
            .cloned()
            .collect()
    };
    let (sa, sb, sc) = (spend(&a), spend(&b), spend(&c));
    let flips_phi = a.iter().filter(|p| p.flip_no_phi).count();
    let flips_goal = a.iter().filter(|p| p.flip_no_goal).count();
    // Verdi's obedience test: for each order, how far is the nearest own unit
    // from its target BEFORE the turn vs AFTER — under the directive, and
    // under no directive at all. The control is the point: an order whose
    // target gets closer just as fast without the order was never followed,
    // it was coincided with.
    let size = view.state.settings.size;
    let dist = |state: &crate::states::GameState, target: i32| -> Option<i32> {
        let t = crate::coords::Coords::from_index(target, size);
        state
            .tribes
            .get(&pov)?
            .units
            .iter()
            .map(|u| u.coords.chebyshev_distance_to(&t))
            .min()
    };
    let owned = |state: &crate::states::GameState, target: i32| -> bool {
        state
            .tiles
            .get(&target)
            .map_or(false, |t| t.owner == pov as i32)
    };
    let orders: Vec<String> = picked
        .orders
        .iter()
        .map(|(kind, t)| {
            let f = |x: Option<i32>| x.map(|v| v.to_string()).unwrap_or("null".into());
            format!(
                "{{\"kind\":\"{kind:?}\",\"target\":{t},\"d_pre\":{},\"d_pick\":{},\"d_none\":{},\
\"owned_pre\":{},\"owned_pick\":{}}}",
                f(dist(&view.state, *t)),
                f(dist(&after_pick.state, *t)),
                f(dist(&after_none.state, *t)),
                owned(&view.state, *t),
                owned(&after_pick.state, *t),
            )
        })
        .collect();
    let row = format!(
        "{{\"turn\":{},\"pov\":{},\"diverged\":{},\"plies\":{},\"spend_plies\":{},\
\"overlap_pick_base\":{:.4},\"overlap_pick_none\":{:.4},\
\"spend_overlap_pick_base\":{:.4},\"spend_overlap_pick_none\":{:.4},\
\"flip_no_phi\":{},\"flip_no_goal\":{},\"orders\":[{}]}}\n",
        view.state.settings.turn,
        pov,
        diverged,
        a.len(),
        sa.len(),
        overlap(&a, &b),
        overlap(&a, &c),
        overlap(&sa, &sb),
        overlap(&sa, &sc),
        flips_phi,
        flips_goal,
        orders.join(","),
    );
    use std::io::Write;
    if let Ok(mut fh) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = fh.write_all(row.as_bytes());
    }
}

impl<'a> MacroMctsAgent<'a> {
    pub fn new(evaluator: &'a crate::ai::eval_server::Evaluator, params: MacroParams) -> Self {
        Self {
            evaluator,
            params,
            stance_commit: StanceCommit::default(),
            lane_state: LaneState::default(),
            counters: TurnCounters::default(),
            plan_key: None,
            turn_goal: None,
            divergent_turns: 0,
            planned_turns: 0,
            last_stats: MacroMctsStats::default(),
            belief: None,
            mat_capital_turns: 0,
            mat_units: 0,
            class_picks: [0; crate::ai::macro_agent::CANDIDATE_CLASSES],
            belief_repicks: 0,
            last_belief_target: None,
            recent_goals: std::collections::VecDeque::new(),
            intra_strips: 0,
            unit_goals: crate::ai::search::unit_goals::UnitGoalStore::default(),
            micro_carry: None,
        }
    }

    pub fn set_belief(&mut self, belief: crate::ai::belief::BeliefState) {
        self.belief = Some(belief);
    }

    /// The directive committed for the current turn — the goal that actually
    /// drove this ply, for feature painting (Stage 3: recorded features must
    /// carry the goal the agent pursued, not the scripted base).
    pub fn committed_goal(&self) -> Option<&MacroGoal> {
        self.turn_goal.as_ref()
    }

    /// Tier-1 state: the committed lane plus its tenure/budget/score
    /// bookkeeping — the top of the `ply <- order <- playstyle` attribution
    /// chain. The macro agent owns its own `LaneState`; the script
    /// path's copy (arena/self_play) is a different seat's.
    pub fn committed_playstyle(&self) -> &LaneState {
        &self.lane_state
    }

    /// Root value of this turn's committed directive, for the TD bootstrap.
    /// Only meaningful under a NET leaf — a heuristic-leaf Q is an
    /// `evaluate_state` number, and feeding that back into the value target
    /// would train the head toward the evaluator it is supposed to beat.
    pub fn last_root_value(&self) -> Option<f32> {
        match self.params.leaf {
            MacroLeaf::Net | MacroLeaf::NetAsym | MacroLeaf::NetAsymPaint => {
                self.last_stats.root_q
            }
            MacroLeaf::Heuristic => None,
        }
    }

    /// The current turn's root ballot: candidate directives and the tree's
    /// own post-search visit count per candidate, parallel arrays. Available
    /// under every leaf kind (unlike `last_root_value`) — the visit
    /// distribution is real search output regardless of what scored the
    /// leaves. Empty before the first search of the run.
    pub fn last_root_ballot(&self) -> (&[MacroGoal], &[f32]) {
        (&self.last_stats.root_candidates, &self.last_stats.root_visits)
    }

    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        let pov = game.state.settings.current_player_turn_id;
        let key = (game.state.settings.turn, pov);
        if self.plan_key != Some(key) {
            // A new turn invalidates any micro-mcts subtree from the last
            // one — it was built off a different goal/lane/state entirely.
            self.micro_carry = None;
            use crate::ai::macro_agent::{BeliefMode, CandidateClass};
            let mut view0 = game.clone_for_mcts(pov);
            let use_world =
                matches!(self.params.belief_mode, BeliefMode::World | BeliefMode::Both);
            let use_cand =
                matches!(self.params.belief_mode, BeliefMode::Candidates | BeliefMode::Both);
            if use_world {
                if let Some(b) = &self.belief {
                    let st = crate::ai::belief::materialize_into(&mut view0, b);
                    if st.capital {
                        self.mat_capital_turns += 1;
                    }
                    self.mat_units += st.ghost_units + st.residual_units;
                }
            }
            let base =
                commit_macro_goal(&view0.state, pov, &mut self.stance_commit, self.counters.tier3_bought);
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
            let candidates: Vec<MacroGoal> =
                tagged.iter().map(|(g, _)| g.clone()).collect();
            let (pick, stats) = MacroMctsSearch::run(
                &view0,
                pov,
                candidates.clone(),
                self.counters,
                &self.lane_state,
                &self.params,
                self.evaluator,
            );
            self.last_stats = stats;
            self.planned_turns += 1;
            if pick != 0 {
                self.divergent_turns += 1;
            }
            let picked_class = tagged.get(pick).map(|(_, c)| *c);
            if let Some(c) = picked_class {
                self.class_picks[c as usize] += 1;
            }
            // Plan-stability: the same belief fog-target winning consecutive
            // planned turns means units aren't being yanked mid-approach.
            let belief_target = match picked_class {
                Some(CandidateClass::ClaimSafe) | Some(CandidateClass::Contest) => tagged
                    .get(pick)
                    .and_then(|(g, _)| {
                        g.orders
                            .iter()
                            .find(|o| !base.orders.contains(o))
                            .map(|(_, t)| *t)
                    }),
                _ => None,
            };
            if belief_target.is_some() && belief_target == self.last_belief_target {
                self.belief_repicks += 1;
            }
            self.last_belief_target = belief_target;
            self.turn_goal = candidates.into_iter().nth(pick);
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
        let unit_status =
            crate::ai::search::unit_goals::reconcile_unit_goals(&view.state, pov, &goal, &mut self.unit_goals);
        let mut ranked = crate::ai::macro_agent::rank_view(
            &mut view,
            pov,
            &goal,
            &mut self.lane_state,
            &mut self.counters,
            self.params.lambda,
            Some(&self.unit_goals),
        );
        let mut pending_micro_carry: Option<(
            serde_json::Value,
            crate::ai::search::micro_mcts::MicroTreeCarry,
        )> = None;
        if let Some(micro_params) = crate::ai::search::micro_mcts::micro_mcts_params() {
            let star_gate = crate::ai::oracle_macro::tech_discipline_active(&view.state, pov, &goal);
            let aux = crate::ai::search::goal_aux::compute_goal_aux(
                &view.state,
                pov,
                &goal,
                self.counters.techs_bought,
                self.counters.tier3_bought,
                Some(&self.lane_state),
            );
            let (pick, next_carry) = crate::ai::search::micro_mcts::micro_search_pick(
                &view,
                pov,
                &goal,
                &ranked,
                &aux,
                star_gate,
                self.evaluator,
                &micro_params,
                self.micro_carry.take(),
            );
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
            dump_ply_decision(path, turn, pov, &goal, candidates, unit_goals_trace, m.as_ref());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::eval_server::{DummyEvalHandle, Evaluator};
    use crate::ai::oracle_macro::{OrderKind, Stance};

    fn generated_game(seed: i64) -> Game {
        let mut game = Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: crate::types::MapSize::Tiny,
            map_type: crate::types::MapType::Drylands,
            tribes: vec![crate::types::TribeType::Imperius, crate::types::TribeType::Bardur],
            seed,
            version: 115,
        });
        game.post_load();
        game
    }

    /// EXP_ELO_036b: the negamax-with-edge-rewards backup, checked against a
    /// hand computation on a prebuilt 4-node chain. A sign error here
    /// silently inverts the shaping incentive at alternating depths.
    /// Chain: n0(pov, r1) → n1(opp, 0) → n2(pov, r2) → n3(opp, frozen v).
    /// Hand: n2.Q = r2−v; n1.Q = v−r2; n0.Q = r1+r2−v. With w=0 (all r=0)
    /// the backup must reduce to the plain negation it replaced.
    #[test]
    fn shaped_backup_matches_hand_computation() {
        let game = generated_game(3);
        let root_turn = game.state.settings.turn;
        let params = MacroParams::default();
        let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
        let heur = |s: &crate::states::GameState, p: PlayerId, _t3: u32| {
            crate::ai::evaluate_state(s, p)
        };
        let mk = |player: PlayerId, shape: f32, frozen: Option<f32>| {
            let mut n = Node::new(
                game.clone(),
                player,
                [TurnCounters::default(); 2],
                Default::default(),
                root_turn,
                1,
                None,
                &heur,
            );
            n.candidates = vec![MacroGoal::default()];
            n.children = vec![None];
            n.edge_visits = vec![0.0];
            n.edge_values = vec![0.0];
            n.edge_shape = vec![shape];
            n.frozen_value = frozen;
            n
        };
        let (r1, r2, v) = (0.3f32, 0.2f32, 0.5f32);
        for (s1, s2, expect_root, expect_n1, expect_n2) in [
            (r1, r2, r1 + r2 - v, v - r2, r2 - v), // shaped
            (0.0, 0.0, -v, v, -v),                 // w=0 regression: plain negation
        ] {
            let mut search = MacroMctsSearch {
                nodes: vec![
                    mk(1, s1, None),
                    mk(2, 0.0, None),
                    mk(1, s2, None),
                    mk(2, 0.0, Some(v)),
                ],
                pov: 1,
                eval: &evaluator,
                leaf: crate::ai::macro_agent::MacroLeaf::Heuristic,
                stats: MacroMctsStats::default(),
            };
            search.nodes[0].children[0] = Some(1);
            search.nodes[1].children[0] = Some(2);
            search.nodes[2].children[0] = Some(3);
            search.simulate(0, root_turn, &params);
            assert!((search.nodes[0].edge_values[0] - expect_root).abs() < 1e-6,
                "root Q {} != {expect_root}", search.nodes[0].edge_values[0]);
            assert!((search.nodes[1].edge_values[0] - expect_n1).abs() < 1e-6,
                "n1 Q {} != {expect_n1}", search.nodes[1].edge_values[0]);
            assert!((search.nodes[2].edge_values[0] - expect_n2).abs() < 1e-6,
                "n2 Q {} != {expect_n2}", search.nodes[2].edge_values[0]);
        }
    }

    /// EXP_ELO_036b w-dial (q-gap method): root q spread vs shaped-edge
    /// magnitudes at candidate w values, on mid-window scripted states with
    /// belief-conditioned candidates. Run manually:
    ///   cargo test --lib ai::macro_mcts -- --ignored shape_w_dial --nocapture
    #[test]
    #[ignore]
    fn shape_w_dial_probe() {
        use crate::ai::belief::BeliefState;
        use crate::ai::macro_agent::enumerate_candidates_with_belief;
        for seed in [11i64, 12, 13, 14] {
            let mut game = generated_game(9_600_000 + seed);
            // Advance to the belief window.
            let mut lane_state = LaneState::default();
            let mut counters = TurnCounters::default();
            for _ in 0..12 {
                if game.state.settings._game_over {
                    break;
                }
                let player = game.state.settings.current_player_turn_id;
                let goal = compute_macro_goal(&game.state, player, 0);
                if !macro_exec::execute_turn(&mut game, player, &goal, &mut lane_state, &mut counters, 1.0)
                {
                    break;
                }
            }
            let pov = game.state.settings.current_player_turn_id;
            let opp: PlayerId = if pov == 1 { 2 } else { 1 };
            let own = game
                .state
                .tiles
                .iter()
                .find(|(_, t)| t.capital_of == pov)
                .map(|(&i, _)| i)
                .unwrap_or(24);
            let b = BeliefState::new(11, 2, own, pov, opp);
            let view = game.clone_for_mcts(pov);
            let mut commit = StanceCommit::default();
            let base = commit_macro_goal(&view.state, pov, &mut commit, 0);
            let tagged = enumerate_candidates_with_belief(
                &view.state,
                pov,
                base,
                TurnCounters::default(),
                7,
                Some(&b),
            );
            let cands: Vec<MacroGoal> = tagged.iter().map(|(g, _)| g.clone()).collect();
            let classes: Vec<String> =
                tagged.iter().map(|(_, c)| format!("{c:?}")).collect();
            for w in [0.0f32, 1e-4, 3e-4, 1e-3] {
                let params = MacroParams {
                    k: 7,
                    sims: 48,
                    shape_w: w,
                    ..MacroParams::default()
                };
                println!("  seed {seed} t{} w={w}: classes={classes:?}", view.state.settings.turn);
                let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
                let (pick, _) = MacroMctsSearch::run_probed(
                    &view,
                    pov,
                    cands.clone(),
                    TurnCounters::default(),
                    &LaneState::default(),
                    &params,
                    &evaluator,
                );
                println!("    -> pick={pick} [{}]", classes.get(pick).cloned().unwrap_or_default());
            }
        }
    }

    /// Negamax correctness depends on this: evaluate_state must be
    /// antisymmetric in a 2-player game, including after several executed
    /// turns of drift.
    #[test]
    fn evaluate_state_is_antisymmetric() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            for _ in 0..6 {
                let s = evaluate_asym(&sim.state);
                assert!(s < 1e-4, "seed {seed}: antisymmetry violated by {s}");
                if sim.state.settings._game_over {
                    break;
                }
                let player = sim.state.settings.current_player_turn_id;
                let goal = compute_macro_goal(&sim.state, player, 0);
                let mut lane_state = LaneState::default();
                let mut counters = TurnCounters::default();
                if !macro_exec::execute_turn(&mut sim, player, &goal, &mut lane_state, &mut counters, 1.0)
                {
                    break;
                }
            }
        }
    }

    fn evaluate_asym(state: &GameState) -> f32 {
        (crate::ai::evaluate_state(state, 1) + crate::ai::evaluate_state(state, 2)).abs()
    }

    #[test]
    fn tree_goes_deeper_than_the_root() {
        for seed in 0..4i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let view = game.clone_for_mcts(pov);
            let base = compute_macro_goal(&view.state, pov, 0);
            let cands = enumerate_candidates(&view.state, pov, base, TurnCounters::default(), 4);
            let k = cands.len();
            let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
            let (_, stats) = MacroMctsSearch::run(
                &view,
                pov,
                cands,
                TurnCounters::default(),
                &LaneState::default(),
                &MacroParams { sims: 32, ..Default::default() },
                &evaluator,
            );
            assert!(
                stats.nodes > k + 1,
                "seed {seed}: only {} nodes for k={k} at 32 sims — no second-level expansion",
                stats.nodes
            );
            assert!(stats.max_depth >= 2, "seed {seed}: max depth {} < 2", stats.max_depth);
        }
    }

    /// Stage 3 net-leaf path: the tree must run and return a true-legal
    /// move with `leaf: Net` (Dummy evaluator — exercises the feature-encode
    /// + evaluate plumbing, not the values).
    #[test]
    fn net_leaf_tree_returns_true_legal_move() {
        let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
        let mut game = generated_game(5);
        let mut agent = MacroMctsAgent::new(
            &evaluator,
            MacroParams {
                sims: 16,
                leaf: crate::ai::macro_agent::MacroLeaf::Net,
                ..Default::default()
            },
        );
        let m = agent.select_move(&mut game).unwrap();
        let legal: Vec<String> =
            game.legal_moves().iter().map(|x| x.serialize().to_string()).collect();
        assert!(legal.contains(&m.serialize().to_string()), "net-leaf true-illegal move");
    }

    /// EXP_ELO_047: negamax negates a child's value to get the parent's, so a
    /// leaf scorer must be zero-sum or every backup is off by the gap. The
    /// net is not (measured median 0.40 on fogged macro roots, ~13x the
    /// q-spread); `NetAsym` restores the identity by construction, and this
    /// pins it — a Dummy evaluator returning per-seat-asymmetric values still
    /// has to come out antisymmetric.
    #[test]
    fn net_asym_leaf_is_zero_sum() {
        use crate::ai::macro_agent::MacroLeaf;
        let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
        let game = generated_game(3);
        let a = leaf_value(&evaluator, MacroLeaf::NetAsym, &game.state, 1, 0, None);
        let b = leaf_value(&evaluator, MacroLeaf::NetAsym, &game.state, 2, 0, None);
        assert!(
            (a + b).abs() < 1e-6,
            "NetAsym leaf must be zero-sum: v(p1)={a} v(p2)={b} sum={}",
            a + b
        );
        // B2 paints the two halves DIFFERENTLY (mover gets the edge directive,
        // the other side its scripted goal), so the identity is worth pinning
        // again: it survives because both perspectives reuse the same pair of
        // numbers with the sign swapped, not because the paintings match.
        let edge = compute_macro_goal(&game.state, 1, 0);
        let from = Some((1 as PlayerId, &edge));
        let c = leaf_value(&evaluator, MacroLeaf::NetAsymPaint, &game.state, 1, 0, from);
        let d = leaf_value(&evaluator, MacroLeaf::NetAsymPaint, &game.state, 2, 0, from);
        assert!(
            (c + d).abs() < 1e-6,
            "NetAsymPaint leaf must stay zero-sum: v(p1)={c} v(p2)={d} sum={}",
            c + d
        );
    }

    /// Per-unit-goal design (Aug 2026), Step 3 verification: extends
    /// `macro_exec::executor_is_deterministic`'s pattern to the full agent
    /// with the `UnitGoalStore` wired in (`reconcile_unit_goals` +
    /// `Some(&self.unit_goals)` in `select_move`). Two independent, real
    /// two-player games from the same seed must produce byte-identical
    /// `_history` -- correctness now additionally depends on
    /// `tribe.units`' Vec order staying stable, a pre-existing invariant
    /// (see `executor_is_deterministic`) this is a new consumer of, not a
    /// new one.
    #[test]
    fn mcts_agent_with_unit_goals_is_deterministic() {
        for seed in 0..2i64 {
            let base = generated_game(seed);
            let mut histories = Vec::new();
            for _ in 0..2 {
                let mut game = base.clone();
                let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
                let params = MacroParams { sims: 8, ..Default::default() };
                let mut agent1 = MacroMctsAgent::new(&evaluator, params.clone());
                let mut agent2 = MacroMctsAgent::new(&evaluator, params);
                for _ in 0..40 {
                    if game.state.settings._game_over {
                        break;
                    }
                    let pid = game.state.settings.current_player_turn_id;
                    let agent = if pid == 1 { &mut agent1 } else { &mut agent2 };
                    let Some(m) = agent.select_move(&mut game) else { break };
                    game.play_move(m.as_ref());
                }
                histories.push(game.state._history.clone());
            }
            assert_eq!(
                histories[0], histories[1],
                "seed {seed}: two MacroMctsAgent (UnitGoalStore wired in) runs diverged"
            );
        }
    }

    #[test]
    fn mcts_agent_returns_true_legal_move() {
        for seed in 0..2i64 {
            let mut game = generated_game(seed);
            let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
            let mut agent =
                MacroMctsAgent::new(&evaluator, MacroParams { sims: 16, ..Default::default() });
            let m = agent.select_move(&mut game).unwrap();
            let legal: Vec<String> =
                game.legal_moves().iter().map(|x| x.serialize().to_string()).collect();
            assert!(legal.contains(&m.serialize().to_string()), "seed {seed}: true-illegal move");
        }
    }

    /// The macro agent must reach `Brain` with the params it was configured
    /// with. Regression for a silently-live bug: `Brain::think` hard-coded
    /// `None`, so every MACRO_GEN training round ran the HEURISTIC leaf no
    /// matter what the run intended (and reported no root value, zeroing the
    /// TD bootstrap).
    #[test]
    fn macro_params_reach_the_macro_agent_through_brain() {
        use crate::ai::brain::{Brain, SearchBackend};
        let mut game = generated_game(0);
        let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
        let mut brain = Brain::with_backend(&evaluator, 8, SearchBackend::MacroMcts)
            .with_macro_params(MacroParams { sims: 5, leaf: MacroLeaf::Net, ..Default::default() });
        let (mv, _) = brain.think_decomposed(&game, 0);
        assert!(mv.is_some());
        // A net leaf reports a root value; the heuristic leaf must not (its Q
        // is an evaluate_state number and would train the head toward the
        // evaluator it exists to beat).
        assert!(brain.last_root_value().is_some(), "net leaf must expose a root value");

        let mut heur = Brain::with_backend(&evaluator, 8, SearchBackend::MacroMcts)
            .with_macro_params(MacroParams { sims: 5, ..Default::default() });
        let _ = heur.think_decomposed(&game, 0);
        assert!(heur.last_root_value().is_none(), "heuristic leaf must report no root value");
        let _ = &mut game;
    }

    /// EXP_ELO_033 smoke probe (manual): advances each game to mid-game with
    /// the executor, then prints per-edge root Q/visits so tuning can tell
    /// genuine directive ties from an exploration term swamping the signal.
    /// Run: cargo test --lib ai::macro_mcts -- --ignored --nocapture
    #[test]
    #[ignore]
    fn smoke_stats_probe() {
        for seed in 0..8i64 {
            let game = generated_game(seed);
            let pov = game.state.settings.current_player_turn_id;
            let mut sim = game.clone_for_mcts(pov);
            let mut lane_states: [LaneState; 2] = Default::default();
            let mut counters = [TurnCounters::default(); 2];
            for _ in 0..8 {
                if sim.state.settings._game_over {
                    break;
                }
                let p = sim.state.settings.current_player_turn_id;
                let goal = compute_macro_goal(&sim.state, p, counters[seat(p)].tier3_bought);
                if !macro_exec::execute_turn(
                    &mut sim,
                    p,
                    &goal,
                    &mut lane_states[seat(p)],
                    &mut counters[seat(p)],
                    1.0,
                ) {
                    break;
                }
            }
            if sim.state.settings._game_over {
                continue;
            }
            let pov = sim.state.settings.current_player_turn_id;
            let base = compute_macro_goal(&sim.state, pov, counters[seat(pov)].tier3_bought);
            let cands =
                enumerate_candidates(&sim.state, pov, base, counters[seat(pov)], 4);
            let k = cands.len();
            let sims = std::env::var("PROBE_SIMS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32);
            let t0 = std::time::Instant::now();
            let evaluator = Evaluator::Dummy(DummyEvalHandle::new());
            let (pick, stats) = MacroMctsSearch::run_probed(
                &sim,
                pov,
                cands,
                counters[seat(pov)],
                &lane_states[seat(pov)],
                &MacroParams { sims, ..Default::default() },
                &evaluator,
            );
            println!(
                "seed {seed} t{}: k={k} pick={pick} nodes={} depth={} share={:.2} ({}ms)",
                sim.state.settings.turn,
                stats.nodes,
                stats.max_depth,
                stats.root_visit_max_share,
                t0.elapsed().as_millis()
            );
        }
    }

    #[test]
    fn derived_counters_track_discovered_techs() {
        let game = generated_game(0);
        for pid in [1, 2] {
            let c = derive_counters(&game.state, pid);
            assert_eq!(c.techs_bought, 0, "fresh game should derive 0 bought techs");
            assert_eq!(c.tier3_bought, 0);
        }
    }

    /// War-room item 3: `decode_macro_prior` should rank a candidate whose
    /// stance and orders match the head's high-probability picks above one
    /// that doesn't, and always return a valid distribution (sums to 1).
    #[test]
    fn decode_macro_prior_favors_matching_stance_and_orders() {
        let map_size = 3; // board = 9, keep the order maps small and readable
        let board = map_size * map_size;
        let mut stance_probs = vec![0.1f32; 4];
        stance_probs[Stance::Arm as usize] = 0.7; // head strongly prefers Arm
        let mut order_maps = vec![0.05f32; 3 * board];
        order_maps[OrderKind::Attack as usize * board + 4] = 0.9; // Attack@tile4 strongly preferred

        let matching = MacroGoal {
            orders: vec![(OrderKind::Attack, 4)],
            stance: Stance::Arm,
            save_target: None,
        };
        let mismatched = MacroGoal {
            orders: vec![(OrderKind::Defend, 0)],
            stance: Stance::Grow,
            save_target: None,
        };
        let candidates = vec![matching, mismatched];

        let prior = decode_macro_prior(&stance_probs, &order_maps, &candidates, map_size);
        assert_eq!(prior.len(), 2);
        assert!(
            (prior.iter().sum::<f32>() - 1.0).abs() < 1e-5,
            "prior must sum to 1, got {prior:?}"
        );
        assert!(
            prior[0] > prior[1],
            "the stance+order-matching candidate should outrank the mismatched one: {prior:?}"
        );
    }

    /// Degenerate input (all-zero maps) must fall back to uniform, not NaN
    /// or a divide-by-zero panic -- `select_edge`'s cold-start path does a
    /// `max_by` over this vector and a NaN would silently wreck the pick.
    #[test]
    fn decode_macro_prior_degenerate_input_is_uniform_not_nan() {
        let map_size = 3;
        let board = map_size * map_size;
        let stance_probs = vec![0.0f32; 4];
        let order_maps = vec![0.0f32; 3 * board];
        let candidates = vec![MacroGoal::default(), MacroGoal::default(), MacroGoal::default()];
        let prior = decode_macro_prior(&stance_probs, &order_maps, &candidates, map_size);
        for p in prior {
            assert!((p - 1.0 / 3.0).abs() < 1e-6, "expected ~1/3, got {p}");
        }
    }

    fn bare_node(game: &Game, root_turn: i32, candidates: usize) -> Node {
        let heur = |s: &crate::states::GameState, p: PlayerId, _t3: u32| crate::ai::evaluate_state(s, p);
        let mut n = Node::new(
            game.clone(),
            1,
            [TurnCounters::default(); 2],
            Default::default(),
            root_turn,
            candidates,
            None,
            &heur,
        );
        n.candidates = vec![MacroGoal::default(); candidates];
        n.children = vec![None; candidates];
        n.edge_visits = vec![0.0; candidates];
        n.edge_values = vec![0.0; candidates];
        n
    }

    /// With `edge_prior` empty (the default, and the case whenever
    /// `root_prior_w == 0.0`), `select_edge` must behave exactly as before
    /// this change: unvisited edges in list order, then plain UCT.
    #[test]
    fn select_edge_unchanged_when_prior_empty() {
        let game = generated_game(1);
        let root_turn = game.state.settings.turn;
        let mut n = bare_node(&game, root_turn, 3);
        assert_eq!(n.select_edge(), 0, "first unvisited edge, list order");

        n.edge_visits = vec![5.0, 5.0, 5.0];
        n.edge_values = vec![1.0, 3.0, 2.0]; // edge 1 has the best mean Q
        n.visits = 15.0;
        assert_eq!(n.select_edge(), 1, "highest-Q edge once all are visited, prior absent");
    }

    /// Cold-start stays list-order even with `edge_prior` populated — the
    /// prior only acts once every edge has a visit (see `select_edge`'s doc
    /// comment: a first version let the prior reorder cold-start too, which
    /// turned out to be a hard on/off switch rather than a dial, since
    /// argmax(w*p) doesn't depend on w for any w>0).
    #[test]
    fn select_edge_cold_start_ignores_prior() {
        let game = generated_game(1);
        let root_turn = game.state.settings.turn;
        let mut n = bare_node(&game, root_turn, 3);
        n.edge_prior = vec![0.1, 0.7, 0.2]; // edge 1 is the prior's clear favorite
        assert_eq!(n.select_edge(), 0, "cold-start picks list order regardless of prior");
    }

    /// Once every edge has visits, a strong prior on a mediocre-Q edge
    /// should be able to overturn the plain-UCT pick -- otherwise the
    /// prior would only ever affect cold start and never bias search
    /// toward the head's judgment during actual exploration.
    #[test]
    fn select_edge_prior_can_overturn_plain_uct_pick() {
        let game = generated_game(1);
        let root_turn = game.state.settings.turn;
        let mut n = bare_node(&game, root_turn, 2);
        n.edge_visits = vec![5.0, 5.0];
        n.edge_values = vec![1.0, 0.8]; // edge 0 wins on Q alone
        n.visits = 10.0;
        assert_eq!(n.select_edge(), 0, "sanity: edge 0 wins without a prior");

        n.edge_prior = vec![0.0, 5.0]; // large weight, all on edge 1
        assert_eq!(n.select_edge(), 1, "a strong prior should be able to flip the pick");
    }
}
