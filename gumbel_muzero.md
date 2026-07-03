Gumbel MuZero Search Rewrite — Implementation Plan

0. Ground truth confirmed this session

- PolyZeroNet::forward() is literally forward_t(map, player, false) (polyfish-rs/src/ai/network.rs:296-302) — so bug #7 in the brief (Gumbel using forward instead of forward_t(..., false)) is cosmetically inconsistent but not a correctness bug. Still worth fixing for uniformity when the file is rewritten anyway.
- Brain (polyfish-rs/src/ai/brain.rs) has no backend field; think is hard-typed to return Option<ZeroMctsAgent<'_>>; arena.rs and self_play.rs both construct agents directly/via Brain::new, no backend switch exists anywhere.
- MoveVisit.visits: f32 (polyfish-rs/src/ai/mcts_types.rs) is consumed only by DecomposedMapper::move_visit_to_targets (polyfish-rs/src/ai/mapper.rs:128-156), which reads only move-identity fields, never visits itself — confirmed fully decoupled, safe to populate with any nonnegative weight.
- self_play.rs's visit-aggregation loop (polyfish-rs/src/bin/self_play.rs ~178-228) sums mv.visits per decomposed target index and normalizes by total_visits — confirmed backend-agnostic.
- policy_composer.rs exposes both compute_move_priors(...) -> Vec<f32> (normalized probs, softmax per head) and compute_move_log_probs(...) -> Vec<f32> (log-probs, log-softmax per head) — Gumbel needs raw logits summed across heads pre-softmax for its transform; need to verify/decide whether compute_move_log_probs gives combined per-move log-probability (sum of head log-probs) suitable as logit(a), or whether a new compute_move_logits is needed that returns unnormalized/pre-softmax scores. This must be checked in Step (a) below — if compute_move_log_probs already sums per-head log-softmax outputs (which behave like a valid joint log-probability, itself usable as "logit" in the Gumbel-top-k sense per the paper, since Gumbel-max works with any monotonic transform of the categorical log-probabilities), it can be reused as-is; no need for a separate raw-logit head extraction. This is a scoping question for the implementer, not fully resolved by the read-only exploration I could do.
- CLAUDE.md (/Users/verdi/Development/Polyfish/CLAUDE.md) confirms the dual-network-sync constraint is about network.rs / train.py byte-compatibility (layer shapes, channel counts, head sizes) — this Gumbel rewrite is purely a search-time consumer of existing PolicyOutput/ValueOutput; it does not touch network.rs, features.rs, mapper.rs constants, or train.py. Confirmed zero risk to the sync constraint, and this should be stated explicitly in the PR description of every step below so reviewers don't go looking for a Python-side change.

---
1. Module structure

Create polyfish-rs/src/ai/mcts_common.rs, and add pub mod mcts_common; next to the existing pub mod mcts_zero; / pub mod gumbel_mcts; in polyfish-rs/src/ai/mod.rs.

Move here (shared, byte-identical behavior for both agents):
- pub(crate) struct LeafData { path_indices: Vec<usize>, path_players: Vec<i32>, features: Option<GameFeatures>, legal_moves: RefCell<Vec<Box<dyn Move>>>, map_size: usize, terminal_value: Option<f32> } — structurally identical in both files today; unify into one definition.
- pub(crate) fn compute_terminal_outcome(game: &Game) -> f32 — extract the score-comparison block duplicated in mcts_zero.rs:614-634 and gumbel_mcts.rs:638-658 verbatim.
- pub(crate) fn extract_leaf_data(game: &Game, device: &Device, indices_stack: Vec<usize>, path_players: Vec<i32>, needs_expansion: bool) -> LeafData — the "at the leaf, before undo" extraction block (terminal / needs_expansion / horizon three-way branch), currently duplicated near-verbatim in both select_and_extract_leaf implementations.
- pub(crate) trait BackpropNode { fn visits_mut(&mut self) -> &mut f32; fn value_sum_mut(&mut self) -> &mut f32; fn virtual_loss(&self) -> &RefCell<f32>; fn children_mut(&mut self) -> &mut [Self] where Self: Sized; } plus one free function pub(crate) fn backpropagate_and_remove_virtual_loss<N: BackpropNode>(root: &mut N, indices: &[usize], path_players: &[i32], virtual_loss_amount: f32, value: f32). This captures the player-aware sign-flip fix exactly once. ZeroNode and GumbelNode each implement BackpropNode with trivial field accessors. This is the single highest-value unification: right now the sign-flip logic (the "recent correctness fix" called out as load-bearing in the brief) is duplicated byte-for-byte in mcts_zero.rs:725-778 and gumbel_mcts.rs:724-778 — a future fix to one and not the other is a real risk today.
- pub(crate) fn get_node_by_path<'b, N>(root: &'b N, indices: &[usize]) -> Option<&'b N> / get_node_by_path_mut — generic over any node type with a children()/children_mut() accessor (via a small TreeNode trait, or just duplicate — these are ~6 lines each, low priority to unify vs. the backprop fn).
- pub(crate) const VIRTUAL_LOSS: f32 = 1.0; and pub(crate) const DEFAULT_BATCH_SIZE: usize = 24; — currently magic-numbered separately in each file (Zero's virtual_loss: 1.0 field vs Gumbel's hardcoded local let virtual_loss = 1.0;).

Do NOT unify (stay Gumbel-specific in the rewritten gumbel_mcts.rs):
- The σ/completed-Q transform (sigma_completed_q, compute_v_mix) — Zero has no analogue.
- The round-robin Sequential-Halving root controller — structurally nothing like Zero's flat PUCT loop.
- Interior (non-root) selection formula (softmax(logit + sigma(Q)) then probs(a) - visits(a)/(1+sum visits)) — different from Zero's PUCT.
- Policy-target extraction over the full legal set via π' — Zero's target is raw visit counts, structurally simpler.
- GumbelNode itself (extra logit/gumbel fields vs Zero's prior) — do not try to force a shared Node struct; the field-set divergence (prior vs logit+gumbel, plus different selection-time bookkeeping needs) makes a shared struct more confusing than two structs sharing helper functions/traits. Zero's expand_node_single/expand_node_from_network_output/expand_node_from_precomputed are prior-normalization-specific (they call compute_move_priors and normalize to sum-1) — Gumbel's expansion instead needs raw logits per child with no normalization, so these three functions are not directly reusable; instead write Gumbel-specific expand_gumbel_node_from_precomputed(node: &mut GumbelNode, legal_moves, map_size, policy: &PolicyOutput, sample_gumbel: bool) in gumbel_mcts.rs that mirrors their shape (same signature pattern, same is_expanded-guard-and-early-return-on-empty-legal-moves logic) but calls compute_move_log_probs instead of compute_move_priors and skips the sum-normalization step. Keeping this Gumbel-local avoids polluting the shared module with Zero's prior-specific normalization assumption.

Where the σ/completed-Q code lives: new private submodule inside gumbel_mcts.rs, e.g. a mod qtransform { ... } block or a sibling file polyfish-rs/src/ai/gumbel_qtransform.rs if the implementer wants it independently unit-testable without touching search internals (recommended — see Testing plan, Step (a) needs to unit test this in total isolation from Game/Move/tree machinery, so a standalone file with pure f32-slice-in/out functions is cleaner than embedding it as a private submodule of gumbel_mcts.rs). Recommendation: polyfish-rs/src/ai/gumbel_qtransform.rs, pub(crate) visibility, pub mod gumbel_qtransform; added to ai/mod.rs.

---
2. Exact signatures

2.1 gumbel_qtransform.rs (new file, pure functions, no Game/Move/tree dependency — critical for isolated unit testing)

/// Constants from mctx qtransform_completed_by_mix_value defaults.
pub const C_VISIT: f32 = 50.0;      // maxvisit_init
pub const C_SCALE: f32 = 0.1;       // value_scale
pub const EPSILON: f32 = 1e-8;

/// v_mix at a node: imputed value used for Q of unvisited children.
/// `raw_value` = this node's own NN value prediction (this node's perspective).
/// `child_visit_counts[i]`, `child_priors[i]`, `child_qvalues[i]` all same length,
/// one entry per child of this node. `child_qvalues[i]` is meaningless/unused when
/// `child_visit_counts[i] == 0.0` (not read in that branch).
pub fn compute_v_mix(
    raw_value: f32,
    child_priors: &[f32],
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
) -> f32 {
    let sum_visits: f32 = child_visit_counts.iter().sum();
    if sum_visits == 0.0 {
        return raw_value;
    }
    let sum_probs_visited: f32 = child_priors.iter().zip(child_visit_counts)
        .filter(|(_, &v)| v > 0.0)
        .map(|(p, _)| p)
        .sum();
    let weighted_q: f32 = if sum_probs_visited > EPSILON {
        child_priors.iter().zip(child_qvalues).zip(child_visit_counts)
            .filter(|((_, _), &v)| v > 0.0)
            .map(|((p, q), _)| p * q / sum_probs_visited)
            .sum()
    } else {
        0.0
    };
    (raw_value + sum_visits * weighted_q) / (sum_visits + 1.0)
}

/// Completed Q per child: real Q if visited, else v_mix.
pub fn compute_completed_qvalues(
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
    v_mix: f32,
) -> Vec<f32> {
    child_qvalues.iter().zip(child_visit_counts)
        .map(|(&q, &v)| if v > 0.0 { q } else { v_mix })
        .collect()
}

/// Optional min-max rescale to [0,1] over the given values (only visited-child Q's
/// per mctx semantics — but simplest/matches mctx close enough to rescale the full
/// completed_qvalues vector post-imputation; degenerate min==max -> all zeros).
pub fn rescale_min_max(values: &[f32]) -> Vec<f32> {
    let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() < EPSILON {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&v| (v - min) / (max - min)).collect()
}

/// sigma(completed_Q) = (C_VISIT + max_visit) * C_SCALE * completed_Q
pub fn sigma(completed_q: &[f32], max_child_visits: f32) -> Vec<f32> {
    let scale = (C_VISIT + max_child_visits) * C_SCALE;
    completed_q.iter().map(|&q| scale * q).collect()
}

/// Convenience: full pipeline used by both root-recommendation and interior selection.
/// rescale=true reproduces mctx's default `rescale_values=True`.
pub fn sigma_completed_q(
    raw_value: f32,
    child_priors: &[f32],
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
    rescale: bool,
) -> Vec<f32> {
    let v_mix = compute_v_mix(raw_value, child_priors, child_qvalues, child_visit_counts);
    let mut completed = compute_completed_qvalues(child_qvalues, child_visit_counts, v_mix);
    if rescale {
        completed = rescale_min_max(&completed);
    }
    let max_visits = child_visit_counts.iter().cloned().fold(0.0f32, f32::max);
    sigma(&completed, max_visits)
}

Note raw_value at the root is the root's own NN value prediction (captured at root expansion time and stored on GumbelNode/GumbelMctsAgent local state — Zero doesn't currently store this anywhere, Gumbel will need to). At interior nodes, raw_value is that node's own NN value prediction from when it was expanded — so GumbelNode needs a new field own_value: f32 populated at expansion time (from value_out.win_value for that node, same batched NN call that produces its children's logits).

2.2 gumbel_mcts.rs — GumbelNode

struct GumbelNode {
    visits: f32,
    value_sum: f32,
    logit: f32,
    gumbel: f32,        // 0.0 for non-root nodes (unchanged from today)
    own_value: f32,      // NEW: this node's own NN value prediction at expansion time (0.0 until expanded)
    children: Vec<GumbelNode>,
    move_to_here: Option<Box<dyn Move>>,
    is_expanded: bool,
    virtual_loss: RefCell<f32>,
}

impl GumbelNode {
    fn new(logit: f32, gumbel: f32, move_to_here: Option<Box<dyn Move>>) -> Self { ... }
    fn q_value(&self) -> f32 { ... } // unchanged: value_sum/visits, 0.0 if unvisited
}

2.3 GumbelMctsAgent constructor — fix param-not-field bug

pub struct GumbelMctsAgent<'a> {
    pub network: &'a PolyZeroNet,
    pub iterations: usize,
    pub k: usize,
    pub batch_size: usize,
    pub c_visit: f32,   // exposed for tuning/testing, default C_VISIT
    pub c_scale: f32,   // exposed for tuning/testing, default C_SCALE
}

impl<'a> GumbelMctsAgent<'a> {
    pub fn new(network: &'a PolyZeroNet, iterations: usize, k: usize) -> Self {
        Self { network, iterations, k, batch_size: mcts_common::DEFAULT_BATCH_SIZE,
               c_visit: gumbel_qtransform::C_VISIT, c_scale: gumbel_qtransform::C_SCALE }
    }
}

This directly fixes the brief's complaint that k is a post-construction field mutation — new() now takes it as a real parameter, matching how ZeroMctsAgent::new takes iterations.

2.4 Root sequential-halving controller (new, replaces the three near-duplicate select_move* bodies)

/// Runs the full root-level search: initial top-k Gumbel cut, then Sequential
/// Halving rounds with round-robin equal-visit allocation, batched leaf collection
/// per round-step. Returns the fully-populated root (all initially-sampled top-k
/// children present with their final visit counts; eliminated candidates keep
/// their partial visit counts, they are NOT removed from `root.children` -- see
/// 2.6, policy-target extraction needs them).
fn run_search(&self, game: &mut Game, root_value: f32, root: &mut GumbelNode, start_turn: i32) {
    let considered_visits_table = sequence_of_considered_visits(self.k, self.iterations);
    // considered_visits_table: Vec<(usize /*round idx*/, usize /*num_considered this round*/, usize /*new visits per candidate this round*/)>
    let mut num_considered = self.k.min(root.children.len());

    for (round_idx, round_considered, visits_this_round) in considered_visits_table {
        if round_considered <= 1 { break; }
        // Restrict "in play" set to the first `round_considered` children,
        // pre-sorted by current sigma(completed_Q)+logit+gumbel score (round-robin
        // fairness within the round is enforced by the batch-collection loop in 2.7,
        // NOT by this sort -- this sort only determines WHICH `round_considered`
        // survive into the round).
        self.rerank_root_children(root, round_idx); // no-op on round 0 (all k candidates already in initial order)
        self.run_round_robin_round(game, root, round_considered, visits_this_round, start_turn);
    }
}

sequence_of_considered_visits(k, num_simulations) -> Vec<(round_idx, num_considered, visits_per_candidate_this_round)> implements the mctx table exactly:

fn sequence_of_considered_visits(max_num_considered: usize, num_simulations: usize) -> Vec<(usize, usize, usize)> {
    if max_num_considered <= 1 {
        return vec![(0, max_num_considered.max(1), num_simulations)];
    }
    let log2max = (max_num_considered as f32).log2().ceil() as usize;
    let mut num_considered = max_num_considered;
    let mut rounds = Vec::new();
    let mut round_idx = 0;
    let mut total_assigned = 0usize;
    while total_assigned < num_simulations && num_considered >= 2 {
        let extra = ((num_simulations / (log2max.max(1) * num_considered)).max(1));
        rounds.push((round_idx, num_considered, extra));
        total_assigned += extra * num_considered;
        num_considered = (num_considered / 2).max(2);
        round_idx += 1;
        // mctx terminates the loop naturally because num_considered stabilizes at 2
        // and total_assigned keeps growing; guard against infinite loop if extra==0 is
        // impossible since extra is max(1,...), but add a round_idx cap as defensive belt:
        if round_idx > 2 * log2max + 4 { break; }
    }
    rounds
}

This must be unit-tested directly (see Testing plan) against hand-computed small cases, e.g. k=4, num_simulations=16 → rounds [(0,4,2),(1,2,6)] or similar (exact numbers to be verified against mctx's own test suite / worked by hand during implementation, not asserted here without running it — flag this as an implementer TODO: literally port mctx's get_sequence_of_considered_visits test cases as Rust unit tests, they're public and small).

2.5 Interior (non-root) selection formula

/// Only called for depth >= 1 (i.e., current node is NOT the root). No Gumbel term.
fn select_child_interior(&self, node: &GumbelNode) -> Option<usize> {
    let n = node.children.len();
    if n == 0 { return None; }
    let child_qvalues: Vec<f32> = node.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = node.children.iter().map(|c| c.effective_visits()).collect();
    let child_priors: Vec<f32> = node.children.iter().map(|c| c.logit).collect(); // NOTE: raw logits, not softmaxed -- softmax happens below
    let sigma_q = gumbel_qtransform::sigma_completed_q(
        node.own_value, &child_priors, &child_qvalues, &child_visits, true,
    );
    // probs = softmax(prior_logits + sigma_q)
    let combined: Vec<f32> = child_priors.iter().zip(&sigma_q).map(|(l, s)| l + s).collect();
    let probs = softmax(&combined); // new small helper, or reuse policy_composer's softmax_1d if it operates on Vec<f32> not just Tensor -- check and reuse if signature matches, else add a tiny local softmax(&[f32]) -> Vec<f32>
    let sum_visits: f32 = child_visits.iter().sum();
    (0..n).max_by(|&a, &b| {
        let score_a = probs[a] - child_visits[a] / (1.0 + sum_visits);
        let score_b = probs[b] - child_visits[b] / (1.0 + sum_visits);
        score_a.partial_cmp(&score_b).unwrap()
    })
}

This fully replaces select_child_in_tree's argmax(logit + effective_value) (bug #1 in the brief).

2.6 Root selection during search (per-simulation-step, within a round-robin round)

/// Picks the next child to descend into, restricted to the top `round_considered`
/// candidates (by current index order after rerank), masked so only candidates whose
/// visit count is exactly `target_visit_this_step` are eligible -- enforces round-robin.
fn select_root_child_for_step(
    &self, root: &GumbelNode, round_considered: usize, target_visit_this_step: f32,
) -> Option<usize> {
    let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = root.children.iter().map(|c| c.effective_visits()).collect();
    let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();
    let sigma_q = gumbel_qtransform::sigma_completed_q(
        root.own_value, &child_priors, &child_qvalues, &child_visits, true,
    );
    (0..round_considered.min(root.children.len()))
        .filter(|&i| (root.children[i].visits - target_visit_this_step).abs() < 0.5) // "considered_visit" gating
        .max_by(|&a, &b| {
            let score_a = root.children[a].gumbel + child_priors[a] + sigma_q[a];
            let score_b = root.children[b].gumbel + child_priors[b] + sigma_q[b];
            score_a.partial_cmp(&score_b).unwrap()
        })
}

2.7 Final move recommendation (after search loop ends)

fn recommend_final_move(&self, root: &GumbelNode) -> usize {
    let max_visit = root.children.iter().map(|c| c.visits).fold(0.0f32, f32::max);
    let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = root.children.iter().map(|c| c.visits).collect();
    let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();
    let sigma_q = gumbel_qtransform::sigma_completed_q(
        root.own_value, &child_priors, &child_qvalues, &child_visits, true,
    );
    root.children.iter().enumerate()
        .filter(|(i, c)| (c.visits - max_visit).abs() < 0.5)
        .max_by(|(a, ca), (b, cb)| {
            let sa = ca.gumbel + child_priors[*a] + sigma_q[*a];
            let sb = cb.gumbel + child_priors[*b] + sigma_q[*b];
            sa.partial_cmp(&sb).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

2.8 Public API — matching Zero's signature exactly

pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>>;

pub fn select_move_with_decomposed_visits(
    &self,
    game: &mut Game,
    move_count: usize,
) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>);

pub fn select_move_with_stats(&self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>);

All three share one internal core:

fn search_and_extract(&self, game: &mut Game) -> GumbelNode /* fully-populated root */ { ... }

select_move_with_decomposed_visits calls search_and_extract, then builds Vec<MoveVisit> via the π' formula (2.9 below), then picks the final action via recommend_final_move (2.7) — not via the π' weights and not via temperature-sampled WeightedIndex over visits.

Decision: do NOT apply Zero's TEMPERATURE_MOVE_THRESHOLD/WeightedIndex temperature-sampling logic to Gumbel. Justification: Zero's temperature sampling exists because Zero's exploration is entirely front-loaded into Dirichlet-noised priors at the root plus PUCT's exploration term — visit counts are the only signal carrying exploration diversity forward into move selection, so sampling by visits^(1/T) is how Zero injects per-game stochasticity for training data diversity. Gumbel's exploration mechanism is structurally different and already stochastic by construction: the root-level Gumbel(0,1) noise is freshly resampled every call to select_move_with_decomposed_visits (new rand::thread_rng() draw), so distinct self-play games already get distinct root candidate sets and distinct tie-breaks in the Sequential-Halving survival ranking and in recommend_final_move's masked-argmax — this alone provides the same "diverse early-game training data" property Zero gets from temperature sampling. Layering WeightedIndex sampling over visits on top would double up two different exploration mechanisms and, worse, would sample over the raw post-halving visit distribution, which (per the bug analysis) is exactly the object that must NOT be used as an action-selection distribution outside of recommend_final_move's specific masked-argmax semantics — an ordinary visits-proportional sample would frequently pick a candidate that Sequential Halving explicitly demoted for having a low completed-Q, undermining the entire point of Halving. Recommendation: bypass temperature sampling entirely for Gumbel; keep the move_count parameter only so the signature matches Zero's (needed for Brain::think_decomposed's call site) but it is unused inside Gumbel's implementation (mark with let _ = move_count; and a comment explaining why, so it doesn't look like a forgotten wire-up).

2.9 Policy target extraction (π') — see section 5 for the algorithm; signature:

fn extract_policy_targets(&self, root: &GumbelNode) -> Vec<crate::ai::mcts_types::MoveVisit>;

---
3. Batching implications — round-robin-aware Phase 1/2/3

Zero's parallel_search_batch (mcts_zero.rs:370-494) collects batch_size (24) leaves via repeated independent tree descents from the root, each descent picking whatever PUCT says is best given current (virtual-loss-adjusted) stats — there's no structural grouping by candidate. This doesn't map cleanly onto Sequential Halving's round-robin constraint (every considered candidate must receive exactly visits_this_round new visits before the round ends and re-ranking happens — mixing rounds together in one big batch would break the "equal visits per candidate per round" invariant that the algorithm's fairness guarantee depends on).

Design: batch within a round, across that round's round_considered candidate subtrees, one NN call per "wave."

fn run_round_robin_round(
    &self, game: &mut Game, root: &mut GumbelNode,
    round_considered: usize, visits_per_candidate: usize, start_turn: i32,
) {
    // visits_per_candidate leaves must be collected under EACH of the round_considered
    // candidates. We do this in "waves": each wave collects up to `batch_size` leaves
    // total, drawn round-robin across candidates (one leaf per candidate per pass through
    // the candidate list, cycling), so that virtual loss spreads evenly and a single NN
    // call at the end of the wave batches leaves from potentially many different
    // candidate subtrees at once -- this preserves Zero's batching win (one NN call per
    // ~24 leaves) while respecting round-robin fairness, since within a wave we visit
    // each candidate the same number of times (0 or 1, plus leftover distribution) before
    // moving to the next wave.
    let total_leaves_needed = round_considered * visits_per_candidate;
    let mut collected_per_candidate = vec![0usize; round_considered];
    let mut total_collected = 0;

    while total_collected < total_leaves_needed {
        let mut leaves: Vec<(usize /*candidate_idx*/, mcts_common::LeafData)> = Vec::with_capacity(self.batch_size);
        // Round-robin: walk candidates in order, skip any that already hit
        // visits_per_candidate this round, collect one leaf each, until batch_size
        // leaves collected or all candidates exhausted for this round.
        'wave: loop {
            let mut made_progress = false;
            for cand_idx in 0..round_considered {
                if leaves.len() >= self.batch_size { break 'wave; }
                if collected_per_candidate[cand_idx] >= visits_per_candidate { continue; }
                if let Some(leaf) = self.select_and_extract_leaf_under_candidate(root, cand_idx, game, /* turn_horizon */ ..., &device) {
                    leaves.push((cand_idx, leaf));
                    collected_per_candidate[cand_idx] += 1;
                    made_progress = true;
                }
            }
            if !made_progress { break 'wave; } // all candidates in this round hit their target or terminal/dead-ends
        }
        if leaves.is_empty() { break; }
        total_collected += leaves.len();

        // Phase 2: exactly Zero's batched-NN-call pattern, unchanged shape --
        // stack all leaves (regardless of which root candidate they descend from)
        // into ONE forward_t call.
        let values = self.batched_evaluate_and_expand(root, &leaves);

        // Phase 3: backprop each leaf via the shared mcts_common::backpropagate_and_remove_virtual_loss.
        for ((_, leaf), value) in leaves.iter().zip(values.iter()) {
            mcts_common::backpropagate_and_remove_virtual_loss(root, &leaf.path_indices, &leaf.path_players, mcts_common::VIRTUAL_LOSS, *value);
        }
    }
}

select_and_extract_leaf_under_candidate(root, cand_idx, ...) is a thin wrapper: it descends starting from root.children[cand_idx] (forcing indices_stack = vec![cand_idx] as the first entry) and below that point uses select_child_interior (2.5) at every subsequent level — i.e., the round-robin/Gumbel/σ selection logic (2.6) only ever fires at the first step (choosing which candidate's subtree to descend one level further into), and everything below depth 1 uses the interior formula. This matches the paper: Gumbel noise and root-only sequential halving apply strictly at the root; everything else in the tree uses the interior rule.

This design keeps NN-call batching (the actual performance-critical property of Zero's Phase 1/2/3 split) while making round-robin fairness a hard structural invariant (collected_per_candidate[cand_idx] >= visits_per_candidate gate) rather than a probabilistic byproduct of a shared score function, and avoids trying to force Zero's parallel_search_batch to also understand halving-round boundaries, which would make that function do two unrelated things.

Note on own_value: each root candidate subtree, when first expanded (its own is_expanded transition), gets its NN value prediction from the same batched forward_t call — store this into own_value on that node exactly like a normal leaf expansion, no special-casing needed (this is the same field/write both for the root itself, computed once before the round-robin loop starts, and for every interior node as it's expanded).

---
4. Wiring into Brain / self_play.rs

4.1 Enum vs trait object — recommend enum

Given Brain::think currently returns a concretely-typed Option<ZeroMctsAgent<'_>>, and both agents' relevant methods (select_move, select_move_with_decomposed_visits, select_move_with_stats) have matching signatures once §2.8 lands, a dyn Trait object is possible but adds an unnecessary vtable/allocation layer and an awkward trait definition (three methods, each taking &mut Game and returning owned Box<dyn Move> — fine, but there's no polymorphic call site that needs runtime type erasure beyond a single match at construction time). Recommend a plain enum wrapping the two constructed agents, matched once per think_decomposed/think_with_stats call:

// polyfish-rs/src/ai/brain.rs
use crate::ai::gumbel_mcts::GumbelMctsAgent;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchBackend { Zero, Gumbel { k: usize } }

impl Default for SearchBackend {
    fn default() -> Self { SearchBackend::Zero }
}

pub struct Brain<'a> {
    pub network: &'a PolyZeroNet,
    pub max_iterations: usize,
    pub backend: SearchBackend,   // NEW field
}

enum SearchAgent<'a> {
    Zero(ZeroMctsAgent<'a>),
    Gumbel(GumbelMctsAgent<'a>),
}

impl<'a> SearchAgent<'a> {
    fn select_move_with_decomposed_visits(&self, game: &mut Game, move_count: usize)
        -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_decomposed_visits(game, move_count),
            SearchAgent::Gumbel(a) => a.select_move_with_decomposed_visits(game, move_count),
        }
    }
    fn select_move_with_stats(&self, game: &mut Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        match self {
            SearchAgent::Zero(a) => a.select_move_with_stats(game),
            SearchAgent::Gumbel(a) => a.select_move_with_stats(game),
        }
    }
}

impl<'a> Brain<'a> {
    pub fn new(network: &'a PolyZeroNet, max_iterations: usize) -> Self {
        Self { network, max_iterations, backend: SearchBackend::default() }
    }
    pub fn with_backend(network: &'a PolyZeroNet, max_iterations: usize, backend: SearchBackend) -> Self {
        Self { network, max_iterations, backend }
    }

    fn think(&'_ self, game: &Game) -> (Option<SearchAgent<'_>>, Vec<Box<dyn Move>>) {
        let moves = generate_legal_moves(&game.state);
        if moves.len() == 1 { return (None, moves); }
        let agent = match self.backend {
            SearchBackend::Zero => SearchAgent::Zero(ZeroMctsAgent::new(self.network, self.max_iterations)),
            SearchBackend::Gumbel { k } => SearchAgent::Gumbel(GumbelMctsAgent::new(self.network, self.max_iterations, k)),
        };
        (Some(agent), moves)
    }

    pub fn think_decomposed(&self, game: &Game, move_count: usize) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        let (agent, mut moves) = self.think(game);
        if agent.is_none() { return (moves.pop(), Vec::new()); }
        agent.unwrap().select_move_with_decomposed_visits(&mut game.clone(), move_count)
    }

    pub fn think_with_stats(&self, game: &Game) -> (Option<Box<dyn Move>>, Vec<f32>) {
        let (agent, mut moves) = self.think(game);
        if agent.is_none() { return (moves.pop(), Vec::new()); }
        agent.unwrap().select_move_with_stats(&mut game.clone())
    }
}

This keeps Brain::new backward-compatible (existing call sites Brain::new(network1, mcts_iters) in self_play.rs:120-121 keep compiling unchanged, defaulting to SearchBackend::Zero — zero behavior change for existing runs, directly satisfying the sequencing goal in §7).

4.2 self_play.rs — new CLI flags

// in the Args clap struct, alongside existing fields:

/// Search backend to use for MCTS.
#[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
search_backend: SearchBackendArg,

/// Gumbel: number of initial top-k candidates sampled at the root (only used
/// when --search-backend gumbel).
#[arg(long, default_value_t = 16)]
gumbel_k: usize,

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum SearchBackendArg { Zero, Gumbel }

Construction site (play_single_game, currently Brain::new(network1, mcts_iters) at lines ~120-121) becomes:

let backend = match args.search_backend {
    SearchBackendArg::Zero => SearchBackend::Zero,
    SearchBackendArg::Gumbel => SearchBackend::Gumbel { k: args.gumbel_k },
};
let agent1 = Brain::with_backend(network1, mcts_iters, backend);
let agent2 = Brain::with_backend(network2, mcts_iters, backend);

play_single_game's signature needs backend: SearchBackend (or the two already-constructed Brains) threaded through from main's per-game closure — currently mcts_iters etc. are passed as plain args into that function (per the brief's citation of lines ~120-121, 170); add search_backend: SearchBackend as one more parameter alongside mcts_iters.

4.3 arena.rs — future Zero-vs-Gumbel validation

arena.rs currently hardcodes ZeroMctsAgent::new(net1, mcts) / ZeroMctsAgent::new(net2, mcts) (lines 68-69) with no Brain involved at all — it calls the agent's select_move directly. To support Zero-vs-Gumbel matches, add mirrored flags:

#[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
backend1: SearchBackendArg,
#[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
backend2: SearchBackendArg,
#[arg(long, default_value_t = 16)]
gumbel_k: usize,

and replace the two hardcoded ZeroMctsAgent::new(...) constructions with a small local fn make_agent(backend: SearchBackendArg, net: &PolyZeroNet, mcts: usize, k: usize) -> SearchAgent returning the same SearchAgent enum from brain.rs (re-exported as pub so arena.rs can use it directly, bypassing Brain's book/decomposed-visits machinery since arena.rs only needs plain select_move). This is deferred to Step (d) in the sequencing plan (§7) — not needed until validation time, and not part of the core rewrite PRs.

---
5. Policy target extraction (π')

Confirmed: this only needs a single root-level v_mix computation once search ends — not per-round. The formula π'(a) ∝ exp(logit(a) + sigma(completed_Q(a))) is evaluated once, over the full legal move set at the root (not just the post-halving-survivor set currently in root.children after truncation — critically, this rewrite must stop truncating root.children; see below).

5.1 Structural change: never truncate root.children

The current buggy code (gumbel_mcts.rs:181-183, 278) calls root.children.truncate(kept_count) after each halving round, physically discarding eliminated candidates — this is bug #4 in the brief and the direct cause of the "policy target support collapses to ~1-2 actions" problem. The rewrite must never truncate; instead, sequential halving tracks an active: Vec<bool> mask (or just uses round_considered as a prefix-length index into a root.children list that stays in ranking-order but retains all k initially-sampled candidates for the whole search) alongside the visit-count-based round-robin gating in §2.6. All k initially top-k-cut candidates remain present in root.children at the end of search, each with whatever partial visit count they accumulated before being dropped out of "considered" status.

5.2 Two categories of un/under-visited actions requiring v_mix imputation

1. Legal moves outside the initial top-k Gumbel cut — never became a GumbelNode at all (the root_candidates.truncate(self.k) step at root construction, kept from the current code, still applies — only the top-k by logit+gumbel become tree nodes; this part of the existing logic is correct per the paper, Gumbel-top-k selection is intentional). These need to be represented in the π' output but have no GumbelNode/visit-count/own-Q at all.
2. In-cut candidates eliminated by halving before exhausting the sim budget — these do have a GumbelNode with visits > 0 (assuming round-robin gave them at least their round-0 allocation) but stopped receiving new visits after being dropped from "considered" status; per §2.6/§2.9's v_mix formula, since visits > 0 for these, they use their real accumulated Q, not v_mix — only category-1 actions need imputation via v_mix.

5.3 Algorithm

fn extract_policy_targets(&self, root: &GumbelNode, all_legal_moves_at_root: &[Box<dyn Move>], root_map_size: usize) -> Vec<MoveVisit> {
    // root.children: the k candidates that were Gumbel-top-k-selected and searched
    //   (all present, none truncated, per 5.1).
    // all_legal_moves_at_root: the FULL legal move list at the root (captured once,
    //   before the initial top-k cut, at root expansion time -- this must be stored
    //   or passed through from wherever select_move_with_decomposed_visits first
    //   calls game.legal_moves()).

    let child_qvalues: Vec<f32> = root.children.iter().map(|c| c.q_value()).collect();
    let child_visits: Vec<f32> = root.children.iter().map(|c| c.visits).collect();
    let child_priors: Vec<f32> = root.children.iter().map(|c| c.logit).collect();

    // v_mix computed ONCE, using only the k searched candidates (matches mctx: v_mix
    // is defined over the node's own children, which for the root IS the k-cut set --
    // moves outside the k-cut never had a node, so they can't contribute to v_mix's
    // weighted-Q sum; they are exactly the actions v_mix exists to impute FOR).
    let v_mix = gumbel_qtransform::compute_v_mix(root.own_value, &child_priors, &child_qvalues, &child_visits);

    // Completed Q for all k in-cut candidates (real Q if visited, v_mix if visits==0
    // -- can happen if round-robin allocation rounds down to 0 for some candidate in
    // a very-low-budget search, e.g. iterations < k).
    let completed_q_in_cut = gumbel_qtransform::compute_completed_qvalues(&child_qvalues, &child_visits, v_mix);

    // Build a lookup: move-identity -> (logit, completed_q) for in-cut moves.
    // For out-of-cut legal moves, completed_q = v_mix, logit = their own raw logit
    // (must be computed for ALL legal moves at root expansion time, not just the
    // top-k survivors -- i.e., logits for the full legal set must be retained, not
    // discarded after the top-k truncation happens today).

    let mut targets = Vec::with_capacity(all_legal_moves_at_root.len());
    let mut raw_scores = Vec::with_capacity(all_legal_moves_at_root.len());
    for m in all_legal_moves_at_root {
        let (logit, completed_q) = match find_in_cut_match(root, m) {
            Some(idx) => (child_priors[idx], completed_q_in_cut[idx]),
            None => (lookup_full_legal_logit(m), v_mix), // logit must come from the
                // SAME initial compute_move_log_probs call that produced child_priors,
                // evaluated over the full legal set before truncation -- store this
                // Vec<f32> alongside the initial top-k selection step instead of
                // discarding it.
        };
        raw_scores.push(logit + gumbel_qtransform::sigma(&[completed_q], child_visits.iter().cloned().fold(0.0,f32::max))[0]);
    }
    let probs = softmax(&raw_scores); // pi'(a)

    for (m, &p) in all_legal_moves_at_root.iter().zip(probs.iter()) {
        targets.push(MoveVisit {
            move_type: m.move_type(),
            visits: p, // NOTE: field named `visits` but semantically pi'(a) here --
                       // confirmed safe since mapper.rs never reads `visits` itself
                       // (only move-identity fields), and self_play.rs's aggregation
                       // loop treats it as an opaque weight, sums + normalizes -- a
                       // second normalization over an already-normalized pi' is a
                       // no-op (sums to 1 already) so no double-counting risk.
            source_idx: m.source_idx().ok(),
            target_idx: m.target_idx().ok(),
            structure_type: m.structure_type().ok(),
            unit_type: m.unit_type().ok(),
            tech_type: m.tech_type().ok(),
            ability_type: m.ability_type().ok(),
        });
    }
    targets
}

Key implementation requirement this surfaces: the initial root expansion must retain logits for the full legal move set, not just the top-k slice, for the duration of the search (small Vec<(Box<dyn Move>, f32)> or keep the untruncated Vec<GumbelNode> around separately from the truncated root.children actually-searched list — cheapest approach: don't truncate a Vec<GumbelNode> at all; instead keep a full_root_candidates: Vec<GumbelNode> of length = legal-move-count, and a separate Vec<usize> in_cut_indices marking which entries got promoted into the "searched" set, so root.children conceptually still means "the k that get tree-descended into" via an index-filtered view, while the π' extraction step at the end iterates the full untruncated list). This is a real structural decision the implementer needs to make explicitly — recommend: root.children holds ALL legal moves' GumbelNodes (full legal-move count, no k-truncation of the storage at all); a separate Vec<usize> in_cut: Vec<usize> (length k, sorted by initial logit+gumbel, values are indices into root.children) drives §2.6/§3's root-selection/round-robin logic, restricted to in_cut. This avoids the two-list-reconciliation problem above entirely and is a cleaner data model than the current code's destructive truncate-based approach.

find_in_cut_match/two-list-lookup complexity above disappears entirely under this revised data model — §5.3 simplifies to: iterate root.children (now = full legal move set), for i in in_cut use real Q, for i not in in_cut use v_mix, done, no matching-by-move-identity needed at all. Recommend updating §2's GumbelNode/root-construction to build root.children = full legal set immediately (this also changes §2.4's run_search signature slightly: it operates on in_cut indices instead of a physically-truncated root.children.len()), and softening the "batching implications" section (§3) to say leaf collection always indexes through in_cut[cand_idx] rather than cand_idx directly into root.children.

---
6. Testing plan

6.1 Pure-function unit tests (no Game, no network — fast, run every CI build)

In polyfish-rs/src/ai/gumbel_qtransform.rs (#[cfg(test)] mod tests):
- test_v_mix_no_visits_returns_raw_value — all child_visit_counts == 0 → v_mix == raw_value.
- test_v_mix_hand_computed — small 2-3 child example, hand-computed expected value per the formula in the brief; assert within 1e-5.
- test_completed_q_uses_real_q_when_visited — visited children keep their own Q untouched.
- test_completed_q_imputes_v_mix_when_unvisited.
- test_rescale_min_max_degenerate_single_visited_child — min==max case returns all-zeros (or whatever the chosen convention is — decide and assert, since mctx's own handling of this edge case should be checked directly rather than assumed; flag as an open question requiring an actual mctx source read for the degenerate branch before finalizing).
- test_sigma_matches_formula — (C_VISIT + max_visits) * C_SCALE * completed_q, hand-checked.

In a new polyfish-rs/src/ai/gumbel_mcts.rs (or mcts_common.rs) #[cfg(test)] module, or a new polyfish-rs/tests/test_sequential_halving.rs:
- test_sequence_of_considered_visits_k4 — port mctx's own small worked examples (k=4/k=8 with small num_simulations) as literal expected Vec<(usize,usize,usize)> outputs. This is the single most valuable correctness test since it's pure arithmetic with a known-correct reference implementation to port from.
- test_round_robin_allocation_is_equal_per_candidate — run run_round_robin_round against a stub/fake tree (not real Game) where every leaf immediately resolves to a fixed terminal value, and assert collected_per_candidate[i] == visits_per_candidate for all considered i after the round completes (this needs a test-only shortcut — likely easiest to test at the level of run_round_robin_round's bookkeeping in isolation from real Game/Move, using a minimal fake Game with a trivial always-terminal state, or by factoring the round-robin bookkeeping itself into a pure function testable without Game at all — recommend factoring collected_per_candidate tracking + wave-loop into a small pure state machine that's unit-testable independent of NN/Game, with the NN-eval/backprop wired around it only in the real agent).
- test_sequential_halving_toy_2_action — construct a toy scenario (2 candidates, deterministic/mocked Q feedback where one candidate is clearly better, e.g. via a stub evaluator returning +1.0 for all leaves under candidate A and -1.0 under candidate B) and assert the algorithm converges to selecting candidate A. This requires either (a) a lightweight fake PolyZeroNet/mock evaluation path, or (b) restructuring run_search so the "evaluate leaf" step is injectable (a closure/trait param) for testing — recommend (b), since it also benefits future Zero-side testability (the brief flags that neither agent currently has numeric-correctness tests for backprop, and Zero's parallel_search_batch has the same untestable-without-real-network problem).
- test_sequential_halving_toy_3_action — same idea with 3 candidates, one dominant.
- test_policy_target_sums_to_one — after a full search, extract_policy_targets's output weights sum to 1.0 within float tolerance, over the FULL legal move count (not just k).
- test_policy_target_covers_full_legal_set — assert targets.len() == all_legal_moves_at_root.len(), i.e. every legal move gets a nonzero-support entry (some may be near-zero probability but none should be silently dropped) — this is the direct regression test for bug #4.

6.2 Structural-invariant tests mirroring test_value_backprop.rs

New polyfish-rs/tests/test_gumbel_value_backprop.rs, porting test_value_backprop.rs's exact pattern (verified via read of that 206-line file's approach) but exercised against GumbelMctsAgent:
- Undo fully restores game.state.settings.current_player_turn_id after a full select_move_with_decomposed_visits call (compare before/after).
- Terminal-state score→outcome conversion (compute_terminal_outcome once unified into mcts_common.rs, testable once for both agents rather than twice).
- Player-sequence-tracking preconditions for the sign-flip logic (same as Zero's test, but exercised through Gumbel's round-robin leaf collection path instead of Zero's flat batch, to make sure the shared backpropagate_and_remove_virtual_loss fn behaves identically when called from Gumbel's calling pattern).
- New, genuinely missing today per the brief's gap-flag: an actual numeric-correctness test on backprop sign — construct a tiny 2-ply scenario with known player IDs at each depth and a known terminal value, call backpropagate_and_remove_virtual_loss directly (now unit-testable in isolation since it's a free function in mcts_common.rs taking a generic BackpropNode), and assert the exact expected value_sum at root and at each path node, including the flip-vs-no-flip cases. Do this once in mcts_common.rs's test module, covering both ZeroNode and GumbelNode (or a minimal TestNode stub) via the same test, since it's shared logic — no need to duplicate across both agent test files.

6.3 Smoke tests (replace/extend existing test_gumbel_mcts.rs)

Update the existing 39-line polyfish-rs/tests/test_gumbel_mcts.rs:
- Fix the struct-literal construction (GumbelMctsAgent { network, iterations, k, batch_size }) to use the new GumbelMctsAgent::new(&network, iterations, k) constructor (required anyway once k moves into new()'s params per §2.3 — this is a compile-break the rewrite must fix).
- Keep best_move.is_some() as a baseline smoke assertion, but add: select_move_with_decomposed_visits returns a Vec<MoveVisit> whose length equals the number of legal moves at that state (regression test for bug #4, at the integration level).
- Add a smoke test that runs 3-4 consecutive select_move_with_decomposed_visits calls on a real (small map) game loop end-to-end without panicking, exercising the round-robin batching path against the real Game/Move/network stack (catches integration bugs the pure unit tests can't).

6.4 arena.rs-based end-to-end validation (go/no-go signal)

Per the brief's explicit ask: before running any Zero-vs-Gumbel arena comparison, save a checkpoint of the current production model.safetensors as a fixed anchor (cp model.safetensors checkpoints/pre_gumbel_validation_<date>.safetensors or equivalent — this is a manual/CI step, not code). This directly addresses the gap the brief calls out (most recent training run had no pre-run checkpoint to compare against).

Validation run design:
cargo run --release --bin arena -- --model1 <anchor_checkpoint> --model2 <anchor_checkpoint> \
  --backend1 zero --backend2 gumbel --gumbel-k 16 --mcts 200 --games 100
(mcts=200 applied to backend1/Zero; Gumbel's --mcts/iterations flag should probably be split into --mcts1/--mcts2 if arena.rs's current single --mcts flag is reused for both sides — needed anyway once two different backends can have different natural iteration counts, e.g. Zero at 200 vs Gumbel at 32-64 per the brief's suggested comparison; add --mcts1/--mcts2 overriding a shared --mcts default, or two dedicated flags, as part of the same arena.rs change in §4.3).

Go/no-go: Gumbel at 32-64 simulations should roughly match Zero at 200 simulations in win rate (within normal noise band for a 100-game sample, e.g. win rate confidence interval overlapping ~50%, not a strict pass/fail threshold — recommend treating "not catastrophically worse" i.e. Gumbel win rate not below ~35-40% against Zero-at-200 as the practical bar, since Gumbel MuZero's whole selling point is achieving comparable quality at a fraction of the simulation budget, not necessarily beating a 3-6x-larger-budget baseline).

---
7. Sequencing (ordered, independently-verifiable PRs)

(a) Shared σ/completed-Q helpers, isolated, no wiring changes.
Add polyfish-rs/src/ai/gumbel_qtransform.rs with the pure functions from §2.1, plus the sequence_of_considered_visits table function (§2.4) — arguably belongs in the same file or a sibling gumbel_sequential_halving.rs, implementer's call — and their full unit test suite (§6.1). Nothing in gumbel_mcts.rs, brain.rs, or self_play.rs changes. Zero risk to any existing behavior; this PR is pure addition. Also extract mcts_common.rs's shared LeafData/backpropagate_and_remove_virtual_loss/compute_terminal_outcome in this same PR (or a preceding micro-PR) since it's needed by (b) and is itself independently verifiable — refactor mcts_zero.rs to use the shared fn, run existing Zero test suite (test_value_backprop.rs, all existing mcts_zero-touching integration tests) to confirm zero behavior change from the extraction. This sub-step (shared-module extraction) is the highest-leverage/lowest-risk first move and should land before anything Gumbel-specific, since it directly eliminates the "sign-flip fix duplicated in two places, can silently diverge" risk called out in the brief, independent of whether the rest of the Gumbel rewrite proceeds on schedule.

(b) Rewrite GumbelMctsAgent's search core.
Full rewrite of gumbel_mcts.rs: new GumbelNode (with own_value), GumbelMctsAgent::new(network, iterations, k), root candidate construction over the full legal set (§5.3's revised data model — no truncation), run_search/run_round_robin_round/batching per §3, interior selection per §2.5, root-during-search selection per §2.6, final recommendation per §2.7, and select_move/select_move_with_decomposed_visits(game, move_count)/select_move_with_stats per §2.8-2.9. Update test_gumbel_mcts.rs per §6.3, add test_gumbel_value_backprop.rs per §6.2, add the round-robin/toy-halving tests per §6.1. Brain, brain.rs, self_play.rs, arena.rs remain untouched in this PR — GumbelMctsAgent is fully rewritten and tested standalone via its own test files and direct construction (bypassing Brain), exactly as test_gumbel_mcts.rs does today. This isolates "did I implement Gumbel MuZero correctly" from "did I wire it in correctly" as two separate reviewable/revertable units.

(c) Wire into Brain/self_play.rs behind a flag defaulting to Zero.
Add SearchBackend enum, Brain::with_backend, SearchAgent dispatch enum per §4.1; add --search-backend/--gumbel-k flags to self_play.rs per §4.2, default zero — existing self_play invocations (and run_training_loop.sh, which doesn't pass --search-backend) get zero behavior change, since Brain::new (unchanged signature) still defaults to SearchBackend::Zero. This is the point at which gumbel_mcts.rs's previously-dead // use crate::ai::gumbel_mcts::GumbelMctsAgent; comment in brain.rs:1 gets resolved (module import now live via SearchAgent::Gumbel). Verify with a short manual self_play run using --search-backend gumbel --num-games 2 --mcts-iters 32 to confirm no panics end-to-end through the full self-play data pipeline (features → decomposed policy → .safetensors write), independent of whether the resulting policy targets are good, just that the pipeline doesn't break structurally.

(d) Arena validation.
Add --backend1/--backend2/--gumbel-k/(--mcts1/--mcts2 if needed) flags to arena.rs per §4.3, reusing the same SearchAgent enum (make it pub from brain.rs or move it to mcts_common.rs if arena.rs shouldn't depend on brain.rs's book/decomposed-visit-specific code — worth a quick look at whether arena.rs importing brain::SearchAgent pulls in unwanted deps, otherwise a thin arena-local re-implementation of the same 2-variant match is only ~10 lines and avoids coupling). Before running, save the checkpoint-anchor per §6.4. Run the Zero-at-200-vs-Gumbel-at-32/64 comparison; record win rate, average game length, and (if easy to add) average wall-clock time per move for both backends as a secondary "is Gumbel actually cheaper" signal (the entire practical motivation for Gumbel MuZero is fewer simulations for comparable strength — if the validation run doesn't also confirm the speed win, the rewrite hasn't achieved its purpose even if correctness is fine).

(e) Only after (d) passes, consider changing self_play.rs/run_training_loop.sh defaults to Gumbel.
Out of scope for this plan beyond noting it as the natural next step — do not fold into any of the above PRs. Changing the default production self-play backend is a training-pipeline-wide decision that should be made by a human reviewing (d)'s actual numbers, not bundled into a code-correctness PR.

Why this order minimizes risk: each step is independently buildable, testable, and revertable; (a) has no behavioral surface at all (pure functions); (b) is testable in complete isolation from the production self-play/training pipeline (exactly as the current broken code already is — same blast radius, but now correct); (c) is a strictly additive, default-off wiring change verifiable by confirming existing zero-path behavior is byte-identical (the Brain::new call sites don't even change); (d) is a read-only comparison run with no code changes to the training pipeline itself. None of (a)-(d) touch network.rs, features.rs, mapper.rs's constants, or train.py — confirmed via CLAUDE.md's description of the dual-network-sync boundary and via direct inspection of what Gumbel actually consumes (PolicyOutput/ValueOutput from the existing forward_t, nothing more) — so the dual-network-sync constraint is inapplicable to this entire body of work; this should be stated explicitly in each PR description so reviewers don't spend time checking train.py diffs that don't exist.

---
Critical Files for Implementation

- /Users/verdi/Development/Polyfish/polyfish-rs/src/ai/gumbel_mcts.rs
- /Users/verdi/Development/Polyfish/polyfish-rs/src/ai/mcts_zero.rs
- /Users/verdi/Development/Polyfish/polyfish-rs/src/ai/brain.rs
- /Users/verdi/Development/Polyfish/polyfish-rs/src/ai/mcts_common.rs (new)
- /Users/verdi/Development/Polyfish/polyfish-rs/src/ai/gumbel_qtransform.rs (new)
- /Users/verdi/Development/Polyfish/polyfish-rs/src/bin/self_play.rs
- /Users/verdi/Development/Polyfish/polyfish-rs/src/bin/arena.rs
- /Users/verdi/Development/Polyfish/polyfish-rs/src/ai/policy_composer.rs
