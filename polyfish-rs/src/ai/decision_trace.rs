//! Serializable snapshot of a single root MCTS decision, captured on demand
//! by `GumbelMctsAgent` (see `arm_trace`/`take_trace`). Self-play attaches
//! run/game/trigger metadata around this and writes it to `decision_traces/`
//! for manual inspection — see notes.md for the village-approach diagnostic
//! this exists for.

use serde::Serialize;

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMode {
    /// move_count < TEMPERATURE_MOVE_THRESHOLD: chosen by visit-weighted
    /// sample, not by `chosen_tiebreak_score` — see each candidate's `visits`.
    Sampled,
    /// `recommend_final_move`: argmax visits, `chosen_tiebreak_score` breaks ties.
    Argmax,
}

#[derive(Serialize, Clone)]
pub struct CandidateTrace {
    pub description: String,
    /// Debug-formatted move type (e.g. "Step", "Capture") for readability;
    /// `MoveType` itself serializes as an integer repr elsewhere.
    pub move_type: String,
    pub source_idx: Option<usize>,
    pub target_idx: Option<usize>,

    /// NN value at this candidate's resulting state, root player's frame.
    /// `None` if the candidate was never expanded (not visited in search).
    pub own_value: Option<f32>,
    /// value_sum / visits after search; 0.0 if unvisited.
    pub q_value: f32,
    pub visits: f32,

    /// softmax(raw network logit), pre heuristic-blend.
    pub raw_net_prob: f32,
    /// Raw `ordering::score_move` scale (not a probability), always computed
    /// regardless of whether blending is actually live this call.
    pub heuristic_score: f32,
    /// softmax(logit) after blending — the prior actually used to seed search.
    /// Equals raw_net_prob when prior_heuristic_weight == 0.
    pub search_prior_prob: f32,
    pub gumbel_noise: f32,
    /// Was this candidate in the top-k Gumbel cut (i.e. actually searched)?
    pub in_top_k: bool,
}

#[derive(Serialize, Clone)]
pub struct RoundCandidate {
    /// Index into `DecisionTrace::candidates`.
    pub candidate_idx: usize,
    /// gumbel + logit + sigma(completed-Q) at this point in the search.
    pub score: f32,
    pub visits: f32,
    pub q_value: f32,
}

#[derive(Serialize, Clone)]
pub struct RoundSnapshot {
    pub round_idx: usize,
    pub round_considered: usize,
    pub visits_per_candidate: usize,
    /// Survivors this round (`in_cut[..round_considered]`), ranked by score,
    /// captured after this round's visits were allocated.
    pub survivors: Vec<RoundCandidate>,
}

#[derive(Serialize, Clone)]
pub struct DecisionTrace {
    pub root_own_value: f32,
    pub prior_heuristic_weight: f32,
    /// Full legal root move set, not just the top-k in-cut.
    pub candidates: Vec<CandidateTrace>,
    /// Sequential Halving narrowing, round by round.
    pub rounds: Vec<RoundSnapshot>,
    pub selection_mode: SelectionMode,
    pub chosen_candidate_idx: usize,
    /// Only the actual selection mechanism when selection_mode == Argmax;
    /// informational only under Sampled (see each candidate's `visits`).
    pub chosen_tiebreak_score: f32,
}

/// Accumulates a `DecisionTrace` across one search call. Lives behind
/// `GumbelMctsAgent::trace: RefCell<Option<TraceBuilder>>` so `&self` methods
/// deep in the search can record into it.
#[derive(Default)]
pub struct TraceBuilder {
    pub root_own_value: f32,
    pub candidates: Vec<CandidateTrace>,
    pub rounds: Vec<RoundSnapshot>,
    pub chosen: Option<(SelectionMode, usize, f32)>,
}

impl TraceBuilder {
    /// `None` if search never reached a final selection (e.g. armed then
    /// short-circuited by a book move or an empty legal-move root).
    pub fn finish(self, prior_heuristic_weight: f32) -> Option<DecisionTrace> {
        let (selection_mode, chosen_candidate_idx, chosen_tiebreak_score) = self.chosen?;
        Some(DecisionTrace {
            root_own_value: self.root_own_value,
            prior_heuristic_weight,
            candidates: self.candidates,
            rounds: self.rounds,
            selection_mode,
            chosen_candidate_idx,
            chosen_tiebreak_score,
        })
    }
}
