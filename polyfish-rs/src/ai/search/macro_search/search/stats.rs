use crate::ai::oracle_macro::MacroGoal;

/// Search telemetry from the last `run` call (smoke instrumentation).
#[derive(Clone, Debug, Default)]
pub struct MacroMctsStats {
    pub nodes: usize,
    pub max_depth: usize,
    pub effective_map_particles: u32,
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
    /// EXP_ELO_165: index into `root_candidates`/`root_visits` of the
    /// net-synthesized candidate (`net_proposed_candidate`), if one was
    /// added this decision. `None` when `net_candidates_w == 0.0`, or the
    /// eval call failed, or the synthesized candidate duplicated an
    /// existing one. Lets a caller check `root_visits[i]` (did search take
    /// it seriously) and whether the final `pick` equals `i` (did it win)
    /// without re-deriving the candidate itself.
    pub net_candidate_index: Option<usize>,
    /// Count of branch-local high-confidence hypotheses materialized while
    /// expanding this root. Zero is meaningful: the threshold never fired.
    pub branch_capital_materializations: u32,
    pub branch_village_materializations: u32,
    pub branch_revealed_tiles: u32,
    pub branch_resource_reveals: u32,
    pub branch_capital_refutations: u32,
    pub branch_capital_confirmations: u32,
    pub branch_village_confirmations: u32,
    pub branch_unit_materializations: u32,
    pub branch_belief_candidate_nodes: u32,
    pub branch_belief_candidates: u32,
}
