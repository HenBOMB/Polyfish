/// Leaf scorer for the Stage-1 rollouts.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MacroLeaf {
    // standard heuristic state evaluator
    Heuristic,
    // new value-target version using evaluate_state_v2 (see ai::evaluator::oracle_v2)
    HeuristicV2,
    // neural network value
    Net,
    // NetAsym with player-committed goal painting for training alignment
    NetAsymPaint,
    // antisymmetric net: (V(s,p) - V(s,opp))/2, for zero-sum negamax backup
    NetAsym,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MacroParams {
    // Max candidate directives per turn (scripted base always kept)
    pub k: usize,
    // Number of own turns simulated per rollout (includes candidate turn)
    pub horizon: u32,
    // Leaf scorer type for stage-1 rollouts
    pub leaf: MacroLeaf,
    // Lambda for Δgoal_potential in ply ranking
    pub lambda: f32,
    // Lambda for internal rollout ranking (tree-internal only)
    pub rollout_lambda: f32,
    // Simulations per macro-MCTS tree search (per turn)
    pub sims: usize,
    // Belief integration mode; how the agent uses harness-fed belief
    pub belief_mode: BeliefMode,
    // Weight for potential-based Δφ edge rewards (macro tree)
    pub shape_w: f32,
    // Weight on macro policy head's prior at search root (PUCT style)
    pub root_prior_w: f32,
    // Weight for NN rollout freeze estimator (`pi_rollout_value`)
    pub rollout_nn_w: f32,
    // Minimum depth for NN rollout freeze to apply
    pub rollout_nn_min_depth: usize,
    // Batch size of leaves per evaluator call in cheap rollout estimator
    pub leaf_batch: usize,
    // Weight for synthesizing net-proposed root candidates
    pub net_candidates_w: f32,
    // Enable tree candidate for continuing last directive (non-root nodes)
    pub tree_continuation: bool,
    // Enable sampling generator-grounded branch-local fog world per rollout
    pub tree_belief_materialization: bool,
    // Number of latent world map particles to average
    pub map_particles: usize,
}

impl Default for MacroParams {
    /// Defaults here are legacy/baseline: tests and non-NN entrypoints expect these.
    /// Production/main.rs override to NetAsym/0.05/1.0/1; arena deliberately stays baseline.
    fn default() -> Self {
        Self {
            k: 4,
            horizon: 2,
            leaf: MacroLeaf::Heuristic,
            lambda: 1.0,
            rollout_lambda: 1.0,
            sims: 32,
            belief_mode: BeliefMode::Off,
            shape_w: 0.0,
            root_prior_w: 0.0,
            rollout_nn_w: 0.0,
            rollout_nn_min_depth: usize::MAX,
            leaf_batch: 1,
            net_candidates_w: 0.0,
            tree_continuation: false,
            tree_belief_materialization: false,
            map_particles: 1,
        }
    }
}

/// EXP_ELO_035/036: how a macro-mcts agent consumes its harness-fed belief.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(super) enum BeliefMode {
    /// Belief ignored (production baseline).
    Off,
    /// 035: materialize the MAP world into the plan view before the tree.
    World,
    /// 036 rung 1: belief-conditioned fog-expansion candidates at the root
    /// (claim safe side / contest enemy side over predicted villages).
    Candidates,
    /// World + Candidates.
    Both,
}
