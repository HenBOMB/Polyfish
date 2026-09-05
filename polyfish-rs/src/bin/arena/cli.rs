//! The `arena` command line: two search/model configurations to play
//! head to head, plus the diagnostic dumps.
//! 
//! Flag names are a contract for the eval scripts that drive this binary.

use clap::Parser;
use polyfish::ai::brain::SearchBackendArg;
use polyfish::ai::macro_agent::{BeliefMode, MacroLeaf};

/// Arena: battle two configurations head-to-head.
/// Each seed is played twice with sides swapped; wins are attributed to the
/// configuration, not the seat. Per-move decision time is recorded per config.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    /// Path to configuration 1's model.
    #[arg(long)]
    pub(crate) model1: String,

    /// Path to configuration 2's model.
    #[arg(long)]
    pub(crate) model2: String,

    /// Number of seeds (each played twice with swapped sides = 2 * games).
    #[arg(long, default_value_t = 10)]
    pub(crate) games: usize,

    /// MCTS iterations per move (override per side with --mcts1 / --mcts2).
    /// Inherits MCTS_ITERS so a reading defaults to the budget the model is
    /// trained at; win rates are only comparable at a fixed (mcts, gumbel_k).
    #[arg(long, env = "MCTS_ITERS", default_value_t = 64)]
    pub(crate) mcts: usize,

    /// Override MCTS iterations for configuration 1.
    #[arg(long)]
    pub(crate) mcts1: Option<usize>,

    /// Override MCTS iterations for configuration 2.
    #[arg(long)]
    pub(crate) mcts2: Option<usize>,

    /// Search backend for configuration 1.
    #[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
    pub(crate) backend1: SearchBackendArg,

    /// Search backend for configuration 2.
    #[arg(long, value_enum, default_value_t = SearchBackendArg::Zero)]
    pub(crate) backend2: SearchBackendArg,

    /// Gumbel top-k at the root (only when a backend is gumbel).
    #[arg(long, env = "GUMBEL_K", default_value_t = 16)]
    pub(crate) gumbel_k: usize,

    /// Max game turns. Higher = more decisive but slower.
    #[arg(long, default_value_t = 30)]
    pub(crate) max_turns: i32,

    /// Game mode (2 = Domination, the training mode). The mode is a net
    /// input feature and steers the heuristic evaluator — match training.
    #[arg(long, default_value_t = 2)]
    pub(crate) gamemode: u8,

    /// Number of concurrent match-worker threads (games in flight).
    /// Independent of CPU core count — workers park while awaiting
    /// eval-server replies (same eval-serving design as self_play), so
    /// oversubscribing past core count is fine and is what produces the fat
    /// coalesced batches that make the fast eval backends fast. 0 = auto:
    /// 4x core count, clamped to the total game count (2 * --games) — sized
    /// for a single EXP-10-style 32-64 seed reading; raise it by hand for
    /// much larger batches.
    #[arg(long, default_value_t = 0)]
    pub(crate) concurrency: usize,

    /// Deprecated alias for --concurrency (previously capped rayon Metal
    /// devices; eval no longer runs on match-worker threads, so that
    /// rationale no longer applies). Kept for old shell history only.
    #[arg(long)]
    pub(crate) workers: Option<usize>,

    /// NN inference backend: "candle" (Metal/CUDA/CPU), "tch" (libtorch/MPS,
    /// ~19x faster on Metal, requires --features tch-eval), or "metal"
    /// (MPSGraph, bypasses libtorch's serial MPS dispatch queue, requires
    /// --features metal-eval — see metal_network.rs). Empty = auto: "metal"
    /// if the metal-eval feature is compiled in, else "tch" if tch-eval is,
    /// else "candle".
    #[arg(long, default_value = "")]
    pub(crate) eval_backend: String,

    /// Number of concurrent eval-server threads (shards) per config. Each
    /// owns its own weights copy + LRU cache. 0 = auto (3 on metal, 1 on
    /// tch/candle). See self_play's --eval-servers doc for the measured
    /// rationale (tch serializes across shards; candle rejects >1).
    #[arg(long, default_value_t = 0)]
    pub(crate) eval_servers: usize,

    /// Metal backend only: pipelined GPU worker threads per eval server.
    /// Ignored by candle/tch.
    #[arg(long, default_value_t = 2)]
    pub(crate) eval_workers: usize,

    /// Eval-server batch cap: max leaves coalesced into one forward_t.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_batch: usize,

    /// Eval-server coalescing flush timeout in microseconds.
    #[arg(long, default_value_t = 1000)]
    pub(crate) coalesce_timeout_us: u64,

    /// Eval-cache LRU capacity (number of cached NN evaluations), split
    /// across shards. 0 disables the cache.
    #[arg(long, default_value_t = 524288)]
    pub(crate) cache_cap: usize,

    /// Per-game virtual-loss mini-batch size (leaves coalesced per NN call
    /// within a single game's search tree). None keeps each MCTS agent's own
    /// default (24). Once cross-match coalescing exists (this eval-server
    /// setup), self_play measured a larger value as a net throughput loss —
    /// see self_play's --leaf-batch doc. Changing this from the default
    /// alters Gumbel move selection, same as --eval-backend; sweep before
    /// trusting a strength-gauge run against it.
    #[arg(long)]
    pub(crate) leaf_batch: Option<usize>,

    /// Write per-turn stat samples (score/SPT/stars/cities/units/unit-cost/
    /// techs per config) as one JSON per game into this directory — the
    /// EXP_ELO_001 loss-autopsy instrument.
    #[arg(long)]
    pub(crate) dump_stats_dir: Option<String>,

    /// Write one start-of-turn ground-truth snapshot per turn (both players'
    /// cities/units + the model player's FOW-visible neutral villages) as one
    /// JSONL file per game into this directory — the vs-Greedy 3rd-city
    /// pursuit instrument (config1 = the model/gumbel seat).
    #[arg(long)]
    pub(crate) dump_turn_states: Option<String>,

    /// EXP_ELO_034: maintain a per-seat belief state (capital posterior +
    /// score-delta inference) from legal observables only, and log one
    /// belief-vs-truth row per player-turn into each game's dump JSON.
    /// Observation-only — no agent reads it. Requires --dump-stats-dir.
    #[arg(long, default_value_t = false)]
    pub(crate) belief_calib: bool,

    /// EXP_ELO_035: config 1's macro-mcts plans on a belief-materialized
    /// view — believed capital, ghost units, and the inferred residual army
    /// are written into the fogged root before the tree runs. Requires
    /// --backend1 macro-mcts.
    #[arg(long, default_value_t = false)]
    pub(crate) macro_belief1: bool,

    /// EXP_ELO_035: same as --macro-belief1, for config 2.
    #[arg(long, default_value_t = false)]
    pub(crate) macro_belief2: bool,

    /// Capture the full root decision (priors, top-k cut, visits, Q) on every
    /// model ply where this tech is unowned, unlocked by its prerequisite, and
    /// affordable — i.e. the plies where the purchase is a live choice.
    /// Requires --dump-stats-dir. Arming invalidates the reused tree, so a
    /// traced game diverges from an untraced one at the first matching ply.
    #[arg(long)]
    pub(crate) trace_tech: Option<String>,

    /// Tribe for both seats. Spawn terrain is tribe-specific -- Bardur forest,
    /// XinXi mountain/metal, Kickoo water/fruit -- and a hub's ceiling is a
    /// property of the terrain around it, so any statement about hub quality is
    /// tribe-scoped until this is varied.
    #[arg(long, default_value = "imperius")]
    pub(crate) tribe: String,

    /// EXP_ELO_026 oracle-macro steer for config 1 (gumbel backend only):
    /// while it holds <3 cities, focus the pursuit channel on one sticky
    /// FOW-visible neutral village (nearest to its units).
    #[arg(long, default_value_t = false)]
    pub(crate) macro_commit: bool,

    /// EXP_ELO_026 oracle-macro steer for config 1 (gumbel backend only):
    /// while a commitment is active, drop every root Research move. v9 removed
    /// the old 5-star reserve escape, so this arm is now a hard block rather
    /// than the affordability test the original experiment measured.
    #[arg(long, default_value_t = false)]
    pub(crate) macro_star_gate: bool,

    /// Base map seed (seed i = base + i). 0 = derive from the wall clock.
    /// Fix it to play identical maps across separate arena runs (paired
    /// A/B arms).
    #[arg(long, default_value_t = 0)]
    pub(crate) base_seed: u64,

    /// JSON file with a fixed `{"seeds": [...]}` list (see
    /// eval_seeds.json) — seed pair idx uses seeds[idx] instead of
    /// base_seed + idx. Errors if --games exceeds the list length rather
    /// than wrapping. Unset: --base-seed behavior is unchanged.
    #[arg(long)]
    pub(crate) seed_file: Option<String>,

    /// EXP_ELO_028: drive config 1's goal channels with the Stage-1 scripted
    /// goal-setter (orders + stance + star gate) each ply. Gumbel backend1
    /// only. For probing goal-conditioned nets; a net trained without goal
    /// channels ignores the (zero-initialized) planes.
    #[arg(long, default_value_t = false)]
    pub(crate) goal_script: bool,

    /// EXP_ELO_028 Phase 1c: weight on the goal potential in config 1's
    /// in-tree edge rewards (stance/order priced shaping). Requires
    /// --goal-script. 0.0 = off.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) goal_w_tree: f32,

    /// EXP_ELO_032: leaf scorer for a macro-lookahead backend.
    #[arg(long, value_enum, default_value_t = MacroLeaf::Heuristic)]
    pub(crate) macro_leaf: MacroLeaf,

    /// EXP_ELO_032: max candidate directives per turn (base always kept).
    #[arg(long, default_value_t = 4)]
    pub(crate) macro_k: usize,

    /// EXP_ELO_032: own turns simulated per rollout, incl. the candidate turn.
    #[arg(long, default_value_t = 2)]
    pub(crate) macro_horizon: u32,

    /// EXP_ELO_032: λ on Δgoal_potential in the macro executor's ply ranking.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) macro_lambda: f32,

    /// EXP_ELO_033: simulations per turn-level tree search (macro-mcts only).
    #[arg(long, default_value_t = 32)]
    pub(crate) macro_sims: usize,

    /// Override --macro-sims for config 1 (sims-sweep rungs).
    #[arg(long)]
    pub(crate) macro_sims1: Option<usize>,

    /// Override --macro-sims for config 2.
    #[arg(long)]
    pub(crate) macro_sims2: Option<usize>,

    /// Override --macro-k for config 1 (candidate-width rungs, EXP_ELO_036).
    #[arg(long)]
    pub(crate) macro_k1: Option<usize>,

    /// Override --macro-k for config 2.
    #[arg(long)]
    pub(crate) macro_k2: Option<usize>,

    /// EXP_ELO_035/036: config 1's belief consumption (macro-mcts only).
    /// `world` = materialize the plan view (035); `candidates` =
    /// belief-conditioned fog-expansion candidates at the root (036 rung 1);
    /// `both`. Overrides --macro-belief1 when set.
    #[arg(long, value_enum, default_value_t = BeliefMode::Off)]
    pub(crate) macro_belief_mode1: BeliefMode,

    /// Same as --macro-belief-mode1, for config 2.
    #[arg(long, value_enum, default_value_t = BeliefMode::Off)]
    pub(crate) macro_belief_mode2: BeliefMode,

    /// EXP_ELO_039: override --macro-leaf for config 1 (net-vs-heuristic
    /// leaf A/B needs per-side leaves).
    #[arg(long, value_enum)]
    pub(crate) macro_leaf1: Option<MacroLeaf>,

    /// Same as --macro-leaf1, for config 2.
    #[arg(long, value_enum)]
    pub(crate) macro_leaf2: Option<MacroLeaf>,

    /// EXP_ELO_036b: config 1's weight on potential-based Δφ edge rewards in
    /// the macro tree (own edges only; 0 = off). Credits the WORK of advance
    /// toward the active directive inside the search objective — the 028
    /// lesson at the macro layer.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_shape_w1: f32,

    /// Same as --macro-shape-w1, for config 2.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_shape_w2: f32,

    /// War-room item 3: config 1's weight on the macro policy head's
    /// PUCT-style prior at the search root (0 = off, plain UCT). Requires
    /// an eval-server call the heuristic path otherwise never makes.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_root_prior_w1: f32,

    /// Same as --macro-root-prior-w1, for config 2.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_root_prior_w2: f32,

    /// EXP_ELO_125 (piece 4): weight on the cheap `pi_rollout_value` NN
    /// estimator, config 1 (0 = off, the default).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_rollout_nn_w1: f32,

    /// Same as --macro-rollout-nn-w1, for config 2.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_rollout_nn_w2: f32,

    /// EXP_ELO_125 (piece 4): minimum tree depth at which
    /// --macro-rollout-nn-w1 may freeze an edge, config 1.
    #[arg(long, default_value_t = usize::MAX)]
    pub(crate) macro_rollout_nn_min_depth1: usize,

    /// Same as --macro-rollout-nn-min-depth1, for config 2.
    #[arg(long, default_value_t = usize::MAX)]
    pub(crate) macro_rollout_nn_min_depth2: usize,
}
