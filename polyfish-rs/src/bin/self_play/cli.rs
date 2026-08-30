//! The `self_play` command line.
//!
//! Flag NAMES here are an external contract, not an implementation
//! detail: run_training_loop.sh, worker/worker_loop.sh, benchmark_self_play.py,
//! bench_eval_sweep.sh and bisect_arm.sh all pass them by name, and
//! worker/publish_manifest.py scrapes the *live process argv* and forwards
//! every flag it does not recognize to remote workers verbatim. Renaming a
//! field silently breaks distributed runs.

use clap::Parser;
use polyfish::ai::brain::SearchBackendArg;
use polyfish::ai::macro_agent::MacroLeaf;
use polyfish::ai::reward;
use crate::labels::{LAMBDA_RETURN, MissingBootstrap, TD_W};
use crate::traces::TraceTrigger;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub(crate) struct Args {
    #[arg(long, default_value_t = 2)]
    pub(crate) gamemode: u8,

    /// Turn cap for generated games. Default 50 matches the flat cap
    /// Verdi set deliberately (games shorter than this couldn't mature
    /// a hub economy or a giants push) -- lowering it is a real
    /// speed/data-quality tradeoff, not a free win, for runs that don't
    /// need full-maturity games (e.g. throughput experiments).
    #[arg(long, default_value_t = 50)]
    pub(crate) max_turns: i32,

    /// Number of games to play
    #[arg(long, default_value_t = 10)]
    pub(crate) num_games: usize,

    /// MCTS iterations per move
    #[arg(long, default_value_t = 64)]
    pub(crate) mcts_iters: usize,

    /// Optional opponent model path (if not set, plays against self)
    #[arg(long)]
    pub(crate) opponent: Option<String>,

    /// STARTING fraction of games (0..1) played against the network-free
    /// Heuristic search backend as an anchor opponent (seat alternates
    /// between anchor games). Anchor games break mirror-play symmetry: a
    /// passive net LOSES them, so the relative value label finally
    /// carries an anti-passivity gradient. The anchor side's data is
    /// recorded too (fresh teacher data, same as the BC corpus). Decays
    /// with `iteration` the same way `prior_heuristic_weight` does (see
    /// `decay_crutch`), then fully to 0 at --decay-last-iter. Mutually
    /// exclusive with --opponent.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) anchor_frac: f32,

    /// Iteration at which both heuristic crutches (the search-prior
    /// blend and anchor-game rate) hard-cut to 0, having spent the
    /// iterations before that decaying down to a 10% floor. Default is
    /// effectively "never" so standalone/benchmark runs aren't
    /// surprised; the training loop passes an explicit value (see
    /// DECAY_LAST_ITER in run_training_loop.sh).
    #[arg(long, default_value_t = usize::MAX)]
    pub(crate) decay_last_iter: usize,

    /// EXP_ELO_004: weight of the TD(lambda) delta vs the final-outcome
    /// tail in the value target (no-op if --no-reward-shaping is set; see
    /// the TD_W const rationale). Default preserves production behavior.
    #[arg(long, default_value_t = TD_W)]
    pub(crate) td_w: f32,

    /// TD(lambda) trace decay in the value label. Sets the credit window's
    /// center of mass to 1/(1-lambda) turns (0.8 -> 5, 0.875 -> 8) and, as
    /// the same parameter, the lambda^n terminal tail INSIDE the TD arm —
    /// the two cannot be dialed apart. The flat 30% outcome share is
    /// `1 - td_w`, independent of this.
    #[arg(long, default_value_t = LAMBDA_RETURN)]
    pub(crate) td_lambda: f32,

    /// EXP_ELO_021: scale on the relative final-outcome ratio before the
    /// [-1,1] clamp in the value LABEL (label-only — not the in-tree
    /// backup, so no EXP_ELO_005 search-disruption risk). Default 3.0
    /// saturates ~32% of outcomes at ±1; lowering it de-saturates so the
    /// value head can learn to distinguish "ahead" from "crushing".
    #[arg(long, default_value_t = 3.0)]
    pub(crate) outcome_scale: f32,

    /// EXP_ELO_006: relative weight used ONLY for TD(lambda) label
    /// windows; the in-tree backup keeps reward::REL_W. Default
    /// preserves production behavior (labels match the backup).
    #[arg(long, default_value_t = reward::REL_W)]
    pub(crate) label_rel_w: f32,

    /// EXP_ELO_002: iteration where the anchor-frac decay clock starts —
    /// the anchor's effective decay iteration is `iteration - this`
    /// (clamped at 0). The loop passes the current iteration to HOLD
    /// anchor_frac at its starting rate until the model crosses 50% vs
    /// Greedy, then pins the crossing iteration so decay runs from
    /// there. The prior-blend decay is unaffected.
    #[arg(long, default_value_t = 0)]
    pub(crate) anchor_decay_start: usize,

    /// Force both heuristic crutches to 0 immediately, regardless of
    /// iteration or --decay-last-iter. Integration point for a future
    /// strength-gated phase-out (model consistently beats the
    /// heuristic-only backend) — not wired to any automatic check yet.
    #[arg(long, default_value_t = false)]
    pub(crate) force_zero_crutches: bool,

    /// Value-head trust in [0,1]: β on σ(completed-Q) both inside the
    /// search tree and in exported policy targets. Overrides the
    /// iteration-based ramp (min(1, iteration/20)), which saturates
    /// uselessly when ITER_OFFSET-shifted runs start at high effective
    /// iterations. Drive this from the loop script (run-relative ramp or
    /// measured value-head calibration).
    #[arg(long)]
    pub(crate) value_trust: Option<f32>,

    /// First tribe (optional, defaults to random)
    #[arg(long)]
    pub(crate) tribe1: Option<String>,

    /// Second tribe (optional, defaults to random)
    #[arg(long)]
    pub(crate) tribe2: Option<String>,

    /// Opt out of reward shaping (the blended per-step TD(lambda) +
    /// final-outcome value target). On by default — EXP_ELO_004 (Jul 13)
    /// found the flat final-outcome-only fallback trains markedly
    /// slower/weaker at matched budget. Pass this to fall back to a flat
    /// final-outcome value for every action (e.g. to reproduce pre-Jul-13
    /// runs or isolate a regression).
    #[arg(long, default_value_t = false)]
    pub(crate) no_reward_shaping: bool,

    /// EXP_ELO_011/025: ±1 win/loss value labels from the adjudicated
    /// winner. Since Jul 28 (025) this flips BOTH arms — flat outcome AND
    /// the TD arm (outcome space, γ=1, root-value q-target bootstrap);
    /// EXP_ELO_011 tested the flat arm alone. Composes with --td-w.
    #[arg(long, default_value_t = false)]
    pub(crate) wl_labels: bool,

    /// EXP_ELO_016: weight on the development potential Φ in TD-label
    /// snapshots (`score + w·Φ`). 0 = raw score deltas (legacy).
    #[arg(long, default_value_t = 0.0)]
    pub(crate) shape_w_label: f32,

    /// EXP_ELO_016: weight on Φ in the Gumbel in-tree edge rewards.
    /// Threaded separately from the label weight (EXP_ELO_005 lesson:
    /// search reacts violently to reward changes). 0 = legacy.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) shape_w_tree: f32,

    /// EXP_ELO_018: weight on the isolated pursuit-progress potential Φ
    /// in TD-label snapshots (`score + w·Φ_pursuit`), independent of
    /// --shape-w-label. 0 = off.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) pursuit_w_label: f32,

    /// EXP_ELO_018: weight on the pursuit-progress Φ in the Gumbel
    /// in-tree edge rewards, independent of --shape-w-tree. Half-dose
    /// vs the label weight (EXP_ELO_005 lesson). 0 = off.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) pursuit_w_tree: f32,

    /// EXP_ELO_017: unfreeze the opponent during in-tree EndTurn
    /// crossings (Gumbel backend only) — each intervening opponent
    /// plays a real deterministic-Greedy turn instead of the engine's
    /// blind auto-skip. Training-data generation only; arena/gauge
    /// binaries always search frozen so every prior strength reading
    /// stays a valid yardstick.
    #[arg(long, default_value_t = false)]
    pub(crate) unfreeze_opponent: bool,

    /// EXP_ELO_020: DAgger expert dose. At each net-seat decision, blend
    /// Greedy's move-ranking at the MODEL'S OWN state into the policy
    /// target: `(1-a)*mcts + a*greedy`. 0 = off. Corrects the collapsed
    /// capture prior on-distribution (unlike BC, which labels Greedy's
    /// states). Net seats only; frozen search recommended to isolate.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) dagger_alpha: f32,

    /// EXP_ELO_028 Stage 1: drive the appended goal channels with the
    /// scripted goal-setter (orders painted + stance + star gate) on net
    /// seats, in both the recorded features and the search. Off = all
    /// goal planes stay zero ("no goal set").
    #[arg(long, default_value_t = false)]
    pub(crate) goal_channels: bool,

    /// EXP_ELO_028 Phase 1c: weight on the goal potential (stance/order
    /// priced in-tree shaping) in net seats' edge rewards. Requires
    /// --goal-channels. 0.0 = off.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) goal_w_tree: f32,

    /// Base map seed (game i plays seed base + i). 0 = derive from the
    /// wall clock, which is right for training but makes any two runs
    /// play different maps. Fix it to pair A/B arms on identical maps —
    /// map variance across 128 games is large enough to swamp the
    /// behavioral effects these runs are usually measuring (EXP_GATE_001).
    #[arg(long, default_value_t = 0)]
    pub(crate) base_seed: u64,

    /// JSON file with a fixed `{"seeds": [...]}` list (see
    /// eval_seeds.json) — game i plays seeds[i] instead of base_seed + i.
    /// Errors if --num-games exceeds the list length rather than
    /// wrapping. Unset: --base-seed behavior is unchanged.
    #[arg(long)]
    pub(crate) seed_file: Option<String>,

    /// Current training iteration (for curriculum learning)
    #[arg(long, default_value_t = 1)]
    pub(crate) iteration: usize,

    /// Search backend to use for MCTS.
    #[arg(long, value_enum, default_value_t = SearchBackendArg::Gumbel)]
    pub(crate) search_backend: SearchBackendArg,

    /// macro-mcts leaf evaluator. `heuristic` = `evaluate_state`;
    /// `net` consults the network (EXP_ELO_039). Until this existed the
    /// backend silently ran the heuristic leaf in every MACRO_GEN round.
    #[arg(long, value_enum, default_value_t = MacroLeaf::Heuristic)]
    pub(crate) macro_leaf: MacroLeaf,

    /// macro-mcts: simulations per turn-level search.
    #[arg(long, default_value_t = 32)]
    pub(crate) macro_sims: usize,

    /// macro-mcts: max candidate directives on the root ballot.
    #[arg(long, default_value_t = 4)]
    pub(crate) macro_k: usize,

    /// macro-mcts: λ on Δφ in per-ply executor ranking. Applies to the
    /// ONE real per-ply commit (rank_view, once per game ply).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) macro_lambda: f32,

    /// macro-mcts: λ for the INTERNAL search tree's own turn rollouts
    /// (expand-one-per-sim -- up to `macro_sims` calls per real turn,
    /// vs macro_lambda's one). Defaults to macro_lambda (current
    /// behavior, unchanged) when unset. EXP_ELO_061 throughput
    /// investigation: profiling found the Delta-phi ranking pass
    /// (goal_potential's city_risks) dominating actor CPU time --
    /// setting this to 0.0 skips it entirely for the 64x-more-frequent
    /// rollout calls while the real per-ply decision keeps full
    /// quality. Real tradeoff, not a free win: 0.0 rollouts rank
    /// candidates by score_move alone, so the tree's leaf values
    /// reflect a less goal-aware simulated policy -- measure before
    /// shipping as a default.
    #[arg(long)]
    pub(crate) macro_rollout_lambda: Option<f32>,

    /// macro-mcts: weight on potential-based edge shaping in the tree.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_shape_w: f32,

    /// War-room item 3: weight on the macro policy head's PUCT-style
    /// prior at the search root (0 = off, plain UCT — the default).
    /// Costs one eval-server call per real turn decision (not per
    /// rollout) when nonzero; the heuristic path otherwise never
    /// touches the eval server at all.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) macro_root_prior_w: f32,

    /// What an n-step return does when its checkpoint reports no root
    /// value. `zero` bootstraps with 0.0 (legacy); `mc` carries the
    /// weight to the terminal return instead of pulling the label toward
    /// zero — which is what a heuristic-leaf macro run needs, since it
    /// reports no root value at all.
    #[arg(long, value_enum, default_value_t = MissingBootstrap::Zero)]
    pub(crate) td_missing_bootstrap: MissingBootstrap,

    /// Gumbel: number of initial top-k candidates sampled at the root.
    /// Only used when --search-backend gumbel.
    #[arg(long, default_value_t = 16)]
    pub(crate) gumbel_k: usize,

    /// Number of concurrent game actor threads. Each holds a Game clone
    /// + MCTS tree, so RAM (not CPU) is the real ceiling — actors block
    /// (parking, no CPU used) while awaiting eval-server replies, so
    /// oversubscribing past core count is fine. 0 = use core count.
    #[arg(long, default_value_t = 0)]
    pub(crate) actors: usize,

    /// Eval-server batch cap: max leaves coalesced into one forward_t.
    #[arg(long, default_value_t = 256)]
    pub(crate) max_batch: usize,

    /// Eval-server coalescing flush timeout in microseconds.
    #[arg(long, default_value_t = 1000)]
    pub(crate) coalesce_timeout_us: u64,

    /// Per-game virtual-loss mini-batch size (leaves coalesced per NN
    /// call within a single game's search tree). Cross-game batching via
    /// the eval server now supplies GPU efficiency independently, so
    /// this can shrink toward sequential per-game search. Measured
    /// (2026-07-05, 96 actors / 3 metal shards): raising this to 6 DID
    /// fatten coalesced batches (avg 47→60) but was a net ~10%
    /// throughput LOSS — more leaf evals per move, worse cache hit rate
    /// (0.19→0.17), and slower per-forward. Fatter batches via this
    /// knob are added work, not amortization. Keep at 4.
    #[arg(long, default_value = "4")]
    pub(crate) leaf_batch: Option<usize>,

    /// Eval-cache LRU capacity (number of cached NN evaluations). 0
    /// disables the cache. Default is 524288 (512K entries, ~900 MB at
    /// ~1.8 KB per row). The cache lives on the eval-server thread and
    /// skips the GPU for any leaf whose RawFeatures hash to a cached
    /// entry — the only lever that reduces GPU work rather than
    /// reshuffling it. Hit rate is reported in EVAL_SERVER_STATS.
    #[arg(long, default_value_t = 524288)]
    pub(crate) cache_cap: usize,

    /// NN inference backend: "candle" (Metal/CUDA/CPU), "tch"
    /// (libtorch/MPS, ~19x faster on Metal, requires --features
    /// tch-eval), or "metal" (MPSGraph, bypasses libtorch's serial MPS
    /// dispatch queue, requires --features metal-eval — see
    /// metal_network.rs). Empty = auto: "metal" if the metal-eval
    /// feature is compiled in, else "tch" if tch-eval is, else "candle".
    #[arg(long, default_value = "")]
    pub(crate) eval_backend: String,

    /// Number of concurrent eval-server threads (shards). Each owns its
    /// own weights copy + LRU cache; leaves are routed by hash so cache
    /// locality is preserved. Never use >1 on tch: measured (2026-07-05)
    /// that 2 tch shards HALVE throughput (156.6 moves/s @ 1 vs 83.3 @ 2)
    /// because libtorch's MPS backend serializes across threads at the
    /// C++ level. candle rejects >1 (Metal corrupts when >1 thread
    /// encodes on the same device — see the bug_handoff invariant in
    /// eval_server.rs). On metal, 3 shards × 2 workers is the measured
    /// best (~610–650 moves/s, see expert_boost_throughput.md).
    /// 0 = auto (3 on metal, 1 on tch/candle). Overridable.
    #[arg(long, default_value_t = 0)]
    pub(crate) eval_servers: usize,

    /// Metal backend only: pipelined GPU worker threads per eval server.
    /// Each owns its own MTLCommandQueue, so N coalesced batches can be
    /// in flight on the GPU while the coalescer collects the next one —
    /// unlike --eval-servers sharding, the batch stream and cache stay
    /// unified. Ignored by candle/tch.
    #[arg(long, default_value_t = 2)]
    pub(crate) eval_workers: usize,

    /// Capture MCTS decision traces at village-approach moments (see
    /// decision_trace.rs) to decision_traces/*.json. Forces a fresh
    /// (non-reused) tree build only for the traced decision, and only
    /// once per game (first trigger), so normal runs are unaffected.
    #[arg(long, default_value_t = false)]
    pub(crate) trace_villages: bool,

    /// Which village-approach moment to trace. Ignored unless
    /// --trace-villages.
    #[arg(long, value_enum, default_value_t = TraceTrigger::Adjacent)]
    pub(crate) trace_trigger: TraceTrigger,

    /// Max decision-trace JSON files written across the whole run.
    /// Ignored unless --trace-villages.
    #[arg(long, default_value_t = 20)]
    pub(crate) trace_max: usize,

    /// Diagnostics: dump games where NO village was captured by either
    /// player into this dir — <base>.replay.json (watcher-loadable) plus
    /// <base>.decisions.json (search trace for every decision; forces
    /// fresh root builds, so within-turn tree reuse is off).
    #[arg(long)]
    pub(crate) dump_failed_dir: Option<String>,

    /// Observability: dump EVERY game into this dir, not just the
    /// zero-capture ones — <base>.replay.json (watcher-loadable) plus
    /// <base>.decisions.json. Same machinery as --dump-failed-dir, no
    /// capture filter. Forces fresh root builds (tree reuse off) and
    /// writes a lot: use with a handful of games. For macro-mcts games,
    /// also defaults --dump-macro-policy and POLYFISH_PLY_TRACE to this
    /// same directory when neither is set explicitly (see below) — the
    /// whole point of "dump everything for this game" is to not have to
    /// remember three separate flags to actually get everything.
    #[arg(long)]
    pub(crate) dump_games_dir: Option<String>,

    /// Pin the Greedy anchor to this seat (1 or 2) instead of
    /// alternating by game ordinal. Lets a debug run put the NET in a
    /// chosen seat, and therefore on a chosen tribe (--tribe1/--tribe2
    /// are seat-keyed). Ignored unless --anchor-frac > 0.
    #[arg(long)]
    pub(crate) anchor_seat: Option<u8>,

    /// Trajectory diagnostics: append one JSON record per player-turn
    /// (at turn start, before any moves) to <dir>/game<idx>.jsonl — the
    /// acting player's owned cities, FOW-visible uncaptured villages, and
    /// unit tiles. Ungated; the Python analysis does all filtering.
    #[arg(long)]
    pub(crate) dump_turn_states: Option<String>,

    /// Diagnostics: append one JSON record per city level-up reward
    /// choice (turn, player, city level/population/stars pre-choice,
    /// reward type chosen) to <dir>/game<idx>.jsonl. Ungated, ply-cheap
    /// (no MCTS trace overhead) — ordinary self-play run.
    #[arg(long)]
    pub(crate) dump_city_rewards: Option<String>,

    /// Value-head calibration: append one JSON record per net-seat step to
    /// <file> — {turn, my_score, opp_score, root_value, final_outcome,
    /// value_target}. For measuring whether the value head's prediction
    /// beats a plain current-score-ratio baseline at predicting the game
    /// outcome (does it have foresight, or just read the scoreboard?).
    #[arg(long)]
    pub(crate) dump_value_calib: Option<String>,

    /// Diagnostics: append one JSON record per Research/Harvest/Build/
    /// Summon move executed — (turn, player, move type, stars spent,
    /// read as the real tribe.stars delta) to <dir>/game<idx>.jsonl.
    #[arg(long)]
    pub(crate) dump_star_spend: Option<String>,

    /// Q-gap diagnostics: append one JSON record per city-reward choice
    /// ply (the modal Explorer/Workshop-style pair) with per-candidate
    /// post-search Q, visits and priors to <dir>/game<idx>.jsonl. Traces
    /// only those plies; not combinable with --dump-failed-dir.
    #[arg(long)]
    pub(crate) dump_reward_choices: Option<String>,

    /// v6 diagnostics: one JSON record per executed Harvest/Build with
    /// the owning city's level/progress and tribe stars before/after —
    /// the per-city level-completion discipline metric.
    #[arg(long)]
    pub(crate) dump_level_completion: Option<String>,

    /// v6 Q-gap diagnostics: sampled traces (turn <= 15, stars >= 2, max
    /// 12/game, one per turn) with per-candidate root Q for economy
    /// candidates (Harvest/Build/Summon/Research/EndTurn). Not
    /// combinable with --dump-failed-dir.
    #[arg(long)]
    pub(crate) dump_pop_spend_choices: Option<String>,

    /// Stage 3b (macro policy head, first step): one JSON record per
    /// macro root decision (turn, pov, candidate ballot, post-search
    /// visit counts) to <dir>/game<idx>.jsonl. Raw supervision for a
    /// future macro policy head — no encoding decisions baked in yet.
    /// Only macro-mcts backends produce rows; a no-op otherwise.
    #[arg(long)]
    pub(crate) dump_macro_policy: Option<String>,
}
