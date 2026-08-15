# Review of Verdi's Attack Plan

I've read the plan, the full MCTS/self-play/training code, the Gumbel draft, the
book/heuristics, the recorder, and your session logs. Here's the report.

# TL;DR

Your instincts are broadly right: batched cross-game inference is the correct
throughput fix, Gumbel is the correct sample-efficiency fix, and human/heuristic
bootstrapping is a sensible accelerant. But throughput is not your binding
constraint on _model quality_ — correctness of the learning signal is. I found
what I believe are two genuine bugs in the search (value backpropagation sign
handling, and terminal/horizon states evaluating to 0) plus a value-target
scaling problem that together mean your MCTS is currently operating almost
entirely on policy priors, with the value network contributing near-zero (and
sometimes inverted) signal. Speeding up game generation 20× before fixing these
would just generate corrupted training data 20× faster. The good news: the fixes
are days, not weeks, and they compound with everything else you're planning.

---

# Part 1: The batched-inference plan itself

The architecture in the plan (async actors + dedicated eval thread coalescing
across games) is the standard, proven design — this is essentially how KataGo,
and most modern AlphaZero reimplementations feed their GPUs. The plan's key
invariant (undo before await, `LeafData` owns its data) is correctly identified,
and sequencing it before 16×16 maps is right. Four gaps:

1. **You may be leaving the easiest speedup on the table already.**
   `run_training_loop.sh` builds without the CUDA feature — the line with
   `--features cuda` is commented out:

```16:18:polyfish-rs/run_training_loop.sh
echo "Building binaries..."
# cargo build --bin polyfish --bin self_play --release --features cuda
cargo build --bin polyfish --bin self_play --release
```

Candle compiled without the `cuda` feature always returns CPU from
`cuda_if_available`, so on your RunPod boxes **self-play inference runs on CPU
while the GPU idles** except during `train.py`. Even before the eval server
lands, flipping this on GPU machines is nearly free throughput.

2. **The batch server is only as good as its feed.** `GAMES_PER_ITER=10` means
   at most 10 concurrent games today. The plan's `--actors` knob is right, but
   the loop script must also raise `--num-games` substantially (64–256+) or the
   coalescer will starve. Worth stating explicitly in the plan since the loop
   script is listed only for "defaults."

3. **Drop the per-game virtual-loss batch to ~1–4, not 8.** With only 200 sims
   per move, collecting 24 leaves per batch from one tree forces breadth
   regardless of what the priors/values say — the first batch visits ~24 paths
   before any value feedback arrives. That's a 12% quality tax you no longer
   need once cross-game batching supplies GPU efficiency. The plan says
   "consider lowering to ~8"; I'd go further and make near-sequential per-game
   search the goal, since search quality is the product you're selling to the
   trainer.

4. **Two cheap multipliers are missing from the plan:** (a) an **evaluation
   cache** keyed by state hash — you already have `hash.rs`, and MCTS with undo
   revisits transpositions constantly; (b) **tree reuse between moves** — in
   Polytopia the same player makes ~8 consecutive moves, so the next root is
   literally the chosen child's subtree, and you're currently throwing it away
   every move. Both effectively multiply sims-per-second without touching the
   NN.

---

# Part 2: Issues that throughput won't fix (read this first)

## 2a. Value backpropagation flips sign on every edge — but Polytopia players don't alternate every move

```647:660:polyfish-rs/src/ai/mcts_zero.rs
        // Update each node along the path
        let mut current = root;
        for &idx in indices {
            value = -value; // Flip value for opponent

            if let Some(child) = current.children.get_mut(idx) {
                child.remove_virtual_loss(self.virtual_loss);
                child.visits += 1.0;
                child.value_sum += value;
                current = child;
            } else {
                break;
            }
        }
```

Your own notes say it: _"you must look ~8 steps deep just to complete one game
turn."_ Most tree edges are same-player moves; the sign should flip only when
the acting player changes (i.e., across `EndTurn`), and the sign should be
anchored to the leaf's evaluated perspective, not to the root by depth parity.
As written, a P1 move followed by another P1 move gets alternating signs, so
during selection the parent frequently _prefers children whose evaluations were
bad for it_. The identical pattern exists in `gumbel_mcts.rs` (lines 664–687).
The fix is small: record the player-to-move along the descent path and credit
`±value` by comparing each node's mover to the leaf's POV.

## 2b. Terminal and horizon leaves backpropagate 0.0 (a draw)

In `select_and_extract_leaf`, any leaf where `_game_over` or
`turn > turn_horizon` gets `features: None`, and `values` defaults to `0.0` for
it (`mcts_zero.rs` lines 383, 486–494, 595–603). Consequences: capturing the
enemy capital looks identical to a draw inside the search, and every deep path
that reaches the 2–5-turn horizon returns "nothing here." Terminal states should
return the true outcome (±1 or your score-based value), and horizon cutoffs
should be evaluated by the value head — that bootstrap is the whole point of
having one.

## 2c. Value targets are compressed to ≈ ±0.05, so the value head is nearly mute

The training target is `tanh(score_diff / 5000)` (`self_play.rs` line 667,
`default_max_score()` = 5000). Your session logs show P1/P2 average scores
around 1850–2350 with typical differentials of 100–400 points → targets of
tanh(0.02–0.08) ≈ 0.02–0.08. The value head is being trained to whisper. In
search, Q ≈ 0 for everything, so PUCT reduces to prior-following — which
explains how the sign bug in 2a has gone unnoticed: the values are too small to
matter either way.

**Important interaction: fix 2a and 2c together.** If you rescale values without
fixing the sign handling, you'll amplify inverted signals and the model may get
_worse_. Rescale by a realistic denominator (e.g., normalize by combined score,
or an empirical std per turn) or move to win/loss primary + score margin as an
auxiliary target.

## 2d. Smaller but real issues

- **EndTurn is banned at the root.**
  `expand_node_single(&mut root, game, false)` filters EndTurn whenever any
  other move exists, so the agent can never stop early — no saving stars, no
  declining to walk a unit into fog. In-tree it's allowed; the root should allow
  it too and let the network learn when.
- **No temperature sampling.** Self-play always plays argmax-visits. With
  mirrored Imperius-vs-Imperius on tiny Drylands, your data diversity rests
  entirely on Dirichlet noise, book shuffle, and map seeds. Standard practice —
  sample proportional to visits for the first ~15–20 moves — is a few lines.
- **Book moves pollute policy targets.** When the book fires (turns 0–1, every
  game), the training target becomes a one-hot on a _randomly chosen_ book move
  with `visits = iterations` (`mcts_zero.rs` lines 262–279). You're teaching the
  net full confidence in a coin flip, forever, since the book never turns off.
- **`train.py` round-trips weights through f16 every iteration** (line 387–388)
  and re-initializes Adam + cosine-warm-restart every iteration. Each sample
  also gets seen ~60 times over its 30-iteration buffer life (2 epochs × full
  buffer, every iteration) — an overfitting recipe on ~1k new samples/iter. Keep
  an f32 master (quantize a copy for inference if you need it), and sample a
  fixed number of minibatches from the window instead of full epochs.
- **Landmine:** the glob `games_*.safetensors` in `train.py` _matches_
  `games_human_1782729427.safetensors`, which is sitting in `polyfish-rs/` right
  now with `win = 0.0` value targets from `recorder.rs`. It's being trained on
  as "fresh" data (dragging the value head toward 0 on human states), then the
  loop's `mv games_*.safetensors archive/` will archive it and eventually delete
  it in the 30-file pruning. Rename the prefix or exclude it.

---

# Part 3: Your three follow-up ideas

**Gumbel — right call, but "finish it" is bigger than it sounds.** For your
regime (branching 8–25, compute-bound, need for policy improvement at low sim
counts) Gumbel is exactly the right tool, and the payoff is real: reliable
improvement at 32–64 sims instead of 200, i.e., a 3–6× throughput gain that
stacks multiplicatively with the eval server. But the current `gumbel_mcts.rs`
diverges from the paper in the three places that produce its guarantees: (1)
in-tree selection is `argmax(logit + Q)` with no visit-count term — the paper's
deterministic selection `argmax π′(a) − N(a)/(1+ΣN)` is what spreads visits
correctly; (2) there's no completed-Q / σ(q̂) scaling, so raw Q in [−1,1] is
added to log-probs of wildly different magnitude; (3) the policy target
extracted is visit counts of the surviving-after-halving root children only
(`root.children.truncate` throws the rest away before `move_visits` is built) —
the paper trains toward the _improved policy over all legal actions_, which is
where most of the sample-efficiency gain lives. Budget for a faithful
implementation, not a completion pass. It also cleanly deletes the
Dirichlet-alpha and root-EndTurn questions.

**Human replay seeding — worthwhile, with three caveats.** The plumbing largely
exists (`recorder.rs`, `replayer.rs`, and one converted file already). Caveats:
(1) value targets must be masked or computed from the replay's actual outcome —
training `win` toward 0.0 on strong human positions actively fights the value
head; (2) map-size reality check: your feature extractor and net are hard-coded
to 11×11, and most ELO-1300+ human games are on larger maps — 100 _tiny-map 1v1_
replays is a much scarcer resource than 100 replays; (3) do it as a distinct
imitation _pretraining stage_ (policy-loss-heavy, few epochs, then hand off to
self-play) rather than mixing the files into the RL buffer permanently, or
you'll anchor the policy to human play long after the model should have
surpassed it.

**Heuristic/book prior shaping — correct diagnosis, use mixing not forcing.**
Right now the heuristics are entirely absent from the Zero path — `ordering.rs`
and the evaluator are only consumed by `heuristic_mcts.rs` (UI/tests), and the
book _replaces_ search instead of informing it. The high-value pattern: at
expansion, mix `p = (1−λ)·p_nn + λ·softmax(heuristic_scores/T)`, and optionally
blend leaf values `v = (1−β)·v_nn + β·evaluate_state()` during early iterations,
annealing λ and β to zero on a schedule. That bootstraps sensible play without
ever contaminating training targets (train on visit distributions as usual), and
converting the book from "forced move + fake one-hot target" into a prior boost
fixes 2d's book issue as a side effect.

---

# Part 4: Opportunities not on your list

The single biggest omission: **you have no strength measurement**, so you can't
tell whether any of these initiatives works. `training_log.csv` tracks avg score
and loss — neither is monotonic with playing strength. `arena.rs` exists but is
manual. An automated Elo ladder (every N iterations, play the current model
against fixed anchors: random agent, `HeuristicMctsAgent`, and 2–3 frozen
checkpoints; log Elo to the dashboard) is ~a day of work and is the instrument
every other decision depends on. I'd build it before anything else on this page.

Others, folded into the matrix below: D4 data augmentation (already drafted and
commented out in `train.py` — the game's rules are dihedral-symmetric, so this
is a legitimate ~8× effective-data multiplier), playout-cap randomization
(KataGo's trick: most moves get cheap searches for value data, a fraction get
full searches for policy targets — 3–4× more games per compute), auxiliary value
targets (score-margin/eco/mil heads — scaffolding already exists in
`recorder.rs` and commented in `train.py`; KataGo showed large sample-efficiency
gains from these), and eventually a continuous decoupled generate/train pipeline
and a larger net (both correctly deferred until after throughput).

# Part 5: Effort × reward

| Initiative                                                      | Effort   | Reward               | Notes                                    |
| --------------------------------------------------------------- | -------- | -------------------- | ---------------------------------------- |
| Fix backprop sign (player-aware) in both MCTS impls             | Hours    | Very high            | Do first; pairs with value rescale       |
| Terminal → true outcome, horizon → NN eval                      | Hours    | High                 | Same PR as above                         |
| Rescale value target                                            | Hours    | High                 | Only together with sign fix              |
| Enable `--features cuda` on GPU boxes                           | Minutes  | High (on GPU)        | Verify with device log line              |
| Temperature sampling, first ~15–20 moves                        | Hours    | Medium-high          | Data diversity                           |
| Allow EndTurn at root                                           | Hours    | Medium               | Validate via arena                       |
| Guard `games_human_*` from glob/archival; f32 master weights    | Minutes  | Medium               | Silent-corruption insurance              |
| D4 augmentation (finish the commented draft)                    | ~1 day   | Medium-high          | Verify spatial-head transforms           |
| **Automated Elo ladder vs fixed anchors**                       | ~1 day   | **Highest leverage** | Prerequisite for judging everything else |
| Eval server plan (as written + games/iter ↑, per-game batch ↓)  | 1–2 wks  | High                 | Your current plan, amended               |
| Eval cache (hash.rs) + tree reuse between moves                 | 2–4 days | High                 | Multiplies sims/sec, stacks with server  |
| Faithful Gumbel (completed-Q, σ scaling, π′ targets)            | 1–2 wks  | High                 | Enables 32–64 sims/move                  |
| Heuristic prior mixing + annealing (book → priors)              | ~1 wk    | Medium-high          | Early-phase accelerant                   |
| Human replay imitation pretraining                              | 1–2 wks  | Medium-high          | Gated on tiny-map replay supply          |
| Playout cap randomization                                       | ~1 wk    | Medium-high          | Stacks with Gumbel                       |
| Training regime cleanup (sampled batches, persistent optimizer) | 2–3 days | Medium               |                                          |
| Auxiliary heads (score margin, eco/mil)                         | 2–3 wks  | Medium-high          | Dual-arch sync cost is real              |
| Continuous async gen/train pipeline                             | Multi-wk | Medium               | After eval server                        |
| Bigger net, 16×16, size-agnostic arch                           | Multi-wk | High, later          | Correctly sequenced last                 |

**Suggested order:** (1) the hours-scale correctness fixes plus the Elo ladder —
roughly a week including validation, and everything downstream is measured
against it; (2) the eval server as planned, amended per Part 1, with cache and
tree reuse riding along; (3) faithful Gumbel; (4) heuristic prior mixing and
human imitation pretraining as parallel accelerants once the measured baseline
is trustworthy. The one thing I'd explicitly _not_ do is land the throughput
refactor first and start burning GPU-days on long runs while the backprop sign
bug and mute value head are still in the loop.

## CPU Speedup

The hot path itself, ranked by expected cost Reading parallel_search_batch and
friends, per wave of 24 leaves:

Game logic dominates: legal_moves() + simulate_move()/undo. Every descent edge
executes a full move with a heap-allocated Box<dyn FnOnce> undo closure
(game.rs:243 — EndTurn allocates a whole Vec of them and ends every player's
turn). Every expandable leaf runs generate_legal_moves, which scans all units ×
attack/step ranges plus all econ options and boxes each move individually
(~10–40 allocations per leaf). This is intrinsic work, but the allocation churn
on top of it is not.

state_to_tensor per leaf (features.rs:218): fills 154×11×11 = 18,634 floats with
a full state scan, then Tensor::from_vec allocates a Metal device buffer per
leaf on the actor thread. The eval-server refactor fixes the second half for
free (actors will ship plain Vec<f32>; tensorization happens once per big batch
on the eval thread). The fill itself stays on the actor and is unavoidable — but
it's cache-friendly and cheap once native.

Quadratic path re-walk in selection. The descent loop in select_and_extract_leaf
can't hold a borrow across iterations, so at each depth step it calls
get_node_by_path(root, &indices_stack) — a fresh walk from the root. Descending
to depth D costs O(D²) node hops. With your 2–5-turn horizon at ~8 moves/turn, D
is 16–40, so ~300–800 pointer-chasing hops per descent instead of ~30. Fixable
with an arena-based tree (nodes in a flat Vec, children as index ranges) — which
also fixes the cache locality of Vec<ZeroNode>-of-Vecs and makes backprop walks
cheaper.

PUCT selection details (mcts_zero.rs:63): effective_value/effective_visits do a
RefCell::borrow() per child per comparison inside max_by — that's ~2 runtime
borrow checks × children × nodes-per-descent. Cell<f32> would make it free.
Minor, but it's in the innermost loop.

So — are you in "optimal territory"? No, comfortably not. In rough order of
leverage:

Fix	Effort	Expected gain Native arm64 toolchain	Minutes–hours	1.5–3× on all CPU
work Eval server (already planned)	As planned	GPU-side; also removes per-leaf
Metal buffer allocs from actors Eval cache (hash.rs) + tree reuse between
moves	Days	Multiplies effective sims/sec — skips whole descents/evals, biggest
algorithmic lever Arena-allocated tree (fixes O(D²) walk + locality)	~1–2
days	Moderate; grows with horizon depth Allocation churn: undo-closure boxing,
per-move Box<dyn Move>Days, invasive	Moderate; measure after the above My
recommendation on ordering: the toolchain fix immediately, then re-profile
natively (a native binary gives clean symbolicated sample/Instruments output,
which the Rosetta run couldn't) before touching the tree internals — the game
logic vs. tree-bookkeeping split may look quite different at native speed, and
tree reuse + eval cache likely beat micro-optimizing either one.
