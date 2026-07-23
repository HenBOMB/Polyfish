training a simple 1v1 small drylands map. max 10 turns

going with 25 mcts for ok depth and search. usually the best moves are always the first few available.

using complex action mapper because the legal moves generation is always different and indexed in different orders. deterministic != psudo-random (basically)

self play formula for eco and mil is crucial

proper type mapping is crucial

mcts uses only win policy, which is fine for now.

mil and eco are used to guide the model

training with FOW ENABLED. maximum difficulty for ai but will trail a lot smarter.
will prevent the ai from lerning FOWless behaviour and strategies.

allow panicing, encourage it. this will trigger bugs.

the simulator is "perfect" it will not error unless some bug occurs.

the legal move generator perfectly generates legal moves so .execute shouldn't in theory panic

target_index is used by economy moves, mostly
and src_index is used by army moves, mostly

just spent 20k ARS in runpod. referral got meself ~+10 USD for free


IM ENABLING FOW AGAIN!!

im guessing it will increase the complexity but will eventually line up with what truly matters.
fow disabled = cheating (basically)

adding heuristics to guide model, and self play


early turns dont require such deep mcts.
mcts requires dynamic depth.
played 1k games 1v1, TINY (11x11) map, DRYLANDS, branching factor analysis:
Turn   | Avg                  | Max       
-------|------------|----------------------|-----------
1      | 7.52                 | 18        
2      | 6.93                 | 20        
3      | 7.02                 | 27        
4      | 7.82                 | 31        
5      | 9.39                 | 40        
6      | 10.93                | 44        
7      | 11.77                | 45        
8      | 12.79                | 66        
9      | 13.78                | 67        
10     | 14.55                | 66        
11     | 15.30                | 71        
12     | 16.10                | 76        
13     | 16.65                | 72        
14     | 17.45                | 84        
15     | 18.04                | 86        
16     | 18.74                | 91        
17     | 19.36                | 113       
18     | 19.87                | 122       
19     | 20.56                | 123       
20     | 20.93                | 112       
21     | 21.72                | 125       
22     | 22.02                | 122       
23     | 22.17                | 131       
24     | 22.89                | 159       
25     | 19.26                | 159       
26     | 19.37                | 163       
27     | 17.20                | 175       
28     | 24.31                | 172       
29     | 20.88                | 193       
30     | 24.99                | 152 

polytopia has a narrower but much deeper search tree per turn compared to Chess
you must look ~8 steps deep just to complete one game turn

--
# Verdi, Jul 3, 2026
Thanks to a deep review from Claude Fable I was able to find and fix a couple of bugs that would hurt the quality of the training. Namely:
- A bug in flipping sign of the NN value head on every move instead of when player turn changes
- Weak gradient causing the value head to barely learn and thus MCTS relying only on priors
- Add more diversity in training data generation both by using temperature sampling over MCTS visited in the early moves instead of argsmax() and by randomizing the tribe per game in each call of self-play

I made a couple of improvements for faster training on Apple Metal. A core bottleneck to training is being able to generate lots of data in as little time as possible. Before it took me 60s to generate 5 games with 50 mcts iters. Now I'm down to 15s. The goal is to get sub 1s.

I have a couple of other ideas on this front to make things better. I did a training run, and brought my loss from 42 down to 34 in a very small run but it took 10min. Too slow. Furthermore, the loss is heavily dominated by the policy loss rather than the value. Could be the data sample + size of the run was too small to get a real signal here.

CPU speedup, baseline:
arch=x86_64 (Rosetta)  backend=zero  mcts=200  games=20
games_duration: 167.56s   avg_s/game: 8.38s   avg_moves: 95.3
moves/sec: 11.38   cores: 14 (M3 Max, all logical cores)

native arm64:
arch=arm64   backend=zero   mcts= 200 games= 20  moves/sec=11.88  avg_s/game=8.06

I reckon our current bottleneck isn't CPU yet it's still NN forward passes, so pausing on this optimization work for now until the bottleneck moves back to the CPU and then we can optimize since this will be the long-term bottleneck to our training regimen.
# Verdi, July 4, 2026
I tested, with having set up the Eval Server that batches GPU calls across all my actors I reach a top speed of ~31 moves/sec with 64 iters per move which is a meaningful speed-up from before I can get a game every 3s.

I am now bottlenecked on CPU which I knew I would end up here. Effect plateaus around 32 actors on this 14-core machine. My ultimate goal is to get to 100 moves/sec not sure how feasile this is on my hardware but will try.

Update: I was actually bottlenecked on GPU but it was GPU idle time. Digging more.

After lots of profiling I found that the problem was the poor MPS kernel of candle. I went on a very long journey to make things run better on my hardware and get closer to the actor ceiling.


| Milestone | moves/sec |
|---|---|
| candle Metal eval server (baseline) | ~31 |
| tch/libtorch MPS backend swap | 161 |
| + cache-hash offload to actor threads | 195 |
| tch, tuned (single readback, 32 actors) | 245 → later re-measured 157–242* |
| metal (MPSGraph) backend, cached executables, 32 actors / 1 server | 242* |
| metal + actor scaling (96 actors / 3 sharded servers) | 435* |
| **metal + pipelined workers (128 actors / 2 servers × 2 workers)** | **~578*** |
| **Actor ceiling** (dummy evaluator, no GPU — the hard cap) | **~1,500** |

# TRAINING CAMPAIGN LOG (Jul 5-6, 2026) — from heuristic-blend self-play to behavior-cloning bootstrap. 
Written so the arc survives even if CSVs/models
get deleted.

Setup at the start of this arc: Gumbel MCTS (64 sims, k=16), heuristic
score_move (ordering.rs) blended into ROOT priors only, weight
`w = 0.5 * decay^EFF_ITER`. Reward shaping always on. Value target =
score-ratio outcome (+ per-step progress blend). Tiny map, FOW, 2p.

Run-by-run (run_id, what happened, lesson):
- 1783242299 (36 iters, decay=0.865)
captures collapsed 1.98 -> ~0.06 in lockstep with the heuristic weight decaying. Lesson: the net was not internalizing captures before the crutch faded; slowed decay to 0.97.
- 1783245126 (20 iters, decay=0.97)
decline softened to 1.61 -> 1.48.
- 1783261893 (25 iters, -g 128)
behavior improved WITHIN each curriculum regime while the weight halved underneath (captures 1.34 -> 1.93 in the 10-turn regime; attacks 3.9 -> 6.5 in the 15-turn regime) = real learning, just slow. 
Also: totals step-function at CSV iter 14 was the 10->15 turn ratchet (`EFF_ITER = (i-1)*g/64+1` doubles per row at -g 128), not learning. Very low value loss ~= the value head reads the scoreboard; it rose when games got longer (healthy).

- BUG FOUND (Jul 5)
illegal moves were reaching real games. Reused Gumbel root children are built under simulate_move semantics (no FOW reveals, no discovery) but the game advances via play_move; play_move executes without re-validating legality and spend_stars only warned on overdraft. Result: negative stars, corrupt recorded games, replay desyncs, tainted training data. 
Fixed: reused children multiset-checked vs generate_legal_moves (mismatch -> fresh root), panic on real-move overdraft, catch_unwind per game so one bad game doesn't kill the run. All prior data + model trashed.
- 1783285900 (11 iters, ITER_OFFSET=76 -> 30-turn games with w~0.05 from the
  start)
monotonic REGRESSION. captures 6.5 -> 3.2, attacks 31 -> 15, moves/game 592 -> 460 (earlier EndTurns = passivity), villages t2c p50 11.8 -> 18.6 — while policy loss FELL 3.12 -> 2.70. 
Classic unanchored self-play collapse: the net confidently learned its own increasingly passive play. Paused.

Diagnosis (why self-play alone couldn't learn the basics at this compute):
the heuristic only nudged root priors; the training target re-ranks that
nudge by Q-values that are noise early on; 64 sims -> winning root child
gets ~15 visits, so the tree is 2-3 plies deep vs ~8 plies per Polytopia
turn — search can never SEE that a step toward a village pays off; and the
steps that lead to captures are rare, weakly-targeted, and value-diluted
all at once. Pure outcome-driven AlphaZero from scratch here costs
DeepMind-scale compute. Shortcut required.

### PHASE CHANGE (Jul 6): behavior-cloning bootstrap.
- Generated ~1,024 30-turn games (8 files x 128) with the network-free
  Heuristic search backend (already wired via --search-backend heuristic).
- Pre-trained fresh weights on them directly: batch 256, LR 2e-3 (sqrt-scaled
  from 64/1e-3), ~13 epochs total. Policy loss 4.53 -> 2.47. Notes learned:
  cross-entropy vs soft visit-count targets has an entropy floor (~2-ish
  here), so don't chase zero; each new train.py invocation restarts the
  cosine LR at max and undoes fine-tuning (use one long run, or TRAIN_LR
  lower for continuations). Env knobs added: TRAIN_EPOCHS, TRAIN_CHUNK_FILES,
  TRAIN_LR, REPLAY_BUFFER_FILES=0 to exclude archive.
- EXAM (network-only, --iteration 999 so w~0, 32 games, 30 turns):
  captures 7.72/game (teacher ~8), villages t2c first/p50/p80/all =
  6.9/9.1/14.9/15.1, ruins p50 11.0, moves 538, SPT 2.2 -> 10.9 by t25.
  Best behavior ever measured in this project. BC bakes the heuristic into
  the weights at full cross-entropy strength instead of whispering it
  through a noisy search operator — this was the unlock.

### PHASE 2 BASELINE (self-play starts from the BC checkpoint):
  captures 7.7 / villages p50 9.1 at iteration 0. Self-play must hold these
  from iter 1, then beat them. Slide toward the 1783285900 numbers = drift;
  strengthen the anchor (floor prior_w at ~0.1) or re-clone. Launch rules:
  never --reset (deletes the BC model), move BC corpus files out of
  polyfish-rs/ first, ITER_OFFSET=76 to keep 30-turn games matching the BC
  data distribution. p1 vs p2 score gap (~4256 vs 3291) is seat advantage,
  both sides were the same model.

# SELF-PLAY FIX ATTEMPTS (Jul 6-7, condensed)

### ABS-YARDSTICK VALUE
since mirror-match score ratio is passivity-symmetric, I tried adding a absolute
target to the score (`final_outcome = 0.6*rel + 0.4*abs-vs-8K`). Hoped: external yardstick
punishes joint laziness. Got: no effect (1783350651/  1783359971); teacher mix + heuristic 
floor 0.1 slowed decay ~2x only but did not stop it.

### 4-TURN FORWARD-DELTA LABEL
The idea is that learning over the score of the entire game propagates too small of a signal
to learn quickly that we should eagerly capture villages. This adds a component to the value target
so model wants to make moves that are learning to better score outcomes 4 turns away.
```
value = 0.7*near + 0.3*final
near = clamp((adv(t+4turns)-adv(t))/norm, ±1)
norm = max(600, 0.15*combined)
```

Hoped: dense per-action credit (vloss >> 0.02, captures recover). 
Got (1783379532): vloss 0.026 -> 0.092, later floors ~0.052 — label works, head fine — but behavior
decayed FASTER (cap 8.39 -> 6.79 in 3 iters). Also measured ~17 EndTurn edges/move in-tree: "search never sees tomorrow" was wrong.

### β-GATE on σ(Q) in policy targets only (β = min(1, iter/20))
Since that Q rescales, it could cause bad moves to be closer in score than good ones. Therfore
negatiely impacting the policy target's ability to learn. Tried fazing NN's Q prediction slowly
over iter. Hoped: noisy Q out of pi' stops early corrosion. Got (1783386024): best iter-1 ever (cap 8.14/atk 41.2), but crash slope at β=0.05-0.15 MATCHED β=1.0 — σ(Q) not the dominant channel.

### Log decision traces in a JSON
To troubleshoot I decided to instrument the whole decision process. What heuristic are hinted at root, what policy head suggests, what the value head evaluates each leaf node at, the ultimate Q computed, and then the move chosen.
I sampled across multiple games and iterations and I can in the data what confirms without a shadow of a doubt what I knew from earlier attempts: our value head sucks. It is not properly learning what is good for the game and it is THE thing that drags down the bot's performance.

THE NUMBERS (Jul 7-8, 2026):
Tooling: self_play --trace-villages --trace-trigger adjacent|on-village
--trace-max N; one JSON per game (first trigger only) into decision_traces/,
carrying per root candidate: raw_net_prob (pre-blend policy), heuristic_score,
blended prior, gumbel noise, own_value (NN value of resulting state),
post-search Q, visits, per-round Sequential-Halving survivor lists, chosen move.
ADJACENT = unit 1 tile (chebyshev) from an open village, deciding to Step onto
it. ON-VILLAGE = standing on it, deciding to Capture (a separate, later-turn
move from the step — two different decisions). Raw JSONs were scratch-only;
regenerate with the flags above.

Model: run 1783446473 (DETACH_VALUE_TRUNK=1, ITER_OFFSET=76, -r).
"before" = after iter 25 (checkpoints/model_pre_iter26_backup.safetensors),
"after" = after iter 27 (checkpoints/model_iter27_snapshot.safetensors).
53 argmax-mode traces (1 sampled-mode excluded). "value ranks it best" =
fraction of traces where the village candidate's own_value ties the max
own_value over all root candidates.

| decision | when   | n  | chose village move | raw_net mean | value ranks it best |
|----------|--------|----|--------------------|--------------|---------------------|
| Capture  | before | 14 | 43%                | 0.48         | 36%                 |
| Capture  | after  | 10 | 70%                | 0.80         | 60%                 |
| Approach | before | 18 | 78%                | 0.75         | 22%                 |
| Approach | after  | 11 | 36%                | 0.62         | 18%                 |

- Real vs noise: capture raw_net 0.48->0.80 solid (t~3.5); approach choose-rate
  78->36% is the one behavioral shift under p<0.05 (~0.03); capture choose-rate
  43->70% direction-only (p~0.19). Approach confidence grew a low tail: 3/11
  traces under 0.25 raw_net vs 1/18 before (going bimodal).
- Macro corroboration: captures/game at iters 26/27 = 4.16/4.03, continuing the
  slide from ~6.4 at run start.
- The invariant across all 53: own_value squashed to ~±0.3 around 0; approach
  move rated best-available only ~20% of the time before AND after; mean gap to
  the best alternative -0.03..-0.09. Policy raw_net is often 0.9+ on the same
  moves. Two iterations swung policy behavior hard in both directions and did
  not move the value ordering. Heuristic hints are NOT the gap: Capture 99.8
  (uniquely dominant), village Step ~121-140 vs ~35-90 for other steps.
- Vivid single case (before, on-village): raw_net 0.87 for Capture, own_value
  0.13, Q crashed to -0.30 after search, move not chosen — search destroying a
  correct prior read.
- WHY (the refinement that matters): the head is NOT failing to learn —
  value_loss converges (~0.16). The LABEL is empty. value = 0.7*near_delta +
  0.3*final_outcome; near_delta is a 100% RELATIVE 4-turn swing, final_outcome
  a per-game constant. In mirror play both copies capture on the same clock, so
  the relative swing for capture states nets ~0 -> label ~0 -> head correctly
  predicts ~0. Total abs share of the signal = 0.3*0.6 = 18% (abs only ever
  went into the 30% tail; the 70% term stayed pure relative). The
  passivity-symmetric trap, now measured at the decision level.
- Transmission into the policy: sigma(completed-Q) steers IN-SEARCH selection at
  full strength always (β only gates exported targets, never the tree). Flat Q
  -> approach moves survive Sequential Halving by gumbel luck -> fewer captures
  get played -> policy trains on those games. Fits every null so far: detach
  (not gradients), β-gate (not targets alone), low-LR arm (not optimizer).
- Caveats: trace runs used --iteration 1 => ~48% heuristic root blend vs the
  0.1 live floor, so choose-rates are flattered (raw_net / own_value /
  heuristic_score are pre-blend, unaffected); live approach behavior is likely
  worse than 36%. n=10-18 per cell. Contamination: 4 stray trace-run games
  files (~2% of samples, 10-turn iteration-1 games) got swept into iter-26
  training as fresh and sit in the archive window; trace runs must end with
  rm -f games_*.safetensors.

# Fable 5, Jul 8, 2026 — Phase-1 training-signal fixes (post-diagnosis)
Code review of the Jul 7-8 decision-trace diagnosis confirmed it, and added
one mechanism the traces couldn't see: sigma_completed_q MIN-MAX RESCALES
completed-Q into [0,1] before scaling by (C_VISIT+maxvisit)*C_SCALE (~5-6
logits at 64 sims). Whatever Q spread exists — signal or noise — gets
amplified to full amplitude in every selection step. An empty-label value
head therefore injects ~6 logits of noise into selection/halving, which is
exactly the "Q crashed a 0.87-prior capture to -0.30" trace. It also explains
the β-gate null: β only gated exported targets; the tree always ran full σ(Q).

Also flagged: the abs-dominant label (rel_w 0.4, from "definitive proof"
commit) conflicts with the negamax backup. mcts_common.rs negates value at
every player-turn boundary, which is only valid for antisymmetric v; absolute
own-progress is not antisymmetric (opponent progress != my loss), so abs
share gets corrupted through every EndTurn-crossing line — mildly today
(2-3 ply trees), worse the deeper search gets. Reverted, and the
mirror-symmetry problem is attacked in the DATA instead:

1. ANCHOR GAMES (--anchor-frac, loop default 0.25 via ANCHOR_FRAC env):
   that fraction of every selfplay iteration is played vs the network-free
   Heuristic backend, seat alternating. Passivity now actually LOSES games,
   so the relative label carries an anti-passivity gradient in the zero-sum
   frame the backup understands. Anchor-side data is recorded (fresh teacher
   data, same family as the BC corpus). Mutually exclusive with --opponent.
2. IN-TREE TRUST GATE (tree_q_weight): β_tree scales σ(completed-Q) in
   interior selection, the halving re-rank, and final recommendation.
   β_tree=0 degenerates to prior+gumbel — the BC-anchored behavior that
   produced the best exam ever. Driven with --value-trust (both β_tree and
   target β), which the loop ramps RUN-relative: min(1, i/30) * VALUE_TRUST_CAP
   (VALUE_TRUST_RAMP_ITERS env; EFF_ITER ramps saturate instantly under
   ITER_OFFSET=76 and were useless).
3. LABEL BACK TO ZERO-SUM: NEAR_DELTA_REL_W and FINAL_OUTCOME_REL_W both
   1.0. The label is antisymmetric again; signal comes from opponent
   diversity, not from breaking the value algebra.
4. LEAGUE = HISTORICAL-ONLY: latest-checkpoint league games were mirror play
   with extra steps; selection now prefers old checkpoints (falls back to
   any non-latest).
5. TRACE QUARANTINE: --trace-villages runs now write trace_games_*.safetensors,
   which the training glob/archive never matches — the 2% contamination
   path is closed.

Launch guidance for the next run: keep the BC checkpoint (never --reset),
ITER_OFFSET=76, -r; ANCHOR_FRAC=0.25; watch captures/game and villages-p50 —
phase-2 gate is still hold-then-beat 7.7 / 9.1. If captures still slide with
anchors on, next lever is a potential-based shaping bonus from an auxiliary
own-progress head (dense credit without touching the zero-sum backup), NOT
more abs share in the label.

## Loss floors decomposed + move_option target bug (Jul 9, 2026)

Question posed: policy_loss floored ~3.0, value_loss ~0.5, R² hovering
~0.4-0.5 for 30 iters — proof the 586K net is capacity-limited? Measured
instead of argued; numbers from archive/games_1783587735.safetensors
(run 1783582724 iter 11, zero-sum + anchor + trust-gate regime).

1. Policy loss is pinned to TARGET ENTROPY, not capacity. Intrinsic CE
   floor (scale-adjusted E[rowsum·H]) = 2.92 nats on that file
   (action .27 + source .43 + target 1.72 + option .50); live
   policy_loss = 2.98. CE cannot go below H(targets) at any parameter
   count. λ-era floor was ~1.9 because targets were peakier — the floor
   moved because the regime changed (trust gate + anchors spread search
   visits), not because the net hit a wall. Best-ever policy_loss: 1.74.
2. Value targets got 3x harder at the regime change: var 0.31 (std .56,
   13% saturated at ±1) vs 0.11 in the λ run. R² is variance-normalized,
   so cross-regime R²/loss comparisons are meaningless. value_loss
   column now = 3·win_mse + progress_mse (progress targets live) — not
   comparable to pre-progress rows either.
3. Convergence probe (scratchpad convergence_probe.py): current net on
   that file, 16K train / 6K holdout (block split), Adam 3e-4..1e-3,
   30 epochs. Train R² reaches 0.83-0.90 repeatedly → the net FITS the
   current labels; capacity not binding. Holdout ceiling ~0.52-0.58 vs
   live 0.43-0.48 → ~0.1 left on the table (2 hot epochs, fresh Adam,
   cosine restarts per iteration). Pre-registered widen-the-value-pool
   trigger (R² plateau ≤0.4 on clean labels) still NOT tripped.
4. BUG (since Feb 5, commit 1269e22): self_play.rs normalized p_action/
   p_source/p_target by total_visits but NEVER p_option — move_option
   rows carried raw visit counts (sums 2..64, mean 30, ~1.9% of rows).
   Each corrupted row ≈ a 30-64x-weighted sample; ~10 per 512-batch →
   roughly a third of policy-gradient mass. Fixed in self_play.rs
   (normalize p_option) + defensive renorm of rows with sum>1 at chunk
   load in train.py (archives still hold corrupted files). Verified:
   fresh game max rowsum = 1.0000.
5. Training is UNSTABLE even with clip_grad_norm 1.0: probe R²
   oscillates (0.90 → -0.65 within 3 epochs at lr 1e-3; live trains at
   2e-3 with cosine restarts). Renormalizing option rows calms it
   noticeably (longest stable stretch, CE -0.5 nats) but one excursion
   remains → a second destabilizer exists, value-side suspected (tanh
   saturation on ±1 labels, VALUE_LOSS_WEIGHT=3, or BN churn). Live
   per-iteration R² is a snapshot of this thrash, not a plateau.

Conclusion: neither loss floor evidences a capacity limit — one is an
entropy floor (policy fits ~perfectly), the other is label noise plus
optimization thrash. The capacity question is now empirically testable
at any time via the probe; re-run it after stabilization before any
trunk-growth decision.

## Observation memory + opponent aggregates + BN recalibration (Jul 9, 2026)

Shipped in one change (plan: kind-jumping-catmull), NUM_CHANNELS 154→161:

1. BN TRAIN/SERVE FIX (train.py): after the epoch loop, BatchNorm running
   stats are recalibrated (reset + momentum=None cumulative pass over up to
   16K fresh positions) before the f16 save. Verified in a sandboxed run:
   eval-mode R² 0.653 vs train-mode 0.709 — gap 0.056, down from 1.2-2.5
   pre-fix. Every value the search consumes now matches what the trainer
   reports.
2. ENGINE MEMORY (states.rs, actions/memory.rs): per-tribe
   `enemy_types_seen: HashSet<UnitType>` + `enemy_ghosts: HashMap<tile,
   GhostRecord{type,owner,turn}>`. Explorers archetype: real moves only
   (_are_you_sure), permanent, no undo — valid because the sim EndTurn
   fast-forward never executes opponent actions. Hooks: end of step_unit
   (post-embark/Hide, walks full_path, ghost at first fog-side tile),
   attack_unit (symmetric evidence; ghost for ranged-from-fog),
   discover_tiles real branch (reveal-then-kill coverage), end_turn sweep
   after process_start_turn_effects (spawn/upgrade/growth catch-all, drops
   ghosts on explored tiles, prunes age>10). obscure_fog strips non-POV
   tribes' memory. 8 tests in tests/observation_memory.rs.
3. NEW INPUT CHANNELS (features.rs): global block 20→24 — CH_OPP_SCORE
   (public scoreboard; the zero-sum label's other half), CH_FRAC_EXPLORED,
   CH_VISIBLE_ENEMY_PRODUCTION, CH_VISIBLE_ENEMY_UNITS,
   CH_ENEMY_TYPES_SEEN, CH_ENEMY_MAX_TIER (star-cost tier of strongest
   type evidenced) — plus a 3-channel ghost block (present/type/age).
   Also fixed a FOW leak: invisible (Cloak) enemy units were encoded on
   explored tiles while legal-move gen hid them; features now skip them.
4. COMPAT: conv1.weight padded 154→161 with zeros (play-identical until
   trained) across model.safetensors + all checkpoints (+ .pre_memory.bak
   backups); train.py zero-pads legacy-width archive files at load
   (channels appended at layout end → index-stable). Smoke game verified
   all new channels live (opp_score from sample 1, frac_explored
   0.21→0.95, ghosts in 84% of samples).

Watch after relaunch: value R² against the probe ceiling (opp-score input
should raise the ceiling itself), behavioral profile, and whether the
value head finally drives behavior now that play-side V isn't garbage.

## BatchNorm → GroupNorm swap (Jul 9, 2026)

Bundled into the from-scratch reseed. GroupNorm normalizes per-sample, so
train and eval run the identical function — train/serve gap structurally 0.0
(verified: max|train−eval| = 0.00e0); train.py's recalibration block deleted.

Design: trunk norms → GroupNorm(GN_GROUPS=8, 64), same bn1.weight/bias keys,
eps 1e-5, mirrored 4 ways (train.py, candle, tch, hand-built MPSGraph
subgraph). The 1-channel pool norms (p_pool_bn, v_pool_bn) were REMOVED, not
converted: a per-sample norm on one channel deletes the map's overall level —
the exact signal the value head reads. Parity square on a fresh GN model:
PyTorch ≡ candle ≡ tch-CPU ≡ tch-MPS ≡ MPSGraph (value Δ 0.0, softmax
Δ ≤ 8e-6); sandbox train epoch clean.

Fallout handled:
- All backends reject BN-era checkpoints loudly (bn1.running_mean check) —
  they'd otherwise load silently and play garbage. BN-era weights moved to
  checkpoints/bn_era/ (outside the league glob; usable only with pre-GN
  binaries, e.g. f847bc6).
- Parity exposed a pre-existing bug: src/main.rs loaded weights via
  VarMap::load on an EMPTY VarMap (silent no-op) — the :3000 server's NN
  agent had been running on RANDOM weights. Fixed with a file-backed
  from_mmaped_safetensors builder (same fix in examples/tch_parity.rs).

Reseed order: rm model + games/archive → init_model.py (GN weights) →
generate teachers → BC train → launch loop.

## Train/serve skew: BatchNorm running stats break the acting value head (Jul 9, 2026)

Measured on model.safetensors vs archive/games_1783587735.safetensors,
same weights, same samples:
- train-mode forward (batch BN stats): win R² = 0.497 — matches dashboard.
- eval-mode forward (running BN stats): win R² = **-0.75 to -2.0** —
  worse than predicting the mean. Policy heads barely affected (CE 3.37
  vs 3.40; softmax cancels shared normalization shift) but the value
  head reads absolute activations and is destroyed.
- 32 no-grad forward passes in train mode (refreshing running stats
  only, zero weight updates) recover eval-mode R² to **+0.42**.

Self-play/candle/MPSGraph inference all run eval-mode with these stored
running stats → every value the search consumes (own_value, TD(λ) root
bootstraps, trust-gated σ(Q)) comes from the broken calibration, while
train-mode dashboard R² reports the healthy fit. Root cause: BN EMA
(momentum 0.1) gets only ~2 epochs of updates per iteration while trunk
activations drift (running_var grows to 3.1e3 by block 5; v_pool_bn
running_mean = -794), so the EMA is always calibrated for an old net;
training thrash makes the lag worse. f16 storage adds minor precision
loss (~0.5 absolute at magnitude 794) but no inf/nan.

Fix (NOT yet applied — loop was live): after the epoch loop in
train.py, before saving, run ~32-64 no-grad train-mode batches over
fresh data to recalibrate running stats. Durable option: BN→GroupNorm
(no train/eval duality; requires candle mirror + likely retune).
Interaction warning: the VALUE_TRUST ramp injects the EVAL-mode value
head into search/targets — until recalibration lands, high trust is
amplifying a value function that plays at negative R².

Also confirmed same day: player_state input is 10 self-only scalars
(features.rs player_vec) while labels are zero-sum my-minus-opp — the
input contains no opponent aggregates (opp score is public in-game).
Part of the probe's ~0.56 holdout ceiling is information-theoretic:
candidate fix is player_state 10→~13 (opp score, opp visible cities,
opp last-seen units), dual-net sync + key migration required. Falsify/
confirm by re-running the convergence probe after the feature lands.

## Four auxiliary training-only heads (Jul 9, 2026)

Four extra heads give the trunk dense, exact supervision on the game's
hidden variables — territory, opponent presence, economy, opponent
tech — instead of hoping one noisy value scalar per state teaches them.
The memory channels feed evidence in; these force calibrated beliefs out.

| tensor | shape | loss (weight env) | target |
|---|---|---|---|
| aux_ownership | (N,121) | MSE (AUX_OWN_W=0.3) | end-of-game tile owner, +1/-1/0 from POV |
| aux_fog_units | (N,121) | BCE (AUX_FOG_W=0.2) | true enemy-unit occupancy now, incl. fogged |
| aux_spt | (N,2) | MSE (AUX_SPT_W=0.1) | [my, opp] SPT five turns ahead, /20 |
| aux_opp_tech | (N,42) | BCE (AUX_TECH_W=0.1) | opponent's end-of-game tech set |

self_play computes the targets while the real (unfogged) state is live
and ships them in the games file; train.py owns the heads, reading the
trunk and a new global pool (not v_latent). Rust inference is untouched —
every backend loads weights by name and ignores the aux keys. Samples
from files without aux targets are masked out, never zero-filled. The
four losses chart on the dashboard ("Aux losses").

Traps for future edits: never save the model from src/bin/train.rs (it
strips the aux weights; it warns at runtime); keep aux targets out of
the policy renorm guard and out of value_loss (value_r2 stays clean);
the tech multi-hot indexes by enum position, not discriminant; and
forward() now returns (policy, values, aux).

Verified: unit tests green; a smoke game emitted all four tensors
correctly (POV sign-mirrored ownership, binary fog); a sandboxed real
train() epoch handled mixed new+legacy files and saved all 8 aux
params; arena loaded an aux-era model; a 30-epoch probe showed all four
aux losses falling on train and holdout.

Live-run rules: aux losses should trend down within ~10 iters with
policy CE still at its floor; if win-MSE degrades >10% for 5+ iters,
halve the AUX_*_W weights. Aux losses stuck high while policy/value
keep fitting = trunk saturation (the capacity trigger).

## GN migration killed the pool convs — value_r2 pinned at 0 (Jul 9, 2026)

The first two GN-era runs (12 iterations total) logged value_r2 ≈ -0.000x
every iteration. That signature means "predicting the target mean exactly":
the value pool conv's pre-activations sat at mean -58 with **zero** positive
entries, so its ReLU output exact zeros for every sample — v_latent was a
constant, and win/progress could only fit their means (value_loss landed
precisely on the predict-the-mean floor, 3·var(win)+var(prog) ≈ 0.85).
p_pool_conv was equally dead, making action_type/move_option constant per
state; only the direct spatial convs and the aux heads (which bypass the
pools) were actually learning. Cause: the BN-era p_pool_bn/v_pool_bn had
absorbed the convs' output offset in running_mean; the GN swap removed the
pool norms without folding those stats into the conv weights, and a fully
dead ReLU gets zero gradient — no amount of training recovers it.

Fix: the 1-channel pool convs are now fully linear — no norm (see the GN
section) and no activation — mirrored in train.py, network.rs,
tch_network.rs, metal_network.rs. The fc+ReLU right after keeps the
nonlinearity; a linear pool has no death mode and passes the map-level
signal in both signs.

Verified via the canonical parity harness (examples/tch_parity,
examples/metal_parity, scripts_tch_parity_check.py): train.py ≡ tch-CPU ≡
tch-MPS ≡ MPSGraph, softmax Δ ≤ 5e-6 (candle matches on logits to 1e-4;
its lone 2.5e-1 target-head softmax delta is a near-tie flip at logit
scale -155 on the broken model, not a port bug). Fresh model at prod
hyperparams on the exact data that flatlined: train win R² 0.47 after 1
epoch, 0.91 after 4. 1-game self_play smoke clean.

Every pre-fix weight file assumes the pool ReLU — the current
model.safetensors and checkpoints/model_gn_v1.safetensors misbehave under
new binaries (league glob matches neither; --reset replaces the live
model). Relaunch from scratch.

## Slow first-village capture: broken map geometry + broken approach gradient (Jul 10, 2026)

The bar: first village captured by turn ~4-5 in nearly every game (mean
≤4.5). Measured (run 1783633575, iters 37-46): censored mean 6.4, ~1 in 10
games capturing nothing by turn 15.

Two causes, measured with `cargo run --example map_geometry` (400
training-identical maps):

1. **Training maps lack suburbs.** Real Polytopia guarantees 1-2 villages
   within radius 3 of every capital. Our mapgen implements that suburb
   phase only for Lakes/Archipelago; training uses Drylands, which gets a
   uniform ≥3-spacing fill instead. Result: 31% of capitals have NO
   village within 3 tiles (turn-4 capture physically impossible), 13%
   none within 4, mean nearest 3.44 → perfect play ≈ mean 4.4 on these
   maps. Lakes for contrast: 100% within 3, mean 2.55. Decision: map
   stays as-is for now (user call); judge t2c against ~4.4 as the
   environment optimum, not 4.0.

2. **The approach gradient used Manhattan distance on a Chebyshev-movement
   game.** `nearest_visible_capturable` + the Step closing bonus in
   ordering.rs scored diagonal-closing steps as non-closing (no +20) and
   understated urgency for diagonal offsets. Fixed to Chebyshev. This
   gradient is the teacher's entire steering (anchor games = zero-search
   score_move softmax) and reaches net games through the root prior blend
   (0.5 → 0.1 floor by ~iter 47), so the fix hits both data sources.
   Also added a matching approach term to the expansion evaluator
   (`(0.35 - 0.05·d).max(0)` city-points per unit, always below the 0.5
   standing bonus) so evaluator-based play sees the same pull.

Deeper Gumbel is NOT the fix: capture-on-turn-4 sits ~14+ plies from a
turn-1 root (~2 plies/turn × 4 own turns + interleaved opponent turns);
64→256 sims buys ~1-2 plies of principled depth at 4× the wall-clock. The
value net + dense priors are the horizon mechanism, not search depth.

Metrics: `villages_t2c_first` stays censored (max_turns when nothing
captured — never compare across curriculum steps at iters 26/51/76).
Added `villages_first_rate` (share of games with ≥1 village capture),
`villages_t2c_first_cond` (mean turn among those, -1 when none), and
`tribes` (per-iteration pair — the harvest/build sawtooth in behavioral
charts is tribe-matchup noise, not learning). CSV auto-migrates on the
next append-row; dashboard plots rate (yellow, right axis) + conditional
t2c on the village chart. The bar is now directly readable: rate → 1.0
and cond ≤ ~4.5-5 (vs the 4.4 environment optimum).

## The anchor was never the teacher — Greedy anchor swap (Jul 10, 2026)

Follow-up to the above, all measured at n=32, Bardur+Imperius (worst-case
1-move pair), iteration 80, 64 mcts-iters:

- Deeper search: 64→256 sims fixed the never-capture tail (rate 0.81 →
  1.00, net-only games) but not speed (cond 7.9 → 7.3) at 2.3× wall-clock.
  Depth can't buy a multi-turn walk; it only rescues in-horizon captures.
- Opening-exploration work in `score_move`: replaced the +2-adjacent
  resource sniffer with an unexplained-resource beacon (explored resource
  with no explored village within 2 certifies a hidden village — mapgen
  spawns resources only in village/capital rings; Starfish excluded).
  First version regressed (teacher 8.07 → 9.53): a unit whose sight can't
  reach the village parks on the resource. Scaling the beacon by the
  resource area's `regional_openness` fixed the parking (8.68). Replay
  analysis (examples/analyze_replays) then showed the real approach bug:
  at d=2 from a *visible* village the mover fled toward fog 52% of the
  time — `newly_revealed×4` + openness outbid the closing bonus once the
  village tile itself was explored. Damped curiosity to (openness×2,
  reveal×1) when a visible capturable is ≤2 away and steepened urgency to
  (18−4d). None of this moved the *anchor* aggregate…
- …because anchor games ran `SearchBackend::Heuristic` — the network-free
  ROLLOUT MCTS (64 evaluator-free rollouts, noise swamps the ordering
  gradient) — not the Greedy score_move argmax everything above tunes.
  Greedy teacher measured 1.00 / cond 6.47 vs the rollout anchor's 0.94 /
  8.9. Swapped the anchor seat to `SearchBackend::Greedy`
  (self_play.rs:1531) — which is also the exact distribution
  `blend_heuristic_prior` injects into Gumbel roots, so anchor data and
  root priors now agree. Production mix (25% anchor) after the swap:
  rate 1.00, cond 7.34.

Ceiling revision (user replay evidence + recompute): competent fog play on
these maps supports mean ~5.0-5.5, not 6.5 — the demonstrator still owed
turns. Post-swap replay analysis found the biggest leak: **attack-vs-capture
inversion** — a kill on territory-relevant ground scored 95+15=110 > Village
capture 99.8, so a unit standing on a village with an adjacent enemy
attacked every turn instead of banking the city. Raised the whole Capture
band to 115+ (ordering.rs). Post-fix analyzer: d=0 capture 100% (was 91%),
d=2 close 60% (was 33%), approach steps t3-4 all-toward. Greedy teacher
6.47 → 6.22 (rate 1.00); production mix 7.34 → 6.42.

Tried and reverted (measured no-gain): doubling the center pull when no
capturable/beacon evidence is visible (6.22 → 6.39, rate dip).

Remaining gap to ~5.5 is blind-phase direction luck, the 13% of spawns
with nearest village at 5+, and village races. The greedy anchor is a
bootstrap, not a gold standard: the net can exceed it by learning
map-prior direction cues (biome/terrain in its features) that a stateless
scorer cannot see. Suburbs remain the unlock for ≤5.

Beacon rule revision (user screenshot: spawn vision = territory + 1 ring,
and border fruits ARE the human's direction hint): the explained-veto
version discarded all spawn evidence because the capital's own Village
structure "explained" every resource in vision. Replaced with the
frontier rule — beacon iff the resource still has unexplored tiles within
Chebyshev 2 (a hidden generator could sit there), regardless of nearby
known cities. Measured (greedy, n=32×2): cond 6.20/5.97 vs veto 6.22,
rate 0.94-0.97 vs 1.00 — statistically a wash, kept for concept fidelity;
nearest-single-fruit can't read the clustering/side signal a human uses
(that's the net's job). Production mix: rate 0.97, cond ~6.5.

## July 20, 2026
I played a Dryland 1x1 domination so I can have some context to anchor performance. By turn 10 I had +22 spts (the ideal for most tribes is to be ~20 spts by turn 10 means you're doing well). By turn 20 I had 90 spts however. This is mainly because I was playing against CPU easy so I took control of the whole map quickly and then just maxed out on greed with sawmills, markets, forges, parks, etc. I lost 0 units. I had 3 warriors, 5 riders, and 4 giants.

### Third-city stall decision-trace diagnosis: proposal-side, confirmed

EXP_ELO_016 (reward-shaped label + tree) came back statistically indistinguishable
from baseline (hypothesis_driven_improvements.md) despite directly paying SPT/army
in the label and the in-tree backup. Before escalating to opponent-unfreezing
(the fallback EXP_ELO_016 named), checked which of the two proposal-vs-valuation
mechanisms is actually broken, using the same decision-trace instrument as the
Jul 7-8 diagnosis above, extended with a new trigger.

Tooling: `TraceTrigger::ThirdCity` (self_play.rs, landed with EXP_ELO_016) — fires
whenever pov is stalled at exactly 2 cities, turn>=15, with an unmoved unit; no
discovered-village requirement, since failing to develop is the failure being
hunted. Captures the full root candidate set (raw policy prior pre-blend,
blended prior, gumbel noise, in_top_k, post-search Q, visits) regardless of what
gets chosen. Two pre-existing capture sets, both --iteration 200, turns 15-24:
`decision_traces_towered/` (99 traces, known-towered reference checkpoint) and
`decision_traces/` (128 traces, later checkpoint).

Per-candidate raw_net_prob (pre-blend policy; sums to 1.0/trace, verified) by
move_type, towered -> current:

| move_type | towered mean (n) | current mean (n) | in_top_k rate (towered -> current) |
|---|---|---|---|
| Capture  | 0.79 (15)   | 0.93 (19)   | 93% -> 100% |
| Attack   | 0.047 (129) | 0.18 (188)  | 36% -> 66%  |
| Step     | 0.018 (3836)| 0.015 (5171)| 30% -> 32%  |
| Build    | 0.0029 (3163), median 0.00003 | 0.0001 (3040), median 0.00000 | 6.5% -> 8.1% |
| Harvest  | 0.0001 (20) | 0.00000 (18)| 5% -> 11%   |
| Research | 0.0003 (526)| 0.00000 (711)| 3.6% -> 4.2%|

Build/Harvest's in_top_k hits are explained by numerosity alone (30+ near-zero-logit
candidates per trace; Gumbel noise occasionally lifts one through by chance) not by
the policy proposing them — median raw prior on an individual Build candidate is at
or below float display precision in both sets.

Yet when a Build/Harvest candidate DOES survive to get evaluated, the value estimate
does not agree it's bad: best-dev-candidate q_value beats the chosen move's q_value
on average in BOTH sets (towered 0.205 vs 0.021; current 0.136 vs 0.089 — gap
narrowed, never flipped). So the rare times search looks, it does not confirm the
prior's aversion.

**Conclusion: proposal-side, not valuation-side.** The policy head suppresses
Build/Harvest to 2-4 orders of magnitude below Capture/Attack/Step at the exact
fork where the tower forms, in both checkpoints, and the suppression got MORE
severe from towered -> current (Build mean raw prob 0.0029 -> 0.0001) even as the
chosen move's own confidence sharpened (mean chosen raw_prob 0.24 -> 0.48) —
consistent with a bootstrap trap: starved-of-visits moves never win Sequential
Halving, so self-play visit-count targets never correct the prior toward them.

Implications: (a) opponent-unfreezing (the EXP_ELO_017 fallback) attacks a
valuation-side mechanism — it changes what the leaf value should be, not what
gets sampled at the root — so it will not touch this fork, since the moves it
would need to reconsider are never sampled in the first place; (b) EXP_ELO_016's
null result is now explained rather than merely observed: it paid the label/tree
more for development but never touched the policy prior that drives Gumbel top-k
candidate selection. The lever this points to is exploration-side: forced
minimum visits / progressive widening on economy action types at the root, or
BC-style prior injection (book.rs), not deeper or adversarial search.

Caveat: chosen-move-type mix differs between the two sets (towered chose
Attack+Capture 19% of traces vs current 33%) — may be confounded by different
game/opponent context at capture time, not purely the checkpoint. The
per-candidate raw_net_prob/in_top_k signature, independent of what actually got
chosen, is the reliable half of this result.

### Location check: does the suppression hold in an unambiguous position? (Jul 21, 2026)

The ThirdCity data alone can't rule out "the stall genuinely has nothing worth
building" (§ above) — score_move itself rates Build low there too, and the
best-visited-Build-beats-chosen-Q evidence was survivorship-biased. Built a
second, purpose-built control: `TraceTrigger::HarvestReady` (self_play.rs) —
fires once per game when pov has exactly 2 cities AND a currently *legal*
(engine-affordable) Harvest move exists for a population-granting resource,
then captures one trace per turn for the trigger turn + the next 3 turns
(not single-shot — real play may delay the upgrade) into
`decision_traces_harvest/`. 40 mirror-self-play games on the current
checkpoint (model.safetensors = the EXP_ELO_016 extension's iter-40 gauge
checkpoint), 64 mcts-iters, 138 traces (33-35 per turns-since-trigger bucket
0-3), 0 sim failures.

**Headline: the suppression is not stall-specific.** Build/Harvest's raw
policy prior stays crushed at every point in the 4-turn window — median
per-candidate raw_net_prob 0.000002-0.00006, same order of magnitude as the
ThirdCity read — in an early-game (~turn 4-14), actively-expanding position,
not just the late stall. Chosen move is Step 66-70% of the time across all
four turns; Harvest itself gets chosen at the captured ply only 2/138 times.

**But this location wasn't as unambiguous as hoped, and both independent
judges say so.** score_move rates Step ~45-49 vs Build ~22-24 / Harvest
~20-27 at every turn in the window (a real, consistent heuristic preference
for Step here) — and unlike ThirdCity, best-visited-dev-candidate Q now LOSES
to chosen-move Q most of the time (68-75% of traces), not the reverse. Reason:
median trace has ~24-27 competing Step candidates (one per unmoved unit) —
early game, most turns genuinely are "keep exploring/expanding," and both the
heuristic and the network's own value estimates mostly agree Harvesting one
population point is lower priority than that.

**What survives: the suppression is real but disproportionate to even that
legitimate preference.** score_move's own ~25-point Step-vs-Build gap,
softmaxed at its own GREEDY_SOFTMAX_TEMP=5, would still leave Build
~0.5-1%+ relative probability. The trained network gives it ~0.0001-0.0006%
— 2-3 orders of magnitude more extreme than even a heuristic that agrees with
the direction. And in the n=3 traces with <=2 competing Step candidates
(too small to trust alone, but directionally sharp), Harvest's raw prior
jumps to 10-70% instead of ~0 — consistent with the extreme case being
mostly a numerosity/softmax-competition effect (many similar Step options
collectively outvoting a few Build/Harvest ones) layered on top of a real
but much milder underlying preference, rather than a categorical "the
network has zero belief in building" story.

**Instrumentation caveat:** one trace captured per turn (first ply only) —
the triggering tile's legal-Harvest status drops from 32/33 at the trigger
ply to 7/35 next turn to single digits after, while "chosen=Harvest at the
captured ply" is ~0 throughout. Can't distinguish "harvested later that same
turn, after other units moved" from "tile lost to territory/structure
changes" — the instrument samples one decision per turn, not the full
within-turn sequence, so this doesn't resolve whether the opportunity
actually gets taken within the window.

**Net read:** confirms the raw-magnitude finding (policy prior suppression
of Build/Harvest is real, extreme, and present everywhere checked, not a
ThirdCity artifact) but complicates the "wrongful suppression" framing from
the ThirdCity data alone — here value mostly agrees with skipping Harvest,
just not by nearly as much as the prior implies. Points at the Gate-1
numerosity/competition mechanism specifically (many Step candidates
collectively starving a few Build candidates via softmax normalization,
amplified further through v_mix training rounds) rather than a blanket
anti-economy policy. A forced-sampling experiment should show whether
guaranteeing Build/Harvest a fair shot at this fork recovers proportionate
(not necessarily dominant) selection.

### Correction: the comparison itself was wrong (Jul 21, 2026)

User pushback, correctly: Step and Harvest/Build aren't exclusive within a
turn. You move all units first (many Step plies), THEN spend remaining stars
on Harvest/Build once movement is exhausted — a single ply is one action, but
the turn isn't a one-shot choice between "move" and "develop." Comparing
Build's per-candidate prior against ~25 live Step candidates in the same ply
(as both analyses above did) measures the wrong thing. Also flagged:
score_move agreeing with the network isn't independent validation — it's
coded by the same project as a performance floor, not an oracle, so shared
bias is as plausible as shared correctness.

Fixed the instrument: HarvestReady now captures **every** ply belonging to
the triggering player across the 4-turn window (previously first-ply-only,
which structurally favored Step), and fixed a real bug the redesign exposed
— the window wasn't gated to the triggering player, so the opponent's
interleaved plies (same shared `turn` counter) were leaking into the sample.
30 games, mcts=64, 681 traces, 0 sim failures, bucketed post-hoc by how many
Step candidates were still live at capture time:

| Step candidates live | n traces | Build/Harvest present | chosen when present | Harvest per-cand prior (mean/median) |
|---|---|---|---|---|
| 0 (exhausted)  | 278 | 168 | **49%** (82/168) | **0.264 / 0.087** |
| 1-3            |   9 |   4 |   0%  (n too small) | ~0 |
| 4-10           | 111 |  92 |   7%  (6/92) | 0.032 / 0.000 |
| 11+            | 283 | 238 |   6%  (14/238) | 0.011 / 0.000 |

**Reversed, cleanly.** Once Step candidates hit zero, Harvest's per-candidate
prior jumps from float-zero to a median 8.7% — competitive, not suppressed —
and Build/Harvest gets chosen essentially a coin flip of the time against the
other live options (Summon, Ability, Reward, Research, Attack — all
legitimate uses of a turn's remaining budget, not noise). The earlier
"extreme suppression" reading was real as a *measurement* but wrong as a
*diagnosis*: it was measuring Build losing a popularity contest it was never
supposed to win ply-by-ply against every unit's movement options, not the
network refusing to develop its economy. `avg_builds ≈ 29.8/game` (CSV) was
sitting right there the whole time as the same signal at the macro level and
should have been weighted more against the per-ply framing sooner.

**What's real and what isn't, updated:** the Gate-1/v_mix sampling mechanism
(gumbel_mcts.rs, verified in source) still stands as a general property of
the search — it's not specific to this finding. But the ThirdCity and
first-HarvestReady analyses' headline claim ("Build/Harvest is suppressed by
2-4 orders of magnitude at this fork") measured a same-ply popularity contest
against Step, not a fair test of whether development gets its turn. Build
does show weaker recovery than Harvest even at Step=0 (median per-candidate
still ~0.0000, though aggregate mass/trace across all Build tiles jumps to
mean 0.265) — plausibly the per-tile dilution effect (many candidate
build-sites splitting one action-type's mass) still applies within this
subgroup even without Step in the mix; unconfirmed.

**Where this leaves the original tower question:** if the mechanism for
picking up development once movement is exhausted basically works, the
city-level deficit (§ ThirdCity diagnosis: 8.36 vs Greedy's 10.5 @t25) more
likely comes from pace/volume (fewer harvest-eligible turns reached, slower
expansion delaying when "movement exhausted" happens, structure/target
choice quality) than from a policy that avoids development outright. Revisits
whether forced-sampling-at-the-root (the standing next experiment) is even
the right lever — it targets the Step-competition mechanism this correction
just showed matters less than assumed.

**Correction (user, Jul 21):** don't use Greedy's own city-leveling as the
comparison bar — score_move (§ above) is understood by the user to favor
training lots of units over leveling cities, so it isn't a reliable reference
for "how well should this be done," only for "does the net at least keep up
with a cheap baseline" on other axes. Drop the "compare pace against Greedy"
suggestion; analyze the net's own turns directly instead (below).

### Whole-turn reconstruction: where do the stars go when leveling loses? (Jul 21, 2026)

Standing rule going forward, now in CLAUDE.md and memory: **analyze decision
traces across a whole turn, never a single ply** — a same-ply prior against
~25 Step candidates measures the wrong thing (see correction above). Applied
that here: grouped the 681 HarvestReady plies by (game_idx, turn), sorted by
move_count, reconstructed each turn's full chosen-move sequence for the
triggering player.

98 distinct (game, turn) groups; 90 had a Harvest/Build (population) opportunity
present at some ply that turn.

- **69/90 (77%) took it somewhere in the turn.** Matches the Step=0 bucket's
  49%-when-competing reading — at the whole-turn level it's actually a clear
  majority, not a coin flip.
- **21/90 (23%) never took it all turn.** Of those, 19/21 did reach a genuine
  head-to-head moment (Step exhausted, dev still available, not just "the
  window ended first"). What got chosen there instead of Harvest/Build:
  **Research 11, Summon 7, Attack 1.** Pooled across every ply of all 21
  missed turns, the actions that actually compete for the same star budget
  (excluding Step/Attack/Capture, which don't cost stars): **Research 50%,
  Summon 47%, Ability 3%.**

**Answer to "where is it going instead": almost entirely Research and
Summon, roughly evenly split, not idle movement and not nothing.** When a
city-leveling opportunity is skipped, the net is choosing to research a tech
or train a unit with those stars instead — a real opportunity-cost tradeoff,
not a policy blind spot. Whether that tradeoff is *correct* more often than
not is the open question this doesn't resolve — Research is instant/riskless
score (§7, hypothesis_driven_improvements.md — tech ≈15-25 pts/star vs a
population point's more indirect, delayed payoff), so a model that's already
over-indexed on tech for exactly that pricing reason skipping a harvest for
one more piece of research is the SAME mechanism restated at the star-budget
level, not a new one.

**Tech-purpose check (Jul 21):** pulled which specific techs got researched
at these 19 head-to-head moments from the existing trace descriptions (no
rerun needed) — Riding×3, Hunting×2, Strategy×2, Mining×2, Construction,
Ramming, Roads, Farming, Philosophy (1 each; tiers 1-3, no repeat/filler
pattern). Notably 3 of these unlock economy structures directly (Farming→Farm,
Construction→Windmill, Roads→trade connectivity) — so some of these are
themselves economy investments, just a different lever than the specific
harvest skipped. The rest (Riding→Rider, Strategy→Defender, Ramming→
Rammership, Hunting, Philosophy) look more opportunistic. Sample too thin
(14 instances, most techs appearing once or twice) to call this resolved
either way — not "purely wasted," not obviously purposeful.

### City-level reward-choice audit — the 77% figure didn't mean what it looked like (Jul 21, 2026)

User pushback: 77% "opportunity taken" is suspiciously high and doesn't
reconcile with known median/max city-level data. Right call — that figure
measured "did the model spend stars on SOME Harvest/Build in a turn," a much
lower bar than "did the city actually level up" (leveling needs population
accumulated past a threshold across possibly many such actions, and a single
resource tile can only be harvested once). Built a direct, cheap diagnostic
instead of inferring: `--dump-city-rewards <dir>` (self_play.rs) logs one
JSONL record per city level-up reward choice — turn, player, city level/
population/stars pre-choice, reward type picked. No MCTS trace overhead
(Reward moves are forced — `generate_reward_moves` preempts all other legal
moves when a choice is pending, moves/mod.rs — so this is a clean, zero-
competition read of what the policy wants at each level). 50 games, mcts=64,
456 reward choices logged across 224 distinct cities, 0 sim failures.

**Level distribution reached (224 cities, this batch):** level 2: 71 (32%),
level 3: 97 (43%), level 4: 28 (13%), level 5: 23 (10%), level 6: 5 (2%).
**Most cities cap out at level 2-3 (75% of them).** This is the real
reconciliation — 77% per-opportunity take-rate and a median level of 2-3
are NOT contradictory: hitting "take some harvest sometimes" often is
compatible with slow leveling once you account for how many accumulated
harvests a level threshold actually needs, plus each resource tile being a
one-time trigger. Roughly matches the campaign's existing aggregate
city_levels reads (~7.5-8.4 summed across ~2-3 cities @t20-25 ⇒ ~2.5-3.5
avg/city) — consistent, not a new discrepancy.

**Reward chosen, by level (the game only offers 2 options per level-tier,
confirmed clean/deterministic in this data):**

| level | options offered | chosen |
|---|---|---|
| 2 (n=218) | Workshop / Explorer | Workshop 69%, Explorer 31% |
| 3 (n=149) | Resources / CityWall | Resources 78%, CityWall 21% |
| 4 (n=56)  | BorderGrowth / PopGrowth | BorderGrowth 86%, **PopGrowth 14%** |
| 5-6 (n=33)| SuperUnit / Park | SuperUnit 67%, Park 33% |

**PopGrowth — the one reward that's a DIRECT population/leveling boost — is
the least-picked option at the one level where it's offered**, 14% vs
BorderGrowth's 86%. BorderGrowth (territory expansion) isn't a bad pick —
more territory means more future harvest/build tiles — but it's an indirect,
delayed lever compared to PopGrowth's immediate push toward the next level.
Consistent with the same tech-over-economy-adjacent pricing pattern already
diagnosed elsewhere in this campaign (§7, hypothesis_driven_improvements.md):
given the choice between something with a fast, legible payoff and something
that specifically accelerates further city growth, the model leans away from
the growth-accelerant even when it's free (reward choices cost no stars).

### Correction #2 + close-out: policy/value "misalignment" was confirmation bias, star-spend refutes the over-research claim (Jul 21, 2026)

Checked Research-vs-Dev head-to-head across all three trace sets (raw prior
AND Q, among visited candidates) to answer "are policy and value voting in
sync." First pass led with the one dataset (harvest-window) that fit the
user's prior (value favors Dev, policy favors Research) — but the other two
sets flip the direction, and the value comparisons there are n=13 and n=9,
unusable. **Correct read: no robust misalignment, direction unstable across
datasets, i.e. noise around zero.** More important: in all three sets, **Dev
wins the head-to-head plurality more often than Research** (33v25, 5v2, 8v0)
— directly against the "2/3 of the time it chooses research" impression
(which the user themselves flagged as unmeasured).

Built `--dump-star-spend <dir>` to check the actual claim (stars, not action
counts — research costs scale with cities×tier so a few expensive research
actions could still dominate the budget even if outnumbered). Reads the real
`tribe.stars` delta around `game.play_move`, so exact under discounts (e.g.
Philosophy). 40 games, 4035 star-costing actions:

| type | stars | share | n actions | avg cost |
|---|---|---|---|---|
| Build | 6389 | 42.6% | 1922 | 3.32 |
| Research | 5747 | 38.3% | 811 | 7.09 |
| Summon | 2170 | 14.5% | 949 | 2.29 |
| Harvest | 706 | 4.7% | 353 | 2.00 |

**Economy (Harvest+Build) = 47.3% of the star budget vs Research's 38.3% —
economy gets MORE stars, not less.** Research costs ~2-3x more per action
than economy moves, which is why its star-share (38%) outruns its action-
count share (20%), but even accounting for that, it doesn't dominate the
budget. This refutes the over-research hypothesis as stated. Combined with
the head-to-head result above: every "the per-decision policy is broken"
reading generated this session deflated under better measurement (ThirdCity
suppression → per-ply artifact; "value disagrees" → survivorship bias; now
"over-researching" → not supported by stars OR action counts). The
per-decision policy keeps not being the smoking gun.

**What IS real, decisively — SPT decomposition:** SPT = cities × income/city.
Pulled both from tempo_by_turn (net role): income/city climbs steadily and
healthily (2.13 → 4.76, turns 0→30) — **not** the problem. City count is:
**1.81 cities at turn 10, 2.60 by turn 30**, against a map that supports 3-4
before contesting the opponent (7-8 villages/11x11, 1 owned each, 5-6 open).
This is an expansion-pace deficit, not an income-efficiency deficit, and it
matches the user's own Jul 20 note (their 22-SPT game was expansion-driven,
"took control of the whole map quickly") and the OLDEST diagnosis in this
campaign (§6, hypothesis_driven_improvements.md — out-raced, not out-fought)
better than anything this session's Research-vs-Build angle turned up. Best-
fitting still-open explanation: the Jul 7-8 decision-trace finding above —
approach-toward-a-village undervalued because the label is empty in mirror
self-play (both sides capture on the same clock, so the relative-reward
signal nets to ~0 for expansion specifically). Not re-tested with this
session's tooling; the natural next step, not another economy-policy check.

**Metrics-tracking decision:** user asked whether training-evolution metrics
are tracked well enough to debug efficiently. Applied one filter: outcome/
behavior metrics (city count, SPT, t2c rates, techs, win rate) are robust and
are what actually located the real problem this session — trend those.
Internal-signal metrics (per-ply policy priors, Q-values, Research-vs-Dev
head-to-heads) misled this session on every use (see above) — dashboarding
them would enshrine the traps, not avoid them; keep them on-demand only.
Checked first rather than assuming a gap: t2c_2nd/3rd/4th, avg_builds/
harvests/research, value_r2, avg_spt_t{0..30} are already per-iteration CSV
columns; cities/spt/city_levels per turn per role already live in
tempo_by_turn.json. The gap was surfacing, not tracking — added `SPT per
city — net vs Greedy` to the dashboard (App.tsx `combinedRows`, free
division of two already-tracked fields) since that's the specific
decomposition that resolved the question. Did NOT wire `--dump-star-spend`
or `--dump-city-rewards` into the training loop — the star-spend read came
back null (no skew to chase) and both are exactly the kind of expensive,
point-in-time, easy-to-misread internal-signal captures this note just
argued against enshrining; promote only if a future read shows a genuine,
robust skew worth trending.

## "Purposefulness" hypotheses and the Layer-2 test (Jul 21, 2026)

Follow-up to the tempo/race findings above (`hypothesis_driven_improvements.md`
§4-§7): framed the city-count deficit as one instance of a general
"purposefulness" problem (does the model sustain a chosen course of action
across plies?) that should also show up in eco-dev and military behavior if
the mechanism is real. See `failure_mode.md` (new file) for the tracked
numbers — FM-1 (uncontested capture rate 73-78%, should be ~99%+), FM-2
(3rd-city pursuit rate 91-98%, should be ~100%), FM-3 (new: turns/tile ≈
1.8-1.9x geometric distance, identical across self-play/vs-Greedy/contested/
uncontested — an opponent-independent movement-pacing deficit, from a
per-turn distance-trajectory reanalysis of the existing `turnstates_*`
dumps: ~41-43% of turns during an active pursuit close zero distance).

Proposed 3-layer frame for where "purposefulness" leaks (proposal / reward /
representation) and picked Layer 2 (reward) to test first — see chat log for
the full hypothesis writeup. Cheapest test: does EXP_ELO_016's already-shipped
`dev_potential` proximity term (a real potential-based shaping term, verified
by reading `reward.rs`) already fix FM-3 once actually trained in?

**Result: no.** Re-measured FM-3 against `gauge_1784500013_iter20.safetensors`
(the EXP_ELO_016 run's own cities high-water-mark checkpoint) — 40 games vs
Greedy, 128/16, same instrumentation. Uncontested capture 72.2% (baseline
73.0/78.0%), turns/tile 1.79 (baseline 1.85-1.94), no-progress-per-turn 41.2%
(baseline 39.6-43.1%) — statistically indistinguishable from a model that
never saw the shaping. Caveat this narrows: `dev_potential`'s terms aren't
balanced — tech de-weight ≈-150/tech and SPT/army terms dominate; proximity
is the smallest term (+12/tile, capped at 7 tiles). EXP_ELO_016 was an
anti-tech/pro-economy repricing with a proximity garnish, not an isolated
urgency treatment — this falsifies that specific weak+diluted implementation,
not "a strong isolated per-turn-progress reward."

Sanity-checked the obvious confound (a stationary garrison could pin the
all-units min-distance and read as a false "stall"): tracked whether the same
exact unit-tile holds the distance minimum turn to turn. Only ~30% of stall
steps show a truly unmoving unit; the other ~65-74% show a *different*
unit/tile at the same distance — i.e. units are actively moving near the
target without net progress (milling, not pure inaction). Garrison-pinning
explains at most a third of FM-3; most of it is real.

**Discriminating trace (the Layer-1-vs-Layer-2 fork):** built
`TraceTrigger::VillagePursuit` (self_play.rs) — same "opportunity" definition
as the FM-1/FM-3 measurement (pov at exactly 2 cities, a discovered village
>1 tile away), captures every ply of the triggering player for the trigger
turn + 3 more turns, into `decision_traces_pursuit/`. Ran 40 games (128/16)
against the EXP_ELO_016 iter20 checkpoint (temporarily swapped into
`model.safetensors`, byte-identical restore verified via sha256 after), 400
traces (cap).

For each trace, identified the "Step toward the pursued village" candidate
(target tile strictly closer than source tile) and compared it against
sibling candidates:

- Raw policy prior is **healthy, usually dominant** — median 0.193 vs 0.00087
  for other Step candidates in the same trace (~220x), mean percentile rank
  0.94 among same-trace Step candidates, in_top_k 94.9%. This is the opposite
  signature from Build/Harvest's crushed prior — **not a proposal-side
  suppression** at this fork.
- The toward-village Step gets fewer search visits than whatever beats it
  (13.76 vs 32.54 mean) and its post-search **Q-value loses to the winner's**
  most of the time — proposal is fine, valuation isn't.

**Correction (same day) — the "chosen X% of the time" figures below are
ply-level artifacts; use the identity-tracked pursuer metric instead.** The
first pass reported "toward-Step chosen 41.8% of plies," then a whole-turn
re-bucket bumped it to "81.4% of turns." BOTH are misleading: they count
"*some* unit took *a* toward-village step at *some* ply," not "the unit that
should be taking the village actually advanced this turn." The correct metric
designates the pursuer = unit nearest the village at opportunity start and
follows THAT unit's exact tile forward via the chain of chosen Step moves
(immune to the garrison/milling confounds). Re-ran 120 self-play games (iter20,
byte-identical model restore verified) with both the pursuit trace and
`--dump-turn-states` on identical games; 43 pursuit windows / 150
pursuer-turns:

| pursuer-turn outcome | rate |
|---|---|
| PROGRESS (advance 37.3% + capture 10.7%) | **48.0%** |
| WASTED (stall 15.3% + sidestep 14.0% + retreat 22.7%) | **52.0%** |

FM-3's min-distance metric on the SAME games: 39.0% no-progress + ~12%
net-backward = ~51% wasted — matches to within 1pt. **Confirmed truth: ~50%
of pursuit-turns make progress, ~50% wasted, robust across two metrics on
identical data.** So FM-3 was right; the 81% was the artifact. Two facts that
reshape the fix: (1) the single largest waste bucket is **RETREAT (22.7%)** —
the pursuer actively moves *away* from its target, not just idles; (2) only
27.9% of pursuit windows are a clean every-turn march — bimodal (some get
`5→4→3→2→1`, others `STALL×4` / `RETREAT→RETREAT→RETREAT` walking off).

**Read:** valuation-side, confirmed. The toward-village Step is proposed fine
(healthy, usually-dominant prior) but the model *values continuing the
pursuit below its alternatives* — decisively enough to walk the unit backward
~23% of turns. The lever: make the reward for continuing a committed pursuit
large enough (and less diluted than EXP_ELO_016's 12-pt/tile garnish, which
sits next to a +100 raw capture payoff and a min-not-sum multi-unit proximity
term) that a well-evaluated "keep going" actually scores above "go do
something else." Whether a stronger reward alone also fixes the visit
under-allocation is a secondary thing to watch, not a separate prerequisite.

Not yet implemented: an isolated, stronger per-turn-progress reward
(decoupled from EXP_ELO_016's tech/eco repricing) is the next concrete step,
pending design + pre-registration as a new EXP_ELO entry.

## Inner mechanics of the away-from-village pursuer decision (Jul 22, 2026)

After EXP_ELO_018 was falsified, decomposed WHY the pursuer steps away and
WHAT it does instead, joining pursuit traces (per-candidate prior/visits/q/
own_value + chosen move) with self-play turn-states (visible villages, own
cities) by (game,turn,pov). Ran on clone (unshaped baseline) and 018
(pursuit-shaped); `pursuer_mechanics.py` in the session scratchpad.

**Intent of away-steps (never combat — 0% Attack/Capture in both models):**
| intent | clone | 018 |
|---|---|---|
| toward OWN city/territory | 48% | 38% |
| lateral / no clear target | 23% | 31% |
| into open / exploring | 17% | 21% |
| toward a DIFFERENT village | 12% | 10% |
The dominant away-move is the pursuer stepping BACK toward its own city, not
diverting to combat or another target.

**Mechanism (toward-village candidate vs chosen away-move, on wrong-move turns):**
| | clone | 018 |
|---|---|---|
| toward prior (raw_net_prob) | 0.16 | 0.21 |
| away prior | 0.44 | 0.29 |
| toward visits | 10.8 | 14.2 |
| away visits | 33.0 | 36.5 |
| toward LEAF own_value | 0.378 | 0.198 |
| away LEAF own_value | 0.399 | 0.214 |
| toward leaf < away leaf | 53% | 56% |

**Key finding — the value head is INDIFFERENT, not opposed.** The leaf
own_value gap between step-toward and step-away is ~zero (coin-flip on which
is higher) in BOTH models. So the post-search Q gap (toward loses ~83%),
which the earlier EXP_ELO_018 diagnosis called "valuation-side," is mostly a
CONSEQUENCE of the visit gap (toward gets ~1/3 the visits → shallow/imputed
Q), NOT the value head pushing toward down. Corrects the earlier
"valuation-side confirmed" framing: the true levers are the **policy prior**
(mildly favors going home) and the **visit allocation** (Gumbel starves the
lower-prior toward move), with the value head a neutral bystander that
provides no corrective pull toward completion.

**Why EXP_ELO_018 failed, precisely:** the reward aimed to raise toward-village
Q/value, but (a) the value head was never the blocker (indifferent, not
negative), and (b) the reward did NOT move the leaf-value gap — 018's is still
~0 (0.198 vs 0.214). It DID slightly narrow the prior gap (toward 0.16→0.21),
so the policy head learned to propose it a touch more, but not enough to beat
the visit dynamics + value indifference.

**Deeper root — connects to the Jul 7-8 "empty label" diagnosis:** in mirror
self-play, stepping toward a village vs staying home genuinely leads to
~equal-outcome states (both sides expand on the same clock), so the value head
CORRECTLY learned indifference — the label is flat for expansion timing. This
is why reward shaping in the absolute channel didn't move the leaf-value
ordering, and points the fix at either (a) the prior / visit allocation
(search-side: force the toward move a fair visit count so its Q is a real
estimate, not imputed), or (b) breaking the mirror-play label symmetry so the
value head has a non-flat signal to learn expansion urgency from.

---

## 2026-07-22 — CPU vs GPU NN eval speed (per-call latency), for horizontal-scale planning

Measured to settle the "is a single CPU NN eval ~100ms?" question. **It is not.**
The ~94-100ms figure in our old docs was candle **Metal** at batch-128 (the broken
backend), NOT CPU, and NOT per-call. Real numbers below.

Bench: `examples/bench_forward.rs` (BENCH_DEVICE=cpu|metal, zero-init weights so
timing is weight-independent), M3 Max, `--features metal,accelerate`, profiling
build. PolyZeroNet at its real shape (161ch, 11x11, 64 filters). Full forward +
CPU readback per call, steady-state after warmup.

| backend | batch | per-call | rows/s |
|---|---|---|---|
| candle CPU (Accelerate BLAS) | 1   | **2.65 ms** | 377 |
| candle CPU (Accelerate BLAS) | 8   | 6.94 ms | 1,152 |
| candle CPU (Accelerate BLAS) | 32  | 18.3 ms | 1,747 |
| candle CPU (Accelerate BLAS) | 128 | 58.4 ms | 2,192 |
| candle Metal (broken backend, ref) | 1   | 3.78 ms | 265 |
| candle Metal (broken backend, ref) | 128 | 76.6 ms | 1,671 |

Reproduce: `BENCH_DEVICE=cpu ./bench_forward <batch> <iters>` (build with
`--features metal,accelerate`).

Takeaways:
- **A single NN eval on CPU is ~2.65 ms, not ~100 ms.** CPU eval is perfectly
  viable for self-play; the "100ms" mental model was wrong by ~40x.
- **candle CPU (Accelerate) BEATS candle Metal** at every batch (58 vs 77ms @128).
  Confirms candle-Metal is broken (see [[candle-metal-19x-slower]]); on candle
  specifically, CPU > Metal.
- The real hierarchy: **MPSGraph/PyTorch-MPS (~5ms/128 = ~26-40K rows/s) >> candle
  CPU (~58ms/128 = ~2.2K rows/s) > candle Metal (~77ms/128 = ~1.7K rows/s).** A
  *good* GPU path is ~12-18x candle-CPU throughput; a good GPU is the eval win,
  but a CPU is only ~12-18x slower in THROUGHPUT, not 40x in latency.
- CPU rows/s barely scales with batch (377 -> 2192 for 1 -> 128): no GPU-style
  massive parallelism, and Accelerate is already multi-threaded so concurrent
  eval processes contend for cores.

Architecture implication (horizontal scale): eval must stay LOCAL to each rented
box (round-trip = us), NOT streamed to the laptop GPU over the internet
(round-trip = 10-100ms on the latency-critical MCTS blocking path -> fatal; also
one laptop GPU caps the whole fleet, and features are 78KB/leaf). Choice per box:
cheap CPU-only box (local candle-CPU eval, ~2.2K rows/s/process, run several) vs
GPU box (local candle-cuda — VERIFY it's cuDNN/cuBLAS-fast, not candle-Metal-broken).
Decide on $/move by reading self_play METRICS moves/s on each box type.

Caveats: profiling build (release-lto slightly faster); M3 Max CPU + Accelerate is
a Mac proxy — a Linux rental (EPYC + MKL or default gemm) will differ; measure on
the actual box before committing.

### Budget-independent throughput metric (moves/s is NOT it)

moves/s depends on the MCTS budget, so it's the wrong unit for capacity planning
when the budget may change. The budget-INDEPENDENT unit is **NN evals/s** (rows/s;
self_play/actor_ceiling report it as `Leaves/s`). Demonstrated (actor_ceiling,
dummy eval, 8 actors, post-batch-1, profiling build):

| mcts_iters | moves/s | Leaves/s (engine) | leaves/move |
|---|---|---|---|
| 32  | 4782 | 128,910 | 26.96 |
| 64  | 2287 | 124,315 | 54.36 |
| 128 | 948  | 104,848 | 110.56 |
| 256 | 330  | 73,031  | 221.03 |

- **moves/s ~ 1/budget** (8x budget -> ~14x fewer moves). Bad planning unit.
- **leaves/move ≈ 0.85 × mcts_iters** (clean; the 0.85 is the ~15% eval-cache hit).
- Identity: `moves/s = Leaves/s ÷ leaves/move`.
- `Leaves/s` (ENGINE descent rate, dummy eval) is budget-ROBUST but not invariant:
  it drifts 129K->73K (32->256 iters) because deeper trees cost more per descent.
- **The truly budget-invariant number is the EVAL rows/s** from bench_forward
  (fixed per-forward cost): candle-CPU ~2,200, good-GPU ~30K. And EVAL is the
  binding constraint, not the engine: engine can do ~124K leaves/s but a single
  candle-CPU eval process only ~2,200 rows/s. Real GPU self_play at 64 iters =
  ~578 moves/s = 31K rows/s ÷ 54 leaves/move ✓ (eval-bound, matches MPSGraph).

**Capacity-planning formula (use this for the fleet):**
`fleet moves/s = (total eval rows/s across all boxes) ÷ (0.85 × mcts_iters)`.
So boosting the MCTS budget 4× needs 4× the eval capacity (rows/s) to hold the
same game-generation rate. Plan the fleet in **rows/s**, convert to moves/s only
after fixing a budget.

### CORRECTION/refinement: eval/s IS budget-invariant (backpressure); the drift was a dummy-eval artifact

The `Leaves/s` drift table above was measured with a FREE dummy eval, so it
measured the **engine descent rate** (which drops as trees deepen), NOT eval
throughput. With a REAL (finite) NN, backpressure pins eval/s at the NN's
capacity: the tree can only yield evals as fast as the NN accepts them, so
**eval/s = NN rows/s = constant regardless of MCTS budget.** Empirical (rate-
limited sim eval, 3ms/batch × 2 lanes): moves/s 146→47 from iters 64→256 (falls
~3×) while eval/s stayed ~8-10K (~flat). So:

- **eval/s (= NN rows/s, from bench_forward: 2,192 CPU / 30K GPU) is the stable,
  budget-invariant capacity.** Plan with it. bench_forward has no budget knob —
  that's the tell.
- **node-visits/s (descents/s) is the ENGINE's invariant** (factors out depth,
  which is why Leaves/s drifted). It only BINDS at extreme budgets where trees
  get so deep the engine can't feed the NN (engine descent rate < NN rows/s);
  there the NN idles and eval/s falls below capacity.
- Crossover: at 64 iters engine ~124K leaves/s vs GPU ~30K rows/s → eval binds,
  4× headroom; by 256 iters headroom ~2.4×; ~thousands of iters → engine-bound.

**Full model:** `moves/s = min( eval_rows_s / (0.85·iters),  node_visits_s / visits_per_move )`.
Normal budgets → NN-bound (plan in eval/s = rows/s). Extreme budgets → engine-
bound (plan in node-visits/s).

### moves-per-game (for games/hr and $/game conversion)
From `training_log.csv` col 58 `avg_moves` (897 logged iters): **range 91-632,
mean ~457**. Curriculum-driven via `max_turns`: **~118** at the current short
(~15-20 turn) curriculum, **~450-630** for full 30-turn games. Use ~450 for
full-length-game cost planning, ~118 for the current curriculum. It's a ~4x lever
on games/hr and $/game, so pick the one matching the phase you're scaling.
Reproduce: `awk -F, 'NR>1&&$58>0{...}' training_log.csv`, or read METRICS avg_moves.
