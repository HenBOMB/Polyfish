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
