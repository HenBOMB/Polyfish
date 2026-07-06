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
Verdi, Jul 3, 2026
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
--
Verdi, July 4, 2026
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
--
BUG FOUND (Jul 4, 2026): candle cross-attention runs with UNTRAINED weights.

While building the tch/libtorch eval path I ran a parity test and found candle's
forward diverges from PyTorch (train.py) by up to 0.99 in softmax on the same
model.safetensors. Root cause:

- model.safetensors stores cross-attention the way PyTorch's nn.MultiheadAttention
  does: PACKED `cross_attention.attn.in_proj_weight [192,64]` + `out_proj`.
- candle's network.rs CrossAttention reads SEPARATE keys
  `cross_attention.q_proj/k_proj/v_proj/o_proj.*` — which DO NOT EXIST in the file.
  No conversion step exists (checked init_model.py + train.py).
- candle's `VarBuilder::from_varmap` does not error on missing keys; it silently
  creates fresh (untrained, default-init) tensors for them. So candle's Q/K/V/O
  projections have never used the trained attention weights. The conv/bn/policy/
  value blocks load fine — only the attention block is corrupted, which matches
  the partial (not total) output divergence.

Implication: self-play (candle) and training (PyTorch) have been evaluating
DIFFERENT networks — a latent train/inference mismatch. The tch/MPS eval swap is
verified to match PyTorch to ~1e-6, so moving self-play onto it is a correctness
fix as well as a ~19x speedup. If we ever want the candle path correct too, fix
network.rs to load `in_proj_weight` and split it into q/k/v (and load out_proj),
matching nn.MultiheadAttention's packing.
--
TRAINING CAMPAIGN LOG (Jul 5-6, 2026) — from heuristic-blend self-play to
behavior-cloning bootstrap. Written so the arc survives even if CSVs/models
get deleted.

Setup at the start of this arc: Gumbel MCTS (64 sims, k=16), heuristic
score_move (ordering.rs) blended into ROOT priors only, weight
w = 0.5 * decay^EFF_ITER. Reward shaping always on. Value target =
score-ratio outcome (+ per-step progress blend). Tiny map, FOW, 2p.

Run-by-run (run_id, what happened, lesson):
- 1783242299 (36 iters, decay=0.865): captures collapsed 1.98 -> ~0.06 in
  lockstep with the heuristic weight decaying. Lesson: the net was not
  internalizing captures before the crutch faded; slowed decay to 0.97.
- 1783245126 (20 iters, decay=0.97): decline softened to 1.61 -> 1.48.
- 1783261893 (25 iters, -g 128): behavior improved WITHIN each curriculum
  regime while the weight halved underneath (captures 1.34 -> 1.93 in the
  10-turn regime; attacks 3.9 -> 6.5 in the 15-turn regime) = real learning,
  just slow. Also: totals step-function at CSV iter 14 was the 10->15 turn
  ratchet (EFF_ITER = (i-1)*g/64+1 doubles per row at -g 128), not learning.
  Very low value loss ~= the value head reads the scoreboard; it rose when
  games got longer (healthy).
- BUG FOUND (Jul 5): illegal moves were reaching real games. Reused Gumbel
  root children are built under simulate_move semantics (no FOW reveals, no
  discovery) but the game advances via play_move; play_move executes without
  re-validating legality and spend_stars only warned on overdraft. Result:
  negative stars, corrupt recorded games, replay desyncs, tainted training
  data. Fixed: reused children multiset-checked vs generate_legal_moves
  (mismatch -> fresh root), panic on real-move overdraft, catch_unwind per
  game so one bad game doesn't kill the run. All prior data + model trashed.
- 1783285900 (11 iters, ITER_OFFSET=76 -> 30-turn games with w~0.05 from the
  start): monotonic REGRESSION. captures 6.5 -> 3.2, attacks 31 -> 15,
  moves/game 592 -> 460 (earlier EndTurns = passivity), villages t2c p50
  11.8 -> 18.6 — while policy loss FELL 3.12 -> 2.70. Classic unanchored
  self-play collapse: the net confidently learned its own increasingly
  passive play. Paused.

Diagnosis (why self-play alone couldn't learn the basics at this compute):
the heuristic only nudged root priors; the training target re-ranks that
nudge by Q-values that are noise early on; 64 sims -> winning root child
gets ~15 visits, so the tree is 2-3 plies deep vs ~8 plies per Polytopia
turn — search can never SEE that a step toward a village pays off; and the
steps that lead to captures are rare, weakly-targeted, and value-diluted
all at once. Pure outcome-driven AlphaZero from scratch here costs
DeepMind-scale compute. Shortcut required.

PHASE CHANGE (Jul 6): behavior-cloning bootstrap.
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

PHASE 2 BASELINE (self-play starts from the BC checkpoint):
  captures 7.7 / villages p50 9.1 at iteration 0. Self-play must hold these
  from iter 1, then beat them. Slide toward the 1783285900 numbers = drift;
  strengthen the anchor (floor prior_w at ~0.1) or re-clone. Launch rules:
  never --reset (deletes the BC model), move BC corpus files out of
  polyfish-rs/ first, ITER_OFFSET=76 to keep 30-turn games matching the BC
  data distribution. p1 vs p2 score gap (~4256 vs 3291) is seat advantage,
  both sides were the same model.
