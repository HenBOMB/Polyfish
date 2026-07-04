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

After lots of profiling I found that the problem was the poor MPS kernel of candle. It is not as performant as the official one so we're now using `tch-rs` bindings and it's much more performant. This is the benchmark (batch 128, full model, fresh host→MPS upload, full readback to CPU):
backend	            ms/forward
candle (integrated)	71.7ms
tch (integrated)	15.2ms
pure PyTorch/MPS	12.0ms
Now we should have significantly more performance.

Update: It was a massive success. This was the results on:
./self_play --num-games 32 --mcts-iters 64 --actors 32 ==> 170 moves/sec
which is a 6.5x speedup. Now we're in territory to do a lot of good work. Striving for +200.

I'm now eval-bound. This is the best place to be but I'm looking to squeeze a bit more from my hardware.
I tried --max-batch 512 --coalesce-timeout-us 2000 to do more per GPU hit but that didn't move the needle. 

I moved up hash() calls since that's a CPU-bound task anyways out of the eval-server and it didn't make sense to blokck every GPU forward call on that. It can parallelize. This boosted my throughput to 195 moves/sec!

I did a clean test of actor ceiling benchmark assuming no eval server and I saw speeds up to 1,500 moves/sec. This tells me there's a lot of headroom to scale.

One issue found in `tch_network` is the fact we were wasting multiple roundtrips to the GPU to readback tensors. Now we just concat all the values, do 1 call, and split back up on the other end. This brought me up to 211 moves/sec.
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
