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
