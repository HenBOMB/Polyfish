# Behaviour fixture — net (XinXi) vs Greedy (Imperius), seed 1786807403

Files: `replays/debug_xinxi_g0.json` (watchable), `replays/debug_xinxi/
game_iter135_game0_seed1786807403.decisions.json` (per-decision search
trace), `replays/debug_xinxi_turns/game0.jsonl` (per-turn directive).
Tools: `cargo test --lib replay_turn_audit -- --ignored --nocapture` with
`REPLAY_FILE=` prints the per-turn economy/ownership audit used below.

Final score 4685–3535 to the net, **and that is the first finding.**

## 1. Tier 1 lane: ForgeGiants all game, never expressed

`playstyle=ForgeGiants` from t0 to t30, zero pivots. The lane bought
Mining on t0 (the goal potential gave on-lane techs e=0.333 vs 0.167 for
off-lane, and the net took it, 34/64 visits) — and then **Smithery did
not arrive until t24**. Units built all game: 20 Warriors, 5 Swordsmen,
**0 Giants**. The lane is a label on the turn dump, not a build order.

## 2. Why Organization on t4 (move 37): nothing chose it

| candidate | edge_reward | own_value | visits | q | net prior |
|---|---|---|---|---|---|
| Research Hunting | 0.1667 | 0.389 | 15 | **1.3435** | 0.344 |
| Research Organization | 0.1667 | 0.389 | 15 | 1.3408 | **0.411** |

Identical shaping bonus, identical value-head reading to three decimals,
and Hunting actually had the higher q. The tie broke on the **raw policy
prior**. Neither tech is on the ForgeGiants lane; the lane's own next
tech was unaffordable that turn, so the goal potential — which does
discriminate when an on-lane tech is affordable (t0: 0.333 vs 0.167) —
had nothing to say and priced both off-lane options the same. The
purchase was not driven by a plan; it was a coin flip decided by the
policy head.

## 3. The t8 "free upgrade" was actually taken

Audit: after t7 P1 has 7 stars and @24 at lvl2 pop4/3 (level-up pending).
On t8 the net played Train Warrior@24, Train Warrior@79, Harvest Game@25,
**Choose reward Resources for @24 (+5 stars)**, Harvest Game@80, Harvest
Game@90 — ending on 9 stars with @24 at lvl3. The +5 was claimed.

The economy miss on that stretch is one turn earlier and is the same
mechanism as §2: on t7 the reward choice for @79 was Explorer vs
Workshop, the value head read them as **−0.985 vs −0.990** (indifferent),
and the prior picked Explorer (0.82). Scouting over production, decided
by the prior because evaluation had no opinion.

## 4. Losing the capital on t10 — both tiers, in sequence

| move | turn | who | what |
|---|---|---|---|
| 96 | t9 | NET | `Step: 24 → 36`, **prior 0.914** — vacates the capital with an enemy on 25, adjacent |
| — | t9 | — | alternative `Attack: 24 → 25` was in the ballot: prior **0.011**, 1 visit |
| 101 | t9 | GREEDY | `Step: 25 → 24` — walks onto the empty capital |
| 110 | t10 | NET | `Attack: 36 → 24` (strike the occupier) was in the ballot: prior **0.0000**, 1 visit of 64. Played `Step: 36 → 47` instead |
| 114 | t10 | GREEDY | `Capture Village at 24` — capital lost |

Tier 2's directive on t9 was `stance=Grow, orders=[Expand 71, Expand 85]`
— **no Defend order on the capital** while an enemy stood adjacent to it
and the garrison was the only unit on it. `Defend 24` first appears on
**t10**, one turn after the city was already occupied.

So: T2 failed to name the threat until it was too late, and T3 gave the
one move that answers it a prior of ~0. This is the general 049 result in
a single game — Defend orders arrive late and move the executor least.

Cost: spt 7 → 2, score 1895 → 1520, one city for the next ten turns.

## 5. ⚠️ The metric counted this as a WIN

`is_game_over` (functions.rs:40) ends a game only on elimination or
`turn > max_turns`. `win_by_capital` exists in settings but is **read
nowhere** — capturing a capital is not a win condition in this engine,
in any mode. With `--max-turns 30` the game was decided on score, and the
net won it 4685–3535 by turtling one city to level 7 while P2 sat on
three cities at spt 15.

From t10 to t20 the net held 1 city at spt 2 against P2's 3-4 cities at
spt 17. **Every arena win rate in this ledger scores that game as a
win.** That is a live validity problem for the primary metric, not a
detail of this replay.
