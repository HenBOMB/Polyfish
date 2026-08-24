# Map belief SSOT — one surface for village existence, tribe attribution, and the enemy capital

**Status: STAGES 0 + 1a IMPLEMENTED, Aug 24 2026.** See §15 for the implementation
record and the three places measurement contradicted this design. Stages 1b–4 remain
unbuilt and gated on the A/Bs described below. Originally proposed by
Verdi: unify "what do we believe about the map under fog" into a single API, computed
deterministically from geometric/generator evidence and refined incrementally as fog clears.

---

## TL;DR — five findings and one recommendation

1. **There are four non-unified mechanisms today, not three.** `guess_villages`
   (`ai/belief/prediction.rs:90`), `BeliefState.capital_posterior`
   (`ai/belief/mod.rs:100`), the `aux_*` heads (`train.py:58`), and — the one not in the
   brief — `PredictionState` / `update_predictions` (`prediction.rs:367`, called from
   `main.rs:343,369`), which caches a *third* copy of the village guess plus a terrain
   guess into `GameState._prediction` for the server/UI and for `obscure_fog`.

2. **The central open question is answered, and the answer is a clean NO.**
   `aux_ownership` is **not** "chance of ownership by the opponent." Its target is
   `ownership_from_pov(&result.final_owner, p_id)` (`self_play.rs:4507`), where
   `final_owner` is snapshotted once at **game end** (`self_play.rs:2847`) and reused as
   the label for *every ply of that game*. It is a spatially-decomposed **outcome
   forecast** — "who will own this tile when the game ends" — structurally a spatial
   value head, not a belief about the present hidden state. Mirroring it the way
   `aux_fog` was mirrored would deliver Verdi nothing he asked for. **A separate
   deterministic ownership/attribution layer is justified.** (§2)

3. **The cross-candidate propagation Verdi asked for is not a heuristic — the generator
   hands it to us as two hard implication constraints and one likelihood.** Village
   placement runs to **saturation** (`mapgen.rs:877-909`, `loop { … if candidates.is_empty() { break } }`),
   which yields a **maximality invariant**: every land, non-mountain, edge-legal tile is
   within Chebyshev 2 of a village or capital. So revealing a legal tile as *empty* (with
   no village or city already known within 2) **proves** an undiscovered village sits
   within 2 of it. Separately, resources spawn
   **only** within Chebyshev 2 of a village site (`mapgen.rs:954-976`), 3:1 weighted
   inner:outer — so a revealed orphan resource **proves** the same thing. And
   `tile.climate` is a jittered multi-source Voronoi flood-fill seeded at the capitals
   (`mapgen.rs:790-828`), immutable after generation — a direct likelihood on capital
   location. These are the novel core. (§3)

4. **All three signals are a pure function of the currently-explored set** — no reveal
   history required. That means **no new `GameState` field, no serialization, no
   `obscure_fog` change, no `set_belief` wiring** — which is precisely why `BeliefState`
   is dormant today (`macro_mcts.rs:716,956`: it must be externally fed, and `self_play`
   never feeds it). A recomputable SSOT is callable anywhere `guess_villages` is called
   today, including `self_play`, with zero plumbing. It is also frozen during search for
   free, because sims never touch `explorers` (`discovery.rs:81-110`). (§5)

5. **Recommendation: deterministic-first, exactly as Verdi's instinct says — but scripted
   consumers before network channels.** Build `MapBelief` as a pure derivation, calibrate
   it offline in the existing arena harness, migrate `guess_villages`'s production callers
   behind a byte-parity gate, and only then propose 2–3 append-only input channels. The
   channel step is dual-network-sync territory and a checkpoint migration; it should be
   gated on a measured behavior win, not shipped on faith. (§9, §10)

---

## 1. What exists today

| # | Mechanism | Where | Persistent? | Live in production? |
|---|---|---|---|---|
| 1 | `guess_villages` → `Vec<VillageGuess>` | `ai/belief/prediction.rs:90` | **No** — fully recomputed per call | **Yes, unconditionally** |
| 2 | `BeliefState.capital_posterior` | `ai/belief/mod.rs:100`, updated by `on_explored:139` | **Yes** — genuine Bayesian posterior | **No** — dormant |
| 3 | `aux_ownership` / `aux_fog_units` | `train.py:58,214,270,389` | n/a (learned) | Training-only, except `aux_fog` |
| 4 | `PredictionState` / `update_predictions` | `prediction.rs:367`, `states.rs:580,623` | Cached in `GameState._prediction` | Server/UI + `obscure_fog` only |

### 1.1 `guess_villages` — the live one

Two-phase, stateless (`prediction.rs:90-215`):

- **Where** (selection, `:118-152`): every unexplored tile passing
  `validate_village_candidate` (`:23-46`) — no Ocean cardinal neighbour, `edge_dist >= 2 && edge_dist != 3`,
  and Chebyshev ≥3 from every *known* village/city — ranked nearest-to-anchor, then
  spread across distinct quadrants around the anchor centroid (a fix for a measured
  "88% of guesses in one bearing" bug, documented at `:70-77`).
- **How confident / which tribe** (`:157-213`): local evidence only — orphan resource
  adjacent (+5 each), ≥2 resource neighbours (+10), climate-mismatched tile within 2
  (+1, and stamps `guessed_tribe`), Bardur-plus-crop contradiction (−20). Confidence is
  `(0.3 + score/20).clamp(0.05, 1.0)`.

**The gap, precisely.** Evidence is scored *per site, from that site's own neighbourhood*.
It cannot move mass between sites, cannot create a site, and cannot destroy one. The
`climate_evidence` variable (`:167,188`) is last-write-wins inside a loop with no
aggregation. This is exactly the hole Verdi's tile-58 → tile-35 example points at.

**Consumers (all live):** `expand_targets` (`oracle_macro.rs:675-695`) tops up this turn's
Expand orders to `EXPAND_TARGET_MIN = 2` from guessed sites while under
`COMMIT_CITY_TARGET` cities; `compute_macro_goal` reaches it through the same path via
`GoalCache::village_guesses` (`oracle_macro.rs:89-96`, keyed on
`(turn, explored_tile_count)`); `enumerate_candidates_with_belief` (`macro_agent.rs:278`)
uses it for ClaimSafe/Contest directives; `update_predictions` (`prediction.rs:369`)
copies it into `_prediction._villages`.

### 1.2 `BeliefState` — the real posterior, unplugged

`capital_posterior: Vec<(i32, f32)>` seeded from `opponent_capital_prior`
(`mod.rs:63-78`) — uniform over `capital_support_by_quad` (`mod.rs:21-53`, the exact
generator support, test-verified) minus the observer's own quadrant.
`on_explored` (`mod.rs:139-158`) does exactly Verdi's collapse-and-renormalize: drop
disconfirmed cells, renormalize to 1.0, or collapse to a point mass on sighting. Measured
in EXP_ELO_034: **0.33 → 0.86 (t10) → 0.98 (t20), zero wrong collapses.**

It is dormant because `MacroMctsAgent.belief` (`macro_mcts.rs:716`) defaults to `None` and
is only ever set by `set_belief` (`:956`), which only `arena.rs:849` calls, only when
`--macro-belief-mode != Off`. `self_play` never calls it.

**Scoping the falsification (aa27bb5).** Six arms were falsified: 035 MAP
materialization (49.7%), 036 belief-conditioned candidates (gated null, picked 3.8% of
turns), 036b Δφ shaping (45.3%), 037 stickiness, 038 strategist memory (48.6%). The
program verdict is quoted precisely: *"macro directive selection starves for EVALUATION,
not options — a heuristic leaf 2-3 own-turns deep cannot distinguish the futures these
strategies create."* Every one of those arms is a **consumer at the directive-selection
layer**. What was *confirmed* in the same thread is the **representation** (EXP_ELO_034:
capital posterior calibrated, hidden-army MAE halved). **Feeding belief to the network as
a dense input feature has never been run.** Do not let the consumer nulls kill it.

### 1.3 `PredictionState` — the fourth mechanism

`update_predictions` (`prediction.rs:367-392`) writes `_villages` (a lossy re-encoding of
`guess_villages`: `(tile, (tribe, true))`, confidence discarded), `_terrain`, and
`_enemy_capital_suspects` into `GameState._prediction`. Two facts matter:

- `predict_enemy_capitals` (`prediction.rs:398-425`) constructs a **fresh** `BeliefState`
  on every call and immediately reads `capital_top(8)`. It is therefore a **pure prior,
  never updated** — the same cells all game, minus whatever's been explored. The Bayesian
  machinery is one line away and unused.
- `predict_terrain` (`prediction.rs:254-345`) picks a biome by *deterministic
  pseudo-random* draw, `((tile_idx * 12345 + 67890) % 100) / 100.0` (`:322`). That is a
  **sampled point estimate wearing the word "prediction"** — the SSOT must use the biome
  *rates* as probabilities and never call it.

**Verdict:** `PredictionState` becomes a **consumer** of the SSOT (the server/UI's
serialized view), not a peer.

---

## 2. The central question: does `aux_ownership` already solve the ownership piece?

**No. Take the position: it does not, and mirroring it is the wrong cheap move.**

The evidence chain:

| Fact | Where |
|---|---|
| `AUX_DIMS['aux_ownership'] = 121`, dense per-tile, `tanh`, MSE, weight `AUX_OWN_W=0.3` | `train.py:58,80,214,270,389` |
| Target tensor is written as `collected_aux_own.push(ownership_from_pov(&result.final_owner, p_id))` | `self_play.rs:4507` |
| `ownership_from_pov` maps `+1` = mine, `-1` = any opponent's, `0` = unowned | `self_play.rs:1078-1090` |
| `final_owner` is snapshotted **once, from the terminal state**, in the game's post-processing | `self_play.rs:2847-2853` |

So `aux_ownership` is **end-of-episode territory, POV-signed** — a *constant label per
(game, POV)* applied to every ply. Its `tanh` range encodes *side*, not confidence: −1 is
"the opponent ends up holding this," +1 "I do," 0 "nobody does."

This is a **spatially-decomposed value head.** It answers "how will the map be divided
when this is over," which is a different question from "who controls that fogged area
right now," and a *completely* different question from "is there an undiscovered village
at tile 81."

Three further reasons mirroring it does not help:

- **It has no village-existence or capital-location semantics at all.** There is no aux
  head for either. `aux_fog_units` is the only present-tense spatial head, and it is about
  enemy *units* (`self_play.rs:800`: "Ground-truth (unfogged) non-invisible enemy-unit
  occupancy **at decision time**"). That contrast — decision-time vs game-end — is the
  whole distinction.
- **It is causally disconnected from inference.** `current_understanding.md:162`, already
  established: *"zeroing `aux_ownership` leaves the value output bit-identical across 256
  states. Aux heads shape only the shared trunk during training."* Mirroring it into
  `network.rs` gets you a tensor to read; making it *matter* is a separate change.
- **The deterministic derivation is exact where the head is approximate.** Climate
  affinity is a closed-form consequence of capital geometry (§3.3) and is *immutable*.
  Spending network capacity learning something a 20-line derivation gives exactly is the
  wrong trade.

**What would be the right learned move, if you wanted one:** re-target `aux_ownership` from
`final_owner` to **decision-time** owner (POV-signed, unfogged) — mechanically a one-line
change at `self_play.rs:4507` reading the live `tile.owner` instead of the terminal
snapshot. That yields a genuine present-tense ownership head, comparable to `aux_fog_units`.
It is a retrain and it destroys the existing head's meaning. **Deferred, not recommended
now** — and note it still would not give you village existence or capital location.

---

## 3. What the generator actually guarantees

Everything below is read off `mapgen.rs`. This is the evidence base the SSOT is built on,
and it is far stronger than what `guess_villages` currently exploits.

### C1 — Maximality (village packing). **The novel core.**

The post-terrain village pass (`mapgen.rs:872-909`) is:

```
loop {
    candidates = { i : is_land[i] && village_map[i]==0 && terrain[i] != Mountain
                       && edge_dist(i) >= 2 && edge_dist(i) != 3
                       && ∀v with village_map[v]>0 : cheb(i,v) >= 3 }
    if candidates.is_empty() { break }
    place a village at a uniformly random candidate
}
```

It terminates **only when no legal tile remains**. Therefore, at generation:

> **Maximality invariant.** For every tile `i` that is land, non-mountain and edge-legal,
> there exists a village or capital `v` with `cheb(i, v) ≤ 2`.

(`distance` in `mapgen.rs` is `get_chebyshev_distance`, aliased at `mapgen.rs:7-8`.
Capitals are seeded `village_map[cap] = 2` at `mapgen.rs:287-289`, so they count as
blockers *and* as satisfiers.)

The observer can run this backwards. For an **explored** tile `i` that is legal
(land, non-mountain, edge-legal) and carries **no** village, and has **no known**
village/city within Chebyshev 2:

> **∃ an undiscovered village inside the disc `D₂(i) = { j : cheb(i,j) ≤ 2 }`.**

Its unexplored, spacing-legal members are the only places that can satisfy it.

This is Verdi's "reveal 81, it's empty, mass moves to 70 and 59" — but *derived*, with the
correct radius, and **strictly stronger than his sketch**: it also creates mass in discs
you had no candidate in. It is the mechanism that makes evidence at one tile change belief
about a *different, distant* tile.

The complementary half is already in the code: **exclusion** — `p_village(j) = 0` for
`cheb(j, K) ≤ 2` — is what `validate_village_candidate`'s `>= 3` check
(`prediction.rs:44`) implements. Today's code has the exclusion and not the existence.
Those are the two halves of a maximal packing.

**Scope.** C1 holds for the quadrant maps' post-terrain loop
(Drylands / Lakes / Archipelago / WaterWorld, `mapgen.rs:872-876`) and for Pangea's fill
loop (`mapgen.rs:678-710`). It does **not** hold for the Lakes/Archipelago *pre-terrain*
pass (`mapgen.rs:317-355`), which places a fixed count at density 0.3/0.1, nor for
Continents' per-landmass phase. **Training runs Drylands** (`self_play.rs:1478`), where
`is_land` is forced all-true (`mapgen.rs:453-458`) and the predicate reduces to
non-mountain + edge + spacing. Gate C1 on `map_type`; fall back to the geometric prior
elsewhere.

### C2 — Resource spawn zone (an even sharper implication)

`mapgen.rs:954-976`: *"Resources exist only within 2 tiles of a village site: full rate at
Chebyshev ≤1 ('inner city territory'), 1/3 of it at distance 2 ('border expansion'), zero
beyond."* The `spawn_zone` array is built exactly that way (`:963-972`) and every resource
roll is gated on `zone != 0` (`:975-977`).

> An explored tile carrying a resource, with no known village/city within Chebyshev 2,
> **proves** an undiscovered village sits within Chebyshev 2 of it.

Same constraint form as C1, far rarer, and near-certain per instance. It also carries a
**shape**: `get_resource_prob(..., inner)` (`mapgen.rs:136-146`) makes the inner ring 3×
the outer, so posterior mass should split **3:1 in favour of `D₁(r)` over the
Chebyshev-2 ring**. That is a generator-derived weighting, not a tuned constant.

Two concrete fidelity bugs in today's code fall out:

- `is_orphan` (`prediction.rs:162`) uses `cheb(res_idx, k) > 1`. The generator says the
  zone is `≤ 2`. A resource two tiles from a known village is **fully explained** and is
  not orphaned — today it scores `+5` anyway.
- The evidence is applied only as a `+5` bump to sites *already selected* by geometry
  (`prediction.rs:169-174`), never to create or move a candidate.

### C3 — Climate is a Voronoi fingerprint of capital geometry

`tile.climate = classic_climate_id(tribe_affinity)` (`mapgen.rs:1299-1302`). And
`tribe_affinity` is assigned by a **simultaneous randomized flood-fill seeded at the
capital cells** (`mapgen.rs:790-828`): each tribe expands one cell per round from a
randomly chosen active frontier cell, 8-neighbourhood (`get_square(cell, 1)`), land
preferred; orphan land falls back to strict nearest-capital (`mapgen.rs:832-852`).

That is a jittered multi-source Chebyshev Voronoi over the land graph. **Verified: nothing
outside `mapgen.rs` ever writes `.climate`** (`grep '\.climate = '` returns mapgen only) —
so climate survives capture and conquest and is a permanent fingerprint of where the
capitals were placed.

This gives a real likelihood for the capital posterior. For an explored land tile `t`
carrying the opponent's climate, and a capital hypothesis cell `k`:

```
P(climate(t) = opp | opp capital at k)  ≈  σ( ( d(t, own_cap) − d(t, k) ) / τ )
```

with `d` = Chebyshev and `τ` the boundary jitter width (≈1 tile; **measure it**, §11 probe
14, don't guess). Tiles carrying *our* climate give the mirror term. Multiply across
explored land tiles, renormalize over the support — the same collapse-and-renormalize
machinery `on_explored` already has, generalized from a `{0,1}` likelihood to a graded one.

**This is Verdi's tile-58 → tile-35 example done properly.** Seeing Imperius climate at 58
does not "bump a nearby guess" — it shifts the *capital posterior* toward support cells
closer to 58 than ours, and that posterior is what attributes the village guessed at 35.

### C4 — Capital support

`capital_support_by_quad` (`mod.rs:21-53`) replicates the generator's quadrant boxes
exactly and is test-verified (`capital_support_matches_generator_*`). On Tiny Drylands 1v1
the support after removing the observer's quadrant is **3 cells** (EXP_ELO_034 measured
the full support as `{24,29,79,84}`). A 3-atom posterior is trivially cheap and C3
typically resolves it within a few reveals — consistent with the measured 0.98 by t20.

### Fidelity caveats

- `validate_village_candidate` (`prediction.rs:23-46`) diverges from the generator
  predicate in two ways. (i) It checks **"no Ocean cardinal neighbour."** That is not a
  generator rule — the generator requires the *tile itself* be land. Read charitably it is
  a **proxy** for the unobservable `is_land(i)`, and a defensible one, but it is applied as
  a **hard filter** where the honest form is a probability, and on Drylands (training) it
  is near-vacuous since `is_land` is forced all-true (`mapgen.rs:453-458`). The SSOT should
  fold it into `P(land)` in the legality mask, not keep it as a veto. (ii) It **omits the
  Mountain exclusion** entirely, which is a genuine miss — mountain rates run 0.14–0.20 by
  tribe, so it over-admits ~15–20% of fog tiles. Both currently shape production expansion
  targeting, so fix them inside the SSOT as a **separately measured delta**
  (§10, Stage 1b).
- Villages are always placed on Field (forest is converted, `mapgen.rs:903-906`), so a
  revealed **Forest** tile is additional proof of "never a village here," and a revealed
  **Mountain** tile proves the tile was never legal and **voids** its C1 constraint.
- **Real Polytopia uses ≥2 village spacing; ours uses ≥3**, and the real game has *more*
  villages (`mapgen_research.md:189`). So the maximality radius is ≤1, not ≤2, in real
  games. This belief is calibrated to **our** generator. On `mod_replay_*` / live Steam
  states, C1 will over-concentrate. Flag it at every such call site.

---

## 4. The SSOT surface

**Location: `polyfish-rs/src/ai/belief/map.rs`** — extend the existing `ai/belief/`
module, do not create a new tree. Division of labour, stated as an invariant:

- **`MapBelief`** (new) — everything derivable from the *current* observable state:
  village existence, tribe attribution, capital location. **Stateless, recomputable.**
- **`BeliefState`** (existing, `mod.rs:95`) — only what genuinely needs *event history*
  and cannot be recomputed: `residual_army_stars`, `hidden_cities`, `hidden_techs`,
  `events` (`mod.rs:163-227`, the score-delta log). Its `capital_posterior` delegates to
  `MapBelief` after Stage 3.
- **`prediction.rs`** shrinks to terrain prediction plus the `PredictionState` serializer.

### Types

```rust
pub const N: usize = features::MAP_SIZE * features::MAP_SIZE; // 121

/// Cache key. Deliberately excludes `turn`: nothing in the derivation reads it,
/// and excluding it makes the belief stable across a simulated turn advance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeliefKey {
    explored: u32,     // explored_tile_count(state, observer)
    known_sites: u32,  // explored villages + all observer-visible cities
}

/// Why a tile carries the mass it does — telemetry, UI overlay, and the only
/// honest way to debug a probabilistic derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    Prior,              // generator legality only
    Packing(i32),       // C1: forced by the explored empty legal tile at idx
    ResourceZone(i32),  // C2: forced by the orphan resource at idx
    Sighted,            // a real village the observer has explored
}

/// What: the single source of truth for everything the observer believes about
/// the map it cannot see — where undiscovered villages are, whose region a tile
/// belongs to, and where the enemy capital sits.
/// How: a pure derivation from the observer's explored set against the map
/// generator's own placement rules (packing maximality, resource spawn zones,
/// climate Voronoi, capital quadrant support). No history, no persistence.
pub struct MapBelief {
    pub observer: PlayerId,
    pub opponent: PlayerId,
    size: i32,
    key: BeliefKey,

    /// P(an undiscovered village site sits here). Sighted villages are 1.0.
    village: Box<[f32; N]>,
    /// P(this tile's generator affinity region is the opponent's).
    affinity: Box<[f32; N]>,
    /// Opponent-capital posterior, dense (0.0 off-support). Sums to 1.
    capital: Box<[f32; N]>,
    capital_confirmed: Option<i32>,

    why: Vec<(i32, Evidence)>,
}
```

### Functions

```rust
impl MapBelief {
    /// The ONLY constructor. Pure: same explored set in, same belief out.
    pub fn observe(state: &GameState, observer: PlayerId) -> MapBelief;

    /// Cheap fingerprint for cache invalidation; never allocates the grids.
    pub fn key_of(state: &GameState, observer: PlayerId) -> BeliefKey;

    pub fn p_village(&self, idx: i32) -> f32;
    pub fn p_opponent_affinity(&self, idx: i32) -> f32;
    pub fn p_capital(&self, idx: i32) -> f32;

    /// Confirmed sighting if any, else the posterior argmax.
    pub fn capital_map(&self) -> Option<i32>;
    /// Mass on the MAP cell — preserves `BeliefState::capital_confidence` semantics.
    pub fn capital_confidence(&self) -> f32;
    pub fn capital_top(&self, n: usize) -> Vec<(i32, f32)>;

    /// Legacy shape, for the `guess_villages` migration. Applies the same
    /// mutual-spacing and quadrant-spread selection today's picker uses, then
    /// fills `confidence` from `p_village` and `tribe` from `affinity`.
    pub fn top_village_sites(&self, max_sites: usize) -> Vec<VillageGuess>;

    pub fn evidence_at(&self, idx: i32) -> Option<Evidence>;
}

/// Turn-scoped memo, the `GoalCache` pattern (`oracle_macro.rs:74-96`).
#[derive(Default, Clone, Debug)]
pub struct MapBeliefCache { key: Option<BeliefKey>, belief: Option<Arc<MapBelief>> }

impl MapBeliefCache {
    pub fn get(&mut self, state: &GameState, observer: PlayerId) -> Arc<MapBelief>;
}
```

### Two rules that make this safe inside search

1. **`BeliefKey` omits `turn`.** Verified: nothing in the derivation reads
   `settings.turn` (`guess_villages` doesn't; the constraints don't). Including it — as
   `GoalCache` does (`oracle_macro.rs:90`) — would force a recompute on every simulated
   turn advance. Omitting it makes the key constant through a whole search, because
   **sims never mutate `explorers`** (`discovery.rs:81-110`: the simulating branch writes
   only to the `_sim_explored` shadow set and leaves `explorers` untouched). The belief is
   frozen during search *by construction* — the `enemy_ghosts` archetype's guarantee
   without any of its real-moves-only discipline.
2. **`known_sites` is in the key** because capturing a village turns it into a city
   without changing the explored count, and cities are spacing sources
   (`prediction.rs:114-118`). Today's `GoalCache` key misses this — a small, real
   staleness hole. Note that this makes the SSOT *fresher* than the cached path, so
   parity (§10) must be measured against the **uncached** `guess_villages`.
   Corollary: inside a tree, take an `Arc<MapBelief>` computed at the **root** and hold it
   — do not recompute per node, where `state.structures` mutates hypothetically.

---

## 5. Representation: dense, with a sparse view on top

**Decision: dense `[f32; 121]` per signal, with `top_village_sites()` as the sparse
adapter.**

- **Memory:** 3 × 121 × 4 B = **1452 bytes** per belief. One per real ply, `Arc`-shared
  across the tree. This is not a consideration at 11×11.
- **Update cost:** dense makes the constraint sweeps trivial array indexing. Sparse would
  need a candidate set that *grows* as C1/C2 create mass in discs with no prior candidate
  — which is the entire point of the design. Sparse is the wrong shape for a mechanism
  whose job is to discover new locations.
- **Network readiness:** if Stage 4 fires, a dense grid *is already the channel*.
  A sparse list would need a rasterization step with its own bugs.
- **Sparse is still what consumers want.** `expand_targets` needs ≤2 tiles, not a
  distribution. So the sparse view is an accessor, not the representation — which also
  keeps the migration honest (§10).

The one thing kept sparse is `why: Vec<(i32, Evidence)>` — provenance is naturally sparse
and only exists for telemetry and the UI overlay.

---

## 6. The Bayesian merge mechanic

`BeliefState::on_explored` (`mod.rs:139-158`) generalizes from one entity to a field in
seven steps. All of it runs inside `observe()`.

**Step 1 — legality mask `L(i) ∈ [0,1]`**, the generator's placement predicate:
`edge_dist >= 2 && edge_dist != 3` (deterministic, known for all tiles) × `P(land)` ×
`P(non-mountain)` × `[cheb(i, K) >= 3]`. For explored tiles the terrain terms are 0 or 1.
For fog tiles on Drylands `P(land) = 1` (`mapgen.rs:453-458`); elsewhere use
`predict_terrain`'s adaptive local-land estimate (`prediction.rs:293-303`). `P(non-mountain)`
comes from `get_tribe_biome_rates` weighted by the affinity posterior — **as a rate, never
via the `(idx*12345+67890)%100` pseudo-draw**.

**Step 2 — direct collapse.** Explored tile with a Village structure → `1.0`, evidence
`Sighted`. Explored tile without one → `0.0`. Exact; no probability involved. This is the
"reveal 81, it's empty, confidence collapses to 0" half, and today's code already gets it
right by filtering candidates to `!explored(idx)`.

**Step 3 — exclusion.** `p_village(j) = 0` for all `j` with `cheb(j, K) ≤ 2`. Already
implemented as `validate_village_candidate`'s ≥3 check.

**Step 4 — C1 existence constraints.** For each explored tile `i` with `L_terrain(i) = 1`,
no village on it, and no known site within 2: emit a constraint over
`S_i = { j ∈ D₂(i) : j unexplored, L(j) > 0 }` requiring `Σ_{j ∈ S_i} p_village(j) ≥ 1`.

**Step 5 — C2 existence constraints.** For each explored tile `r` carrying a resource with
no known site within 2: emit a constraint over `S_r = { j ∈ D₂(r) : j unexplored, L(j) > 0 }`
with per-member weight `3` for `cheb(j,r) ≤ 1` and `1` for `cheb(j,r) = 2`, requiring
`Σ p_village ≥ 1`.

**Step 6 — reconcile by iterative proportional fitting.** Initialize
`p_village(j) ← L(j) · p_base` (`p_base` = the generator's marginal village density on
legal tiles, measurable offline — see §11 probe 12). Then run **3 sweeps**: for each
constraint whose weighted sum is `< 1`, scale its support up proportionally; clamp every
tile to `[0, 1]` between sweeps. Three sweeps is enough at this scale (constraint supports
are ≤25 tiles and overlap shallowly); it is a knob, not a convergence risk.

> **Mass conservation, stated as an invariant:** a reveal never *increases* total village
> mass in a region. Revealing tile 81 as empty removes 81's own mass and *forces* the
> remainder onto `D₂(81)`. This is the precise version of Verdi's ".75 → 0, and 70/59 rise
> to .83," and it is tested (§11 test 7).

**Step 7 — C3 → capital posterior → affinity.** Start from `opponent_capital_prior`
(`mod.rs:63-78`). Multiply in the C3 likelihood over every explored land tile whose
climate is known and non-zero. Renormalize. If the opponent's capital has actually been
sighted, collapse to a point mass — **preserving `on_explored`'s existing branch exactly**
(`mod.rs:140-144`). Then `affinity(i) = Σ_k p_capital(k) · [ d(i,k) < d(i, own_cap) ]`,
softened by the same `τ`, giving a per-tile P(opponent's region) for **every** tile
including fog. `VillageGuess.tribe` becomes `affinity(site) > 0.5 ? Some(opponent) : None`
— an aggregated posterior instead of today's last-write-wins climate stamp
(`prediction.rs:167-188`).

---

## 7. Ownership probability — three different things, named apart

The word "ownership" is doing three jobs in this codebase. The SSOT should keep them
distinct and never let a caller confuse them:

| Sense | Question | Where it lives | Status |
|---|---|---|---|
| (a) **Affinity** | "Whose *region* of the generated map is this tile?" | `MapBelief::p_opponent_affinity` | **Build now** — derives from C3 + the capital posterior; defined for every tile including fog; immutable ground truth exists (`tile.climate`) so it is directly calibratable |
| (b) **Territory** | "Is this tile inside an opponent city's border *right now*?" | `MapBelief` stub | **Stage 3** — needs believed cities (capital hypothesis + a captured-village model), then `get_square(city, border_size)` |
| (c) **Terminal ownership** | "Who will own this tile when the game ends?" | `aux_ownership` (`train.py:214`) | **Already exists, learned, unrelated to (a)/(b)** |

Verdi's stated ask — *"per-tile/per-village probability of which tribe controls that
area"* — is **(a)** for the fog-heavy early game and **(b)** for contested midgame. `(c)`
is neither. v1 ships (a); (b) returns known territory only until Stage 3.

---

## 8. Capital signal

`MapBelief` **absorbs** `capital_posterior`; it does not wrap it.

- The prior (`opponent_capital_prior`, `mod.rs:63-78`) and support
  (`capital_support_by_quad`, `mod.rs:21-53`) move into `MapBelief::observe` unchanged —
  they are already exact and test-verified.
- The `{0,1}` elimination update in `on_explored` (`mod.rs:145-157`) is preserved as the
  special case of the C3 likelihood where an explored support cell has no capital
  (likelihood 0).
- The sighting collapse (`mod.rs:140-144`) is preserved verbatim.
- **What is new is C3**: today the posterior only *shrinks* by elimination. With C3 it
  also *reweights* by terrain evidence — which is how a climate reveal at tile 58 can
  raise confidence in a capital at 35 without ever exploring 35.
- `BeliefState` keeps `capital_confirmed` as a convenience mirror and delegates
  `capital_top` / `capital_confidence`. `predict_enemy_capitals` (`prediction.rs:398-425`)
  — which builds a fresh, never-updated prior on every call — becomes a one-line
  delegation to `MapBelief::capital_top(n)`, which is a **strict improvement to the
  server/UI path for free**.

**Non-regression bar:** the EXP_ELO_034 measurement (0.33 → 0.86 at t10 → **0.98 at t20**,
zero wrong collapses). C3 must not make it worse. A graded likelihood *can* put mass on
the wrong cell where pure elimination could not — that is the specific risk, and probe 14
(measuring `τ` rather than guessing it) is the mitigation.

---

## 9. Deterministic vs learned, and the network-input path

**Recommendation: deterministic. Verdi's instinct holds, and for sharper reasons than
"it's cheaper."**

1. **`aux_ownership` does not answer the question** (§2), and there is *no* learned head at
   all for village existence or capital location. There is nothing to cheaply mirror.
2. **The signals are exactly computable.** C1/C2/C4 are logical consequences of the
   generator; C3 is a closed-form likelihood with one measurable parameter. Asking a
   network to *learn* what a 20-line derivation gives exactly is the wrong use of
   capacity — and the deterministic version already has a measured 0.98 accuracy result
   (EXP_ELO_034) that a fresh head would have to beat from scratch.
3. **The project's own strongest precedent supports feeding it as input.** The
   pursuit-representation finding (memory `pursuit-failure-representation-gap`): the value
   head was ~blind (3e-5) to village pursuit because progress existed **only as an in-tree
   reward, never as an input feature** — and the fix was *representation* (a feature
   channel + aux target), not more shaping. **Village/capital belief is in the identical
   position today.** `guess_villages`'s output reaches the network only indirectly, via the
   ≤2 guesses that survive into `expand_targets` and get painted as `CH_ORDER_*` proximity
   blobs. The *belief itself* — the distribution, the confidence, the attribution — has
   never been an input.
4. **But the falsification history says: calibrate before you consume.** aa27bb5's lesson
   is not "belief is useless" — 034 confirmed the representation by calibrating offline
   *first*, and the arms that failed were consumers shipped ahead of any evidence they
   would land. Repeat 034's ordering, not 035–038's.

### If/when channels ship (Stage 4)

Append-only at the end of the layout, after `CH_STANCE_END` (`features.rs:174`):

| Channel | Contents |
|---|---|
| `CH_BELIEF_VILLAGE` (1) | `p_village` per tile, `[0,1]` |
| `CH_BELIEF_CAPITAL` (1) | capital posterior per tile, `[0,1]` |
| `CH_BELIEF_AFFINITY` (1) | `p_opponent_affinity`, `[0,1]` — **hold this one back in the first cut**; it is the weakest independently-evidenced of the three |

`NUM_CHANNELS` 169 → 171 (or 172 with affinity). **This is dual-network-sync territory**
per `CLAUDE.md`: `features.rs` painting + `train.py`'s mirrored layout + the zero-pad list
must move together. Two properties make it the *low-risk* form of that change:

- **Channels are append-only, so old training data zero-pads at load** — the established
  convention (154 → 161 → 162 → 169). Not a breaking migration.
- **Input channels never touch `EvalResult`**, so unlike a new *head* they need **no tch
  and no Metal backend plumbing** — sidestepping the `progress`-stubbed-to-0.0 trap
  `CLAUDE.md` warns about. That asymmetry is a genuine argument for input channels over a
  head here.
- **Painting happens inside `encode`**, exactly like the ghost block (`features.rs:782-822`)
  and the pursuit block (`features.rs:824+`). Because the belief is a pure function of
  state, **train/inference parity is automatic** — no agent-side threading, no new
  safetensors target. An agent-side belief would have to be threaded into every `encode`
  call site; that is the decisive argument for keeping it derivable.

**One hard prerequisite:** a channel bump requires migrating `checkpoints/` too — the Rust
opponent loader is strict and does **not** zero-pad (memory `migrate-checkpoints-on-arch-change`),
so an unmigrated `checkpoints/` crashes the first league iteration of the next run.

---

## 10. Migration and staging

`guess_villages` drives every game's early-game expansion targeting. It cannot be deleted,
and its behavior must not drift until we intend it to.

### Stage 0 — instrument, change nothing

Build `MapBelief::observe` in `src/ai/belief/map.rs`. **Nothing in production calls it.**
Extend the arena `--belief-calib` harness (`CalibHarness`, `mod.rs:471-757` — it already
holds ground truth and already logs per-turn rows) to record: Brier on `p_village` against
true village tiles, capital posterior mass on the true cell, and affinity accuracy against
true `tile.climate`.

**Gate:** `p_village` Brier must beat today's flat 0.3-floor confidence, and the capital
posterior must be **≥ EXP_ELO_034's baseline** at t10/t20. This is the step the 035–038
consumers skipped and 034 did not.

### Stage 1a — parity adapter, bugs preserved

Reimplement `guess_villages` as `MapBelief::observe(..).top_village_sites(max_sites)`,
**deliberately reproducing today's known-wrong rules** (the Ocean-cardinal check, the
`is_orphan` radius-1, the missing Mountain exclusion). Gate on a pinned parity harness:
replay a corpus of self-play states through both paths and assert **byte-identical**
`Vec<VillageGuess>` — tile order, `tribe`, and `confidence`. Compare against the
**uncached** path (§4, rule 2).

### Stage 1b — fidelity fixes, measured as their own delta

Fix the three: demote the Ocean-cardinal veto into `P(land)`, add the Mountain exclusion,
widen `is_orphan` to the generator's `> 2`. **These change production expansion targeting**
— measure as a paired A/B on the frozen seed-770425 gauge (memory
`seed-770425-gauge-harness`). Do not bundle with Stage 1a; the whole point of 1a is that it
is a no-op.

### Stage 2 — turn on the new information, scripted consumer first

Enable C1/C2/C3 and let `expand_targets` (`oracle_macro.rs:685-695`) rank guesses by
`p_village` rather than nearest-anchor. This is the first real behavior change, and it
targets the **scripted executor that actually drives production expansion** — not
`enumerate_candidates_with_belief` (`macro_agent.rs:225`), which aa27bb5 measured at 3.8%
of picks.

Two guardrails: **keep the quadrant-spread tie-break** (`prediction.rs:135-152`) — it
exists to fix a measured "88% of guesses in one bearing" bug and `p_village` ranking could
reintroduce it. And judge on **behavior curves** (t2c speed, cities at t15/t25,
first-capture turn), not the 64-game gauge (memory `metrics-noise-floor`: a ±12pp ruler).

### Stage 3 — fold in the capital signal and demote the peers

`BeliefState::capital_posterior` / `on_explored` become thin delegations;
`predict_enemy_capitals` becomes `MapBelief::capital_top(n)`; `PredictionState` /
`update_predictions` becomes a pure serializer of the belief for the server/UI. `BeliefState`
keeps only `residual_army_stars` / `hidden_cities` / `hidden_techs` / `events`, which
genuinely need history. Ship the web-UI belief overlay in this stage — it is the cheapest
possible way for Verdi to eyeball whether the grid is sane.

### Stage 4 — network channels

Gated on Stage 2 showing a behavior win *and* Stage 0's calibration holding. Migrate
`checkpoints/` in the same commit.

---

## 11. Verification plan

### Generator ground-truth probes — **run these first**, `#[ignore]` (the `test_min_capital_distance_1v1` pattern)

12. `maximality_holds_on_1000_generated_drylands_maps` — for every land, non-mountain,
    edge-legal tile, assert a village or capital within Chebyshev 2. **The entire C1
    design rests on this. If it fails, C1 dies and the design reduces to C2 + C3.**
    Also record the marginal village density on legal tiles → `p_base` (§6, step 6).
13. `resources_only_within_2_of_a_village_on_1000_maps`, and measure the realized
    inner:outer ratio against the nominal 3:1.
14. `climate_boundary_width` — over generated maps, fit the logistic width `τ` in
    `P(affinity = tribe k | d differences)`. **Measure `τ`; do not guess it.**

### Unit tests on the merge mechanics

1. `explored_empty_legal_tile_creates_existence_mass_in_disc2` — assert `Σ p_village` over
   the unexplored legal members of `D₂` reaches 1, and a tile outside `D₂` is untouched.
2. `revealed_mountain_voids_the_existence_constraint`.
3. `known_village_within_2_discharges_the_constraint`.
4. `revealed_resource_creates_mass_and_weights_inner_ring_3x`.
5. `resource_at_chebyshev_2_from_a_known_village_is_not_orphaned` (the `is_orphan` fix).
6. `spacing_exclusion_zeros_disc2_of_every_known_site`.
7. `mass_is_conserved_under_reveal` — Verdi's exact scenario: three sites at .75, reveal
   one as empty, assert it goes to 0, the others rise, and total regional mass does not
   increase.
8. `climate_evidence_moves_capital_posterior_toward_the_nearer_support_cell` — plus the
   negative control: our own climate moves it away.
9. `capital_sighting_collapses_to_a_point_mass` — preserves `mod.rs:140-144` exactly.
10. `belief_is_a_pure_function_of_the_explored_set` — same final explored set reached by
    two different reveal orders ⇒ identical belief. **This is the test that licenses "no
    persistence, no serialization, no `obscure_fog` handling."**
11. `belief_key_is_stable_across_a_simulated_turn` — `obscure_fog` a state, run an EndTurn
    fast-forward, assert `key_of` is unchanged. Proves the search freeze.

### Regression

15. The Stage-1a parity harness (byte-identical `VillageGuess` on a state corpus).
16. `cargo test --lib --tests --bin self_play` green. ⚠️ Build with `--features apple` or a
    scratch `CARGO_TARGET_DIR` if a training run may be live (memory
    `shared-target-dir-clobbers-training`).
17. Paired seed-770425 gauge A/B at each behavior-changing stage (1b, 2), judged on
    behavior curves. Pre-register each in `hypothesis_driven_improvements.md` first
    (memory `hypothesis-driven-loop`).

---

## 12. Cost — the O(n²) concern, with numbers

Per `observe()` call at 121 tiles, `|K| ≈ 4-10`, `|E| ≤ 121`:

| Step | Cost | Worst case |
|---|---|---|
| Legality mask | `O(N · |K|)` | 121 × 10 = 1 210 |
| Direct collapse + exclusion | `O(N + |K| · 25)` | ~370 |
| C1 constraints (emit) | `O(|E| · 25)` | 3 025 |
| C2 constraints (emit) | `O(|R| · 25)` | ~750 |
| IPF, 3 sweeps | `3 · O(|C| · 25)` | ~9 000 |
| C3 capital update | `O(|support| · |E|)` | 12 × 121 = 1 452 |
| Affinity fill | `O(N · |support|)` | 1 452 |
| **Total** | | **≈ 17 000 float ops** |

One `PolyZeroNet` forward is on the order of 10⁷ FLOPs, and self-play does one per MCTS
node. **The belief recompute is ≈0.2% of a single network evaluation**, and it runs
**once per real ply** (the key is search-stable, §4). Today's `guess_villages` is already
`O(N · |K|)` selection plus `O(picks · 25)` evidence and was cached precisely because it
ran on every ply of a turn (`oracle_macro.rs:62-73`, EXP_ELO_056) — same order, same
cache, no new cost class.

**The genuinely quadratic thing is avoided by construction.** Naive "propagate evidence
between every pair of candidates" would be `O(|candidates|²)` per ply and would get worse
as the candidate set grows. This design never loops over candidate pairs: propagation
flows through **constraints anchored on explored tiles**, each with a bounded ≤25-tile
support, and the constraint count is bounded by `|E| ≤ 121`. There is no pair loop to
optimize away.

---

## 13. Honesty — what this does not solve

- **It does not make the search evaluate better.** aa27bb5's verdict stands: directive
  selection starves for *evaluation*, not options. A better belief is *more and better
  options*. If the only consumer ends up being directive selection at heuristic-leaf
  strength, expect another null. That is exactly why Stage 2's consumer is the scripted
  `expand_targets` and not `enumerate_candidates_with_belief`.
- **It is calibrated to our generator, not the real game.** ≥3 vs ≥2 spacing
  (`mapgen_research.md:189`) means the C1 radius is wrong on real Steam states. Fine for
  self-play and the simulator; flag it wherever `mod_replay_*` states are involved.
- **C1 is map-type-scoped** (§3). Guard on `map_type`; Drylands (training) is covered,
  Lakes/Archipelago's pre-terrain pass and Continents are not.
- **C1's usefulness inverts over the game.** Early: few explored tiles, so few constraints
  — but wide discs and few known sites. Late: the map is mostly explored and the
  constraints are mostly discharged. The useful window is the early/mid game, which is
  where expansion targeting matters — but do not expect it to help at t30.
- **Terrain under fog is still genuinely uncertain.** On Drylands the land term is free
  but `P(non-mountain)` is not — mountain rates run ~0.14–0.20 by tribe
  (`get_tribe_biome_rates`), so roughly 15–20% of the legality mask on fog tiles is real
  uncertainty and must not be rounded to 1. If it is, C1 will emit constraints that the
  generator never made.
- **Territory-under-fog (sense (b)) is deferred to Stage 3.**
- **Tribe attribution is nearly free in 1v1 and does real work only in FFA.** With one
  opponent, "P(opponent affinity)" is "not ours." The design generalizes; the immediate
  payoff is the *capital posterior it feeds*, not the attribution itself. Do not oversell it.
- **Risk: two sources of truth during Stages 1–3.** `guess_villages` coexists with
  `MapBelief` until Stage 3 completes, held honest only by the parity harness. If the
  migration stalls mid-way the repo is *worse* than before it started. Mitigation: 1a and
  1b are small and should land together or not at all.
- **Risk: C3 can be actively wrong where elimination could only be uninformative.** A
  graded likelihood can put mass on the wrong support cell. The 034 baseline (zero wrong
  collapses) is the non-regression bar, and measuring `τ` (probe 14) rather than guessing
  it is the mitigation.
- **Risk: the IPF reconciliation is a heuristic solver, not exact inference.** It enforces
  the constraints as soft lower bounds over 3 sweeps. It will not produce a calibrated
  joint distribution over village configurations — only per-tile marginals that respect
  the constraints. That is adequate for ranking expansion targets and for a network input
  channel; it is not adequate if someone later wants to *sample worlds* from it (the
  determinization-ensemble idea in `current_understanding.md:77`). Flag it there.
- **Latent leak, pre-existing, worth fixing separately:** `GameState::obscure_fog`
  (`states.rs:691-718`) strips terrain, owner, resources, structures and unit ownership on
  unexplored tiles but **does not strip `tile.climate`**. `features.rs` is safe (all tile
  painting is gated on `is_explored`, `features.rs:505-517`) and `guess_villages` is safe
  (it checks `explored(n)` before reading climate, `prediction.rs:169`) — but **any new
  code that reads `.climate` on an obscured view reads ground truth through fog.** Since
  C3 is built entirely on climate, this must be called out in `MapBelief`'s module doc and
  every climate read gated on `explorers`.

---

## 14. Critical files

| File | Lines | Role |
|---|---|---|
| `polyfish-rs/src/ai/belief/prediction.rs` | 23-46, 52-56, 85-215, 254-345, 367-425 | `validate_village_candidate`, `VillageGuess`, `guess_villages`, `predict_terrain`, `update_predictions`, `predict_enemy_capitals` — the code being absorbed |
| `polyfish-rs/src/ai/belief/mod.rs` | 21-53, 63-78, 95-260, 303-436, 471-757 | Capital support/prior, `BeliefState`, `on_explored`, `materialize_into`, `CalibHarness` |
| `polyfish-rs/src/ai/belief/map.rs` | — | **NEW** — `MapBelief`, `MapBeliefCache`, `BeliefKey`, `Evidence` |
| `polyfish-rs/src/mapgen.rs` | 287-289, 317-355, 678-710, 790-852, 872-909, 954-976, 1299-1302 | The evidence base: capital seeding, pre-terrain pass, Pangea fill, affinity flood-fill, **post-terrain saturation loop (C1)**, **resource spawn zones (C2)**, **climate assignment (C3)** |
| `polyfish-rs/src/ai/oracle_macro.rs` | 62-96, 436-447, 516, 666-695 | `GoalCache` (the caching pattern to mirror), the `guess_villages` re-export, `expand_targets` (**the production consumer to migrate**) |
| `polyfish-rs/src/ai/search/macro_agent.rs` | 90-99, 225-330 | `BeliefMode`, `enumerate_candidates_with_belief` (the falsified consumer — do not target it) |
| `polyfish-rs/src/ai/search/macro_mcts.rs` | 716, 956, 1001-1030 | `belief: Option<BeliefState>`, `set_belief`, the consumption site — why it's dormant |
| `polyfish-rs/src/ai/features.rs` | 42-176, 505-517, 782-822, 824+ | Channel layout, the `is_explored` gate, the **ghost painting block** (Stage-4 template), the pursuit block |
| `polyfish-rs/src/actions/memory.rs` | 12, 34-92, 186-232 | The persistent-belief archetype: real-moves-only, decayed, swept. Studied as a template — and consciously *not* needed here |
| `polyfish-rs/src/actions/discovery.rs` | 23-150, esp. 81-110 | `explorers` written only on real moves; sims use the `_sim_explored` shadow — **why the belief is search-stable for free** |
| `polyfish-rs/src/states.rs` | 376-384, 444, 580-623, 691-745 | `GhostRecord`, `enemy_ghosts`, `PredictionState`, `obscure_fog` (**the climate leak**) |
| `polyfish-rs/src/bin/self_play.rs` | 800, 1078-1090, 1478, 2847-2853, 4507 | `aux_fog_units` doc, `ownership_from_pov`, Drylands default, `final_owner` snapshot, aux target push — **the §2 evidence chain** |
| `polyfish-rs/train.py` | 58, 80, 214, 270, 389, 639-645 | `AUX_DIMS`, `AUX_OWN_W`, `aux_own`/`aux_fog` convs, tanh/MSE, the per-key aux mask |
| `polyfish-rs/src/ai/network.rs` | 177-180, 258-266, 307, 382-392 | The mirrored `aux_fog` head — the shipped template for mirroring a learned per-tile head, if Stage 4 ever needs one |
| `mapgen_research.md` | 71, 122-150, 189 | Resource-zone rule provenance; **the ≥3-vs-≥2 spacing divergence from the real game** |
| `current_understanding.md` | 56-77, 162 | The belief thread's design record; **"zeroing `aux_ownership` leaves the value output bit-identical"** |

---

## Appendix — Verdi's two examples, resolved

**"I believe there could be a village at 81, 70, 59 with confidence .75; I reveal 81 and
it's empty, so it collapses to 0 and the mass redistributes to 70 and 59 (now .83)."**

Correct, and the generator makes it sharper. Revealing 81 as empty **and legal** does not
merely free .75 of mass to be shared — it emits a **hard existence constraint** over
`D₂(81)`. If 70 and 59 are both inside that disc they rise; if only 70 is, 70 rises and 59
does not; **and if neither is, brand-new mass appears on unexplored tiles inside `D₂(81)`
that were never candidates.** If 81 turns out to be a Mountain, nothing happens at all —
the tile was never legal, so its emptiness carries no information.

**"Revealing tile 58 and seeing Imperius-style terrain should raise confidence that a
village guessed near 35 is Imperius-owned or even their capital."**

Correct, and the mechanism is C3, not a local bump. Climate at 58 is a Voronoi label from
the capital flood-fill (`mapgen.rs:790-828`), immutable since generation. Seeing it shifts
the **capital posterior** toward support cells closer to 58 than our own capital is — and
the posterior is what attributes the village at 35, and what raises `p_capital` on the
cells near it. Two distinct tiles, one shared latent variable. That is the propagation all
four existing mechanisms are missing.

---

## 15. Implementation record — Stages 0 and 1a (Aug 24 2026)

Ledger entry: **EXP_ELO_068** in `hypothesis_driven_improvements.md`.

**Shipped:** `polyfish-rs/src/ai/belief/map.rs` (`MapBelief`, `MapBeliefCache`, `BeliefKey`,
`Evidence`, `known_sites`), three `#[ignore]` generator probes in `mapgen::tests`, 15 unit
tests on the merge mechanics, a Stage-1a parity harness built against a **frozen verbatim copy** of the pre-migration
implementation (comparing the live path to a function it delegates to would pass by
construction and pin nothing; both this and the fog gate are **mutation-verified** to fail
when the behaviour they pin is broken), and `MapBelief` metrics in
`CalibHarness::turn_row`. `cargo test --lib --tests --bin self_play`: **280 lib tests + all
integration suites green.** Production behaviour is **unchanged** — nothing consumes the
belief yet.

### 15.1 The probes ran, and C1 lives

| Probe | Result |
|---|---|
| 12 — maximality | **0 violations / 85 551 legal tiles** over 3 000 Tiny Drylands maps (3 tribe pairs × 1 000 seeds). C1 is real. `p_base` measured at **0.1664**. |
| 13 — resource zone | **0 resources outside every village zone** over 1 000 maps. Realized inner:outer **2.69**, not the nominal 3.0 — the pinned constant uses the measured value. |
| 14 — climate | See below; the design's likelihood form was wrong. |

### 15.2 Three places the design was wrong

1. **The C3 likelihood is not a symmetric logistic.** §3's `σ((d_own − d_k)/τ)` predicts
   0.5 at `delta = 0`. The generator produces **0.045** — and `delta = 0` is the single
   largest bucket (23% of tiles). Cause: the affinity flood-fill (`mapgen.rs:790-828`) is a
   round-robin over seats in index order, so **seat 1 wins every tie**. Fitting τ anyway
   gives 0.45 and an 11× error on that bucket. Replaced with a measured 11-entry table
   keyed on the **seat-ordered** distance difference, verified identical across three tribe
   orderings (max spread 0.0000 — the fill is keyed on seat index, not tribe).

2. **Per-tile likelihoods cannot be multiplied.** The tiles are a Voronoi field, not
   independent draws. A flat per-tile temper produced **241 / 2084 calibration rows at
   confidence ≥ 0.9 with the wrong argmax** (pure elimination: zero) and four hard
   collapses onto a wrong cell — exactly the §13 risk. C3 now contributes the **mean**
   log-likelihood scaled by `C3_EVIDENCE`, which bounds total evidence regardless of how
   much is explored.

3. **C3 buys nothing measurable on the training config.** §8 expected C3 to improve the
   capital posterior. Over 4 × ~12 400 calibration rows it moves it by **+0.001 at t10 and
   +0.003 at t20** against elimination alone. The reason is C4, which the doc already
   stated: the support after removing the observer's quadrant is **three cells**, and
   elimination alone nearly saturates it (0.892 @ t10, 0.982 @ t20). Kept on at
   `C3_EVIDENCE = 1.0` — provably regressing nothing (zero overconfident rows, zero wrong
   collapses) — because the support grows with player count and FFA is where it would earn
   its place. Raising it is not free: w = 2 buys +0.010 at t10 but reintroduces
   overconfident rows; w = 3 costs accuracy outright.

### 15.3 Stage-0 gate: PASSED, and the village grid is the real win

Tiny Drylands 1v1, `arena --belief-calib`, 150 seeds × 2 sides, ~12 400 per-turn rows.

| Metric | `MapBelief` | Today's `guess_villages` | Uninformed prior |
|---|---|---|---|
| `p_village` Brier (unexplored tiles) | **0.0224** | 0.0542 | 0.0479 |
| Hidden villages found in top 8 | **1.113** | 0.729 | — |
| (of truly hidden) | **60%** | 41% | — |

The belief is **2.4× better on Brier** and finds **+53% more hidden villages** in the same
number of guesses. Note the legacy guesser scores *worse than the uninformed prior* — its
confidences are actively miscalibrated, which is what the `0.3 + score/20` floor produces.

Capital, paired against the live `BeliefState` (the EXP_ELO_034 baseline) in the same games:
t10 0.877 vs 0.876, t20 0.988 vs 0.984, top-1 0.773 vs 0.772. **≥ baseline everywhere, zero
wrong collapses.** Affinity accuracy ~0.72, essentially flat in `C3_EVIDENCE` — consistent
with §13's "nearly free in 1v1."

### 15.4 Design changes made while implementing

- **`BeliefKey` splits `villages` and `cities`.** §4's single `known_sites` sum is broken: a
  village→city capture decrements one and increments the other, leaving the sum — and the
  explored count — unchanged while the spacing sources really did move. Pinned by
  `key_separates_villages_from_cities`.
- **`top_village_sites` takes `&GameState`.** Selection is anchored on the observer's units
  and cities, which are not part of the belief. Corollary: the purity test (10) applies to
  the **grids**, not to this adapter, which is anchor-dependent by design.
- **Dense grids are `Vec<f32>`, not `Box<[f32; 121]>`.** §5 pins N to `features::MAP_SIZE`,
  but the server/UI path (Stage 3) loads real states at other map sizes.
- **Empty-support constraints are dropped, not scaled.** After exclusion zeroes a disc, a
  C1/C2 support can be empty; IPF would divide by zero.
- **`guess_villages` does NOT build a `MapBelief`.** §10's Stage 1a wording implies routing
  it through `observe()`, which would make every production caller pay the full derivation
  to run legacy selection that consults none of it. The seam is the shared
  `legacy_village_sites`; `MapBelief::top_village_sites_legacy` is for callers already
  holding a belief. Parity is pinned either way.
- **One `known_sites(state, observer)`** for exclusion, discharge and spacing, so C1 cannot
  emit constraints against a site set the exclusion pass disagrees with.

### 15.4b Stage-1b delta, isolated (EXP_ELO_068b)

`Fidelity::LegacyBugs` reverts only the three fidelity rules inside the same derivation.
Over 12 559 rows: **Brier −12.1%, hits@8 +2.3%, guesses on impossible (Mountain) tiles
−61%.** Decomposed against production, the constraint propagation is **91% of the Brier
gain and 94% of the hit-rate gain**; the fidelity fixes are 9% and 6%.

Why 1b alone is weak — and it is a fact about `obscure_fog`, not about the fixes: it
**fabricates** fog terrain (`_prediction._terrain`, default `Field`). So the Ocean veto
inspects invented neighbours (near-vacuous on Drylands), and a Mountain exclusion on a fog
tile can only ever be a *probability*, never a hard filter — i.e. a Stage-2 ranking effect.
Only the `is_orphan` radius is a genuine hard-selection fix. §10's plan to measure 1b as a
standalone behaviour A/B is therefore not worth its cost; ship it with Stage 2 and attribute
with this isolation.

### 15.4c Stage 2 was run and REVERTED (EXP_ELO_069)

`guess_villages` was swapped to `top_village_sites` and measured on the frozen seed-770425
gauge, then replicated at seed 880533. Run 1's apparent mid/late-game gains did **not**
replicate (7 of 15 metrics agreed on sign — coin-flip). What did replicate is a small
regression in the targeted behaviour: slower first village (+0.17 / +0.04 turns), lower
first-village rate (−0.008 / −0.047), slower 2nd city (+0.26 / +0.22), flat-to-fewer village
captures — at **−17.3% self-play throughput**. `anchor_net_wr` +0.023 / +0.031, under the
0.078 floor both times.

**Why §10's Stage 2 instruction is wrong as written.** "Rank guesses by `p_village` rather
than nearest-anchor" collapses two different objectives. `top_village_sites` sorts by
probability with distance only as a tie-break, so the scout is sent to likelier-but-farther
sites and arrives later. A better *estimate* of where villages are is not a better *target* —
the target has to be walked to. The next form to try is expected value per turn of travel
(`p_village` discounted by distance to the nearest anchor), which keeps the calibration win
and restores tempo. Registered as EXP_ELO_070, not yet run.

`guess_villages` is back on the legacy path; `MapBelief` and everything measured in
§15.1–15.4b stands. **Superseded by EXP_ELO_070** (belief PRUNES, distance DECIDES):
`guess_villages` currently routes to `MapBelief::top_village_sites`, pinned by
`tests::parity::guess_villages_entry_point_uses_the_belief_path`. That experiment repaired
2nd-city tempo and first-village rate but not first-village turn or total villages captured;
see the ledger for the numbers and the open revert decision.

### 15.5 What is NOT done

- **Stage 1b** (fidelity fixes: demote the Ocean-cardinal veto into `P(land)`, add the
  Mountain exclusion, widen `is_orphan` to the generator's `> 2`). These change production
  expansion targeting and need a pre-registered paired seed-770425 A/B.
- **Stage 2** (`expand_targets` ranking by `p_village`). `MapBelief::top_village_sites` is
  written and unwired; §15.3 is its evidence, but the behaviour A/B is the gate.
- **Stage 3** (fold `BeliefState::capital_posterior`, `predict_enemy_capitals` and
  `PredictionState` into the SSOT; ship the UI overlay).
- **Stage 4** (network input channels + `checkpoints/` migration).

⚠️ Until Stage 3 lands, `guess_villages` and `MapBelief` are two sources of truth held
honest only by the parity harness — the §13 risk. 1a and 1b should land together.

### 15.6 File layout (Aug 24 2026)

`map.rs` reached ~2 050 lines and was split into `polyfish-rs/src/ai/belief/map/` with no
behaviour change. `mod.rs` keeps the evidence-base and fog-discipline header and re-exports
every previously-public name, so no consumer path moved:

| file | holds |
| --- | --- |
| `params.rs` | every measured or tuned constant, each with its probe |
| `rules.rs` | the generator's placement predicates + `is_explored`/`known_sites` |
| `belief.rs` | the `MapBelief` type, `observe`/`observe_with`, all accessors |
| `ctx.rs` | the fog-gated observation context — every fog-sensitive read |
| `capital.rs` | C4 → C3 → elimination: capital posterior + affinity field |
| `villages.rs` | legality mask, C1/C2 constraint emission, IPF reconciliation |
| `targets.rs` | consumer views (`top_village_sites`, `top_village_sites_legacy`) |
| `cache.rs` | `BeliefKey`, `key_of`, `MapBeliefCache`, the thread-local memo |
| `tests/` | `fixtures`, `villages`, `capital`, `purity`, `parity` |

Verified bit-identical, not merely green: a corpus of 150 replayed states was **snapshotted
to JSON first** (move-gen order is not reproducible across processes, so a regenerated
corpus would have made the diff noise), then every grid, evidence tag, cache key and
consumer-view result was dumped before and after. The move and the DRY pass each diffed to
zero. The DRY pass folded three copies of the explored-tile closure into `is_explored`, four
copies of the known-site distance test into `Ctx::known_within`, a double-evaluated
`has_resource` in `emit_constraints`, and three copies of the bounds-checked grid read.
