# 05. Technology Pruning & Valuation Engine Specification

## 1. Problem Statement & Motivation

In Polytopia, every turn presents a wide action space (often 30–80+ legal actions). When technologies with zero immediate or near-term strategic payoff are treated as viable candidates during tree search, they waste significant compute:
1. **Rollout Inefficiencies**: In `HeuristicMctsAgent`, research moves are expanded and simulated during greedy rollouts even when they yield no board impact.
2. **Top-K Cut Pollution**: In `GumbelMctsAgent`, Gumbel noise $g \sim \text{Gumbel}(0, 1)$ combined with an artificially clamped positive prior elevates dead technologies into the `in_cut` candidate pool, robbing search iterations from decisive tactical moves.
3. **Miscalibrated Heuristics**: Key early-game technologies like **Climbing** are scored purely on static mountain counts in borders, completely ignoring the primary reasons human grandmasters purchase them: **massive fog discovery (+1 vision radius)** and **mountain combat defense**.

---

## 2. Canonical Rule Parity vs. Search Pruning Boundary

> **Game rule legality must remain untouched.**
> `generate_legal_moves` defines legal state transitions. It is consumed by:
> - The replay playback & validation engine (`polyfish-rs/src/replay/executor.rs`), where historical human games containing suboptimal research moves must validate without errors.
> - The web UI (`polyfish-rs/src/main.rs`), where human players must see all legally permissible moves.
> - Fuzzers and undo integrity tests.
>
> **Pruning strictly belongs in the AI Candidate / Search Layer:**
> 1. Heuristic scoring (`polyfish-rs/src/ai/scoring.rs` and `gene_ordering.rs`) assigns suppressed tech a score of `-f32::INFINITY`.
> 2. Gumbel MCTS prior blending (`polyfish-rs/src/ai/gumbel_mcts.rs`) masks suppressed moves to `logit = -f32::INFINITY`, making selection via Gumbel noise impossible.
> 3. Heuristic MCTS (`polyfish-rs/src/ai/heuristic_mcts.rs`) sorts suppressed moves below `EndTurn` (`0.0`), preventing expansion and rollout selection.

---

## 3. Audit of Current Implementation Deficiencies

Our audit of the current codebase identified four structural bugs and omissions:

### 3.1 Outdated Customs House Logic on Trade
In `polyfish-rs/src/ai/evaluator/research.rs`:
Lines 74–90 score `TechnologyType::Trade` based on **water adjacency to cities** (the legacy Customs House mechanic from old Polytopia!). But in modern Polytopia and in `polyfish-rs/src/settings/technology.rs`, Trade unlocks **Markets**, which require adjacency to Sawmills, Windmills, and Forges. The evaluator currently rewards Trade for water tiles it cannot exploit.

### 3.2 Missing Tech Match Arms
In `evaluate_tech_utility` (`polyfish-rs/src/ai/evaluator/research.rs`), the following technologies are absent and fall through to the default arm `_ => (1.0, 2.0)`:
- `TechnologyType::Ramming`
- `TechnologyType::Aquatism`
- `TechnologyType::Construction`
- `TechnologyType::Meditation`
- `TechnologyType::Spiritualism`
- `TechnologyType::Mathematics`
- All special tribe replacement techs (`Frostwork`, `Sledding`, `IceFishing`, `PolarWarfare`, etc.)

### 3.3 Floor Clamping in Move Scoring
In `polyfish-rs/src/ai/scoring.rs`:
```rust
let base = 8.0 + (utility.clamp(-2.0, 8.0) + 2.0);
```
Even if utility is `-3.0` (e.g. Sailing with 0 water), clamping to `-2.0` results in a score of `8.0`. This is higher than `EndTurn` (`0.0`) and unranked abilities (`5.0`), preventing the engine from ever discarding worthless research.

### 3.4 Inadequate Climbing Modeling
`evaluate_tech_utility` evaluates Climbing strictly as mountain count inside `city._territory`. It provides 0 value for mountains outside borders, 0 value for fog reveal, and 0 value for defense bonuses.

---

## 4. Technology Pruning Taxonomy

### 4.1 Tier 1: Provably Zero-Utility (Hard Suppressed: Score = $-\infty$)

A technology is assigned $-\infty$ when its required physical game entity does not exist on the visible map or within the tribe's expansion horizon:

| Technology | Pruning Condition | Strategic Justification |
| :--- | :--- | :--- |
| **Sailing / Ramming / Navigation / Aquatism** | Map is Dryland OR tribe sees 0 water/ocean tiles | Cannot enter water, cannot build ports, cannot spawn naval units. |
| **Fishing** | 0 fish in territory AND 0 accessible water tiles | 0 population gain, 0 movement capability. |
| **Farming** | 0 crop resources in territory or immediate adjacent ring | Farms can *only* be placed on crop tiles. Yields 0 population. |
| **Mining** | 0 metal resources in territory or immediate adjacent ring | Mines can *only* be placed on metal tiles. Yields 0 population. |
| **Forestry** | 0 forest tiles in territory | Lumber Huts and Clear Forest are physically impossible. |
| **Roads** | Tribe owns 1 city AND has 0 captured villages AND 0 visible uncaptured villages | Connecting a capital to itself yields 0 network bonus and 0 SPT. |
| **Diplomacy** | Number of known opponent tribes = 0 | Cannot establish embassies, cannot gain capital vision, Cloaks have no valid targets. |

### 4.2 Tier 2: Dependency / Adjacency Dead Techs (Suppressed until Prerequisites Built)

Tier 3 economic technologies that require Tier 2 structures to provide any value:

| Technology | Pruning Condition | Strategic Justification |
| :--- | :--- | :--- |
| **Trade (Markets)** | Player has 0 Sawmills, 0 Windmills, and 0 Forges | Markets produce stars strictly via adjacency to production buildings. With 0 hubs, Market yields 0 stars/turn. |
| **Mathematics (Sawmills)** | 0 Lumber Huts built AND cannot afford Catapult (8 stars) | Sawmills yield 0 population without adjacent Lumber Huts. |
| **Construction (Windmills)** | 0 Farms built AND no forests to burn | Windmills yield 0 population without adjacent Farms. |
| **Smithery (Forges)** | 0 Mines built AND cannot afford/need Swordsmen (5 stars) | Forges yield 0 population without adjacent Mines. |

### 4.3 Tier 3: Contextual Traps (Heavy De-prioritization)

Technologies that are mechanically functional but mathematically negative ROI in early/mid game:

| Technology | Condition | Strategic Justification |
| :--- | :--- | :--- |
| **Meditation** | Turn < 15 unless pursuing specific score condition | Provides 0 units, 0 stars, 0 population. Severe early tempo sacrifice. |
| **Spiritualism** | Domination mode / Early game | Planting trees (5 stars) to chop (1 star) or build temples (20 stars) is a severe net-negative investment. |
| **Climbing** | All mountains are already explored, inside territory, and have no metal | 0 fog discovery boost, 0 tactical defense value. |

---

## 5. Climbing Overhaul: Vision & Defense Valuation

In high-Elo Polytopia, Climbing derives its early value from **Fog of War discovery** and **tactical terrain defense**, not passive mountain counting.

### 5.1 Vision Discovery Formula
When a unit steps onto a mountain, its sight radius increases from 1 to 2:
- Standard tile sight: $3 \times 3 = 9$ tiles.
- Mountain sight: $5 \times 5 = 25$ tiles (revealing up to 16 additional tiles).

Let $\mathcal{M}_{\text{acc}}$ be the set of mountain tiles accessible to the tribe's units within 1–2 moves. For each mountain $m \in \mathcal{M}_{\text{acc}}$, let $F(m)$ be the count of unexplored fog tiles within radius 2 of $m$:
$$\text{Utility}_{\text{vision}}(\text{Climbing}) = \alpha \cdot \max_{m \in \mathcal{M}_{\text{acc}}} F(m)$$
where $\alpha \approx 0.35$. If an accessible mountain reveals 10+ fog tiles, it yields an immediate $+3.5$ utility boost, capturing the value of locating early villages and ruins.

### 5.2 Tactical Defense Formula
If an accessible mountain is positioned between an owned city and a known enemy unit within 3 tiles:
$$\text{Utility}_{\text{defense}}(\text{Climbing}) = +2.5$$

---

## 6. Implementation Architecture

### Phase 1: Overhaul `evaluate_tech_utility` (`polyfish-rs/src/ai/evaluator/research.rs`)
- Implement all missing tech arms (`Ramming`, `Aquatism`, `Construction`, `Meditation`, `Spiritualism`, `Mathematics`).
- Replace water checking on `Trade` with Market hub adjacency checks (Sawmills, Windmills, Forges).
- Incorporate fog discovery and tactical defense into `Climbing`.
- Return `-f32::INFINITY` when a Tier 1 or Tier 2 pruning condition is met.

### Phase 2: Unclamp Move Scoring (`scoring.rs` & `gene_ordering.rs`)
- If `evaluate_tech_utility` returns `-f32::INFINITY`, bypass base clamping and return `-f32::INFINITY` (or `-100.0`).
- Ensure suppressed research ranks strictly lower than `EndTurn` (`0.0`).

### Phase 3: Prior Masking in Gumbel MCTS (`gumbel_mcts.rs`)
- In `blend_heuristic_into_logits`, if `heur_scores[i] <= -100.0`, force `logits[i] = -f32::INFINITY`.
- This ensures $p_{\text{net}} = 0$ and prevents Gumbel noise sampling into `in_cut`.

### Phase 4: Test Matrix (`polyfish-rs/tests`)
- `test_dryland_sailing_pruned`: Verify Sailing evaluates to $-\infty$ on dry maps.
- `test_climbing_fog_prioritization`: Verify Climbing near fog scores higher than flat-land techs.
- `test_market_trade_pruned_without_huts`: Verify Trade is suppressed when 0 production buildings exist.
