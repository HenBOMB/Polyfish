# Fog Memory — Spec (v1)

Goal: stop the net being amnesiac about enemy units that disappear into unexplored
tiles (Verdi's rider scenario). Adds 6 spatial channels of decayed "last seen"
enemy info, per POV player, computed from facts stored in `TribeState`.

## Engine facts this design is built on

1. **Exploration is permanent and is the only fog.** Unit encoding gates on
   `tile.explorers.contains(&perspective)` (`features.rs`, unit loop). Explored
   tiles never re-fog, so the *only* way an enemy unit leaves our input tensor
   is by stepping onto a never-explored tile — or dying. Memory therefore only
   needs to answer: "what did I last see, where, and how long ago?"
2. **Real moves are cleanly separated from MCTS sims** via
   `settings._are_you_sure` (set in `Game::play_move`, `game.rs:311`, and
   temporarily inside `post_load`'s initial-vision pass, `:230-236`;
   deliberately NOT set in `Game::simulate_move`, `:385-395`). Memory mutates
   only on real moves → MCTS undo callbacks never need to touch it.
   *(Citations refreshed Aug 18, 2026 — the previous `game.rs:120,185` / `:263`
   had drifted.)*
3. **Features are re-encoded from `GameState` at every use site** (self_play
   history steps, MCTS leaf eval, server). If memory lives inside the state and
   `state_to_cpu_features` reads it, every consumer gets it for free — no
   changes to safetensors game files, mapper, or training targets.

## Data model — store facts, derive decay

No per-turn decay pass. Store raw observations; compute decay at encode time
from `turn - last_seen_turn`. Update is upsert-only + pruning.

```rust
// states.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemUnit {
    pub unit_type: UnitType,
    pub health: f32,          // as last seen
    pub owner: PlayerId,
    pub last_seen_turn: i32,
}

// TribeState — both fields #[serde(default)] so old JSON/replays/mod captures load.
pub memory_units: IndexMap<i32, MemUnit>,   // tile idx -> last enemy unit seen there
pub memory_attacks: IndexMap<i32, i32>,     // tile idx -> last turn one of MY units was hit there
```

IndexMap (not HashMap) to match the rest of `GameState` and keep serialization
order deterministic. Keys are tile idx, same convention as `tiles`/`structures`.

## Update rules (all inside real-move path only)

**U1 — observe.** After every real move (`Game::play_move`, after the move
executes, `_are_you_sure == true`): for every tribe P, for every unit U owned
by someone else, if U's tile is explored by P →
`P.memory_units.insert(idx, MemUnit{...})`. Per-move cost is tiny (≤ ~50 units
× ≤ 4 tribes); this runs on real moves only, never in MCTS sims. Per-move (not
per-turn-boundary) matters: the rider that attacks and retreats *within* the
enemy turn is captured mid-turn, which is the entire point.

**U2 — move-away supersedes.** If the observed unit's `prev_coords` differ from
its current coords, remove P's `memory_units` entry at `prev_coords.idx` before
inserting at the new idx (we watched it move; the ghost travels with it).

**U3 — witnessed death.** When a unit dies on a tile explored by P, remove P's
`memory_units` entry at that idx. Hook where units are removed from
`tribe.units` in the attack/death action (`actions/`); guard on
`_are_you_sure`. Miss this and the net keeps fearing dead riders.

**U4 — attacked-from-fog.** When one of P's units takes damage, set
`P.memory_attacks.insert(defender_idx, current_turn)`. Records "combat happened
to me here" regardless of whether the attacker was ever visible.

**U5 — prune.** At end of each full round (or every U1 pass): drop entries with
`turn - last_seen_turn > MEM_HORIZON` (constant, 8 turns). Keeps maps tiny and
bounds the info to "recent past".

Deliberate v1 non-rules: no ghost diffusion into adjacent fog (net learns
"threat near here" from the decayed marker), no memory update inside MCTS
simulation (leaves inherit the root's memory snapshot — same simplification
Verdi's snapshot+memory input implies).

## Feature encoding — 6 new channels

`features.rs`: new block after city stats. NUM_CHANNELS 136 → 142.

```
CH_MEM_START = CH_CITY_STATS_END          // count: 6
CH_MEM_ENEMY_SEEN     +0  // decay = 0.85^(turn - last_seen_turn), 0 if no entry
CH_MEM_ENEMY_HP       +1  // last-seen hp / max_hp, gated by same entry
CH_MEM_ENEMY_ATTACK   +2  // unit attack stat / 5.0 (settings/units.rs lookup)
CH_MEM_ENEMY_RANGED   +3  // 1.0 if range > 1
CH_MEM_ENEMY_NAVAL    +4  // 1.0 if naval unit type
CH_MEM_ATTACKED_HERE  +5  // 0.85^(turn - memory_attacks[idx])
NUM_CHANNELS = CH_MEM_END = CH_CITY_STATS_END + 6   // 142
```

Encode loop reads `pov_tribe.memory_units` / `memory_attacks` directly — one
pass over the (small) maps, writing into the tensor; not per-tile scans.

Suppression rule: if the remembered tile currently holds a *visible* enemy unit,
skip channels 0–4 there (live unit channels already cover it; don't double-fire).
`CH_MEM_ATTACKED_HERE` always renders.

Decay base 0.85: readable signal (~0.27) at the 8-turn horizon, near-1
gradient for the first 2–3 turns where the tactical info matters.

## Sync fan-out (the dual-network constraint)

Channel count flows from `features::NUM_CHANNELS` on the Rust side —
`network.rs:176`, `metal_network.rs`, `tch_network.rs` all derive from it, so
they recompile clean. Hardcodes to touch by hand:

| File | Change |
|---|---|
| `polyfish-rs/src/ai/features.rs` | channel block, encode logic, `test_num_channels` |
| `polyfish-rs/src/states.rs` | `MemUnit`, 2 fields on `TribeState` |
| `polyfish-rs/src/game.rs` | U1/U2 hook at end of `play_move` real path |
| `polyfish-rs/src/actions/*` (attack/death) | U3, U4 hooks (guard `_are_you_sure`) |
| `polyfish-rs/train.py:398` | `SPATIAL_CHANNELS = 142` |
| `polyfish-rs/init_model.py:22` | `SPATIAL_CHANNELS = 142` |
| `polyfish-rs/migrate_model.py` | optional: zero-pad `conv1.weight` 136→142 |

Not affected: mapper.rs / policy heads (policy space unchanged), self_play
safetensors schema (features encoded from state), hash.rs (memory is
observation bookkeeping, not game-rule state — keep it out of any state hash,
and it can't desync sims because sims never write it).

## Checkpoint compatibility

`conv1` input width changes ⇒ existing `model.safetensors` won't load.
Either train fresh (`run_training_loop.sh --reset`) or zero-pad: new input
channels start with zero weights = memory invisible to the old net until
training picks it up. Zero-pad preserves prior progress; on an experimental
branch fresh is also fine. Pick one before running.

## Verification

1. `cargo test` — channel-layout tests updated, engine tests untouched.
2. Unit test (rider scenario): enemy unit visible on explored tile → real move
   it into unexplored tile → encode features → assert `CH_MEM_ENEMY_SEEN` ≈
   0.85^Δ at last-seen idx and live unit channels are 0. Kill a visible unit →
   assert ghost cleared (U3).
3. Determinism guard: run a short self_play twice with the same seed → same
   game hashes (memory must not perturb rules/RNG).
4. Throughput: compare self_play games/min before/after; U1 is O(units) per
   real move so expected noise-level.
5. Training smoke: `--reset` short run → value/policy losses behave as before;
   the new channels can only help once conv1 learns to read them.

---

## Aug 18, 2026 — one interaction to know about

`Game::clone_for_mcts` now does more than `obscure_fog` when
`game::adversarial_search()` is on: it also clears `tile.explorers` for every
tile the POV player has not explored, so the in-tree opponent generates moves
against the same tiles we can see rather than against terrain, owners and
resources the obscured state has already erased.

That only touches the **search clone**, never a real state, so nothing here
changes: fact 1 above (exploration is permanent and is the only fog) still
describes the real game, and the memory channels are still computed from
`TribeState` facts recorded on real moves only. Worth knowing if you ever read
`explorers` from inside a search: under adversarial search it means "explored by
the root player", not "explored by this tile's viewer".
