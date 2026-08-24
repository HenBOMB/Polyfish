//! Map belief SSOT — one surface for village existence, tribe attribution and
//! the enemy capital, derived from the map generator's own placement rules.
//!
//! [`MapBelief`] is a **pure function of the observer's explored set**: same
//! tiles revealed, same belief, regardless of the order they were revealed in
//! or what happened in between. That is what lets it be recomputed anywhere
//! (including inside `self_play`) with no `GameState` field, no serialization
//! and no `obscure_fog` handling — and it is why the belief is frozen for free
//! during search, since simulated moves never write `explorers`
//! (`actions/discovery.rs`).
//!
//! The evidence base is the generator (`mapgen.rs`), measured by the `#[ignore]`
//! probes in `mapgen::tests`:
//!
//! - **C1 (maximality)** — the post-terrain village pass runs to saturation, so
//!   every land/non-mountain/edge-legal tile lies within Chebyshev 2 of a
//!   village or capital. Revealing such a tile as *empty* therefore PROVES an
//!   undiscovered village sits in its radius-2 disc. Verified at 0 violations
//!   over 85 551 legal tiles (`maximality_holds_on_generated_drylands_maps`).
//! - **C2 (resource zone)** — resources spawn only within Chebyshev 2 of a
//!   village site, at a measured 2.69:1 inner:outer per-tile rate. A revealed
//!   orphan resource proves the same thing, more sharply.
//! - **C3 (climate Voronoi)** — `tile.climate` is a jittered multi-source
//!   flood-fill seeded at the capitals and is never rewritten afterwards, so it
//!   is a permanent fingerprint of capital geometry and a direct likelihood on
//!   capital location.
//!
//! ⚠️ **Fog discipline.** `GameState::obscure_fog` strips terrain, owner,
//! resources and structures on unexplored tiles but does **not** strip
//! `tile.climate`. Every climate read in this module is gated on `explorers`;
//! any new one must be too, or C3 reads ground truth through fog.
//!
//! ⚠️ **Calibrated to OUR generator, not the real game.** Real Polytopia packs
//! villages at ≥2 spacing where we use ≥3 (`mapgen_research.md:189`), so the C1
//! radius is wrong on `mod_replay_*` / live Steam states. C1 is additionally
//! gated on `map_type` — see [`c1_applies`].
//!
//! Where things live: `params` the measured constants, `rules` the generator's
//! placement predicates, `belief` the type and its accessors, `ctx` the
//! fog-gated observation context, `capital`/`villages` the two derivations,
//! `targets` the consumer views, `cache` the fingerprint and its two caches.

mod belief;
mod cache;
mod capital;
mod ctx;
mod params;
mod rules;
mod targets;
mod villages;

#[cfg(test)]
mod tests;

pub use belief::{Evidence, Fidelity, MapBelief};
pub use cache::{observe_cached, BeliefKey, MapBeliefCache};
pub use params::{C3_EVIDENCE, P_BASE, RESOURCE_INNER_OUTER_RATIO};
pub use rules::{c1_applies, edge_legal, known_sites};
