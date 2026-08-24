//! The village-existence grid: the generator's legality predicate as a
//! probability, the C1/C2 existence constraints it licenses, and the IPF
//! pass that reconciles them.

use crate::functions::{get_chebyshev_distance, get_square_indices};
use crate::types::TerrainType;

use super::belief::{Evidence, Fidelity, MapBelief};
use super::ctx::Ctx;
use super::params::{IPF_SWEEPS, P_BASE, RESOURCE_INNER_OUTER_RATIO};
use super::rules::{c1_applies, edge_legal};

/// An existence constraint: at least one undiscovered village lies in `support`.
#[derive(Debug, Clone)]
struct Constraint {
    support: Vec<i32>,
    /// Relative per-member weight; uniform for C1, 2.69:1 inner:outer for C2.
    weights: Vec<f32>,
    source: Evidence,
}

impl MapBelief {
    /// The generator's placement predicate as a probability: edge legality ×
    /// P(land) × P(non-mountain). Exact (0/1) on explored tiles.
    pub(super) fn legality_mask(&self, ctx: &Ctx) -> Vec<f32> {
        let n = self.village.len();
        let mut mask = vec![0.0f32; n];
        for idx in 0..n as i32 {
            if !edge_legal(idx, self.size) {
                continue;
            }
            let i = idx as usize;
            // The legacy Ocean-cardinal veto: not a generator rule, and on an
            // obscured view it reads terrain `obscure_fog` fabricated.
            if ctx.fidelity == Fidelity::LegacyBugs && ctx.has_ocean_cardinal(idx) {
                continue;
            }
            if ctx.explored(idx) {
                let Some(tile) = ctx.state.tiles.get(&idx) else {
                    continue;
                };
                let land = !matches!(tile.terrain_type, TerrainType::Water | TerrainType::Ocean);
                let non_mountain = ctx.fidelity == Fidelity::LegacyBugs
                    || tile.terrain_type != TerrainType::Mountain;
                mask[i] = if land && non_mountain { 1.0 } else { 0.0 };
            } else {
                // Fog. P(non-mountain) is real uncertainty (~0.14-0.20 by
                // tribe) and must NOT be rounded to 1, or C1 emits constraints
                // the generator never made.
                let p_mountain = if ctx.fidelity == Fidelity::LegacyBugs {
                    0.0
                } else {
                    ctx.mountain_rate(self.affinity[i])
                };
                mask[i] = ctx.p_land(idx) * (1.0 - p_mountain);
            }
        }
        mask
    }

    /// Direct collapse → exclusion → C1/C2 constraint emission → IPF.
    pub(super) fn solve_villages(&mut self, ctx: &Ctx, legality: &[f32]) {
        let n = self.village.len();

        // Step 2 — direct collapse. Explored tiles are known, not believed.
        for idx in 0..n as i32 {
            let i = idx as usize;
            if ctx.explored(idx) {
                if ctx.has_village(idx) {
                    self.village[i] = 1.0;
                    self.why.push((idx, Evidence::Sighted));
                }
                continue;
            }
            self.village[i] = legality[i] * P_BASE;
        }

        // Step 3 — exclusion. Nothing can sit within Chebyshev 2 of a known
        // site; this is the half `validate_village_candidate` already had.
        for &k in &ctx.known {
            for j in get_square_indices(k, 2, self.size) {
                if !ctx.explored(j) {
                    self.village[j.max(0) as usize] = 0.0;
                }
            }
        }

        // Steps 4/5 — existence constraints.
        let constraints = self.emit_constraints(ctx, legality);

        // Step 6 — reconcile by iterative proportional fitting. Each
        // constraint is a soft lower bound: at least one village in `support`.
        for _ in 0..IPF_SWEEPS {
            for c in &constraints {
                let held: f32 = c.support.iter().map(|&j| self.village[j as usize]).sum();
                if held >= 1.0 {
                    continue;
                }
                let deficit = 1.0 - held;
                // Allocate the deficit by weight × remaining headroom, so no
                // tile is pushed past 1.0 and the 2.69:1 inner:outer shape of
                // a resource constraint is respected.
                let total: f32 = c
                    .support
                    .iter()
                    .zip(c.weights.iter())
                    .map(|(&j, w)| w * (1.0 - self.village[j as usize]))
                    .sum();
                if total <= 1e-9 {
                    continue;
                }
                for (&j, w) in c.support.iter().zip(c.weights.iter()) {
                    let head = 1.0 - self.village[j as usize];
                    self.village[j as usize] += deficit * w * head / total;
                }
            }
            for v in self.village.iter_mut() {
                *v = v.clamp(0.0, 1.0);
            }
        }

        for c in &constraints {
            for &j in &c.support {
                if self.village[j as usize] > P_BASE && self.evidence_at(j).is_none() {
                    self.why.push((j, c.source));
                }
            }
        }
    }

    /// C1 and C2. Both say "an undiscovered village sits within Chebyshev 2 of
    /// this explored tile"; they differ in what licenses the claim and in how
    /// the mass is shaped across the disc.
    fn emit_constraints(&self, ctx: &Ctx, legality: &[f32]) -> Vec<Constraint> {
        let mut out = Vec::new();
        let c1_on = c1_applies(ctx.state.settings.map_type);

        for idx in 0..self.village.len() as i32 {
            if !ctx.explored(idx) || ctx.has_village(idx) {
                continue;
            }
            // Already explained: a known site inside the disc discharges the
            // constraint. Uses the same `known` set as the exclusion above.
            // LegacyBugs defers this for resource tiles, which it discharges at
            // the wrong radius just below.
            let has_resource = matches!(ctx.state.resources.get(&idx), Some(Some(_)));
            let defer = ctx.fidelity == Fidelity::LegacyBugs && has_resource;
            if !defer && ctx.known_within(idx, 2) {
                continue;
            }

            // The `is_orphan` bug: legacy treats a resource as unexplained
            // unless a known site sits within 1, but the generator's spawn zone
            // is 2, so a resource at distance 2 is already fully accounted for.
            if has_resource && ctx.fidelity == Fidelity::LegacyBugs && ctx.known_within(idx, 1) {
                continue;
            }
            // C1 needs the tile to have been LEGAL at placement time; a
            // revealed mountain or water tile was never legal, so its
            // emptiness carries no information at all.
            let c1 = c1_on && legality[idx as usize] >= 1.0;
            if !c1 && !has_resource {
                continue;
            }

            let mut support = Vec::new();
            let mut weights = Vec::new();
            for j in get_square_indices(idx, 2, self.size) {
                if ctx.explored(j) || legality[j as usize] <= 0.0 {
                    continue;
                }
                // Spacing-excluded tiles were zeroed above and cannot host.
                if ctx.known_within(j, 2) {
                    continue;
                }
                support.push(j);
                weights.push(if has_resource {
                    if get_chebyshev_distance(idx, j, self.size) <= 1 {
                        RESOURCE_INNER_OUTER_RATIO
                    } else {
                        1.0
                    }
                } else {
                    1.0
                });
            }
            // An empty support means every candidate host was ruled out. The
            // constraint cannot be satisfied and scaling into it would divide
            // by zero, so drop it rather than let IPF chase infinity.
            if support.is_empty() {
                continue;
            }
            out.push(Constraint {
                support,
                weights,
                source: if has_resource {
                    Evidence::ResourceZone(idx)
                } else {
                    Evidence::Packing(idx)
                },
            });
        }
        out
    }
}
