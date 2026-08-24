//! Every measured or tuned number the belief depends on, in one place.
//! Each carries the probe or experiment that produced it; none is a guess.

/// Marginal village density on legal tiles, measured over 3 000 Tiny Drylands
/// maps by `mapgen::tests::maximality_holds_on_generated_drylands_maps`.
pub const P_BASE: f32 = 0.1664;

/// Realized inner:outer per-tile resource rate (`resources_only_within_2_of_a_village`).
/// The generator's nominal figure is 3.0; 2.69 is what it actually produces.
pub const RESOURCE_INNER_OUTER_RATIO: f32 = 2.689;

/// P(a tile carries SEAT 2's climate) by seat-ordered `d(t, cap1) - d(t, cap2)`,
/// index 0 = delta −5. Measured, not fitted: the affinity fill is a round-robin
/// in seat order so seat 1 wins ties and the curve is asymmetric (0.045 at delta
/// 0, not 0.5). See `belief_grid_ssot_design.md` §15.2.
const CLIMATE_P_SEAT2: [f32; 11] = [
    0.0002, 0.0013, 0.0030, 0.0197, 0.0879, 0.0447, 0.9080, 0.9796, 0.9966, 0.9995, 1.0000,
];

/// Likelihoods are clamped away from {0,1}: climate is spatially correlated, so
/// one mis-modelled tile must never be able to eliminate the true capital.
pub(super) const LIKELIHOOD_FLOOR: f32 = 1e-3;

/// How many INDEPENDENT observations the whole climate field is worth. The
/// per-tile likelihoods are Voronoi-correlated, so C3 uses the MEAN
/// log-likelihood scaled by this, never the raw product. 0.0 = pure
/// elimination (the EXP_ELO_034 bar). Measured ~nil on Tiny 1v1: see
/// `belief_grid_ssot_design.md` §15.2.
pub const C3_EVIDENCE: f32 = 1.0;

/// Calibration override for [`C3_EVIDENCE`] (`POLYFISH_C3_EVIDENCE`), so the
/// weight can be swept without a rebuild. Unset in normal operation.
pub(super) fn c3_evidence() -> f32 {
    static CACHED: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("POLYFISH_C3_EVIDENCE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(C3_EVIDENCE)
    })
}

/// How many belief-ranked sites enter the distance-ordered pool.
///
/// Belief PRUNES, distance DECIDES. EXP_ELO_069 measured that ranking by
/// `p_village` alone sends the scout to likelier-but-farther sites: slower 2nd
/// city and a lower first-village rate, replicated across two seeds. Expansion
/// is a walk, not a lookup. Smaller = belief steers more; larger = closer to
/// pure nearest-first.
pub(super) const BELIEF_POOL: usize = 8;

/// IPF sweeps used to reconcile the existence constraints. Supports are ≤25
/// tiles and overlap shallowly, so this converges well before it matters.
pub(super) const IPF_SWEEPS: usize = 3;

/// P(a tile carries seat 2's climate) for a seat-ordered distance difference.
pub(super) fn climate_p_seat2(delta: i32) -> f32 {
    let i = (delta + 5).clamp(0, (CLIMATE_P_SEAT2.len() - 1) as i32) as usize;
    CLIMATE_P_SEAT2[i]
}
