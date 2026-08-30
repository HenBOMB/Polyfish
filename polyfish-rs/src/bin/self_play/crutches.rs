//! Training-time crutch schedules. Both the search-prior heuristic blend
//! and the heuristic-anchor game rate decay to a shared floor and then cut
//! hard to zero -- they are scaffolding, not permanent terms.

pub(crate) const HEURISTIC_PRIOR_W0: f32 = 0.5; // net & heur blended 50/50 at start
pub(crate) const HEURISTIC_PRIOR_DECAY: f32 = 0.97; // decays 0.5 -> 0.1 floor by ~iteration 53
pub(crate) const ANCHOR_FRAC_DECAY: f32 = 0.97; // same rate as HEURISTIC_PRIOR_DECAY, own start value
pub(crate) const CRUTCH_FLOOR: f32 = 0.1; // intermediate plateau shared by both crutches below

/// Exponential decay from `w0` toward `CRUTCH_FLOOR`, then a hard cutover to
/// 0 once `iteration >= decay_last_iter` (or immediately if `force_zero`).
/// Shared by `prior_heuristic_weight` (self-play search prior blend) and
/// `anchor_frac` (heuristic-anchor game rate) — both are training-time
/// crutches meant to fully phase out, not asymptote at a permanent floor.
pub(crate) fn decay_crutch(
    w0: f32,
    decay_rate: f32,
    iteration: usize,
    decay_last_iter: usize,
    force_zero: bool,
) -> f32 {
    if force_zero || iteration >= decay_last_iter {
        return 0.0;
    }
    (w0 * decay_rate.powi(iteration as i32)).max(CRUTCH_FLOOR)
}

#[cfg(test)]
#[path = "crutches_tests.rs"]
mod tests;
