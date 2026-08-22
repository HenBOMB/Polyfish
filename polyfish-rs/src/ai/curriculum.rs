//! Iteration-dependent search knobs for training.
//!
//! `self_play` applies these; `arena` must grade the searcher they produce.
//! Both read this module so the gauge cannot silently drift onto a different
//! agent than the one training generates.

/// Net & heuristic priors blended 50/50 at the start of a run.
pub const HEURISTIC_PRIOR_W0: f32 = 0.5;
/// Decays 0.5 -> the 0.1 floor by ~iteration 53.
pub const HEURISTIC_PRIOR_DECAY: f32 = 0.97;
/// Same rate as `HEURISTIC_PRIOR_DECAY`, own start value.
pub const ANCHOR_FRAC_DECAY: f32 = 0.97;
/// Intermediate plateau shared by both crutches.
pub const CRUTCH_FLOOR: f32 = 0.1;

/// Ramp (in iterations) for β on σ(Q) in the exported policy targets:
/// β = min(1, iteration/20). Early on the value head's Q ordering is noise
/// that min-max rescaling amplifies to full strength, so π' corrodes the
/// prior; let search re-ranking into the targets only as the head matures.
pub const POLICY_TARGET_Q_RAMP_ITERS: f32 = 20.0;

/// Converged end of both ramps — the knobs a caller with no iteration in hand
/// should grade at. `converged_knobs_match_the_schedules` pins them to the
/// schedules below.
pub const CONVERGED_PRIOR_HEURISTIC_W: f32 = CRUTCH_FLOOR;
pub const CONVERGED_Q_WEIGHT: f32 = 1.0;

/// Exponential decay from `w0` toward `CRUTCH_FLOOR`, then a hard cutover to 0
/// once `iteration >= decay_last_iter`. Shared by `prior_heuristic_weight`
/// (search prior blend) and `anchor_frac` (greedy-anchor game rate) — both are
/// training-time crutches meant to phase out, not asymptote at a floor.
pub fn decay_crutch(w0: f32, decay_rate: f32, iteration: usize, decay_last_iter: usize) -> f32 {
    if iteration >= decay_last_iter {
        return 0.0;
    }
    (w0 * decay_rate.powi(iteration as i32)).max(CRUTCH_FLOOR.min(w0))
}

/// Prior blend self-play searches with at `iteration`.
pub fn prior_heuristic_weight(iteration: usize, decay_last_iter: usize) -> f32 {
    decay_crutch(
        HEURISTIC_PRIOR_W0,
        HEURISTIC_PRIOR_DECAY,
        iteration,
        decay_last_iter,
    )
}

/// σ(Q) weight, in-tree and in the exported policy targets, at `iteration`.
/// The shell driver overrides this with its own `--value-trust` ramp, which is
/// run-relative rather than `ITER_OFFSET`-shifted.
pub fn policy_target_q_weight(iteration: usize) -> f32 {
    (iteration as f32 / POLICY_TARGET_Q_RAMP_ITERS).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decays_from_w0_toward_the_floor() {
        let start = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 0, 150);
        assert!((start - HEURISTIC_PRIOR_W0).abs() < 1e-6, "got {start}");

        let mid = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 23, 150);
        assert!((mid - 0.25).abs() < 0.01, "got {mid}");

        let floored = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 100, 150);
        assert!((floored - CRUTCH_FLOOR).abs() < 1e-6, "got {floored}");
    }

    #[test]
    fn hard_cuts_to_zero_at_decay_last_iter() {
        assert_eq!(
            decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 150, 150),
            0.0
        );
        assert_eq!(
            decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 500, 150),
            0.0
        );
        let just_before = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 149, 150);
        assert!(
            (just_before - CRUTCH_FLOOR).abs() < 1e-6,
            "got {just_before}"
        );
    }

    #[test]
    fn never_raises_a_rate_that_starts_below_the_floor() {
        let probe = decay_crutch(0.05, ANCHOR_FRAC_DECAY, 40, usize::MAX);
        assert!(probe <= 0.05, "got {probe}");
        assert_eq!(decay_crutch(0.0, ANCHOR_FRAC_DECAY, 0, usize::MAX), 0.0);
    }

    #[test]
    fn anchor_decay_start_holds_the_rate_until_the_clock_starts() {
        // The loop passes --anchor-decay-start == iteration until a gauge
        // reading crosses 50% vs greedy, which pins the exponent at 0.
        let held = decay_crutch(0.25, ANCHOR_FRAC_DECAY, 60usize.saturating_sub(60), 150);
        assert!((held - 0.25).abs() < 1e-6, "got {held}");
        let decaying = decay_crutch(0.25, ANCHOR_FRAC_DECAY, 60usize.saturating_sub(40), 150);
        assert!(
            decaying < 0.25 && decaying >= CRUTCH_FLOOR,
            "got {decaying}"
        );
    }

    /// `arena`'s no-schedule defaults are these constants. If a schedule's
    /// converged value moves and the constant does not, the gauge starts
    /// grading a searcher self-play never converges to.
    #[test]
    fn converged_knobs_match_the_schedules() {
        let late = prior_heuristic_weight(400, 500);
        assert!(
            (late - CONVERGED_PRIOR_HEURISTIC_W).abs() < 1e-6,
            "prior schedule converges to {late}, constant says {CONVERGED_PRIOR_HEURISTIC_W}"
        );
        let saturated = policy_target_q_weight(POLICY_TARGET_Q_RAMP_ITERS as usize + 1);
        assert!(
            (saturated - CONVERGED_Q_WEIGHT).abs() < 1e-6,
            "q ramp saturates at {saturated}, constant says {CONVERGED_Q_WEIGHT}"
        );
    }

    /// The drift #32 documents: at the start of a fresh run the two knobs are
    /// nowhere near their converged values, so a gauge that grades the
    /// converged pair is off-instrument for the readings that matter most.
    #[test]
    fn early_iterations_are_far_from_converged() {
        assert!(prior_heuristic_weight(0, 150) > CONVERGED_PRIOR_HEURISTIC_W * 4.0);
        assert_eq!(policy_target_q_weight(0), 0.0);
    }
}
