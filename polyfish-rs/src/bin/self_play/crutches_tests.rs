//! Split to a separate file so crutches.rs stays one readable unit,
//! matching the goal_potential_tests.rs / goal_aux_tests.rs convention.

    use super::*;

    #[test]
    fn decays_toward_floor_before_taper() {
        let w0 = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 0, 150, false);
        assert!(
            (w0 - HEURISTIC_PRIOR_W0).abs() < 1e-6,
            "iteration 0 should equal w0, got {w0}"
        );

        let mid = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 23, 150, false);
        assert!(
            (mid - 0.25).abs() < 0.01,
            "iteration 23 should be ~0.25, got {mid}"
        );

        let floored = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 100, 150, false);
        assert!(
            (floored - CRUTCH_FLOOR).abs() < 1e-6,
            "past-decay iteration should sit at the floor, got {floored}"
        );
    }

    #[test]
    fn hard_cuts_to_zero_at_decay_last_iter() {
        let at_cutoff = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 150, 150, false);
        assert_eq!(at_cutoff, 0.0);

        let past_cutoff = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 500, 150, false);
        assert_eq!(past_cutoff, 0.0);

        let just_before = decay_crutch(HEURISTIC_PRIOR_W0, HEURISTIC_PRIOR_DECAY, 149, 150, false);
        assert!((just_before - CRUTCH_FLOOR).abs() < 1e-6);
    }

    #[test]
    fn force_zero_overrides_regardless_of_iteration() {
        let forced = decay_crutch(
            HEURISTIC_PRIOR_W0,
            HEURISTIC_PRIOR_DECAY,
            0,
            usize::MAX,
            true,
        );
        assert_eq!(forced, 0.0);
    }
