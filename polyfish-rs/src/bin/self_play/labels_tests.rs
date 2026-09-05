//! Split to a separate file so labels.rs stays under the ~1000-line
//! ceiling, matching goal_potential_tests.rs / goal_aux_tests.rs.
//!
//! Was two modules in self_play.rs: td_lambda_tests and aux_target_tests.

use super::*;
use polyfish::coords::Coords;
use polyfish::states::{TechnologyState, TribeState, UnitState};
use polyfish::types::{TechnologyType, UnitEffect};

    fn step(player_id: PlayerId, turn: i32, my: i32, opp: i32, rv: Option<f32>) -> LabelStep {
        step_h(player_id, turn, my, opp, rv, 0.0)
    }

    fn step_h(
        player_id: PlayerId,
        turn: i32,
        my: i32,
        opp: i32,
        rv: Option<f32>,
        heur_value: f32,
    ) -> LabelStep {
        LabelStep {
            player_id,
            turn,
            my_score: my as f32,
            opp_score: opp as f32,
            root_value: rv,
            heur_value,
        }
    }

    fn finals(pairs: &[(i32, i32)]) -> HashMap<i32, f32> {
        pairs.iter().map(|&(id, s)| (id, s as f32)).collect()
    }

    /// A macro (heuristic-leaf) game reports no root value anywhere, so
    /// under `zero` every label is a truncated return pulled toward 0. Under
    /// `mc` the whole weight reaches the terminal return instead.
    #[test]
    fn mc_fallback_recovers_terminal_return_when_all_roots_missing() {
        let history = vec![
            step(1, 5, 1000, 800, None),
            step(1, 6, 1100, 800, None),
            step(1, 7, 1200, 800, None),
        ];
        let final_scores = finals(&[(1, 1600), (2, 900)]);
        let expected = reward::normalized_reward(1000, 800, 1600, 900).clamp(-1.0, 1.0);

        let mc = td_lambda_labels(&history, &final_scores, 0.5, reward::REL_W, None, MissingBootstrap::Mc);
        assert!(
            (mc[0] - expected).abs() < 1e-6,
            "mc label {} should be the pure terminal return {expected}",
            mc[0]
        );
        let zero = td_lambda_labels(&history, &final_scores, 0.5, reward::REL_W, None, MissingBootstrap::Zero);
        assert!(
            (zero[0] - expected).abs() > 1e-6,
            "zero-bootstrap must still truncate (legacy semantics pinned elsewhere)"
        );
    }

    /// `Heur` must not skip a missing-root checkpoint (unlike `Mc`) and must
    /// not zero it (unlike `Zero`) -- it substitutes a calibrated
    /// `evaluate_state` reading as the bootstrap, at lambda=0 so exactly one
    /// checkpoint is exercised and the arithmetic is checkable by hand.
    #[test]
    fn heur_fallback_bootstraps_on_calibrated_evaluate_state_not_mc_or_zero() {
        let history = vec![
            step_h(1, 5, 1000, 800, None, 0.1), // i (heur_value unused: not a checkpoint bootstrap source here)
            step_h(1, 6, 1100, 800, None, 0.3), // checkpoint n=1, root_value missing
        ];
        let final_scores = finals(&[(1, 5000), (2, 800)]);

        let heur = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W, None, MissingBootstrap::Heur);
        let mc = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W, None, MissingBootstrap::Mc);
        let zero = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W, None, MissingBootstrap::Zero);

        let r = reward::normalized_reward(1000, 800, 1100, 800);
        let bootstrap = (HEUR_TO_OUTCOME_SLOPE * 0.3 + HEUR_TO_OUTCOME_INTERCEPT).clamp(-1.0, 1.0);
        let expected = (r + reward::GAMMA_TURN.powi(1) * bootstrap).clamp(-1.0, 1.0);

        assert!(
            (heur[0] - expected).abs() < 1e-6,
            "heur label {} should bootstrap on the calibrated evaluate_state reading {expected}",
            heur[0]
        );
        assert!(
            (heur[0] - mc[0]).abs() > 1e-6,
            "heur must not degrade to mc's full-terminal-return skip"
        );
        assert!(
            (heur[0] - zero[0]).abs() > 1e-6,
            "heur must not degrade to zero's truncated-to-0.0 bootstrap"
        );
    }

    #[test]
    fn last_decision_of_game_is_pure_terminal_return_at_any_lambda() {
        // Only decision on record for player 1: no checkpoints ahead, so the
        // label must equal the plain (unbootstrapped) reward to final scores
        // regardless of lambda (remaining_weight stays 1.0, loop body never runs).
        let history = vec![step(1, 5, 1000, 800, Some(0.2))];
        let final_scores = finals(&[(1, 1300), (2, 900)]);
        let expected = reward::normalized_reward(1000, 800, 1300, 900).clamp(-1.0, 1.0);

        for lambda in [0.0, 0.5, 0.8, 0.95] {
            let out = td_lambda_labels(&history, &final_scores, lambda, reward::REL_W, None, MissingBootstrap::Zero);
            assert!(
                (out[0] - expected).abs() < 1e-6,
                "lambda={lambda}: got {}, expected {expected}",
                out[0]
            );
        }
    }

    #[test]
    fn lambda_zero_uses_only_the_first_checkpoint() {
        // Two future checkpoints for player 1 at turn 6 and turn 7. At
        // lambda=0 the label must depend ONLY on the turn-6 checkpoint —
        // this is exactly the original 1-step TD bootstrap, reproduced
        // bit-for-bit as the lambda=0 special case of the new formula.
        let history = vec![
            step(1, 5, 1000, 800, Some(0.2)),  // i
            step(1, 6, 1100, 800, Some(0.9)),  // checkpoint n=1 (this player's next turn)
            step(2, 6, 1000, 850, Some(-0.1)), // other player, ignored
            step(1, 7, 1400, 800, Some(-0.9)), // checkpoint n=2: a wildly different root_value
        ];
        let final_scores = finals(&[(1, 5000), (2, 800)]);

        let out = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W, None, MissingBootstrap::Zero);

        let r = reward::normalized_reward(1000, 800, 1100, 800);
        let expected = (r + reward::GAMMA_TURN.powi(1) * 0.9).clamp(-1.0, 1.0);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );

        // Sanity: changing turn 7's root_value must NOT move the lambda=0 label.
        let mut history2 = history.clone();
        history2[3].root_value = Some(12345.0);
        let out2 = td_lambda_labels(&history2, &final_scores, 0.0, reward::REL_W, None, MissingBootstrap::Zero);
        assert!((out2[0] - out[0]).abs() < 1e-6);
    }

    #[test]
    fn weights_blend_geometrically_and_sum_to_one() {
        // One checkpoint ahead + terminal. At lambda=0.5 the checkpoint gets
        // weight 0.5 and the terminal return gets the residual 0.5 — hand
        // computed, not just asserted-to-sum-to-1, so a weighting bug can't
        // hide behind a normalization step.
        let history = vec![
            step(1, 0, 100, 100, Some(0.4)),
            step(1, 1, 300, 100, Some(0.6)),
        ];
        let final_scores = finals(&[(1, 300), (2, 100)]);

        let out = td_lambda_labels(&history, &final_scores, 0.5, reward::REL_W, None, MissingBootstrap::Zero);

        let n1 = reward::normalized_reward(100, 100, 300, 100) + reward::GAMMA_TURN.powi(1) * 0.6;
        let terminal = reward::normalized_reward(100, 100, 300, 100);
        let expected = (0.5 * n1 + 0.5 * terminal).clamp(-1.0, 1.0);

        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );
    }

    #[test]
    fn missing_root_value_at_a_checkpoint_contributes_zero_bootstrap() {
        // Turn 6's only entry has no root value (forced/book/single-legal
        // move) — its n-step return must fall back to pure banked reward
        // (0.0 bootstrap), not skip the checkpoint entirely.
        let history = vec![
            step(1, 5, 1000, 800, Some(0.2)),
            step(1, 6, 1200, 800, None),
        ];
        let final_scores = finals(&[(1, 1200), (2, 800)]);

        let out = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W, None, MissingBootstrap::Zero);
        let expected = reward::normalized_reward(1000, 800, 1200, 800).clamp(-1.0, 1.0);
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );
    }

    #[test]
    fn label_rel_w_reprices_windows() {
        // I gain 100 while the opponent gains 400 over one window: an
        // abs-only weighting must label it positive, a rel-only one negative
        // — proves the flag actually reaches the window pricing.
        let history = vec![
            step(1, 5, 1000, 800, Some(0.0)),
            step(1, 6, 1100, 1200, Some(0.0)),
        ];
        let final_scores = finals(&[(1, 1100), (2, 1200)]);

        let abs_only = td_lambda_labels(&history, &final_scores, 0.0, 0.0, None, MissingBootstrap::Zero);
        let rel_only = td_lambda_labels(&history, &final_scores, 0.0, 1.0, None, MissingBootstrap::Zero);
        assert!(abs_only[0] > 0.0, "abs-only label should be positive, got {}", abs_only[0]);
        assert!(rel_only[0] < 0.0, "rel-only label should be negative, got {}", rel_only[0]);
    }

    #[test]
    fn wl_mode_last_decision_is_pure_z() {
        // No checkpoints ahead: the label must be exactly the ±1 outcome,
        // independent of lambda and of every score in the game.
        let history = vec![step(1, 5, 1000, 800, Some(0.2))];
        let final_scores = finals(&[(1, 1300), (2, 900)]);
        let z = finals(&[(1, 1), (2, -1)]);

        for lambda in [0.0, 0.5, 0.8, 0.95] {
            let out = td_lambda_labels(&history, &final_scores, lambda, reward::REL_W, Some(&z), MissingBootstrap::Zero);
            assert!(
                (out[0] - 1.0).abs() < 1e-6,
                "lambda={lambda}: got {}, expected 1.0",
                out[0]
            );
        }
    }

    #[test]
    fn wl_mode_blends_root_value_with_z_and_ignores_scores() {
        // One checkpoint ahead (V=0.6) + z=-1 tail: at lambda=0.5 the label
        // is 0.5·0.6 + 0.5·(−1) — the q-target blend, hand computed.
        let history = vec![
            step(1, 0, 100, 100, Some(0.4)),
            step(1, 1, 300, 100, Some(0.6)),
        ];
        let final_scores = finals(&[(1, 300), (2, 100)]);
        let z = finals(&[(1, -1), (2, 1)]);

        let out = td_lambda_labels(&history, &final_scores, 0.5, reward::REL_W, Some(&z), MissingBootstrap::Zero);
        let expected = 0.5f32 * 0.6 + 0.5 * -1.0;
        assert!(
            (out[0] - expected).abs() < 1e-6,
            "got {}, expected {expected}",
            out[0]
        );

        // Outcome space must be blind to score magnitudes entirely.
        let history2 = vec![
            step(1, 0, 5000, 1, Some(0.4)),
            step(1, 1, 9000, 1, Some(0.6)),
        ];
        let out2 = td_lambda_labels(&history2, &final_scores, 0.5, reward::REL_W, Some(&z), MissingBootstrap::Zero);
        assert!((out2[0] - out[0]).abs() < 1e-6);
    }

    #[test]
    fn wl_mode_lambda_zero_is_pure_undiscounted_first_root_value() {
        // lambda=0: first checkpoint takes weight 1, z weight 0 — and no
        // GAMMA_TURN discount may be applied (γ=1 in outcome space).
        let history = vec![
            step(1, 5, 1000, 800, Some(0.2)),
            step(1, 6, 1100, 800, Some(0.9)),
        ];
        let final_scores = finals(&[(1, 5000), (2, 800)]);
        let z = finals(&[(1, 1), (2, -1)]);

        let out = td_lambda_labels(&history, &final_scores, 0.0, reward::REL_W, Some(&z), MissingBootstrap::Zero);
        assert!(
            (out[0] - 0.9).abs() < 1e-6,
            "got {}, expected undiscounted 0.9",
            out[0]
        );
    }

    #[test]
    fn macro_ballot_dedups_per_turn_pov_and_retries_on_empty() {
        let goal = polyfish::ai::oracle_macro::MacroGoal::default();
        let ballot = Some((vec![goal], vec![1.0]));
        let mut last_key: Option<(i32, PlayerId)> = None;

        assert!(
            macro_ballot_for_history_step((5, 1), &mut last_key, ballot.clone()).is_some(),
            "first offer for a (turn,pov) must capture"
        );
        assert_eq!(last_key, Some((5, 1)));
        assert!(
            macro_ballot_for_history_step((5, 1), &mut last_key, ballot.clone()).is_none(),
            "same (turn,pov) must dedup"
        );
        assert!(
            macro_ballot_for_history_step((6, 1), &mut last_key, ballot.clone()).is_some(),
            "new turn must re-capture"
        );

        let mut last_key2: Option<(i32, PlayerId)> = None;
        let empty = Some((Vec::new(), Vec::new()));
        assert!(
            macro_ballot_for_history_step((7, 2), &mut last_key2, empty).is_none(),
            "an empty ballot must not be captured"
        );
        assert_eq!(last_key2, None, "empty ballot must not poison the dedup key");
        assert!(
            macro_ballot_for_history_step((7, 2), &mut last_key2, ballot).is_some(),
            "must retry on the same (turn,pov) after an empty offer"
        );
    }


    #[test]
    fn tech_multihot_uses_iter_position_not_discriminant() {
        let mk = |tech_type, discovered| TechnologyState {
            tech_type,
            discovered,
            discovered_turn: 0,
        };
        let techs = vec![
            mk(TechnologyType::Riding, true),
            mk(TechnologyType::ShockTactics, true),
            mk(TechnologyType::Rituals, true),
            mk(TechnologyType::Fishing, false),
        ];
        let v = tech_multihot(&techs);
        let n = TechnologyType::iter().count();
        assert_eq!(v.len(), n);
        assert_eq!(v.iter().filter(|&&x| x == 1.0).count(), 3);
        let rituals_pos = TechnologyType::iter()
            .position(|t| t == TechnologyType::Rituals)
            .unwrap();
        assert_eq!(v[rituals_pos], 1.0);
        // Discriminant-indexed encoding would need a slot at 121 >= n.
        assert!(TechnologyType::Rituals as usize >= n);
    }

    #[test]
    fn ownership_from_pov_maps_signs() {
        let owner = vec![0, 1, 2];
        assert_eq!(ownership_from_pov(&owner, 1), vec![0.0, 1.0, -1.0]);
        assert_eq!(ownership_from_pov(&owner, 2), vec![0.0, -1.0, 1.0]);
    }

    #[test]
    fn enemy_unit_grid_excludes_pov_invisible_and_bounds() {
        let unit = |owner: PlayerId, idx: i32, invisible: bool| {
            let mut u = UnitState {
                owner,
                coords: Coords::from_index(idx, 11),
                ..Default::default()
            };
            if invisible {
                u.effects.insert(UnitEffect::Invisible);
            }
            u
        };
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit(1, 5, false));
        let mut t2 = TribeState::default();
        t2.units.push(unit(2, 17, false));
        t2.units.push(unit(2, 30, true)); // invisible: excluded
        t2.units.push(unit(2, 500, false)); // out of range: excluded
        state.tribes.insert(1, t1);
        state.tribes.insert(2, t2);

        let g = enemy_unit_grid(&state, 1, 121);
        let set: Vec<usize> = g
            .iter()
            .enumerate()
            .filter(|&(_, &v)| v == 1.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(set, vec![17]);
    }

    fn sstep(player_id: PlayerId, turn: i32, my: i32, opp: i32) -> SptStep {
        SptStep {
            player_id,
            turn,
            my_spt: my,
            opp_spt: opp,
        }
    }

    #[test]
    fn spt_checkpoints_keep_first_decision_per_turn() {
        let steps = vec![sstep(1, 3, 5, 4), sstep(1, 3, 9, 9), sstep(1, 4, 6, 5)];
        let cp = spt_checkpoints_by_player(&steps);
        let c1 = &cp[&1];
        assert_eq!(c1.len(), 2);
        assert_eq!((c1[0].turn, c1[0].my_spt), (3, 5));
        assert_eq!((c1[1].turn, c1[1].my_spt), (4, 6));
    }

    #[test]
    fn spt_target_five_turn_lookup_and_final_fallback() {
        let steps = vec![
            sstep(1, 0, 2, 2),
            sstep(1, 3, 4, 3),
            sstep(1, 9, 8, 6),
            sstep(1, 12, 10, 9),
        ];
        let cp = spt_checkpoints_by_player(&steps);
        // T=3: first turn >= 8 is 9.
        assert_eq!(spt_target(cp.get(&1), 3, 99, 99), (8, 6));
        // T=4: exact boundary, turn 9 == 4+5.
        assert_eq!(spt_target(cp.get(&1), 4, 99, 99), (8, 6));
        // T=7: first turn >= 12 is 12 (present exactly).
        assert_eq!(spt_target(cp.get(&1), 7, 0, 0), (10, 9));
        // T=9: nothing at >= 14 -> final fallback.
        assert_eq!(spt_target(cp.get(&1), 9, 99, 98), (99, 98));
        // Unknown player -> final fallback.
        assert_eq!(spt_target(cp.get(&7), 0, 1, 2), (1, 2));
    }

    fn tstep(player_id: PlayerId, turn: i32, my: i32, opp: i32) -> TerritoryStep {
        TerritoryStep {
            player_id,
            turn,
            my_territory: my,
            opp_territory: opp,
        }
    }

    #[test]
    fn territory_checkpoints_keep_first_decision_per_turn() {
        let steps = vec![tstep(1, 3, 12, 8), tstep(1, 3, 20, 8), tstep(1, 4, 14, 9)];
        let cp = territory_checkpoints_by_player(&steps);
        let c1 = &cp[&1];
        assert_eq!(c1.len(), 2);
        assert_eq!((c1[0].turn, c1[0].my_territory), (3, 12));
        assert_eq!((c1[1].turn, c1[1].my_territory), (4, 14));
    }

    #[test]
    fn territory_target_five_turn_lookup_and_final_fallback() {
        let steps = vec![
            tstep(1, 0, 5, 5),
            tstep(1, 3, 10, 6),
            tstep(1, 9, 22, 12),
            tstep(1, 12, 25, 15),
        ];
        let cp = territory_checkpoints_by_player(&steps);
        // T=3: first turn >= 8 is 9.
        assert_eq!(territory_target(cp.get(&1), 3, 999, 999), (22, 12));
        // T=4: exact boundary, turn 9 == 4+5.
        assert_eq!(territory_target(cp.get(&1), 4, 999, 999), (22, 12));
        // T=7: first turn >= 12 is 12 (present exactly).
        assert_eq!(territory_target(cp.get(&1), 7, 0, 0), (25, 15));
        // T=9: nothing at >= 14 -> final fallback.
        assert_eq!(territory_target(cp.get(&1), 9, 99, 98), (99, 98));
        // Unknown player -> final fallback.
        assert_eq!(territory_target(cp.get(&7), 0, 1, 2), (1, 2));
    }

    #[test]
    fn territory_target_reached_then_lost_still_counts_the_earlier_high() {
        // EXP_ELO_120: momentum, not possession -- a city taken then
        // recaptured must not silently erase the earlier tile count from
        // the horizon lookup. Territory drops back down at turn 10 (city
        // lost) but the checkpoint AT the +5 horizon (turn 8) still shows
        // the peak, because it's a snapshot at that turn, not a max-so-far.
        let steps = vec![
            tstep(1, 0, 8, 8),
            tstep(1, 5, 8, 8),
            tstep(1, 8, 18, 8),  // captured a city, territory jumps
            tstep(1, 10, 9, 8),  // city recaptured by opponent, drops back
        ];
        let cp = territory_checkpoints_by_player(&steps);
        // T=3: first turn >= 8 is turn 8 itself -- the peak, not the later drop.
        assert_eq!(territory_target(cp.get(&1), 3, 0, 0), (18, 8));
        // T=5: horizon is turn 10, which already reflects the loss.
        assert_eq!(territory_target(cp.get(&1), 5, 0, 0), (9, 8));
    }

    #[test]
    fn territory_target_h1_one_turn_lookup_and_final_fallback() {
        // Phase-2 spike (EXP_ELO_120): same checkpoints as territory_target,
        // a +1 window instead of +5.
        let steps = vec![
            tstep(1, 0, 5, 5),
            tstep(1, 3, 10, 6),
            tstep(1, 9, 22, 12),
            tstep(1, 12, 25, 15),
        ];
        let cp = territory_checkpoints_by_player(&steps);
        // T=0: first turn >= 1 is 3.
        assert_eq!(territory_target_h1(cp.get(&1), 0, 999, 999), (10, 6));
        // T=3: exact boundary, turn 4 has no checkpoint -> first turn >= 4 is 9.
        assert_eq!(territory_target_h1(cp.get(&1), 3, 999, 999), (22, 12));
        // T=8: first turn >= 9 is present exactly.
        assert_eq!(territory_target_h1(cp.get(&1), 8, 0, 0), (22, 12));
        // T=12: nothing at >= 13 -> final fallback.
        assert_eq!(territory_target_h1(cp.get(&1), 12, 99, 98), (99, 98));
    }

    fn astep(player_id: PlayerId, turn: i32, my: f32, opp: f32) -> ArmyStep {
        ArmyStep { player_id, turn, my_army: my, opp_army: opp }
    }

    #[test]
    fn army_target_five_turn_lookup_and_final_fallback() {
        let steps = vec![
            astep(1, 0, 0.1, 0.1),
            astep(1, 3, 0.3, 0.2),
            astep(1, 9, 0.6, 0.4),
            astep(1, 12, 0.8, 0.5),
        ];
        let cp = army_checkpoints_by_player(&steps);
        assert_eq!(army_target(cp.get(&1), 3, 9.0, 9.0), (0.6, 0.4));
        assert_eq!(army_target(cp.get(&1), 4, 9.0, 9.0), (0.6, 0.4));
        assert_eq!(army_target(cp.get(&1), 7, 0.0, 0.0), (0.8, 0.5));
        assert_eq!(army_target(cp.get(&1), 9, 0.99, 0.98), (0.99, 0.98));
        assert_eq!(army_target(cp.get(&7), 0, 0.1, 0.2), (0.1, 0.2));
    }

    #[test]
    fn siege_pressure_target_windowed_max() {
        // Opponent (id 2) besieged at turns 3 and 9; POV (id 1) besieged at
        // turn 4 -- must not leak into the opponent-only window.
        let events = vec![(3, 2), (4, 1), (9, 2)];
        // T=0: window is (0,5], catches turn 3.
        assert_eq!(siege_pressure_target(&events, 0, 2), 1.0);
        // T=3: window is (3,8], turn 3 itself is excluded (open interval).
        assert_eq!(siege_pressure_target(&events, 3, 2), 0.0);
        // T=4: window is (4,9], catches turn 9 exactly at the boundary.
        assert_eq!(siege_pressure_target(&events, 4, 2), 1.0);
        // T=10: nothing left in (10,15].
        assert_eq!(siege_pressure_target(&events, 10, 2), 0.0);
        // Querying the POV's own besiegement (opp_id=1) must not match the
        // opponent's events.
        assert_eq!(siege_pressure_target(&events, 0, 1), 1.0);
        assert_eq!(siege_pressure_target(&events, 4, 1), 0.0);
    }
