//! Split to a separate file (Aug 2026) so the production module
//! this backs stays well under 1000 lines despite thorough coverage.

use super::*;
use super::super::test_support::*;
use crate::coords::Coords;
use crate::settings::units::get_unit_setting;
use crate::states::{TileState, TribeState, UnitState};
use crate::types::UnitType;

    /// A Smithery/Forge-shaped lane: `cost = tech_cost + structure_cost`.
    fn build_test_lane(cost: i32, tech_cost: i32, structure_cost: i32) -> crate::ai::oracle_macro::SaveLane {
        crate::ai::oracle_macro::SaveLane {
            cost,
            tech_cost,
            structure_cost,
            structure_unit_cost: structure_cost,
            tech: crate::types::TechnologyType::Smithery,
            structure: crate::types::StructureType::Forge,
        }
    }
    /// Full 11×11 field board, both players' explorers everywhere, P1 city
    /// at `city_idx` — reach checks need real tiles to path over.
    fn defense_board(city_idx: i32) -> GameState {
        let mut state = GameState::default();
        state.settings.size = 11;
        for i in 0..121 {
            let mut tile = TileState::default();
            tile.terrain_type = crate::types::TerrainType::Field;
            tile.explorers.insert(1);
            tile.explorers.insert(2);
            state.tiles.insert(i, tile);
        }
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState {
            owner: 1,
            idx: city_idx,
            ..Default::default()
        });
        state.tribes.insert(1, t1);
        state.tribes.insert(2, TribeState::default());
        state
    }
    fn combat_unit(idx: i32, unit_type: UnitType, owner: i32) -> UnitState {
        let mut u = unit_at(idx, unit_type);
        u.owner = owner;
        u.health = crate::functions::get_unit_max_health(&u);
        u
    }
    #[test]
    fn goal_potential_prices_each_stance_and_expand_progress() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        add_visible_village(&mut state, 0);
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(2, UnitType::Warrior)); // 2 tiles from village 0
        state.tribes.insert(1, t1);

        // ARM pays the army's star cost (+ the lighthouse term: the explored
        // village tile 0 is a map corner).
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        let corner = SHAPE_GOAL_LIGHTHOUSE;
        assert!(
            (goal_potential(&state, 1, &arm, None) - SHAPE_GOAL_ARM_PER_COST * cost - corner)
                .abs()
                < 1e-4
        );

        // GROW pays SPT plus the scout term (no EXPAND target known, <3
        // cities, one explored tile in this state) plus the v6 body term
        // (1 unit within the cities+1 cap, map unexplored).
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let spt = crate::functions::get_tribe_spt(&state, state.tribes.get(&1).unwrap()) as f32;
        let expected = SHAPE_GOAL_SPT * spt + SHAPE_GOAL_SCOUT + corner + SHAPE_GOAL_BODY;
        assert!((goal_potential(&state, 1, &grow, None) - expected).abs() < 1e-4);

        // EXPAND order: a one-tile close banks one step of the gradient.
        let ex = |orders| MacroGoal { orders, stance: Stance::Arm, save_target: None };
        let base = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        state.tribes.get_mut(&1).unwrap().units[0] = unit_at(1, UnitType::Warrior);
        let closer = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        assert!((closer - base - SHAPE_GOAL_EXPAND_PER_TILE).abs() < 1e-3);

        // Achieved target holds cap + completion bonus (no cliff on capture);
        // enemy-owned pays 0. Capture makes the tile an owned CITY.
        state.tiles.get_mut(&0).unwrap().owner = 1;
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            ..Default::default()
        });
        let achieved = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        let arm_only = goal_potential(&state, 1, &arm, None);
        let done = SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP as f32 + SHAPE_GOAL_EXPAND_DONE);
        assert!((achieved - arm_only - done).abs() < 1e-3);
        assert!(achieved >= closer);
        state.tiles.get_mut(&0).unwrap().owner = 2;
        let lost = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        // v6: an enemy-taken village pays the retake-weighted approach
        // (unit at 1 is one tile out) instead of dropping to zero.
        let retake = SHAPE_GOAL_RETAKE_W * SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP - 1) as f32;
        assert!((lost - arm_only - retake).abs() < 1e-3);
    }
    #[test]
    fn scout_term_pays_full_then_half_with_target_until_third_city() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };

        // Each newly explored tile banks SHAPE_GOAL_SCOUT.
        let base = goal_potential(&state, 1, &grow, None);
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(50, tile);
        let one = goal_potential(&state, 1, &grow, None);
        assert!((one - base - SHAPE_GOAL_SCOUT).abs() < 1e-4);

        // A known EXPAND target halves the scout term (v4 — info retains
        // value alongside the approach gradient; unit at 60 is cheb 1 from 50).
        let with_target = MacroGoal {
            orders: vec![(OrderKind::Expand, 50)],
            stance: Stance::Grow,
            save_target: None,
        };
        let anchored = goal_potential(&state, 1, &with_target, None);
        let spt0 = crate::functions::get_tribe_spt(&state, state.tribes.get(&1).unwrap()) as f32;
        let approach = SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP - 1) as f32;
        let half_scout = SHAPE_GOAL_SCOUT * 0.5;
        // + v6 body term: 1 unit, 0 cities → cap 1.
        assert!(
            (anchored - SHAPE_GOAL_SPT * spt0 - approach - half_scout - SHAPE_GOAL_BODY).abs()
                < 1e-3
        );
        // ARM never scouts; neither does a 3-city tribe.
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
        let arm_phi = goal_potential(&state, 1, &arm, None);
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        assert!((arm_phi - SHAPE_GOAL_ARM_PER_COST * cost).abs() < 1e-4);
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        let done = goal_potential(&state, 1, &grow, None);
        let spt = crate::functions::get_tribe_spt(&state, state.tribes.get(&1).unwrap()) as f32;
        // Scout retires at 3 cities; the body term still pays its 1 unit
        // while the map stays unexplored.
        assert!((done - SHAPE_GOAL_SPT * spt - SHAPE_GOAL_BODY).abs() < 1e-4);
    }
    #[test]
    fn goal_aux_pays_tech_fit_and_riders() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        use crate::states::TechnologyState;
        use crate::types::TechnologyType;
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Rider));
        state.tribes.insert(1, t1);
        let goal = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let aux = GoalAux {
            recommended_techs: vec![TechnologyType::Mining],
            rider_push: true,
            ..Default::default()
        };
        let base = goal_potential(&state, 1, &goal, None);
        // Rider push pays per living Rider; the unowned recommendation pays 0.
        let with_aux = goal_potential(&state, 1, &goal, Some(&aux));
        assert!((with_aux - base - SHAPE_GOAL_RIDER).abs() < 1e-3);
        // Owning the recommended tech banks the fit bonus.
        state.tribes.get_mut(&1).unwrap().tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Mining,
            discovered: true,
            discovered_turn: 0,
        });
        let owned = goal_potential(&state, 1, &goal, Some(&aux));
        assert!((owned - with_aux - SHAPE_GOAL_TECH_FIT).abs() < 1e-3);
    }
    /// v9 (EXP_ELO_029): ARM holds 85% of plies after turn 10, so an ARM
    /// potential blind to income and city level left the whole mid-game
    /// without an economy gradient. A giant is bought with population.
    #[test]
    fn arm_pays_for_income_and_level_progress() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState {
            idx: 60,
            owner: 1,
            level: 1,
            ..Default::default()
        });
        state.tribes.insert(1, t1);
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };

        // Income is visible under ARM: raising city production raises Phi by
        // exactly the new per-SPT rate.
        let spt = |s: &GameState| {
            crate::functions::get_tribe_spt(s, s.tribes.get(&1).unwrap()) as f32
        };
        let before = goal_potential(&state, 1, &arm, None);
        let spt_before = spt(&state);
        state.tribes.get_mut(&1).unwrap().cities[0].production += 3;
        let richer = goal_potential(&state, 1, &arm, None);
        let spt_gain = spt(&state) - spt_before;
        assert!(spt_gain > 0.0, "test setup did not move SPT");
        assert!(
            (richer - before - SHAPE_GOAL_ARM_SPT * spt_gain).abs() < 1e-3,
            "ARM ignored income: {before} -> {richer} for +{spt_gain} SPT"
        );

        // ...and so is progress toward the next city level. The city must be
        // COMPLETABLE for the term to pay (progress alone clears level+1 here;
        // a bare CityState has no territory routes to harvest).
        state.tribes.get_mut(&1).unwrap().cities[0].progress = 2;
        let progressing = goal_potential(&state, 1, &arm, None);
        assert!(
            (progressing - richer - SHAPE_GOAL_COMPLETION * (2.0 / 2.0)).abs() < 1e-3,
            "ARM ignored level progress: {richer} -> {progressing}"
        );

        // Army still dominates: one warrior outweighs a point of SPT.
        assert!(SHAPE_GOAL_ARM_PER_COST * 2.0 > SHAPE_GOAL_ARM_SPT * 1.0);
    }
    #[test]
    fn explorer_reward_pays_by_hidden_fraction() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::types::CityRewardType;
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 24, ..Default::default() });
        // v8: the first-city discount applies at 1 city, so a second city keeps
        // this test on the full-rate branch it was written to measure.
        t1.cities.push(crate::states::CityState { idx: 108, ..Default::default() });
        state.tribes.insert(1, t1);
        // Unlock stance isolates the explorer term (no SPT/scout/ARM terms).
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let before = goal_potential(&state, 1, &goal, None);
        state.tribes.get_mut(&1).unwrap().cities[0].rewards.push(CityRewardType::Explorer);
        // Fully hidden map, city 24 within EXPLORER_WALK_RANGE of corner 0:
        // full bonus + the lighthouse-chance lift.
        let dark = goal_potential(&state, 1, &goal, None);
        assert!(
            (dark - before - SHAPE_GOAL_EXPLORER - SHAPE_GOAL_EXPLORER_LIGHTHOUSE).abs() < 1e-3
        );
        // A center city reaches all four dark corners (cheb 5) but the lift
        // caps at two — "one, sometimes two lighthouses per explorer".
        let mut mid = GameState::default();
        let mut t2 = TribeState::default();
        let mut c = crate::states::CityState { idx: 60, ..Default::default() };
        c.rewards.push(CityRewardType::Explorer);
        t2.cities.push(c);
        t2.cities.push(crate::states::CityState { idx: 12, ..Default::default() });
        mid.tribes.insert(1, t2);
        let mid_phi = goal_potential(&mid, 1, &goal, None);
        let capped = SHAPE_GOAL_EXPLORER + 2.0 * SHAPE_GOAL_EXPLORER_LIGHTHOUSE;
        assert!((mid_phi - capped).abs() < 1e-3);
        // Fully revealed map: the bonus decays to ~0 (corners add lighthouse).
        for idx in 0..121 {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.explorers.insert(1);
        }
        let lit = goal_potential(&state, 1, &goal, None);
        assert!((lit - before - 4.0 * SHAPE_GOAL_LIGHTHOUSE).abs() < 1e-3);
    }
    #[test]
    fn goal_potential_pays_archetype_preferred_units() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Archer));
        t1.units.push(unit_at(61, UnitType::Warrior));
        state.tribes.insert(1, t1);
        // Unlock stance zeroes the stance term; only the unit bonus differs.
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let aux = GoalAux {
            preferred_units: vec![UnitType::Archer],
            ..Default::default()
        };
        let base = goal_potential(&state, 1, &goal, None);
        let with = goal_potential(&state, 1, &goal, Some(&aux));
        // Cost-scaled (v6): Archer costs 3 → 99, within 1% of the old flat 100.
        let archer_cost = get_unit_setting(UnitType::Archer).cost as f32;
        assert!((with - base - SHAPE_GOAL_ARCHETYPE_PER_COST * archer_cost).abs() < 1e-3);
    }
    #[test]
    fn archetype_per_cost_prices_knight_above_defender() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Knight));
        state.tribes.insert(1, t1);
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let aux = GoalAux {
            preferred_units: vec![UnitType::Knight, UnitType::Defender],
            ..Default::default()
        };
        let knight = goal_potential(&state, 1, &goal, Some(&aux));
        state.tribes.get_mut(&1).unwrap().units[0] = unit_at(60, UnitType::Defender);
        let defender = goal_potential(&state, 1, &goal, Some(&aux));
        let k_cost = get_unit_setting(UnitType::Knight).cost as f32;
        let d_cost = get_unit_setting(UnitType::Defender).cost as f32;
        assert!((knight - SHAPE_GOAL_ARCHETYPE_PER_COST * k_cost).abs() < 1e-3);
        assert!((defender - SHAPE_GOAL_ARCHETYPE_PER_COST * d_cost).abs() < 1e-3);
        assert!(knight > defender, "a knight must out-price a defender head-for-head");
    }
    #[test]
    fn yield_structures_pay_per_partner_beyond_first() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::coords::Coords;
        use crate::states::StructureState;
        use crate::types::StructureType;
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 60, owner: 1, ..Default::default() });
        state.tribes.insert(1, t1);
        let rule = Coords { x: 5, y: 5, idx: 60 };
        for idx in [59, 70] {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.ruling_city_coords = Some(rule.clone());
        }
        // Partner tiles must be FRIENDLY territory to count (real-game rule).
        for idx in [58, 48, 69, 71] {
            state.tiles.entry(idx).or_insert_with(TileState::default).owner = 1;
        }
        let farm = |st: &mut GameState, idx: i32| {
            st.structures.insert(idx, Some(StructureState {
                structure_type: StructureType::Farm,
                ..Default::default()
            }));
        };
        // Unlock stance isolates the term.
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        state.structures.insert(59, Some(StructureState {
            structure_type: StructureType::Windmill,
            ..Default::default()
        }));
        farm(&mut state, 58);
        // One partner: the windmill pays for itself, no bonus.
        let one = goal_potential(&state, 1, &goal, None);
        assert!(one.abs() < 1e-4);
        // Second adjacent farm: +YIELD_ADJ × reward_pop(1) × 1.
        farm(&mut state, 48);
        let two = goal_potential(&state, 1, &goal, None);
        assert!((two - one - SHAPE_GOAL_YIELD_ADJ).abs() < 1e-4);
        // Forge scales by its reward_pop (2 per extra mine).
        state.structures.insert(70, Some(StructureState {
            structure_type: StructureType::Forge,
            ..Default::default()
        }));
        let mine = |st: &mut GameState, idx: i32| {
            st.structures.insert(idx, Some(StructureState {
                structure_type: StructureType::Mine,
                ..Default::default()
            }));
        };
        mine(&mut state, 69);
        mine(&mut state, 71);
        let with_forge = goal_potential(&state, 1, &goal, None);
        assert!((with_forge - two - 2.0 * SHAPE_GOAL_YIELD_ADJ).abs() < 1e-4);
        // Enemy-ruled structures pay nothing.
        state.tribes.get_mut(&1).unwrap().cities[0].owner = 2;
        let enemy = goal_potential(&state, 1, &goal, None);
        assert!(enemy.abs() < 1e-4);
    }
    /// v10 invariant: buying the lane tech must never LOWER Φ. The old ramp
    /// read the star balance alone, so the purchase the plan existed for cost
    /// it the price of the tech.
    #[test]
    fn buying_the_lane_tech_does_not_lower_the_savings_ramp() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.id = 1;
        t1.stars = 16;
        state.tribes.insert(1, t1);
        let goal = MacroGoal {
            orders: vec![],
            stance: Stance::Save,
            save_target: Some(build_test_lane(21, 16, 5)),
        };
        let before = goal_potential(&state, 1, &goal, None);

        // Spend 16 on the tech: stars 16 -> 0, but the tech is now owned.
        let t = state.tribes.get_mut(&1).unwrap();
        t.stars = 0;
        t.tech_vanilla.push(crate::states::TechnologyState {
            tech_type: crate::types::TechnologyType::Smithery,
            discovered: true,
            discovered_turn: 1,
        });
        let after = goal_potential(&state, 1, &goal, None);
        assert!(
            after >= before - 1e-3,
            "buying the lane tech dropped Phi: {before} -> {after}"
        );
    }
    /// v10: the level-5 Park-vs-Giant pick, priced end to end (raw score AND
    /// Phi) under every stance. These ratios are coupled to `SHAPE_GOAL_SPT`
    /// and `SHAPE_GOAL_ARM_PER_COST`; if someone retunes those, this is the
    /// test that catches the pick silently flipping back to Park.
    #[test]
    fn level_five_reward_pick_favours_the_giant_except_on_a_nearly_done_plan() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::types::UnitType;

        // Score side, from functions.rs: Park is +250, a Giant is 5 x cost 10.
        const PARK_SCORE: f32 = 250.0;
        const GIANT_SCORE: f32 = 5.0 * 10.0;

        let mk = |stance, save: Option<crate::ai::oracle_macro::SaveLane>| MacroGoal {
            orders: vec![],
            stance,
            save_target: save,
        };
        let base_state = || {
            let mut s = GameState::default();
            s.settings.size = 11;
            let mut t = TribeState::default();
            t.id = 1;
            t.stars = 0;
            s.tribes.insert(1, t);
            s
        };
        // Park: +1 production on the city that just levelled.
        let with_park = || {
            let mut s = base_state();
            s.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
                idx: 60,
                level: 5,
                production: 1,
                ..Default::default()
            });
            s
        };
        let with_giant = || {
            let mut s = base_state();
            s.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Giant));
            s
        };
        let value = |s: &GameState, g: &MacroGoal, score: f32| {
            score + goal_potential(s, 1, g, None)
        };

        for (label, stance, save, giant_should_win) in [
            ("ARM", Stance::Arm, None, true),
            ("GROW", Stance::Grow, None, true),
            (
                "SAVE, plan barely started",
                Stance::Save,
                Some(build_test_lane(20, 16, 4)),
                true,
            ),
        ] {
            let g = mk(stance, save);
            let park = value(&with_park(), &g, PARK_SCORE);
            let giant = value(&with_giant(), &g, GIANT_SCORE);
            assert_eq!(
                giant > park,
                giant_should_win,
                "{label}: park {park}, giant {giant}"
            );
        }

        // A SAVE plan at full progress SHOULD prefer the Park: +1 production
        // finishes the batch, and that is worth more than the unit right now.
        let lane = build_test_lane(20, 16, 4);
        let g = mk(Stance::Save, Some(lane.clone()));
        let mut park_full = with_park();
        park_full.tribes.get_mut(&1).unwrap().stars = lane.cost;
        let mut giant_full = with_giant();
        giant_full.tribes.get_mut(&1).unwrap().stars = lane.cost;
        let park = value(&park_full, &g, PARK_SCORE);
        let giant = value(&giant_full, &g, GIANT_SCORE);
        assert!(
            park > giant,
            "a nearly-complete SAVE plan should take the Park: park {park}, giant {giant}"
        );
    }
    /// v7: holding stars must climb a gradient. Before this, banked stars
    /// appeared nowhere in Phi, so spending them on anything scored strictly
    /// beat holding and the measured policy was hand-to-mouth.
    #[test]
    fn savings_ramp_pays_for_banked_stars_and_keeps_the_economy_potential() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.id = 1;
        t1.stars = 0;
        state.tribes.insert(1, t1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let saving = MacroGoal {
            orders: vec![],
            stance: Stance::Save,
            save_target: Some(build_test_lane(20, 16, 4)),
        };

        // Empty bank: SAVE must equal GROW — the stance itself costs nothing.
        let base = goal_potential(&state, 1, &grow, None);
        assert!((goal_potential(&state, 1, &saving, None) - base).abs() < 1e-3);

        // Half banked pays half the ramp; full banked pays all of it.
        state.tribes.get_mut(&1).unwrap().stars = 10;
        let half = goal_potential(&state, 1, &saving, None);
        assert!((half - base - SHAPE_GOAL_SAVE / 2.0).abs() < 1e-3);
        state.tribes.get_mut(&1).unwrap().stars = 20;
        let full = goal_potential(&state, 1, &saving, None);
        assert!((full - base - SHAPE_GOAL_SAVE).abs() < 1e-3);

        // Overshooting the target pays no more — the ramp is not a hoard bonus.
        state.tribes.get_mut(&1).unwrap().stars = 60;
        assert!((goal_potential(&state, 1, &saving, None) - full).abs() < 1e-3);

        // Under GROW the same bank is worth nothing: the ramp is stance-gated.
        assert!((goal_potential(&state, 1, &grow, None) - base).abs() < 1e-3);

        // A full bank must never outweigh spending it — otherwise the agent
        // banks forever rather than buying the batch it saved for.
        assert!(SHAPE_GOAL_SAVE < SHAPE_GOAL_SPT);
    }
    #[test]
    fn contested_target_pays_one_extra_converger() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        add_visible_village(&mut state, 5);
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(3, UnitType::Warrior)); // assigned, d=2
        t1.units.push(unit_at(8, UnitType::Warrior)); // second, d=3
        state.tribes.insert(1, t1);
        let ex = MacroGoal {
            orders: vec![(OrderKind::Expand, 5)],
            stance: Stance::Arm,
            save_target: None,
        };
        let uncontested = goal_potential(&state, 1, &ex, None);

        // Enemy squatter on the village: the second unit's gradient pays at
        // half weight on top.
        let mut t2 = TribeState::default();
        t2.id = 2;
        t2.units.push(unit_at(5, UnitType::Warrior));
        state.tribes.insert(2, t2);
        let contested = goal_potential(&state, 1, &ex, None);
        let second = SHAPE_GOAL_CONTEST_SECOND
            * SHAPE_GOAL_EXPAND_PER_TILE
            * (SHAPE_PROX_CAP - 3) as f32;
        assert!(
            (contested - uncontested - second).abs() < 1e-3,
            "contested village must pay the second unit ({contested} vs {uncontested})"
        );
    }
    #[test]
    fn grow_body_term_pays_to_cap_only() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };

        // 0 cities → cap 1: the second unit adds nothing.
        let one = goal_potential(&state, 1, &grow, None);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(61, UnitType::Warrior));
        let two = goal_potential(&state, 1, &grow, None);
        assert!((two - one).abs() < 1e-4, "beyond-cap unit must not pay");

        // A city raises the cap to 2: the second unit now pays.
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            owner: 1,
            ..Default::default()
        });
        let capped_two = goal_potential(&state, 1, &grow, None);
        let one_city_one_unit = {
            let mut s2 = state.clone();
            s2.tribes.get_mut(&1).unwrap().units.pop();
            goal_potential(&s2, 1, &grow, None)
        };
        assert!(
            (capped_two - one_city_one_unit - SHAPE_GOAL_BODY).abs() < 1e-3,
            "unit within raised cap must pay"
        );
    }
    #[test]
    fn market_pays_star_adjacency_beyond_first_partner() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::coords::Coords;
        use crate::states::StructureState;
        use crate::types::StructureType;
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 60, owner: 1, ..Default::default() });
        state.tribes.insert(1, t1);
        let rule = Coords { x: 5, y: 5, idx: 60 };
        state.tiles.entry(59).or_insert_with(TileState::default).ruling_city_coords =
            Some(rule);
        for idx in [58, 48, 59] {
            state.tiles.entry(idx).or_insert_with(TileState::default).owner = 1;
        }
        let put = |st: &mut GameState, idx: i32, s: StructureType| {
            st.structures.insert(idx, Some(StructureState {
                structure_type: s,
                ..Default::default()
            }));
        };
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        put(&mut state, 59, StructureType::Market);
        put(&mut state, 58, StructureType::Windmill);
        // One hub partner: no bonus (the market pays for itself).
        let one = goal_potential(&state, 1, &goal, None);
        assert!(one.abs() < 1e-4);
        // Second hub: +YIELD_ADJ_STARS × reward_stars(1) × 1.
        put(&mut state, 48, StructureType::Sawmill);
        let two = goal_potential(&state, 1, &goal, None);
        assert!((two - one - SHAPE_GOAL_YIELD_ADJ_STARS).abs() < 1e-4);
    }
    #[test]
    fn standing_forest_in_territory_holds_option_value() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::types::TerrainType;
        let mut state = GameState::default();
        state.settings.size = 11;
        state.tribes.insert(1, TribeState::default());
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let tile = state.tiles.entry(60).or_insert_with(TileState::default);
        tile.owner = 1;
        tile.terrain_type = TerrainType::Forest;
        let with = goal_potential(&state, 1, &goal, None);
        assert!((with - SHAPE_GOAL_FOREST_STANDING).abs() < 1e-4);
        // Cleared (Field) or enemy-owned forest pays nothing.
        state.tiles.get_mut(&60).unwrap().terrain_type = TerrainType::Field;
        assert!(goal_potential(&state, 1, &goal, None).abs() < 1e-4);
        state.tiles.get_mut(&60).unwrap().terrain_type = TerrainType::Forest;
        state.tiles.get_mut(&60).unwrap().owner = 2;
        assert!(goal_potential(&state, 1, &goal, None).abs() < 1e-4);
    }
    /// Miniature of the fixture-seed t3 walk-off: a load-bearing garrison
    /// must out-price stepping off, and cover must out-price leaving the
    /// leash entirely.
    #[test]
    fn defend_order_prices_hold_cover_and_leash() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = defense_board(60);
        state.tribes.get_mut(&1).unwrap().units.push(combat_unit(60, UnitType::Rider, 1));
        state.tribes.get_mut(&2).unwrap().units.push(combat_unit(59, UnitType::Swordsman, 2));
        let goal = MacroGoal {
            orders: vec![(OrderKind::Defend, 60)],
            stance: Stance::Arm,
            save_target: None,
        };
        let phi_hold = goal_potential(&state, 1, &goal, None);
        // Step off to an adjacent tile: still full cover, hold term lost.
        state.tribes.get_mut(&1).unwrap().units[0].coords = Coords::from_index(48, 11);
        let phi_adjacent = goal_potential(&state, 1, &goal, None);
        // March to the far corner: out of the leash, only the recall
        // gradient pays.
        state.tribes.get_mut(&1).unwrap().units[0].coords = Coords::from_index(0, 11);
        let phi_far = goal_potential(&state, 1, &goal, None);
        assert!(
            phi_hold > phi_adjacent && phi_adjacent > phi_far,
            "leash ordering violated: hold {phi_hold} adjacent {phi_adjacent} far {phi_far}"
        );
        assert!((phi_hold - phi_adjacent - SHAPE_GOAL_DEFEND_HOLD).abs() < 1e-3);
    }
    /// The prep mechanism in miniature: a NEW unit that lands inside the
    /// coverage ring is worth more Φ than the same unit far away — this is
    /// what makes the executor's per-ply Δφ pay a defensive train/road/tech
    /// chain step by step (outcome pricing, no discrete planner).
    #[test]
    fn new_unit_inside_the_ring_outprices_the_same_unit_far_away() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = defense_board(60);
        state.tribes.get_mut(&1).unwrap().units.push(combat_unit(60, UnitType::Rider, 1));
        state.tribes.get_mut(&2).unwrap().units.push(combat_unit(59, UnitType::Swordsman, 2));
        let goal = MacroGoal {
            orders: vec![(OrderKind::Defend, 60)],
            stance: Stance::Arm,
            save_target: None,
        };
        // Same purchase, two landing spots: covering (cheb 2, rider reach)
        // vs remote corner. Army/stance terms cancel; coverage does not.
        let mut near = state.clone();
        near.tribes.get_mut(&1).unwrap().units.push(combat_unit(38, UnitType::Rider, 1));
        let mut far = state.clone();
        far.tribes.get_mut(&1).unwrap().units.push(combat_unit(10, UnitType::Rider, 1));
        let phi_near = goal_potential(&near, 1, &goal, None);
        let phi_far = goal_potential(&far, 1, &goal, None);
        assert!(
            phi_near > phi_far + SHAPE_GOAL_DEFEND_COVER * 0.5,
            "coverage must separate the landing spots: near {phi_near} far {phi_far}"
        );
    }
    /// EXP_ELO_042: a unit standing on an enemy city keeps its siege-hold
    /// pay by state-fact — no Attack order involved — and stepping off
    /// costs exactly the latch.
    #[test]
    fn siege_hold_outprices_stepping_off() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = defense_board(29);
        state.tribes.get_mut(&2).unwrap().cities.push(crate::states::CityState {
            owner: 2,
            idx: 79,
            ..Default::default()
        });
        state.tribes.get_mut(&1).unwrap().units.push(combat_unit(79, UnitType::Rider, 1));
        let goal = MacroGoal {
            orders: vec![],
            stance: Stance::Grow,
            save_target: None,
        };
        let phi_on = goal_potential(&state, 1, &goal, None);
        state.tribes.get_mut(&1).unwrap().units[0].coords = Coords::from_index(80, 11);
        let phi_off = goal_potential(&state, 1, &goal, None);
        assert!(
            (phi_on - phi_off - SHAPE_GOAL_ATTACK_PRESS * SHAPE_GOAL_SIEGE_HOLD_MULT).abs() < 1e-3,
            "latch delta wrong: on {phi_on} off {phi_off}"
        );
    }
    /// EXP_ELO_042: the shortfall recall gradient never conscripts an
    /// attack-committed unit — the same unit at the same distance from the
    /// threatened city pays recall when free and nothing when committed.
    #[test]
    fn recall_skips_attack_committed_units() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = defense_board(29);
        state.tribes.get_mut(&1).unwrap().units.push(combat_unit(29, UnitType::Rider, 1));
        state.tribes.get_mut(&2).unwrap().units.push(combat_unit(28, UnitType::Swordsman, 2));
        let goal = MacroGoal {
            orders: vec![(OrderKind::Defend, 29), (OrderKind::Attack, 79)],
            stance: Stance::Arm,
            save_target: None,
        };
        // Helper warrior, cheb 5 from B both times; H-side (35: cheb H=4,
        // committed) vs far side (87: cheb H=8, free). Neither position
        // covers B or presses H, so the recall term is the only difference.
        let mut committed = state.clone();
        committed.tribes.get_mut(&1).unwrap().units.push(combat_unit(35, UnitType::Warrior, 1));
        let mut free = state.clone();
        free.tribes.get_mut(&1).unwrap().units.push(combat_unit(87, UnitType::Warrior, 1));
        let phi_committed = goal_potential(&committed, 1, &goal, None);
        let phi_free = goal_potential(&free, 1, &goal, None);
        let recall = SHAPE_GOAL_DEFEND_COVER * 1.0 * 0.5
            * ((SHAPE_PROX_CAP - 5).max(0) as f32 / SHAPE_PROX_CAP as f32);
        assert!(
            (phi_free - phi_committed - recall).abs() < 1e-3,
            "recall exemption delta wrong: free {phi_free} committed {phi_committed} expected {recall}"
        );
    }
    /// CONTRACT CHANGED (EXP_ELO_050, was `no_defend_order_means_no_defense
    /// _pricing`): risk to a city is priced whether or not a Defend order
    /// names it. The 040 rule — enemy presence must not leak into Φ without
    /// an order — is exactly what lost the capital in the seed-1786807403
    /// fixture: the directive was still Grow/Expand on the turn the garrison
    /// walked off, and `Defend 24` only appeared after the city was already
    /// occupied. The ORDER-keyed defend terms below still require an order;
    /// the RISK term does not.
    #[test]
    fn city_risk_is_priced_without_any_defend_order() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = defense_board(60);
        state.tribes.get_mut(&1).unwrap().units.push(combat_unit(60, UnitType::Rider, 1));
        let goal = MacroGoal {
            orders: vec![],
            stance: Stance::Arm,
            save_target: None,
        };
        let aux = |s: &crate::states::GameState| {
            crate::ai::oracle_macro::compute_goal_aux(s, 1, &goal, 0, 0, None)
        };
        let quiet = goal_potential(&state, 1, &goal, Some(&aux(&state)));
        state.tribes.get_mut(&2).unwrap().units.push(combat_unit(59, UnitType::Swordsman, 2));
        let besieged = goal_potential(&state, 1, &goal, Some(&aux(&state)));
        assert!(
            besieged < quiet,
            "a reachable enemy must cost potential even with no Defend order: \
             quiet {quiet}, besieged {besieged}"
        );
    }
