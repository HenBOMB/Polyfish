//! Split to a separate file (Aug 2026) so the production module
//! this backs stays well under 1000 lines despite thorough coverage.

use super::*;
use super::super::test_support::*;
use crate::coords::Coords;
use crate::settings::units::get_unit_setting;
use crate::states::{TileState, TribeState, UnitState};
use crate::types::UnitType;

    /// A Smithery/Forge-shaped lane: `cost = tech_cost + structure_cost`.
    fn build_test_lane(cost: i32, tech_cost: i32, structure_cost: i32) -> crate::ai::oracle_macro::SaveTarget {
        crate::ai::oracle_macro::SaveTarget {
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
    /// v11: ARM's Φ should blend LINEARLY with `arm_strength` — full intensity
    /// reproduces the old flat formula exactly (no aux defaults to full, so
    /// this also pins the no-aux fallback), and the same army composition at
    /// half intensity must land exactly at the midpoint between the full- and
    /// zero-intensity readings (a straight line, not just "somewhere lower").
    #[test]
    fn arm_phi_blends_linearly_with_intensity() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState {
            idx: 60,
            owner: 1,
            level: 1,
            production: 3,
            ..Default::default()
        });
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };

        let full = GoalAux { arm_strength: 1.0, ..Default::default() };
        let zero = GoalAux { arm_strength: 0.0, ..Default::default() };
        let half = GoalAux { arm_strength: 0.5, ..Default::default() };

        let arm_full = goal_potential(&state, 1, &arm, Some(&full));
        let arm_default = goal_potential(&state, 1, &arm, None);
        assert!(
            (arm_full - arm_default).abs() < 1e-3,
            "no-aux fallback should match full intensity: {arm_default} vs {arm_full}"
        );

        let arm_zero = goal_potential(&state, 1, &arm, Some(&zero));
        assert!(
            (arm_full - arm_zero).abs() > 1.0,
            "test setup: ARM's and GROW's SPT rate must actually differ here: \
             full {arm_full}, zero {arm_zero}"
        );

        let arm_half = goal_potential(&state, 1, &arm, Some(&half));
        let expected_half = 0.5 * arm_full + 0.5 * arm_zero;
        assert!(
            (arm_half - expected_half).abs() < 1e-3,
            "half intensity did not land at the midpoint: {arm_half} vs {expected_half}"
        );
    }
    /// v11: army VALUE must price identically regardless of intensity — only
    /// the SPT rate blends. A besieged state (same army, an enemy now near)
    /// must not score higher than a quiet one purely because the higher
    /// intensity "reveals" the value of units that were there all along;
    /// `city_risk_is_priced_without_any_defend_order` is the sharper version
    /// of this with a real risk assessment — this pins the mechanism directly.
    #[test]
    fn arm_army_value_does_not_depend_on_intensity() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
        let low = GoalAux { arm_strength: 0.1, ..Default::default() };
        let high = GoalAux { arm_strength: 0.9, ..Default::default() };

        // Same army, no SPT (no cities), so the SPT-rate blend contributes
        // nothing at all — any remaining gap would be the army term leaking.
        let phi_low = goal_potential(&state, 1, &arm, Some(&low));
        let phi_high = goal_potential(&state, 1, &arm, Some(&high));
        assert!(
            (phi_low - phi_high).abs() < 1e-3,
            "army value must not depend on intensity: low {phi_low} vs high {phi_high}"
        );
    }
    /// v11: a live SAVE plan must not go fully dark just because a marginal
    /// ARM signal won the discrete stance pick — it pays the
    /// `(1 - arm_strength)` remainder of the ramp instead of zero.
    #[test]
    fn savings_ramp_stays_partially_live_under_marginal_arm() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.id = 1;
        t1.stars = 10;
        state.tribes.insert(1, t1);
        let lane = build_test_lane(20, 16, 4);
        let bare_arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
        let arming = MacroGoal {
            orders: vec![],
            stance: Stance::Arm,
            save_target: Some(lane.clone()),
        };
        let saving = MacroGoal { orders: vec![], stance: Stance::Save, save_target: Some(lane) };
        let bare_grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };

        let ramp_at_save =
            goal_potential(&state, 1, &saving, None) - goal_potential(&state, 1, &bare_grow, None);
        assert!(ramp_at_save > 1.0, "test setup: half-banked ramp should be well above zero");

        // Full-intensity ARM: the plan carries none of the ramp, exactly like
        // before this change (a real emergency shouldn't pay to keep banking).
        let full = GoalAux { arm_strength: 1.0, ..Default::default() };
        let paid_full = goal_potential(&state, 1, &arming, Some(&full))
            - goal_potential(&state, 1, &bare_arm, Some(&full));
        assert!(paid_full.abs() < 1e-3, "full-intensity ARM should pay zero savings ramp: {paid_full}");

        // Marginal ARM (0.3 intensity): the plan carries 70% of the ramp it
        // would under SAVE outright.
        let marginal = GoalAux { arm_strength: 0.3, ..Default::default() };
        let paid_marginal = goal_potential(&state, 1, &arming, Some(&marginal))
            - goal_potential(&state, 1, &bare_arm, Some(&marginal));
        assert!(
            (paid_marginal - 0.7 * ramp_at_save).abs() < 1e-3,
            "marginal ARM paid {paid_marginal}, expected 70% of {ramp_at_save}"
        );
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
    fn goal_potential_pays_lane_preferred_units() {
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
        assert!((with - base - SHAPE_GOAL_LANE_PER_COST * archer_cost).abs() < 1e-3);
    }
    #[test]
    fn lane_per_cost_prices_knight_above_defender() {
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
        assert!((knight - SHAPE_GOAL_LANE_PER_COST * k_cost).abs() < 1e-3);
        assert!((defender - SHAPE_GOAL_LANE_PER_COST * d_cost).abs() < 1e-3);
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

        let mk = |stance, save: Option<crate::ai::oracle_macro::SaveTarget>| MacroGoal {
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

    /// Per-unit-goal design (Aug 2026), Step 3 verification: the legacy
    /// EXPAND pricing (`None`) re-matches unit<->target fresh on every
    /// `goal_potential` call, so one unit's candidate move can change which
    /// target a completely different, NON-moving unit is priced against.
    /// Threading a frozen `UnitGoalStore` (`Some`) must make that
    /// non-moving unit's contribution invariant to the other unit's move.
    ///
    /// Fixture: two Warriors, two explored villages, positioned so the
    /// nearest-pair greedy match picks (A->V1, B->V2) initially, then A
    /// teleports next to V2 -- flipping the fresh match to (A->V2, B->V1)
    /// even though B never moved. All non-Expand Φ terms are position-
    /// independent here (no cities/economy/other orders), so the total Φ
    /// delta between the two states is exactly the Expand block's delta --
    /// no decomposition needed to observe the effect.
    #[test]
    fn unit_goal_store_makes_a_non_moving_units_pricing_invariant_to_a_teammates_move() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        use crate::ai::search::unit_goals::{UnitGoal, UnitGoalStore};

        const V1: i32 = 5; // row0,col5
        const V2: i32 = 115; // row10,col5
        const A_ID: u32 = 1;
        const B_ID: u32 = 2;

        let mut before = GameState::default();
        before.settings.size = 11;
        add_visible_village(&mut before, V1);
        add_visible_village(&mut before, V2);
        let mut t1 = TribeState::default();
        t1.units.push(UnitState { id: A_ID, ..unit_at(0, UnitType::Warrior) }); // row0,col0: d(V1)=5, d(V2)=10
        t1.units.push(UnitState { id: B_ID, ..unit_at(110, UnitType::Warrior) }); // row10,col0: d(V1)=10, d(V2)=5
        before.tribes.insert(1, t1);

        let mut after = before.clone();
        // A teleports to row10,col4: d(V1)=10, d(V2)=1. B is untouched.
        after.tribes.get_mut(&1).unwrap().units[0] = UnitState { id: A_ID, ..unit_at(114, UnitType::Warrior) };

        let goal = MacroGoal {
            orders: vec![(OrderKind::Expand, V1), (OrderKind::Expand, V2)],
            stance: Stance::Grow,
            save_target: None,
        };

        // Legacy (None): fresh greedy match every call -- B's implied
        // target flips from V2 (d=5) to V1 (d=10) purely because A moved.
        let phi_before_none = goal_potential(&before, 1, &goal, None);
        let phi_after_none = goal_potential(&after, 1, &goal, None);
        let term = |d: i32| SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP - d).max(0) as f32;
        let predicted_none_delta = (term(1) + term(10)) - (term(5) + term(5));
        assert!(
            (phi_after_none - phi_before_none - predicted_none_delta).abs() < 1e-3,
            "None: expected legacy delta {predicted_none_delta}, got {}",
            phi_after_none - phi_before_none
        );
        assert!(
            phi_after_none != phi_before_none,
            "sanity: the fixture must actually trigger a reassignment under the legacy path"
        );

        // Store-backed (Some): the assignment made at `before` (mirroring
        // what the legacy match would have picked at that same state) is
        // frozen and reused for `after` -- B's target stays V2, d=5,
        // completely unaffected by A having moved.
        let mut store = UnitGoalStore::default();
        store.assign(A_ID, UnitGoal { kind: OrderKind::Expand, target: V1 });
        store.assign(B_ID, UnitGoal { kind: OrderKind::Expand, target: V2 });
        let phi_before_some = goal_potential_with_unit_goals(&before, 1, &goal, None, None, Some(&store));
        let phi_after_some = goal_potential_with_unit_goals(&after, 1, &goal, None, None, Some(&store));
        // Before-state pricing is identical either way (same assignment).
        assert!((phi_before_some - phi_before_none).abs() < 1e-3);
        let predicted_some_delta = (term(10) + term(5)) - (term(5) + term(5)); // only A's own term moves
        assert!(
            (phi_after_some - phi_before_some - predicted_some_delta).abs() < 1e-3,
            "Some: expected store-backed delta {predicted_some_delta}, got {}",
            phi_after_some - phi_before_some
        );
        assert!(
            phi_after_some != phi_after_none,
            "the frozen store must diverge from the legacy re-match once A has moved"
        );
    }

    /// Regression for the seed 1787500020 double-dip: a unit already
    /// pursuing its OWN Expand goal must not also move the "unassigned
    /// target" gradient for a completely different, unclaimed order target
    /// just by wandering near it -- only a genuinely idle unit could ever
    /// end up claiming that target, so only idle units should move its
    /// pricing. This is what let a unit walk past an adjacent, capturable
    /// village toward its own (bad) guess: the incidental "unassigned"
    /// credit for a THIRD village out-scored actually reaching the one next
    /// to it.
    ///
    /// Fixture: A holds its own goal (OWN) and is always too far from OWN
    /// to score there (pinned at 0 both before/after, isolating the effect
    /// under test). A starts farther than B from OTHER (an unclaimed order
    /// target), then moves closer than B. Pre-fix, the "closest unit"
    /// search included A, so OTHER's gradient would track A's shrinking
    /// distance even though A was never going to pursue it. Post-fix, A is
    /// excluded (it has its own goal); B never moves, so OTHER's gradient
    /// -- and the whole Φ delta, since nothing else in this bare fixture
    /// depends on position -- must stay flat despite A's approach.
    #[test]
    fn unassigned_target_gradient_ignores_a_unit_with_its_own_goal() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        use crate::ai::search::unit_goals::{UnitGoal, UnitGoalStore};

        const OWN: i32 = 0; // row0,col0
        const OTHER: i32 = 115; // row10,col5
        const A_ID: u32 = 1;
        const B_ID: u32 = 2;

        let mut before = GameState::default();
        before.settings.size = 11;
        add_visible_village(&mut before, OWN);
        add_visible_village(&mut before, OTHER);
        let mut t1 = TribeState::default();
        t1.units.push(UnitState { id: A_ID, ..unit_at(110, UnitType::Warrior) }); // row10,col0: d(OWN)=10, d(OTHER)=5
        t1.units.push(UnitState { id: B_ID, ..unit_at(22, UnitType::Warrior) }); // row2,col0: d(OTHER)=8, constant
        before.tribes.insert(1, t1);

        let mut after = before.clone();
        // A moves to row10,col4: d(OWN)=10 (unchanged, still pinned at 0),
        // d(OTHER)=1 (much closer). B is untouched.
        after.tribes.get_mut(&1).unwrap().units[0] = UnitState { id: A_ID, ..unit_at(114, UnitType::Warrior) };

        let goal = MacroGoal {
            orders: vec![(OrderKind::Expand, OWN), (OrderKind::Expand, OTHER)],
            stance: Stance::Grow,
            save_target: None,
        };
        let mut store = UnitGoalStore::default();
        store.assign(A_ID, UnitGoal { kind: OrderKind::Expand, target: OWN });

        let phi_before = goal_potential_with_unit_goals(&before, 1, &goal, None, None, Some(&store));
        let phi_after = goal_potential_with_unit_goals(&after, 1, &goal, None, None, Some(&store));
        assert!(
            (phi_after - phi_before).abs() < 1e-3,
            "A approaching a target it doesn't own must not move Φ: before {phi_before}, after {phi_after}"
        );

        // Sanity: the fixture must actually be capable of triggering the
        // bug -- confirm the pre-fix formula (closest of ALL units, A
        // included) really would have produced a nonzero delta here.
        let term = |d: i32| SHAPE_UNIT_GOAL_PER_TILE * (SHAPE_PROX_CAP - d).max(0) as f32;
        let predicted_buggy_delta = term(1) - term(5); // A's distance to OTHER: 5 -> 1
        assert!(
            predicted_buggy_delta.abs() > 1e-3,
            "sanity: fixture must exercise a real distance change for the pre-fix formula"
        );
    }

    /// Regression for the turn-1 capital-return incident (seed 1787500002):
    /// a unit sitting on an owned city that still has open Train capacity
    /// must cost potential on the real trajectory, refunded the instant it
    /// steps off -- `moves/summon.rs` cannot legally train there while
    /// occupied.
    #[test]
    fn parking_on_a_city_with_train_capacity_costs_potential_when_a_store_is_threaded() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::ai::search::unit_goals::UnitGoalStore;
        let goal = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let store = UnitGoalStore::default();

        let mut occupied = defense_board(60);
        occupied.tribes.get_mut(&1).unwrap().units.push(combat_unit(60, UnitType::Warrior, 1));
        let phi_occupied =
            goal_potential_with_unit_goals(&occupied, 1, &goal, None, None, Some(&store));

        let mut vacated = occupied.clone();
        vacated.tribes.get_mut(&1).unwrap().units[0].coords = Coords::from_index(48, 11);
        let phi_vacated =
            goal_potential_with_unit_goals(&vacated, 1, &goal, None, None, Some(&store));

        assert!(
            (phi_vacated - phi_occupied - SHAPE_CITY_TRAIN_BLOCKED).abs() < 1e-3,
            "stepping off a train-capable city must refund exactly SHAPE_CITY_TRAIN_BLOCKED: \
             occupied {phi_occupied}, vacated {phi_vacated}"
        );
    }

    /// The legacy/rollout path (`unit_goals: None`, what every internal
    /// macro-mcts rollout call passes) must be completely untouched -- this
    /// term only prices the real trajectory, same scope as
    /// `SHAPE_UNIT_GOAL_PER_TILE`/`COMPLETE`.
    #[test]
    fn city_train_block_is_invisible_without_a_unit_goal_store() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let goal = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };

        let mut occupied = defense_board(60);
        occupied.tribes.get_mut(&1).unwrap().units.push(combat_unit(60, UnitType::Warrior, 1));
        let phi_occupied = goal_potential(&occupied, 1, &goal, None);

        let mut vacated = occupied.clone();
        vacated.tribes.get_mut(&1).unwrap().units[0].coords = Coords::from_index(48, 11);
        let phi_vacated = goal_potential(&vacated, 1, &goal, None);

        assert!(
            (phi_vacated - phi_occupied).abs() < 1e-3,
            "occupied {phi_occupied} vacated {phi_vacated} must match without a unit-goal store"
        );
    }

    /// A city already at (or over) its unit cap has no train capacity to
    /// protect -- occupying it must not be penalized.
    #[test]
    fn city_train_block_skips_a_city_already_at_unit_cap() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::ai::search::unit_goals::UnitGoalStore;
        let goal = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let store = UnitGoalStore::default();

        let mut state = defense_board(60);
        // CityState::default() level is 1 -- two units homed here already
        // makes count(2) > level(1), the same over-cap Summon gates on.
        for home_at in [70, 71] {
            let mut homed = unit_at(home_at, UnitType::Warrior);
            homed.owner = 1;
            homed.home_coords = Some(Coords::from_index(60, 11));
            state.tribes.get_mut(&1).unwrap().units.push(homed);
        }
        state.tribes.get_mut(&1).unwrap().units.push(combat_unit(60, UnitType::Warrior, 1));
        let phi_occupied = goal_potential_with_unit_goals(&state, 1, &goal, None, None, Some(&store));

        state.tribes.get_mut(&1).unwrap().units[2].coords = Coords::from_index(48, 11);
        let phi_vacated = goal_potential_with_unit_goals(&state, 1, &goal, None, None, Some(&store));

        assert!(
            (phi_vacated - phi_occupied).abs() < 1e-3,
            "a city already at its unit cap has nothing to protect: occupied {phi_occupied} \
             vacated {phi_vacated}"
        );
    }

    /// Aug 2026 regression: a Ruin's "free unit" reward summons the new unit
    /// ONTO the tile, and if the capturing unit is already standing there,
    /// the engine displaces it to an adjacent tile to make room -- found by
    /// watching a real replay where the capturer stepped OFF a Ruin it had
    /// an active Expand goal for, because the completion check required
    /// *that exact unit* to still be on the target. It must fire off
    /// occupancy by ANY of our units instead (matching `goal_outcome`'s
    /// already-correct pattern), since the displaced unit and the newly
    /// summoned one are equally good evidence the capture happened.
    #[test]
    fn ruin_completion_fires_even_when_the_capturer_is_displaced() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        use crate::ai::search::unit_goals::{UnitGoal, UnitGoalStore};

        const RUIN_IDX: i32 = 60;
        const CAPTURER_ID: u32 = 1;

        let mut state = GameState::default();
        state.settings.size = 11;
        state.tiles.insert(RUIN_IDX, TileState::default());
        // No entry in state.structures for RUIN_IDX -- captured, destroyed.
        let mut t1 = TribeState::default();
        // The goal-holder, displaced one tile off the Ruin after capture.
        t1.units.push(UnitState { id: CAPTURER_ID, owner: 1, ..unit_at(59, UnitType::Warrior) });
        // The newly-summoned unit, standing where the Ruin was.
        t1.units.push(UnitState { id: 2, owner: 1, ..unit_at(RUIN_IDX, UnitType::Swordsman) });
        state.tribes.insert(1, t1);

        let mut store = UnitGoalStore::default();
        store.assign(CAPTURER_ID, UnitGoal { kind: OrderKind::Expand, target: RUIN_IDX });

        let goal = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let (_, bd) = goal_potential_breakdown(&state, 1, &goal, None, None, Some(&store));
        let complete: f32 =
            bd.iter().filter(|(l, _)| *l == "unit_goal_complete").map(|(_, v)| v).sum();
        assert!(
            complete > 0.0,
            "Ruin completion must pay out even when the capturer was displaced by the reward, \
             got breakdown {bd:?}"
        );
    }

    /// Verdi's call (Aug 2026): a Ruin is a one-time reward, a Village a
    /// permanent second city, so a close Ruin shouldn't outbid the search
    /// for a first village just because it's nearer. The discount must be
    /// live before any village is found and lift exactly once one is.
    #[test]
    fn ruin_pull_is_discounted_before_the_first_village_and_at_parity_after() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        use crate::states::{CityState, StructureState};
        use crate::types::StructureType;

        const RUIN_IDX: i32 = 60;
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(RUIN_IDX, tile);
        state.structures.insert(
            RUIN_IDX,
            Some(StructureState { structure_type: StructureType::Ruin, level: 0, founded: 0 }),
        );
        let mut t1 = TribeState::default();
        t1.units.push(UnitState { id: 1, owner: 1, ..unit_at(50, UnitType::Warrior) });
        t1.cities.push(CityState { idx: 999, owner: 1, ..Default::default() }); // capital only
        state.tribes.insert(1, t1);

        let goal =
            MacroGoal { orders: vec![(OrderKind::Expand, RUIN_IDX)], stance: Stance::Grow, save_target: None };
        let approach = |s: &GameState| -> f32 {
            let (_, bd) = goal_potential_breakdown(s, 1, &goal, None, None, None);
            bd.iter().filter(|(l, _)| *l == "expand_approach").map(|(_, v)| v).sum()
        };

        let no_village = approach(&state);
        // A second city -- a village found -- lifts the discount.
        state.tribes.get_mut(&1).unwrap().cities.push(CityState { idx: 998, owner: 1, ..Default::default() });
        let with_village = approach(&state);

        assert!(
            no_village > 0.0 && with_village > 0.0,
            "sanity: the Ruin must actually be pulling in both states"
        );
        assert!(
            (no_village - SHAPE_GOAL_RUIN_W * with_village).abs() < 1e-3,
            "before the first village, Ruin pull must be exactly SHAPE_GOAL_RUIN_W of its \
             full-parity value: no_village={no_village} with_village={with_village}"
        );
    }
