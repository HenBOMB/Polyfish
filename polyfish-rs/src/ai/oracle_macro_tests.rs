//! Split to a separate file (Aug 2026) so the production module
//! this backs stays well under 1000 lines despite thorough coverage.

use super::*;
use super::test_support::*;
use crate::states::{TileState, TribeState};
use crate::types::TechnologyType;

/// A bare city with nothing happening: no military pressure either way.
#[test]
fn stance_strength_is_zero_arm_in_a_quiet_position() {
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
    state.tribes.insert(1, t1);
    let s = stance_pressure(&state, 1);
    assert_eq!(s.arm, 0.0);
    assert_eq!(s.cause, ArmCause::None);
}
/// The distinction the categorical stance throws away: one enemy scout near
/// one of three cities is a weak signal; a stack pressing the only city I
/// have, with nothing defending, is near-maximal. Both are `Stance::Arm`.
#[test]
fn threat_strength_scales_with_how_much_is_pressed() {
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    for idx in [12, 60, 108] {
        t1.cities.push(crate::states::CityState { idx, ..Default::default() });
    }
    // Own defenders sitting on each city.
    for idx in [12, 60, 108] {
        t1.units.push(unit_at(idx));
    }
    state.tribes.insert(1, t1);
    let mut t2 = TribeState::default();
    t2.units.push(unit_at(61)); // adjacent to city 60 only
    state.tribes.insert(2, t2);

    let weak = stance_pressure(&state, 1);
    assert_eq!(weak.cause, ArmCause::Threat);
    assert!(weak.arm > 0.0 && weak.arm < 0.25, "one of three cities, defended: {}", weak.arm);

    // Now: a single undefended city with three enemies on it.
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
    state.tribes.insert(1, t1);
    let mut t2 = TribeState::default();
    for idx in [59, 61, 71] {
        t2.units.push(unit_at(idx));
    }
    state.tribes.insert(2, t2);

    let dire = stance_pressure(&state, 1);
    assert_eq!(dire.cause, ArmCause::Threat);
    assert!(dire.arm > 0.9, "sole city, undefended, surrounded: {}", dire.arm);
    assert!(dire.arm > weak.arm * 3.0);
}
/// The other route to a high ARM: overwhelming force with somewhere to put
/// it. Reported as MOMENTUM, not THREAT — they want opposite economies.
#[test]
fn army_dominance_reads_as_momentum_not_threat() {
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    t1.cities.push(crate::states::CityState { idx: 0, ..Default::default() });
    // Six attackers massed on the enemy city at 60, far from home.
    for idx in [48, 49, 50, 59, 61, 70] {
        t1.units.push(unit_at(idx));
    }
    state.tribes.insert(1, t1);
    let mut t2 = TribeState::default();
    t2.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
    state.tribes.insert(2, t2);

    let s = stance_pressure(&state, 1);
    assert_eq!(s.cause, ArmCause::Momentum);
    assert!(s.arm > 0.9, "total army dominance with a target: {}", s.arm);
}
/// Parity is not momentum, however many units are on the board.
#[test]
fn even_armies_produce_no_momentum() {
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    t1.cities.push(crate::states::CityState { idx: 0, ..Default::default() });
    let mut t2 = TribeState::default();
    t2.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
    for idx in [30, 31] {
        t1.units.push(unit_at(idx));
        t2.units.push(unit_at(idx + 50));
    }
    state.tribes.insert(1, t1);
    state.tribes.insert(2, t2);
    let s = stance_pressure(&state, 1);
    assert_eq!(s.arm, 0.0, "parity must not read as momentum");
}
/// GROW tracks available economy: open villages to take, or stars that
/// could already be converted into population.
#[test]
fn grow_strength_rises_with_capturable_villages() {
    let quiet = {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        state.tribes.insert(1, t1);
        stance_pressure(&state, 1).grow
    };
    let mut state = state_with_villages(0, &[3, 5, 7]);
    state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
        idx: 60,
        ..Default::default()
    });
    let rich = stance_pressure(&state, 1).grow;
    assert!(rich > quiet, "three open villages must beat none: {rich} vs {quiet}");
    assert!(rich >= 1.0);
}
#[test]
fn retakeable_village_predicate_and_radius() {
    let mut state = state_with_villages(13, &[12]);
    state.settings.size = 11;
    // Neutral: not retakeable (still_capturable covers it).
    assert!(!retakeable_village(&state, 12, 1));
    // Enemy-owned, explored, within radius of our unit at 10: retakeable.
    state.tiles.get_mut(&12).unwrap().owner = 2;
    assert!(retakeable_village(&state, 12, 1));
    // Enemy capital: never painted.
    state.tiles.get_mut(&12).unwrap().capital_of = 2;
    assert!(!retakeable_village(&state, 12, 1));
    state.tiles.get_mut(&12).unwrap().capital_of = 0;
    // Beyond RETAKE_PAINT_RADIUS: not painted (move our unit far away).
    state.tribes.get_mut(&1).unwrap().units[0] = unit_at(120);
    assert!(!retakeable_village(&state, 12, 1));
}
#[test]
fn real_target_outranks_fog_guess_in_assignment() {
    // One unit, two targets: a fog guess NEARBY and a real explored
    // village further out — the unit must pair with the real one.
    let mut state = state_with_villages(0, &[5]);
    state.settings.size = 11;
    // Fog guess at 2 (no tile entry → unexplored), real village at 5.
    let pairs = assign_expand_targets(&state, 1, &[2, 5]);
    assert_eq!(pairs, vec![(0, 5)]);
}
#[test]
fn commitment_picks_nearest_is_sticky_and_retires_at_three_cities() {
    let mut state = state_with_villages(0, &[3, 5]);
    // Fresh pick: village at idx 3 is 3 tiles away vs 5 for idx 5.
    assert_eq!(update_commitment(&state, 1, None), Some(3));
    // Sticky: an existing valid commitment survives a nearer alternative.
    assert_eq!(update_commitment(&state, 1, Some(5)), Some(5));
    // Retires once the third city exists.
    let t1 = state.tribes.get_mut(&1).unwrap();
    for _ in 0..3 {
        t1.cities.push(Default::default());
    }
    assert_eq!(update_commitment(&state, 1, Some(5)), None);
}
#[test]
fn commitment_repicks_when_target_is_captured() {
    let mut state = state_with_villages(0, &[3, 5]);
    state.tiles.get_mut(&5).unwrap().owner = 2;
    assert_eq!(update_commitment(&state, 1, Some(5)), Some(3));
}
/// EXP_ELO_051 — Verdi: "we should be saving towards a lane if that is
/// what T1 says … the best computed path for that giant spam is forges."
/// A tribe holding Climbing+Mining is walking the Forge lane even when a
/// Windmill is cheaper, and a mountain that could take a Mine counts as a
/// Forge partner before the Mine is standing — otherwise the plan waits
/// on builds that nothing is planning.
#[test]
fn the_invested_lane_wins_and_future_mines_count_as_partners() {
    use crate::types::{TechnologyType as T, TerrainType};
    let mut state = state_with_villages(0, &[3, 5]);
    {
        let t1 = state.tribes.get_mut(&1).unwrap();
        for tech in [T::Climbing, T::Mining, T::Organization, T::Farming] {
            t1.tech_vanilla.push(crate::states::TechnologyState {
                tech_type: tech,
                discovered: true,
                discovered_turn: 0,
            });
        }
        t1.stars = 30;
        t1.cities.push(crate::states::CityState {
            idx: 60,
            owner: 1,
            _territory: vec![60, 61, 50, 72],
            production: 3,
            ..Default::default()
        });
    }
    // 61 is bare field (a Forge site); 50 and 72 are ore mountains with no
    // mine on them yet — the exact board that used to price zero.
    for idx in [60, 61, 50, 72] {
        let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
        tile.owner = 1;
        // Resource visibility is FOW-honest: unexplored ground is not a plan.
        tile.explorers.insert(1);
        tile.terrain_type = if idx == 50 || idx == 72 {
            TerrainType::Mountain
        } else {
            TerrainType::Field
        };
    }
    for idx in [50, 72] {
        state.resources.insert(
            idx,
            Some(crate::states::ResourceState {
                resource_type: crate::types::ResourceType::Metal,
            }),
        );
    }
    let plan = pick_save_lane(&state, 1, 0).expect("an unbuilt mine still makes a site");
    assert_eq!(
        plan.structure,
        crate::types::StructureType::Forge,
        "the invested lane must win, got {:?}",
        plan.structure
    );
    assert_eq!(plan.tech, T::Smithery);

    // …and the batch never grows past the next two placements.
    assert!(
        plan.structure_cost <= plan.structure_unit_cost * SAVE_MAX_PLACEMENTS,
        "structure_cost {} exceeds two placements",
        plan.structure_cost
    );
}
/// v7: SAVE fires only for a batch that is out of pocket now but inside
/// SAVE_MAX_TURNS of income, so it self-terminates instead of becoming an
/// open-ended hoard — the failure mode a savings reward invites.
#[test]
fn save_stance_targets_a_reachable_batch_and_self_terminates() {
    use crate::types::{StructureType, TechnologyType};
    let mut state = state_with_villages(0, &[3, 5]);
    {
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.stars = 40;
        for tech in [
            TechnologyType::Organization,
            TechnologyType::Farming,
            TechnologyType::Construction,
        ] {
            t1.tech_vanilla.push(crate::states::TechnologyState {
                tech_type: tech,
                discovered: true,
                discovered_turn: 0,
            });
        }
        t1.cities.push(crate::states::CityState {
            idx: 60,
            owner: 1,
            _territory: vec![60, 61, 50, 72],
            production: 2, // income, so the batch is reachable at all
            ..Default::default()
        });
    }
    // Two standing Farms around the empty field at 61 → a Windmill worth
    // banking for (2 partners clears SAVE_MIN_PARTNERS).
    for idx in [50, 72] {
        state.structures.insert(
            idx,
            Some(crate::states::StructureState {
                structure_type: StructureType::Farm,
                ..Default::default()
            }),
        );
    }
    for idx in [60, 61, 50, 72] {
        let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
        tile.owner = 1;
        tile.terrain_type = crate::types::TerrainType::Field;
    }
    assert_eq!(pick_save_lane(&state, 1, 0).map(|l| l.cost), Some(5),
        "one 5-star windmill, tech owned");

    // The lane is what costs: drop Construction and the batch must absorb
    // the tier-3 tech price, which is the thing actually worth banking for.
    {
        state
            .tribes
            .get_mut(&1)
            .unwrap()
            .tech_vanilla
            .retain(|t| t.tech_type != TechnologyType::Construction);
        let tech_cost = crate::functions::get_tech_cost(
            &state.tribes[&1],
            TechnologyType::Construction,
        );
        assert!(tech_cost > 0);
        assert_eq!(pick_save_lane(&state, 1, 0).map(|l| l.cost), Some(5 + tech_cost));
        state.tribes.get_mut(&1).unwrap().tech_vanilla.push(
            crate::states::TechnologyState {
                tech_type: TechnologyType::Construction,
                discovered: true,
                discovered_turn: 0,
            },
        );
    }

    // Broke but within reach → SAVE with the batch named.
    state.tribes.get_mut(&1).unwrap().stars = 1;
    let g = compute_macro_goal(&state, 1, 0);
    assert_eq!(g.stance, Stance::Save);
    assert_eq!(g.save_target.as_ref().map(|l| l.cost), Some(5));

    // Already affordable → nothing to save for, back to GROW.
    state.tribes.get_mut(&1).unwrap().stars = 5;
    let g = compute_macro_goal(&state, 1, 0);
    assert_eq!(g.stance, Stance::Grow);
    assert_eq!(g.save_target, None);

    // Out of reach (no income, batch unaffordable for SAVE_MAX_TURNS) →
    // GROW rather than an indefinite hoard.
    state.tribes.get_mut(&1).unwrap().stars = 0;
    state.tribes.get_mut(&1).unwrap().cities[0].production = 0;
    let far = compute_macro_goal(&state, 1, 0);
    assert!(
        far.stance != Stance::Save || far.save_target.is_some(),
        "SAVE is only ever set together with a named target"
    );
}
/// v7: a discretionary stance swing must hold for STANCE_SWITCH_TURNS
/// turns, and re-running the same turn's plies must not advance the streak
/// (the goal-setter runs every ply, the commitment counts turns).
#[test]
fn stance_commitment_damps_discretionary_swings_across_turns() {
    let mut st = StanceCommit::default();
    let mut state = state_with_villages(0, &[3, 5]);
    state.settings.turn = 1;

    // First read commits immediately — nothing to be loyal to yet.
    assert_eq!(commit_macro_goal(&state, 1, &mut st, 0).stance, Stance::Grow);
    assert_eq!(st.stance, Some(Stance::Grow));

    // Force the script to want ARM: post-expansion "prepare" phase — an
    // explored enemy city we outweigh but cannot yet storm, at 3+ cities.
    let mut t2 = TribeState::default();
    t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
    t2.units.push(unit_at(41));
    state.tribes.insert(2, t2);
    explore_tile(&mut state, 40);
    let t1 = state.tribes.get_mut(&1).unwrap();
    t1.units.push(unit_at(29));
    for _ in 0..3 {
        t1.cities.push(Default::default());
    }
    assert_eq!(
        compute_macro_goal(&state, 1, 0).stance,
        Stance::Arm,
        "precondition: script wants ARM here"
    );

    // Same turn, several plies: the challenger must not accrue a streak.
    for _ in 0..4 {
        assert_eq!(
            commit_macro_goal(&state, 1, &mut st, 0).stance,
            Stance::Grow,
            "extra plies of one turn must not buy a stance switch"
        );
    }
    // Next turn: streak reaches STANCE_SWITCH_TURNS and the switch lands.
    state.settings.turn = 2;
    assert_eq!(commit_macro_goal(&state, 1, &mut st, 0).stance, Stance::Arm);
    assert_eq!(st.stance_flips, 1);
}
/// Threat responses bypass the hysteresis — a DEFEND order means an enemy
/// is already inside the threat radius, and arriving two turns late is the
/// same as not arriving.
#[test]
fn stance_commitment_lets_threat_response_switch_immediately() {
    let mut st = StanceCommit::default();
    let mut state = state_with_villages(0, &[3, 5]);
    state.settings.turn = 1;
    assert_eq!(commit_macro_goal(&state, 1, &mut st, 0).stance, Stance::Grow);

    // Visible deliverable strike on an own city → DEFEND → ARM (040:
    // threat math, not the old position count — stats must be real).
    state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
        idx: 0,
        ..Default::default()
    });
    let t2 = state.tribes.entry(2).or_insert_with(TribeState::default);
    t2.id = 2;
    t2.units.push(unit_at(1));
    t2.units.push(unit_at(11));
    explore_tile(&mut state, 0);
    explore_tile(&mut state, 1);
    explore_tile(&mut state, 11);
    {
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units[0].owner = 1;
        t1.units[0].health = 10.0;
    }
    for u in state.tribes.get_mut(&2).unwrap().units.iter_mut() {
        u.owner = 2;
        u.health = 10.0;
    }
    let g = commit_macro_goal(&state, 1, &mut st, 0);
    assert!(g.orders.iter().any(|(k, _)| *k == OrderKind::Defend));
    assert_eq!(g.stance, Stance::Arm, "threat response must not wait");
    assert_eq!(st.stance_flips, 1);
}
#[test]
fn scripted_goal_paints_expand_attack_defend_and_sets_stance() {
    let mut state = state_with_villages(0, &[3, 5]);
    // Under 3 cities with two capturable villages → two EXPAND orders,
    // sorted, GROW stance, star gate active.
    let g = compute_macro_goal(&state, 1, 0);
    assert_eq!(
        g.orders,
        vec![(OrderKind::Expand, 3), (OrderKind::Expand, 5)]
    );
    assert_eq!(g.stance, Stance::Grow);
    assert!(tech_discipline_active(&state, 1, &g));

    // Explored enemy city at 40 = (3,7), two own units within Chebyshev 3
    // (39 = (3,6) and 29 = (2,7)), no defenders → superiority → ATTACK.
    let mut t2 = TribeState::default();
    t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
    state.tribes.insert(2, t2);
    explore_tile(&mut state, 40);
    let t1 = state.tribes.get_mut(&1).unwrap();
    t1.units.push(unit_at(39));
    t1.units.push(unit_at(29));
    let g = compute_macro_goal(&state, 1, 0);
    assert!(g.orders.contains(&(OrderKind::Attack, 40)));
    assert_eq!(g.stance, Stance::Grow);

    // Threatened own city → DEFEND + ARM stance. 040 contract: a single
    // VISIBLE enemy that can reach the unguarded city suffices (the old
    // `near >= 2` proxy is gone, and hidden units never count).
    state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
        idx: 0,
        ..Default::default()
    });
    let t2 = state.tribes.get_mut(&2).unwrap();
    t2.units.push(unit_at(1));
    t2.units.push(unit_at(12));
    explore_tile(&mut state, 0);
    explore_tile(&mut state, 1);
    explore_tile(&mut state, 12);
    // 040 threat math reads real stats: garrison + attackers need owner
    // and HP (the old proxy counted bare positions).
    {
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units[0].owner = 1;
        t1.units[0].health = 10.0;
    }
    for u in state.tribes.get_mut(&2).unwrap().units.iter_mut() {
        u.owner = 2;
        u.health = 10.0;
    }
    let g = compute_macro_goal(&state, 1, 0);
    assert!(g.orders.contains(&(OrderKind::Defend, 0)));
    assert_eq!(g.stance, Stance::Arm);
}
#[test]
fn attack_requires_local_superiority() {
    let mut state = state_with_villages(0, &[3]);
    let mut t2 = TribeState::default();
    t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
    // Two defenders within Chebyshev 2 of their city match our two
    // attackers' value — no superiority, no ATTACK order.
    t2.units.push(unit_at(41));
    t2.units.push(unit_at(51));
    state.tribes.insert(2, t2);
    explore_tile(&mut state, 40);
    let t1 = state.tribes.get_mut(&1).unwrap();
    t1.units.push(unit_at(39));
    t1.units.push(unit_at(29));
    let g = compute_macro_goal(&state, 1, 0);
    assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));

    // A third attacker reaches parity-plus but not the 1.5x margin.
    state.tribes.get_mut(&1).unwrap().units.push(unit_at(30));
    let g = compute_macro_goal(&state, 1, 0);
    assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));

    // A fourth clears the margin → ATTACK.
    state.tribes.get_mut(&1).unwrap().units.push(unit_at(20));
    let g = compute_macro_goal(&state, 1, 0);
    assert!(g.orders.contains(&(OrderKind::Attack, 40)));

    // Unexplored enemy city never draws an order.
    state.tiles.get_mut(&40).unwrap().explorers.clear();
    let g = compute_macro_goal(&state, 1, 0);
    assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));
}
#[test]
fn prepare_arms_post_expansion_when_massing_would_win() {
    // Explored enemy city, one own unit in approach range (cheb 4), army
    // outweighs the garrison but local force is short → prepare.
    let mut state = state_with_villages(0, &[3]);
    let mut t2 = TribeState::default();
    t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
    t2.units.push(unit_at(41));
    state.tribes.insert(2, t2);
    explore_tile(&mut state, 40);
    let t1 = state.tribes.get_mut(&1).unwrap();
    t1.units.push(unit_at(29));

    // Still expanding (<3 cities): prepare must NOT override GROW.
    let g = compute_macro_goal(&state, 1, 0);
    assert_eq!(g.stance, Stance::Grow);

    let t1 = state.tribes.get_mut(&1).unwrap();
    for _ in 0..3 {
        t1.cities.push(Default::default());
    }
    let g = compute_macro_goal(&state, 1, 0);
    assert_eq!(g.stance, Stance::Arm);
    assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));
}
#[test]
fn expand_persists_past_third_city_but_gate_retires() {
    let mut state = state_with_villages(0, &[3, 5]);
    let t1 = state.tribes.get_mut(&1).unwrap();
    for _ in 0..3 {
        t1.cities.push(Default::default());
    }
    let g = compute_macro_goal(&state, 1, 0);
    assert!(g.orders.contains(&(OrderKind::Expand, 3)));
    assert_eq!(g.stance, Stance::Grow);
    assert!(!tech_discipline_active(&state, 1, &g));
}
/// A lane's price is the whole path to it, not just the last tech.
#[test]
fn tech_chain_cost_prices_undiscovered_prerequisites() {
    use crate::settings::technology::get_technology_setting;
    let mut t1 = TribeState::default();
    t1.cities.push(crate::states::CityState::default());
    let direct = crate::functions::get_tech_cost(&t1, TechnologyType::Trade);
    let chain = tech_chain_cost(&t1, TechnologyType::Trade);
    assert!(
        chain > direct,
        "Trade sits behind Roads behind Riding — the chain must cost more \
         than the tech alone ({chain} vs {direct})"
    );
    // Owning the prerequisite removes its cost from the chain.
    let req = get_technology_setting(TechnologyType::Trade).requires.unwrap();
    t1.tech_vanilla.push(crate::states::TechnologyState {
        tech_type: req,
        discovered: true,
        discovered_turn: 0,
    });
    assert_eq!(tech_chain_cost(&t1, TechnologyType::Trade), direct);
}
/// A lane the tier-3 cap will refuse is not a plan — it is a hoard with no
/// exit. v7 shipped priced-but-unbuyable lanes; this pins the fix.
#[test]
fn save_batch_skips_lanes_the_tier3_cap_will_refuse() {
    use crate::types::{StructureType, TechnologyType, TerrainType};
    let mut state = state_with_villages(0, &[3, 5]);
    {
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.stars = 40;
        for tech in [TechnologyType::Organization, TechnologyType::Farming] {
            t1.tech_vanilla.push(crate::states::TechnologyState {
                tech_type: tech,
                discovered: true,
                discovered_turn: 0,
            });
        }
        t1.cities.push(crate::states::CityState {
            idx: 60,
            owner: 1,
            _territory: vec![60, 61, 50, 72],
            production: 2,
            ..Default::default()
        });
    }
    for idx in [50, 72] {
        state.structures.insert(
            idx,
            Some(crate::states::StructureState {
                structure_type: StructureType::Farm,
                ..Default::default()
            }),
        );
    }
    for idx in [60, 61, 50, 72] {
        let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
        tile.owner = 1;
        tile.terrain_type = TerrainType::Field;
    }
    // Construction unowned: the lane is priced with its full chain.
    let with_budget = pick_save_lane(&state, 1, 0).expect("lane priced").cost;
    assert!(with_budget > 5, "chain cost must be included, got {with_budget}");
    // Tier-3 budget spent: the same lane is unreachable and must vanish.
    assert!(pick_save_lane(&state, 1, TIER3_CAP_PER_GAME).is_none());
}
#[test]
fn recommended_techs_follow_the_environment() {
    use crate::states::TechnologyState;
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    t1.units.push(unit_at(0));
    // EXP_ELO_055: recommended_techs now ranks via evaluate_tech_utility,
    // which counts resources/terrain from CITY TERRITORY, not every explored
    // tile — so the fixture needs a city whose territory covers the ridge.
    t1.cities.push(crate::states::CityState {
        idx: 12,
        owner: 1,
        _territory: (10..16).collect(),
        ..Default::default()
    });
    state.tribes.insert(1, t1);
    // Explored mountain ridge with metal → mountain line: Climbing first.
    for idx in 10..16 {
        let mut tile = TileState::default();
        tile.terrain_type = crate::types::TerrainType::Mountain;
        tile.owner = 1;
        tile.explorers.insert(1);
        state.tiles.insert(idx, tile);
    }
    state.resources.insert(
        11,
        Some(crate::states::ResourceState {
            resource_type: crate::types::ResourceType::Metal,
        }),
    );
    let recs = recommended_techs(&state, 1);
    assert_eq!(recs, vec![TechnologyType::Climbing]);

    // Owning Climbing + Mining advances the line to Smithery.
    let t1 = state.tribes.get_mut(&1).unwrap();
    for tech in [TechnologyType::Climbing, TechnologyType::Mining] {
        t1.tech_vanilla.push(TechnologyState {
            tech_type: tech,
            discovered: true,
            discovered_turn: 0,
        });
    }
    let recs = recommended_techs(&state, 1);
    assert_eq!(recs, vec![TechnologyType::Smithery]);
}
#[test]
fn guessed_sites_respect_generator_rules_and_spread() {
    // Capital city at the center, nothing else explored: guesses must be
    // unexplored, on the legal edge bands, >=3 from the capital and from
    // each other, and nearest-first from the unit.
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    t1.units.push(unit_at(60));
    t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
    state.tribes.insert(1, t1);
    explore_tile(&mut state, 60);

    let sites: Vec<i32> = guess_villages(&state, 1, 2).iter().map(|g| g.tile).collect();
    assert_eq!(sites.len(), 2);
    let cheb = |a: i32, b: i32| ((a / 11) - (b / 11)).abs().max(((a % 11) - (b % 11)).abs());
    for &s in &sites {
        let (r, c) = (s / 11, s % 11);
        let edge = r.min(10 - r).min(c).min(10 - c);
        assert!(edge >= 2 && edge != 3, "site {s} off the generator's bands");
        assert!(cheb(s, 60) >= 3, "site {s} too close to the known capital");
        assert!(cheb(s, 60) <= 4, "site {s} not nearest-first");
    }
    assert!(cheb(sites[0], sites[1]) >= 3, "guesses must spread");

    // A known village nearby suppresses guesses in its exclusion zone.
    add_visible_village(&mut state, 24); // (2,2)
    let sites = guess_villages(&state, 1, 4);
    assert!(sites.iter().all(|g| cheb(g.tile, 24) >= 3));

    // And compute_macro_goal paints guesses whenever real targets run short.
    let g = compute_macro_goal(&state, 1, 0);
    let expands: Vec<i32> = g
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Expand)
        .map(|(_, i)| *i)
        .collect();
    assert!(expands.contains(&24)); // the real village
    assert_eq!(expands.len(), EXPAND_TARGET_MIN); // topped up with a guess
}
#[test]
fn expand_assignment_is_unique_and_nearest_first() {
    let mut state = state_with_villages(0, &[4, 44]);
    state.tribes.get_mut(&1).unwrap().units.push(unit_at(40));
    // Two units, two targets: greedy nearest-pair-first must cover both
    // targets with distinct units — never two scouts on one target.
    let pairs = assign_expand_targets(&state, 1, &[4, 44]);
    assert_eq!(pairs.len(), 2);
    let units: std::collections::HashSet<i32> = pairs.iter().map(|(u, _)| *u).collect();
    let targets: std::collections::HashSet<i32> = pairs.iter().map(|(_, t)| *t).collect();
    assert_eq!(units.len(), 2, "each unit assigned at most once");
    assert_eq!(targets.len(), 2, "each target assigned at most once");
}
#[test]
fn guessed_sites_spread_across_quadrants() {
    // Anchor in the center; legal spots exist in multiple quadrants.
    let state = state_with_villages(60, &[]);
    let picks: Vec<i32> = guess_villages(&state, 1, 2).iter().map(|g| g.tile).collect();
    assert_eq!(picks.len(), 2);
    let size = 11;
    let q = |idx: i32| ((idx % size > 5) as u8) * 2 + ((idx / size > 5) as u8);
    assert_ne!(q(picks[0]), q(picks[1]), "guesses should span distinct quadrants");
}
#[test]
fn goal_star_gate_is_stance_aware() {
    let mut state = state_with_villages(0, &[3]);
    // ARM gates regardless of expansion state.
    let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
    assert!(tech_discipline_active(&state, 1, &arm));
    // GROW gates only inside the expansion window.
    let grow = MacroGoal {
        orders: vec![(OrderKind::Expand, 3)],
        stance: Stance::Grow,
        save_target: None,
    };
    assert!(tech_discipline_active(&state, 1, &grow));
    let t1 = state.tribes.get_mut(&1).unwrap();
    for _ in 0..3 {
        t1.cities.push(Default::default());
    }
    assert!(!tech_discipline_active(&state, 1, &grow));
}
