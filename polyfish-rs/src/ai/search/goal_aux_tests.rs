//! Split to a separate file (Aug 2026) so the production module
//! this backs stays well under 1000 lines despite thorough coverage.

use super::*;
use crate::ai::oracle_macro::test_support::*;
use crate::ai::oracle_macro::{scripted_goal, MacroGoal, Stance};
use crate::moves::EndTurnMove;
use crate::moves::research::ResearchMove;
use crate::states::{StructureState, TileState, TribeState};
use crate::types::{TechnologyType, TerrainType};

/// End-to-end: a real generated Drylands game must report `water_dead`
/// through the same path self_play uses, and mask the naval lane there.
#[test]
fn a_generated_drylands_game_masks_the_water_lane() {
    let mut game = crate::game::Game::new();
    for seed in 0..8 {
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: crate::types::MapSize::Tiny,
            map_type: crate::types::MapType::Drylands,
            tribes: vec![crate::types::TribeType::Imperius, crate::types::TribeType::Bardur],
            seed,
            version: 115,
        });
        game.post_load();
        let goal = MacroGoal::default();
        let aux = scripted_goal_aux(&game.state, 1, &goal, 0, 0, None);
        assert!(aux.water_dead, "seed {seed}: generated Drylands still reads wet");
        assert!(
            !passes_tech_caps(&ResearchMove::new(TechnologyType::Fishing), &aux),
            "seed {seed}: Fishing survived the mask"
        );
    }
}
#[test]
fn capture_first_gate_blocks_attacks_from_capturable_tiles() {
    use crate::moves::attack::AttackMove;
    let mut state = state_with_villages(10, &[10]);
    state.settings.current_player_turn_id = 1;
    let attack = AttackMove::new(10, 11);

    // Standing on a neutral village: attack blocked.
    assert!(!passes_capture_first(&state, &attack));

    // Enemy-owned village (their city): still blocked — recapture instead.
    state.tiles.get_mut(&10).unwrap().owner = 2;
    assert!(!passes_capture_first(&state, &attack));

    // Own city tile: attack allowed.
    state.tiles.get_mut(&10).unwrap().owner = 1;
    assert!(passes_capture_first(&state, &attack));

    // Ruin: blocked.
    state.structures.insert(
        10,
        Some(StructureState {
            structure_type: StructureType::Ruin,
            level: 0,
            founded: 0,
        }),
    );
    assert!(!passes_capture_first(&state, &attack));

    // Plain tile: allowed; non-attack moves always pass.
    state.structures.shift_remove(&10);
    assert!(passes_capture_first(&state, &attack));
    assert!(passes_capture_first(&state, &EndTurnMove));
}
/// While banking, research that is not the batch's own next step is the
/// purchase that delays it — the Organization buy Verdi flagged.
#[test]
fn banking_gates_research_that_is_not_the_plan() {
    use crate::types::TechnologyType as T;
    let mut aux = GoalAux::default();
    aux.save_next_tech = Some(T::Mining);
    aux.recommended_techs = vec![T::Mining];
    assert!(passes_tech_caps(&ResearchMove::new(T::Mining), &aux));
    assert!(!passes_tech_caps(&ResearchMove::new(T::Organization), &aux));
    // No batch: the committed lane's recommendations are the whitelist.
    aux.save_next_tech = None;
    assert!(passes_tech_caps(&ResearchMove::new(T::Mining), &aux));
    assert!(!passes_tech_caps(&ResearchMove::new(T::Organization), &aux));
    // No opinion at all: nothing is gated on lane grounds.
    aux.recommended_techs.clear();
    assert!(passes_tech_caps(&ResearchMove::new(T::Organization), &aux));
}
#[test]
fn legacy_star_gate_blocks_research_at_any_star_count() {
    // Legacy (stance-less, EXP_ELO_026) arm: every tech is gated, and v9
    // removed the reserve escape — being rich no longer lifts it.
    let mut state = state_with_villages(0, &[3]);
    state.settings.current_player_turn_id = 1;
    let research = ResearchMove::new(TechnologyType::Organization);

    for stars in [0, 5, 50, 500] {
        state.tribes.get_mut(&1).unwrap().stars = stars;
        assert!(!passes_star_gate(&state, &research, None, None));
    }

    // Non-research moves always pass, regardless of stars.
    state.tribes.get_mut(&1).unwrap().stars = 0;
    assert!(passes_star_gate(&state, &EndTurnMove, None, None));
}
/// v9: the whole point of the dual-class exemption — Smithery opens the
/// Forge (giants) and fields a Swordsman, so no economy-or-army stance may
/// drop it. Same for Mathematics (Sawmill + Catapult).
#[test]
fn dual_class_tech_is_never_stance_gated() {
    let mut state = state_with_villages(0, &[3]);
    state.settings.current_player_turn_id = 1;
    state.tribes.get_mut(&1).unwrap().stars = 0;
    for tech in [TechnologyType::Smithery, TechnologyType::Mathematics] {
        let m = ResearchMove::new(tech);
        for stance in [Stance::Grow, Stance::Arm, Stance::Save] {
            assert!(
                passes_star_gate(&state, &m, Some(stance), None),
                "{tech:?} gated under {stance:?}"
            );
        }
    }
}
#[test]
fn stance_gate_is_granular_by_tech_class() {
    let mut state = state_with_villages(0, &[3]);
    state.settings.current_player_turn_id = 1;
    // Broke: nothing can meet the reserve, so gated == blocked.
    state.tribes.get_mut(&1).unwrap().stars = 0;
    let eco = ResearchMove::new(TechnologyType::Organization);
    let combat = ResearchMove::new(TechnologyType::Riding);
    let passage = ResearchMove::new(TechnologyType::Climbing);
    let mixed = ResearchMove::new(TechnologyType::Smithery);

    // GROW gates PURE-combat tech; eco, passage and dual-class flow freely
    // (Climbing carries a defense bonus but fields no unit).
    let grow = Some(Stance::Grow);
    assert!(passes_star_gate(&state, &eco, grow, None));
    assert!(passes_star_gate(&state, &passage, grow, None));
    assert!(!passes_star_gate(&state, &combat, grow, None));
    assert!(passes_star_gate(&state, &mixed, grow, None));

    // ARM flips it: pure-eco tech gated, unit tech (incl. mixed) free.
    let arm = Some(Stance::Arm);
    assert!(!passes_star_gate(&state, &eco, arm, None));
    assert!(passes_star_gate(&state, &combat, arm, None));
    assert!(passes_star_gate(&state, &mixed, arm, None));

    // SAVE is an economy stance and gates the same class GROW does — it
    // must not block the tech chain its own batch is priced to buy.
    let save = Some(Stance::Save);
    assert!(passes_star_gate(&state, &eco, save, None));
    assert!(passes_star_gate(&state, &mixed, save, None));
    assert!(!passes_star_gate(&state, &combat, save, None));

    // v9: no reserve — being rich no longer lifts a gated class.
    state.tribes.get_mut(&1).unwrap().stars = 500;
    assert!(!passes_star_gate(&state, &combat, grow, None));

    // UNLOCK gates nothing (no unlock policy yet).
    state.tribes.get_mut(&1).unwrap().stars = 0;
    assert!(passes_star_gate(&state, &combat, Some(Stance::Unlock), None));

    // v6: an active knight commitment makes its lane stance-coherent —
    // FreeSpirit passes under ARM and Chivalry under GROW, even broke;
    // without the commit both stay gated by their stance class.
    let free_spirit = ResearchMove::new(TechnologyType::FreeSpirit);
    let chivalry = ResearchMove::new(TechnologyType::Chivalry);
    let mut committed = GoalAux::default();
    committed.overlays.knight_commit = true;
    // Aug 14: the ARM eco-mask is intensity-conditional — it fires only
    // at near-certain pressure (arm_strength >= 0.98). A covered
    // skirmish (low strength) must NOT lock the eco lanes.
    let mut uncommitted = GoalAux::default();
    uncommitted.arm_strength = 1.0;
    assert!(passes_star_gate(&state, &chivalry, grow, Some(&committed)));
    assert!(passes_star_gate(&state, &free_spirit, arm, Some(&committed)));
    assert!(!passes_star_gate(&state, &chivalry, grow, Some(&uncommitted)));
    assert!(!passes_star_gate(&state, &free_spirit, arm, Some(&uncommitted)));
    let mut covered = GoalAux::default();
    covered.arm_strength = 0.3;
    assert!(
        passes_star_gate(&state, &free_spirit, arm, Some(&covered)),
        "low-intensity ARM must not mask eco tech"
    );
}
#[test]
fn market_ready_needs_three_cities_and_a_hub() {
    let mut state = GameState::default();
    state.settings.size = 11;
    let mut t1 = TribeState::default();
    for i in 0..3 {
        t1.cities.push(crate::states::CityState { idx: i, owner: 1, ..Default::default() });
    }
    state.tribes.insert(1, t1);
    // Three cities but no hub yet.
    assert!(!market_ready(&state, 1));
    // A windmill on own territory opens the lane.
    state.structures.insert(
        40,
        Some(StructureState {
            structure_type: StructureType::Windmill,
            level: 0,
            founded: 0,
        }),
    );
    state.tiles.entry(40).or_insert_with(TileState::default).owner = 1;
    assert!(market_ready(&state, 1));
    // Two cities: not ready even with the hub.
    state.tribes.get_mut(&1).unwrap().cities.pop();
    assert!(!market_ready(&state, 1));
}
#[test]
fn tier3_cap_exempts_chivalry_under_knight_commit() {
    let chivalry = ResearchMove::new(TechnologyType::Chivalry);
    let math = ResearchMove::new(TechnologyType::Mathematics);
    let mut aux = GoalAux::default();
    aux.tier3_bought = TIER3_CAP_PER_GAME;
    aux.overlays.knight_commit = true;
    aux.eco_tier3_owned = true; // v7: economy first, then the combat lane
    // Cap spent: Chivalry still passes under the commit; other tier-3s
    // stay blocked; without the commit Chivalry is blocked too (by the
    // stepping-stone rule AND the cap).
    assert!(passes_tech_caps(&chivalry, &aux));
    assert!(!passes_tech_caps(&math, &aux));
    aux.overlays.knight_commit = false;
    assert!(!passes_tech_caps(&chivalry, &aux));
}
/// v7 (Verdi): players almost never take knights before the level-3 pop
/// buildings, because those are what lead to giants. A combat tier-3 waits
/// for an economic one — and OWNERSHIP is the predicate, so a free
/// economy tier-3 out of a ruin unblocks it immediately.
#[test]
fn combat_tier3_waits_for_an_economic_tier3() {
    let chivalry = ResearchMove::new(TechnologyType::Chivalry);
    let construction = ResearchMove::new(TechnologyType::Construction);
    let mut aux = GoalAux::default();
    aux.overlays.knight_commit = true; // clears the stepping-stone rule
    aux.tier3_bought = 0; // budget available

    assert!(
        !passes_tech_caps(&chivalry, &aux),
        "combat tier-3 blocked while no economic tier-3 is owned"
    );
    assert!(
        passes_tech_caps(&construction, &aux),
        "the economic tier-3 itself is never blocked by the ordering rule"
    );
    aux.eco_tier3_owned = true;
    assert!(passes_tech_caps(&chivalry, &aux), "economy first, then knights");

    // Two slots now, so economy + combat both fit in one game.
    assert_eq!(TIER3_CAP_PER_GAME, 2);
    aux.tier3_bought = 1;
    assert!(passes_tech_caps(&chivalry, &aux));
    aux.tier3_bought = 2;
    assert!(!passes_tech_caps(&construction, &aux), "cap still binds at 2");
}
/// The economic/combat split must come from the settings tables, not a
/// hand list — the exact discipline `max_affordable_pop` failed at.
#[test]
fn eco_tier3_classification_is_table_derived() {
    use crate::settings::technology::is_eco_tier3;
    for t in [
        TechnologyType::Construction,
        TechnologyType::Mathematics,
        TechnologyType::Smithery,
        TechnologyType::Trade,
        TechnologyType::Philosophy,
    ] {
        assert!(is_eco_tier3(t), "{t:?} unlocks a yielding structure");
    }
    for t in [TechnologyType::Chivalry, TechnologyType::Navigation] {
        assert!(!is_eco_tier3(t), "{t:?} unlocks no yielding structure");
    }
    // Not tier 3 at all.
    assert!(!is_eco_tier3(TechnologyType::Farming));
}
/// On a dry map the naval lane unlocks nothing, so it is masked at the
/// root — but only the lane that dead-ends, and only when the map is dry.
#[test]
fn water_techs_are_masked_only_on_a_map_without_water() {
    let mut aux = GoalAux::default();
    aux.overlays.knight_commit = true;
    aux.eco_tier3_owned = true;

    let wet = [
        TechnologyType::Fishing,
        TechnologyType::Sailing,
        TechnologyType::Ramming,
        TechnologyType::Aquatism,
        TechnologyType::Navigation,
    ];
    aux.water_dead = false;
    for t in wet {
        assert!(passes_tech_caps(&ResearchMove::new(t), &aux), "{t:?} legal with water");
    }
    aux.water_dead = true;
    for t in wet {
        assert!(!passes_tech_caps(&ResearchMove::new(t), &aux), "{t:?} dead without water");
    }
    // Land techs are untouched, and a non-Research move never sees the gate.
    for t in [TechnologyType::Construction, TechnologyType::Chivalry, TechnologyType::Riding] {
        assert!(passes_tech_caps(&ResearchMove::new(t), &aux), "{t:?} unaffected");
    }
}
/// Aquatism yields population, so the table calls it an economic tier-3 —
/// but a WaterTemple can never be built on a dry map, and letting it pass
/// the economy-first rule would hand the combat lane a free unlock.
#[test]
fn a_water_tier3_does_not_satisfy_the_economy_first_rule_when_dry() {
    use crate::states::TechnologyState;
    let mut state = GameState::default();
    let mut t1 = TribeState::default();
    t1.tech_vanilla.push(TechnologyState {
        tech_type: TechnologyType::Aquatism,
        discovered: true,
        discovered_turn: 3,
    });
    state.tribes.insert(1, t1);
    // No tiles at all -> no water.
    let goal = MacroGoal::default();
    let dry = scripted_goal_aux(&state, 1, &goal, 0, 0, None);
    assert!(dry.water_dead);
    assert!(!dry.eco_tier3_owned, "a dead water temple is not an economy");

    // Same tech, same seat, on a map that has water: it counts.
    let mut wet_tile = TileState::default();
    wet_tile.terrain_type = TerrainType::Water;
    state.tiles.insert(0, wet_tile);
    let wet = scripted_goal_aux(&state, 1, &goal, 0, 0, None);
    assert!(!wet.water_dead);
    assert!(wet.eco_tier3_owned);
}
/// The rider push must judge the ROUTE, not the global terrain census: a
/// forest pocket off the path is irrelevant; a forest corridor on the
/// path erases the 2-tile advantage.
#[test]
fn rider_push_is_path_aware() {
    use crate::types::TerrainType;
    let terrain_tile = |terrain: TerrainType| {
        let mut tile = TileState::default();
        tile.terrain_type = terrain;
        tile.explorers.insert(1);
        tile
    };
    // Unit at (0,0), village at (4,0). A big explored forest pocket in
    // the far corner outnumbers explored fields — the old global census
    // would veto riders; the route doesn't care.
    let mut state = state_with_villages(0, &[44]);
    for r in 8..11 {
        for c in 8..11 {
            state.tiles.insert(r * 11 + c, terrain_tile(TerrainType::Forest));
        }
    }
    let goal = scripted_goal(&state, 1, 0, None);
    assert!(scripted_goal_aux(&state, 1, &goal, 0, 0, None).rider_push);
    assert!(rider_turns_saved(&state, 1, &[44]) >= 2);

    // A thin band is NOT enough: a rider weaves open-step + forest-step
    // (2 tiles/turn, real rider mechanics) and still saves a turn.
    for r in 1..4 {
        for c in 0..3 {
            state.tiles.insert(r * 11 + c, terrain_tile(TerrainType::Forest));
        }
    }
    let goal = scripted_goal(&state, 1, 0, None);
    assert!(scripted_goal_aux(&state, 1, &goal, 0, 0, None).rider_push);

    // Only when the whole approach region is rough does the advantage
    // vanish: forest block rows 0-4 x cols 0-4 (minus start and target).
    // Judged per-target — the aux flag may still fire via guessed sites
    // whose routes run through open unexplored ground (by design).
    for r in 0..5 {
        for c in 0..5 {
            let idx = r * 11 + c;
            if idx != 0 && idx != 44 {
                state.tiles.insert(idx, terrain_tile(TerrainType::Forest));
            }
        }
    }
    assert_eq!(rider_turns_saved(&state, 1, &[44]), 0);
}
#[test]
fn tech_caps_and_rider_push() {
    let mut state = state_with_villages(0, &[3]);
    state.settings.current_player_turn_id = 1;
    // Open fields around the spawn → rider-friendly terrain.
    for idx in 20..30 {
        let mut tile = TileState::default();
        tile.terrain_type = crate::types::TerrainType::Field;
        tile.explorers.insert(1);
        state.tiles.insert(idx, tile);
    }
    let goal = scripted_goal(&state, 1, 0, None); // EXPAND on village 3
    let aux = scripted_goal_aux(&state, 1, &goal, 0, 0, None);
    assert!(aux.rider_push);
    assert_eq!(aux.recommended_techs.first(), Some(&TechnologyType::Riding));

    // Without an EXPAND order there is no rider push.
    let quiet = MacroGoal::default();
    assert!(!scripted_goal_aux(&state, 1, &quiet, 0, 0, None).rider_push);

    // Caps: 8 bought blocks all research; one tier-3 blocks further tier-3.
    let research1 = ResearchMove::new(TechnologyType::Organization);
    let research3 = ResearchMove::new(TechnologyType::Smithery);
    let mut capped = aux.clone();
    capped.techs_bought = TECH_CAP_PER_GAME;
    assert!(!passes_tech_caps(&research1, &capped));
    assert!(passes_tech_caps(&EndTurnMove, &capped));
    // Isolate the tier-3 cap from the lane whitelist: this fixture has no
    // cities, so EXP_ELO_055's territory-scoped recommended_techs recommends
    // nothing of its own here (only the rider-push insert), which would
    // otherwise gate Organization out for lane reasons unrelated to what
    // this assertion is testing.
    let mut t3 = aux.clone();
    t3.tier3_bought = TIER3_CAP_PER_GAME;
    t3.recommended_techs.clear();
    assert!(passes_tech_caps(&research1, &t3));
    assert!(!passes_tech_caps(&research3, &t3));
}
#[test]
fn ability_gate_blocks_destroy_and_resource_clearing() {
    use crate::moves::abilities::{BurnForestMove, ClearForestMove, DestroyMove};
    let mut state = GameState::default();
    assert!(!passes_ability_gate(&state, &DestroyMove::new(5)));
    assert!(passes_ability_gate(&state, &EndTurnMove));
    assert!(passes_ability_gate(&state, &ResearchMove::new(TechnologyType::Organization)));
    // Bare forest may still be cleared — that trade is priced, not masked.
    assert!(passes_ability_gate(&state, &ClearForestMove::new(5)));
    // v8: a forest carrying a resource may not be — clearing DELETES the
    // Game sitting on it for one star.
    state.resources.insert(
        5,
        Some(crate::states::ResourceState { resource_type: crate::types::ResourceType::Game }),
    );
    assert!(!passes_ability_gate(&state, &ClearForestMove::new(5)));
    assert!(!passes_ability_gate(&state, &BurnForestMove::new(5)));
    assert!(passes_ability_gate(&state, &ClearForestMove::new(6)));
}
