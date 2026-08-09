//! Guards for the rules that were duplicated and drifted (Aug 2026).
//!
//! Each of these covers a divergence that shipped silently because the rule was
//! implemented in more than one place. They fail if a copy reappears.

use polyfish::game::Game;
use polyfish::rules;
use polyfish::states::{CityState, GameState, StructureState, TileState, TribeState, UnitState};
use polyfish::types::{ResourceType, StructureType, TechnologyType, TerrainType, TribeType, UnitType};

fn board() -> GameState {
    let mut state = GameState::default();
    state.settings.size = 11;
    state.settings.current_player_turn_id = 1;
    state.settings.version = 115;
    for idx in 0..121 {
        let mut t = TileState::default();
        t.owner = 1;
        state.tiles.insert(idx, t);
    }
    state
}

/// The resource actually stored on the tile. `get_resource_at` is filtered by
/// tech visibility, so it cannot tell "crushed" from "not yet discovered".
fn raw_resource(state: &GameState, idx: i32) -> Option<ResourceType> {
    state
        .resources
        .get(&idx)
        .and_then(|r| r.as_ref())
        .map(|r| r.resource_type)
}

fn unit(kind: UnitType, idx: i32, owner: i32) -> UnitState {
    let mut u = UnitState {
        unit_type: kind,
        coords: polyfish::Coords::from_index(idx, 11),
        ..Default::default()
    };
    u.owner = owner;
    u.health = polyfish::settings::units::get_unit_setting(kind).health;
    u
}

/// A ranged attacker takes no retaliation. `calculate_combat_preview` used to
/// report the counter anyway, and `ai::scoring` then priced ranged attacks as
/// suicide (1.0 instead of 50-95).
#[test]
fn ranged_attacker_is_not_predicted_to_take_retaliation() {
    let state = board();
    let archer = unit(UnitType::Archer, 60, 1);
    let warrior = unit(UnitType::Warrior, 62, 2); // distance 2, Warrior range 1
    assert!(
        !rules::combat::can_retaliate(&state, &archer, &warrior),
        "a range-2 attack is outside the defender's reach"
    );

    let adjacent_warrior = unit(UnitType::Warrior, 61, 1);
    assert!(
        rules::combat::can_retaliate(&state, &adjacent_warrior, &warrior),
        "an adjacent melee attack must still draw a counter"
    );
}

/// A fresh city rules its 3x3 — nine tiles, its own included. `post_load` used
/// to rebuild territory as a fixed radius 2 with the centre dropped, silently
/// reshaping every city on load.
#[test]
fn a_fresh_city_rules_nine_tiles_including_its_own() {
    let mut state = board();
    let mut tribe = TribeState::default();
    tribe.id = 1;
    tribe.cities.push(CityState {
        idx: 60,
        owner: 1,
        level: 1,
        border_size: 1,
        ..Default::default()
    });
    state.tribes.insert(1, tribe);

    let mut game = Game::new();
    game.state = state;
    game.post_load();

    let city = &game.state.tribes.get(&1).unwrap().cities[0];
    assert_eq!(city._territory.len(), 9, "3x3 square");
    assert!(city._territory.contains(&60), "the city tile is its own");
}

/// Hub level is the count of friendly adjacent partners, and it is
/// PLAYER-scoped — a partner across a city border still feeds it.
#[test]
fn partner_count_is_player_scoped_not_city_scoped() {
    let mut state = board();
    let put = |s: &mut GameState, idx: i32, k: StructureType| {
        s.structures.insert(
            idx,
            Some(StructureState {
                structure_type: k,
                level: 1,
                founded: 0,
            }),
        );
    };
    put(&mut state, 60, StructureType::Sawmill);
    for hut in [48, 49, 50] {
        put(&mut state, hut, StructureType::LumberHut);
    }
    assert_eq!(
        rules::economy::partner_count(&state, 60, StructureType::Sawmill, 1),
        3
    );

    // An enemy-owned partner does not count.
    state.tiles.get_mut(&49).unwrap().owner = 2;
    assert_eq!(
        rules::economy::partner_count(&state, 60, StructureType::Sawmill, 1),
        2
    );
}

/// Levels cost L+1 pop each, so 9 pop reaches level 4 and super units start at 5.
#[test]
fn level_thresholds_and_super_unit_slots_agree_with_the_engine() {
    assert_eq!(rules::economy::pop_to_reach(1, 4), 9);
    assert_eq!(rules::economy::pop_to_reach(1, 5), 14);
    assert_eq!(rules::economy::level_at_pop(8), 3);
    assert_eq!(rules::economy::level_at_pop(9), 4);
    assert_eq!(rules::economy::super_units_at_level(4), 0);
    assert_eq!(rules::economy::super_units_at_level(5), 1);
    assert_eq!(rules::economy::super_units_at_level(7), 3);
}

/// Every tribe's super unit, not just Imperius' Giant. Both the summon heuristic
/// and the `giants_made` metric used to test `== UnitType::Giant`.
#[test]
fn each_tribe_has_its_own_super_unit() {
    use polyfish::settings::units::get_super_unit;
    assert_eq!(get_super_unit(TribeType::Imperius), UnitType::Giant);
    assert_eq!(get_super_unit(TribeType::Polaris), UnitType::Gaami);
    assert_ne!(get_super_unit(TribeType::Aquarion), UnitType::Giant);
    assert_ne!(get_super_unit(TribeType::Cymanti), UnitType::Giant);
}

/// A unit's worth is its cost plus its passenger's, and zero once converted —
/// ten AI sites summed the bare cost instead.
#[test]
fn unit_worth_counts_passengers_and_ignores_converted() {
    let plain = unit(UnitType::Warrior, 0, 1);
    let base = polyfish::settings::units::get_unit_setting(UnitType::Warrior).cost;
    assert_eq!(rules::combat::unit_worth(&plain), base);

    let mut loaded = unit(UnitType::Raft, 1, 1);
    loaded.passenger_type = Some(UnitType::Warrior);
    assert_eq!(
        rules::combat::unit_worth(&loaded),
        polyfish::settings::units::get_unit_setting(UnitType::Raft).cost + base,
        "a carrier is worth its passenger too"
    );

    let mut taken = plain.clone();
    taken.converted = true;
    assert_eq!(rules::combat::unit_worth(&taken), 0);
}

/// Resource visibility reads the settings table. `visible_required` was declared
/// and populated but never read, with a different rule hardcoded in functions.rs.
#[test]
fn resource_visibility_comes_from_the_settings_table() {
    use polyfish::functions::is_resource_visible_to_tribe;
    let mut state = board();
    let mut tribe = TribeState::default();
    tribe.id = 1;
    state.tribes.insert(1, tribe);

    let learn = |state: &mut polyfish::states::GameState, tech| {
        state
            .tribes
            .get_mut(&1)
            .unwrap()
            .tech_vanilla
            .push(polyfish::states::TechnologyState {
                tech_type: tech,
                discovered: true,
                discovered_turn: 0,
            });
    };

    // On the map from turn 1 — only Organization, Climbing and Navigation
    // reveal anything in the real game.
    for always in [
        ResourceType::Fruit,
        ResourceType::Game,
        ResourceType::Fish,
        ResourceType::AquaCrop,
    ] {
        assert!(
            is_resource_visible_to_tribe(&state, always, 1, None),
            "{always:?} needs no tech to see"
        );
    }

    // Starfish waits for Navigation specifically, not the rest of its branch.
    assert!(!is_resource_visible_to_tribe(&state, ResourceType::Starfish, 1, None));
    learn(&mut state, TechnologyType::Fishing);
    learn(&mut state, TechnologyType::Sailing);
    assert!(
        !is_resource_visible_to_tribe(&state, ResourceType::Starfish, 1, None),
        "Fishing and Sailing must not reveal starfish"
    );
    learn(&mut state, TechnologyType::Navigation);
    assert!(is_resource_visible_to_tribe(&state, ResourceType::Starfish, 1, None));

    // Any one of the listed techs suffices for Crop.
    assert!(!is_resource_visible_to_tribe(&state, ResourceType::Crop, 1, None));
    learn(&mut state, TechnologyType::Construction);
    assert!(is_resource_visible_to_tribe(&state, ResourceType::Crop, 1, None));
}

/// Replacement techs inherit the tier of the tech they replace — reading
/// `.tier.unwrap_or(1)` priced all 13 of them as tier 1.
#[test]
fn replacement_techs_price_at_the_tier_they_replace() {
    use polyfish::settings::technology::{get_technology_setting, tech_tier};
    use strum::IntoEnumIterator;

    for tech in TechnologyType::iter() {
        let Some(vanilla) = get_technology_setting(tech).replaces_tech else {
            continue;
        };
        assert_eq!(
            tech_tier(tech),
            tech_tier(vanilla),
            "{tech:?} replaces {vanilla:?} and must cost the same tier"
        );
    }
}

/// Terrain is irrelevant to the rule above — this pins that the table drives it,
/// so adding a resource without visibility techs stays visible by default.
#[test]
fn spores_need_no_tech_to_see() {
    let mut state = board();
    state.tiles.get_mut(&60).unwrap().terrain_type = TerrainType::Field;
    let mut tribe = TribeState::default();
    tribe.id = 1;
    state.tribes.insert(1, tribe);
    assert!(polyfish::functions::is_resource_visible_to_tribe(
        &state,
        ResourceType::Spores,
        1,
        None
    ));
}

/// Build legality never looks at resources, so a Market may be sited on an
/// undeveloped Crop or Fruit field — and building there CRUSHES it.
/// `build_structure` used to leave the resource standing under the new
/// structure, where it kept feeding the feature planes and the map render.
#[test]
fn building_over_an_undeveloped_resource_crushes_it() {
    let mut state = board();
    state.tiles.get_mut(&60).unwrap().terrain_type = TerrainType::Field;
    state.resources.insert(
        60,
        Some(polyfish::states::ResourceState {
            resource_type: ResourceType::Crop,
        }),
    );
    let mut tribe = TribeState::default();
    tribe.id = 1;
    tribe.stars = 50;
    state.tribes.insert(1, tribe);

    let undo = polyfish::actions::structure::build_structure(
        &mut state,
        60,
        StructureType::Market,
    );
    assert!(
        raw_resource(&state, 60).is_none(),
        "the Market crushed the Crop it was built over"
    );

    undo(&mut state);
    assert_eq!(
        raw_resource(&state, 60),
        Some(ResourceType::Crop),
        "and undo puts it back"
    );
}

/// The structure that WORKS a resource keeps it: a Farm stands on its Crop, a
/// Mine on its Metal. `StructureSetting.resource_type` reads that pairing the
/// other way round, so the two tables are held to each other here.
#[test]
fn the_structure_that_works_a_resource_does_not_crush_it() {
    use polyfish::settings::structures::get_structure_setting;
    use strum::IntoEnumIterator;

    for resource in ResourceType::iter() {
        let Some(worker) = rules::economy::worker_structure(resource) else {
            continue;
        };
        assert!(
            !rules::economy::build_consumes_resource(worker, resource),
            "{worker:?} works {resource:?} rather than crushing it"
        );
        assert_eq!(
            get_structure_setting(worker).resource_type,
            Some(resource),
            "settings/resources.rs and settings/structures.rs disagree about \
             which structure works {resource:?}"
        );
        assert!(
            rules::economy::build_consumes_resource(StructureType::Market, resource),
            "any other structure crushes {resource:?}"
        );
    }

    let mut state = board();
    state.tiles.get_mut(&60).unwrap().terrain_type = TerrainType::Field;
    state.resources.insert(
        60,
        Some(polyfish::states::ResourceState {
            resource_type: ResourceType::Crop,
        }),
    );
    let mut tribe = TribeState::default();
    tribe.id = 1;
    tribe.stars = 50;
    state.tribes.insert(1, tribe);

    polyfish::actions::structure::build_structure(&mut state, 60, StructureType::Farm);
    assert_eq!(
        raw_resource(&state, 60),
        Some(ResourceType::Crop),
        "the Farm keeps the Crop it works"
    );
}

/// A Road shares its tile rather than claiming it, which is why it is the one
/// structure allowed onto an occupied tile — and why it crushes nothing.
#[test]
fn a_road_claims_nothing_and_crushes_nothing() {
    assert!(!rules::economy::occupies_tile(StructureType::Road));
    assert!(rules::economy::occupies_tile(StructureType::Market));
    for resource in [ResourceType::Fruit, ResourceType::Crop, ResourceType::Metal] {
        assert!(!rules::economy::build_consumes_resource(
            StructureType::Road,
            resource
        ));
    }
}

/// A hub's partners arrive turns after its tile is chosen, so `partner_count`
/// cannot inform placement — `partner_ceiling` is what the site could ever
/// collect. Terrain rule, unoccupied, and resource-worked partners (Farm on
/// Crop) only where the resource stands.
#[test]
fn partner_ceiling_sees_the_sites_future_not_its_present() {
    let mut state = board();
    let put = |s: &mut GameState, idx: i32, k: StructureType| {
        s.structures.insert(
            idx,
            Some(StructureState { structure_type: k, level: 1, founded: 0 }),
        );
    };
    // Sawmill on 60. Neighbours 48,49,50,59,61,70,71,72.
    for idx in [48, 49, 50, 59, 61, 70, 71, 72] {
        state.tiles.get_mut(&idx).unwrap().terrain_type = TerrainType::Forest;
    }
    put(&mut state, 60, StructureType::Sawmill);

    assert_eq!(
        rules::economy::partner_count(&state, 60, StructureType::Sawmill, 1),
        0,
        "nothing is standing yet"
    );
    assert_eq!(
        rules::economy::partner_ceiling(&state, 60, StructureType::Sawmill, 1),
        8,
        "every adjacent forest could take a LumberHut"
    );

    // A realized partner stays inside the ceiling rather than adding to it.
    put(&mut state, 48, StructureType::LumberHut);
    assert_eq!(rules::economy::partner_count(&state, 60, StructureType::Sawmill, 1), 1);
    assert_eq!(rules::economy::partner_ceiling(&state, 60, StructureType::Sawmill, 1), 8);

    // Wrong terrain, someone else's tile, and an occupied tile all drop out.
    state.tiles.get_mut(&49).unwrap().terrain_type = TerrainType::Field;
    state.tiles.get_mut(&50).unwrap().owner = 2;
    put(&mut state, 59, StructureType::Temple);
    assert_eq!(
        rules::economy::partner_ceiling(&state, 60, StructureType::Sawmill, 1),
        5,
        "8 less a field, an enemy tile and an occupied one"
    );
}

/// A Farm works a Crop, so a Windmill's ceiling is bounded by the crops on the
/// map — not by how many fields happen to sit next to it.
#[test]
fn partner_ceiling_respects_resource_worked_partners() {
    let mut state = board();
    for idx in [48, 49, 50, 59, 61, 70, 71, 72] {
        state.tiles.get_mut(&idx).unwrap().terrain_type = TerrainType::Field;
    }
    state.structures.insert(
        60,
        Some(StructureState {
            structure_type: StructureType::Windmill,
            level: 1,
            founded: 0,
        }),
    );
    assert_eq!(
        rules::economy::partner_ceiling(&state, 60, StructureType::Windmill, 1),
        0,
        "eight empty fields and no crop: a Farm cannot stand on any of them"
    );

    for idx in [48, 61] {
        state.resources.insert(
            idx,
            Some(polyfish::states::ResourceState {
                resource_type: ResourceType::Crop,
            }),
        );
    }
    assert_eq!(
        rules::economy::partner_ceiling(&state, 60, StructureType::Windmill, 1),
        2,
        "only the crops count"
    );
}
