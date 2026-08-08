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
