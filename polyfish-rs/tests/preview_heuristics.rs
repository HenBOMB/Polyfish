use polyfish::actions::units::{remove_unit, summon_unit};
use polyfish::ai::evaluator::army::assess_unit_power;
use polyfish::states::{GameState, TileState, TribeState};
use polyfish::types::{TerrainType, TribeType};

fn setup_basic_state() -> GameState {
    let mut state = GameState::default();
    state.settings.size = 11;
    let tribe_id = 1;
    state.settings.current_player_turn_id = tribe_id;

    let mut tribe = TribeState::default();
    tribe.id = tribe_id;
    tribe.tribe_type = TribeType::Imperius;
    state.tribes.insert(tribe_id, tribe);

    // Fill map with fields
    for i in 0..121 {
        let mut tile = TileState::default();
        tile.terrain_type = TerrainType::Field;
        state.tiles.insert(i, tile);
    }

    state
}

fn evaluate_unit(state: &mut GameState, unit_type: polyfish::UnitType) -> f32 {
    let tribe_id = 1;
    let tile_idx = 0; // Use tile 0 for testing

    // Ensure tile is empty/valid
    if let Some(tile) = state.tiles.get_mut(&tile_idx) {
        tile._unit_owner_id = None;
    }

    // Summon unit for tribe 1
    let _ = summon_unit(state, unit_type, tile_idx, false, false);

    // Get the unit (it should be the last one added)
    let unit_score = if let Some(tribe) = state.tribes.get(&tribe_id) {
        if let Some(unit) = tribe.units.last() {
            assess_unit_power(state, unit)
        } else {
            0.0
        }
    } else {
        0.0
    };

    // Clean up: Remove the unit
    if let Some(tribe) = state.tribes.get(&tribe_id) {
        let unit_idx = tribe.units.len() - 1;
        let _ = remove_unit(state, tribe_id, unit_idx, None, None);
    }

    unit_score
}

#[test]
fn preview_heuristics() {
    let mut state = setup_basic_state();

    println!(
        "Warrior: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Warrior)
    );
    println!(
        "Archer: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Archer)
    );
    println!(
        "Amphibian: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Amphibian)
    );
    println!(
        "BattleSled: {}",
        evaluate_unit(&mut state, polyfish::UnitType::BattleSled)
    );
    println!(
        "Bomber: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Bomber)
    );
    println!(
        "Boomchi: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Boomchi)
    );
    println!(
        "Catapult: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Catapult)
    );
    println!(
        "BabyDragon: {}",
        evaluate_unit(&mut state, polyfish::UnitType::BabyDragon)
    );
    println!(
        "Centipede: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Centipede)
    );
    println!(
        "Cloak: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Cloak)
    );
    println!(
        "Crab: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Crab)
    );
    println!(
        "Dagger: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Dagger)
    );
    println!(
        "Defender: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Defender)
    );
    println!(
        "Dinghy: {}",
        evaluate_unit(&mut state, polyfish::UnitType::CloakBoat)
    );
    println!(
        "Doomux: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Doomux)
    );
    println!(
        "DragonEgg: {}",
        evaluate_unit(&mut state, polyfish::UnitType::DragonEgg)
    );
    println!(
        "Exida: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Exida)
    );
    println!(
        "FireDragon: {}",
        evaluate_unit(&mut state, polyfish::UnitType::FireDragon)
    );
    println!(
        "Gaami: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Gaami)
    );
    println!(
        "Giant: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Giant)
    );
    println!(
        "Hexapod: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Hexapod)
    );
    println!(
        "IceArcher: {}",
        evaluate_unit(&mut state, polyfish::UnitType::IceArcher)
    );
    println!(
        "IceFortress: {}",
        evaluate_unit(&mut state, polyfish::UnitType::IceFortress)
    );
    println!(
        "InsectEgg: {}",
        evaluate_unit(&mut state, polyfish::UnitType::InsectEgg)
    );
    println!(
        "Juggernaut: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Juggernaut)
    );
    println!(
        "Kiton: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Kiton)
    );
    println!(
        "Knight: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Knight)
    );
    println!(
        "Larva: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Larva)
    );
    println!(
        "LivingIsland: {}",
        evaluate_unit(&mut state, polyfish::UnitType::LivingIsland)
    );
    println!(
        "Mantis: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Mantis)
    );
    println!(
        "MindBender: {}",
        evaluate_unit(&mut state, polyfish::UnitType::MindBender)
    );
    println!(
        "Mooni: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Mooni)
    );
    println!(
        "Moth: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Moth)
    );
    println!(
        "Phychi: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Phychi)
    );
    println!(
        "Pirate: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Pirate)
    );
    println!(
        "Polytaur: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Polytaur)
    );
    println!(
        "Raft: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Transportship)
    );
    println!(
        "Rammer: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Rammership)
    );
    println!(
        "Raychi: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Raychi)
    );
    println!(
        "Rider: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Rider)
    );
    println!(
        "Scout: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Scoutship)
    );
    println!(
        "Segment: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Segment)
    );
    println!(
        "Swordsman: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Swordsman)
    );
    println!(
        "Shaman: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Shaman)
    );
    println!(
        "Tridention: {}",
        evaluate_unit(&mut state, polyfish::UnitType::Tridention)
    );
}
