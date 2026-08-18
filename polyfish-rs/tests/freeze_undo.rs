//! Regression for `actions::freeze_area`: it used to push its only mutation into the
//! undo chain, so the forward call was a no-op and undoing turned Water into Ice
//! permanently. MCTS relies on undo, so a non-round-tripping undo corrupts the tree.

use polyfish::actions::freeze_area;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, SkillType, TerrainType, TribeType, UnitType};

fn state_hash(game: &Game) -> u32 {
    let bytes = serde_json::to_vec(&game.state).unwrap();
    polyfish::hash::xxhash32(&bytes, 0)
}

fn polaris_game(seed: i64) -> Game {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size: MapSize::Small,
        map_type: MapType::Continents,
        tribes: vec![TribeType::Polaris, TribeType::Imperius],
        seed,
        ..Default::default()
    });
    game.post_load();
    game
}

/// A land tile owned by nobody in particular that has at least `n` adjacent water tiles.
fn land_next_to_water(game: &Game, n: usize) -> Option<i32> {
    let mut candidates: Vec<i32> = game
        .state
        .tiles
        .iter()
        .filter(|(_, t)| !t.is_water_terrain() && t.terrain_type != TerrainType::Mountain)
        .map(|(&idx, _)| idx)
        .collect();
    candidates.sort();
    candidates.into_iter().find(|&idx| {
        polyfish::functions::get_adjacent_indices(&game.state, idx, 1)
            .iter()
            .filter(|&&a| {
                game.state
                    .tiles
                    .get(&a)
                    .map(|t| matches!(t.terrain_type, TerrainType::Water | TerrainType::Ocean))
                    .unwrap_or(false)
            })
            .count()
            >= n
    })
}

#[test]
fn freeze_area_freezes_water_and_undo_restores_it() {
    let mut checked = 0;

    for seed in 0..12i64 {
        let mut game = polaris_game(seed);
        let Some(idx) = land_next_to_water(&game, 1) else {
            continue;
        };
        let pov = game.state.settings.current_player_turn_id;

        let adjacent = polyfish::functions::get_adjacent_indices(&game.state, idx, 1);
        let before_terrain: Vec<TerrainType> = adjacent
            .iter()
            .map(|a| game.state.tiles[a].terrain_type)
            .collect();
        let before_hash = state_hash(&game);

        let undo = freeze_area(&mut game.state, pov, idx);

        let froze = adjacent
            .iter()
            .zip(&before_terrain)
            .any(|(a, old)| {
                matches!(old, TerrainType::Water | TerrainType::Ocean)
                    && game.state.tiles[a].terrain_type == TerrainType::Ice
            });
        assert!(
            froze,
            "seed {seed}: freeze_area at {idx} left every adjacent water unfrozen"
        );

        undo(&mut game.state);

        for (a, old) in adjacent.iter().zip(&before_terrain) {
            assert_eq!(
                game.state.tiles[a].terrain_type, *old,
                "seed {seed}: tile {a} not restored by undo"
            );
        }
        assert_eq!(
            state_hash(&game),
            before_hash,
            "seed {seed}: freeze_area undo did not round-trip the state hash"
        );
        checked += 1;
    }

    assert!(checked > 0, "no seed produced a land tile next to water");
}

/// The FreezeArea unit action, end to end through move generation and `simulate_move`.
#[test]
fn freeze_area_move_undo_round_trips() {
    let mut game = polaris_game(3);
    let idx = land_next_to_water(&game, 1).expect("no land tile next to water");
    let pov = game.state.settings.current_player_turn_id;

    let tribe = game.state.tribes.get_mut(&pov).unwrap();
    let mut gaami = polyfish::states::UnitState::default();
    gaami.owner = pov;
    gaami.unit_type = UnitType::Gaami;
    gaami.health = 30.0;
    gaami.coords.set_at(idx, game.state.settings.size);
    tribe.units.push(gaami);
    game.state.tiles.get_mut(&idx).unwrap()._unit_owner_id = Some(pov);

    assert!(
        polyfish::settings::units::get_unit_setting(UnitType::Gaami)
            .skills
            .contains(&SkillType::FreezeArea)
    );

    let before_hash = state_hash(&game);
    let moves = game.legal_moves();
    let m = moves
        .iter()
        .find(|m| {
            m.ability_type().ok() == Some(polyfish::types::AbilityType::FreezeArea)
                && m.source_idx().ok() == Some(idx as usize)
        })
        .expect("FreezeAreaMove was not generated next to unfrozen water");

    let undo = game.simulate_move(m.as_ref()).expect("simulation refused");
    assert_ne!(
        state_hash(&game),
        before_hash,
        "FreezeAreaMove changed nothing"
    );

    undo(&mut game.state);
    assert_eq!(
        state_hash(&game),
        before_hash,
        "FreezeAreaMove undo did not round-trip the state hash"
    );
}

/// AutoFreeze fires on movement, and its undo round-trips.
#[test]
fn auto_freeze_on_step_undo_round_trips() {
    let mut game = polaris_game(3);
    let dest = land_next_to_water(&game, 1).expect("no land tile next to water");
    let pov = game.state.settings.current_player_turn_id;

    let from = polyfish::functions::get_adjacent_indices(&game.state, dest, 1)
        .into_iter()
        .find(|a| {
            game.state
                .tiles
                .get(a)
                .map(|t| !t.is_water_terrain() && t.terrain_type != TerrainType::Mountain)
                .unwrap_or(false)
                && polyfish::functions::get_unit_at(&game.state, *a).is_none()
        })
        .expect("no free land tile adjacent to the destination");

    let tribe = game.state.tribes.get_mut(&pov).unwrap();
    let mut mooni = polyfish::states::UnitState::default();
    mooni.owner = pov;
    mooni.unit_type = UnitType::Mooni;
    mooni.health = 15.0;
    mooni.coords.set_at(from, game.state.settings.size);
    let unit_idx = tribe.units.len();
    tribe.units.push(mooni);
    game.state.tiles.get_mut(&from).unwrap()._unit_owner_id = Some(pov);

    let before_hash = state_hash(&game);
    let undo = polyfish::actions::units::step_unit(&mut game.state, pov, unit_idx, dest, false);

    let frozen_any = polyfish::functions::get_adjacent_indices(&game.state, dest, 1)
        .iter()
        .any(|a| game.state.tiles[a].terrain_type == TerrainType::Ice);
    assert!(frozen_any, "AutoFreeze did not freeze any adjacent water");

    undo(&mut game.state);
    assert_eq!(
        state_hash(&game),
        before_hash,
        "AutoFreeze step undo did not round-trip the state hash"
    );
}
