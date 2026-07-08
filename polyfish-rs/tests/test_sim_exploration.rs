//! Exploration score during MCTS simulations: simulated reveals must credit
//! +5/tile exactly once per simulation line (via `_sim_explored`) WITHOUT
//! marking `tile.explorers` — the search learns THAT a direction reveals,
//! never WHAT is under the fog. See `actions::discovery::discover_tiles`.

use polyfish::actions::discovery::discover_tiles;
use polyfish::actions::try_discover_other_tribes;
use polyfish::game::Game;
use polyfish::mapgen::{MapGenSettings, generate};
use polyfish::types::{MapSize, MapType, TribeType};

fn make_game(seed: i64) -> Game {
    let mut game = Game::new();
    let gen_settings = MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    };
    game.state = generate(gen_settings);
    game.post_load();
    game
}

// Fog tiles with no unit or structure, so really-revealing them can't
// trigger side rewards (tribe discovery, lighthouse) that would shift score.
fn unexplored_tiles(game: &Game, pov: i32, n: usize) -> Vec<i32> {
    game.state
        .tiles
        .iter()
        .filter(|(idx, t)| {
            !t.explorers.contains(&pov)
                && t._unit_owner_id.is_none()
                && game
                    .state
                    .structures
                    .get(*idx)
                    .map_or(true, |s| s.is_none())
        })
        .map(|(idx, _)| *idx)
        .take(n)
        .collect()
}

#[test]
fn sim_discovery_credits_once_without_revealing() {
    let mut game = make_game(7);
    let pov = 1;

    // Flush any pending tribe discovery so the score baseline is stable.
    let _ = try_discover_other_tribes(&mut game.state);

    let tiles = unexplored_tiles(&game, pov, 4);
    assert!(!tiles.is_empty(), "map should have fog for player 1");
    let score0 = game.state.tribes.get(&pov).unwrap().score;

    // _are_you_sure is false outside play_move => simulation semantics.
    assert!(!game.state.settings._are_you_sure);
    let undo1 = discover_tiles(&mut game.state, pov, None, Some(tiles.clone()));

    let gained = game.state.tribes.get(&pov).unwrap().score - score0;
    assert_eq!(
        gained,
        5 * tiles.len() as i32,
        "sim reveal should credit +5 per fog tile"
    );
    for &idx in &tiles {
        assert!(
            !game.state.tiles.get(&idx).unwrap().explorers.contains(&pov),
            "sim reveal must NOT mark tile {idx} explored"
        );
    }

    // Second contact with the same tiles in the same sim line: no re-credit.
    let undo2 = discover_tiles(&mut game.state, pov, None, Some(tiles.clone()));
    assert_eq!(
        game.state.tribes.get(&pov).unwrap().score - score0,
        gained,
        "same sim line must not re-credit already-counted tiles"
    );

    // Unwinding restores score and empties the shadow set.
    undo2(&mut game.state);
    undo1(&mut game.state);
    assert_eq!(game.state.tribes.get(&pov).unwrap().score, score0);
    assert!(
        game.state
            .settings
            ._sim_explored
            .get(&pov)
            .map_or(true, |s| s.is_empty()),
        "shadow set must be empty after full unwind"
    );

    // After unwind, a fresh sim line credits again (determinism).
    let undo3 = discover_tiles(&mut game.state, pov, None, Some(tiles.clone()));
    assert_eq!(game.state.tribes.get(&pov).unwrap().score - score0, gained);
    undo3(&mut game.state);
}

#[test]
fn real_move_discovery_still_reveals() {
    let mut game = make_game(7);
    let pov = 1;
    let _ = try_discover_other_tribes(&mut game.state);

    let tiles = unexplored_tiles(&game, pov, 3);
    assert!(!tiles.is_empty());
    let score0 = game.state.tribes.get(&pov).unwrap().score;

    game.state.settings._are_you_sure = true;
    let _undo = discover_tiles(&mut game.state, pov, None, Some(tiles.clone()));
    game.state.settings._are_you_sure = false;

    assert_eq!(
        game.state.tribes.get(&pov).unwrap().score - score0,
        5 * tiles.len() as i32
    );
    for &idx in &tiles {
        assert!(
            game.state.tiles.get(&idx).unwrap().explorers.contains(&pov),
            "real reveal must mark tile {idx} explored"
        );
    }
    assert!(
        game.state
            .settings
            ._sim_explored
            .get(&pov)
            .map_or(true, |s| s.is_empty()),
        "real moves must not touch the sim shadow set"
    );
}
