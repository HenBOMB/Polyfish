use polyfish::research::{self, MatchConfig, Opponent};
use polyfish::{Action, ActionError, Coords, Game, MapSize, TribeType, TurnRotation};

fn match_game() -> Game {
    research::new_match(
        MapSize::Tiny,
        [TribeType::Imperius, TribeType::Bardur],
        42,
        MatchConfig::new(Opponent::Human, 2),
    )
}

fn step(game: &Game) -> Action {
    game.legal_actions()
        .into_iter()
        .find(|action| matches!(action, Action::Step { .. }))
        .expect("opening step")
}

fn assert_rejected(game: &mut Game, action: &Action, error: ActionError) {
    let before = serde_json::to_value(&game.state).unwrap();
    let simulated_discovery = game.state.settings._sim_explored.clone();

    assert_eq!(game.play(action).err(), Some(error), "{action:?}");
    assert_eq!(serde_json::to_value(&game.state).unwrap(), before);
    for rotation in [TurnRotation::NextPlayer, TurnRotation::SamePlayer] {
        assert_eq!(game.simulate(action, rotation).err(), Some(error));
        assert_eq!(serde_json::to_value(&game.state).unwrap(), before);
        assert_eq!(game.state.settings._sim_explored, simulated_discovery);
    }
}

#[test]
fn hand_built_and_stale_steps_leave_the_game_unchanged() {
    let mut game = match_game();
    let action = step(&game);
    let Action::Step { source, target } = action else {
        unreachable!()
    };
    let tiles = game.map_size().pow(2);
    let opposite_edge = if source < tiles / 2 { tiles - 1 } else { 0 };
    let wrong_player = game.state.tribes[&2].units[0].coords.idx;
    for (source, target) in [
        (source, opposite_edge),
        (source, tiles),
        (source, -1),
        (wrong_player, target),
    ] {
        assert_rejected(
            &mut game,
            &Action::Step { source, target },
            ActionError::IllegalAction,
        );
    }

    let mut occupied = game.clone();
    let mut ally = occupied.state.tribes[&1].units[0].clone();
    ally.coords = Coords::from_index(target, occupied.map_size());
    occupied.state.tribes.get_mut(&1).unwrap().units.push(ally);
    assert_rejected(&mut occupied, &action, ActionError::IllegalAction);

    let _undo = game.play(&action).unwrap();
    assert_rejected(&mut game, &action, ActionError::IllegalAction);
    let reverse = Action::Step {
        source: target,
        target: source,
    };
    assert_rejected(&mut game, &reverse, ActionError::IllegalAction);
}

#[test]
fn mandatory_rewards_and_game_over_are_respected() {
    let mut game = match_game();
    let action = step(&game);
    game.state.tribes.get_mut(&1).unwrap().cities[0].level = 2;
    assert!(
        game.legal_actions()
            .iter()
            .all(|action| matches!(action, Action::Reward { .. }))
    );
    assert_rejected(&mut game, &action, ActionError::IllegalAction);
    assert_rejected(&mut game, &Action::EndTurn, ActionError::IllegalAction);

    let mut hinted = game.legal_actions().into_iter().next().unwrap();
    if let Action::Reward { revealed_tiles, .. } = &mut hinted {
        *revealed_tiles = Some(vec![0]);
    }
    assert_rejected(&mut game, &hinted, ActionError::IllegalAction);

    game.state.settings._game_over = true;
    assert!(game.legal_actions().is_empty());
    assert_rejected(&mut game, &Action::EndTurn, ActionError::GameOver);
}

#[test]
fn simulation_rotation_is_per_call() {
    let mut game = match_game();
    let before = serde_json::to_value(&game.state).unwrap();
    let first = game.current_player_id();
    assert_eq!(
        game.decision()
            .simulate_index(usize::MAX, TurnRotation::NextPlayer)
            .err(),
        Some(ActionError::IllegalAction)
    );
    assert_eq!(serde_json::to_value(&game.state).unwrap(), before);
    let undo = game
        .simulate(&Action::EndTurn, TurnRotation::NextPlayer)
        .unwrap();
    assert_ne!(game.current_player_id(), first);
    undo(&mut game.state);
    assert_eq!(serde_json::to_value(&game.state).unwrap(), before);

    let undo = game
        .simulate(&Action::EndTurn, TurnRotation::SamePlayer)
        .unwrap();
    assert_eq!(game.current_player_id(), first);
    undo(&mut game.state);
    assert_eq!(serde_json::to_value(&game.state).unwrap(), before);
}

#[test]
fn committed_undo_keeps_fog_discoveries() {
    let mut game = match_game();
    let source = game.state.tribes[&1].units[0].coords.idx;
    let size = game.map_size();
    for tile in game.state.tiles.values_mut() {
        if polyfish::functions::get_chebyshev_distance(source, tile.coords.idx, size) > 1 {
            tile.explorers.remove(&1);
        }
    }
    let explored = |game: &Game| {
        game.state
            .tiles
            .values()
            .filter(|tile| tile.explorers.contains(&1))
            .count()
    };
    let before = explored(&game);
    for action in game
        .legal_actions()
        .into_iter()
        .filter(|action| matches!(action, Action::Step { .. }))
    {
        let mut committed = game.clone();
        let undo = committed.play(&action).unwrap();
        if explored(&committed) == before {
            continue;
        }
        undo(&mut committed.state);
        assert!(explored(&committed) > before);

        let mut simulated = game.clone();
        let undo = simulated
            .simulate(&action, TurnRotation::NextPlayer)
            .unwrap();
        undo(&mut simulated.state);
        assert_eq!(explored(&simulated), before);
        return;
    }
    panic!("opening fixture has no exploratory move");
}
