use polyfish::actions::city::capture_city;
use polyfish::game::Game;
use polyfish::moves::EndTurnMove;
use polyfish::research::{
    self, EXPERT_BUDGET, MatchConfig, MatchResult, Opponent, OpponentPreset, observation,
};
use polyfish::types::{MapSize, ModeType, ResourceType, TerrainType, TribeType};

fn fixture(size: MapSize) -> Game {
    research::new_match(
        size,
        [TribeType::Imperius, TribeType::Bardur],
        42,
        MatchConfig::new(Opponent::Expert, 2),
    )
}

fn turn(game: &mut Game, preset: OpponentPreset) -> Vec<serde_json::Value> {
    let pov = game.current_player_id();
    let mut actions = Vec::new();
    while game.current_player_id() == pov && research::result(game) == MatchResult::InProgress {
        let action = research::choose_move(game, preset).expect("a legal bot move");
        actions.push(action.serialize());
        let _undo = game.play_move(action.as_ref()).expect("move accepted");
        assert!(actions.len() <= 100, "bot did not finish the turn");
    }
    actions
}

#[test]
fn all_presets_finish_the_same_turn_after_a_save_load() {
    for size in [MapSize::Tiny, MapSize::Small, MapSize::Normal] {
        for preset in [
            OpponentPreset::Easy,
            OpponentPreset::Normal,
            OpponentPreset::Expert,
        ] {
            let mut game = fixture(size);
            let saved = game.to_json().unwrap();
            let mut loaded = Game::from_json(&saved).unwrap();
            assert_eq!(
                loaded.state.settings.research.unwrap().preset_version,
                research::PRESET_VERSION
            );
            assert_eq!(loaded.tile_count(), size.get_size().pow(2));
            assert_eq!(turn(&mut game, preset), turn(&mut loaded, preset));
            assert_eq!(game.current_player_id(), 2);
        }
    }
}

#[test]
fn expert_uses_the_budget_and_ignores_hidden_state() {
    let game = fixture(MapSize::Tiny);
    let mut changed = game.clone();
    let hidden = game
        .state
        .tiles
        .values()
        .find(|tile| !tile.explorers.contains(&1))
        .unwrap()
        .coords
        .idx;
    let visible = game
        .state
        .tiles
        .values()
        .find(|tile| {
            tile.explorers.contains(&1)
                && !game
                    .state
                    .structures
                    .get(&tile.coords.idx)
                    .is_some_and(Option::is_some)
                && !game
                    .state
                    .resources
                    .get(&tile.coords.idx)
                    .is_some_and(Option::is_some)
        })
        .unwrap()
        .coords
        .idx;
    changed.state.tiles[&hidden].terrain_type = TerrainType::Mountain;
    changed.state.tiles[&hidden].has_road = true;
    changed.state.tribes[&2].stars += 1000;
    changed.state.resources.insert(
        visible,
        Some(polyfish::states::ResourceState {
            resource_type: ResourceType::Metal,
        }),
    );
    assert_eq!(
        observation(&game, 1).state.tiles[&hidden].terrain_type,
        TerrainType::Field
    );
    assert!(observation(&changed, 1).state.resources[&visible].is_none());
    for preset in [
        OpponentPreset::Easy,
        OpponentPreset::Normal,
        OpponentPreset::Expert,
    ] {
        let (expected, advances) = research::choose_with_budget(&game, preset, EXPERT_BUDGET);
        let (actual, changed_advances) =
            research::choose_with_budget(&changed, preset, EXPERT_BUDGET);
        assert_eq!(expected.unwrap().serialize(), actual.unwrap().serialize());
        assert_eq!(advances, changed_advances);
        assert_eq!(
            advances,
            if preset == OpponentPreset::Expert {
                EXPERT_BUDGET
            } else {
                0
            }
        );
    }
}

#[test]
fn might_capital_capture_ends_the_match_and_undo_restores_it() {
    let mut game = fixture(MapSize::Tiny);
    let original_capital = game.state.tribes[&2].starting_tile_coords.idx;
    let village = game
        .state
        .structures
        .iter()
        .find_map(|(&idx, structure)| {
            structure
                .as_ref()
                .filter(|s| s.structure_type == polyfish::StructureType::Village)
                .filter(|_| game.state.tiles[&idx].owner == 0)
                .map(|_| idx)
        })
        .unwrap();
    game.state.settings.current_player_turn_id = 2;
    let _found_city = capture_city(&mut game.state, village).unwrap();
    game.state.settings.current_player_turn_id = 1;

    let undo = capture_city(&mut game.state, original_capital).unwrap();
    assert_eq!(game.state.tribes[&2].cities.len(), 1);
    assert_eq!(research::result(&game), MatchResult::Won { winner: 1 });
    assert!(game.legal_moves().is_empty());
    let loaded = Game::from_json(&game.to_json().unwrap()).unwrap();
    assert_eq!(research::result(&loaded), MatchResult::Won { winner: 1 });

    undo(&mut game.state);
    assert_eq!(research::result(&game), MatchResult::InProgress);
    assert_eq!(game.state.tribes[&2].cities.len(), 2);
    game.state.settings.mode = ModeType::Domination;
    let _capture = capture_city(&mut game.state, original_capital).unwrap();
    assert_eq!(research::result(&game), MatchResult::InProgress);
}

#[test]
fn research_turn_limit_is_a_truncation() {
    let mut game = fixture(MapSize::Tiny);
    game.state.settings.max_turns = game.turn();
    let _first = game.play_move(&EndTurnMove).unwrap();
    let _second = game.play_move(&EndTurnMove).unwrap();
    assert_eq!(research::result(&game), MatchResult::Truncated);
    assert_eq!(game.state._history.len(), 2);
}
