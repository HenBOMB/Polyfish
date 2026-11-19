# Polyfish Rust Simulator API

Use the typed action list for policies that inspect the full board. For a fog-of-war
policy, pass the observation and the filtered move list to the policy instead.
The observation reports `TerrainType::Fog` for unexplored tiles and the actual
terrain for explored tiles. When adapting an observation for Java Tribes, map
`TerrainType::Fog` to Java's `FOG` tile value.

```rust
use polyfish::research::{self, MatchConfig, Opponent};
use polyfish::{Action, Game, MapSize, TribeType};

fn advance_match() -> Result<Game, polyfish::ActionError> {
    let mut game = research::new_match(
        MapSize::Tiny,
        [TribeType::Imperius, TribeType::Bardur],
        42,
        MatchConfig::new(Opponent::Human, 2),
    );
    let observation = research::observation(&game, game.current_player_id());
    let available = research::legal_actions(&game, &observation);
    let action = Action::from_move(available[0].as_ref())
        .expect("generated moves have typed identities");
    let _undo = game.play(&action)?;
    Ok(game)
}
```

`Game::play` checks the action against the current authoritative legal list.
If a policy already needs that list, use `Game::decision()` to generate it once
and apply a selection while holding the same game borrow.

`Game::simulate` also checks legality and takes an explicit `TurnRotation`.
`NextPlayer` advances one seat on EndTurn and `SamePlayer` advances back to the
mover where the game permits it. Undo callbacks apply to the same game state in
reverse order. Simulation undo reverses simulated discoveries. Committed undo
keeps newly revealed tiles visible. Replay seeks rebuild from the saved start.
