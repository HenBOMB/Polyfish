# Polyfish research release candidate 1

## Play locally

From `polyfish-rs/`, start the Rust server and visit [the simulator](http://127.0.0.1:3000/simulator/). A full checkout serves the graphical board from `src/public/`. The research archive uses the compact board from `polyfish-ui/public/research/`. That board is also available at [/simulator/plain/](http://127.0.0.1:3000/simulator/plain/) in a full checkout.

```sh
POLYFISH_LISTEN=127.0.0.1:3000 cargo run --locked --release --no-default-features --bin polyfish
```

Select Drylands, Lakes, Continents, Pangea, Archipelago, or Water World. Each supports the standard boards of 121, 196, 256, 324, 400, and 900 tiles in 1v1 Might with fog of war. Both seats can choose any of the 12 regular tribes. Random uses the match seed with the same four-tribe pool as v1: Xin-Xi, Imperius, Bardur, and Oumaji. Enter a seed to reproduce the matchup and map or leave it blank for a random one.

Select one of your units and a highlighted destination, or choose a move in the action panel. Drag to pan, use the mouse wheel or pinch to zoom, and select **Fit Board** to see the full map. The save and replay controls open a local state or canonical replay. A compatible `model.safetensors` in the server's working directory enables the learned opponent on 121-tile Drylands.

On water maps, Fishing unlocks Ports and shallow-water travel. Move a land unit onto your Port to embark. Sailing unlocks deep-ocean travel and the Scout upgrade. Land on a coast to disembark. Easy, Normal, and Expert use visible shores and objectives to build Ports and plan crossings. The Expert preset still uses a 20-action horizon and a 256-action simulation budget for search.

The 1v1 generator gives Lakes a continuous land border and land routes to neutral villages, puts both Pangea capitals on a central mainland, and puts Continents capitals on separate landmasses. Archipelago settlements have shallow coasts, and Water World has small settlement islands. Saved games and replays carry their original terrain. Starting a fresh water match with a seed from an earlier v1.1 draft uses the corrected generator. [Polytopia map generation reference](https://wiki.polytopia.com/wiki/map-generation)

To generate a model-free replay, run:

```sh
cargo run --locked --release --no-default-features --bin research_match -- \
  --map-type archipelago --map-size 900 --tribe1 kickoo --tribe2 hoodrick --seed 42 \
  --preset1 normal --preset2 expert --replay /tmp/polyfish-match.json
```

For checkpoint training, run `self_play --opponent-preset easy|normal|expert` with your usual self-play options. This mode uses 11×11 Might, alternates the learner's seat, and records samples from learner decisions. Select a preset, checkpoint opponent, or anchor mode per run.

## Reproduce the pilots

Pin [GAIGResearch/Tribes](https://github.com/GAIGResearch/Tribes) to `015a551d1a6be69b5931788fcc418e4c51eb8ea1`. A local JDK and the corresponding [JavaFX Base](https://repo.maven.apache.org/maven2/org/openjfx/javafx-base/22.0.2/) platform jar are required. The paired pilot builds both sources and records elapsed times, legal-action counts, and runtime details.

The archived native match timings in [engine-pilot.json](results/engine-pilot.json) used the Drylands generator at `c038708d`. The current Drylands generator produces different maps from those seeds. [simulator-smoke.jsonl](results/simulator-smoke.jsonl) records the v1 browser and headless match verification.

```sh
python3 research/run_engine_pilot.py /path/to/Tribes /path/to/javafx-base.jar /tmp/engine-pilot.json
```

For the historical tournament, compile the Tribes source using that jar and the repository's `lib` directory on the classpath. From the pinned Tribes checkout, run `Tournament` in a headless JVM with the absolute path to `research/tribes-pilot.json` as its argument. The original tournament runner generates game and agent seeds from the clock. The records under `results/` retain those seeds and the level seeds.

With the research server running in a separate terminal, exercise the browser API, compare all bot turns to the headless matches, and reconstruct each full replay:

```sh
python3 research/verify_surface.py polyfish-rs/target/release/research_match --full-api
python3 research/verify_surface.py polyfish-rs/target/release/research_match --water --full-api
```

See the [literature and asset audit](AUDIT.md) and [paper draft](PAPER.md). The versioned source archive is generated from a committed tree:

```sh
python3 research/package_release.py /tmp/polyfish-research-v1-rc.1.zip
```

## Rust API

Use the typed action list for policies that inspect the full board. For a fog-of-war
policy, pass the observation and the filtered move list to the policy instead.

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
