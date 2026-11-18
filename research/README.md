# Polyfish research release candidate 1

## Play locally

From `polyfish-rs/`, start the Rust server and visit [the simulator](http://127.0.0.1:3000/simulator/). The browser uses the files in `polyfish-ui/public/research/`.

```sh
POLYFISH_LISTEN=127.0.0.1:3000 cargo run --locked --release --no-default-features --bin polyfish
```

Select a board size, tribes, and opponent. Enter a seed to reproduce the map or leave it blank for a random one. Select one of your units and a highlighted destination, or choose a move in the action panel. Use the save and replay controls to open a local state or canonical replay. A compatible `model.safetensors` in the server's working directory enables the learned opponent on an 11×11 board.

To generate a model-free replay, run:

```sh
cargo run --locked --release --no-default-features --bin research_match -- \
  --map-size 196 --tribe1 imperius --tribe2 bardur --seed 42 \
  --preset1 normal --preset2 expert --replay /tmp/polyfish-match.json
```

For checkpoint training, run `self_play --opponent-preset easy|normal|expert` with your usual self-play options. This mode uses 11×11 Might, alternates the learner's seat, and records samples from learner decisions. Select a preset, checkpoint opponent, or anchor mode per run.

## Reproduce the pilots

Pin [GAIGResearch/Tribes](https://github.com/GAIGResearch/Tribes) to `015a551d1a6be69b5931788fcc418e4c51eb8ea1`. A local JDK and the corresponding [JavaFX Base](https://repo.maven.apache.org/maven2/org/openjfx/javafx-base/22.0.2/) platform jar are required. The paired pilot builds both sources and records elapsed times, legal-action counts, and runtime details.

```sh
python3 research/run_engine_pilot.py /path/to/Tribes /path/to/javafx-base.jar /tmp/engine-pilot.json
```

For the historical tournament, compile the Tribes source using that jar and the repository's `lib` directory on the classpath. From the pinned Tribes checkout, run `Tournament` in a headless JVM with the absolute path to `research/tribes-pilot.json` as its argument. The original tournament runner generates game and agent seeds from the clock. The records under `results/` retain those seeds and the level seeds.

With the research server running in a separate terminal, exercise the browser API, compare all bot turns to the headless matches, and reconstruct each full replay:

```sh
python3 research/verify_surface.py polyfish-rs/target/release/research_match --full-api
```

See the [literature and asset audit](AUDIT.md) and [paper draft](PAPER.md). The versioned source archive is generated from a committed tree:

```sh
python3 research/package_release.py /tmp/polyfish-research-v1-rc.1.zip
```
