# Polyfish research release candidate 1

[Polyfish Rust Simulator API](simulator-api.md)

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

## Benchmark dependencies

Pin [GAIGResearch/Tribes](https://github.com/GAIGResearch/Tribes) to `015a551d1a6be69b5931788fcc418e4c51eb8ea1`. A local JDK and the corresponding [JavaFX Base](https://repo.maven.apache.org/maven2/org/openjfx/javafx-base/22.0.2/) platform jar are required.

## Reproduce the v1.1 engine benchmark

Use the pinned Tribes checkout and JavaFX jar above. Run the benchmark from the repository root on a quiet host with BSD `time` (macOS) or GNU `time` (Linux).

```sh
python3 research/run_engine_benchmark.py /path/to/Tribes /path/to/javafx-base.jar /tmp/engine-benchmark.json
```

Each engine runs a flat opening with two matching capitals and one warrior per seat. The benchmark runs seven fresh process pairs per workload on each of six board sizes and interleaves the engines and sizes. The operation tests time 20,000 repetitions each of state copy, player-view creation, legal enumeration, a one-tile step, and End Turn. The throughput test plays 250 fixed 40-action traces per process and verifies every step and seat change. The engines enumerate their native legal actions before each scripted advance. Each timed workload gets two warm-up iterations per measured iteration.

Peak resident set size (RSS) covers the engine process, including the runtime and warm-up. A separate run holds 512 copies of the opening state at once. The Java runs use `-Xms32m -Xmx512m`, and Rust uses a model-free release build. The output reports paired bootstrap intervals from seven process pairs. The script writes a JSON file containing the raw measurements and build metadata and a CSV of the summaries. [Baseline runs](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/engine-benchmark-v1.1.json) · [Optimized runs](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/engine-benchmark-v1.1-optimized.json) · [Optimized CSV](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/engine-benchmark-v1.1-optimized.csv)

The Rust before-and-after comparison uses the same protocol and two engine revisions, `5e9905dd` and `f7444048`. The macOS sampling profile uses the repeated 900-tile trace and reports overlapping, inclusive stack counts. [Profiling data](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/engine-profile-v1.1.json)

## Reproduce the midgame search comparison

Use the pinned Tribes checkout and JavaFX jar above. Build the Rust harness at each revision you want to compare and save each executable before rebuilding. The first benchmark revision, `e200bc39`, contains the harness and the engine before board compaction.

```sh
cargo build --manifest-path polyfish-rs/Cargo.toml --locked --release --no-default-features --example midgame_bench
python3 research/run_midgame_benchmark.py /path/to/Tribes /path/to/javafx-base.jar /path/to/midgame_bench /tmp/midgame-preflight.json --preflight verify
python3 research/run_midgame_benchmark.py /path/to/Tribes /path/to/javafx-base.jar /path/to/midgame_bench /tmp/midgame-results.json --rust-revision <engine-commit>
```

The [scenario files](fixtures/midgame) describe turn-16 positions with mixed terrain, roads, developed cities, and armies of seven to fifteen units per seat. The matched workload uses full information. Each engine generates its entire native legal-action list on each cloned state and applies one validated branch. The harness times batches of 512 retained child states. Separate timed runs break down copying, legal generation, lookup, and application. Preflight verifies a shared state projection covering seat, round, stars, city production and progress, units and HP, ownership, terrain, resources, and roads. The [preflight report](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/midgame-preflight-v1.json) records the action mix and differences in native legal lists.

The native MCTS workload alternates seats on fog-of-war views, runs 512 iterations per decision, and uses a 20-action rollout cap. The reported decisions per second include observation creation. The engines use their own search trees, evaluators, and legal actions. The memory workload runs separate processes holding 0, 128, 512, and 1,024 child states and reports peak resident memory. All workloads use seven independent interleaved process pairs by default. Timing runs warm up with twice the measured batch or decision count.

The [baseline runs](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/midgame-benchmark-v1-baseline.json), [tuned runs](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/midgame-benchmark-v1-tuned.json), and [per-board comparison](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/midgame-comparison-v1.json) record the paired measurements and intervals for the engine revisions identified in the reports. New runs compare with `python3 research/summarize_midgame_benchmark.py BASELINE_JSON TUNED_JSON` when their host fingerprints match. Add `--allow-unverified-host` for the two archived reports, which were collected on the same host before the runner recorded fingerprints. To check replay parity, build the release `research_match` binary at both revisions and run `python3 research/verify_board_compaction.py BASELINE_BINARY TUNED_BINARY`.

To repeat the [Rust opening regression check](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/engine-opening-rust-paired-v1.json), save the release `research_bench` binary from each revision and run `python3 research/compare_opening_rust.py BASELINE_BINARY TUNED_BINARY BASELINE_COMMIT TUNED_COMMIT`. The [paired opening results with Java](https://github.com/HenBOMB/Polyfish/blob/964278e95d9f6cd943f9eaa279b0747e0dad40f8/research/results/engine-benchmark-after-compaction-v1.json) use the original v1.1 opening protocol.

## Reproduce the original v1 pilots

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

See the [literature and asset audit](AUDIT.md) and [paper draft](PAPER.md). The versioned source archive uses the committed tree selected by `--ref`, which defaults to `HEAD`. Earlier revisions can be packaged without the midgame benchmark example.

```sh
python3 research/package_release.py /tmp/polyfish-research-v1-rc.1.zip
```
