# Polyfish: a Rust forward model for Polytopia research

Manuscript draft for AIIDE 2027. Author and affiliation review pending.

## Abstract

Polytopia-style strategy games combine partial observation, economic development, combat, and several sequential decisions per turn. These properties make them useful for testing planning agents with a forward model. We present Polyfish, a Rust simulator with a browser interface and a seeded headless runner. The initial research scope is two-player Might on 121-, 196-, and 256-tile Drylands maps. A shared interface supports rule-based and rolling-horizon opponents in both interactive games and headless training. We describe the adaptation from earlier Tribes work, test reproducible play under fog of war, and report an initial operations-level comparison against the Java Tribes reference.

## Introduction and related work

A planning agent needs to enumerate legal actions and evaluate successor states while respecting information boundaries. In a game with several decisions per player turn, the cost of copying or advancing a state can affect how much search fits into a decision. An interactive interface also makes it possible to inspect the concrete actions behind aggregate outcomes.

Java Tribes established a Polytopia-based forward-model benchmark and compared rule-based and search agents. Subsequent work used the environment for pruning and for generating diverse play styles. Polyfish adds an independent Rust environment with the same core research concerns and a local browser interface. Nearby implementations include the C-based Tribes port in `tribes-rl` and the C++ and Python environment PolyEnv. These projects differ in rules lineage, public interfaces, and what has been validated. [Tribes](https://doi.org/10.1609/aiide.v16i1.7438) · [Pruning](https://ceur-ws.org/Vol-2862/paper27.pdf) · [Play styles](https://doi.org/10.1109/cog52621.2021.9619094) · [tribes-rl](https://github.com/pham-tuan-binh/tribes-rl) · [PolyEnv](https://polyenv.readthedocs.io/en/latest/index.html)

## Methods

**Environment.** The release supports two seats, three board sizes, fog of war, and original-capital capture on the profiled Might maps. It records the seed, preset version, and replay commands. We test deterministic turns through save and load and compare the live browser route against the headless runner under the same starting position and human decisions.

**Agents.** The rule-based presets evaluate legal actions according to published Tribes priorities adapted to Polyfish's actions. The rolling-horizon agent evolves a population of one sequence by mutation and evaluates the simulated state at the end of a 20-action horizon. Its input is a belief-state copy built from the acting player's view. The interactive preset uses an explicit budget of 256 advance attempts. The original Java comparison uses the published limit of 2,000 forward-model calls and the original search settings where recoverable. We report these protocols independently.

**Performance.** We benchmark Java Tribes and Rust Polyfish on flat two-capital openings with one warrior per seat and board sizes of 121, 196, 256, 324, 400, and 900 tiles. The operations benchmark times state copy, player-view construction, legal generation, a one-tile movement, and End Turn. A separate trace measures throughput for ten rounds of a fixed script. Both seats alternate one-tile steps and end turns, and each engine enumerates its own legal actions before every advance. We use seven interleaved process pairs per workload and size on one Apple M3 Max with 36 GiB RAM. Each timed process warms up with twice its measured iteration count. The reported intervals resample process pairs. Peak resident set size (RSS) includes runtime memory. One separate memory workload holds 512 copies of the opening state. [Benchmark implementation](run_engine_benchmark.py) · [Reproduction instructions](README.md#reproduce-the-v11-engine-benchmark)

**Profiling.** On the 900-tile trace, the original Rust build spent 2,484 of 6,731 sampled stacks in build-action generation. The profiler showed repeated board scans for neutral roads even before the player had unlocked them, along with allocation of unit, technology, and structure settings. The revised engine shares immutable rule settings, skips unavailable road scans, checks occupied tiles for first contact, and bypasses path search for adjacent steps when the direct move cannot be improved. We also reduced undo bookkeeping for empty work and avoided cloning cities when computing production. Sixty seeded matches produced identical replay commands and results across the two Rust builds. [Profile summary](results/engine-profile-v1.1.json)

## Preliminary evidence and limitations

The pinned Java tournament finished four Capitals games using two published level seeds and both tribe assignments. RHEA and MCTS won two games each. This is a smoke pilot. The upstream runner chooses game and agent seeds from the clock, and its copied forward states can draw from newly initialized randomness. The recorded seeds help identify the runs. [Pilot input](tribes-pilot.json) · [Raw pilot log](results/tribes-pilot.txt)

Before optimization, Rust copied opening states 1.09–1.78× faster across the six board sizes. Java constructed player views, generated legal moves, and applied Step and End Turn faster on the same fixture. Java processed 187,041 scripted actions/s on 256 tiles and Rust processed 42,361 actions/s. Across sizes, the baseline Java throughput advantage ranged from 4.37× to 6.02×. [Baseline data and intervals](results/engine-benchmark-v1.1.json)

After optimization, Rust trace throughput increased 5.1–6.7× across board sizes. In the paired rerun, Rust processed 6.5–11.3% more scripted actions/s than Java. On 900 tiles, Rust rose from 32,592 to 218,319 actions/s, and the paired Java median was 203,916 actions/s. Rust legal generation fell from 27.36 to 3.81 µs per call, near Java's 3.79 µs. Rust Step fell from 10.87 to 2.09 µs and End Turn from 2.47 to 1.33 µs. Java still applied isolated Step and End Turn faster at 1.10 and 0.22 µs, respectively. The engines emitted 13 and 15 legal actions in the opening, respectively, and use different bookkeeping, including Rust's undo callbacks and movement-path visibility. The measurements cover a sparse opening and a scripted trace. [Optimized data and intervals](results/engine-benchmark-v1.1-optimized.json)

In the rerun, median peak RSS on the 256-tile trace was 332.7 MiB for Java and 7.0 MiB for Rust. Holding 512 opening-state copies on the same board size yielded 60.2 MiB for Java and 36.1 MiB for Rust. On 900 tiles the retained-copy workload reached 86.3 MiB for Java and 103.8 MiB for Rust. Process RSS includes runtime and allocator overhead and depends on heap settings. The original pilot's native matches use each environment's own rules and agents and remain a separate workload. [Optimized CSV summary](results/engine-benchmark-v1.1-optimized.csv) · [Original native-match pilot](results/engine-pilot.json)

The browser interface passed setup, tile inspection, turn, board fit, zoom, save, and replay checks. A controlled workflow comparison with the pinned commercial build is still needed before drawing fidelity conclusions.

## Artifact

The research source archive includes a model-free quickstart, the browser surface, the Rust runner, benchmark scripts, and pilot data. Full experimental results, formal paper authorship, and the archival DOI are pending review.
