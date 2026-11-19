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

**Performance.** The pilot times copy, player-view construction, legal generation, and EndTurn on flat two-capital openings. Both programs run on the same CPU. Each process warms up with twice as many iterations as the measurement. Five independent processes per engine and board size supply raw totals and a bootstrap interval for the median timing ratio. The report records process-inclusive native match timings separately. Additional matched mid- and late-game positions would establish how the findings change as the board fills.

## Preliminary evidence and limitations

The pinned Java tournament finished four Capitals games using two published level seeds and both tribe assignments. RHEA and MCTS won two games each. This is a smoke pilot. The upstream runner chooses game and agent seeds from the clock, and its copied forward states can draw from newly initialized randomness. The recorded seeds help identify the runs. [Pilot input](tribes-pilot.json) · [Raw pilot log](results/tribes-pilot.txt)

On the Apple M3 Max pilot, Java-to-Rust state-copy time ratios were 2.44×, 1.78×, and 1.52× on the 11×11, 14×14, and 16×16 openings, respectively. The flat openings agree on board dimensions and EndTurn rotation. Legal candidate counts differ (13 in Java and 15 in Rust), and the engines perform different bookkeeping. Java measured faster for legal enumeration and EndTurn in these cases. Native match timings use each environment's own game rules and rule-based agents. A general end-to-end speed advantage remains unestablished. [Raw timing records and intervals](results/engine-pilot.json)

The browser interface passed setup, tile inspection, turn, board fit, zoom, save, and replay checks. A controlled workflow comparison with the pinned commercial build is still needed before drawing fidelity conclusions.

## Artifact

The research source archive includes a model-free quickstart, the browser surface, the Rust runner, benchmark scripts, and pilot data. Full experimental results, formal paper authorship, and the archival DOI are pending review.
