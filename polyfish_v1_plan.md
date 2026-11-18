# Polyfish v1 research simulator

Polyfish v1 targets two-player Might on Drylands with fog of war at 121, 196, and 256 tiles. The initial validated tribes are Xin-Xi, Imperius, Bardur, and Oumaji. The first research release will document simulator fidelity and performance, provide usable reference opponents, and give downstream work a versioned artifact to cite.

## Acceptance checklist

- [x] Verify an upstream source build, audit the browser entry points and assets, and pin commercial reference version 2.16.8.
- [x] Make `/simulator/` the primary play surface. Check browser play, mobile board fit, zoom, saves, and replay stepping.
- [x] Add map-size and opponent settings, preserve them in saves and replays, and verify original-capital victory under Might.
- [x] Add Easy, Normal, and Expert. Verify nine complete matches through the browser API, headless runner, and replay player, covering all three board sizes. Validate fog isolation and rolling-horizon advance counts.
- [x] Add opt-in 11×11 preset training with alternating seats and learner-only samples. Exercise the checkpoint-opponent and anchor paths.
- [x] Run the pinned four-game Java Tribes pilot and fixed-board timing pilot. Record native whole-game timings separately.
- [x] Draft the AIIDE paper and prepare the model-free source archive, citation metadata, quickstart, scripts, and pilot records.
- [ ] Finish the citation audit. The OpenAlex snapshot and primary sources are recorded, and Google Scholar returned HTTP 403.
- [ ] Complete the commercial workflow comparison and expand the Java/Rust comparison to equivalent mid- and late-game workloads.
- [ ] Review authorship and claims, confirm CI, and register the reviewed artifact with Zenodo.

The [research quickstart](research/README.md), [audit](research/AUDIT.md), [paper draft](research/PAPER.md), and [pilot results](research/results/) live alongside the implementation.

## References

- [Tribes: A New Turn-Based Strategy Game for AI Research](https://ojs.aaai.org/index.php/AIIDE/article/view/7438)
- [Java Tribes reference implementation](https://github.com/GAIGResearch/Tribes)
- [MCTS Pruning in Turn-Based Strategy Games](https://ceur-ws.org/Vol-2862/paper27.pdf)
- [Generating Diverse and Competitive Play-Styles for Strategy Games](https://arxiv.org/abs/2104.08641)
- [Polytopia Might reference rules](https://polytopia.io/World%20Championship/)
