# Polyfish v1 research simulator

Polyfish v1 targets two-player Might on Drylands with fog of war at 121, 196, and 256 tiles. The initial validated tribes are Xin-Xi, Imperius, Bardur, and Oumaji. The first research release will document simulator fidelity and performance, provide usable reference opponents, and give downstream work a versioned artifact to cite.

## Acceptance checklist

- [ ] Record the Tribes citation search, available implementations, research settings, and evidence of downstream use. Compare Java Tribes, tribes-rl, and PolyEnv.
- [ ] Verify an upstream source build, audit both browser entry points and asset provenance, and pin the commercial reference version.
- [ ] Make `/simulator/` the primary play surface and verify sizing, hit targets, navigation, and the full play workflow across the three sizes.
- [ ] Add validated size and opponent configuration to new games. Preserve it through save and replay. Verify Might ends when the opponent's original capital is captured.
- [ ] Ship Easy and Normal rule-based presets and an Expert Rolling Horizon Evolution preset from the published Tribes algorithms. Verify the same seeded decisions in the browser and model-free headless runner. Keep the learned agent available at 121 tiles.
- [ ] Add an opt-in opponent preset to 11×11 training. Alternate the learner's seat, keep bot actions in game and reward transitions, and write only learner training samples. Exercise existing checkpoint and anchor modes.
- [ ] Verify legal play, deterministic full turns, fog isolation, live and loaded games, and training data. Reproduce the historical Java RHEA-versus-MCTS experiment where recoverable and validate the Rust adaptation on equivalent fixtures.
- [ ] Compare headless Java and Rust performance on semantically matched workloads. Document native whole-game throughput and interface workflows as separate measurements.
- [ ] Prepare the paper for AIIDE 2027 and build a reproducible release with citation metadata, a Zenodo record, a model-free quickstart, and the experiment inputs, scripts, and raw outputs. Review authorship, claims, and asset provenance before publication.

## References

- [Tribes: A New Turn-Based Strategy Game for AI Research](https://ojs.aaai.org/index.php/AIIDE/article/view/7438)
- [Java Tribes reference implementation](https://github.com/GAIGResearch/Tribes)
- [MCTS Pruning in Turn-Based Strategy Games](https://ceur-ws.org/Vol-2862/paper27.pdf)
- [Generating Diverse and Competitive Play-Styles for Strategy Games](https://arxiv.org/abs/2104.08641)
- [Polytopia Might reference rules](https://polytopia.io/World%20Championship/)
