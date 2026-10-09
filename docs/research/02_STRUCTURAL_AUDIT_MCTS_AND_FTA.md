# Structural Audit: MCTS Solitaire Assumption, In-Tree Opponent Modeling, and First-Turn Advantage

This document details the architectural findings uncovered in the `polyfish-rs` engine regarding in-tree MCTS simulation semantics, seat asymmetry physics, and training label dynamics.

---

## 1. The Core Finding: MCTS is Single-Player Solitaire by Default

In standard AlphaZero (Chess, Go), MCTS models alternating two-player zero-sum moves: Player A selects a move, then Player B responds with their optimal countermove.

In `polyfish-rs`, **this does NOT occur by default**.

### 1.1 In-Tree Turn Deletion in `game.rs`
In [`polyfish-rs/src/game.rs#L486-L498`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/game.rs#L486-L498):
```rust
// Legacy single-player search: keep ending turns until control is
// back with the mover, i.e. delete every opponent turn in between.
if !adversarial_search() {
    let max_players = 16; // Safety limit to prevent infinite loop
    let mut iterations = 0;
    while self.state.settings.current_player_turn_id != original_player
        && iterations < max_players
        && !self.state.settings._game_over
    {
        undos.push(self.end_turn());
        iterations += 1;
    }
}
```

- `adversarial_search()` is **`false` by default** in both self-play and arena evaluation.
- When the search tree simulates `EndTurn`, it calls `end_turn()` in a loop until control cycles back to the mover.
- `end_turn()` ([`game.rs#L557-L600`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/game.rs#L557-L600)) only resets unit moved/attacked flags, distributes star production, and increments `current_player_turn_id`. **It executes zero moves for the opponent.**

### 1.2 The Solitaire Search Consequence
Because the search tree skips the opponent's turn, in-tree rollouts are essentially single-agent puzzle solvers:
- The tree evaluates forward lines under the implicit assumption that the opponent takes no tactical or reactive moves between turns.
- Any tactical overextension appears viable in-tree because opponent counter-actions are never executed during simulation.
- As a consequence, MCTS policy distributions ($\pi$) optimize for greedy expansion rather than robust tactical balance.

### 1.3 The Search Backend Mechanics
- In commit `88f70422`, the engine introduced `game::adversarial_search()` (controllable via `POLYFISH_ADVERSARIAL_SEARCH=1` and `arena --adversarial`), which allows `simulate_move(EndTurn)` to hand control over to the opponent.
- However, adversarial search was kept **default off** because historical baselines had never evaluated its impact, and Gumbel MCTS was originally designed for single-agent Sequential Halving.
- In [`polyfish-rs/src/ai/gumbel_mcts.rs#L537-L542`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/ai/gumbel_mcts.rs#L537-L542), `EndTurn` is also stripped from root candidate actions when other moves exist to prevent passive play, enforcing aggressive move exhaustion.

---

## 2. Physics of First-Turn Advantage (FTA) on 11×11 Maps

Our empirical benchmarks revealed that **100% of our wins against Iter 156 occurred as Player 1**, and all decisive eliminations occurred as Player 2.

### 2.1 The Geometry of 11×11 Tiny Maps
- On an 11×11 map, capital cities spawn approximately 6 to 8 tiles apart.
- An equidistant central village is reached by Player 1 on Turn 1 or 2. Player 2 always arrives 1 ply later.
- First capture grants immediate border expansion, vision into the fog, and higher stars-per-turn (SPT).

### 2.2 Asymmetric Combat Dynamics
- Combat in *Polytopia* is non-simultaneous: the attacker deals damage first at 100% effectiveness. The defender retaliates only with their surviving HP.
- The player holding tempo initiative dictates when and where engagements occur, dealing maximum damage and receiving attenuated counter-attacks.
- Player 2 is perpetually forced to react. Under a solitaire search tree that does not model opponent threats, Player 2 is at a severe structural disadvantage.

---

## 3. Dissecting Mirror Labels (`--mirror-labels`)

A common question is whether `--mirror-labels` can teach defensive play for Player 2. The source code clarifies its exact function:

### 3.1 Implementation in `self_play.rs`
In [`polyfish-rs/src/bin/self_play.rs#L2345-L2357`](file:///mnt/hen480/henry/Escritorio/Coding/PolyAI/polyfish-rs/src/bin/self_play.rs#L2345-L2357):
```rust
let final_outcome = match partner_outcomes {
    Some((pz1, pz2)) => {
        let z_counterpart = if p_id == 1 { pz2 } else { pz1 };
        let pair_mean = 0.5 * (own_outcome + z_counterpart);
        let u = if hist_len > 1 {
            step_idx as f32 / (hist_len - 1) as f32
        } else {
            1.0
        };
        (1.0 - u) * pair_mean + u * own_outcome
    }
    None => own_outcome,
};
```

### 3.2 Value De-biasing vs Policy Guidance
1. **Mirror labels were ALREADY active**: In `self_play.rs`, `mirror_labels` defaults to `true` (CLI arg line 1394). Every game generated across our DAgger runs had mirror partners (`[MirrorLabels] 100/100 games have a live mirror partner`).
2. **Value Variance Reduction ($V$)**:
   - If Player 1 wins both paired games due to first-mover advantage, the mirror label adjusts early game value targets toward $0.5 \times (1.0 + (-1.0)) = 0.0$.
   - This prevents the value head from falsely predicting that Seat 1 always wins from turn 1 regardless of skill.
3. **Zero Policy Impact ($\pi$)**:
   - Mirror labels only modify the scalar $z$ used to train the value head.
   - The policy head targets ($\pi$) in DAgger came from the teacher.
   - Mirror labels tell the value function that opening positions are neutral, but cannot generate tactical policy moves.
