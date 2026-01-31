# Polyfish Architecture Notes

## Move Generation vs. Execution Trust Model

Our architecture relies on a "Trust Model" between the Move Generator (`generate_legal_moves`) and the Move Executors (`execute`).

1.  **Validation Responsibility**: The `generate_legal_moves` function is solely responsible for ensuring a move is valid, legal, and affordable for the player.
    *   **Economy**: It checks if the player has enough stars.
    *   **Tech**: It checks if the player has the required technology.
    *   **Gamestate**: It checks rules like "can only build in own territory" or "cannot move into mountains without climbing".

2.  **Execution Trust**: The `Move::execute` methods **assume** the move is valid.
    *   **No re-validation**: They do NOT re-check if the player has enough stars or tech.
    *   **Blind Execution**: For example, `SummonMove` calls `spend_stars`, which subtracts stars (clamping to 0) but does not return an error if funds were insufficient. It trusts that we wouldn't be here if we didn't have the money.

3.  **The Exception: Hidden Information**:
    *   The only time `execute` performs validation-like logic is when interacting with **Hidden State** that the generator is *not allowed* to check.
    *   *Example*: `StepMove`. The generator allows moving to an empty tile. The `execute` logic checks if that "empty" tile actually contains an `Invisible` unit (e.g., a Cloak). If it does, the move is "interrupted" (unit bumps into cloak), revealing the enemy instead of moving. This logic *must* live in execution to prevent cheating.

## Known Limitations & Technical Debt

### 1. Passenger State Loss
*   **Issue**: Currently, when a unit acts as a passenger (e.g., in a boat), it is stored as an `Option<UnitType>` (enum) rather than a full `UnitState` object.
*   **Consequence**: We lose all state specific to that unit when it embarks, including:
    *   Health (it heals to full/max of boat type?)
    *   Veteran status
    *   Age (Critical for Dragon Eggs/Babies - they stop aging while in a boat!)
    *   Kill count
*   **Future Fix**: We would need to refactor `UnitState` to allow `passenger: Option<Box<UnitState>>` or similar.

### 2. Lint Warnings
*   The codebase currently has numerous Rust lint warnings (unused imports, unused variables). These should be cleaned up to reduce noise during compilation.

### 3. Structure Blocking Rules
*   **Rule**: Structures block "natural" abilities like Clear Forest, Grow Forest, or Burn Forest.
*   **Exception**: **Roads** are the only exception. You *can* modify the forest on a tile that has a Road.

## AI Architecture
*   **Algorithm**: Monte Carlo Tree Search (MCTS).
*   **Evaluator**: Heuristic evaluation at leaf nodes.
*   **Rollouts**: Shallow random rollouts (depth ~20) to simulate tactical outcomes.
*   **Integration**: The AI uses the same `generate_legal_moves` and `play_move` interface as the game engine, ensuring it adheres to all rules.

## Map Generation
*   **Procedural**: Uses seed-based procedural generation.
*   **Pipeline**:
    1.  **intermediate representation**: Uses `GenTile` to build up terrain features, resources, and tribe affinities.
    2.  **Conversion**: Converts `GenTile` grid into final `GameState` (Tiles, Resources, Structures).
*   **Smoothing**: Uses cellular automata-like smoothing passes to create organic terrain shapes.

## Economy & Connectivity Logic
### 1. City Connections (`connection.rs`)
*   **Algorithm**: Breadth-First Search (BFS) starting from the Capital.
*   **Rules**:
    *   Standard: Roads (on land) and Ports (on water).
    *   Cymanti: Mycelium acts as a hub, connecting via Algae or other Mycelium (range 3).
*   **Undo Complexity**: The undo logic for connection updates manually tracks which cities "flipped" their connection status. This is slightly fragile and requires careful synchronization with the forward logic.

### 2. Population & Score (`city.rs`)
*   **Score Trigger**: Score for population is **only awarded on level up**, not per population point added (unless it causes a level up).
*   **Leveling**: Cities level up when `progress >= level + 1`.
*   **Missing Features**: Reward generation (e.g., choosing "Workshop" vs "Explorer") is currently a TODO in `city.rs`.

