//! Main Game struct
//!
//! This is the primary interface for running the Polytopia simulation.
//! Translated from the TypeScript Game class.

use crate::actions::{
    self, UndoCallback, end_unit_turn, gain_stars, has_effect, start_unit_turn,
    try_discover_other_tribes, try_remove_effect, update_exploration,
};
use crate::functions::{
    get_adjacent_indices, get_pov_tribe, get_total_production, is_game_over, sync_scores,
};
use crate::moves::{Move, generate_legal_moves};
use crate::settings::has_technology;
use crate::states::*;
use crate::types::*;
use std::fs;
use std::path::Path;

/// Starting owner ID (first player)
pub const STARTING_OWNER_ID: PlayerId = 1;

/// Diagnostic: how many EndTurn edges MCTS actually simulates (i.e. how often
/// search crosses a turn boundary in-tree). Read/reset by self_play's summary.
pub static SIM_END_TURN_EDGES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// The main game controller
///
/// Provides the interface for loading game states, playing moves, and managing turns.
#[derive(Debug)]
pub struct Game {
    pub state: GameState,
}

impl Game {
    /// Create a new game with default state
    pub fn new() -> Self {
        let mut state = GameState::default();
        state.initial_seed = state.settings.seed;
        Self { state }
    }

    /// Load game state from a JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let state: GameState = serde_json::from_str(json)?;
        let mut game = Self { state };
        game.post_load();
        Ok(game)
    }

    /// Load game state from a JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        let game = Self::from_json(&json)?;
        Ok(game)
    }

    /// Post-load initialization (visibility, coord indices, etc.)
    pub fn post_load(&mut self) {
        let map_size = self.state.settings.size;

        // 1. Compute coord indexes for all tiles
        for (_idx, tile) in self.state.tiles.iter_mut() {
            tile.coords.compute_idx(map_size);
            // if *idx == 147 && tile.owner != 0 {
            //     println!("🐛 Tile 147 is already owned by {}!", tile.owner);
            // }
            if let Some(ref mut rc) = tile.ruling_city_coords {
                rc.compute_idx(map_size);
            }
        }

        // 2. Compute coord indices and scale health for units
        for tribe in self.state.tribes.values_mut() {
            for unit in &mut tribe.units {
                unit.coords.compute_idx(map_size);
                unit.prev_coords.compute_idx(map_size);
                if let Some(ref mut hc) = unit.home_coords {
                    hc.compute_idx(map_size);
                }

                // Set tile unit owner correctly
                if let Some(tile) = self.state.tiles.get_mut(&unit.coords.idx) {
                    tile._unit_owner_id = Some(unit.owner);
                }
            }
            tribe.starting_tile_coords.compute_idx(map_size);
        }

        // 3. Calculate _territory for cities
        let mut territory_updates = Vec::new();
        for tribe in self.state.tribes.values() {
            for city in &tribe.cities {
                // fetch nearby tiles and filter the ones we own
                let territory = get_adjacent_indices(&self.state, city.idx, 2)
                    .into_iter()
                    .filter(|&idx| {
                        self.state
                            .tiles
                            .get(&idx)
                            .map(|t| t.owner == city.owner)
                            .unwrap_or(false)
                    })
                    .collect::<Vec<i32>>();
                territory_updates.push((tribe.id, city.idx, territory));
            }
        }

        for (tribe_id, city_idx, territory) in territory_updates {
            if let Some(tribe) = self.state.tribes.get_mut(&tribe_id) {
                if let Some(city) = tribe.cities.iter_mut().find(|c| c.idx == city_idx) {
                    city._territory = territory;
                }
            }
        }

        // 4. Update initial vision for all tribes
        let old_sure = self.state.settings._are_you_sure;
        self.state.settings._are_you_sure = true;
        let ids: Vec<PlayerId> = self.state.tribes.keys().cloned().collect();
        for id in ids {
            let _ = update_exploration(&mut self.state, id);
        }
        self.state.settings._are_you_sure = old_sure;

        // 5. Update scores (first time)
        sync_scores(&mut self.state);
    }

    /// Serialize game state to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.state)
    }

    /// Clone the entire game
    pub fn clone_game(&self) -> Self {
        // Deep clone via serialization
        let json = self.to_json().unwrap_or_default();
        Self::from_json(&json).unwrap_or_else(|_| Self::new())
    }

    /// Clone the entire game and obscure hidden information for MCTS simulations
    pub fn clone_for_mcts(&self, pov_id: PlayerId) -> Self {
        let mut cloned = self.clone(); // Fast direct clone
        cloned.state.obscure_fog(pov_id);
        cloned
    }

    /// Clone just the game state
    pub fn clone_state(&self) -> GameState {
        // Deep clone via serialization
        let json = serde_json::to_string(&self.state).unwrap_or_default();
        serde_json::from_str(&json).unwrap_or_default()
    }

    /// Get all legal moves for the current player
    pub fn legal_moves(&self) -> Vec<Box<dyn Move>> {
        generate_legal_moves(&self.state)
    }

    /// Play a sequence of moves by their indices in the legal moves list
    pub fn play_sequence(&mut self, move_indices: &[usize]) -> Vec<UndoCallback> {
        let mut undos = Vec::new();

        for &idx in move_indices {
            let legal = self.legal_moves();
            if idx < legal.len() {
                if let Some(undo) = self.play_move(legal[idx].as_ref()) {
                    undos.push(undo);
                }
            }
        }

        undos
    }

    /// Play a move and return an undo callback
    /// Returns None if the game is over.
    pub fn play_move(&mut self, game_move: &dyn Move) -> Option<UndoCallback> {
        if self.state.settings._game_over {
            return None;
        }

        self.state.settings._are_you_sure = true;

        let undo = if game_move.move_type() == MoveType::EndTurn {
            let end_undo = self.end_turn();
            let old_recent_moves = std::mem::take(&mut self.state.settings._recent_moves);
            Box::new(move |s: &mut GameState| {
                end_undo(s);
                s.settings._recent_moves = old_recent_moves;
            }) as UndoCallback
        } else if game_move.move_type() == MoveType::Resign {
            let res = game_move.execute(&mut self.state).unwrap();
            let move_undo = res.undo;
            let end_undo = self.end_turn();
            let old_recent_moves = std::mem::take(&mut self.state.settings._recent_moves);
            self.state._history.push(game_move.serialize());

            Box::new(move |s: &mut GameState| {
                s._history.pop();
                end_undo(s);
                move_undo(s);
                s.settings._recent_moves = old_recent_moves;
            }) as UndoCallback
        } else {
            let result = game_move.execute(&mut self.state);
            if let Err(e) = result {
                eprintln!(
                    "Error executing move [{}]: {}",
                    game_move.describe(&self.state),
                    e
                );
                self.state.settings._are_you_sure = false;
                return None;
            }
            let res = result.unwrap();

            // Try discovering new tribes after the move
            let mut discover_undo: UndoCallback = actions::noop_undo();

            // If FOW is disabled, then preload discoveries
            if self.state.settings._fow {
                discover_undo = try_discover_other_tribes(&mut self.state);
            }

            // Track the move type in recent moves
            self.state
                .settings
                ._recent_moves
                .push(game_move.move_type());

            // Add to history
            self.state._history.push(game_move.serialize());

            // Collect undos
            let move_undo = res.undo;

            Box::new(move |s: &mut GameState| {
                s.settings._recent_moves.pop();
                s._history.pop();
                discover_undo(s);
                move_undo(s);
            }) as UndoCallback
        };

        // Fog memory: record post-move observations (real moves only).
        let mem_undo = crate::memory::observe_all(&mut self.state);

        self.state.settings._are_you_sure = false;

        Some(Box::new(move |s: &mut GameState| {
            mem_undo(s);
            undo(s);
        }))
    }

    /// Simulate a move for MCTS (does NOT set _are_you_sure, preventing exploration)
    ///
    /// This is specifically for MCTS simulations where we don't want to
    /// permanently reveal tiles on the map. For EndTurn moves, this implements
    /// single-player MCTS by skipping enemy turns and cycling back to the original player.
    pub fn simulate_move(&mut self, game_move: &dyn Move) -> Option<UndoCallback> {
        if self.state.settings._game_over {
            return None;
        }

        // NOTE: We deliberately do NOT set _are_you_sure = true here
        // This prevents exploration during MCTS simulations

        let undo = if game_move.move_type() == MoveType::EndTurn {
            SIM_END_TURN_EDGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Single-player MCTS: skip enemy turns and return to original player
            let original_player = self.state.settings.current_player_turn_id;
            let mut undos = Vec::new();

            // End our turn
            undos.push(self.end_turn());
            let old_recent_moves = std::mem::take(&mut self.state.settings._recent_moves);

            // Keep ending turns until we're back at original player
            // This effectively "skips" all enemy turns
            let max_players = 16; // Safety limit to prevent infinite loop
            let mut iterations = 0;
            while self.state.settings.current_player_turn_id != original_player
                && iterations < max_players
                && !self.state.settings._game_over
            {
                undos.push(self.end_turn());
                iterations += 1;
            }

            // Combine all undos
            Box::new(move |s: &mut GameState| {
                while let Some(undo) = undos.pop() {
                    undo(s);
                }
                s.settings._recent_moves = old_recent_moves;
            }) as UndoCallback
        } else if game_move.move_type() == MoveType::Resign {
            let res = game_move.execute(&mut self.state).unwrap();
            let move_undo = res.undo;
            let end_undo = self.end_turn();
            let old_recent_moves = std::mem::take(&mut self.state.settings._recent_moves);

            Box::new(move |s: &mut GameState| {
                end_undo(s);
                move_undo(s);
                s.settings._recent_moves = old_recent_moves;
            }) as UndoCallback
        } else {
            let result = game_move.execute(&mut self.state);
            if let Err(e) = result {
                eprintln!("Error executing simulated move: {}", e);
                return None;
            }
            let res = result.unwrap();

            // Try discovering new tribes after the move (won't actually reveal in simulation)
            let discover_undo = try_discover_other_tribes(&mut self.state);

            // Track the move type in recent moves
            self.state
                .settings
                ._recent_moves
                .push(game_move.move_type());

            // Collect undos
            let move_undo = res.undo;

            Box::new(move |s: &mut GameState| {
                s.settings._recent_moves.pop();
                discover_undo(s);
                move_undo(s);
            }) as UndoCallback
        };

        Some(undo)
    }

    /// End the current tribe's turn
    ///
    /// This handles:
    /// - Changing to the next alive tribe
    /// - Incrementing turn counter when wrapping around
    /// - Updating visibility for new tribe
    /// - Discovering tribes that moved into view
    /// - Rewarding production stars
    /// - Resetting/updating unit states (frozen, moved, attacked)
    fn end_turn(&mut self) -> UndoCallback {
        let state = &mut self.state;

        // Save old state for undo
        let old_pov = state.settings.current_player_turn_id;
        let old_turn = state.settings.turn;
        let old_last = state.settings._last_player_turn_id;
        let old_game_over = state.settings._game_over;

        // Track all undos in a chain
        let mut undos: Vec<UndoCallback> = Vec::new();

        // === CHANGE TURN === //
        let active_pov = state.settings.current_player_turn_id;

        // Update pacifist turns
        if let Some(tribe) = state.tribes.get_mut(&active_pov) {
            let old_pacifist_turns = tribe.pacifist_turns;
            let old_attacked_this_turn = tribe.attacked_this_turn;
            if tribe.attacked_this_turn {
                tribe.pacifist_turns = 0;
            }
            // Only track if required meditation tech has been unlocked
            else if has_technology(&tribe.tech_vanilla, TechnologyType::Meditation) {
                tribe.pacifist_turns += 1;
            }
            tribe.attacked_this_turn = false;

            undos.push(Box::new(move |s| {
                if let Some(t) = s.tribes.get_mut(&active_pov) {
                    t.pacifist_turns = old_pacifist_turns;
                    t.attacked_this_turn = old_attacked_this_turn;
                }
            }));
        }

        undos.push(actions::process_end_turn_effects(state, active_pov));

        state.settings._last_player_turn_id = active_pov;
        state.settings.current_player_turn_id += 1;

        // Wrap around to first player
        if state.settings.current_player_turn_id > state.settings._max_tribe_count {
            state.settings.current_player_turn_id = STARTING_OWNER_ID;
        }

        // Skip dead/resigned/missing tribes
        loop {
            // Unwrapping to `true` ensures that if a tribe ID doesn't exist in the map
            // (e.g. empty slots up to max_tribe_count), we correctly skip them.
            let should_skip = state
                .tribes
                .get(&state.settings.current_player_turn_id)
                .map(|t| t.killed_turn > 0 || t.resigned_turn > 0)
                .unwrap_or(true);

            if !should_skip {
                break;
            }

            state.settings.current_player_turn_id += 1;
            if state.settings.current_player_turn_id > state.settings._max_tribe_count {
                state.settings.current_player_turn_id = STARTING_OWNER_ID;
            }
        }

        // If we wrapped around to the start, increment turn
        if state.settings.current_player_turn_id == STARTING_OWNER_ID {
            state.settings.turn += 1;
        }

        // Check for game over
        if is_game_over(state) {
            state.settings._game_over = true;
            return Box::new(move |s| {
                for undo in undos.into_iter().rev() {
                    undo(s);
                }
                s.settings._game_over = old_game_over;
                s.settings.current_player_turn_id = old_pov;
                s.settings.turn = old_turn;
                s.settings._last_player_turn_id = old_last;
            });
        }

        // === NEW TRIBE TURN === //

        let new_pov = state.settings.current_player_turn_id;

        // Update exploration for new tribe
        undos.push(update_exploration(state, new_pov));

        // Process start turn effects
        undos.push(actions::process_start_turn_effects(state, new_pov));

        // Try discovering tribes that moved into view
        if state.settings._fow {
            undos.push(try_discover_other_tribes(state));
        }

        // Reward production if not the first turn
        if state.settings.turn > 0 {
            if let Some(tribe) = state.tribes.get(&new_pov) {
                let cities: Vec<_> = tribe.cities.clone();
                let spt = get_total_production(state, &cities);
                if spt > 0 {
                    undos.push(gain_stars(state, spt));
                }
            }
        }

        // Update all unit states
        if let Some(tribe) = state.tribes.get(&new_pov) {
            let _unit_count = tribe.units.len();
            let frozen_units: Vec<(usize, bool)> = tribe
                .units
                .iter()
                .enumerate()
                .map(|(i, u)| (i, has_effect(u, UnitEffect::Frozen)))
                .collect();

            for (unit_idx, is_frozen) in frozen_units {
                if is_frozen {
                    // Frozen units get unfrozen but their turn ends
                    undos.push(try_remove_effect(
                        state,
                        new_pov,
                        unit_idx,
                        UnitEffect::Frozen,
                    ));
                    undos.push(end_unit_turn(state, new_pov, unit_idx));
                } else {
                    // Normal units get their turn reset
                    undos.push(start_unit_turn(state, new_pov, unit_idx));

                    // // add debug log
                    // println!(
                    //     "Unit {} started turn",
                    //     state.tribes.get(&new_pov).unwrap().units[unit_idx]
                    //         .coords
                    //         .idx
                    // );

                    // // print attacked and moved
                    // println!(
                    //     "Unit {} attacked: {}, moved: {}",
                    //     state.tribes.get(&new_pov).unwrap().units[unit_idx]
                    //         .coords
                    //         .idx,
                    //     state.tribes.get(&new_pov).unwrap().units[unit_idx].attacked,
                    //     state.tribes.get(&new_pov).unwrap().units[unit_idx].moved
                    // );
                }
            }
        }

        // Create the combined undo callback
        Box::new(move |s| {
            // Undo all collected operations in reverse
            for undo in undos.into_iter().rev() {
                undo(s);
            }

            // Restore turn state
            s.settings._game_over = old_game_over;
            s.settings.current_player_turn_id = old_pov;
            s.settings.turn = old_turn;
            s.settings._last_player_turn_id = old_last;
        })
    }

    // === Convenience accessors === //

    /// Get the current player's tribe
    pub fn current_tribe(&self) -> Option<&TribeState> {
        get_pov_tribe(&self.state)
    }

    /// Check if the game is over
    pub fn is_game_over(&self) -> bool {
        is_game_over(&self.state)
    }

    /// Get the current turn number
    pub fn turn(&self) -> i32 {
        self.state.settings.turn
    }

    /// Get the map size
    pub fn map_size(&self) -> i32 {
        self.state.settings.size
    }

    /// Get current player ID
    pub fn current_player_id(&self) -> PlayerId {
        self.state.settings.current_player_turn_id
    }

    /// Get total tile count
    pub fn tile_count(&self) -> i32 {
        self.state.settings.size * self.state.settings.size
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Game {
    fn clone(&self) -> Self {
        // Direct clone without JSON serialization
        // This preserves exact state without re-running post_load()

        let cloned = Self {
            state: self.state.clone(),
        };

        cloned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = Game::new();
        assert_eq!(game.state.settings.turn, 0);
        assert_eq!(game.state.settings.mode, ModeType::Domination);
    }

    #[test]
    fn test_legal_moves_includes_end_turn() {
        let game = Game::new();
        let moves = game.legal_moves();
        assert!(moves.iter().any(|m| m.move_type() == MoveType::EndTurn));
    }

    #[test]
    fn test_clone_game() {
        let game = Game::new();
        let cloned = game.clone_game();
        assert_eq!(game.map_size(), cloned.map_size());
        assert_eq!(game.turn(), cloned.turn());
    }

    #[test]
    fn test_initial_vision() {
        use crate::mapgen::{MapGenSettings, generate};
        let settings = MapGenSettings {
            size: MapSize::Tiny,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            ..Default::default()
        };
        let mut game = Game::new();
        game.state = generate(settings);
        game.post_load();

        let p1_explored = game
            .state
            .tiles
            .values()
            .filter(|t| t.explorers.contains(&1))
            .count();
        let p2_explored = game
            .state
            .tiles
            .values()
            .filter(|t| t.explorers.contains(&2))
            .count();

        // Each player should have ~25 tiles explored (radius 2 around capital)
        assert!(
            p1_explored >= 9,
            "P1 should have at least city radius 1 explored"
        );
        assert!(
            p2_explored >= 9,
            "P2 should have at least city radius 1 explored"
        );
    }
}
