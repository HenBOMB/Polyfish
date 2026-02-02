//! Main Game struct
//!
//! This is the primary interface for running the Polytopia simulation.
//! Translated from the TypeScript Game class.

use crate::actions::{
    self, end_unit_turn, gain_stars, has_effect, start_unit_turn, try_discover_other_tribes,
    try_remove_effect, update_exploration, UndoCallback,
};
use crate::functions::{get_pov_tribe, get_total_production, is_game_over, sync_scores};
use crate::moves::{generate_legal_moves, Move};
use crate::states::*;
use crate::types::*;
use std::fs;
use std::path::Path;

/// Starting owner ID (first player)
pub const STARTING_OWNER_ID: PlayerId = 1;

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
        Self {
            state: GameState::default(),
        }
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

        // Compute coord indices for all tiles
        for (_idx, tile) in self.state.tiles.iter_mut() {
            tile.coords.compute_idx(map_size);
            if let Some(ref mut rc) = tile.ruling_city_coords {
                rc.compute_idx(map_size);
            }
        }

        // Compute coord indices for all units
        for tribe in self.state.tribes.values_mut() {
            for unit in &mut tribe.units {
                unit.coords.compute_idx(map_size);
                unit.prev_coords.compute_idx(map_size);
                if let Some(ref mut hc) = unit.home_coords {
                    hc.compute_idx(map_size);
                }
            }
            tribe.starting_tile_coords.compute_idx(map_size);
        }

        // Set initial exploration for all tribes (real move, so this will work)

        let ids: Vec<PlayerId> = self.state.tribes.keys().cloned().collect();
        for id in ids {
            actions::update_exploration(&mut self.state, id);
        }

        // Ensure exploration is specifically set for the current player
        let pov_id = self.state.settings.current_player_turn_id;
        actions::update_exploration(&mut self.state, pov_id);

        // Update scores
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
    ///
    /// Returns None if the game is over.
    pub fn play_move(&mut self, game_move: &dyn Move) -> Option<UndoCallback> {
        if self.state.settings._game_over {
            return None;
        }

        self.state.settings._are_you_sure = true;

        let undo = if game_move.move_type() == MoveType::EndTurn {
            let end_undo = self.end_turn();
            self.state.settings._recent_moves.clear();
            end_undo
        } else {
            let result = game_move.execute(&mut self.state);
            if let Err(e) = result {
                eprintln!("Error executing move: {}", e);
                self.state.settings._are_you_sure = false;
                return None;
            }
            let res = result.unwrap();

            // Try discovering new tribes after the move
            let discover_undo = try_discover_other_tribes(&mut self.state);

            // Sync scores after move
            sync_scores(&mut self.state);

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

        self.state.settings._are_you_sure = false;

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
            if tribe.attacked_this_turn {
                tribe.pacifist_turns = 0;
            } else {
                tribe.pacifist_turns += 1;
            }
            tribe.attacked_this_turn = false;
        }

        undos.push(actions::process_end_turn_effects(state, active_pov));

        state.settings._last_player_turn_id = active_pov;
        state.settings.current_player_turn_id += 1;

        // Wrap around to first player
        if state.settings.current_player_turn_id > state.settings._max_tribe_count {
            state.settings.current_player_turn_id = STARTING_OWNER_ID;
        }

        // Skip dead/resigned tribes
        loop {
            let should_skip = state
                .tribes
                .get(&state.settings.current_player_turn_id)
                .map(|t| t.killed_turn > 0 || t.resigned_turn > 0)
                .unwrap_or(false);

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
                s.settings._game_over = old_game_over;
                s.settings.current_player_turn_id = old_pov;
                s.settings.turn = old_turn;
                s.settings._last_player_turn_id = old_last;
            });
        }

        // === NEW TRIBE TURN === //

        let new_pov = state.settings.current_player_turn_id;

        // Update exploration for new tribe
        update_exploration(state, new_pov);

        // Process start turn effects
        undos.push(actions::process_start_turn_effects(state, new_pov));

        // Try discovering tribes that moved into view
        undos.push(try_discover_other_tribes(state));

        // Reward production if not the first turn
        if state.settings.turn > 1 {
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
                .map(|(i, u)| (i, has_effect(u, EffectType::Frozen)))
                .collect();

            for (unit_idx, is_frozen) in frozen_units {
                if is_frozen {
                    // Frozen units get unfrozen but their turn ends
                    undos.push(try_remove_effect(
                        state,
                        new_pov,
                        unit_idx,
                        EffectType::Frozen,
                    ));
                    undos.push(end_unit_turn(state, new_pov, unit_idx));
                } else {
                    // Normal units get their turn reset
                    undos.push(start_unit_turn(state, new_pov, unit_idx));
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

    /// Static version: play a move on a state without a Game instance
    pub fn play_move_static(state: &mut GameState, game_move: &dyn Move) -> Option<UndoCallback> {
        if state.settings._game_over {
            return None;
        }

        state.settings._are_you_sure = true;

        let undo = if game_move.move_type() == MoveType::EndTurn {
            let end_undo = Self::end_turn_static(state);
            state.settings._recent_moves.clear();
            end_undo
        } else {
            let result = game_move.execute(state);
            if let Err(e) = result {
                eprintln!("Error executing move (static): {}", e);
                state.settings._are_you_sure = false;
                return None;
            }
            let res = result.unwrap();
            let discover_undo = try_discover_other_tribes(state);
            sync_scores(state);
            state.settings._recent_moves.push(game_move.move_type());

            let move_undo = res.undo;
            Box::new(move |s: &mut GameState| {
                s.settings._recent_moves.pop();
                discover_undo(s);
                move_undo(s);
            }) as UndoCallback
        };

        state.settings._are_you_sure = false;

        Some(undo)
    }

    /// Static version: end turn on a state without a Game instance
    pub fn end_turn_static(state: &mut GameState) -> UndoCallback {
        // Save old state
        let old_pov = state.settings.current_player_turn_id;
        let old_turn = state.settings.turn;

        let mut undos: Vec<UndoCallback> = Vec::new();

        // Change turn
        let active_pov = state.settings.current_player_turn_id;

        // Update pacifist turns
        if let Some(tribe) = state.tribes.get_mut(&active_pov) {
            if tribe.attacked_this_turn {
                tribe.pacifist_turns = 0;
            } else {
                tribe.pacifist_turns += 1;
            }
            tribe.attacked_this_turn = false;
        }

        undos.push(actions::process_end_turn_effects(state, active_pov));

        state.settings.current_player_turn_id += 1;
        if state.settings.current_player_turn_id > state.settings._max_tribe_count {
            state.settings.current_player_turn_id = STARTING_OWNER_ID;
        }

        // Skip dead tribes
        loop {
            let should_skip = state
                .tribes
                .get(&state.settings.current_player_turn_id)
                .map(|t| t.killed_turn > 0 || t.resigned_turn > 0)
                .unwrap_or(false);

            if !should_skip {
                break;
            }

            state.settings.current_player_turn_id += 1;
            if state.settings.current_player_turn_id > state.settings._max_tribe_count {
                state.settings.current_player_turn_id = STARTING_OWNER_ID;
            }
        }

        if state.settings.current_player_turn_id == STARTING_OWNER_ID {
            state.settings.turn += 1;
        }

        if is_game_over(state) {
            state.settings._game_over = true;
            return Box::new(move |s| {
                s.settings._game_over = false;
                s.settings.current_player_turn_id = old_pov;
                s.settings.turn = old_turn;
            });
        }

        let new_pov = state.settings.current_player_turn_id;
        update_exploration(state, new_pov);
        undos.push(actions::process_start_turn_effects(state, new_pov));
        undos.push(try_discover_other_tribes(state));

        if state.settings.turn > 1 {
            if let Some(tribe) = state.tribes.get(&new_pov) {
                let cities: Vec<_> = tribe.cities.clone();
                let spt = get_total_production(state, &cities);
                if spt > 0 {
                    undos.push(gain_stars(state, spt));
                }
            }
        }

        if let Some(tribe) = state.tribes.get(&new_pov) {
            let frozen_units: Vec<(usize, bool)> = tribe
                .units
                .iter()
                .enumerate()
                .map(|(i, u)| (i, has_effect(u, EffectType::Frozen)))
                .collect();

            for (unit_idx, is_frozen) in frozen_units {
                if is_frozen {
                    undos.push(try_remove_effect(
                        state,
                        new_pov,
                        unit_idx,
                        EffectType::Frozen,
                    ));
                    undos.push(end_unit_turn(state, new_pov, unit_idx));
                } else {
                    undos.push(start_unit_turn(state, new_pov, unit_idx));
                }
            }
        }

        Box::new(move |s| {
            for undo in undos.into_iter().rev() {
                undo(s);
            }
            s.settings.current_player_turn_id = old_pov;
            s.settings.turn = old_turn;
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
        self.clone_game()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = Game::new();
        assert_eq!(game.state.settings.turn, 1);
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
}
