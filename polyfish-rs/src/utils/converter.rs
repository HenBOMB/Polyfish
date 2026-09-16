//! Utility functions for converting between different types.

use crate::states::PlayerId;

/// Converts a `PlayerId` to a `usize` (0 or 1).
pub fn player_id_to_usize(player_id: PlayerId) -> usize {
    match player_id {
        1 => 0,
        2 => 1,
        _ => panic!("Invalid player id: {}", player_id),
    }
}

// Returns the opponent of the given player.
fn opponent_player_id(player_id: PlayerId) -> PlayerId {
    if player_id == 1 { 2 } else { 1 }
}
