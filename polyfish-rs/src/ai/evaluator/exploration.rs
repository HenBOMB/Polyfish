use crate::ai::genes::AIGenes;
use crate::states::{GameState, PlayerId};

/// Evaluates the exploration score (Fog of War revealed) for a given player.
/// Formula from notes-heuristics.md, clarified by user:
/// "if i had explored 80% of the map, i would get 1 score. the extra 20% doesnt add more"
///
/// Mapping:
/// - < 20% explored -> 0.0
/// - target% explored -> 1.0
/// - > target% explored -> 1.0
pub fn evaluate_exploration(state: &GameState, player_id: PlayerId, genes: &AIGenes) -> f32 {
    let tribe_opt = state.tribes.get(&player_id);
    if tribe_opt.is_none() {
        return 0.0;
    }
    // We don't need tribe object, we check tiles directly.

    let total_tiles = state.tile_count() as f32;
    if total_tiles == 0.0 {
        return 0.0;
    }

    let mut explored_count = 0.0;
    for tile in state.tiles.values() {
        if tile.explorers.contains(&player_id) {
            explored_count += 1.0;
        }
    }

    let max_exploration_target = genes.exploration.max_exploration_target;
    let min_threshold_pct = 1.0 - max_exploration_target;
    let spread = max_exploration_target - min_threshold_pct;

    let score = (explored_count - total_tiles * min_threshold_pct) / (total_tiles * spread);

    score.clamp(0.0, 1.0)
}
