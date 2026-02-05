// Policy composition helper for decomposed heads

use crate::ai::mapper::DecomposedMapper;
use crate::ai::network::PolicyOutput;
use crate::game::Game;
use crate::moves::Move;
use crate::types::*;
use candle_core::Tensor;

/// Compute prior probabilities for legal moves from decomposed policy heads
///
/// For each move, compute prob = product of relevant head probabilities:
/// - Action type (always)
/// - Source tile (for Step, Attack, Ability, Capture)
/// - Target tile (for Step, Attack, Ability, Summon, Harvest, Build)
/// - Option (Structure, Unit, Tech, Ability, Reward)
pub fn compute_move_priors(
    policy: &PolicyOutput,
    legal_moves: &[Box<dyn Move>],
    game: &Game,
    _allow_end_turn: bool,
) -> Vec<f32> {
    let map_size = game.state.settings.size as usize;
    let num_moves = legal_moves.len();

    // Convert tensors to probability distributions (Softmax)
    let action_probs = softmax_1d(&policy.action_type).unwrap_or_else(|_| vec![1.0 / 11.0; 11]);
    let spatial_size = (features::MAP_HEIGHT * features::MAP_WIDTH) as f32;

    let source_probs = softmax_1d(&policy.source_spatial)
        .unwrap_or_else(|_| vec![1.0 / spatial_size; spatial_size as usize]);
    let target_probs = softmax_1d(&policy.target_spatial)
        .unwrap_or_else(|_| vec![1.0 / spatial_size; spatial_size as usize]);

    // The option head is now 192 slots
    let option_probs = softmax_1d(&policy.move_option).unwrap_or_else(|_| vec![1.0 / 192.0; 192]);

    let mut priors = Vec::with_capacity(num_moves);

    for mv in legal_moves {
        let move_type = mv.move_type();
        let mut prob = 1.0;

        // 1. Action type probability
        let action_idx = DecomposedMapper::move_type_to_idx(move_type);

        if action_idx < action_probs.len() {
            prob *= action_probs[action_idx];
        }

        // 2. Source tile probability
        if let Ok(source_idx) = mv.source_idx() {
            let spatial_idx = coord_to_flat_index(source_idx, map_size);
            if spatial_idx < source_probs.len() {
                prob *= source_probs[spatial_idx];
            }
        }

        // 3. Target tile probability
        if let Ok(target_idx) = mv.target_idx() {
            let spatial_idx = coord_to_flat_index(target_idx, map_size);
            if spatial_idx < target_probs.len() {
                prob *= target_probs[spatial_idx];
            }
        }

        // 4. Move option probability (Robust mapping)
        let option_idx = match move_type {
            MoveType::Build => mv
                .structure_type()
                .ok()
                .and_then(|s| DecomposedMapper::map_structure(s)),
            MoveType::Summon => mv
                .unit_type()
                .ok()
                .and_then(|u| DecomposedMapper::map_unit(u)),
            MoveType::Research => mv
                .tech_type()
                .ok()
                .and_then(|t| DecomposedMapper::map_tech(t)),
            MoveType::Ability => mv
                .ability_type()
                .ok()
                .and_then(|a| DecomposedMapper::map_ability(a)),
            MoveType::Reward => Some(191),
            _ => None,
        };

        if let Some(idx) = option_idx {
            if idx < option_probs.len() {
                prob *= option_probs[idx];
            } else {
                eprintln!(
                    "[Composer Warning] Option index {} out of range (max 191)",
                    idx
                );
                prob *= 1e-10;
            }
        }

        priors.push(prob.max(1e-10)); // Ensure non-zero
    }

    priors
}

use crate::ai::features;

/// Convert 2D coords to flat index
fn coord_to_flat_index(idx: usize, map_size: usize) -> usize {
    let y = idx / map_size;
    let x = idx % map_size;
    (y * features::MAP_WIDTH + x).min((features::MAP_HEIGHT * features::MAP_WIDTH) - 1)
}

/// Softmax over 1D tensor
fn softmax_1d(tensor: &Tensor) -> Result<Vec<f32>, candle_core::Error> {
    let logits = tensor.flatten_all()?.to_vec1::<f32>()?;
    if logits.is_empty() {
        return Ok(vec![]);
    }

    // Find max for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(logit - max)
    let exp_values: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let exp_sum: f32 = exp_values.iter().sum();

    // Normalize
    Ok(exp_values
        .into_iter()
        .map(|v| (v / exp_sum).max(1e-12))
        .collect())
}
