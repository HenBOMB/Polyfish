// Policy composition helper for decomposed heads

use crate::ai::mapper::DecomposedMapper;
use crate::ai::network::{PolicyOutput, RawPolicyOutput};
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
    map_size: usize,
    _allow_end_turn: bool,
) -> Vec<f32> {
    // Convert tensors to probability distributions (Softmax)
    let action_probs = softmax_1d(&policy.action_type).unwrap_or_else(|_| vec![1.0 / 11.0; 11]);
    let spatial_size = (features::MAP_SIZE * features::MAP_SIZE) as f32;

    let source_probs = softmax_1d(&policy.source_spatial)
        .unwrap_or_else(|_| vec![1.0 / spatial_size; spatial_size as usize]);
    let target_probs = softmax_1d(&policy.target_spatial)
        .unwrap_or_else(|_| vec![1.0 / spatial_size; spatial_size as usize]);

    // The option head is now 192 slots
    let option_probs = softmax_1d(&policy.move_option).unwrap_or_else(|_| vec![1.0 / 192.0; 192]);

    compute_move_priors_from_probs(&action_probs, &source_probs, &target_probs, &option_probs, legal_moves, map_size)
}

/// Same as [`compute_move_priors`] but operates on a device-free
/// [`RawPolicyOutput`] row (plain logit `Vec<f32>`s). Used by the batched
/// eval path, where per-leaf policy slices are extracted as CPU floats
/// rather than `Tensor`s (see `RawPolicyOutput`).
pub fn compute_move_priors_raw(
    policy: &RawPolicyOutput,
    legal_moves: &[Box<dyn Move>],
    map_size: usize,
    _allow_end_turn: bool,
) -> Vec<f32> {
    let action_probs = softmax(&policy.action_type);
    let source_probs = softmax(&policy.source_spatial);
    let target_probs = softmax(&policy.target_spatial);
    let option_probs = softmax(&policy.move_option);

    compute_move_priors_from_probs(&action_probs, &source_probs, &target_probs, &option_probs, legal_moves, map_size)
}

fn compute_move_priors_from_probs(
    action_probs: &[f32],
    source_probs: &[f32],
    target_probs: &[f32],
    option_probs: &[f32],
    legal_moves: &[Box<dyn Move>],
    map_size: usize,
) -> Vec<f32> {
    let num_moves = legal_moves.len();
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
            MoveType::Reward => mv
                .reward_type()
                .ok()
                .and_then(DecomposedMapper::map_reward),
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

/// Compute log-probabilities for legal moves from decomposed policy heads
///
/// Log-prob = log(P(action_type)) + log(P(source)) + log(P(target)) + log(P(option))
/// These are used as "logits" for Gumbel-Top-k sampling in Gumbel AlphaZero.
pub fn compute_move_log_probs(
    policy: &PolicyOutput,
    legal_moves: &[Box<dyn Move>],
    map_size: usize,
) -> Vec<f32> {
    // Convert tensors to log-probabilities (LogSoftmax)
    let action_logs = log_softmax_1d(&policy.action_type).unwrap_or_else(|_| vec![-2.4; 11]);
    let spatial_size = (features::MAP_SIZE * features::MAP_SIZE) as f32;
    let log_spatial_unif = -(spatial_size.ln());

    let source_logs = log_softmax_1d(&policy.source_spatial)
        .unwrap_or_else(|_| vec![log_spatial_unif; spatial_size as usize]);
    let target_logs = log_softmax_1d(&policy.target_spatial)
        .unwrap_or_else(|_| vec![log_spatial_unif; spatial_size as usize]);

    let option_logs = log_softmax_1d(&policy.move_option).unwrap_or_else(|_| vec![-5.25; 192]);

    compute_move_log_probs_from_logs(&action_logs, &source_logs, &target_logs, &option_logs, legal_moves, map_size)
}

/// Same as [`compute_move_log_probs`] but operates on a device-free
/// [`RawPolicyOutput`] row. See [`compute_move_priors_raw`] for why this
/// variant exists.
pub fn compute_move_log_probs_raw(
    policy: &RawPolicyOutput,
    legal_moves: &[Box<dyn Move>],
    map_size: usize,
) -> Vec<f32> {
    let action_logs = log_softmax(&policy.action_type);
    let source_logs = log_softmax(&policy.source_spatial);
    let target_logs = log_softmax(&policy.target_spatial);
    let option_logs = log_softmax(&policy.move_option);

    compute_move_log_probs_from_logs(&action_logs, &source_logs, &target_logs, &option_logs, legal_moves, map_size)
}

fn compute_move_log_probs_from_logs(
    action_logs: &[f32],
    source_logs: &[f32],
    target_logs: &[f32],
    option_logs: &[f32],
    legal_moves: &[Box<dyn Move>],
    map_size: usize,
) -> Vec<f32> {
    let num_moves = legal_moves.len();
    let mut move_logs = Vec::with_capacity(num_moves);

    for mv in legal_moves {
        let move_type = mv.move_type();
        let mut log_prob = 0.0;

        // 1. Action type log-prob
        let action_idx = DecomposedMapper::move_type_to_idx(move_type);
        if action_idx < action_logs.len() {
            log_prob += action_logs[action_idx];
        }

        // 2. Source tile log-prob
        if let Ok(source_idx) = mv.source_idx() {
            let spatial_idx = coord_to_flat_index(source_idx, map_size);
            if spatial_idx < source_logs.len() {
                log_prob += source_logs[spatial_idx];
            }
        }

        // 3. Target tile log-prob
        if let Ok(target_idx) = mv.target_idx() {
            let spatial_idx = coord_to_flat_index(target_idx, map_size);
            if spatial_idx < target_logs.len() {
                log_prob += target_logs[spatial_idx];
            }
        }

        // 4. Move option log-prob
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
            MoveType::Reward => mv
                .reward_type()
                .ok()
                .and_then(DecomposedMapper::map_reward),
            _ => None,
        };

        if let Some(idx) = option_idx {
            if idx < option_logs.len() {
                log_prob += option_logs[idx];
            } else {
                log_prob += -23.0; // Very small probability
            }
        }

        move_logs.push(log_prob);
    }

    move_logs
}

/// Log-Softmax over 1D tensor
fn log_softmax_1d(tensor: &Tensor) -> Result<Vec<f32>, candle_core::Error> {
    let logits = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok(log_softmax(&logits))
}

/// Log-Softmax over a plain logit slice (device-free counterpart of
/// `log_softmax_1d`, for `RawPolicyOutput` rows).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }

    // Find max for numerical stability
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute log(sum(exp(xi - max)))
    let log_sum_exp = logits
        .iter()
        .map(|&l| (l - max_logit).exp())
        .sum::<f32>()
        .ln();

    // xi - max - log_sum_exp
    logits
        .iter()
        .map(|&l| l - max_logit - log_sum_exp)
        .collect()
}

/// Softmax over a plain logit slice (device-free counterpart of
/// `softmax_1d`, for `RawPolicyOutput` rows).
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_values: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let exp_sum: f32 = exp_values.iter().sum();
    exp_values
        .into_iter()
        .map(|v| (v / exp_sum).max(1e-12))
        .collect()
}

use crate::ai::features;

/// Convert 2D coords to flat index
fn coord_to_flat_index(idx: usize, map_size: usize) -> usize {
    let y = idx / map_size;
    let x = idx % map_size;
    (y * features::MAP_SIZE + x).min((features::MAP_SIZE * features::MAP_SIZE) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::EndTurnMove;
    use candle_core::{DType, Device};

    /// Build a batch-of-2 synthetic `PolicyOutput` with distinct, non-uniform
    /// logits per row so a row-extraction bug (e.g. wrong stride) would show
    /// up as a mismatch rather than accidentally cancelling out.
    fn synthetic_policy_output() -> PolicyOutput {
        let device = Device::Cpu;
        let spatial = features::MAP_SIZE * features::MAP_SIZE;
        let action_type = Tensor::arange(0f32, (2 * 11) as f32, &device)
            .unwrap()
            .reshape((2, 11))
            .unwrap();
        let source_spatial = Tensor::arange(0f32, (2 * spatial) as f32, &device)
            .unwrap()
            .reshape((2, spatial))
            .unwrap();
        let target_spatial = Tensor::arange(0f32, (2 * spatial) as f32, &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .affine(0.5, 1.0)
            .unwrap()
            .reshape((2, spatial))
            .unwrap();
        let move_option = Tensor::arange(0f32, (2 * 192) as f32, &device)
            .unwrap()
            .affine(-0.25, 3.0)
            .unwrap()
            .reshape((2, 192))
            .unwrap();
        PolicyOutput {
            action_type,
            source_spatial,
            target_spatial,
            move_option,
        }
    }

    #[test]
    fn raw_priors_match_tensor_priors_per_row() {
        let policy = synthetic_policy_output();
        let raw_rows = policy.to_raw_rows(None).unwrap();
        assert_eq!(raw_rows.len(), 2);

        let legal_moves: Vec<Box<dyn Move>> = vec![Box::new(EndTurnMove)];
        let map_size = features::MAP_SIZE;

        for row in 0..2 {
            let row_policy = PolicyOutput {
                action_type: policy.action_type.get(row).unwrap().unsqueeze(0).unwrap(),
                source_spatial: policy
                    .source_spatial
                    .get(row)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap(),
                target_spatial: policy
                    .target_spatial
                    .get(row)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap(),
                move_option: policy.move_option.get(row).unwrap().unsqueeze(0).unwrap(),
            };

            let tensor_priors = compute_move_priors(&row_policy, &legal_moves, map_size, true);
            let raw_priors =
                compute_move_priors_raw(&raw_rows[row], &legal_moves, map_size, true);
            assert_eq!(tensor_priors, raw_priors, "priors mismatch at row {row}");

            let tensor_logs = compute_move_log_probs(&row_policy, &legal_moves, map_size);
            let raw_logs = compute_move_log_probs_raw(&raw_rows[row], &legal_moves, map_size);
            assert_eq!(tensor_logs.len(), raw_logs.len());
            for (t, r) in tensor_logs.iter().zip(raw_logs.iter()) {
                assert!((t - r).abs() < 1e-5, "log-prob mismatch at row {row}: {t} vs {r}");
            }
        }
    }
}

/// Softmax over 1D tensor
fn softmax_1d(tensor: &Tensor) -> Result<Vec<f32>, candle_core::Error> {
    let logits = tensor.flatten_all()?.to_vec1::<f32>()?;
    Ok(softmax(&logits))
}
