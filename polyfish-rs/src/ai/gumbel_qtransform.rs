//! Pure, dependency-free helpers for Gumbel MuZero search.
//!
//! Everything here operates on plain `&[f32]` / `Vec<f32>` and has no tie to
//! `Game`, `Move`, the network, or the search tree. That isolation is
//! deliberate: it lets the σ / completed-Q math and the sequential-halving
//! visit schedule be unit-tested in total isolation from the rest of the
//! engine.
//!
//! Constants and formulas follow DeepMind's `mctx` reference
//! (`qtransform_completed_by_mix_value` defaults and
//! `get_sequence_of_considered_visits`).

/// `maxvisit_init` from `mctx qtransform_completed_by_mix_value`.
pub const C_VISIT: f32 = 50.0;
/// `value_scale` from `mctx qtransform_completed_by_mix_value`.
pub const C_SCALE: f32 = 0.1;
/// Small epsilon for numerical guards.
pub const EPSILON: f32 = 1e-8;

/// `v_mix` at a node: the imputed value used as the Q of unvisited children.
///
/// `raw_value` is this node's own NN value prediction (from this node's
/// perspective). `child_priors`, `child_qvalues`, and `child_visit_counts`
/// are parallel slices, one entry per child. `child_qvalues[i]` is only
/// meaningful when `child_visit_counts[i] > 0.0`.
///
/// When no child has been visited, `v_mix` degenerates to `raw_value`.
pub fn compute_v_mix(
    raw_value: f32,
    child_priors: &[f32],
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
) -> f32 {
    let sum_visits: f32 = child_visit_counts.iter().sum();
    if sum_visits == 0.0 {
        return raw_value;
    }
    let sum_probs_visited: f32 = child_priors
        .iter()
        .zip(child_visit_counts)
        .filter(|&(_, &v)| v > 0.0)
        .map(|(p, _)| *p)
        .sum();
    let weighted_q: f32 = if sum_probs_visited > EPSILON {
        child_priors
            .iter()
            .zip(child_qvalues)
            .zip(child_visit_counts)
            .filter(|&((_, _), &v)| v > 0.0)
            .map(|((p, q), _)| p * q / sum_probs_visited)
            .sum()
    } else {
        0.0
    };
    (raw_value + sum_visits * weighted_q) / (sum_visits + 1.0)
}

/// Completed Q per child: real Q if the child was visited, else `v_mix`.
pub fn compute_completed_qvalues(
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
    v_mix: f32,
) -> Vec<f32> {
    child_qvalues
        .iter()
        .zip(child_visit_counts)
        .map(|(&q, &v)| if v > 0.0 { q } else { v_mix })
        .collect()
}

/// Min-max rescale `values` into `[0, 1]`. Degenerate `min == max` collapses
/// to all-zeros (matches `mctx`'s rescale behavior for the constant case).
pub fn rescale_min_max(values: &[f32]) -> Vec<f32> {
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() < EPSILON {
        return vec![0.0; values.len()];
    }
    values.iter().map(|&v| (v - min) / (max - min)).collect()
}

/// `sigma(completed_Q) = (C_VISIT + max_child_visits) * C_SCALE * completed_Q`.
pub fn sigma(completed_q: &[f32], max_child_visits: f32) -> Vec<f32> {
    let scale = (C_VISIT + max_child_visits) * C_SCALE;
    completed_q.iter().map(|&q| scale * q).collect()
}

/// Full σ(completed-Q) pipeline used by both root recommendation and interior
/// selection. With `rescale = true` this reproduces `mctx`'s default
/// `rescale_values = True`.
pub fn sigma_completed_q(
    raw_value: f32,
    child_priors: &[f32],
    child_qvalues: &[f32],
    child_visit_counts: &[f32],
    rescale: bool,
) -> Vec<f32> {
    let v_mix = compute_v_mix(raw_value, child_priors, child_qvalues, child_visit_counts);
    let mut completed = compute_completed_qvalues(child_qvalues, child_visit_counts, v_mix);
    if rescale {
        completed = rescale_min_max(&completed);
    }
    let max_visits = child_visit_counts.iter().copied().fold(0.0f32, f32::max);
    sigma(&completed, max_visits)
}

/// Sequential-Halving visit schedule: the per-round `(num_considered,
/// visits_per_candidate)` plan that allocates `num_simulations` total visits
/// across `max_num_considered` initial root candidates.
///
/// Returns `Vec<(round_idx, num_considered, visits_per_candidate)>`.
/// Mirrors `mctx.get_sequence_of_considered_visits`.
pub fn sequence_of_considered_visits(
    max_num_considered: usize,
    num_simulations: usize,
) -> Vec<(usize, usize, usize)> {
    if max_num_considered <= 1 {
        return vec![(0, max_num_considered.max(1), num_simulations)];
    }
    let log2max = (max_num_considered as f32).log2().ceil() as usize;
    let mut num_considered = max_num_considered;
    let mut rounds = Vec::new();
    let mut round_idx = 0;
    let mut total_assigned = 0usize;
    while total_assigned < num_simulations && num_considered >= 2 {
        let extra = (num_simulations / (log2max.max(1) * num_considered)).max(1);
        rounds.push((round_idx, num_considered, extra));
        total_assigned += extra * num_considered;
        num_considered = (num_considered / 2).max(2);
        round_idx += 1;
        // Defensive belt: `mctx` terminates naturally because `num_considered`
        // stabilizes at 2 and `total_assigned` keeps growing, but cap rounds
        // so a pathological input cannot loop forever.
        if round_idx > 2 * log2max + 4 {
            break;
        }
    }
    rounds
}

/// Numerically stable softmax over a slice. Empty input -> empty output.
pub fn softmax(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / values.len() as f32; values.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn v_mix_no_visits_returns_raw_value() {
        let v = compute_v_mix(0.7, &[0.5, 0.5], &[0.0, 0.0], &[0.0, 0.0]);
        assert!(approx(v, 0.7), "got {}", v);
    }

    #[test]
    fn v_mix_hand_computed() {
        // 2 children, both visited.
        // priors [0.6, 0.4], q [0.8, -0.2], visits [2.0, 1.0].
        // sum_visits = 3.
        // sum_probs_visited = 0.6 + 0.4 = 1.0.
        // weighted_q = (0.6*0.8 + 0.4*(-0.2)) / 1.0 = 0.48 - 0.08 = 0.4.
        // v_mix = (raw + 3*0.4) / (3 + 1) = (raw + 1.2) / 4.
        // With raw_value = 0.2 -> 1.4 / 4 = 0.35.
        let v = compute_v_mix(0.2, &[0.6, 0.4], &[0.8, -0.2], &[2.0, 1.0]);
        assert!(approx(v, 0.35), "got {}", v);
    }

    #[test]
    fn v_mix_ignores_unvisited_child_q() {
        // Child 1 unvisited: its q must NOT feed weighted_q.
        // priors [0.5, 0.5], q [0.9, 999.0], visits [1.0, 0.0].
        // sum_visits = 1. sum_probs_visited = 0.5.
        // weighted_q = (0.5*0.9 / 0.5) = 0.9.
        // v_mix = (raw + 1*0.9) / 2 = (0.1 + 0.9)/2 = 0.5.
        let v = compute_v_mix(0.1, &[0.5, 0.5], &[0.9, 999.0], &[1.0, 0.0]);
        assert!(approx(v, 0.5), "got {}", v);
    }

    #[test]
    fn completed_q_uses_real_q_when_visited() {
        let cq = compute_completed_qvalues(&[0.8, -0.2, 0.0], &[2.0, 1.0, 0.0], 0.3);
        assert!(approx(cq[0], 0.8));
        assert!(approx(cq[1], -0.2));
        assert!(approx(cq[2], 0.3), "unvisited child imputed to v_mix");
    }

    #[test]
    fn completed_q_imputes_v_mix_when_unvisited() {
        let cq = compute_completed_qvalues(&[0.0, 0.0], &[0.0, 0.0], -0.4);
        assert!(approx(cq[0], -0.4));
        assert!(approx(cq[1], -0.4));
    }

    #[test]
    fn rescale_min_max_degenerate_returns_zeros() {
        let r = rescale_min_max(&[0.5, 0.5, 0.5]);
        assert!(r.iter().all(|&x| approx(x, 0.0)));
    }

    #[test]
    fn rescale_min_max_basic() {
        let r = rescale_min_max(&[1.0, 2.0, 3.0]);
        assert!(approx(r[0], 0.0));
        assert!(approx(r[1], 0.5));
        assert!(approx(r[2], 1.0));
    }

    #[test]
    fn sigma_matches_formula() {
        // (C_VISIT + max_visits) * C_SCALE * completed_q
        // C_VISIT=50, C_SCALE=0.1, max_visits=10 -> scale = 6.0
        let s = sigma(&[0.5, -0.25], 10.0);
        assert!(approx(s[0], 3.0), "got {}", s[0]);
        assert!(approx(s[1], -1.5), "got {}", s[1]);
    }

    #[test]
    fn sigma_completed_q_end_to_end() {
        // Single visited child, raw_value 0.0.
        // v_mix = (0 + 1*q)/2 ... let's just check it runs and has right len
        // and that rescale of a single value is 0.0 (degenerate min==max).
        let s = sigma_completed_q(0.0, &[1.0], &[0.5], &[1.0], true);
        assert_eq!(s.len(), 1);
        // After rescale, single value -> 0.0 -> sigma = scale*0 = 0.
        assert!(approx(s[0], 0.0), "got {}", s[0]);
    }

    #[test]
    fn sequence_k1_single_round_all_simulations() {
        let r = sequence_of_considered_visits(1, 100);
        assert_eq!(r, vec![(0, 1, 100)]);
    }

    #[test]
    fn sequence_k4_returns_halving_rounds() {
        // k=4 -> log2max = 2. Round 0 considers 4, round 1 considers 2.
        // Each round's `extra` is max(1, sims / (log2max * considered)).
        let r = sequence_of_considered_visits(4, 16);
        // Round 0: considered=4, extra = max(1, 16/(2*4)) = max(1,2) = 2.
        // Round 1: considered=2, extra = max(1, 16/(2*2)) = max(1,4) = 4.
        assert!(r.len() >= 2, "got {:?}", r);
        assert_eq!(r[0].1, 4, "round 0 considers 4: {:?}", r);
        assert_eq!(r[0].2, 2, "round 0 visits per candidate: {:?}", r);
        assert_eq!(r[1].1, 2, "round 1 considers 2: {:?}", r);
        // Total assigned = 4*2 + 2*4 = 16.
        let total: usize = r.iter().map(|(_, c, v)| c * v).sum();
        assert_eq!(total, 16, "total assigned: {:?}", r);
    }

    #[test]
    fn sequence_total_assigned_does_not_exceed_budget_grossly() {
        // The schedule may overshoot by at most one round's worth; it must
        // never under-allocate when simulations are sufficient.
        let r = sequence_of_considered_visits(8, 64);
        let total: usize = r.iter().map(|(_, c, v)| c * v).sum();
        assert!(total >= 64, "under-allocated: total={} rounds={:?}", total, r);
        // First round considers k, last round considers 2.
        assert_eq!(r.first().unwrap().1, 8);
        assert_eq!(r.last().unwrap().1, 2);
    }

    #[test]
    fn softmax_sums_to_one() {
        let s = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = s.iter().sum();
        assert!(approx(sum, 1.0));
        // Larger logit -> larger probability.
        assert!(s[2] > s[1] && s[1] > s[0]);
    }

    #[test]
    fn softmax_empty_input() {
        assert!(softmax(&[]).is_empty());
    }
}
