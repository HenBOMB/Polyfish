//! C4 → C3 → elimination: the opponent-capital posterior, and the per-tile
//! affinity field it induces. Nothing here touches village existence.

use crate::ai::belief::{capital_support_by_quad, quad_of};

use super::belief::MapBelief;
use super::ctx::Ctx;

impl MapBelief {
    /// C4 prior → C3 climate likelihood → explored-cell elimination, or a hard
    /// collapse if the capital has actually been sighted.
    pub(super) fn solve_capital(&mut self, ctx: &Ctx) {
        if let Some(seen) = ctx.sighted_capital {
            self.capital_confirmed = Some(seen);
            if let Some(slot) = self.capital.get_mut(seen.max(0) as usize) {
                *slot = 1.0;
            }
            return;
        }

        let support = capital_support_by_quad(self.size, ctx.player_count);
        let own_quad = ctx.own_capital.and_then(|c| quad_of(c, &support));
        let cells: Vec<i32> = support
            .iter()
            .enumerate()
            .filter(|(q, _)| Some(*q) != own_quad)
            .flat_map(|(_, c)| c.iter().copied())
            // Elimination: an explored support cell that holds no capital
            // cannot be one. This is the {0,1} special case of C3 and
            // reproduces `BeliefState::on_explored` exactly.
            .filter(|&c| !ctx.explored(c))
            .collect();
        if cells.is_empty() {
            return;
        }

        // C3: reweight the surviving hypotheses by how well each explains the
        // climate actually observed on explored land tiles.
        let mut log_w: Vec<f32> = cells
            .iter()
            .map(|&k| ctx.climate_log_likelihood(k))
            .collect();
        let max = log_w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            log_w.iter_mut().for_each(|w| *w = 0.0);
        }
        let mut total = 0.0f32;
        let mut w: Vec<f32> = log_w
            .iter()
            .map(|&l| {
                let v = (l - max).exp();
                total += v;
                v
            })
            .collect();
        if total <= 0.0 || !total.is_finite() {
            let u = 1.0 / cells.len() as f32;
            w = vec![u; cells.len()];
            total = 1.0;
        }
        for (&c, wi) in cells.iter().zip(w.iter()) {
            if let Some(slot) = self.capital.get_mut(c.max(0) as usize) {
                *slot = wi / total;
            }
        }
    }

    /// P(tile is in the opponent's generator region), for EVERY tile. Explored
    /// tiles read their (known) climate directly; fog tiles take the mixture
    /// over the capital posterior.
    pub(super) fn fill_affinity(&mut self, ctx: &Ctx) {
        let n = self.affinity.len();
        for idx in 0..n as i32 {
            if ctx.explored(idx) {
                if let Some(c) = ctx.climate_of(idx) {
                    self.affinity[idx as usize] = if c == ctx.opp_climate { 1.0 } else { 0.0 };
                    continue;
                }
            }
            let mut acc = 0.0;
            let mut mass = 0.0;
            for (k, p) in self.capital.iter().enumerate() {
                if *p <= 0.0 {
                    continue;
                }
                acc += p * ctx.p_opponent_climate(idx, k as i32);
                mass += p;
            }
            self.affinity[idx as usize] = if mass > 0.0 { acc / mass } else { 0.0 };
        }
    }
}
