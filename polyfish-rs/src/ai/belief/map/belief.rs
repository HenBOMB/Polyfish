//! The [`MapBelief`] type itself: what it holds, how it is constructed, and
//! every read a consumer is offered. The derivations live in `capital.rs`
//! and `villages.rs`; this file is the surface.

use crate::states::{GameState, PlayerId};

use super::cache::BeliefKey;
use super::ctx::Ctx;

/// Which placement rules the legality mask uses. Exists so Stage 1b's three
/// fidelity fixes can be measured on their own, holding C1/C2/C3 constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fidelity {
    /// The generator's actual rules.
    #[default]
    Generator,
    /// The pre-SSOT guesser's three known-wrong rules: the Ocean-cardinal veto
    /// as a hard filter, no Mountain exclusion, and the resource zone at
    /// radius 1 where the generator's is 2.
    LegacyBugs,
}

/// Why a tile carries the mass it does — telemetry, UI overlay, and the only
/// honest way to debug a probabilistic derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    /// Generator legality only.
    Prior,
    /// C1: forced by the explored, empty, legal tile at this index.
    Packing(i32),
    /// C2: forced by the orphan resource at this index.
    ResourceZone(i32),
    /// A real village the observer has explored.
    Sighted,
}

/// What: the single source of truth for everything the observer believes about
/// the map it cannot see — where undiscovered villages are, whose region a tile
/// belongs to, and where the enemy capital sits.
/// How: a pure derivation from the observer's explored set against the map
/// generator's own placement rules. No history, no persistence.
#[derive(Debug, Clone)]
pub struct MapBelief {
    pub observer: PlayerId,
    pub opponent: PlayerId,
    pub(super) size: i32,
    pub(super) key: BeliefKey,

    /// P(a village site sits here). Sighted villages are 1.0, explored empty
    /// tiles 0.0, everything else the reconciled posterior.
    pub(super) village: Vec<f32>,
    /// P(this tile's generator affinity region is the opponent's).
    pub(super) affinity: Vec<f32>,
    /// Opponent-capital posterior, dense (0.0 off-support). Sums to 1.
    pub(super) capital: Vec<f32>,
    pub(super) capital_confirmed: Option<i32>,

    pub(super) why: Vec<(i32, Evidence)>,
}

/// Out-of-range reads answer 0.0 and negative ones clamp to tile 0, rather than
/// panicking: consumers index by tile and a stale index must not kill a search.
fn at(grid: &[f32], idx: i32) -> f32 {
    grid.get(idx.max(0) as usize).copied().unwrap_or(0.0)
}

impl MapBelief {
    /// The ONLY constructor. Pure: same explored set in, same belief out.
    pub fn observe(state: &GameState, observer: PlayerId) -> MapBelief {
        Self::observe_with(state, observer, Fidelity::Generator)
    }

    /// As [`observe`](Self::observe), with the placement rules selectable. Only the
    /// calibration harness passes anything but [`Fidelity::Generator`].
    pub fn observe_with(
        state: &GameState,
        observer: PlayerId,
        fidelity: Fidelity,
    ) -> MapBelief {
        let size = state.settings.size;
        let n = (size * size).max(0) as usize;
        let opponent = state
            .tribes
            .keys()
            .copied()
            .find(|&p| p != observer)
            .unwrap_or(observer);

        let mut belief = MapBelief {
            observer,
            opponent,
            size,
            key: Self::key_of(state, observer),
            village: vec![0.0; n],
            affinity: vec![0.0; n],
            capital: vec![0.0; n],
            capital_confirmed: None,
            why: Vec::new(),
        };
        if n == 0 {
            return belief;
        }

        let ctx = Ctx::new(state, observer, opponent, fidelity);

        // Village existence needs no opponent — it is pure generator geometry.
        // Only the capital posterior and the affinity field do, so a solo state
        // (synthetic fixtures, or every rival eliminated) still gets a full
        // village grid with those two left empty.
        if ctx.has_opponent {
            // The capital posterior comes first: the legality mask's mountain
            // term is conditioned on the affinity field it generates.
            belief.solve_capital(&ctx);
            belief.fill_affinity(&ctx);
        }
        let legality = belief.legality_mask(&ctx);
        belief.solve_villages(&ctx, &legality);
        belief
    }

    pub fn p_village(&self, idx: i32) -> f32 {
        at(&self.village, idx)
    }

    pub fn p_opponent_affinity(&self, idx: i32) -> f32 {
        at(&self.affinity, idx)
    }

    pub fn p_capital(&self, idx: i32) -> f32 {
        at(&self.capital, idx)
    }

    /// Confirmed sighting if any, else the posterior argmax.
    pub fn capital_map(&self) -> Option<i32> {
        if self.capital_confirmed.is_some() {
            return self.capital_confirmed;
        }
        self.capital
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as i32)
    }

    /// Mass on the MAP cell — preserves `BeliefState::capital_confidence` semantics.
    pub fn capital_confidence(&self) -> f32 {
        self.capital.iter().copied().fold(0.0, f32::max)
    }

    /// Posterior cells, most probable first.
    pub fn capital_top(&self, n: usize) -> Vec<(i32, f32)> {
        let mut v: Vec<(i32, f32)> = self
            .capital
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 1e-9)
            .map(|(i, p)| (i as i32, *p))
            .collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v
    }

    /// Live (non-eliminated) capital hypotheses.
    pub fn capital_live(&self) -> usize {
        self.capital.iter().filter(|p| **p > 1e-6).count()
    }

    pub fn evidence_at(&self, idx: i32) -> Option<Evidence> {
        self.why.iter().find(|(i, _)| *i == idx).map(|(_, e)| *e)
    }

    pub fn key(&self) -> BeliefKey {
        self.key
    }
}
