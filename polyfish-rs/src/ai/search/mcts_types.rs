#[derive(Debug, Clone)]
pub struct MoveVisit {
    pub move_type: crate::types::MoveType,
    pub visits: f32,
    pub source_idx: Option<usize>,
    pub target_idx: Option<usize>,
    pub structure_type: Option<crate::types::StructureType>,
    pub unit_type: Option<crate::types::UnitType>,
    pub tech_type: Option<crate::types::TechnologyType>,
    pub ability_type: Option<crate::types::AbilityType>,
    pub reward_type: Option<crate::types::CityRewardType>,
}

impl MoveVisit {
    /// A single executed move as a unit-mass policy target — the
    /// behavior-cloning shape emitted by deterministic (searchless-policy)
    /// generators like macro-mcts (Stage 3 data generation).
    pub fn one_hot(m: &dyn crate::moves::Move) -> Self {
        Self::weighted(m, 1.0)
    }

    /// Same shape as [`Self::one_hot`], but with a real search-derived
    /// weight instead of a hardcoded unit mass — e.g. micro-mcts's own
    /// post-search visit count per root candidate (EXP_ELO_155: distilling
    /// the search's actual distribution instead of collapsing it to a
    /// single winner).
    pub fn weighted(m: &dyn crate::moves::Move, visits: f32) -> Self {
        MoveVisit {
            move_type: m.move_type(),
            visits,
            source_idx: m.source_idx().ok(),
            target_idx: m.target_idx().ok(),
            structure_type: m.structure_type().ok(),
            unit_type: m.unit_type().ok(),
            tech_type: m.tech_type().ok(),
            ability_type: m.ability_type().ok(),
            reward_type: m.reward_type().ok(),
        }
    }
}
