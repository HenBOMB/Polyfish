use super::super::super::super::super::states::{GameState, PlayerId};
use super::super::super::super::oracle_macro::{GoalAux, LaneState, MacroGoal};
use super::super::super::macro_exec::TurnCounters;
use super::super::config::MacroParams;

/// EdgeShaper is used to shape the edge of the macro search tree.
pub(super) struct EdgeShaper {
    /// The pre-move potential for the edge.
    pre: Option<(f32, GoalAux)>,
    /// The weight of the edge.
    w: f32,
}

impl EdgeShaper {
    /// Starts the edge shaper.
    pub(super) fn start(
        params: &MacroParams,
        pre_state: &GameState,
        player: PlayerId,
        goal: &MacroGoal,
        counters: &TurnCounters,
        lane_states: &[LaneState; 2],
    ) -> Self {
        EdgeShaper { pre: None, w: 0.0 }
    }

    /// Finishes the edge shaper. Post - pre, w·Δ
    pub(super) fn finish(self, post_state: &GameState) -> f32 {
        0.0
    }
}
