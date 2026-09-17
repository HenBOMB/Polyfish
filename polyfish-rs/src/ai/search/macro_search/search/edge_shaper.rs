use super::super::super::super::super::game::Game;
use super::super::super::super::super::states::{GameState, PlayerId};
use super::super::super::super::oracle_macro::{GoalAux, LaneState, MacroGoal};
use super::super::super::macro_exec::TurnCounters;
use super::super::belief::BranchBelief;
use super::super::config::MacroParams;
use crate::ai::reward::goal_potential;
use crate::utils::converter::player_id_to_usize;

/// EdgeShaper is used to shape the edge of the macro search tree.
pub(super) struct EdgeShaper {
    /// The pre-move potential for the edge.
    pub pre: Option<(f32, GoalAux)>,
    /// The weight of the edge.
    pub w: f32,
    /// The pov of the edge.
    pov: PlayerId,
    // The goal of the edge.
    goal: MacroGoal,
}

pub(super) struct EdgeShaperSnapshot {
    pub game: Game,
    pub player: PlayerId,
    pub counters: [TurnCounters; 2],
    pub lane_states: [LaneState; 2],
    pub goal: MacroGoal,
    pub parent_from: Option<(PlayerId, MacroGoal)>,
    pub branch_belief: Option<BranchBelief>,
}

impl EdgeShaper {
    /// Starts the edge shaper.
    pub(super) fn start(
        params: &MacroParams,
        snapshot: &EdgeShaperSnapshot,
        pov: PlayerId,
    ) -> Self {
        // EXP_ELO_036b: pre-move potential for this edge's directive via GoalAux diff
        let seat_idx = player_id_to_usize(snapshot.player);
        let shape_pre = if params.shape_w != 0.0 && snapshot.player == pov {
            // Compute the GoalAux for the edge's directive.
            let aux = crate::ai::oracle_macro::compute_goal_aux(
                &snapshot.game.state,
                snapshot.player,
                &snapshot.goal,
                snapshot.counters[seat_idx].techs_bought,
                snapshot.counters[seat_idx].tier3_bought,
                Some(&snapshot.lane_states[seat_idx]),
            );
            Some((
                goal_potential(
                    &snapshot.game.state,
                    snapshot.player,
                    &snapshot.goal,
                    Some(&aux),
                ),
                aux,
            ))
        } else {
            None
        };

        EdgeShaper {
            pre: shape_pre,
            w: params.shape_w,
            pov,
            goal: snapshot.goal.clone(),
        }
    }

    /// Finishes the edge shaper. (Post - pre) * w
    pub(super) fn finish(self, post_state: &GameState) -> f32 {
        match self.pre {
            Some((pre, aux)) => {
                let post = goal_potential(post_state, self.pov, &self.goal, Some(&aux));
                self.w * (post - pre)
            }
            None => 0.0,
        }
    }
}
