use super::belief::BranchBelief;
use crate::ai::macro_agent::enumerate_candidates_with_belief;
use crate::ai::macro_exec::TurnCounters;
use crate::ai::oracle_macro::{LaneState, MacroGoal, OrderKind, compute_macro_goal};
use crate::ai::search::macro_mcts::{TURN_DEPTH_CAP, fog_order_dead, terminal_value};
use crate::ai::search::mcts_common::VIRTUAL_LOSS;
use crate::game::Game;
use crate::states::{GameState, PlayerId};
use crate::utils::converter::player_id_to_usize;

/// UCT exploration constant, tuned so a 0.03 q01 gap is decisive (~10 visits); ties split evenly.
pub(crate) const UCT_EXPLORATION_CONSTANT: f32 = 0.05;

/// A node represents a turn boundary in the tree.
pub(super) struct Node {
    /// The game state.
    game: Game,
    /// The player whose turn it is.
    player: PlayerId,
    /// The counters for the two players.
    counters: [TurnCounters; 2],
    /// The lane states of the two players.
    lane_states: [LaneState; 2],
    /// The candidate macro goals for this node.
    candidates: Vec<MacroGoal>,
    /// The indices of the children of this node.
    children: Vec<Option<usize>>,
    /// The number of visits to each child.
    edge_visits: Vec<f32>,
    /// The values of each child.
    edge_values: Vec<f32>,
    /// EXP_ELO_036b: potential-based edge reward w·(φ(s',g)−φ(s,g)) from the
    /// EDGE OWNER's perspective; nonzero only on the root player's edges.
    edge_shape: Vec<f32>,
    /// Number of visits to this node.
    visits: f32,
    /// Terminal/game over or depth-capped leaf value from this player's perspective.
    frozen_value: Option<f32>,
    /// Edge that produced this node: (player, directive) or None at root.
    from: Option<(PlayerId, MacroGoal)>,
    /// Root-only prior over candidates from the macro policy head.
    edge_prior: Vec<f32>,
    /// Cached rollout_value for frozen edge; None if edge has child or before frozen.
    edge_frozen: Vec<Option<f32>>,
    /// Per-edge virtual loss charged during wave-batching, removed after backup.
    edge_virtual_loss: Vec<f32>,
    /// Sum of edge_virtual_loss, mirrors visits for UCT sqrt/exploration term.
    virtual_visits: f32,
    /// Synthetic, FOW-honest world state for branch-local materialization.
    branch_belief: Option<BranchBelief>,
}

/// Node implementation for the macro search tree.
impl Node {
    /// Create a new node for the given game state, player, and counters.
    pub(super) fn new(
        game: Game,
        player: PlayerId,
        counters: [TurnCounters; 2],
        mut lane_states: [LaneState; 2],
        root_turn: i32,
        k: usize,
        from: Option<(PlayerId, MacroGoal)>,
        leaf_fn: &dyn Fn(&crate::states::GameState, PlayerId, u32) -> f32,
        own_last_goal: Option<&MacroGoal>,
        branch_belief: Option<BranchBelief>,
    ) -> Self {
        let frozen_value = if game.state.settings._game_over {
            Some(terminal_value(&game.state, player))
        } else if game.state.settings.turn - root_turn >= TURN_DEPTH_CAP {
            Some(leaf_fn(
                &game.state,
                player,
                counters[player_id_to_usize(player)].tier3_bought,
            ))
        } else {
            None
        };

        let mut candidates = Vec::new();

        if frozen_value.is_none() {
            recompute_playstyle_goal(&game, player, &mut lane_states);

            let seat_idx = player_id_to_usize(player);
            let tier3_bought = counters[seat_idx].tier3_bought;
            let macro_goal = compute_macro_goal(&game.state, player, tier3_bought);
            let belief = branch_belief
                .as_ref()
                .and_then(|b| b.candidate_belief_for(player));
            let mut cands: Vec<MacroGoal> = enumerate_candidates_with_belief(
                &game.state,
                player,
                macro_goal,
                counters[seat_idx],
                k,
                belief,
            )
            .into_iter()
            .map(|(g, _)| g)
            .collect();

            if let Some(last_g) = own_last_goal {
                // Filter out fogged expand orders.
                let mut cand = last_g.clone();
                cand.orders.retain(|(kind, t)| {
                    *kind != OrderKind::Expand || !fog_order_dead(&game.state, *t, player)
                });
                cand.orders.sort();

                if !cands.contains(&cand) {
                    cands.push(cand);
                }
            }
            candidates = cands;
        };

        let n = candidates.len();
        Node {
            game,
            player,
            counters,
            lane_states,
            candidates,
            children: vec![None; n],
            edge_visits: vec![0.0; n],
            edge_values: vec![0.0; n],
            edge_shape: vec![0.0; n],
            visits: 0.0,
            frozen_value,
            from,
            edge_prior: Vec::new(),
            edge_frozen: vec![None; n],
            edge_virtual_loss: vec![0.0; n],
            virtual_visits: 0.0,
            branch_belief: None,
        }
    }
    /// UCT: cold start ignores prior; picks unvisited edge in base order.
    /// After all visited, adds PUCT-style prior/(1+n) bonus to UCT score.
    /// Only here does root prior act; ensures unbiased cold start.
    pub(super) fn select_edge(&self) -> usize {
        // Cold start UCT selection rule: Finds the first unvisited edge.
        let unvisited_edge_idx = (0..self.edge_visits.len())
            .find(|&i| self.edge_visits[i] + self.edge_virtual_loss[i] == 0.0);
        if let Some(unvisited_edge_idx) = unvisited_edge_idx {
            return unvisited_edge_idx;
        }

        // After all visited, adds PUCT-style prior/(1+n) bonus to UCT score.
        let effective_n = self.visits + self.virtual_visits;
        let ln_n = effective_n.max(1.0).ln();
        let sqrt_n = effective_n.max(1.0).sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for i in 0..self.candidates.len() {
            // Pending (unbacked) visits are scored as losses (value -1).
            // Zero virtual loss reduces to plain UCT.
            let ev = self.edge_visits[i] + self.edge_virtual_loss[i];
            let q_val = ((self.edge_values[i] - self.edge_virtual_loss[i]) / ev + 1.0) / 2.0;
            let mut score = q_val + UCT_EXPLORATION_CONSTANT * (ln_n / ev).sqrt();
            if let Some(&p) = self.edge_prior.get(i) {
                score += p * sqrt_n / (1.0 + ev);
            }
            if score > best_score {
                best_score = score;
                best = i;
            }
        }
        best
    }

    /// Calculate Q values for each edge, excluding edges with no visits.
    pub(super) fn edge_q_values(&self) -> Vec<Option<f32>> {
        self.edge_visits
            .iter()
            .enumerate()
            .map(|(i, visit)| (*visit > 0.0).then_some(self.edge_values[i] / visit))
            .collect()
    }

    /// Charge virtual loss to an edge.
    pub(super) fn charge_virtual_loss(&mut self, edge: usize) {
        self.edge_virtual_loss[edge] += VIRTUAL_LOSS;
        self.virtual_visits += VIRTUAL_LOSS;
    }

    /// Converts the value from the child's POV to this node's POV and records it.
    pub(super) fn backup(&mut self, edge: usize, mut value: f32, times: f32) -> f32 {
        value = self.edge_shape[edge] - value;

        // Update node statistics.
        self.visits += times;
        self.edge_visits[edge] += times;
        self.edge_values[edge] += value * times;
        self.edge_virtual_loss[edge] -= VIRTUAL_LOSS * times;
        self.virtual_visits -= VIRTUAL_LOSS * times;

        value
    }

    pub(super) fn tier_3_bought(&self) -> u32 {
        let seat_idx = player_id_to_usize(self.player);
        self.counters[seat_idx].tier3_bought
    }

    pub(super) fn game_state(&self) -> &GameState {
        &self.game.state
    }

    pub(super) fn player(&self) -> PlayerId {
        self.player
    }

    pub(super) fn counters(&self) -> &[TurnCounters; 2] {
        &self.counters
    }

    pub(super) fn game(&self) -> &Game {
        &self.game
    }

    pub(super) fn lane_states(&self) -> &[LaneState; 2] {
        &self.lane_states
    }

    pub(super) fn candidates(&self) -> &[MacroGoal] {
        &self.candidates
    }

    pub(super) fn frozen_value(&self) -> Option<f32> {
        self.frozen_value
    }

    pub(super) fn from(&self) -> Option<(PlayerId, &MacroGoal)> {
        self.from.as_ref().map(|(p, g)| (*p, g))
    }

    pub(super) fn candidate_for_edge(&self, edge: usize) -> &MacroGoal {
        &self.candidates[edge]
    }

    pub(super) fn branch_belief(&self) -> Option<&BranchBelief> {
        self.branch_belief.as_ref()
    }

    /// Get the index of the child by edge.
    pub(super) fn get_child_by_edge(&self, edge: usize) -> Option<usize> {
        self.children[edge]
    }

    pub(super) fn get_edge_frozen(&self, edge: usize) -> Option<f32> {
        self.edge_frozen[edge]
    }

    pub(super) fn set_edge_frozen(&mut self, edge: usize, value: f32) {
        self.edge_frozen[edge] = Some(value);
    }

    pub(super) fn set_child_by_edge(&mut self, edge: usize, child: Option<usize>) {
        self.children[edge] = child;
    }

    pub(super) fn set_edge_shape(&mut self, edge: usize, value: f32) {
        self.edge_shape[edge] = value;
    }

    /// Selects the next edge and charges its virtual loss.
    pub(super) fn prepare_next_for_descend(&mut self) -> usize {
        let edge = self.select_edge();
        self.charge_virtual_loss(edge);
        edge
    }
}

/// Recompute the playstyle goal for the given player.
fn recompute_playstyle_goal(game: &Game, player: PlayerId, lane_states: &mut [LaneState; 2]) {
    let seat_idx = player_id_to_usize(player);
    crate::ai::oracle_macro::observe_lane_state(&game.state, player, &mut lane_states[seat_idx]);
    crate::ai::oracle_macro::select_lane(&game.state, player, &mut lane_states[seat_idx], None);
}
