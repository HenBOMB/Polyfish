use crate::ai::evaluator;
use crate::ai::genes::AIGenes;
use crate::ai::mcts_types::MoveVisit;
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::states::PlayerId;
use crate::types::MoveType;

/// Search agent driven by evolved parameter sets (AIGenes, Candidate 896).
/// When `iterations == 0`, acts as pure greedy argmax over `gene_ordering::score_move`.
/// When `iterations > 0`, runs heuristic MCTS with `gene_ordering::score_move` guiding
/// tree expansion and rollouts.
pub struct GeneAgent {
    pub genes: AIGenes,
    pub iterations: usize,
    pub exploration_constant: f32,
    rng: std::cell::RefCell<rand::rngs::SmallRng>,
}

struct GeneNode {
    visits: f32,
    value: f32,
    children: Vec<GeneNode>,
    move_to_here: Option<Box<dyn Move>>,
    untried_moves: Option<Vec<Box<dyn Move>>>,
}

impl GeneNode {
    fn new(move_to_here: Option<Box<dyn Move>>, game: &mut Game, genes: &AIGenes) -> Self {
        let is_end_turn = move_to_here
            .as_ref()
            .map_or(false, |m| m.move_type() == MoveType::EndTurn);

        let untried = if game.state.settings._game_over || is_end_turn {
            None
        } else {
            let mut moves = game.legal_moves();
            // Move Ordering: Sort by heuristic score ascending (best moves at the end for .pop())
            moves.sort_by(|a, b| {
                let score_a = crate::ai::gene_ordering::score_move(game, a.as_ref(), genes);
                let score_b = crate::ai::gene_ordering::score_move(game, b.as_ref(), genes);
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            Some(moves)
        };

        Self {
            visits: 0.0,
            value: 0.0,
            children: Vec::new(),
            move_to_here,
            untried_moves: untried,
        }
    }

    fn uct_select_child(&mut self, c: f32) -> &mut GeneNode {
        let parent_visits = self.visits;
        self.children
            .iter_mut()
            .max_by(|a, b| {
                let a_val = if a.visits > 0.0 {
                    (a.value / a.visits) + c * (parent_visits.ln() / a.visits).sqrt()
                } else {
                    f32::INFINITY
                };
                let b_val = if b.visits > 0.0 {
                    (b.value / b.visits) + c * (parent_visits.ln() / b.visits).sqrt()
                } else {
                    f32::INFINITY
                };
                a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }

    fn is_fully_expanded(&self) -> bool {
        match &self.untried_moves {
            Some(v) => v.is_empty(),
            None => true,
        }
    }
}

impl GeneAgent {
    pub fn new(genes: AIGenes, iterations: usize) -> Self {
        let c = genes.mcts.exploration_constant;
        Self {
            genes,
            iterations,
            exploration_constant: if c > 0.0 { c } else { 0.6 },
            rng: std::cell::RefCell::new(crate::ai::mcts_common::next_search_rng()),
        }
    }

    pub fn select_move(&mut self, game: &mut Game) -> Option<Box<dyn Move>> {
        self.select_move_with_decomposed_visits(game, usize::MAX).0
    }

    pub fn select_move_with_decomposed_visits(
        &mut self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        if self.iterations == 0 {
            self.select_greedy_move(game, move_count)
        } else {
            self.select_mcts_move(game)
        }
    }

    fn select_greedy_move(
        &self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        let mut moves = game.legal_moves();
        let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
        if has_other && !crate::game::end_turn_retained() {
            moves.retain(|m| m.move_type() != MoveType::EndTurn);
        }
        if moves.is_empty() {
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }

        let scores: Vec<f32> = moves
            .iter()
            .map(|m| crate::ai::gene_ordering::score_move(game, m.as_ref(), &self.genes))
            .collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let temp: f32 = 5.0;
        let probs: Vec<f32> = {
            let exps: Vec<f32> = scores
                .iter()
                .map(|s| ((s - max_score) / temp).exp())
                .collect();
            let sum: f32 = exps.iter().sum();
            if sum > 0.0 {
                exps.iter().map(|e| e / sum).collect()
            } else {
                vec![1.0 / moves.len() as f32; moves.len()]
            }
        };

        let visits: Vec<MoveVisit> = moves
            .iter()
            .zip(&probs)
            .map(|(m, &p)| MoveVisit {
                move_type: m.move_type(),
                visits: p,
                source_idx: m.source_idx().ok(),
                target_idx: m.target_idx().ok(),
                structure_type: m.structure_type().ok(),
                unit_type: m.unit_type().ok(),
                tech_type: m.tech_type().ok(),
                ability_type: m.ability_type().ok(),
                reward_type: m.reward_type().ok(),
            })
            .collect();

        let chosen = if move_count < crate::ai::mcts_zero::ZeroMctsAgent::TEMPERATURE_MOVE_THRESHOLD
            && moves.len() > 1
        {
            use rand::distr::{Distribution, weighted::WeightedIndex};
            WeightedIndex::new(&probs)
                .ok()
                .map(|d| d.sample(&mut *self.rng.borrow_mut()))
        } else {
            None
        };
        let idx = chosen.unwrap_or_else(|| {
            scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        });

        (Some(moves.swap_remove(idx)), visits)
    }

    fn select_mcts_move(
        &mut self,
        game: &mut Game,
    ) -> (Option<Box<dyn Move>>, Vec<MoveVisit>) {
        let player_id = game.state.settings.current_player_turn_id;
        let mut root = GeneNode::new(None, game, &self.genes);

        let mut filtered_end_turn = false;
        if let Some(moves) = &mut root.untried_moves {
            let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
            if has_other {
                moves.retain(|m| m.move_type() != MoveType::EndTurn);
                filtered_end_turn = true;
            }
        }

        for _ in 0..self.iterations {
            self.search_iteration(game, &mut root, player_id);
        }

        let total_visits: f32 = root.children.iter().map(|c| c.visits).sum();
        let visits: Vec<MoveVisit> = root
            .children
            .iter()
            .filter_map(|c| {
                let m = c.move_to_here.as_ref()?;
                let p = if total_visits > 0.0 {
                    c.visits / total_visits
                } else {
                    0.0
                };
                Some(MoveVisit {
                    move_type: m.move_type(),
                    visits: p,
                    source_idx: m.source_idx().ok(),
                    target_idx: m.target_idx().ok(),
                    structure_type: m.structure_type().ok(),
                    unit_type: m.unit_type().ok(),
                    tech_type: m.tech_type().ok(),
                    ability_type: m.ability_type().ok(),
                    reward_type: m.reward_type().ok(),
                })
            })
            .collect();

        let mut best_move = root
            .children
            .into_iter()
            .max_by(|a, b| a.visits.partial_cmp(&b.visits).unwrap_or(std::cmp::Ordering::Equal))
            .and_then(|n| n.move_to_here);

        if best_move.is_none() && filtered_end_turn {
            best_move = Some(Box::new(EndTurnMove));
        }

        (best_move, visits)
    }

    fn search_iteration(&self, game: &mut Game, node: &mut GeneNode, pov: PlayerId) -> f32 {
        // 1. Selection
        if node.is_fully_expanded() && !node.children.is_empty() {
            let child = node.uct_select_child(self.exploration_constant);
            if let Some(m) = &child.move_to_here {
                if let Some(undo) = game.simulate_move(m.as_ref()) {
                    let val = self.search_iteration(game, child, pov);
                    undo(&mut game.state);

                    node.visits += 1.0;
                    node.value += val;
                    return val;
                }
            }
        }

        // 2. Expansion
        if let Some(untried) = &mut node.untried_moves {
            if !untried.is_empty() {
                let m = untried.pop().unwrap();
                if let Some(undo) = game.simulate_move(m.as_ref()) {
                    let mut child = GeneNode::new(Some(m), game, &self.genes);
                    let val = self.simulate_to_turn_end(game, pov);
                    undo(&mut game.state);

                    child.visits = 1.0;
                    child.value = val;
                    node.children.push(child);

                    node.visits += 1.0;
                    node.value += val;
                    return val;
                }
            }
        }

        // 3. Rollout / Terminal
        let val = self.simulate_to_turn_end(game, pov);
        node.visits += 1.0;
        node.value += val;
        val
    }

    fn simulate_to_turn_end(&self, game: &mut Game, pov: PlayerId) -> f32 {
        let start_turn = game.state.settings.turn;
        let current_player = game.state.settings.current_player_turn_id;
        let mut undos: Vec<crate::actions::UndoCallback> = Vec::new();
        let max_rollout = 30;

        for _ in 0..max_rollout {
            if game.state.settings.turn != start_turn
                || game.state.settings.current_player_turn_id != current_player
                || game.state.settings._game_over
            {
                break;
            }

            let mut moves = game.legal_moves();
            if moves.is_empty() {
                break;
            }

            if moves.len() == 1 && moves[0].move_type() == MoveType::EndTurn {
                if let Some(undo) = game.simulate_move(moves.remove(0).as_ref()) {
                    undos.push(undo);
                }
                break;
            }

            moves.retain(|m| m.move_type() != MoveType::EndTurn);
            if moves.is_empty() {
                break;
            }

            let best_idx = moves
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let sa = crate::ai::gene_ordering::score_move(game, a.as_ref(), &self.genes);
                    let sb = crate::ai::gene_ordering::score_move(game, b.as_ref(), &self.genes);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let best_move = moves.swap_remove(best_idx);
            if let Some(undo) = game.simulate_move(best_move.as_ref()) {
                undos.push(undo);
            } else {
                break;
            }
        }

        let score = evaluator::evaluate_state(&game.state, pov);

        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        (score + 1.0) / 2.0
    }
}
