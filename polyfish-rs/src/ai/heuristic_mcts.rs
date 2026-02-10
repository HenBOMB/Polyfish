use crate::ai::evaluator;
use crate::ai::mcts::{MctsAnalysis, MoveEvaluation};
use crate::game::Game;
use crate::moves::{EndTurnMove, Move};
use crate::states::PlayerId;
use crate::types::MoveType;

pub struct HeuristicMctsAgent {
    pub iterations: usize,
    pub exploration_constant: f32,
}

struct Node {
    visits: f32,
    value: f32,
    children: Vec<Node>,
    move_to_here: Option<Box<dyn Move>>,
    untried_moves: Option<Vec<Box<dyn Move>>>,
}

impl Node {
    fn new(move_to_here: Option<Box<dyn Move>>, game: &mut Game) -> Self {
        let untried = if game.state.settings._game_over {
            None
        } else {
            let book_moves = crate::ai::book::Book::recommend(game);

            let mut filtered = if !book_moves.is_empty() {
                // If we have book recommendations, strictly follow them
                book_moves
            } else {
                game.legal_moves()
            };

            // Move Ordering: Sort by heuristic score ascending (best moves at the end for .pop())
            filtered.sort_by(|a, b| {
                let score_a = crate::ai::ordering::score_move(game, a.as_ref());
                let score_b = crate::ai::ordering::score_move(game, b.as_ref());
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            Some(filtered)
        };

        Self {
            visits: 0.0,
            value: 0.0,
            children: Vec::new(),
            move_to_here,
            untried_moves: untried,
        }
    }

    fn uct_select_child(&mut self, c: f32) -> &mut Node {
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
                a_val.partial_cmp(&b_val).unwrap()
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

impl HeuristicMctsAgent {
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            exploration_constant: 1.414,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let (best_move, _) = self.select_move_with_analysis(game);
        best_move
    }

    pub fn select_move_with_analysis(
        &self,
        game: &mut Game,
    ) -> (Option<Box<dyn Move>>, MctsAnalysis) {
        let player_id = game.state.settings.current_player_turn_id;
        let mut root = Node::new(None, game);

        // Filter out EndTurn from root moves if there are other options
        // This encourages the AI to do actions before ending turn
        let mut filtered_end_turn = false;
        if let Some(moves) = &mut root.untried_moves {
            let has_other_moves = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
            if has_other_moves {
                moves.retain(|m| m.move_type() != MoveType::EndTurn);
                filtered_end_turn = true;
            }
        }

        for _ in 0..self.iterations {
            self.search_iteration(game, &mut root, player_id);
        }

        // Analysis extraction
        let mut evaluations: Vec<MoveEvaluation> = root
            .children
            .iter()
            .filter_map(|child| {
                let m = child.move_to_here.as_ref()?;
                let json = m.serialize();

                let src = json
                    .get("src")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .or_else(|| {
                        json.get("tileIndex")
                            .and_then(|v| v.as_i64())
                            .map(|v| v as i32)
                    })
                    .unwrap_or(-1);

                let target = json
                    .get("target")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(src);

                let win_rate = if child.visits > 0.0 {
                    child.value / child.visits
                } else {
                    0.0
                };

                Some(MoveEvaluation {
                    src,
                    target,
                    visits: child.visits,
                    win_rate,
                    move_type: m.move_type(),
                })
            })
            .collect();

        evaluations.sort_by(|a, b| b.visits.partial_cmp(&a.visits).unwrap());

        let analysis = MctsAnalysis {
            evaluations,
            total_iterations: self.iterations,
        };

        let mut best_move = root
            .children
            .into_iter()
            .max_by(|a, b| a.visits.partial_cmp(&b.visits).unwrap())
            .and_then(|n| n.move_to_here);

        if best_move.is_none() && filtered_end_turn {
            best_move = Some(Box::new(EndTurnMove));
        }

        (best_move, analysis)
    }

    fn search_iteration(&self, game: &mut Game, node: &mut Node, pov: PlayerId) -> f32 {
        // 1. Selection
        if node.is_fully_expanded() && !node.children.is_empty() {
            let child = node.uct_select_child(self.exploration_constant);
            if let Some(m) = &child.move_to_here {
                if let Some(undo) = game.simulate_move(m.as_ref()) {
                    let val = self.search_iteration(game, child, pov);
                    undo(&mut game.state);

                    // Update stats
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
                    let mut child = Node::new(Some(m), game);
                    let val = self.simulate_heuristic(game, pov); // Use Heuristic here
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

        // Terminal or Stalled
        let val = self.simulate_heuristic(game, pov);
        node.visits += 1.0;
        node.value += val;
        val
    }

    /// Instead of random rollout, we just evaluate the current state directly
    /// This is "Heuristic MCTS"
    fn simulate_heuristic(&self, game: &Game, pov: PlayerId) -> f32 {
        let score = evaluator::evaluate_state(&game.state, pov);
        // Map rough score -inf..inf to -1..1 or 0..1 for MCTS value
        // evaluate_state generally returns (MyScore - OppScore)
        // Let's sigmoid it to get 0.0 - 1.0

        let sigmoid = |x: f32| 1.0 / (1.0 + (-x * 0.1).exp());
        sigmoid(score)
    }
}
