use crate::ai::evaluator;
use crate::ai::genes::AIGenes;
use crate::game::Game;
use crate::moves::EndTurnMove;
use crate::moves::Move;
use crate::states::PlayerId;
use crate::types::MoveType;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Serialize;

/// Evaluation data for a single move candidate
#[derive(Debug, Clone, Serialize)]
pub struct MoveEvaluation {
    pub src: i32,
    pub target: i32,
    pub visits: f32,
    pub win_rate: f32,
    pub move_type: MoveType,
    pub description: String,
}

/// Analysis results from MCTS search
#[derive(Debug, Clone, Serialize)]
pub struct MctsAnalysis {
    pub evaluations: Vec<MoveEvaluation>,
    pub total_iterations: usize,
    pub principal_variation: Vec<String>,
    pub tree: Option<MctsNodeData>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MctsNodeData {
    pub visits: f32,
    pub value: f32,
    pub move_description: String,
    pub children: Vec<MctsNodeData>,
}

pub struct MctsAgent {
    pub iterations: usize,
    pub exploration_constant: f32,
    pub genes: AIGenes,
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
            Some(game.legal_moves())
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
                let a_val = (a.value / a.visits) + c * (parent_visits.ln() / a.visits).sqrt();
                let b_val = (b.value / b.visits) + c * (parent_visits.ln() / b.visits).sqrt();
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

impl MctsAgent {
    pub fn new(iterations: usize) -> Self {
        let genes = AIGenes::default();
        Self {
            iterations,
            exploration_constant: genes.mcts.exploration_constant,
            genes,
        }
    }

    pub fn with_genes(iterations: usize, genes: AIGenes) -> Self {
        let c = genes.mcts.exploration_constant;
        Self {
            iterations,
            exploration_constant: c,
            genes,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let (best_move, _) = self.select_move_with_analysis(game);
        best_move
    }

    /// Run MCTS and return both the best move and analysis data for visualization
    pub fn select_move_with_analysis(
        &self,
        game: &mut Game,
    ) -> (Option<Box<dyn Move>>, MctsAnalysis) {
        let player_id = game.state.settings.current_player_turn_id;
        let mut root = Node::new(None, game);

        // Filter out EndTurn from root moves if there are other options
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

        // Extract evaluations from all children
        let mut evaluations: Vec<MoveEvaluation> = root
            .children
            .iter()
            .filter_map(|child| {
                let m = child.move_to_here.as_ref()?;
                let json = m.serialize();

                // Extract src and target from move JSON
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
                    description: format!("{:?}", m.move_type()),
                })
            })
            .collect();

        // Sort by visits (most visited first)
        evaluations.sort_by(|a, b| b.visits.partial_cmp(&a.visits).unwrap());

        let analysis = MctsAnalysis {
            evaluations,
            total_iterations: self.iterations,
            principal_variation: Vec::new(),
            tree: None,
        };

        // Return child with most visits
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
                    let val = self.simulate(game, pov);
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

        // Game over or can't move
        let val = evaluator::evaluate_state(&game.state, pov, &self.genes);
        node.visits += 1.0;
        node.value += val;
        val
    }

    fn simulate(&self, game: &mut Game, pov: PlayerId) -> f32 {
        let mut moves_played = 0;
        let mut undos = Vec::new();

        // Random rollout for simulation
        let max_rollout_depth = 20;
        let mut rng = thread_rng();

        while !game.state.settings._game_over && moves_played < max_rollout_depth {
            let moves = game.legal_moves();
            if moves.is_empty() {
                break;
            }

            let m = moves.choose(&mut rng).unwrap();
            if let Some(undo) = game.simulate_move(m.as_ref()) {
                undos.push(undo);
                moves_played += 1;
            } else {
                break;
            }
        }

        let val = evaluator::evaluate_state(&game.state, pov, &self.genes);

        // Undo all rollout moves
        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        val
    }
}
