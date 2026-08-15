use crate::ai::evaluator;
use crate::ai::mcts::{MctsAnalysis, MctsNodeData, MoveEvaluation};
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
        let is_end_turn = move_to_here
            .as_ref()
            .map_or(false, |m| m.move_type() == MoveType::EndTurn);

        let untried = if game.state.settings._game_over || is_end_turn {
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
                let score_a = crate::ai::scoring::score_move(game, a.as_ref());
                let score_b = crate::ai::scoring::score_move(game, b.as_ref());
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

/// Zero-search heuristic teacher: one movegen + one `score_move` pass per
/// move, policy = softmax over scores. This is the same distribution
/// `blend_heuristic_prior` injects into Gumbel roots, produced ~1000x cheaper
/// than the rollout MCTS — built for bulk imitation-corpus generation.
pub struct GreedyHeuristicAgent;

/// Softmax temperature over raw `score_move` values. At 1.0 the 40+-point
/// score bands make the distribution one-hot; 5.0 keeps the best move
/// dominant while preserving the ordering of runners-up (a ~10-point gap ≈
/// 7:1 odds), which is real signal for the policy head to learn.
const GREEDY_SOFTMAX_TEMP: f32 = 5.0;

impl GreedyHeuristicAgent {
    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        self.select_move_with_decomposed_visits(game, usize::MAX).0
    }

    /// Same contract as the search agents' training-data API: played move +
    /// per-move weights (softmax probabilities stand in for visit counts —
    /// downstream normalizes to a distribution either way). Samples the
    /// played move from the softmax for early-game diversity, argmax after.
    pub fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        use crate::ai::mcts_types::MoveVisit;

        let mut moves = game.legal_moves();
        // Mirror the search backends' root EndTurn suppression.
        let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
        if has_other {
            moves.retain(|m| m.move_type() != MoveType::EndTurn);
        }
        if moves.is_empty() {
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }

        let scores: Vec<f32> = moves
            .iter()
            .map(|m| crate::ai::scoring::score_move(game, m.as_ref()))
            .collect();
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = {
            let exps: Vec<f32> = scores
                .iter()
                .map(|s| ((s - max_score) / GREEDY_SOFTMAX_TEMP).exp())
                .collect();
            let sum: f32 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
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
                .map(|d| d.sample(&mut rand::rng()))
        } else {
            None
        };
        let idx = chosen.unwrap_or_else(|| {
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        });

        (Some(moves.swap_remove(idx)), visits)
    }
}

/// Uniform-random legal-move agent. Exists as the fixed 0-Elo anchor for the
/// rating ladder (`elo.py`): it never changes, so every rating ever computed
/// against it stays comparable across runs and architectures.
pub struct RandomAgent;

impl RandomAgent {
    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        self.select_move_with_decomposed_visits(game, usize::MAX).0
    }

    /// Mirrors the other backends' root EndTurn suppression so "random" means
    /// random play, not random early turn-ending.
    pub fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
        _move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        use crate::ai::mcts_types::MoveVisit;
        use rand::Rng;

        let mut moves = game.legal_moves();
        let has_other = moves.iter().any(|m| m.move_type() != MoveType::EndTurn);
        if has_other {
            moves.retain(|m| m.move_type() != MoveType::EndTurn);
        }
        if moves.is_empty() {
            return (Some(Box::new(EndTurnMove)), Vec::new());
        }

        let p = 1.0 / moves.len() as f32;
        let visits: Vec<MoveVisit> = moves
            .iter()
            .map(|m| MoveVisit {
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

        let idx = rand::rng().random_range(0..moves.len());
        (Some(moves.swap_remove(idx)), visits)
    }
}

impl HeuristicMctsAgent {
    pub fn new(iterations: usize) -> Self {
        Self {
            iterations,
            exploration_constant: 0.6,
        }
    }

    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let (best_move, _) = self.select_move_with_analysis(game);
        best_move
    }

    /// Build the root (EndTurn filtered out when other moves exist, to
    /// encourage acting before passing) and run the configured number of
    /// search iterations. Returns (root, filtered_end_turn).
    fn run_root_search(&self, game: &mut Game) -> (Node, bool) {
        let player_id = game.state.settings.current_player_turn_id;
        let mut root = Node::new(None, game);

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
        (root, filtered_end_turn)
    }

    /// Network-free analogue of the NN agents' training-data API: returns the
    /// played move plus root visit counts (the policy target), sampling by
    /// visits for early-game diversity at the same threshold the Zero agent
    /// uses. Lets self_play generate imitation corpora from the heuristic.
    pub fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
        move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        use crate::ai::mcts_types::MoveVisit;

        let (root, filtered_end_turn) = self.run_root_search(game);

        let visits: Vec<MoveVisit> = root
            .children
            .iter()
            .filter_map(|child| {
                let m = child.move_to_here.as_ref()?;
                Some(MoveVisit {
                    move_type: m.move_type(),
                    visits: child.visits,
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

        let sampled_idx = if move_count
            < crate::ai::mcts_zero::ZeroMctsAgent::TEMPERATURE_MOVE_THRESHOLD
            && root.children.len() > 1
        {
            use rand::distr::{Distribution, weighted::WeightedIndex};
            let weights: Vec<f32> = root.children.iter().map(|c| c.visits.max(0.0)).collect();
            // All-zero weights error out; fall through to argmax below.
            WeightedIndex::new(&weights)
                .ok()
                .map(|dist| dist.sample(&mut rand::rng()))
        } else {
            None
        };

        let best_move = match sampled_idx {
            Some(i) => root
                .children
                .into_iter()
                .nth(i)
                .and_then(|n| n.move_to_here),
            None => root
                .children
                .into_iter()
                .max_by(|a, b| a.visits.partial_cmp(&b.visits).unwrap())
                .and_then(|n| n.move_to_here),
        };

        if best_move.is_none() && filtered_end_turn {
            return (Some(Box::new(EndTurnMove)), visits);
        }
        (best_move, visits)
    }

    pub fn select_move_with_analysis(
        &self,
        game: &mut Game,
    ) -> (Option<Box<dyn Move>>, MctsAnalysis) {
        let (root, filtered_end_turn) = self.run_root_search(game);

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
                    description: m.describe(&game.state),
                })
            })
            .collect();

        evaluations.sort_by(|a, b| b.visits.partial_cmp(&a.visits).unwrap());

        // Extract principal variation (best line) by following most-visited children
        let mut pv = Vec::new();
        let mut current = &root;
        for _ in 0..10 {
            if current.children.is_empty() {
                break;
            }
            let best_child = current
                .children
                .iter()
                .max_by(|a, b| a.visits.partial_cmp(&b.visits).unwrap());
            match best_child {
                Some(child) if child.visits > 0.0 => {
                    if let Some(m) = &child.move_to_here {
                        pv.push(m.describe(&game.state));
                    }
                    current = child;
                }
                _ => break,
            }
        }

        let analysis = MctsAnalysis {
            evaluations,
            total_iterations: self.iterations,
            principal_variation: pv,
            tree: Some(Self::build_tree_data(&root, &game.state, 0)),
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

        // Terminal or Stalled
        let val = self.simulate_to_turn_end(game, pov);
        node.visits += 1.0;
        node.value += val;
        val
    }

    /// Greedy rollout to the end of the current turn, then evaluate.
    /// Instead of evaluating mid-turn states (which punishes the start of
    /// multi-step plans), we greedily pick the best-scored move until the
    /// turn changes or EndTurn is played, then evaluate at the turn boundary.
    fn simulate_to_turn_end(&self, game: &mut Game, pov: PlayerId) -> f32 {
        let start_turn = game.state.settings.turn;
        let current_player = game.state.settings.current_player_turn_id;
        let mut undos: Vec<crate::actions::UndoCallback> = Vec::new();
        let max_rollout = 30; // Safety cap to prevent infinite loops

        for _ in 0..max_rollout {
            // Stop if turn changed (EndTurn was played) or game over
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

            // If EndTurn is the only move, play it and stop
            if moves.len() == 1 && moves[0].move_type() == crate::types::MoveType::EndTurn {
                if let Some(undo) = game.simulate_move(moves.remove(0).as_ref()) {
                    undos.push(undo);
                }
                break;
            }

            // Remove EndTurn from candidates (we want to play all useful moves first)
            moves.retain(|m| m.move_type() != crate::types::MoveType::EndTurn);
            if moves.is_empty() {
                break;
            }

            // Pick the highest-scored move greedily
            let best_idx = moves
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let sa = crate::ai::scoring::score_move(game, a.as_ref());
                    let sb = crate::ai::scoring::score_move(game, b.as_ref());
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let best_move = moves.swap_remove(best_idx);
            if let Some(undo) = game.simulate_move(best_move.as_ref()) {
                undos.push(undo);
            } else {
                break; // Move failed, stop rollout
            }
        }

        // Evaluate at turn boundary
        let score = evaluator::evaluate_state(&game.state, pov);

        // Undo all rollout moves
        while let Some(undo) = undos.pop() {
            undo(&mut game.state);
        }

        // Score is already clamped -1.0 to 1.0 in evaluate_state.
        // Map linearly to 0.0 - 1.0 for MCTS value
        (score + 1.0) / 2.0
    }

    fn build_tree_data(
        node: &Node,
        state: &crate::states::GameState,
        depth: usize,
    ) -> MctsNodeData {
        let description = if let Some(m) = &node.move_to_here {
            m.describe(state)
        } else {
            "Root".to_string()
        };

        // Recursively build children, limiting depth to avoid huge JSONs
        let mut children: Vec<MctsNodeData> = if depth < 8 {
            node.children
                .iter()
                .map(|child| Self::build_tree_data(child, state, depth + 1))
                .collect()
        } else {
            Vec::new()
        };

        // Sort by visits (descending) and take top 5 to keep it readable
        children.sort_by(|a, b| {
            b.visits
                .partial_cmp(&a.visits)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if children.len() > 5 {
            children.truncate(5);
        }

        MctsNodeData {
            visits: node.visits,
            value: node.value,
            move_description: description,
            children,
        }
    }
}

pub struct StateDiffGreedyAgent;

impl StateDiffGreedyAgent {
    pub fn select_move(&self, game: &mut Game) -> Option<Box<dyn Move>> {
        let mut moves = game.legal_moves();
        if moves.is_empty() {
            return Some(Box::new(crate::moves::EndTurnMove));
        }

        let has_other = moves
            .iter()
            .any(|m| m.move_type() != crate::types::MoveType::EndTurn);
        if has_other {
            moves.retain(|m| m.move_type() != crate::types::MoveType::EndTurn);
        }

        let pid = game.state.settings.current_player_turn_id;
        let mut best_move = None;
        let mut best_score = f32::NEG_INFINITY;

        for m in moves.into_iter() {
            // Execute directly on state — no enemy turns, no history, no FOW.
            // This is O(1) overhead vs simulate_move/play_move which both run
            // through the full game loop (including enemy turn cycling).
            if let Ok(result) = m.execute(&mut game.state) {
                let score = crate::ai::evaluator::player::evaluate_player(&game.state, pid);
                (result.undo)(&mut game.state);

                if score > best_score {
                    best_score = score;
                    best_move = Some(m);
                }
            }
        }

        best_move.or_else(|| Some(Box::new(crate::moves::EndTurnMove)))
    }

    pub fn select_move_with_decomposed_visits(
        &self,
        game: &mut Game,
        _move_count: usize,
    ) -> (Option<Box<dyn Move>>, Vec<crate::ai::mcts_types::MoveVisit>) {
        (self.select_move(game), Vec::new())
    }
}
