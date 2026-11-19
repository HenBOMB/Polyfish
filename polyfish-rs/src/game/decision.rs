use super::Game;
use crate::actions::UndoCallback;
use crate::moves::{Move, ResignMove};
use crate::types::MoveType;

pub use crate::replay::ReplayCommand as Action;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRotation {
    NextPlayer,
    SamePlayer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ActionError {
    #[error("the game is over")]
    GameOver,
    #[error("the action is not in the current legal move list")]
    IllegalAction,
    #[error("the action matches more than one current legal move")]
    AmbiguousAction,
    #[error("the engine could not execute the legal action")]
    ExecutionFailed,
}

/// A game borrowed together with its current legal move list.
///
/// Use this when a policy needs to inspect the same list that it will play from.
///
/// ```compile_fail
/// let mut game = polyfish::Game::new();
/// let decision = game.decision();
/// game.state.settings.turn = 2;
/// decision.play(&polyfish::Action::EndTurn).unwrap();
/// ```
///
/// ```compile_fail
/// let mut game = polyfish::Game::new();
/// let decision = game.decision();
/// decision.play_index(0).unwrap();
/// decision.play_index(0).unwrap();
/// ```
///
/// ```compile_fail
/// let mut game = polyfish::Game::new();
/// let mut other = polyfish::Game::new();
/// let decision = game.decision();
/// std::mem::swap(&mut game, &mut other);
/// decision.play_index(0).unwrap();
/// ```
///
/// ```compile_fail
/// use polyfish::moves::{Move, MoveExecution, StepMove};
/// let mut game = polyfish::Game::new();
/// let _ = StepMove::new(0, 1).execute(&mut game.state, MoveExecution(()));
/// ```
pub struct Decision<'game> {
    game: &'game mut Game,
    moves: Vec<Box<dyn Move>>,
}

impl Game {
    /// List the current player's actions on the authoritative board.
    pub fn legal_actions(&self) -> Vec<Action> {
        self.decision_moves()
            .iter()
            .map(|m| Action::from_move(m.as_ref()).expect("generated move has an action identity"))
            .collect()
    }

    pub(crate) fn decision_moves(&self) -> Vec<Box<dyn Move>> {
        if self.state.settings._game_over {
            return Vec::new();
        }
        let mut moves = self.legal_moves();
        if moves
            .first()
            .is_some_and(|m| m.move_type() == MoveType::EndTurn)
            && self.state.tribes.contains_key(&self.current_player_id())
        {
            moves.push(Box::new(ResignMove));
        }
        moves
    }

    /// Borrow the current game and generate its legal moves once.
    pub fn decision(&mut self) -> Decision<'_> {
        let moves = self.decision_moves();
        Decision { game: self, moves }
    }

    /// Apply a legal action. Undo in reverse order on this game's state.
    /// Committed discoveries remain visible after undo.
    pub fn play(&mut self, action: &Action) -> Result<UndoCallback, ActionError> {
        self.decision().play(action)
    }

    /// Apply a legal search action without committing discoveries or history.
    /// Undo in reverse order on this game's state.
    pub fn simulate(
        &mut self,
        action: &Action,
        rotation: TurnRotation,
    ) -> Result<UndoCallback, ActionError> {
        let decision = self.decision();
        let index = decision.resolve(action)?;
        decision.simulate_index(index, rotation)
    }
}

impl Decision<'_> {
    pub fn game(&self) -> &Game {
        self.game
    }

    /// Moves in engine order. Indices belong to this decision.
    pub fn moves(&self) -> &[Box<dyn Move>] {
        &self.moves
    }

    pub(crate) fn search_moves(&self) -> &[Box<dyn Move>] {
        if self
            .moves
            .last()
            .is_some_and(|m| m.move_type() == MoveType::Resign)
        {
            &self.moves[..self.moves.len() - 1]
        } else {
            &self.moves
        }
    }

    pub fn play(self, action: &Action) -> Result<UndoCallback, ActionError> {
        let index = self.resolve(action)?;
        self.play_index(index)
    }

    /// Apply one of the moves from this decision's list by index.
    pub fn play_index(self, index: usize) -> Result<UndoCallback, ActionError> {
        if self.game.state.settings._game_over {
            return Err(ActionError::GameOver);
        }
        let selected = self.moves.get(index).ok_or(ActionError::IllegalAction)?;
        self.game
            .play_generated(selected.as_ref())
            .ok_or(ActionError::ExecutionFailed)
    }

    /// Simulate one of the moves from this decision's list by index.
    pub fn simulate_index(
        self,
        index: usize,
        rotation: TurnRotation,
    ) -> Result<UndoCallback, ActionError> {
        if self.game.state.settings._game_over {
            return Err(ActionError::GameOver);
        }
        let selected = self.moves.get(index).ok_or(ActionError::IllegalAction)?;
        self.game
            .simulate_generated_with_rotation(selected.as_ref(), Some(rotation))
            .ok_or(ActionError::ExecutionFailed)
    }

    fn resolve(&self, action: &Action) -> Result<usize, ActionError> {
        if self.game.state.settings._game_over {
            return Err(ActionError::GameOver);
        }
        let mut matches = self
            .moves
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                action
                    .matches_live_move(candidate.as_ref())
                    .then_some(index)
            });
        let index = matches.next().ok_or(ActionError::IllegalAction)?;
        if matches.next().is_some() {
            return Err(ActionError::AmbiguousAction);
        }
        Ok(index)
    }
}
