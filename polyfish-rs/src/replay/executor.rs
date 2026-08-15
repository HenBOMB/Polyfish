use crate::game::Game;
use crate::moves::{CaptureMove, Move, RewardMove};

use super::{Replay, ReplayCommand, ReplayError, ReplayMoveContext, validate_replay};

pub trait ReplayObserver {
    fn before_move(
        &mut self,
        _game: &Game,
        _context: &ReplayMoveContext,
        _legal_moves: &[Box<dyn Move>],
        _selected_move: &dyn Move,
        _command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        Ok(())
    }

    fn after_move(
        &mut self,
        _game: &Game,
        _context: &ReplayMoveContext,
        _command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        Ok(())
    }
}

pub struct NoopReplayObserver;
impl ReplayObserver for NoopReplayObserver {}

pub struct ReplayExecutor;

impl ReplayExecutor {
    pub fn initialize(replay: &Replay) -> Result<Game, ReplayError> {
        validate_replay(replay, None)?;
        let mut game = Game::new();
        game.state = replay.initial_state.clone();
        game.post_load();
        Ok(game)
    }

    pub fn execute(replay: &Replay) -> Result<Game, ReplayError> {
        Self::execute_with_observer(replay, &mut NoopReplayObserver)
    }

    pub fn execute_with_observer<O: ReplayObserver>(
        replay: &Replay,
        observer: &mut O,
    ) -> Result<Game, ReplayError> {
        let mut game = Self::initialize(replay)?;
        let mut global = 0;
        for (turn_index, turn) in replay.turns.iter().enumerate() {
            for (command_index, command) in turn.commands.iter().enumerate() {
                let context = ReplayMoveContext {
                    turn_index,
                    turn_number: turn.turn_number,
                    player_id: turn.player_id,
                    command_index,
                    global_command_index: global,
                };
                Self::execute_command(&mut game, command, &context, observer)?;
                global += 1;
            }
        }
        if global != replay.command_count() {
            return Err(ReplayError::validation(format!(
                "executor consumed {global} commands, replay contains {}",
                replay.command_count()
            )));
        }
        Ok(game)
    }

    pub(crate) fn execute_command<O: ReplayObserver>(
        game: &mut Game,
        command: &ReplayCommand,
        context: &ReplayMoveContext,
        observer: &mut O,
    ) -> Result<(), ReplayError> {
        let active = game.state.settings.current_player_turn_id;
        if active != context.player_id {
            return Err(ReplayError::ActivePlayer {
                context: context.clone(),
                declared: context.player_id,
                actual: active,
            });
        }
        let actual_turn = game.state.settings.turn;
        if actual_turn != context.turn_number {
            return Err(ReplayError::TurnNumber {
                context: context.clone(),
                declared: context.turn_number,
                actual: actual_turn,
            });
        }

        let legal_moves = game.legal_moves();
        let matches = matching_move_indices(command, &legal_moves);
        if matches.is_empty() {
            return Err(ReplayError::IllegalCommand {
                context: context.clone(),
                command: command.clone(),
                legal_move_summaries: legal_moves
                    .iter()
                    .take(40)
                    .map(|m| format!("{} | {}", m.describe(&game.state), m.serialize()))
                    .collect(),
            });
        }
        if matches.len() > 1 {
            return Err(ReplayError::AmbiguousCommand {
                context: context.clone(),
                command: command.clone(),
                matching_move_summaries: matches
                    .iter()
                    .map(|&i| {
                        format!(
                            "{} | {}",
                            legal_moves[i].describe(&game.state),
                            legal_moves[i].serialize()
                        )
                    })
                    .collect(),
            });
        }
        let selected = legal_moves[matches[0]].as_ref();
        observer.before_move(game, context, &legal_moves, selected, command)?;

        let played = match command {
            ReplayCommand::Capture {
                source,
                reward,
                revealed_tiles,
                technology,
            } => {
                for &idx in revealed_tiles {
                    if !game.state.tiles.contains_key(&idx) {
                        return Err(ReplayError::validation(format!(
                            "Capture hint contains out-of-bounds tile index {} at {}",
                            idx, context
                        )));
                    }
                }
                let mut hinted = CaptureMove::new(*source);
                hinted.reward = *reward;
                hinted.revealed_tiles = revealed_tiles.clone();
                hinted.tech_hint = *technology;
                game.play_move(&hinted)
            }
            ReplayCommand::Reward {
                target,
                reward,
                revealed_tiles,
            } => {
                for &idx in revealed_tiles {
                    if !game.state.tiles.contains_key(&idx) {
                        return Err(ReplayError::validation(format!(
                            "Reward hint contains out-of-bounds tile index {} at {}",
                            idx, context
                        )));
                    }
                }
                let mut hinted = RewardMove::new(*target, *reward);
                hinted.revealed_tiles = revealed_tiles.clone();
                game.play_move(&hinted)
            }
            _ => game.play_move(selected),
        };
        if played.is_none() {
            return Err(ReplayError::Execution {
                context: context.clone(),
                move_summary: selected.describe(&game.state),
            });
        }
        game.state._messages.clear();
        observer.after_move(game, context, command)
    }
}

pub(crate) fn matching_move_indices(
    command: &ReplayCommand,
    legal_moves: &[Box<dyn Move>],
) -> Vec<usize> {
    legal_moves
        .iter()
        .enumerate()
        .filter_map(|(index, legal_move)| {
            command.matches_move(legal_move.as_ref()).then_some(index)
        })
        .collect()
}
