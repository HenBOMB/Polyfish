use crate::game::Game;

use super::{
    NoopReplayObserver, Replay, ReplayCommand, ReplayError, ReplayExecutor, ReplayMoveContext,
};

/// Seekable playback session. Backward seeks deterministically reconstruct
/// from `initial_state`; no game-rule undo logic is duplicated here.
pub struct ReplayPlayback {
    replay: Replay,
    game: Game,
    cursor: usize,
}

impl ReplayPlayback {
    pub fn new(replay: Replay) -> Result<Self, ReplayError> {
        let game = ReplayExecutor::initialize(&replay)?;
        Ok(Self {
            replay,
            game,
            cursor: 0,
        })
    }

    pub fn replay(&self) -> &Replay {
        &self.replay
    }
    pub fn game(&self) -> &Game {
        &self.game
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }
    pub fn total_commands(&self) -> usize {
        self.replay.command_count()
    }

    pub fn current_command(&self) -> Option<&ReplayCommand> {
        self.command_at(self.cursor).map(|(_, c)| c)
    }

    pub fn advance(&mut self) -> Result<bool, ReplayError> {
        let Some((context, command)) = self
            .command_at(self.cursor)
            .map(|(ctx, cmd)| (ctx, cmd.clone()))
        else {
            return Ok(false);
        };
        ReplayExecutor::execute_command(
            &mut self.game,
            &command,
            &context,
            &mut NoopReplayObserver,
        )?;
        self.cursor += 1;
        Ok(true)
    }

    pub fn seek(&mut self, target: usize) -> Result<(), ReplayError> {
        if target > self.total_commands() {
            return Err(ReplayError::validation(format!(
                "playback index {target} exceeds {} commands",
                self.total_commands()
            )));
        }
        if target < self.cursor {
            self.game = ReplayExecutor::initialize(&self.replay)?;
            self.cursor = 0;
        }
        while self.cursor < target {
            self.advance()?;
        }
        Ok(())
    }

    pub fn context(&self) -> Option<ReplayMoveContext> {
        self.command_at(self.cursor).map(|(ctx, _)| ctx)
    }

    fn command_at(&self, wanted: usize) -> Option<(ReplayMoveContext, &ReplayCommand)> {
        let mut global = 0;
        for (turn_index, turn) in self.replay.turns.iter().enumerate() {
            for (command_index, command) in turn.commands.iter().enumerate() {
                if global == wanted {
                    return Some((
                        ReplayMoveContext {
                            turn_index,
                            turn_number: turn.turn_number,
                            player_id: turn.player_id,
                            command_index,
                            global_command_index: global,
                        },
                        command,
                    ));
                }
                global += 1;
            }
        }
        None
    }
}
