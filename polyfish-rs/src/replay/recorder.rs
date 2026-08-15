use crate::moves::Move;
use crate::states::GameState;

use super::{
    REPLAY_SCHEMA_VERSION, Replay, ReplayCommand, ReplayError, ReplayMetadata, ReplayResult,
    ReplayTurn,
};

/// Records exactly the atomic moves executed by an engine game.
pub struct ReplayRecorder {
    replay: Replay,
}

impl ReplayRecorder {
    pub fn new(initial_state: GameState, metadata: ReplayMetadata) -> Self {
        Self {
            replay: Replay {
                schema_version: REPLAY_SCHEMA_VERSION,
                metadata,
                initial_state,
                turns: Vec::new(),
                result: None,
            },
        }
    }

    pub fn record_move(
        &mut self,
        turn_number: i32,
        player_id: i32,
        game_move: &dyn Move,
    ) -> Result<(), ReplayError> {
        self.record_command(turn_number, player_id, ReplayCommand::from_move(game_move)?)
    }

    pub fn record_command(
        &mut self,
        turn_number: i32,
        player_id: i32,
        command: ReplayCommand,
    ) -> Result<(), ReplayError> {
        if let Some(last) = self.replay.turns.last_mut() {
            if last.turn_number == turn_number && last.player_id == player_id {
                last.commands.push(command);
                return Ok(());
            }
            if turn_number < last.turn_number {
                return Err(ReplayError::validation(format!(
                    "recorder turn went backwards from {} to {turn_number}",
                    last.turn_number
                )));
            }
        }
        self.replay.turns.push(ReplayTurn {
            turn_number,
            player_id,
            commands: vec![command],
        });
        Ok(())
    }

    pub fn finish(mut self, result: Option<ReplayResult>) -> Replay {
        self.replay.result = result;
        self.replay
    }
}
