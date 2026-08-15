use crate::states::PlayerId;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use thiserror::Error;

use super::ReplayCommand;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ReplayMoveContext {
    pub turn_index: usize,
    pub turn_number: i32,
    pub player_id: PlayerId,
    pub command_index: usize,
    pub global_command_index: usize,
}

impl fmt::Display for ReplayMoveContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "turn {}, player {}, command {} (global {}, segment {})",
            self.turn_number,
            self.player_id,
            self.command_index,
            self.global_command_index,
            self.turn_index,
        )
    }
}

#[derive(Debug, Error)]
pub enum ReplayError {
    #[error("cannot read replay {file}: {source}")]
    Io {
        file: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid replay JSON in {file}: {source}")]
    Json {
        file: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("replay validation failed{file}: {message}")]
    Validation { file: String, message: String },
    #[error(
        "illegal replay command at {context}: {command:?}; legal moves: {legal_move_summaries:?}"
    )]
    IllegalCommand {
        context: ReplayMoveContext,
        command: ReplayCommand,
        legal_move_summaries: Vec<String>,
    },
    #[error(
        "ambiguous replay command at {context}: {command:?}; matching legal moves: {matching_move_summaries:?}"
    )]
    AmbiguousCommand {
        context: ReplayMoveContext,
        command: ReplayCommand,
        matching_move_summaries: Vec<String>,
    },
    #[error("active player mismatch at {context}: replay declares {declared}, engine has {actual}")]
    ActivePlayer {
        context: ReplayMoveContext,
        declared: PlayerId,
        actual: PlayerId,
    },
    #[error("engine turn mismatch at {context}: replay declares {declared}, engine has {actual}")]
    TurnNumber {
        context: ReplayMoveContext,
        declared: i32,
        actual: i32,
    },
    #[error("engine refused selected legal move at {context}: {move_summary}")]
    Execution {
        context: ReplayMoveContext,
        move_summary: String,
    },
    #[error("cannot convert engine move {move_summary} to replay command: {message}")]
    CommandConversion {
        move_summary: String,
        message: String,
    },
    #[error("replay is not eligible for training: {message}")]
    TrainingIneligible { message: String },
    #[error("training export failed: {0}")]
    Training(String),
}

impl ReplayError {
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation {
            file: String::new(),
            message: message.into(),
        }
    }

    pub fn validation_at(file: impl Into<String>, message: impl Into<String>) -> Self {
        let file = file.into();
        Self::Validation {
            file: if file.is_empty() {
                String::new()
            } else {
                format!(" in {file}")
            },
            message: message.into(),
        }
    }
}
