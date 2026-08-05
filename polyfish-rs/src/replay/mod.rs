//! Versioned, engine-authoritative replay support.
//!
//! JSON parsing, structural validation, legal-move resolution, playback and
//! training extraction deliberately share this one domain model.

pub mod command;
pub mod errors;
pub mod executor;
pub mod loader;
pub mod playback;
pub mod recorder;
pub mod schema;
pub mod training;
pub mod validator;

pub use command::ReplayCommand;
pub use errors::{ReplayError, ReplayMoveContext};
pub use executor::{NoopReplayObserver, ReplayExecutor, ReplayObserver};
pub use loader::{load_replay, load_replay_reader, save_replay};
pub use playback::ReplayPlayback;
pub use recorder::ReplayRecorder;
pub use schema::*;
pub use validator::{TrainingEligibility, validate_replay, validate_training_eligibility};

pub const REPLAY_SCHEMA_VERSION: u32 = 1;
pub const DATASET_SCHEMA_VERSION: u32 = 1;
pub const FEATURE_SCHEMA_VERSION: u32 = 1;
pub const ACTION_SCHEMA_VERSION: u32 = 1;

#[cfg(test)]
mod tests;
