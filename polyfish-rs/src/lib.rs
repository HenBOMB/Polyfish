//! Polyfish - Polytopia game simulator in Rust
//!
//! This library provides a complete simulation of the Polytopia game engine,
//! translated from the original TypeScript implementation.

// serde_json's json! recurses once per key; training_api's metric row
// outgrew the default limit of 128.
#![recursion_limit = "256"]

pub mod actions;
pub mod ai;
pub mod coords;
pub mod fow;
pub mod functions;
pub mod game;
pub mod hash;
pub mod illegal_move_log;
pub mod mapgen;
pub mod moves;
pub mod prediction;
pub mod recorder;
pub mod replayer;
pub mod score;
pub mod settings;
pub mod states;
pub mod types;
pub mod training_api;
pub mod version_sync;

pub use coords::Coords;
pub use game::Game;
pub use states::*;
pub use types::*;
