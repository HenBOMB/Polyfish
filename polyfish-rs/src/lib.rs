//! Polyfish - Polytopia game simulator in Rust
//!
//! This library provides a complete simulation of the Polytopia game engine,
//! translated from the original TypeScript implementation.

pub mod actions;
pub mod ai;
pub mod coords;
pub mod dotnet_rng;
pub mod fow;
pub mod functions;
pub mod game;
pub mod hash;
pub mod mapgen;
pub mod memory;
pub mod moves;
pub mod prediction;
pub mod recorder;
pub mod replay;
pub mod score;
pub mod settings;
pub mod states;
pub mod training_api;
pub mod types;
pub mod version_sync;

pub use coords::Coords;
pub use game::Game;
pub use states::*;
pub use types::*;
