//! Polyfish - Polytopia game simulator in Rust
//!
//! This library provides a complete simulation of the Polytopia game engine,
//! translated from the original TypeScript implementation.

pub mod actions;
pub mod ai;
pub mod coords;
pub mod functions;
pub mod game;
pub mod mapgen;
pub mod moves;
pub mod settings;
pub mod states;
pub mod types;

pub use coords::Coords;
pub use game::Game;
pub use states::*;
pub use types::*;
