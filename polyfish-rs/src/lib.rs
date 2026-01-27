//! Polyfish - Polytopia game simulator in Rust
//! 
//! This library provides a complete simulation of the Polytopia game engine,
//! translated from the original TypeScript implementation.

pub mod types;
pub mod states;
pub mod coords;
pub mod settings;
pub mod functions;
pub mod actions;
pub mod moves;
pub mod game;

pub use types::*;
pub use states::*;
pub use coords::Coords;
pub use game::Game;
