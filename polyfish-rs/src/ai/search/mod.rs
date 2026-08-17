//! Search: MCTS variants (plain, Gumbel, heuristic, macro-goal) and the
//! opening book / policy assembly / agent wiring around them. Orchestration,
//! not a domain — cuts across combat/movement/economy/belief the same way
//! each of those buckets' own `derived.rs` cuts across complexity.

pub mod lane;
pub mod book;
pub mod brain;
pub mod goal_aux;
pub mod gumbel_mcts;
pub mod gumbel_qtransform;
pub mod heuristic_mcts;
pub mod macro_agent;
pub mod macro_exec;
pub mod macro_mcts;
pub mod mcts;
pub mod mcts_common;
pub mod mcts_types;
pub mod mcts_zero;
pub mod policy_composer;
