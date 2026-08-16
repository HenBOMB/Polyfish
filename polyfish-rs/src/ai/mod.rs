pub mod belief;
pub mod decision_trace;
pub mod defense;
pub mod economy;
pub mod eval_backend;
pub mod eval_server;
pub mod evaluator;
pub mod features;
pub mod mapper;
pub mod movement;
pub mod network;
pub mod oracle_macro;
pub mod search;
#[cfg(feature = "tch-eval")]
pub mod tch_network;
#[cfg(feature = "metal-eval")]
pub mod metal_network;
pub mod scoring;
pub mod reward;
// Flat re-exports so every existing `crate::ai::gumbel_mcts::X`-style path
// (self_play, arena, replayer, defense, reward, ...) keeps resolving after
// these moved into the search/ bucket — see ok-lets-draft-a-gleaming-emerson.
pub use search::{
    book, brain, gumbel_mcts, gumbel_qtransform, heuristic_mcts, macro_agent, macro_exec,
    macro_mcts, mcts, mcts_common, mcts_types, mcts_zero, policy_composer,
};
pub use evaluator::evaluate_state;
pub use mcts::{MctsAgent, MctsAnalysis, MoveEvaluation};
