mod expand;
mod root;
mod stats;
mod wave;

use super::node::Node;
use crate::states::PlayerId;
use stats::MacroMctsStats;

pub(crate) struct MacroMctsSearch<'a> {
    // Macro search nodes.
    nodes: Vec<Node>,
    // Player whose edges get delta-phi shaping rewards.
    pov: PlayerId,
    // Leaf evaluator for scoring (heuristic or NN).
    eval: &'a crate::ai::eval_server::Evaluator,
    // Macro search leaf selection mode.
    leaf: crate::ai::macro_agent::MacroLeaf,
    // Macro search statistics for this run.
    pub stats: MacroMctsStats,
}
