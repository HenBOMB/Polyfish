//! Deterministic economy planner: for each city, the star-efficient frontier of
//! population, city level, giants and SPT, under a set of named strategies.
//!
//! Exists to be the ground truth the evaluator is checked against — every rule
//! is read from the `settings` tables and the engine's own helpers rather than
//! re-derived, so it cannot drift from the game.
//!
//! Two facts make the search tractable:
//!   * adjacency pay is retroactive (`actions::structure`), so a hub's pop is
//!     its adjacent partner count at the END — total pop is a function of the
//!     built SET, not the build order;
//!   * given fixed hub sites every other tile is independent, so the cost→pop
//!     curve is just the tiles sorted by stars-per-pop.
//! Hub sites are the only coupled choice, and they are enumerated.
//!
//! The debug/verification/reporting half (--verify, --explain, --optimal, all
//! print_* output) lives in `src/bin/eco_plan.rs`, which is a thin CLI over
//! this module. Nothing here prints; nothing here is CLI-shaped.
//!
//! Shape: shared vocabulary (`Buy`/`Lane`/`Scenario`/`Goal`) lives here;
//! [`tech`] prices tech chains, [`city`] plans one city's build, [`empire`]
//! allocates territory and enumerates the joint Pareto frontier across cities.

pub mod city;
pub mod empire;
pub mod tech;

pub use city::*;
pub use empire::*;
pub use tech::*;

use crate::types::TechnologyType;

/// Pop to reach the level-4 reward slot, where BorderGrowth/PopGrowth is offered.
pub const POP_FOR_LEVEL_4: i32 = 9;

/// Does this buy place the structure that feeds a hub? Conversions prefix the
/// name (`burn+Farm`), so the match is on the suffix.
pub fn is_partner_buy(b: &Buy, partner_name: &str) -> bool {
    b.what == partner_name || b.what.ends_with(partner_name)
}

/// One buyable thing on one tile.
#[derive(Clone, Debug)]
pub struct Buy {
    pub idx: i32,
    pub what: &'static str,
    pub cost: i32,
    pub pop: i32,
    /// Places a structure, so it competes with a hub for the tile. Harvests
    /// (Fruit/Game/Fish) leave the tile empty and do not.
    pub occupies: bool,
    /// Techs this buy needs. Charged only when the buy is actually taken, so a
    /// plan that skips the Mountains never pays for Mining.
    pub techs: Vec<TechnologyType>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Lane {
    Forest,
    Farm,
    /// Forge eats Mines and pays 2 pop per partner — double the other hubs —
    /// but only a mountainous city can feed it.
    Mine,
}

#[derive(Clone, Copy)]
pub struct Scenario {
    pub name: &'static str,
    pub lane: Lane,
    pub border_growth: bool,
    /// Forest->Field+Crop (BurnForest, with Construction) for the farm lane, or
    /// Field->Forest (GrowForest, needs Spiritualism) for the forest lane.
    pub convert: bool,
}

pub const SCENARIOS: [Scenario; 8] = [
    Scenario { name: "sawmill natural",      lane: Lane::Forest, border_growth: false, convert: false },
    Scenario { name: "sawmill +border",      lane: Lane::Forest, border_growth: true,  convert: false },
    Scenario { name: "sawmill max greed",    lane: Lane::Forest, border_growth: true,  convert: true  },
    Scenario { name: "windmill natural",     lane: Lane::Farm,   border_growth: false, convert: false },
    Scenario { name: "windmill +border",     lane: Lane::Farm,   border_growth: true,  convert: false },
    Scenario { name: "windmill max greed",   lane: Lane::Farm,   border_growth: true,  convert: true  },
    // Forge needs no terrain conversion: it sits on Field or Forest already,
    // and its partners are Mines on mountains, which nothing converts.
    Scenario { name: "forge natural",        lane: Lane::Mine,   border_growth: false, convert: false },
    Scenario { name: "forge +border",        lane: Lane::Mine,   border_growth: true,  convert: false },
];

/// What a plan is optimised FOR. The frontier answers "what are the options";
/// these answer "give me the build for what I need right now".
/// Two ceilings and three knees. The ceilings say what the map can do at any
/// price; the knees say what it is worth paying for, on income, on army, and on
/// the two together.
#[derive(Clone, Copy, PartialEq)]
pub enum Goal {
    Spt,
    Eco,
    Balanced,
    Army,
    Giants,
}
