//! EXP_ELO_028 Phase 1c goal-priced in-tree shaping constants (Aug 2026
//! taxonomy split out of reward.rs to keep every file under ~1000 lines): the
//! painted macro goal's actuator prices, plus `save_progress`, the v10 SAVE
//! ramp helper sized against them. Pure data + one small pure function — no
//! logic changes from the original reward.rs, just relocated. Re-exported
//! through `reward` so existing `crate::ai::reward::X` call sites keep
//! resolving.

use crate::states::GameState;

// ---- EXP_ELO_028 Phase 1c: goal-priced in-tree shaping ------------------
// The painted macro goal gets an actuator: each stance/order prices the
// resource conversion it names, as a potential on the goal-holder's side
// only (the opponent's goal is unknown at search time). Sized like
// SHAPE_PURSUIT_PER_TILE — large enough to flip a decisive Q gap (~0.15-0.2
// normalized through score_norm≈600-700).

/// Rewards higher star income while the plan is to grow or save — the more
/// income, the bigger the reward.
pub const SHAPE_GOAL_SPT: f32 = 150.0;
/// Rewards having a stronger living army while the plan is to arm for war,
/// on top of what the game's normal score already gives units.
pub const SHAPE_GOAL_ARM_PER_COST: f32 = 50.0;
/// Also rewards star income while arming, so the economy isn't ignored just
/// because you're building an army.
pub const SHAPE_GOAL_ARM_SPT: f32 = 75.0;
/// Rewards getting closer to a tile you're expanding into — capturing it
/// locks in the full reward instead of losing it.
pub const SHAPE_GOAL_EXPAND_PER_TILE: f32 = 200.0;
/// Rewards researching techs that fit your current map and situation well.
pub const SHAPE_GOAL_TECH_FIT: f32 = 150.0;
/// Penalizes a city for every road tile still missing to connect it to your
/// capital — the penalty shrinks one tile at a time as the road is built.
pub const SHAPE_GOAL_CONNECT: f32 = 120.0;
/// Rewards exploring new map tiles early in the game — worth less once you
/// already have a specific place you're expanding to.
pub const SHAPE_GOAL_SCOUT: f32 = 25.0;
/// Extra reward the moment you actually capture an expansion target, on top
/// of the distance reward above — makes capturing clearly beat waiting.
pub const SHAPE_GOAL_EXPAND_DONE: f32 = 2.0;
/// Same reward as the tile-distance expansion bonus above, just calculated a
/// different way internally — no difference in effect.
pub const SHAPE_UNIT_GOAL_PER_TILE: f32 = 200.0;
/// Same as the capture bonus above, for that same internal variant.
pub const SHAPE_UNIT_GOAL_COMPLETE: f32 = 2.0;
/// Rewards having Riders alive when a Rider would clearly reach your
/// expansion target faster than walking there.
pub const SHAPE_GOAL_RIDER: f32 = 100.0;
/// Rewards having units alive that match your current tech strategy's
/// preferred unit types — pricier units earn more.
pub const SHAPE_GOAL_LANE_PER_COST: f32 = 33.0;

/// Limits how many explored tiles in one quarter of the map count toward the
/// exploring reward, so a fresh unexplored area keeps paying too.
pub const SCOUT_QUADRANT_CAP: i32 = 20;

/// Reward for uncovering a hidden map corner where lighthouses are placed, giving 1 pop.
pub const SHAPE_GOAL_LIGHTHOUSE: f32 = 120.0;

/// Rewards choosing the Explorer city reward — worth a lot when most of the
/// map is still hidden, almost nothing once it's mostly explored.
pub const SHAPE_GOAL_EXPLORER: f32 = 700.0;
/// Extra reward on top of the Explorer bonus when a still-hidden map corner
/// is within reach of the explorer's walk.
pub const SHAPE_GOAL_EXPLORER_LIGHTHOUSE: f32 = 230.0;
/// How far away a hidden map corner can be and still count as "within
/// reach" for the explorer bonus above.
pub const EXPLORER_WALK_RANGE: i32 = 5;
/// Caps how many hidden map corners can count toward the explorer bonus at
/// once.
pub const EXPLORER_CORNER_CAP: usize = 2;
/// Rewards placing pop-boosting buildings (Windmill, Sawmill, Forge) next to
/// more than one matching structure — each extra neighbor pays more.
pub const SHAPE_GOAL_YIELD_ADJ: f32 = 100.0;
/// Gives partial credit for a building spot that could still gain more
/// neighbors later, not just the ones already built.
pub const SHAPE_GOAL_YIELD_CAPACITY_W: f32 = 0.5;
/// Same idea as the pop-building bonus above, but for star-boosting
/// buildings like Market — worth half as much.
pub const SHAPE_GOAL_YIELD_ADJ_STARS: f32 = 50.0;
/// Rewards keeping a forest tile standing instead of clearing or burning it
/// — the reward is lost the moment it's cut down.
pub const SHAPE_GOAL_FOREST_STANDING: f32 = 50.0;

/// Shrinks the Explorer reward for your very first city reward pick — that
/// pick isn't a real choice yet, so it shouldn't be valued like one.
pub const SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE: f32 = 0.15;

/// Penalizes a city for having growth progress it can no longer finish
/// because the resources it needs have run out. One penalty per city max,
/// and it's waived while the city is under threat.
pub const SHAPE_GOAL_STRANDED: f32 = 75.0;

/// Rewards banking stars toward a specific big purchase you're saving for —
/// the closer you are to affording it, the bigger the reward. Kept small so
/// actually buying it still beats just sitting on the stars.
pub const SHAPE_GOAL_SAVE: f32 = 100.0;

/// Fraction of the SAVE batch already secured, counting BOUGHT components and
/// not just the star balance.
///
/// v10: the ramp used to read `stars/cost` alone, so buying the lane tech —
/// the whole point of banking — dropped Φ by the price of the tech (measured
/// ~76 score-equivalents on the Forge lane at a 21-star target). It paid to
/// accumulate and charged to make progress, and inside a 6.5-ply horizon only
/// the charge is visible. Progress is monotone under plan-advancing purchases.
pub fn save_progress(
    state: &GameState,
    player: i32,
    lane: &crate::ai::oracle_macro::SaveTarget,
) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let mut banked = tribe.stars.max(0);
    if lane.tech_cost > 0
        && crate::settings::technology::has_technology(&tribe.tech_vanilla, lane.tech)
    {
        banked += lane.tech_cost;
    }
    if lane.structure_unit_cost > 0 {
        let built: i32 = tribe
            .cities
            .iter()
            .filter(|c| {
                c._territory.iter().any(|&t| {
                    crate::functions::get_structure_type_at(state, t) == Some(lane.structure)
                })
            })
            .count() as i32;
        banked += built * lane.structure_unit_cost;
    }
    (banked as f32 / lane.cost as f32).clamp(0.0, 1.0)
}

/// Rewards owning a living super unit like a Giant — this is what makes
/// picking a super unit worth it over the alternative city reward.
pub const SHAPE_GOAL_SUPER: f32 = 500.0;

/// Penalizes leaving a city exposed to attack, sized by how much that city
/// is worth losing — active at all times, not just when defending it.
pub const SHAPE_GOAL_CITY_RISK: f32 = 4.0;

/// Penalizes parking a unit on your own city when that blocks you from
/// training a new unit there.
pub const SHAPE_CITY_TRAIN_BLOCKED: f32 = 200.0;

/// Rewards assigning units to defend a threatened city — pays more the more
/// urgent the threat is.
pub const SHAPE_GOAL_DEFEND_COVER: f32 = 600.0;

/// Rewards keeping a unit garrisoned in a city when that city genuinely
/// needs a defender present to stay safe.
pub const SHAPE_GOAL_DEFEND_HOLD: f32 = 400.0;

/// Rewards keeping units focused on attacking a targeted enemy city instead
/// of wandering off — pays up to 4 attackers per target.
pub const SHAPE_GOAL_ATTACK_PRESS: f32 = 500.0;

/// Multiplies the attack reward for a unit already standing on the enemy
/// city itself, so it holds that position instead of leaving early.
pub const SHAPE_GOAL_SIEGE_HOLD_MULT: f32 = 1.5;

/// Shrinks the super-unit reward once your savings plan is almost fully
/// funded, so the pick switches back toward the city reward.
pub const SHAPE_GOAL_SUPER_ECON_DAMP: f32 = 0.6;

/// Rewards growth progress toward a city's next level, but only counts if
/// that level is actually reachable — always worth less than leveling up.
pub const SHAPE_GOAL_COMPLETION: f32 = 75.0;

/// Cuts the expansion reward by a quarter when the target is a village an
/// enemy already took, since retaking it is riskier than a free one.
pub const SHAPE_GOAL_RETAKE_W: f32 = 0.75;

/// Cuts the approach pull toward a Ruin until you have found your first village.
/// A Ruin is one-time, a Village is permanent, so prioritize village early.
/// Approach only — completion bonus for capturing a Ruin is unaffected.
pub const SHAPE_GOAL_RUIN_W: f32 = 0.35;

/// Gives a second unit converging on a contested expansion target (one an
/// enemy is standing on) half the normal reward, so it gets help but not a mob.
pub const SHAPE_GOAL_CONTEST_SECOND: f32 = 0.5;

/// Rewards having more living units early in the game while still expanding
/// or exploring, up to a small cap — makes an extra warrior worth its cost.
pub const SHAPE_GOAL_BODY: f32 = 75.0;

/// The largest the early-game unit-count cap above can grow to as your city
/// count rises.
pub const BODY_CAP_MAX: usize = 3;
