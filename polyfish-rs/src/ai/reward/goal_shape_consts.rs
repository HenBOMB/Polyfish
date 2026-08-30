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
///
/// EXP_ELO_097 (Verdi, Aug 2026): cut from 700 to a small, deliberately
/// sub-winning base. At 700, this alone (times hidden_frac^2) beat
/// Workshop's own ~150-point pick regardless of the city's position —
/// Explorer won by default everywhere, not just where it should. Verdi's
/// rule: Workshop by default, Explorer only where a real signal (frontier
/// reveal chance or a genuinely hard-to-walk-to lighthouse corner, both
/// below) justifies it. Sized so the base alone stays under Workshop's
/// measured dphi even at hidden_frac = 1.0 (turn 0).
pub const SHAPE_GOAL_EXPLORER: f32 = 80.0;
/// Extra reward on top of the Explorer bonus when a still-hidden map corner
/// is within reach of the explorer's walk, scaled by `walkable_weight`
/// below — a corner an ordinary unit could stroll over to on its own soon
/// is worth much less than one genuinely stranded across water.
pub const SHAPE_GOAL_EXPLORER_LIGHTHOUSE: f32 = 230.0;
/// Floor on `walkable_weight`'s discount: even an easily-walkable corner
/// keeps some lighthouse value (getting the reveal sooner still counts),
/// just not enough on its own to beat Workshop — no hard zero cliff.
pub const EXPLORER_LIGHTHOUSE_WALKABLE_FLOOR: f32 = 0.2;
/// How far away a hidden map corner can be and still count as "within
/// reach" for the explorer bonus above.
pub const EXPLORER_WALK_RANGE: i32 = 5;
/// Caps how many hidden map corners can count toward the explorer bonus at
/// once.
pub const EXPLORER_CORNER_CAP: usize = 2;
/// Scales the frontier-weighted bonus on top of the flat Explorer pick —
/// how much a city's dark neighborhood leans enemy/village vs plain fog.
/// First-fit (Verdi, Aug 2026): sized so a maximally enemy-facing
/// neighborhood (avg weight ~= FRONTIER_W_FOG + FRONTIER_W_ENEMY) lands
/// near the old max lighthouse-in-reach bonus (2 * 230 = 460); dial against
/// measured dq once real games are in.
pub const SHAPE_GOAL_EXPLORER_FRONTIER: f32 = 80.0;
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

/// Shrinks the Explorer reward at the capital specifically, every time —
/// not just its first pick. EXP_ELO_097 (Verdi, Aug 2026): the original
/// v8 intent ("the capital's first reward is a constant, not a choice")
/// was implemented as `tribe.cities.len() <= 1`, a proxy that silently
/// stopped applying the moment a second city existed — even if that
/// second city was captured moments before the capital's OWN first
/// reward (exactly what happened in the seed0 turn-3 game: city 49 was
/// captured right before city 84's reward fired, so this never engaged).
/// Checking the tile's own `capital_of` instead of city count fixes that,
/// and applying it unconditionally (not just "the first reward") matches
/// Verdi's stated rule directly: "Capital almost always workshop."
pub const SHAPE_GOAL_EXPLORER_CAPITAL_SCALE: f32 = 0.15;

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

/// Verdi (Aug 2026): a per-held-unit penalty while a tech goal is known,
/// nothing needs defending, and Expand coverage is already saturated (every
/// live target has a unit converging on it) — a new unit under those three
/// conditions is a star not going toward the plan, not genuine need. Sized
/// above SHAPE_GOAL_RIDER/SHAPE_GOAL_BODY so it wins the tradeoff outright
/// rather than merely offsetting them.
pub const SHAPE_GOAL_UNIT_OPPORTUNITY_COST: f32 = 120.0;

// ---- Frontier weighting (Verdi, Aug 2026) --------------------------------
// What "forward" means, in one place: enemy-facing unexplored ground is the
// hardest information to get any other way (can't be walked to safely),
// a possible village site is worth more than plain fog (economic upside),
// and plain fog is still worth something over already-explored ground.
// Consumed by both the explorer term (goal_potential.rs) and the per-city
// completion weighting (economy_completion.rs) via `frontier_weight` below
// — one definition, not two drifting copies.
pub const FRONTIER_W_FOG: f32 = 1.0;
pub const FRONTIER_W_VILLAGE: f32 = 2.0;
pub const FRONTIER_W_ENEMY: f32 = 6.0;

/// P(this tile is worth revealing), tiered fog < village < enemy territory.
/// `belief` is a pure function of currently-explored tiles (`MapBelief`), so
/// this is itself pure — safe to hoist once per ply, same as `threats`.
pub fn frontier_weight(belief: &crate::ai::belief::map::MapBelief, idx: i32) -> f32 {
    FRONTIER_W_FOG
        + FRONTIER_W_VILLAGE * belief.p_village(idx)
        + FRONTIER_W_ENEMY * belief.p_opponent_affinity(idx)
}

/// Chebyshev radius `avg_frontier_in_reach` scans around a city — same
/// range the explorer term reasons over, so "which city should level up"
/// and "what's an explorer pick from there worth" read the same ground.
pub const COMPLETION_FRONTIER_RANGE: i32 = EXPLORER_WALK_RANGE;

/// Mean tile in a frontier-facing neighborhood pulls `city_completion_weight`
/// up to roughly `1 + this * (FRONTIER_W_ENEMY)`; first-fit (Verdi, Aug 2026)
/// at a modest lift so completion progress still dominated by raw pop, not
/// overridden by geography.
pub const COMPLETION_FRONTIER_W: f32 = 0.15;

/// Per-level decay on the frontier lift: a level-0 city's first upgrade
/// (still holding its whole reward menu, Explorer included) matters most;
/// by level 3+ the menu is mostly spent and raw SPT should dominate again.
/// First-fit — Verdi's asked-for ordering rules are still TBD.
pub const COMPLETION_LEVEL_DECAY: f32 = 0.6;

/// Mean `frontier_weight` over still-dark tiles within `radius` of `center`
/// — the same neighborhood scan the explorer term and the per-city
/// completion weight both key off. Plain grid lookups against `belief`'s
/// dense arrays, no pathfinding: bounded by `(2*radius+1)^2` and cheap
/// enough to run on every gated ply (see `goal_potential_with_belief`'s
/// hoisting doc). Returns `FRONTIER_W_FOG` (the fully-lit baseline) if
/// nothing in range is still dark.
pub fn avg_frontier_in_reach(
    state: &crate::states::GameState,
    belief: &crate::ai::belief::map::MapBelief,
    center: i32,
    radius: i32,
) -> f32 {
    let width = state.settings.size;
    if width <= 0 {
        return FRONTIER_W_FOG;
    }
    let (cx, cy) = (center % width, center / width);
    let mut sum = 0.0f32;
    let mut n = 0u32;
    for y in (cy - radius).max(0)..=(cy + radius).min(width - 1) {
        for x in (cx - radius).max(0)..=(cx + radius).min(width - 1) {
            if (x - cx).abs().max((y - cy).abs()) > radius {
                continue;
            }
            let idx = y * width + x;
            let dark = !state
                .tiles
                .get(&idx)
                .map_or(true, |t| t.explorers.contains(&belief.observer));
            if dark {
                sum += frontier_weight(belief, idx);
                n += 1;
            }
        }
    }
    if n == 0 {
        FRONTIER_W_FOG
    } else {
        sum / n as f32
    }
}

/// EXP_ELO_097: cheap proxy for "an ordinary walking unit could plausibly
/// reach `to` from `from` on its own soon" — Verdi's "not likely to be able
/// to walk into and reveal" clause on the lighthouse bonus. A straight-line
/// sample (fixed step count, no pathfinding, no HashMap/HashSet iteration —
/// see EXP_ELO_091) between the two points; each sampled tile's terrain is
/// checked against the tribe's current water/ocean-crossing tech, mirroring
/// `moves/mod.rs`'s land-unit terrain rule for a plain unit (no Fly/
/// Navigate/Water skill — that's what "just walk a unit over" means)
/// without depending on that private function. Returns a continuous weight
/// in `[EXPLORER_LIGHTHOUSE_WALKABLE_FLOOR, 1.0]`: the floor when the whole
/// line is already walkable, rising toward 1.0 as more of it is blocked by
/// water the tribe can't yet cross.
pub fn walkable_weight(state: &GameState, tribe: &crate::states::TribeState, from: i32, to: i32) -> f32 {
    let width = state.settings.size;
    if width <= 0 {
        return EXPLORER_LIGHTHOUSE_WALKABLE_FLOOR;
    }
    let (x0, y0) = (from % width, from / width);
    let (x1, y1) = (to % width, to / width);
    let steps = (x1 - x0).abs().max((y1 - y0).abs()).max(1);
    let crosses = |terrain: crate::types::TerrainType| -> bool {
        use crate::types::TerrainType;
        match terrain {
            TerrainType::Water | TerrainType::Ocean => tribe.tech_vanilla.iter().any(|t| {
                crate::settings::technology::get_technology_setting(t.tech_type).unlocks_terrain
                    == Some(terrain)
            }),
            _ => true,
        }
    };
    let mut blocked = 0u32;
    for i in 1..=steps {
        let t = i as f32 / steps as f32;
        let x = x0 + ((x1 - x0) as f32 * t).round() as i32;
        let y = y0 + ((y1 - y0) as f32 * t).round() as i32;
        let terrain = state
            .tiles
            .get(&(y * width + x))
            .map(|t| t.terrain_type)
            .unwrap_or_default();
        if !crosses(terrain) {
            blocked += 1;
        }
    }
    let blocked_frac = blocked as f32 / steps as f32;
    EXPLORER_LIGHTHOUSE_WALKABLE_FLOOR + (1.0 - EXPLORER_LIGHTHOUSE_WALKABLE_FLOOR) * blocked_frac
}
