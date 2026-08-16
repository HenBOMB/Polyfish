//! EXP_ELO_026 "oracle macro": a hand-scripted macro layer over the unchanged
//! net, testing whether third-city reach fails at the macro level (commitment
//! and star allocation) rather than micro execution. Two independent steers,
//! both inference-only: an expansion commitment (focus the pursuit channel on
//! one sticky capturable village) and a star gate (drop root tech purchases
//! that would leave the capture unfunded). Nothing here touches training.
//!
//! Aug 2026: the Archetype (T1) selector split to `search::archetype`, and
//! `GoalAux`/`scripted_goal_aux`/its gates split to `search::goal_aux`, so no
//! file in `ai::` exceeds ~1000 lines. Both are re-exported below so every
//! `crate::ai::oracle_macro::X` call site keeps resolving unchanged.

use crate::states::{GameState, PlayerId};
use crate::types::StructureType;

/// EXP_ELO_028: order types painted into the goal channels. The discriminant
/// is the channel offset from `features::CH_ORDER_START`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderKind {
    Expand = 0,
    Attack = 1,
    Defend = 2,
}

/// EXP_ELO_028: global spending stance. The discriminant is the channel
/// offset from `features::CH_STANCE_START` (one-hot plane).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stance {
    #[default]
    Grow = 0,
    Arm = 1,
    Unlock = 2,
    /// v7: bank stars toward a named purchase the tribe cannot afford yet.
    /// Held stars appeared nowhere in the potential, so converting them into
    /// any scored asset strictly raised Phi while holding left it flat —
    /// saving was a dominated action by construction, and the measured policy
    /// was hand-to-mouth (median spend/income exactly 1.00). SAVE names the
    /// target so `SHAPE_GOAL_SAVE` can pay the ramp toward it.
    Save = 3,
}

/// EXP_ELO_028 Stage-1 macro goal: concurrent painted orders (each a target
/// tile) plus one global spending stance. Encoded into the appended goal
/// channels; `orders` must stay sorted so identical goals produce identical
/// feature bytes (the eval cache and tree reuse hash them).
/// Moved to `ai::economy` (Aug 2026 taxonomy reorg), re-exported here since
/// `MacroGoal.save_target` below needs it in scope.
pub use crate::ai::economy::SaveLane;

#[derive(Clone, Debug, PartialEq, Default)]
pub struct MacroGoal {
    pub orders: Vec<(OrderKind, i32)>,
    pub stance: Stance,
    /// v7: the economy batch this seat is banking for while the stance is
    /// SAVE. Not encoded into the feature planes (the stance one-hot carries
    /// the categorical); `reward::goal_potential` reads it to pay the savings
    /// ramp and `advances_save_plan` reads it to boost the plan's own moves.
    pub save_target: Option<SaveLane>,
}

/// Stage-1 scripted goal-setter, v2 (recalibrated Jul 29 after the iter-1..4
/// channel audit showed ATTACK lit on 62% of plies): EXPAND on every
/// capturable village until captured; ATTACK only with local force
/// superiority; DEFEND unchanged; ARM gains a post-expansion "prepare" phase.
pub fn scripted_goal(
    state: &GameState,
    player: PlayerId,
    tier3_bought: u32,
    lane: Option<Archetype>,
) -> MacroGoal {
    let size = state.settings.size as i32;
    let cheb =
        |a: i32, b: i32| crate::functions::get_chebyshev_distance(a, b, size);
    let Some(tribe) = state.tribes.get(&player) else {
        return MacroGoal::default();
    };
    // Engine accounting: cost + passenger, zero once converted.
    let unit_cost = |u: &crate::states::UnitState| crate::rules::combat::unit_worth(u);
    let own_units: Vec<(i32, i32)> =
        tribe.units.iter().map(|u| (u.coords.idx, unit_cost(u))).collect();
    let our_army: i32 = own_units.iter().map(|(_, c)| c).sum();
    let mut orders: Vec<(OrderKind, i32)> = Vec::new();

    for &idx in state.structures.keys() {
        if still_capturable(state, idx, player) || retakeable_village(state, idx, player) {
            orders.push((OrderKind::Expand, idx));
        }
    }
    // v2.4: while expanding, keep at least EXPAND_TARGET_MIN targets painted —
    // generator-informed guesses stand in for undiscovered villages, so the
    // approach gradient drives scouting toward likely sites instead of idling.
    if tribe.cities.len() < COMMIT_CITY_TARGET && orders.len() < EXPAND_TARGET_MIN {
        for g in guess_villages(state, player, EXPAND_TARGET_MIN - orders.len()) {
            orders.push((OrderKind::Expand, g.tile));
        }
    }

    // ATTACK needs assembled superiority; a merely winnable-if-massed city
    // sets `prepare` instead (post-expansion ARM below). Defender count is
    // ground truth, not FOW-filtered — acceptable script approximation.
    let mut prepare = false;
    for (id, t) in &state.tribes {
        if *id == player {
            continue;
        }
        for c in &t.cities {
            let explored = state
                .tiles
                .get(&c.idx)
                .map_or(false, |tl| tl.explorers.contains(&player));
            if !explored {
                continue;
            }
            let local: Vec<i32> = own_units
                .iter()
                .filter(|(u, _)| cheb(*u, c.idx) <= 3)
                .map(|(_, cost)| *cost)
                .collect();
            let defenders: i32 = t
                .units
                .iter()
                .filter(|u| cheb(u.coords.idx, c.idx) <= 2)
                .map(unit_cost)
                .sum();
            // 1.5x margin (v2.1): proximity superiority alone kept ATTACK lit
            // on 36-40% of plies; a real edge should be decisive, not marginal.
            if local.len() >= 2 && 2 * local.iter().sum::<i32>() > 3 * defenders {
                orders.push((OrderKind::Attack, c.idx));
            } else if our_army > defenders
                && own_units.iter().any(|(u, _)| cheb(*u, c.idx) <= 4)
            {
                prepare = true;
            }
        }
    }
    // EXP_ELO_040/050: threat-driven Defend, from the single unified risk
    // model (city_risks — EXP_ELO_054 folded the separate strike-only
    // city_threats model into it). `needs_order()` covers both a sieged or
    // next-turn-reachable-and-open city and a garrison under near-lethal
    // strike; the old `near >= 2` proxy was blind to a single sieging unit
    // (fixture 1786670356), and a strike-only model was blind to an EMPTY
    // reachable city (seed-1786807403, capital lost on t9 while the
    // directive read Grow/Expand).
    for r in crate::ai::combat::city_risks(state, player) {
        if r.needs_order() {
            orders.push((OrderKind::Defend, r.city));
        }
    }

    orders.sort();
    // v7: SAVE sits below both ARM branches — a threat or a committed push
    // always outranks banking — and only fires for a batch that is out of
    // pocket now but inside SAVE_MAX_TURNS of income, so it self-terminates
    // rather than becoming an open-ended hoard.
    // EXP_ELO_052: bank for the lane T1 committed to, not for whichever
    // structure happens to be cheapest.
    let save_target = save_batch_plan(state, player, tier3_bought, lane).filter(|l| {
        let spt = crate::functions::get_tribe_spt(state, tribe);
        tribe.stars < l.cost && tribe.stars + spt * SAVE_MAX_TURNS >= l.cost
    });
    let stance = if orders.iter().any(|(k, _)| *k == OrderKind::Defend) {
        Stance::Arm
    } else if prepare && tribe.cities.len() >= COMMIT_CITY_TARGET {
        Stance::Arm
    } else if save_target.is_some() {
        Stance::Save
    } else {
        Stance::Grow
    };
    MacroGoal { orders, stance, save_target }
}

/// Why ARM is elevated. Threat and momentum both want giants, so the planner
/// can read `arm` alone — but they want OPPOSITE economy behaviour (under
/// threat you need stars now; with momentum you can afford to invest), so the
/// cause is kept separate rather than collapsed into the magnitude.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ArmCause {
    #[default]
    None,
    Threat,
    Momentum,
}

/// Continuous magnitudes behind the categorical `Stance`. The if-else ladder in
/// `scripted_goal` thresholds these away — "enemy near a city" and "crushing
/// attack advantage" both emit a bare `Stance::Arm` — so anything that needs to
/// know HOW military the position is has to recompute them. Read-only; nothing
/// in search or the feature planes consumes this yet.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct StanceStrength {
    /// 0 = no military pressure or opportunity, 1 = maximal.
    pub arm: f32,
    /// 0 = no economic upside available, 1 = ample.
    pub grow: f32,
    pub cause: ArmCause,
}

/// Pop the planner treats as "ample immediate economy" for normalisation.
const GROW_POP_FULL: f32 = 5.0;
/// Capturable targets that count as full expansion pressure.
const EXPAND_FULL: f32 = 3.0;
/// Turns of income counted as spendable when sizing economic upside.
const GROW_HORIZON_TURNS: i32 = 3;

/// Magnitudes behind the stance, derived from the same signals the stance
/// ladder tests. Pure function of state.
pub fn stance_strength(state: &GameState, player: PlayerId) -> StanceStrength {
    let size = state.settings.size as i32;
    let cheb =
        |a: i32, b: i32| crate::functions::get_chebyshev_distance(a, b, size);
    let Some(tribe) = state.tribes.get(&player) else {
        return StanceStrength::default();
    };
    // Engine accounting: cost + passenger, zero once converted.
    let unit_cost = |u: &crate::states::UnitState| crate::rules::combat::unit_worth(u);

    let our_army: i32 = tribe.units.iter().map(unit_cost).sum();
    let their_army: i32 = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .flat_map(|(_, t)| t.units.iter())
        .map(unit_cost)
        .sum();

    // THREAT: how much of my territory is contested, weighted by who holds the
    // local balance. All cities pressed by a force I cannot match -> 1.0.
    let (mut threatened, mut enemy_near, mut own_near) = (0, 0, 0);
    for c in &tribe.cities {
        let e: i32 = state
            .tribes
            .iter()
            .filter(|(id, _)| **id != player)
            .flat_map(|(_, t)| t.units.iter())
            .filter(|u| cheb(u.coords.idx, c.idx) <= 2)
            .map(unit_cost)
            .sum();
        if e > 0 {
            threatened += 1;
            enemy_near += e;
            own_near += tribe
                .units
                .iter()
                .filter(|u| cheb(u.coords.idx, c.idx) <= 2)
                .map(unit_cost)
                .sum::<i32>();
        }
    }
    let threat = if tribe.cities.is_empty() || threatened == 0 {
        0.0
    } else {
        let frac = threatened as f32 / tribe.cities.len() as f32;
        let ratio = enemy_near as f32 / (enemy_near + own_near).max(1) as f32;
        (frac * ratio).clamp(0.0, 1.0)
    };

    // MOMENTUM: army edge over the opponent, scaled by whether there is
    // anything to spend it on. Parity or worse is no momentum at all.
    let edge = if our_army + their_army == 0 {
        0.0
    } else {
        let share = our_army as f32 / (our_army + their_army) as f32;
        ((share - 0.5) * 2.0).clamp(0.0, 1.0)
    };
    let mut attackable = 0;
    let mut enemy_cities = 0;
    for (id, t) in &state.tribes {
        if *id == player {
            continue;
        }
        for c in &t.cities {
            enemy_cities += 1;
            let local: Vec<i32> = tribe
                .units
                .iter()
                .filter(|u| cheb(u.coords.idx, c.idx) <= 3)
                .map(unit_cost)
                .collect();
            let defenders: i32 = t
                .units
                .iter()
                .filter(|u| cheb(u.coords.idx, c.idx) <= 2)
                .map(unit_cost)
                .sum();
            // Same 1.5x margin the ATTACK order uses.
            if local.len() >= 2 && 2 * local.iter().sum::<i32>() > 3 * defenders {
                attackable += 1;
            }
        }
    }
    let opportunity = if enemy_cities == 0 {
        0.0
    } else {
        (0.5 + 0.5 * attackable as f32 / enemy_cities as f32).min(1.0)
    };
    let momentum = (edge * opportunity).clamp(0.0, 1.0);

    // GROW: pop I could convert stars into over the next few turns, plus open
    // expansion targets. Uses the same knapsack the evaluator prices cities with.
    let spt = crate::functions::get_tribe_spt(state, tribe);
    let budget = tribe.stars + spt * GROW_HORIZON_TURNS;
    let buyable = tribe
        .cities
        .iter()
        .map(|c| crate::ai::reward::max_affordable_pop(state, player, c, budget))
        .max()
        .unwrap_or(0);
    let expandable = state
        .structures
        .keys()
        .filter(|&&idx| still_capturable(state, idx, player))
        .count();
    let grow = (buyable as f32 / GROW_POP_FULL)
        .max(expandable as f32 / EXPAND_FULL)
        .clamp(0.0, 1.0);

    let (arm, cause) = if threat >= momentum {
        (threat, if threat > 0.0 { ArmCause::Threat } else { ArmCause::None })
    } else {
        (momentum, ArmCause::Momentum)
    };
    StanceStrength { arm, grow, cause }
}

/// Turns a discretionary challenger stance must hold before it takes over.
/// Threat responses bypass this entirely — see `update_goal`.
pub const STANCE_SWITCH_TURNS: u8 = 2;

/// Moved to `ai::economy` (Aug 2026 taxonomy reorg): SAVE_MIN_PARTNERS,
/// lane_investment, potential_partner_count, SAVE_MAX_TURNS,
/// SAVE_MAX_PLACEMENTS, SAVE_LANES, lane_yield_per_star, lane_save_structure,
/// tech_chain_cost, save_batch_plan, advances_save_plan, recommended_techs.
pub use crate::ai::economy::{
    advances_save_plan, lane_investment, lane_save_structure, recommended_techs, save_batch_plan,
    tech_chain_cost, SAVE_MAX_PLACEMENTS, SAVE_MAX_TURNS, SAVE_MIN_PARTNERS,
};

/// v7: the STANDING macro commitment.
///
/// `scripted_goal` is a pure function of the current state and was recomputed
/// every ply, so the "strategy" could contradict itself between plies of the
/// same turn — a reflex, not a plan. Nothing that persists can be committed to,
/// and nothing that flips can be rewarded for being held. This carries the
/// stance across plies with the same hysteresis `ArchetypeState` already uses
/// for doctrine, and counts the flip rates EXP_ELO_028 registered as
/// first-class metrics and never measured.
#[derive(Clone, Debug, Default)]
pub struct StanceCommit {
    /// EXP_ELO_052: the Tier-1 lane this seat has committed to, so the
    /// savings plan banks for what T1 actually chose rather than inferring
    /// it from which techs happen to be owned. `None` falls back to the
    /// spawn prior, then to price.
    pub lane: Option<Archetype>,
    pub stance: Option<Stance>,
    challenger: Option<Stance>,
    streak: u8,
    last_turn: i32,
    last_orders: Vec<(OrderKind, i32)>,
    /// Turns on which the committed stance actually changed.
    pub stance_flips: u32,
    /// Turns on which the painted order set changed.
    pub order_flips: u32,
    /// Turns observed, the denominator for both rates.
    pub turns_seen: u32,
}

/// v7: the goal-setter with memory. Returns the scripted orders unchanged (a
/// painted target is already persistent while it stays capturable) but resolves
/// the stance through `st`, so a discretionary swing must hold for
/// `STANCE_SWITCH_TURNS` turns before it lands.
///
/// Asymmetric on purpose: a DEFEND order means an enemy is inside our cities'
/// threat radius, and a threat response that waits out a hysteresis window is
/// a threat response that arrives after the city falls. Those switch instantly;
/// only discretionary changes are damped.
pub fn update_goal(
    state: &GameState,
    player: PlayerId,
    st: &mut StanceCommit,
    tier3_bought: u32,
) -> MacroGoal {
    let mut goal = scripted_goal(state, player, tier3_bought, st.lane);
    let turn = state.settings.turn;
    let new_turn = turn != st.last_turn;
    if new_turn {
        st.turns_seen = st.turns_seen.saturating_add(1);
        if !st.last_orders.is_empty() && st.last_orders != goal.orders {
            st.order_flips = st.order_flips.saturating_add(1);
        }
        st.last_orders = goal.orders.clone();
    }
    st.last_turn = turn;

    let urgent = goal.orders.iter().any(|(k, _)| *k == OrderKind::Defend);
    let fresh = goal.stance;
    match st.stance {
        None => {
            st.stance = Some(fresh);
            st.challenger = None;
            st.streak = 0;
        }
        Some(cur) if fresh == cur => {
            st.challenger = None;
            st.streak = 0;
        }
        Some(cur) => {
            if urgent {
                st.stance = Some(fresh);
                st.challenger = None;
                st.streak = 0;
                if fresh != cur {
                    st.stance_flips = st.stance_flips.saturating_add(1);
                }
            } else {
                if st.challenger == Some(fresh) {
                    if new_turn {
                        st.streak = st.streak.saturating_add(1);
                    }
                } else {
                    st.challenger = Some(fresh);
                    st.streak = 1;
                }
                if st.streak >= STANCE_SWITCH_TURNS {
                    st.stance = Some(fresh);
                    st.challenger = None;
                    st.streak = 0;
                    st.stance_flips = st.stance_flips.saturating_add(1);
                }
            }
        }
    }
    goal.stance = st.stance.unwrap_or(fresh);
    goal
}

/// Minimum EXPAND targets painted while expanding — real villages first,
/// generator-informed guesses fill the remainder (v2.4).
pub const EXPAND_TARGET_MIN: usize = 2;

/// Moved to `belief::prediction` and merged with `predict_villages` (Aug
/// 2026) — one village-guesser instead of two.
pub use crate::ai::belief::prediction::{guess_villages, VillageGuess};

/// Whether the goal-conditioned research gate is active (v2.2, stance-aware):
/// GROW gates during the expansion window (EXPAND painted, under
/// `COMMIT_CITY_TARGET` cities); ARM gates whenever it holds — each stance
/// gates only the tech class that contradicts it (see `passes_star_gate`).
pub fn goal_star_gate(state: &GameState, player: PlayerId, goal: &MacroGoal) -> bool {
    match goal.stance {
        Stance::Grow => {
            // A live batch keeps star discipline on even while growing —
            // otherwise the gate switches off at the third city and the lane
            // stops mattering (Organization on t11 of seed 1786807403).
            goal.save_target.is_some()
                || goal.orders.iter().any(|(k, _)| *k == OrderKind::Expand)
                    && state
                        .tribes
                        .get(&player)
                        .map_or(false, |t| t.cities.len() < COMMIT_CITY_TARGET)
        }
        Stance::Arm => true,
        // v7: banking for a named batch — every star spent elsewhere competes
        // with it, so the gate is unconditionally active.
        Stance::Save => true,
        Stance::Unlock => false,
    }
}

/// City count at which the commitment retires (the third-city objective).
pub const COMMIT_CITY_TARGET: usize = 3;

/// v2.3 tech-discipline crutch: whole-game cap on techs bought with own
/// stars (Research moves; ruin-granted techs don't count) …
pub const TECH_CAP_PER_GAME: u32 = 8;
/// … of which at most this many tier-3 unlocks.
///
/// v7: 1 → 2 (Verdi). One slot forced the economy lane and the knight lane to
/// compete for the same purchase, and the knight lane usually won — Chivalry
/// was the first tier-3 in 7/14 sampled seats while Construction fell to 1.
/// Two slots plus the economy-first ordering below reproduces the real-game
/// pattern: players take the level-3 pop buildings first (they lead to giants)
/// and only then a combat tier-3.
pub const TIER3_CAP_PER_GAME: u32 = 2;

pub use crate::ai::search::archetype::{
    lane_techs, observe_archetype, select_playstyle, tribe_lane_prior, update_archetype,
    Archetype, ArchetypeState, Overlays, ARCH_ENTRY_MIN, ARCH_SWITCH_MARGIN, ARCH_SWITCH_TURNS,
    DWELL_MIN, HEAVY_DEFENSE_MIN, LANES, LANE_BLOCKED_TRIGGER, LANE_ORDER, MAX_PIVOTS,
    METAL_FORGE_MIN, OPEN_FRAC_RIDER, ROUGH_FRAC_ARCHER, SEEN_CAVALRY_SCREEN, SEEN_HEAVY_COUNTER,
    SEEN_SQUISHY_KNIGHT, SQUISHY_DEFENSE_MAX, TRIBE_PRIOR_BONUS,
};
pub use crate::ai::search::goal_aux::{
    market_ready, passes_ability_gate, passes_capture_first, passes_star_gate, passes_tech_caps,
    scripted_goal_aux, GoalAux,
};
pub use crate::ai::movement::connect_remaining;

/// True while `idx` still holds a village capturable by `player`: Village
/// structure on an unowned tile that `player` has explored (the pursuit
/// channel's predicate — see features.rs).
pub fn still_capturable(state: &GameState, idx: i32, player: PlayerId) -> bool {
    crate::rules::capture::is_capturable(
        state,
        idx,
        player,
        crate::rules::capture::CaptureKind::OPEN_VILLAGE,
        true,
    )
}

/// v6: Chebyshev reach within which a lost/enemy-taken village stays a
/// painted retake target — beyond it the pull would become a cross-map
/// crusade holding the GROW window open artificially.
pub const RETAKE_PAINT_RADIUS: i32 = 6;

/// v6: an enemy-captured village worth retaking — explored, enemy-owned
/// (never a capital: those stay Attack-order territory), within
/// RETAKE_PAINT_RADIUS of one of our units or cities. Recapture is a legal
/// CaptureMove; without this the EXPAND painting dropped the tile the
/// moment its owner flipped and the retake went entirely unpriced.
pub fn retakeable_village(state: &GameState, idx: i32, player: PlayerId) -> bool {
    let is_village = state
        .structures
        .get(&idx)
        .and_then(|s| s.as_ref())
        .map_or(false, |s| s.structure_type == StructureType::Village);
    if !is_village {
        return false;
    }
    let Some(tile) = state.tiles.get(&idx) else {
        return false;
    };
    if tile.owner == 0 || tile.owner == player || tile.capital_of != 0 {
        return false;
    }
    if !tile.explorers.contains(&player) {
        return false;
    }
    let size = state.settings.size as i32;
    let Some(tribe) = state.tribes.get(&player) else {
        return false;
    };
    tribe
        .units
        .iter()
        .map(|u| u.coords.idx)
        .chain(tribe.cities.iter().map(|c| c.idx))
        .any(|a| {
            ((a / size) - (idx / size))
                .abs()
                .max(((a % size) - (idx % size)).abs())
                <= RETAKE_PAINT_RADIUS
        })
}

/// Nearest capturable village by Chebyshev distance to any of `player`'s
/// units (fallback anchor: its cities), lowest tile index on ties.
pub fn nearest_capturable_village(state: &GameState, player: PlayerId) -> Option<i32> {
    let size = state.settings.size as i32;
    let tribe = state.tribes.get(&player)?;
    let anchors: Vec<i32> = if tribe.units.is_empty() {
        tribe.cities.iter().map(|c| c.idx).collect()
    } else {
        tribe.units.iter().map(|u| u.coords.idx).collect()
    };
    if anchors.is_empty() {
        return None;
    }
    let cheb =
        |a: i32, b: i32| crate::functions::get_chebyshev_distance(a, b, size);
    state
        .structures
        .keys()
        .filter(|&&idx| still_capturable(state, idx, player))
        .map(|&idx| {
            let d = anchors.iter().map(|&a| cheb(a, idx)).min().unwrap_or(i32::MAX);
            (d, idx)
        })
        .min()
        .map(|(_, idx)| idx)
}

/// Per-decision commitment update: retired at `COMMIT_CITY_TARGET` cities,
/// sticky while the current target stays capturable, else re-picked nearest.
pub fn update_commitment(
    state: &GameState,
    player: PlayerId,
    prev: Option<i32>,
) -> Option<i32> {
    let tribe = state.tribes.get(&player)?;
    if tribe.cities.len() >= COMMIT_CITY_TARGET {
        return None;
    }
    if let Some(idx) = prev {
        if still_capturable(state, idx, player) {
            return Some(idx);
        }
    }
    nearest_capturable_village(state, player)
}

// ======================= Exploration pack (v4 / bucket B) =======================

/// Moved to `ai::movement` (Aug 2026 taxonomy reorg).
pub use crate::ai::movement::assign_expand_targets;


/// Test-only helpers shared by oracle_macro.rs's own tests and by
/// search::archetype / search::goal_aux (split out of this file, Aug 2026).
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::Coords;
    use crate::states::{StructureState, TileState, TribeState, UnitState};
    use crate::types::UnitType;

    pub(crate) fn unit_at(idx: i32) -> UnitState {
        UnitState {
            unit_type: UnitType::Warrior,
            coords: Coords::from_index(idx, 11),
            ..Default::default()
        }
    }
    /// Village structure at `idx`, unowned, explored by player 1.
    pub(crate) fn add_visible_village(state: &mut GameState, idx: i32) {
        state.structures.insert(
            idx,
            Some(StructureState {
                structure_type: StructureType::Village,
                level: 0,
                founded: 0,
            }),
        );
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(idx, tile);
    }
    pub(crate) fn state_with_villages(unit_idx: i32, villages: &[i32]) -> GameState {
        let mut state = GameState::default();
        for &v in villages {
            add_visible_village(&mut state, v);
        }
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(unit_idx));
        state.tribes.insert(1, t1);
        state
    }
    /// Bare explored tile at `idx` (no structure) — for enemy-city visibility.
    pub(crate) fn explore_tile(state: &mut GameState, idx: i32) {
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(idx, tile);
    }
    /// Explored open fields at 22..42 for archetype map reads.
    pub(crate) fn explore_open_fields(state: &mut GameState) {
        for idx in 22..42 {
            let mut tile = TileState::default();
            tile.terrain_type = crate::types::TerrainType::Field;
            tile.explorers.insert(1);
            state.tiles.insert(idx, tile);
        }
    }
}

#[cfg(test)]
#[path = "oracle_macro_tests.rs"]
mod tests;
