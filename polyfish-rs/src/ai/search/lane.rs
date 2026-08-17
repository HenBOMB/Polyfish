//! T1 doctrine selector (Aug 2026 taxonomy reorg: split out of
//! oracle_macro.rs to keep every file in `ai::` under ~1000 lines). Doctrine
//! chosen from ground-truth predicates, sticky with hysteresis, expressed
//! through tech recommendations, unit pricing, and the stepping-stone tech
//! gate. No new input channels: every predicate is a function of state the
//! net already sees (terrain, ghost units, economy). Re-exported through
//! `oracle_macro` so existing `crate::ai::oracle_macro::X` call sites keep
//! resolving.

use crate::ai::oracle_macro::{expand_targets, COMMIT_CITY_TARGET};
use crate::ai::movement::{rider_turns_saved, RIDER_PUSH_MIN_TURNS_SAVED};
use crate::states::{GameState, PlayerId};
use crate::types::TechnologyType;

// ========================= Lane layer (v3) =========================
// Doctrine chosen from ground-truth predicates, sticky with hysteresis,
// expressed through tech recommendations, unit pricing, and the
// stepping-stone tech gate. No new input channels: every predicate is a
// function of state the net already sees (terrain, ghost units, economy).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// Open map + live expansion race + real route advantage, enemy not
    /// heavy-dominant. Buys Riding→Roads ONLY (FreeSpirit is a stepping
    /// stone redeemed solely by a knight commitment).
    RiderRoads,
    /// Anti-heavy/siege AND push support: range beats high defense, and a
    /// backline wears targets down while warriors advance.
    ArcherLine,
    /// Metal-rich explored map, no immediate threat: Mining→Smithery funds
    /// the giant/swordsman push on this tribe's terrain. Not XinXi-exclusive
    /// in spirit — any tribe whose territory suits a resource-hub economy
    /// (XinXi's Mines here; a forest-rich Kickoo or Imperius could run the
    /// same play through Sawmill) can spam giants, this is just the one
    /// tech chain currently wired to trigger it.
    SpamGiants,
}

/// Reactive overlays — composition adjustments on top of the base doctrine.
/// Monotone within a game: they key off peak seen-counts, so they never flap.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overlays {
    /// Enemy cavalry-heavy → Defenders (bodies deny road corridors, defense 3
    /// punishes cavalry trades, screens our ranged backline).
    pub defender_screen: bool,
    /// Enemy heavy melee (giants/swords) → catapults + archers assist.
    pub catapult_counter: bool,
    /// Enemy squishy spam (defense ≤ 1.5) → knights; Persist chains through
    /// low-defense bodies. Opens the FreeSpirit→Chivalry lane.
    pub knight_commit: bool,
}

/// Per-seat persistent lane state, threaded through the play loop like
/// the tech counters. Peak counts approximate observation memory: a unit
/// seen once stays counted after it retreats into fog (the net's ghost
/// channels carry the same information).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LaneState {
    pub lane: Option<Lane>,
    pub overlays: Overlays,
    pub seen_squishy: u32,
    pub seen_heavy: u32,
    pub seen_cavalry: u32,
    pub seen_ranged: u32,
    challenger: Option<Lane>,
    streak: u8,
    last_turn: i32,
    /// Turn the current lane was committed (Tier-1 tenure).
    pub committed_turn: Option<i32>,
    /// Lane changes spent this game; capped at `MAX_PIVOTS`.
    pub pivots_used: u8,
    /// Turn of the last lane change — enforces `DWELL_MIN`.
    pub last_pivot_turn: i32,
    /// Consecutive turns the lane's next tech was proposed and gate-dropped.
    /// A stranded lane (e.g. pure-eco prerequisites masked by the ARM eco
    /// gate) is evidence for pivoting rather than a reason to keep waiting.
    pub lane_blocked_turns: u8,
    /// Last per-lane scores, in `LANES` order — recorded for the attribution
    /// trace so a lane call is explainable, not merely logged.
    pub last_scores: [f32; LANES],
}

/// Every lane the selector ranks, in a fixed order (also the plane order
/// when the lane is painted into features).
pub const LANES: usize = 3;
pub const LANE_ORDER: [Lane; LANES] =
    [Lane::RiderRoads, Lane::ArcherLine, Lane::SpamGiants];

/// The tech chain that *is* the lane — single source of truth, consumed by
/// `compute_goal_aux`'s recommendations and by the spawn tribe prior.
pub fn lane_techs(a: Lane) -> &'static [TechnologyType] {
    use TechnologyType as T;
    match a {
        Lane::RiderRoads => &[T::Riding, T::Roads],
        Lane::ArcherLine => &[T::Hunting, T::Archery],
        Lane::SpamGiants => &[T::Climbing, T::Mining, T::Smithery],
    }
}

/// Lane the tribe is born into: mapgen stamps one tribe tech at turn 0
/// (`mapgen.rs`), and if it opens a lane's chain that lane starts ahead —
/// "your tribe sets the tone" before any terrain is explored. Derived from
/// `lane_techs`, so the two can never drift apart.
pub fn tribe_lane_prior(state: &GameState, player: PlayerId) -> Option<Lane> {
    let tribe = state.tribes.get(&player)?;
    let spawn_tech: Vec<TechnologyType> = tribe
        .tech_vanilla
        .iter()
        .filter(|t| t.discovered && t.discovered_turn == 0)
        .map(|t| t.tech_type)
        .collect();
    LANE_ORDER
        .iter()
        .copied()
        .find(|a| lane_techs(*a).iter().any(|t| spawn_tech.contains(t)))
}

/// Lane changes allowed per game (Verdi: "pivot at most up to 3 lanes").
pub const MAX_PIVOTS: u8 = 3;
/// Minimum turns a lane must be held before it may be abandoned.
pub const DWELL_MIN: i32 = 5;
/// Turns of a stranded tech lane that count as pivot evidence.
pub const LANE_BLOCKED_TRIGGER: u8 = 3;
/// Score bonus for the tribe's birth lane.
pub const TRIBE_PRIOR_BONUS: i32 = 2;

/// Chainable by knights: at or below this defense a kill likely frees
/// Persist for the next body (Verdi: threshold 1.5 — riders/archers/
/// catapults/knights in; warriors at 2.0 out).
pub const SQUISHY_DEFENSE_MAX: f32 = 1.5;
/// Heavy: defenders/swordsmen (3.0), giants (4.0) — outrange, don't trade.
pub const HEAVY_DEFENSE_MIN: f32 = 3.0;

/// Minimum best-score to commit to a doctrine at all.
pub const LANE_ENTRY_MIN: i32 = 3;
/// A challenger must outscore the incumbent by this margin…
pub const LANE_SWITCH_MARGIN: i32 = 2;
/// …for this many distinct turns before a soft switch (hysteresis).
pub const LANE_SWITCH_TURNS: u8 = 3;
/// Explored-land open-field share for rider terrain.
pub const OPEN_FRAC_RIDER: f32 = 0.45;
/// Explored-land rough share for archer terrain.
pub const ROUGH_FRAC_ARCHER: f32 = 0.30;
/// Peak seen heavy units: fires the catapult overlay AND hard-exits riders.
pub const SEEN_HEAVY_COUNTER: u32 = 2;
/// Peak seen cavalry: fires the defender screen.
pub const SEEN_CAVALRY_SCREEN: u32 = 2;
/// Peak seen squishy units: opens the knight commitment.
pub const SEEN_SQUISHY_KNIGHT: u32 = 4;

/// Explored-map terrain read (FOW-honest, same style as `recommended_techs`).
struct MapRead {
    open_frac: f32,
    rough_frac: f32,
}

fn census_explored_terrain(state: &GameState, player: PlayerId) -> MapRead {
    use crate::types::TerrainType as T;
    let (mut open, mut rough, mut land) = (0i32, 0i32, 0i32);
    for (_, tile) in state.tiles.iter() {
        if !tile.explorers.contains(&player) {
            continue;
        }
        match tile.terrain_type {
            T::Field => {
                open += 1;
                land += 1;
            }
            T::Forest | T::Mountain | T::Wetland | T::Mangrove => {
                rough += 1;
                land += 1;
            }
            _ => {}
        }
    }
    let denom = land.max(1) as f32;
    MapRead { open_frac: open as f32 / denom, rough_frac: rough as f32 / denom }
}

/// Update peak seen-counts from enemy units standing on tiles this player
/// has explored — the script-side proxy for the ghost channels.
fn observe_enemies(state: &GameState, player: PlayerId, st: &mut LaneState) {
    let (mut sq, mut hv, mut cav, mut rng) = (0u32, 0u32, 0u32, 0u32);
    for (id, t) in &state.tribes {
        if *id == player {
            continue;
        }
        for u in &t.units {
            let seen = state
                .tiles
                .get(&u.coords.idx)
                .map_or(false, |tl| tl.explorers.contains(&player));
            if !seen {
                continue;
            }
            let s = crate::settings::units::get_unit_setting(u.unit_type);
            if s.defense <= SQUISHY_DEFENSE_MAX {
                sq += 1;
            }
            if s.defense >= HEAVY_DEFENSE_MIN {
                hv += 1;
            }
            if s.movement >= 2 {
                cav += 1;
            }
            if s.range >= 2 {
                rng += 1;
            }
        }
    }
    st.seen_squishy = st.seen_squishy.max(sq);
    st.seen_heavy = st.seen_heavy.max(hv);
    st.seen_cavalry = st.seen_cavalry.max(cav);
    st.seen_ranged = st.seen_ranged.max(rng);
}

/// Score each doctrine from the predicates. 0 = not viable right now.
///
/// Pure state — no `MacroGoal` input. Until Aug 2026 this read T2's
/// already-assembled `goal.orders` for `race_live`/`mobility`'s targets and
/// for `has_defend`, a T1-depends-on-T2 layering violation with no upside:
/// both signals are available straight from state (`expand_targets` is the
/// same predicate scan `compute_macro_goal` itself runs; `city_risks` is the
/// same pure state read `compute_macro_goal` uses to build Defend orders).
fn lane_scores(state: &GameState, player: PlayerId, st: &LaneState, map: &MapRead) -> [(Lane, i32); 3] {
    let tribe_cities = state.tribes.get(&player).map_or(0, |t| t.cities.len());
    let targets = expand_targets(state, player, None);
    let race_live = tribe_cities < COMMIT_CITY_TARGET || !targets.is_empty();
    let mobility = !targets.is_empty()
        && rider_turns_saved(state, player, &targets) >= RIDER_PUSH_MIN_TURNS_SAVED;

    // Contact: a seen enemy within 3 of our units/cities — the skirmish
    // condition under which an archer backline pays.
    let size = state.settings.size as i32;
    let cheb =
        |a: i32, b: i32| crate::functions::get_chebyshev_distance(a, b, size);
    let contact = state.tribes.iter().filter(|(id, _)| **id != player).any(|(_, t)| {
        t.units.iter().any(|e| {
            let seen = state
                .tiles
                .get(&e.coords.idx)
                .map_or(false, |tl| tl.explorers.contains(&player));
            seen && state.tribes.get(&player).map_or(false, |own| {
                own.units.iter().any(|u| cheb(u.coords.idx, e.coords.idx) <= 3)
                    || own.cities.iter().any(|c| cheb(c.idx, e.coords.idx) <= 3)
            })
        })
    });

    let rider = if st.seen_heavy >= SEEN_HEAVY_COUNTER {
        0 // hard-countered: riders lose into a heavy wall
    } else {
        2 * (map.open_frac >= OPEN_FRAC_RIDER) as i32
            + 2 * mobility as i32
            + race_live as i32
            + (st.seen_ranged >= 2) as i32 // riders punish a ranged backline
    };
    let archer = 2 * (st.seen_heavy >= 1) as i32
        + (map.rough_frac >= ROUGH_FRAC_ARCHER) as i32
        + 2 * contact as i32;
    let has_defend = crate::ai::combat::city_risks(state, player).iter().any(|r| r.needs_order());
    // EXP_ELO_057: whether this territory's best city could produce a giant
    // off a Mine-fed Forge, per `rules::eco_plan::plan_city` (the same
    // single source of truth `bin/eco_plan --verify` checks against) —
    // replacing a flat metal-tile count, which said nothing about whether
    // those tiles could actually feed a hub (adjacency, terrain, already-
    // spoken-for ground).
    let spam_viable = crate::ai::economy::eco_plan_best_city(
        state, player, crate::rules::eco_plan::Lane::Mine,
    )
    .is_some_and(|p| p.giants >= 1);
    let spam = 2 * spam_viable as i32
        + (map.rough_frac >= ROUGH_FRAC_ARCHER) as i32
        + (!has_defend) as i32;
    [
        (Lane::RiderRoads, rider),
        (Lane::ArcherLine, archer),
        (Lane::SpamGiants, spam),
    ]
}

/// Per-ply lane update: observe enemies (peaks), refresh overlays,
/// then enter/hold/switch the base doctrine with hysteresis — hard exits
/// fire immediately (score drops to 0), soft switches need the challenger
/// to outscore by `LANE_SWITCH_MARGIN` for `LANE_SWITCH_TURNS` turns.
/// Per-ply entry point for the script paths: observe every ply, but run the
/// Tier-1 selector only at a turn boundary (or to make the very first
/// commit). Callers that already sit on a turn boundary — the macro agent's
/// replan branch — call `observe_lane_state` + `select_lane` directly.
pub fn update_lane_state(state: &GameState, player: PlayerId, st: &mut LaneState) {
    let before = st.overlays;
    observe_lane_state(state, player, st);
    // Refutation bypasses the turn boundary, mirroring the stance layer's
    // urgent-threat path (`commit_macro_goal`): new counter-evidence — the
    // sighting that flips an overlay is exactly what zeroes a lane's score —
    // must not wait a turn to be acted on. Discretionary switches still wait,
    // and the pivot budget still binds either way.
    let refuted = st.overlays != before;
    if state.settings.turn != st.last_turn || st.lane.is_none() || refuted {
        select_lane(state, player, st, None);
    }
}

/// Per-ply half: peak enemy-type counts and the reactive overlays. Cheap,
/// runs on every executor ply. Selection deliberately does NOT happen here —
/// a lane recomputed 20x a turn is a running average, not an identity.
pub fn observe_lane_state(state: &GameState, player: PlayerId, st: &mut LaneState) {
    observe_enemies(state, player, st);
    st.overlays = Overlays {
        defender_screen: st.seen_cavalry >= SEEN_CAVALRY_SCREEN,
        catapult_counter: st.seen_heavy >= SEEN_HEAVY_COUNTER,
        knight_commit: st.seen_squishy >= SEEN_SQUISHY_KNIGHT,
    };
}

/// Tier-1 selector — called once per turn. Scores EVERY lane (algorithmic
/// census + tribe prior, plus `head` when the net supplies per-lane scores)
/// and returns the committed lane. Switching away from a held lane must
/// additionally clear the budget, the dwell floor, and the existing
/// margin/streak hysteresis, so the call is stable by construction rather
/// than by hoping the scores stay put.
pub fn select_lane(
    state: &GameState,
    player: PlayerId,
    st: &mut LaneState,
    head: Option<&[f32; LANES]>,
) -> Option<Lane> {
    let map = census_explored_terrain(state, player);
    let turn = state.settings.turn;
    let prior = tribe_lane_prior(state, player);

    let mut scores = lane_scores(state, player, st, &map);
    if let Some(p) = prior {
        if let Some(e) = scores.iter_mut().find(|(a, _)| *a == p) {
            e.1 += TRIBE_PRIOR_BONUS;
        }
    }
    for (i, a) in LANE_ORDER.iter().enumerate() {
        let algo = scores.iter().find(|(k, _)| k == a).map_or(0, |(_, s)| *s) as f32;
        st.last_scores[i] = match head {
            Some(h) => algo + h[i],
            None => algo,
        };
    }
    let score_of = |a: Lane| {
        LANE_ORDER.iter().position(|k| *k == a).map_or(0.0, |i| st.last_scores[i])
    };
    let (best, best_score) = LANE_ORDER
        .iter()
        .map(|a| (*a, score_of(*a)))
        .fold((Lane::RiderRoads, f32::NEG_INFINITY), |acc, x| {
            if x.1 > acc.1 { x } else { acc }
        });

    let new_turn = turn != st.last_turn;
    st.last_turn = turn;
    let entry_min = LANE_ENTRY_MIN as f32;

    match st.lane {
        None => {
            let pick = if best_score >= entry_min { Some(best) } else { prior };
            if pick.is_some() {
                st.lane = pick;
                st.committed_turn = Some(turn);
                st.challenger = None;
                st.streak = 0;
            }
        }
        Some(cur) => {
            let budget_left = st.pivots_used < MAX_PIVOTS;
            let dwell_ok = turn - st.last_pivot_turn >= DWELL_MIN;
            let stranded = st.lane_blocked_turns >= LANE_BLOCKED_TRIGGER;
            // Hard exit stays immediate — a lane scored 0 is refuted, not
            // merely out-competed — but still costs budget.
            if score_of(cur) <= 0.0 && budget_left {
                st.lane = (best_score >= entry_min).then_some(best);
                if st.lane.is_some() {
                    st.pivots_used += 1;
                    st.last_pivot_turn = turn;
                    st.committed_turn = Some(turn);
                }
                st.challenger = None;
                st.streak = 0;
                st.lane_blocked_turns = 0;
            } else if budget_left
                && (dwell_ok || stranded)
                && best != cur
                && best_score >= score_of(cur) + LANE_SWITCH_MARGIN as f32
            {
                if st.challenger == Some(best) {
                    if new_turn {
                        st.streak += 1;
                    }
                } else {
                    st.challenger = Some(best);
                    st.streak = 1;
                }
                if st.streak >= LANE_SWITCH_TURNS {
                    st.lane = Some(best);
                    st.pivots_used += 1;
                    st.last_pivot_turn = turn;
                    st.committed_turn = Some(turn);
                    st.challenger = None;
                    st.streak = 0;
                    st.lane_blocked_turns = 0;
                }
            } else {
                st.challenger = None;
                st.streak = 0;
            }
        }
    }
    st.lane
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::oracle_macro::test_support::*;
    use crate::ai::oracle_macro::{compute_macro_goal, compute_goal_aux, passes_tech_purchase_limits};
    use crate::moves::research::ResearchMove;
    use crate::states::{TechnologyState, TribeState};
    use crate::types::{TechnologyType, TribeType, UnitType};

    #[test]
    fn lane_rider_enters_on_open_map_and_hard_exits_on_heavy() {
        let mut state = state_with_villages(0, &[24]);
        state.settings.current_player_turn_id = 1;
        explore_open_fields(&mut state);
        let goal = compute_macro_goal(&state, 1, 0);
        let mut st = LaneState::default();
        update_lane_state(&state, 1, &mut st);
        assert_eq!(st.lane, Some(Lane::RiderRoads));

        // Lane expression: Riding recommended, Rider preferred; FreeSpirit
        // stays blocked without a knight commitment.
        let aux = compute_goal_aux(&state, 1, &goal, 0, 0, Some(&st));
        assert!(aux.recommended_techs.contains(&TechnologyType::Riding));
        assert!(aux.preferred_units.contains(&UnitType::Rider));
        assert!(!passes_tech_purchase_limits(&ResearchMove::new(TechnologyType::FreeSpirit), &aux));

        // Two giants observed → catapult overlay fires and riders hard-exit.
        let mut t2 = TribeState::default();
        for &i in &[23, 25] {
            let mut g = unit_at(i);
            g.unit_type = UnitType::Giant;
            t2.units.push(g);
        }
        state.tribes.insert(2, t2);
        let goal2 = compute_macro_goal(&state, 1, 0);
        // Tier 1: discretionary switches wait for the turn boundary, but
        // REFUTATION does not — the sighting that flips an overlay is what
        // zeroes the lane's score, so it re-selects immediately (same
        // precedent as the stance layer's urgent-threat bypass).
        update_lane_state(&state, 1, &mut st);
        assert!(st.overlays.catapult_counter);
        assert_eq!(st.lane, Some(Lane::ArcherLine), "refutation is immediate");
        assert_eq!(st.pivots_used, 1, "a refuted lane costs budget");
        let aux2 = compute_goal_aux(&state, 1, &goal2, 0, 0, Some(&st));
        assert!(aux2.preferred_units.contains(&UnitType::Catapult));
        assert!(aux2.preferred_units.contains(&UnitType::Archer));
    }
    /// Tier 1: the tribe's birth tech commits a lane before any terrain is
    /// explored, and the mapping is derived from `lane_techs` (not a second
    /// table that can drift from mapgen). `select_lane` only falls back
    /// to the prior when the census has nothing to say at all — a visible
    /// village alone already gives the rider lane a real (non-terrain)
    /// mobility signal, so this isolates the true information vacuum with
    /// no villages and no cities.
    #[test]
    fn tribe_prior_commits_a_lane_before_the_map_speaks() {
        for (tribe, tech, lane) in [
            (TribeType::Oumaji, TechnologyType::Riding, Lane::RiderRoads),
            (TribeType::Hoodrick, TechnologyType::Archery, Lane::ArcherLine),
            (TribeType::XinXi, TechnologyType::Climbing, Lane::SpamGiants),
        ] {
            let mut state = state_with_villages(0, &[]);
            {
                let t1 = state.tribes.get_mut(&1).unwrap();
                t1.tribe_type = tribe;
                t1.tech_vanilla = vec![TechnologyState {
                    tech_type: tech,
                    discovered: true,
                    discovered_turn: 0,
                }];
            }
            assert_eq!(tribe_lane_prior(&state, 1), Some(lane), "{tribe:?}");
            // Explore the whole grid (no structures, no resources) so
            // `guess_villages` has no unexplored legal spot left to guess —
            // real isolation of the prior-fallback mechanism, now that
            // `select_lane` sources its own Expand targets from state
            // instead of a caller-supplied `MacroGoal`. An unexplored
            // fixture is NOT an information vacuum here: FOW-honest
            // pathfinding reads unexplored tiles as open, so `guess_villages`
            // still paints sites and rider mobility fires on them regardless
            // of tribe — the vacuum has to come from exploration, not from a
            // hand-built empty goal.
            let size = state.settings.size;
            for idx in 0..size * size {
                explore_tile(&mut state, idx);
            }
            let mut st = LaneState::default();
            assert_eq!(select_lane(&state, 1, &mut st, None), Some(lane));
            assert_eq!(st.committed_turn, Some(state.settings.turn));
        }
        // A tribe whose birth tech opens no lane gets no prior.
        let mut state = state_with_villages(0, &[]);
        state.tribes.get_mut(&1).unwrap().tribe_type = TribeType::Imperius;
        state.tribes.get_mut(&1).unwrap().tech_vanilla = vec![TechnologyState {
            tech_type: TechnologyType::Organization,
            discovered: true,
            discovered_turn: 0,
        }];
        assert_eq!(tribe_lane_prior(&state, 1), None);
    }
    /// The lane is an identity, not a running recomputation: switching costs
    /// budget, and the budget runs out.
    #[test]
    fn lane_switching_is_budgeted_and_dwell_gated() {
        let mut state = state_with_villages(0, &[24]);
        state.settings.current_player_turn_id = 1;
        explore_open_fields(&mut state);
        let mut st = LaneState::default();
        select_lane(&state, 1, &mut st, None);
        let first = st.lane.expect("a lane is committed on an explored map");
        assert_eq!(st.pivots_used, 0, "the first commit is not a pivot");

        // Force a hard exit (score 0) repeatedly: each re-pick spends budget,
        // and past MAX_PIVOTS the lane is frozen even under refutation.
        let mut t2 = TribeState::default();
        for &i in &[23, 25] {
            let mut g = unit_at(i);
            g.unit_type = UnitType::Giant;
            t2.units.push(g);
        }
        state.tribes.insert(2, t2);
        for turn in 1..=12 {
            state.settings.turn = turn;
            select_lane(&state, 1, &mut st, None);
        }
        assert!(st.pivots_used <= MAX_PIVOTS, "budget must cap at {MAX_PIVOTS}");
        let frozen = st.lane;
        st.pivots_used = MAX_PIVOTS;
        for turn in 13..=20 {
            state.settings.turn = turn;
            select_lane(&state, 1, &mut st, None);
        }
        assert_eq!(st.lane, frozen, "no lane change once the budget is spent");
        let _ = first;
    }
    /// Pinned semantics of the budget's hard edge: `MAX_PIVOTS` is a cap on
    /// ALL lane changes, refutation included. A lane refuted after the
    /// budget is spent therefore stays committed — deliberate (Verdi's spec
    /// is "at most 3 lanes"), and the tradeoff is recorded here rather than
    /// discovered later: if dead-lane lock-in shows up in the data, this is
    /// the assertion to revisit.
    #[test]
    fn a_spent_budget_outranks_refutation() {
        let mut state = state_with_villages(0, &[24]);
        state.settings.current_player_turn_id = 1;
        explore_open_fields(&mut state);
        let mut st = LaneState::default();
        select_lane(&state, 1, &mut st, None);
        st.pivots_used = MAX_PIVOTS;
        let committed = st.lane;

        // Overwhelming counter-evidence: two giants refute a rider lane.
        let mut t2 = TribeState::default();
        for &i in &[23, 25] {
            let mut g = unit_at(i);
            g.unit_type = UnitType::Giant;
            t2.units.push(g);
        }
        state.tribes.insert(2, t2);
        for turn in 1..=6 {
            state.settings.turn = turn;
            update_lane_state(&state, 1, &mut st);
        }
        assert_eq!(st.lane, committed, "the cap binds even under refutation");
        assert_eq!(st.pivots_used, MAX_PIVOTS);
    }
    /// The head's per-lane scores are additive on top of the census, so a
    /// strong enough net opinion can outvote terrain — but only through the
    /// same guards.
    #[test]
    fn head_scores_shift_the_selector() {
        let mut state = state_with_villages(0, &[24]);
        state.settings.current_player_turn_id = 1;
        explore_open_fields(&mut state);

        let mut algo_only = LaneState::default();
        select_lane(&state, 1, &mut algo_only, None);
        let census_pick = algo_only.lane.unwrap();

        let idx = LANE_ORDER.iter().position(|a| *a == census_pick).unwrap();
        let mut head = [0.0f32; LANES];
        head[(idx + 1) % LANES] = 50.0; // overwhelming opinion for another lane
        let mut with_head = LaneState::default();
        let pick = select_lane(&state, 1, &mut with_head, Some(&head)).unwrap();
        assert_ne!(pick, census_pick, "a decisive head score must move the call");
        assert!(with_head.last_scores.iter().any(|s| *s >= 50.0), "scores recorded for the trace");
    }
    #[test]
    fn knight_commit_opens_stepping_stone_lane() {
        let mut state = state_with_villages(0, &[24]);
        state.settings.current_player_turn_id = 1;
        explore_open_fields(&mut state);
        // Four squishy cavalry (riders, defense 1.0) on explored tiles.
        let mut t2 = TribeState::default();
        for &i in &[26, 27, 28, 29] {
            let mut r = unit_at(i);
            r.unit_type = UnitType::Rider;
            t2.units.push(r);
        }
        state.tribes.insert(2, t2);
        let goal = compute_macro_goal(&state, 1, 0);
        let mut st = LaneState::default();
        update_lane_state(&state, 1, &mut st);
        assert!(st.overlays.knight_commit);
        assert!(st.overlays.defender_screen);
        let aux = compute_goal_aux(&state, 1, &goal, 0, 0, Some(&st));
        assert!(passes_tech_purchase_limits(&ResearchMove::new(TechnologyType::FreeSpirit), &aux));
        assert!(aux.preferred_units.contains(&UnitType::Knight));
        assert!(aux.preferred_units.contains(&UnitType::Defender));
    }
}
