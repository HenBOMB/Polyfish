//! EXP_ELO_026 "oracle macro": a hand-scripted macro layer over the unchanged
//! net, testing whether third-city reach fails at the macro level (commitment
//! and star allocation) rather than micro execution. Two independent steers,
//! both inference-only: an expansion commitment (focus the pursuit channel on
//! one sticky capturable village) and a star gate (drop root tech purchases
//! that would leave the capture unfunded). Nothing here touches training.

use crate::moves::Move;
use crate::states::{GameState, PlayerId};
use crate::types::{MoveType, StructureType, TechnologyType};

/// Stars that must remain after a tech purchase for it to pass the gate while
/// a commitment is active — rough price of fielding a capturer.
pub const STAR_GATE_RESERVE: i32 = 5;

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
}

/// EXP_ELO_028 Stage-1 macro goal: concurrent painted orders (each a target
/// tile) plus one global spending stance. Encoded into the appended goal
/// channels; `orders` must stay sorted so identical goals produce identical
/// feature bytes (the eval cache and tree reuse hash them).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct MacroGoal {
    pub orders: Vec<(OrderKind, i32)>,
    pub stance: Stance,
}

/// Stage-1 scripted goal-setter, v2 (recalibrated Jul 29 after the iter-1..4
/// channel audit showed ATTACK lit on 62% of plies): EXPAND on every
/// capturable village until captured; ATTACK only with local force
/// superiority; DEFEND unchanged; ARM gains a post-expansion "prepare" phase.
pub fn scripted_goal(state: &GameState, player: PlayerId) -> MacroGoal {
    let size = state.settings.size as i32;
    let cheb =
        |a: i32, b: i32| ((a / size) - (b / size)).abs().max(((a % size) - (b % size)).abs());
    let Some(tribe) = state.tribes.get(&player) else {
        return MacroGoal::default();
    };
    let unit_cost =
        |u: &crate::states::UnitState| crate::settings::units::get_unit_setting(u.unit_type).cost;
    let own_units: Vec<(i32, i32)> =
        tribe.units.iter().map(|u| (u.coords.idx, unit_cost(u))).collect();
    let our_army: i32 = own_units.iter().map(|(_, c)| c).sum();
    let mut orders: Vec<(OrderKind, i32)> = Vec::new();

    for &idx in state.structures.keys() {
        if still_capturable(state, idx, player) {
            orders.push((OrderKind::Expand, idx));
        }
    }
    // v2.4: while expanding, keep at least EXPAND_TARGET_MIN targets painted —
    // generator-informed guesses stand in for undiscovered villages, so the
    // approach gradient drives scouting toward likely sites instead of idling.
    if tribe.cities.len() < COMMIT_CITY_TARGET && orders.len() < EXPAND_TARGET_MIN {
        for idx in guessed_village_sites(state, player, EXPAND_TARGET_MIN - orders.len()) {
            orders.push((OrderKind::Expand, idx));
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
    let enemy_units: Vec<i32> = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .flat_map(|(_, t)| t.units.iter().map(|u| u.coords.idx))
        .collect();
    for c in &tribe.cities {
        let near = enemy_units.iter().filter(|&&u| cheb(u, c.idx) <= 2).count();
        if near >= 2 {
            orders.push((OrderKind::Defend, c.idx));
        }
    }

    orders.sort();
    let stance = if orders.iter().any(|(k, _)| *k == OrderKind::Defend) {
        Stance::Arm
    } else if prepare && tribe.cities.len() >= COMMIT_CITY_TARGET {
        Stance::Arm
    } else {
        Stance::Grow
    };
    MacroGoal { orders, stance }
}

/// Minimum EXPAND targets painted while expanding — real villages first,
/// generator-informed guesses fill the remainder (v2.4).
pub const EXPAND_TARGET_MIN: usize = 2;

/// Guess likely undiscovered village sites from the generator's own Drylands
/// rules + the observed map (FOW-honest — game knowledge, not map peeking).
/// The generator fills villages to SATURATION over legal spots (land, edge
/// distance ∈ {2,4,5...}, Chebyshev ≥3 from every village/capital), so an
/// UNEXPLORED legal spot ≥3 from everything known must lie near an
/// undiscovered village. Returns up to `max_sites`, nearest-to-units first,
/// mutually ≥3 apart (the "first warrior center, second north/east" spread).
pub fn guessed_village_sites(
    state: &GameState,
    player: PlayerId,
    max_sites: usize,
) -> Vec<i32> {
    let size = state.settings.size as i32;
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let anchors: Vec<i32> = if tribe.units.is_empty() {
        tribe.cities.iter().map(|c| c.idx).collect()
    } else {
        tribe.units.iter().map(|u| u.coords.idx).collect()
    };
    if anchors.is_empty() || size <= 0 {
        return Vec::new();
    }
    let cheb =
        |a: i32, b: i32| ((a / size) - (b / size)).abs().max(((a % size) - (b % size)).abs());
    let explored =
        |idx: i32| state.tiles.get(&idx).map_or(false, |t| t.explorers.contains(&player));

    // Known spacing sources: explored villages + explored cities (capitals
    // and captured villages count as villages in the generator's spacing).
    let mut known: Vec<i32> = state
        .structures
        .iter()
        .filter(|(idx, s)| {
            s.as_ref().map_or(false, |s| s.structure_type == StructureType::Village)
                && explored(**idx)
        })
        .map(|(idx, _)| *idx)
        .collect();
    for t in state.tribes.values() {
        known.extend(t.cities.iter().map(|c| c.idx).filter(|&i| explored(i)));
    }

    let mut cands: Vec<(i32, i32)> = (0..size * size)
        .filter(|&idx| {
            let (r, c) = (idx / size, idx % size);
            let edge = r.min(size - 1 - r).min(c).min(size - 1 - c);
            !explored(idx)
                && edge >= 2
                && edge != 3
                && known.iter().all(|&k| cheb(idx, k) >= 3)
        })
        .map(|idx| {
            let d = anchors.iter().map(|&a| cheb(a, idx)).min().unwrap_or(i32::MAX);
            (d, idx)
        })
        .collect();
    cands.sort_unstable();
    let mut picks: Vec<i32> = Vec::new();
    for (_, idx) in cands {
        if picks.len() >= max_sites {
            break;
        }
        if picks.iter().all(|&p| cheb(p, idx) >= 3) {
            picks.push(idx);
        }
    }
    picks
}

/// Whether the goal-conditioned research gate is active (v2.2, stance-aware):
/// GROW gates during the expansion window (EXPAND painted, under
/// `COMMIT_CITY_TARGET` cities); ARM gates whenever it holds — each stance
/// gates only the tech class that contradicts it (see `passes_star_gate`).
pub fn goal_star_gate(state: &GameState, player: PlayerId, goal: &MacroGoal) -> bool {
    match goal.stance {
        Stance::Grow => {
            goal.orders.iter().any(|(k, _)| *k == OrderKind::Expand)
                && state
                    .tribes
                    .get(&player)
                    .map_or(false, |t| t.cities.len() < COMMIT_CITY_TARGET)
        }
        Stance::Arm => true,
        Stance::Unlock => false,
    }
}

/// City count at which the commitment retires (the third-city objective).
pub const COMMIT_CITY_TARGET: usize = 3;

/// v2.3 tech-discipline crutch: whole-game cap on techs bought with own
/// stars (Research moves; ruin-granted techs don't count) …
pub const TECH_CAP_PER_GAME: u32 = 8;
/// … of which at most this many tier-3 unlocks.
pub const TIER3_CAP_PER_GAME: u32 = 1;

/// Per-ply auxiliary goal context (v2.3), set on the agent alongside the
/// `MacroGoal` but NOT painted into features: environment-fit tech bias and
/// the whole-game purchase counters. Cached tree edges may carry rewards
/// from a slightly older aux — acceptable staleness, like tree reuse itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoalAux {
    /// Environment-recommended techs (owned ones pay the in-tree fit bonus).
    pub recommended_techs: Vec<TechnologyType>,
    /// Path-aware: a Rider reaches some EXPAND target at least
    /// `RIDER_PUSH_MIN_TURNS_SAVED` turns sooner than a walker would.
    pub rider_push: bool,
    /// Research moves this seat has executed this game.
    pub techs_bought: u32,
    /// …of which tier-3.
    pub tier3_bought: u32,
}

/// Minimum turns a Rider must save (vs a movement-1 unit) to some EXPAND
/// target for the rider push to fire.
pub const RIDER_PUSH_MIN_TURNS_SAVED: u32 = 1;

/// Simplified land-movement class of a tile: `None` = impassable,
/// `Some(true)` = passable but movement-ending (rough), `Some(false)` = open.
/// FOW-honest: unexplored tiles read as open (optimistic scouting).
fn move_class(state: &GameState, player: PlayerId, idx: i32, climbing: bool) -> Option<bool> {
    use crate::types::TerrainType as T;
    let Some(tile) = state.tiles.get(&idx) else {
        return Some(false);
    };
    if !tile.explorers.contains(&player) {
        return Some(false);
    }
    match tile.terrain_type {
        T::Field | T::None => Some(false),
        T::Forest | T::Wetland | T::Mangrove => Some(true),
        T::Mountain => climbing.then_some(true),
        T::Water | T::Ocean | T::Ice => None,
    }
}

/// Multi-source turns-to-reach for a land unit with `movement` points under
/// simplified Polytopia rules: 8-directional steps, entering rough terrain
/// ends the turn. Returns per-tile turn counts (`u32::MAX` = unreachable).
fn turns_to_reach(
    state: &GameState,
    player: PlayerId,
    anchors: &[i32],
    movement: i32,
    climbing: bool,
) -> Vec<u32> {
    let width = state.settings.size as i32;
    let n = (width * width).max(0) as usize;
    let mut turns = vec![u32::MAX; n];
    let neighbors = |idx: i32| {
        let (r, c) = (idx / width, idx % width);
        let mut out = Vec::with_capacity(8);
        for dr in -1..=1 {
            for dc in -1..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let (nr, nc) = (r + dr, c + dc);
                if nr >= 0 && nr < width && nc >= 0 && nc < width {
                    out.push(nr * width + nc);
                }
            }
        }
        out
    };
    let mut frontier: Vec<i32> = anchors
        .iter()
        .copied()
        .filter(|&a| (a as usize) < n)
        .collect();
    for &a in &frontier {
        turns[a as usize] = 0;
    }
    let mut t = 0u32;
    while !frontier.is_empty() && t < 64 {
        t += 1;
        let mut next = Vec::new();
        for &p in &frontier {
            for n1 in neighbors(p) {
                let Some(rough1) = move_class(state, player, n1, climbing) else {
                    continue;
                };
                if turns[n1 as usize] > t {
                    turns[n1 as usize] = t;
                    next.push(n1);
                }
                if movement >= 2 && !rough1 {
                    for n2 in neighbors(n1) {
                        if move_class(state, player, n2, climbing).is_some()
                            && turns[n2 as usize] > t
                        {
                            turns[n2 as usize] = t;
                            next.push(n2);
                        }
                    }
                }
            }
        }
        frontier = next;
    }
    turns
}

/// Path-aware rider advantage: max over `targets` of (walker turns − rider
/// turns) along real explored terrain from the player's units (fallback:
/// cities). A forest pocket off the route costs nothing; a forest corridor
/// erases the advantage — exactly the 2-tile-hop question.
pub fn rider_turns_saved(state: &GameState, player: PlayerId, targets: &[i32]) -> u32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0;
    };
    let anchors: Vec<i32> = if tribe.units.is_empty() {
        tribe.cities.iter().map(|c| c.idx).collect()
    } else {
        tribe.units.iter().map(|u| u.coords.idx).collect()
    };
    if anchors.is_empty() || targets.is_empty() {
        return 0;
    }
    let climbing = crate::settings::technology::is_tech_unlocked(
        &tribe.tech_vanilla,
        crate::settings::technology::resolve_tech_for_tribe(
            TechnologyType::Climbing,
            tribe.tribe_type,
        ),
    );
    let walk = turns_to_reach(state, player, &anchors, 1, climbing);
    let ride = turns_to_reach(state, player, &anchors, 2, climbing);
    targets
        .iter()
        .filter_map(|&tg| {
            let (w, r) = (walk.get(tg as usize)?, ride.get(tg as usize)?);
            (*w != u32::MAX && *r != u32::MAX).then(|| w.saturating_sub(*r))
        })
        .max()
        .unwrap_or(0)
}

/// Environment-fit tech lines, scored from the player's EXPLORED tiles
/// (FOW-honest): terrain counts plus double-weighted matching resources.
/// Returns the next unowned tech of the top two lines. Tribe awareness is
/// emergent — tribe spawns generate their signature terrain/resources, so
/// counting the map plays into the natural environment.
pub fn recommended_techs(state: &GameState, player: PlayerId) -> Vec<TechnologyType> {
    use crate::types::{ResourceType as R, TerrainType as T};
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let (mut forest, mut mountain, mut field, mut water) = (0i32, 0i32, 0i32, 0i32);
    let (mut game_r, mut fruit, mut crop, mut metal, mut fish) = (0i32, 0i32, 0i32, 0i32, 0i32);
    for (idx, tile) in state.tiles.iter() {
        if !tile.explorers.contains(&player) {
            continue;
        }
        match tile.terrain_type {
            T::Forest => forest += 1,
            T::Mountain => mountain += 1,
            T::Field => field += 1,
            T::Water | T::Ocean => water += 1,
            _ => {}
        }
        if let Some(Some(r)) = state.resources.get(idx) {
            match r.resource_type {
                R::Game => game_r += 1,
                R::Fruit => fruit += 1,
                R::Crop => crop += 1,
                R::Metal => metal += 1,
                R::Fish => fish += 1,
                _ => {}
            }
        }
    }
    use TechnologyType as Tech;
    let forest_line: &[Tech] = &[Tech::Hunting, Tech::Forestry, Tech::Mathematics];
    let mountain_line: &[Tech] = &[Tech::Climbing, Tech::Mining, Tech::Smithery];
    let farm_line: &[Tech] = &[Tech::Organization, Tech::Farming, Tech::Construction];
    let water_line: &[Tech] = &[Tech::Fishing];
    let mut lines = [
        (forest + 2 * game_r, forest_line),
        (mountain + 2 * metal, mountain_line),
        (field / 2 + 2 * (fruit + crop), farm_line),
        (water / 2 + 2 * fish, water_line),
    ];
    lines.sort_by_key(|(score, _)| -*score);
    let mut recs = Vec::new();
    for (score, line) in lines.iter().take(2) {
        if *score <= 0 {
            continue;
        }
        if let Some(t) = line
            .iter()
            .find(|t| !crate::settings::technology::is_tech_unlocked(&tribe.tech_vanilla, **t))
        {
            recs.push(*t);
        }
    }
    recs
}

/// Build the per-ply `GoalAux` for the scripted driver: environment fit,
/// the path-aware rider push (a Rider beats a walker to some EXPAND target
/// → Riding joins the recommendations while unowned), and the caller-tracked
/// purchase counters.
pub fn scripted_goal_aux(
    state: &GameState,
    player: PlayerId,
    goal: &MacroGoal,
    techs_bought: u32,
    tier3_bought: u32,
) -> GoalAux {
    let mut recommended = recommended_techs(state, player);
    let expand_targets: Vec<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Expand)
        .map(|(_, idx)| *idx)
        .collect();
    let rider_push = !expand_targets.is_empty()
        && rider_turns_saved(state, player, &expand_targets) >= RIDER_PUSH_MIN_TURNS_SAVED;
    if rider_push {
        if let Some(tribe) = state.tribes.get(&player) {
            let riding = crate::settings::technology::resolve_tech_for_tribe(
                crate::types::TechnologyType::Riding,
                tribe.tribe_type,
            );
            if riding == crate::types::TechnologyType::Riding
                && !crate::settings::technology::is_tech_unlocked(&tribe.tech_vanilla, riding)
                && !recommended.contains(&riding)
            {
                recommended.insert(0, riding);
            }
        }
    }
    GoalAux { recommended_techs: recommended, rider_push, techs_bought, tier3_bought }
}

/// Root-only whole-game purchase caps — applied whenever a `GoalAux` is set,
/// independent of the stance gate's window. Non-Research moves always pass.
pub fn passes_tech_caps(m: &dyn Move, aux: &GoalAux) -> bool {
    if m.move_type() != MoveType::Research {
        return true;
    }
    if aux.techs_bought >= TECH_CAP_PER_GAME {
        return false;
    }
    if aux.tier3_bought >= TIER3_CAP_PER_GAME {
        if let Ok(tech) = m.tech_type() {
            if crate::settings::technology::get_technology_setting(tech).tier == Some(3) {
                return false;
            }
        }
    }
    true
}

/// True while `idx` still holds a village capturable by `player`: Village
/// structure on an unowned tile that `player` has explored (the pursuit
/// channel's predicate — see features.rs).
pub fn still_capturable(state: &GameState, idx: i32, player: PlayerId) -> bool {
    let is_village = state
        .structures
        .get(&idx)
        .and_then(|s| s.as_ref())
        .map_or(false, |s| s.structure_type == StructureType::Village);
    is_village
        && state
            .tiles
            .get(&idx)
            .map_or(false, |t| t.owner == 0 && t.explorers.contains(&player))
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
        |a: i32, b: i32| ((a / size) - (b / size)).abs().max(((a % size) - (b % size)).abs());
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

/// Root-only research gate (v2.2, granular). Every non-Research move passes.
/// A gated Research move passes only when the buyer retains
/// `STAR_GATE_RESERVE` stars after the purchase. What counts as gated:
/// - `Some(Grow)`: techs fielding combat units (eco/mobility/defense tech is
///   the point of GROW and passes freely — incl. Climbing/Sailing passage).
/// - `Some(Arm)`: eco tech fielding no combat units (mixed tech like
///   Smithery arms you and passes).
/// - `Some(Unlock)`: nothing gated (no unlock policy yet).
/// - `None`: every tech (the EXP_ELO_026 legacy gate, kept reproducible for
///   arena `--macro-star-gate`).
pub fn passes_star_gate(state: &GameState, m: &dyn Move, stance: Option<Stance>) -> bool {
    if m.move_type() != MoveType::Research {
        return true;
    }
    let player = state.settings.current_player_turn_id;
    let Some(tribe) = state.tribes.get(&player) else {
        return true;
    };
    let Ok(tech) = m.tech_type() else {
        return true;
    };
    let effects = crate::settings::technology::get_tech_effects(tech);
    let gated = match stance {
        None => true,
        Some(Stance::Grow) => !effects.combat_units.is_empty(),
        Some(Stance::Arm) => {
            crate::settings::technology::is_eco_tech(tech) && effects.combat_units.is_empty()
        }
        Some(Stance::Unlock) => false,
    };
    if !gated {
        return true;
    }
    tribe.stars - crate::functions::get_tech_cost(tribe, tech) >= STAR_GATE_RESERVE
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::EndTurnMove;
    use crate::moves::research::ResearchMove;
    use crate::Coords;
    use crate::states::{StructureState, TileState, TribeState, UnitState};
    use crate::types::{TechnologyType, UnitType};

    fn unit_at(idx: i32) -> UnitState {
        UnitState {
            unit_type: UnitType::Warrior,
            coords: Coords::from_index(idx, 11),
            ..Default::default()
        }
    }

    /// Village structure at `idx`, unowned, explored by player 1.
    fn add_visible_village(state: &mut GameState, idx: i32) {
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

    fn state_with_villages(unit_idx: i32, villages: &[i32]) -> GameState {
        let mut state = GameState::default();
        for &v in villages {
            add_visible_village(&mut state, v);
        }
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(unit_idx));
        state.tribes.insert(1, t1);
        state
    }

    #[test]
    fn commitment_picks_nearest_is_sticky_and_retires_at_three_cities() {
        let mut state = state_with_villages(0, &[3, 5]);
        // Fresh pick: village at idx 3 is 3 tiles away vs 5 for idx 5.
        assert_eq!(update_commitment(&state, 1, None), Some(3));
        // Sticky: an existing valid commitment survives a nearer alternative.
        assert_eq!(update_commitment(&state, 1, Some(5)), Some(5));
        // Retires once the third city exists.
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        assert_eq!(update_commitment(&state, 1, Some(5)), None);
    }

    #[test]
    fn commitment_repicks_when_target_is_captured() {
        let mut state = state_with_villages(0, &[3, 5]);
        state.tiles.get_mut(&5).unwrap().owner = 2;
        assert_eq!(update_commitment(&state, 1, Some(5)), Some(3));
    }

    /// Bare explored tile at `idx` (no structure) — for enemy-city visibility.
    fn explore_tile(state: &mut GameState, idx: i32) {
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(idx, tile);
    }

    #[test]
    fn scripted_goal_paints_expand_attack_defend_and_sets_stance() {
        let mut state = state_with_villages(0, &[3, 5]);
        // Under 3 cities with two capturable villages → two EXPAND orders,
        // sorted, GROW stance, star gate active.
        let g = scripted_goal(&state, 1);
        assert_eq!(
            g.orders,
            vec![(OrderKind::Expand, 3), (OrderKind::Expand, 5)]
        );
        assert_eq!(g.stance, Stance::Grow);
        assert!(goal_star_gate(&state, 1, &g));

        // Explored enemy city at 40 = (3,7), two own units within Chebyshev 3
        // (39 = (3,6) and 29 = (2,7)), no defenders → superiority → ATTACK.
        let mut t2 = TribeState::default();
        t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
        state.tribes.insert(2, t2);
        explore_tile(&mut state, 40);
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(39));
        t1.units.push(unit_at(29));
        let g = scripted_goal(&state, 1);
        assert!(g.orders.contains(&(OrderKind::Attack, 40)));
        assert_eq!(g.stance, Stance::Grow);

        // Two enemy units within 2 of an own city → DEFEND + ARM stance.
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            ..Default::default()
        });
        let t2 = state.tribes.get_mut(&2).unwrap();
        t2.units.push(unit_at(1));
        t2.units.push(unit_at(12));
        let g = scripted_goal(&state, 1);
        assert!(g.orders.contains(&(OrderKind::Defend, 0)));
        assert_eq!(g.stance, Stance::Arm);
    }

    #[test]
    fn attack_requires_local_superiority() {
        let mut state = state_with_villages(0, &[3]);
        let mut t2 = TribeState::default();
        t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
        // Two defenders within Chebyshev 2 of their city match our two
        // attackers' value — no superiority, no ATTACK order.
        t2.units.push(unit_at(41));
        t2.units.push(unit_at(51));
        state.tribes.insert(2, t2);
        explore_tile(&mut state, 40);
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(39));
        t1.units.push(unit_at(29));
        let g = scripted_goal(&state, 1);
        assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));

        // A third attacker reaches parity-plus but not the 1.5x margin.
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(30));
        let g = scripted_goal(&state, 1);
        assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));

        // A fourth clears the margin → ATTACK.
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(20));
        let g = scripted_goal(&state, 1);
        assert!(g.orders.contains(&(OrderKind::Attack, 40)));

        // Unexplored enemy city never draws an order.
        state.tiles.get_mut(&40).unwrap().explorers.clear();
        let g = scripted_goal(&state, 1);
        assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));
    }

    #[test]
    fn prepare_arms_post_expansion_when_massing_would_win() {
        // Explored enemy city, one own unit in approach range (cheb 4), army
        // outweighs the garrison but local force is short → prepare.
        let mut state = state_with_villages(0, &[3]);
        let mut t2 = TribeState::default();
        t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
        t2.units.push(unit_at(41));
        state.tribes.insert(2, t2);
        explore_tile(&mut state, 40);
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(29));

        // Still expanding (<3 cities): prepare must NOT override GROW.
        let g = scripted_goal(&state, 1);
        assert_eq!(g.stance, Stance::Grow);

        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        let g = scripted_goal(&state, 1);
        assert_eq!(g.stance, Stance::Arm);
        assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));
    }

    #[test]
    fn expand_persists_past_third_city_but_gate_retires() {
        let mut state = state_with_villages(0, &[3, 5]);
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        let g = scripted_goal(&state, 1);
        assert!(g.orders.contains(&(OrderKind::Expand, 3)));
        assert_eq!(g.stance, Stance::Grow);
        assert!(!goal_star_gate(&state, 1, &g));
    }

    #[test]
    fn star_gate_blocks_only_underfunded_research() {
        // Legacy (stance-less, EXP_ELO_026) arm: every tech is gated.
        let mut state = state_with_villages(0, &[3]);
        state.settings.current_player_turn_id = 1;
        let tech = TechnologyType::Organization;
        let cost = crate::functions::get_tech_cost(state.tribes.get(&1).unwrap(), tech);
        let research = ResearchMove::new(tech);

        state.tribes.get_mut(&1).unwrap().stars = cost + STAR_GATE_RESERVE - 1;
        assert!(!passes_star_gate(&state, &research, None));

        state.tribes.get_mut(&1).unwrap().stars = cost + STAR_GATE_RESERVE;
        assert!(passes_star_gate(&state, &research, None));

        // Non-research moves always pass, regardless of stars.
        state.tribes.get_mut(&1).unwrap().stars = 0;
        assert!(passes_star_gate(&state, &EndTurnMove, None));
    }

    #[test]
    fn stance_gate_is_granular_by_tech_class() {
        let mut state = state_with_villages(0, &[3]);
        state.settings.current_player_turn_id = 1;
        // Broke: nothing can meet the reserve, so gated == blocked.
        state.tribes.get_mut(&1).unwrap().stars = 0;
        let eco = ResearchMove::new(TechnologyType::Organization);
        let combat = ResearchMove::new(TechnologyType::Riding);
        let passage = ResearchMove::new(TechnologyType::Climbing);
        let mixed = ResearchMove::new(TechnologyType::Smithery);

        // GROW gates combat-unit tech only; eco and passage flow freely
        // (Climbing carries a defense bonus but fields no unit).
        let grow = Some(Stance::Grow);
        assert!(passes_star_gate(&state, &eco, grow));
        assert!(passes_star_gate(&state, &passage, grow));
        assert!(!passes_star_gate(&state, &combat, grow));
        assert!(!passes_star_gate(&state, &mixed, grow));

        // ARM flips it: pure-eco tech gated, unit tech (incl. mixed) free.
        let arm = Some(Stance::Arm);
        assert!(!passes_star_gate(&state, &eco, arm));
        assert!(passes_star_gate(&state, &combat, arm));
        assert!(passes_star_gate(&state, &mixed, arm));

        // Reserve still lifts a gated class.
        let cost = crate::functions::get_tech_cost(
            state.tribes.get(&1).unwrap(),
            TechnologyType::Riding,
        );
        state.tribes.get_mut(&1).unwrap().stars = cost + STAR_GATE_RESERVE;
        assert!(passes_star_gate(&state, &combat, grow));

        // UNLOCK gates nothing (no unlock policy yet).
        state.tribes.get_mut(&1).unwrap().stars = 0;
        assert!(passes_star_gate(&state, &combat, Some(Stance::Unlock)));
    }

    #[test]
    fn recommended_techs_follow_the_environment() {
        use crate::states::TechnologyState;
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(0));
        state.tribes.insert(1, t1);
        // Explored mountain ridge with metal → mountain line: Climbing first.
        for idx in 10..16 {
            let mut tile = TileState::default();
            tile.terrain_type = crate::types::TerrainType::Mountain;
            tile.explorers.insert(1);
            state.tiles.insert(idx, tile);
        }
        state.resources.insert(
            11,
            Some(crate::states::ResourceState {
                resource_type: crate::types::ResourceType::Metal,
            }),
        );
        let recs = recommended_techs(&state, 1);
        assert_eq!(recs, vec![TechnologyType::Climbing]);

        // Owning Climbing + Mining advances the line to Smithery.
        let t1 = state.tribes.get_mut(&1).unwrap();
        for tech in [TechnologyType::Climbing, TechnologyType::Mining] {
            t1.tech_vanilla.push(TechnologyState {
                tech_type: tech,
                discovered: true,
                discovered_turn: 0,
            });
        }
        let recs = recommended_techs(&state, 1);
        assert_eq!(recs, vec![TechnologyType::Smithery]);
    }

    /// The rider push must judge the ROUTE, not the global terrain census: a
    /// forest pocket off the path is irrelevant; a forest corridor on the
    /// path erases the 2-tile advantage.
    #[test]
    fn rider_push_is_path_aware() {
        use crate::types::TerrainType;
        let terrain_tile = |terrain: TerrainType| {
            let mut tile = TileState::default();
            tile.terrain_type = terrain;
            tile.explorers.insert(1);
            tile
        };
        // Unit at (0,0), village at (4,0). A big explored forest pocket in
        // the far corner outnumbers explored fields — the old global census
        // would veto riders; the route doesn't care.
        let mut state = state_with_villages(0, &[44]);
        for r in 8..11 {
            for c in 8..11 {
                state.tiles.insert(r * 11 + c, terrain_tile(TerrainType::Forest));
            }
        }
        let goal = scripted_goal(&state, 1);
        assert!(scripted_goal_aux(&state, 1, &goal, 0, 0).rider_push);
        assert!(rider_turns_saved(&state, 1, &[44]) >= 2);

        // A thin band is NOT enough: a rider weaves open-step + forest-step
        // (2 tiles/turn, real rider mechanics) and still saves a turn.
        for r in 1..4 {
            for c in 0..3 {
                state.tiles.insert(r * 11 + c, terrain_tile(TerrainType::Forest));
            }
        }
        let goal = scripted_goal(&state, 1);
        assert!(scripted_goal_aux(&state, 1, &goal, 0, 0).rider_push);

        // Only when the whole approach region is rough does the advantage
        // vanish: forest block rows 0-4 x cols 0-4 (minus start and target).
        // Judged per-target — the aux flag may still fire via guessed sites
        // whose routes run through open unexplored ground (by design).
        for r in 0..5 {
            for c in 0..5 {
                let idx = r * 11 + c;
                if idx != 0 && idx != 44 {
                    state.tiles.insert(idx, terrain_tile(TerrainType::Forest));
                }
            }
        }
        assert_eq!(rider_turns_saved(&state, 1, &[44]), 0);
    }

    #[test]
    fn guessed_sites_respect_generator_rules_and_spread() {
        // Capital city at the center, nothing else explored: guesses must be
        // unexplored, on the legal edge bands, >=3 from the capital and from
        // each other, and nearest-first from the unit.
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60));
        t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        state.tribes.insert(1, t1);
        explore_tile(&mut state, 60);

        let sites = guessed_village_sites(&state, 1, 2);
        assert_eq!(sites.len(), 2);
        let cheb = |a: i32, b: i32| ((a / 11) - (b / 11)).abs().max(((a % 11) - (b % 11)).abs());
        for &s in &sites {
            let (r, c) = (s / 11, s % 11);
            let edge = r.min(10 - r).min(c).min(10 - c);
            assert!(edge >= 2 && edge != 3, "site {s} off the generator's bands");
            assert!(cheb(s, 60) >= 3, "site {s} too close to the known capital");
            assert!(cheb(s, 60) <= 4, "site {s} not nearest-first");
        }
        assert!(cheb(sites[0], sites[1]) >= 3, "guesses must spread");

        // A known village nearby suppresses guesses in its exclusion zone.
        add_visible_village(&mut state, 24); // (2,2)
        let sites = guessed_village_sites(&state, 1, 4);
        assert!(sites.iter().all(|&s| cheb(s, 24) >= 3));

        // And scripted_goal paints guesses whenever real targets run short.
        let g = scripted_goal(&state, 1);
        let expands: Vec<i32> = g
            .orders
            .iter()
            .filter(|(k, _)| *k == OrderKind::Expand)
            .map(|(_, i)| *i)
            .collect();
        assert!(expands.contains(&24)); // the real village
        assert_eq!(expands.len(), EXPAND_TARGET_MIN); // topped up with a guess
    }

    #[test]
    fn tech_caps_and_rider_push() {
        let mut state = state_with_villages(0, &[3]);
        state.settings.current_player_turn_id = 1;
        // Open fields around the spawn → rider-friendly terrain.
        for idx in 20..30 {
            let mut tile = TileState::default();
            tile.terrain_type = crate::types::TerrainType::Field;
            tile.explorers.insert(1);
            state.tiles.insert(idx, tile);
        }
        let goal = scripted_goal(&state, 1); // EXPAND on village 3
        let aux = scripted_goal_aux(&state, 1, &goal, 0, 0);
        assert!(aux.rider_push);
        assert_eq!(aux.recommended_techs.first(), Some(&TechnologyType::Riding));

        // Without an EXPAND order there is no rider push.
        let quiet = MacroGoal::default();
        assert!(!scripted_goal_aux(&state, 1, &quiet, 0, 0).rider_push);

        // Caps: 8 bought blocks all research; one tier-3 blocks further tier-3.
        let research1 = ResearchMove::new(TechnologyType::Organization);
        let research3 = ResearchMove::new(TechnologyType::Smithery);
        let mut capped = aux.clone();
        capped.techs_bought = TECH_CAP_PER_GAME;
        assert!(!passes_tech_caps(&research1, &capped));
        assert!(passes_tech_caps(&EndTurnMove, &capped));
        let mut t3 = aux.clone();
        t3.tier3_bought = TIER3_CAP_PER_GAME;
        assert!(passes_tech_caps(&research1, &t3));
        assert!(!passes_tech_caps(&research3, &t3));
    }

    #[test]
    fn goal_star_gate_is_stance_aware() {
        let mut state = state_with_villages(0, &[3]);
        // ARM gates regardless of expansion state.
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm };
        assert!(goal_star_gate(&state, 1, &arm));
        // GROW gates only inside the expansion window.
        let grow = MacroGoal {
            orders: vec![(OrderKind::Expand, 3)],
            stance: Stance::Grow,
        };
        assert!(goal_star_gate(&state, 1, &grow));
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        assert!(!goal_star_gate(&state, 1, &grow));
    }
}
