//! EXP_ELO_026 "oracle macro": a hand-scripted macro layer over the unchanged
//! net, testing whether third-city reach fails at the macro level (commitment
//! and star allocation) rather than micro execution. Two independent steers,
//! both inference-only: an expansion commitment (focus the pursuit channel on
//! one sticky capturable village) and a star gate (drop root tech purchases
//! that would leave the capture unfunded). Nothing here touches training.

use crate::moves::Move;
use crate::states::{GameState, PlayerId};
use crate::types::{AbilityType, MoveType, StructureType, TechnologyType, TerrainType};

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
/// The economy batch a SAVE stance is banking for, with the lane it belongs to.
///
/// v10: this used to be a bare `Option<i32>` — `save_batch_plan` identified the
/// lane and then discarded everything but the price, so nothing downstream
/// could tell "saving for a Forge" from "saving for 21 stars". Search could not
/// boost the very move the plan existed to reach.
#[derive(Clone, Debug, PartialEq)]
pub struct SaveLane {
    /// `tech_cost + structure_cost`, the number the ramp measures against.
    pub cost: i32,
    /// Chain cost of reaching `tech` from what the tribe owns; 0 once owned.
    pub tech_cost: i32,
    /// Placement cost summed over every city that can legally site one now.
    pub structure_cost: i32,
    /// Per-placement cost, for crediting partial completion.
    pub structure_unit_cost: i32,
    pub tech: TechnologyType,
    pub structure: crate::types::StructureType,
}

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
    // EXP_ELO_040/050: threat-driven Defend, from the single unified risk
    // model (city_risks — EXP_ELO_054 folded the separate strike-only
    // city_threats model into it). `needs_order()` covers both a sieged or
    // next-turn-reachable-and-open city and a garrison under near-lethal
    // strike; the old `near >= 2` proxy was blind to a single sieging unit
    // (fixture 1786670356), and a strike-only model was blind to an EMPTY
    // reachable city (seed-1786807403, capital lost on t9 while the
    // directive read Grow/Expand).
    for r in crate::ai::defense::city_risks(state, player) {
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

/// Minimum friendly partners a multiplier-tier placement must pay before it is
/// worth banking for — a 1-partner Windmill is one pop and affordable out of
/// pocket, so it never justifies holding stars.
pub const SAVE_MIN_PARTNERS: i32 = 1;

/// EXP_ELO_051: how far this tribe has already walked toward `tech` — the
/// number of techs in its prerequisite chain it already owns.
///
/// This is what makes the COMMITTED lane, not the cheapest sticker price,
/// decide what to bank for. Verdi: "we should be saving towards a lane if
/// that is what T1 says … the best computed path for that giant spam is
/// forges, therefore these things should act as the justification to save
/// for and buy forge." A tribe holding Climbing+Mining is walking the Forge
/// lane whether or not a Windmill happens to be five stars cheaper.
pub fn lane_investment(tribe: &crate::states::TribeState, tech: TechnologyType) -> i32 {
    use crate::settings::technology::{get_technology_setting, has_technology};
    let mut owned = 0;
    let mut cur = Some(tech);
    let mut guard = 0;
    while let Some(t) = cur {
        guard += 1;
        if guard > 16 {
            break;
        }
        if has_technology(&tribe.tech_vanilla, t) {
            owned += 1;
        }
        cur = get_technology_setting(t).requires;
    }
    owned
}

/// Partners a hub site would have once the lane's OWN prerequisite builds go
/// up: existing partner structures plus adjacent owned tiles that could take
/// one. Counting only what stands today is circular — a Forge needs Mines,
/// and nothing was banking for the Forge that would justify the Mines. In
/// seed 1786807405 that loop cost the whole plan: `save_lane` stayed None
/// until t9 because the mountains had no mines on them yet.
fn potential_partner_count(
    state: &GameState,
    idx: i32,
    partners: &std::collections::HashSet<crate::types::StructureType>,
    owner: PlayerId,
) -> i32 {
    use crate::settings::structures::get_structure_setting;
    if partners.is_empty() || owner == 0 {
        return 0;
    }
    crate::functions::get_adjacent_indices(state, idx, 1)
        .into_iter()
        .filter(|&adj| {
            let Some(tile) = state.tiles.get(&adj) else { return false };
            if tile.owner != owner {
                return false;
            }
            if let Some(s) = crate::functions::get_structure_at(state, adj) {
                return partners.contains(&s.structure_type);
            }
            partners.iter().any(|p| {
                let ps = get_structure_setting(*p);
                ps.terrain_types.contains(&tile.terrain_type)
                    && ps.resource_type.map_or(true, |r| {
                        // Judged for the OWNER, not `current_player_turn_id`:
                        // this runs inside rollouts where the pov is not the
                        // player being planned for.
                        state.resources.get(&adj).and_then(|x| x.as_ref())
                            .map_or(false, |res| res.resource_type == r)
                            && crate::functions::is_resource_visible_to_tribe(
                                state, r, owner, Some(adj))
                    })
            })
        })
        .count() as i32
}
/// Never bank for a batch further out than this many turns of income; beyond
/// it the plan is a hoard, not a plan.
pub const SAVE_MAX_TURNS: i32 = 3;
/// Placements a single batch may bank for. The plan is "the tech and the
/// first hubs", not "one hub in every city I will ever own".
pub const SAVE_MAX_PLACEMENTS: i32 = 2;

/// The four territory-upgrade LANES and the tier-3 tech that opens each. The
/// tech is the real commitment: `TIER3_CAP_PER_GAME` is 1, so a tribe picks at
/// most one of these per game — which is exactly the "decide which upgrade to
/// lean on this game" choice, and exactly what a savings plan is for.
const SAVE_LANES: [(crate::types::StructureType, TechnologyType); 4] = [
    (crate::types::StructureType::Windmill, TechnologyType::Construction),
    (crate::types::StructureType::Sawmill, TechnologyType::Mathematics),
    (crate::types::StructureType::Forge, TechnologyType::Smithery),
    (crate::types::StructureType::Market, TechnologyType::Trade),
];

/// EXP_ELO_053: what a hub lane is actually worth ON THIS MAP, as population
/// per star. This is the pareto ratio the planner was missing.
///
/// `lane_investment` ranked lanes by how much of their TECH CHAIN the tribe
/// already owned, which is blind to terrain: forest-rich Imperius spawns with
/// Organization, a Construction prerequisite, and so banked for a **Windmill**
/// — a hub that eats Farms, which need a Crop resource the map does not have.
/// Verdi: "It's not that good of a tech since imperius is mostly forest rich
/// rather than crop rich. This is a failure of our computation."
///
/// Yield is the engine's own: a hub pays `reward_pop × partner_count`, and
/// each partner pays its own `reward_pop`. Cost is the full prerequisite
/// chain plus the hub plus the partners still to build. Sites are scored
/// individually because only ADJACENT partners feed a hub.
fn lane_yield_per_star(
    state: &GameState,
    player: PlayerId,
    hub: crate::types::StructureType,
    tech: TechnologyType,
) -> f32 {
    use crate::settings::structures::get_structure_setting;
    let Some(tribe) = state.tribes.get(&player) else { return 0.0 };
    let hs = get_structure_setting(hub);
    let Some(hub_cost) = hs.cost else { return 0.0 };
    let mut best = 0.0f32;
    for city in &tribe.cities {
        for &site in &city._territory {
            let Some(tile) = state.tiles.get(&site) else { continue };
            if !hs.terrain_types.contains(&tile.terrain_type) || tile.is_algae() {
                continue;
            }
            if crate::functions::get_structure_at(state, site).is_some() {
                continue;
            }
            // Adjacent ground that already feeds, or could.
            let mut pop = 0;
            let mut cost = hub_cost;
            let mut partners = 0;
            for adj in crate::functions::get_adjacent_indices(state, site, 1) {
                let Some(t) = state.tiles.get(&adj) else { continue };
                if t.owner != player {
                    continue;
                }
                if let Some(st) = crate::functions::get_structure_at(state, adj) {
                    if hs.adjacent_types.contains(&st.structure_type) {
                        partners += 1; // already standing: free
                    }
                    continue;
                }
                // Buildable partner on this ground?
                if let Some(p) = hs.adjacent_types.iter().find(|p| {
                    let ps = get_structure_setting(**p);
                    ps.terrain_types.contains(&t.terrain_type)
                        && ps.resource_type.map_or(true, |r| {
                            state.resources.get(&adj).and_then(|x| x.as_ref())
                                .map_or(false, |res| res.resource_type == r)
                                && crate::functions::is_resource_visible_to_tribe(
                                    state, r, player, Some(adj))
                        })
                }) {
                    let ps = get_structure_setting(*p);
                    partners += 1;
                    pop += ps.reward_pop;
                    cost += ps.cost.unwrap_or(0);
                }
            }
            if partners == 0 {
                continue;
            }
            pop += hs.reward_pop * partners;
            let total = cost + tech_chain_cost(tribe, tech);
            let ratio = pop as f32 / total.max(1) as f32;
            if ratio > best {
                best = ratio;
            }
        }
    }
    best
}

/// EXP_ELO_052: road tiles still needed to link each unconnected city into
/// the capital's network, as (city, tiles_remaining).
///
/// The engine connects two adjacent tiles only when BOTH carry a road (city
/// tiles and ports count as road for this purpose) and never through enemy
/// ground — see `actions::connection`. So this is a shortest path from the
/// capital's component where standing road/city tiles are free and buildable
/// ground costs one.
///
/// Why it exists: a connection pays +1 population to the city AND +1 to the
/// capital, but that lands only on the LAST road tile. Every earlier tile on
/// the path earns nothing, so it loses every ballot to a harvest and no city
/// ever connects — measured 0.00 connected cities at t10 across 96 games on
/// three tribes. Handing the remaining-tile count down lets T3 price each
/// tile by the progress it makes.
pub fn connect_remaining(state: &GameState, player: PlayerId) -> Vec<(i32, i32)> {
    use crate::types::StructureType;
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    // Roads are the only way to build the path; without the tech there is no
    // plan to price, only a constant.
    if !crate::settings::technology::has_technology(&tribe.tech_vanilla, TechnologyType::Roads) {
        return Vec::new();
    }
    let Some(cap) = crate::functions::get_capital_city(state, player) else {
        return Vec::new();
    };
    let cities: Vec<i32> = tribe.cities.iter().map(|c| c.idx).collect();
    let road_here = |idx: i32| {
        cities.contains(&idx)
            || crate::functions::get_structure_type_at(state, idx) == Some(StructureType::Road)
    };
    let buildable = |idx: i32| {
        let Some(t) = state.tiles.get(&idx) else { return false };
        if t.owner != 0 && t.owner != player {
            return false;
        }
        if crate::functions::get_structure_at(state, idx).is_some() {
            return false;
        }
        crate::settings::structures::get_structure_setting(StructureType::Road)
            .terrain_types
            .contains(&t.terrain_type)
    };
    // 0-1 BFS from the capital: free through standing road, cost 1 per tile
    // we would have to build.
    let mut dist: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    let mut dq: std::collections::VecDeque<i32> = std::collections::VecDeque::new();
    dist.insert(cap.idx, 0);
    dq.push_front(cap.idx);
    while let Some(cur) = dq.pop_front() {
        let d = dist[&cur];
        for n in crate::functions::get_adjacent_indices(state, cur, 1) {
            let free = road_here(n);
            if !free && !buildable(n) {
                continue;
            }
            let nd = d + if free { 0 } else { 1 };
            if dist.get(&n).map_or(true, |&old| nd < old) {
                dist.insert(n, nd);
                if free {
                    dq.push_front(n);
                } else {
                    dq.push_back(n);
                }
            }
        }
    }
    tribe
        .cities
        .iter()
        .filter(|c| c.idx != cap.idx && !c.connected_to_capital)
        .filter_map(|c| dist.get(&c.idx).map(|&d| (c.idx, d)))
        .collect()
}

/// EXP_ELO_052: the economic lane each Tier-1 playstyle banks toward. This
/// is the "T1 says giants, so save for forges" mapping stated directly,
/// replacing the `lane_investment` PROXY that inferred it from owned techs.
/// The proxy is exact for XinXi (Climbing is a Smithery prerequisite) and
/// inverts for Imperius, which spawns with Organization and therefore banked
/// for a Windmill on a RiderRoads seat — gating its own first tech out.
///
/// RiderRoads maps to the Market, whose chain is Trade ← Roads ← Riding: the
/// savings plan then names Riding as its next step, which is exactly the
/// lane's opening move.
pub fn lane_save_structure(a: Archetype) -> crate::types::StructureType {
    use crate::types::StructureType as S;
    match a {
        Archetype::RiderRoads => S::Market,
        Archetype::ArcherLine => S::Sawmill,
        Archetype::ForgeGiants => S::Forge,
    }
}

/// v7: full star cost of REACHING `tech` from what this tribe owns — every
/// undiscovered prerequisite up the `requires` chain plus the tech itself.
///
/// Pricing only the final tech understates a lane badly: Trade sits behind
/// Roads behind Riding, so "5 stars for a Market" can really be 30+. A plan
/// that cannot see the path it has to walk cannot be weighed against a
/// cheaper one.
pub fn tech_chain_cost(tribe: &crate::states::TribeState, tech: TechnologyType) -> i32 {
    use crate::settings::technology::{get_technology_setting, has_technology};
    let mut total = 0;
    let mut cur = Some(tech);
    let mut guard = 0;
    while let Some(t) = cur {
        guard += 1;
        if guard > 16 || has_technology(&tribe.tech_vanilla, t) {
            break;
        }
        total += crate::functions::get_tech_cost(tribe, t);
        cur = get_technology_setting(t).requires;
    }
    total
}

/// v7: cost of the CHEAPEST territory-upgrade lane worth banking for — the
/// full prerequisite chain to the enabling tier-3 tech (when unowned) plus
/// every placement in that lane that would pay at least `SAVE_MIN_PARTNERS`.
///
/// `tier3_bought` gates reachability: a lane whose tech the tier-3 cap will
/// refuse is not a plan, it is a hoard with no exit. v7 priced such lanes and
/// banked toward techs it was structurally forbidden to buy.
///
/// Counting the tech is what makes this a plan rather than a purchase. A lone
/// 5-star Windmill is affordable out of pocket at any realistic income, so a
/// batch of already-unlocked structures never triggers saving — measured: a
/// placeable batch existed on 26% of turns and the SAVE gate fired on 0 of
/// them. The lane (tech + the structures it unlocks) is the ~15-25 star
/// commitment a human actually banks for.
pub fn save_batch_plan(
    state: &GameState,
    player: PlayerId,
    tier3_bought: u32,
    committed: Option<Archetype>,
) -> Option<SaveLane> {
    // Before the selector has committed, the spawn tribe tech already says
    // which lane this tribe is born into — so the plan is right from ply one
    // and every caller resolves it identically.
    let committed = committed.or_else(|| tribe_lane_prior(state, player));
    use crate::settings::structures::get_structure_setting;
    use crate::settings::technology::has_technology;
    use crate::types::StructureType;
    let tribe = state.tribes.get(&player)?;
    let mut best: Option<(SaveLane, i32)> = None;
    for (s_type, tech) in SAVE_LANES {
        let s = get_structure_setting(s_type);
        let Some(cost) = s.cost else { continue };
        if let Some(req) = s.tribe_type {
            if req != tribe.tribe_type {
                continue;
            }
        }
        // Unreachable lane: the tier-3 budget is spent and this tech is not
        // owned, so no amount of banking can ever complete the plan.
        let owned = has_technology(&tribe.tech_vanilla, tech);
        if !owned && tier3_bought >= TIER3_CAP_PER_GAME {
            continue;
        }
        // Market pays stars rather than pop; one partner is enough for it.
        let need = if s_type == StructureType::Market { 1 } else { SAVE_MIN_PARTNERS };
        let mut lane = 0;
        for city in &tribe.cities {
            if s.limited_per_city
                && city._territory.iter().any(|&t| {
                    crate::functions::get_structure_type_at(state, t) == Some(s_type)
                })
            {
                continue;
            }
            let placeable = city._territory.iter().any(|&idx| {
                if crate::functions::get_structure_at(state, idx).is_some() {
                    return false;
                }
                let Some(tile) = state.tiles.get(&idx) else { return false };
                if !s.terrain_types.contains(&tile.terrain_type) || tile.is_algae() {
                    return false;
                }
                potential_partner_count(state, idx, &s.adjacent_types, player) >= need
            });
            if placeable {
                lane += cost;
            }
        }
        if lane == 0 {
            continue;
        }
        // EXP_ELO_051: bank for the next placements, not for every one at
        // once. Summing over all cities made the batch GROW with expansion —
        // 44 stars across five cities in seed 1786807405 — so the plan became
        // unaffordable exactly as the empire that justified it arrived.
        lane = lane.min(cost * SAVE_MAX_PLACEMENTS);
        let tech_cost = if owned { 0 } else { tech_chain_cost(tribe, tech) };
        let plan = SaveLane {
            cost: lane + tech_cost,
            tech_cost,
            structure_cost: lane,
            structure_unit_cost: cost,
            tech,
            structure: s_type,
        };
        // EXP_ELO_052 iter 2: the hub is chosen by reachability and price,
        // NOT by the committed lane. Forcing RiderRoads onto the Market meant
        // banking through Riding→Roads→Trade (30+ stars, as `tech_chain_cost`
        // warns), and those seats then built no hub at all — measured hubs@t15
        // 0.94 → 0.31 on Imperius. The lane gets priority over TECHS instead,
        // in `passes_tech_caps`.
        // A lane you cannot reach is not a plan. Reachability is checked HERE,
        // before ranking, so an unaffordable best-ratio lane yields to the
        // next one instead of leaving the seat with no plan at all — which is
        // what took Bardur to 0.22 hubs and 18/32 wins.
        let spt = crate::functions::get_tribe_spt(state, tribe);
        if plan.cost > tribe.stars + spt * SAVE_MAX_TURNS {
            continue;
        }
        // Pareto: population per star ON THIS MAP. Scaled to an integer so
        // the existing tie-break on price still applies between equals.
        let rank = (lane_yield_per_star(state, player, s_type, tech) * 1000.0) as i32;
        let better = best.as_ref().map_or(true, |(b, bi): &(SaveLane, i32)| {
            rank > *bi || (rank == *bi && plan.cost < b.cost)
        });
        if better {
            best = Some((plan, rank));
        }
    }
    best.map(|(p, _)| p)
}

/// Does `m` advance the banked plan? The whole undiscovered `requires` chain
/// counts, not just the final tech — `save_batch_plan` prices the chain
/// (Market sits behind Roads behind Riding), so boosting only the last step
/// would leave every multi-step lane exactly as stuck as it is today.
///
/// Structurally inert while banking: a Research/Build move is only generated
/// once it is affordable, so this fires exactly when the purchase goes live.
pub fn advances_save_plan(m: &dyn Move, lane: &SaveLane, tribe: &crate::states::TribeState) -> bool {
    use crate::settings::technology::{get_technology_setting, has_technology};
    match m.move_type() {
        MoveType::Research => {
            let Ok(t) = m.tech_type() else { return false };
            let mut cur = Some(lane.tech);
            let mut guard = 0;
            while let Some(c) = cur {
                if guard > 16 {
                    break;
                }
                guard += 1;
                if has_technology(&tribe.tech_vanilla, c) {
                    break;
                }
                if c == t {
                    return true;
                }
                cur = get_technology_setting(c).requires;
            }
            false
        }
        // The hub itself, and the partner builds that make it placeable. A
        // Forge needs Mines; crediting only the Forge left every Mine priced
        // as generic economy, and in seed 1786807405 `Build Mine at 69` lost
        // to a Warrior at t6 despite higher q — which is why the Forge did
        // not land until t15, five turns after the cities started falling.
        MoveType::Build => {
            let Ok(s) = m.structure_type() else { return false };
            s == lane.structure
                || crate::settings::structures::get_structure_setting(lane.structure)
                    .adjacent_types
                    .contains(&s)
        }
        _ => false,
    }
}

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

/// Moved to `belief::prediction` (Aug 2026 taxonomy reorg) — co-located with
/// `predict_villages`, the other FOW village-guesser it duplicates.
pub use crate::ai::belief::prediction::guessed_village_sites;

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

/// Per-ply auxiliary goal context (v2.3), set on the agent alongside the
/// `MacroGoal` but NOT painted into features: environment-fit tech bias and
/// the whole-game purchase counters. Cached tree edges may carry rewards
/// from a slightly older aux — acceptable staleness, like tree reuse itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoalAux {
    /// EXP_ELO_050: T2's risk assessment, handed DOWN to T3 — the threat
    /// facts (who can strike, who can walk in, is a siege breakable, what the
    /// city is worth) for every city under threat. The expensive reachability
    /// search happens here, once; T3 re-resolves only `residual_risk` against
    /// live occupancy, which is what gives its defensive plies a gradient.
    pub city_risk: Vec<crate::ai::defense::CityRisk>,
    /// EXP_ELO_051: the batch this seat is banking for, carried down so the
    /// star gate can ask "does this purchase advance the plan" instead of
    /// only "is this tech the right CLASS for the stance". Class-only gating
    /// is what let Organization through on a SAVE turn in seed 1786807403.
    pub save_lane: Option<SaveLane>,
    /// The one tech that batch is actually waiting on — the deepest unowned
    /// step of its prerequisite chain. While banking, this is the only
    /// research that is not a delay.
    pub save_next_tech: Option<TechnologyType>,
    /// EXP_ELO_052: the committed Tier-1 lane's next unowned tech. This
    /// outranks the hub batch: a lane is a plan for the whole game, the batch
    /// is a plan for the next few turns.
    pub lane_next_tech: Option<TechnologyType>,
    /// EXP_ELO_052: (city, road tiles still needed) for every city not yet
    /// linked to the capital. T2 runs the path search once; T3 prices each
    /// road tile by the progress it makes.
    pub connect_remaining: Vec<(i32, i32)>,
    /// Stance-intensity (Verdi, Aug 14): measured ARM pressure 0..1 from
    /// `stance_strength` — threat-vs-coverage truth, NOT the binary stance.
    /// The eco-tech mask fires only when this is near-certain (>= 0.98);
    /// below that ARM steers pricing, never masks.
    pub arm_strength: f32,
    /// Environment-recommended techs (owned ones pay the in-tree fit bonus).
    pub recommended_techs: Vec<TechnologyType>,
    /// Path-aware: a Rider reaches some EXPAND target at least
    /// `RIDER_PUSH_MIN_TURNS_SAVED` turns sooner than a walker would.
    pub rider_push: bool,
    /// Research moves this seat has executed this game.
    pub techs_bought: u32,
    /// …of which tier-3.
    pub tier3_bought: u32,
    /// v3 archetype: unit types the active doctrine + overlays prefer —
    /// each living one banks `SHAPE_GOAL_ARCHETYPE_UNIT` in the potential.
    pub preferred_units: Vec<crate::types::UnitType>,
    /// v3 reactive overlays; `knight_commit` also opens the
    /// FreeSpirit→Chivalry purchase lane (see `passes_tech_caps`).
    pub overlays: Overlays,
    /// v6 income lane: third city up + a hub structure standing → the
    /// Riding→Roads→Trade lane is recommended and Trade is exempt from the
    /// tier-3 cap (Market is the best ★/SPT purchase in the game).
    pub market_push: bool,
    /// v7: this seat already owns an economic tier-3 (by purchase OR ruin
    /// grant). Until it does, combat tier-3s are blocked — see
    /// `passes_tech_caps`. Ownership rather than purchases on purpose: a free
    /// economy tier-3 out of a ruin has already paid the ordering cost.
    pub eco_tier3_owned: bool,
    /// The map holds no water at all (Drylands), so the whole water tech lane
    /// buys nothing — see `passes_tech_caps`. Read from the true tile set, not
    /// the player's view: map type is public information at game start, unlike
    /// what happens to sit under the fog.
    pub water_dead: bool,
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
    arch: Option<&ArchetypeState>,
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
    // v3 archetype expression: doctrine + overlay tech lanes join the
    // recommendations (next unowned tech per lane), preferred unit classes
    // feed the in-tree unit bonus. FreeSpirit/Chivalry appear ONLY under a
    // knight commitment — the stepping-stone rule (Verdi, Jul 30).
    let mut preferred_units: Vec<crate::types::UnitType> = Vec::new();
    let mut overlays = Overlays::default();
    if let Some(arch) = arch {
        use crate::types::{TechnologyType as Tech, UnitType as U};
        overlays = arch.overlays;
        let owned = |t: Tech| {
            state
                .tribes
                .get(&player)
                .map_or(false, |tr| crate::settings::technology::is_tech_unlocked(&tr.tech_vanilla, t))
        };
        let push_lane = |lane: &[Tech], recs: &mut Vec<Tech>| {
            if let Some(t) = lane.iter().find(|t| !owned(**t)) {
                if !recs.contains(t) {
                    recs.push(*t);
                }
            }
        };
        if let Some(a) = arch.archetype {
            push_lane(lane_techs(a), &mut recommended);
            match a {
                Archetype::RiderRoads => preferred_units.push(U::Rider),
                Archetype::ArcherLine => preferred_units.push(U::Archer),
                Archetype::ForgeGiants => {
                    preferred_units.push(U::Swordsman);
                    preferred_units.push(U::Giant);
                }
            }
        }
        if overlays.defender_screen {
            push_lane(&[Tech::Strategy], &mut recommended);
            preferred_units.push(U::Defender);
        }
        if overlays.catapult_counter {
            push_lane(&[Tech::Forestry, Tech::Mathematics], &mut recommended);
            preferred_units.push(U::Catapult);
        }
        if overlays.knight_commit {
            push_lane(&[Tech::Riding, Tech::FreeSpirit, Tech::Chivalry], &mut recommended);
            preferred_units.push(U::Knight);
        }
    }
    // v6 income lane — archetype-independent: with the third city up and a
    // hub structure standing, Riding→Roads→Trade opens the Market.
    let market_push = market_ready(state, player);
    if market_push {
        use crate::types::TechnologyType as Tech;
        let owned = |t: Tech| {
            state.tribes.get(&player).map_or(false, |tr| {
                crate::settings::technology::is_tech_unlocked(&tr.tech_vanilla, t)
            })
        };
        if let Some(t) = [Tech::Riding, Tech::Roads, Tech::Trade]
            .iter()
            .find(|t| !owned(**t))
        {
            if !recommended.contains(t) {
                recommended.push(*t);
            }
        }
    }
    // EXP_ELO_051: the batch we are banking for is by definition on-plan, so
    // its own next step joins the whitelist — otherwise `passes_tech_caps`
    // would gate the very purchase the savings exist to make.
    // The committed lane's own next step — the chain walked to its deepest
    // unowned tech, so a two-step lane buys Riding before Roads.
    // Only the lane's OPENING tech is privileged. Iteration 3 privileged the
    // whole chain and Imperius spent its early stars walking Riding→Roads, a
    // chain whose payoff (a connection: 9 stars of road for +2 pop) is
    // dominated by a 5-star Windmill for the same +2 — so it lost its hub and
    // its giants (hubs@t15 0.94 → 0.44, giants 1.12 → 0.78). The opening tech
    // is the commitment; the rest of the chain competes on price like
    // everything else.
    let lane_next_tech = arch.and_then(|a| a.archetype).and_then(|a| {
        let tribe = state.tribes.get(&player)?;
        let first = *lane_techs(a).first()?;
        (!crate::settings::technology::has_technology(&tribe.tech_vanilla, first))
            .then_some(first)
    });
    let mut save_next_tech = None;
    if let Some(lane) = goal.save_target.as_ref() {
        if let Some(tribe) = state.tribes.get(&player) {
            let mut cur = Some(lane.tech);
            let mut guard = 0;
            while let Some(t) = cur {
                guard += 1;
                if guard > 16
                    || crate::settings::technology::has_technology(&tribe.tech_vanilla, t)
                {
                    break;
                }
                save_next_tech = Some(t);
                cur = crate::settings::technology::get_technology_setting(t).requires;
            }
            if let Some(t) = save_next_tech {
                if !recommended.contains(&t) {
                    recommended.push(t);
                }
            }
        }
    }
    let water_dead = !state
        .tiles
        .values()
        .any(|t| matches!(t.terrain_type, TerrainType::Water | TerrainType::Ocean));
    // Aquatism's WaterTemple yields population, so it counts as an economic
    // tier-3 by the table — but on a dry map it can never be built, and letting
    // it satisfy the economy-first rule would unblock the combat lane for free.
    let eco_tier3_owned = state.tribes.get(&player).map_or(false, |t| {
        t.tech_vanilla.iter().any(|tech| {
            tech.discovered
                && crate::settings::technology::is_eco_tier3(tech.tech_type)
                && !(water_dead && crate::settings::technology::is_water_tech(tech.tech_type))
        })
    });
    GoalAux {
        // T2 assesses; T3 prices its response against it.
        city_risk: crate::ai::defense::city_risks(state, player),
        save_lane: goal.save_target.clone(),
        save_next_tech,
        lane_next_tech,
        connect_remaining: connect_remaining(state, player),
        arm_strength: stance_strength(state, player).arm,
        recommended_techs: recommended,
        rider_push,
        techs_bought,
        tier3_bought,
        preferred_units,
        overlays,
        market_push,
        eco_tier3_owned,
        water_dead,
    }
}

/// v6: the market limb is worth opening — third city up and at least one
/// hub structure (anything in Market's adjacent_types: Sawmill / Windmill /
/// Forge) standing in own territory. Derived from settings tables.
pub fn market_ready(state: &GameState, player: PlayerId) -> bool {
    let Some(tribe) = state.tribes.get(&player) else {
        return false;
    };
    if tribe.cities.len() < COMMIT_CITY_TARGET {
        return false;
    }
    let hubs = &crate::settings::structures::get_structure_setting(StructureType::Market)
        .adjacent_types;
    state.structures.iter().any(|(idx, s)| {
        s.as_ref().map_or(false, |s| hubs.contains(&s.structure_type))
            && state.tiles.get(idx).map_or(false, |t| t.owner == player)
    })
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
    // EXP_ELO_051 lane discipline. This lives here, not behind `star_gate`,
    // because that gate switches OFF under GROW once the third city is up —
    // which is exactly when the fixtures bought Organization. Tech discipline
    // is not conditional on the spending stance.
    //
    // Verdi: "the terrain is telling us we clearly just need to get metal and
    // forge thats it." While a batch is live, only purchases that advance it
    // pass; before one is affordable, `recommended_techs` — the committed
    // lane's next step PLUS whatever the overlays demand (defender screen,
    // catapult counter, knight commit, the market lane) — is the whitelist.
    // Overlays are what keep this a judgment call rather than a freeze: a
    // real threat still buys its answer.
    if let Ok(tech) = m.tech_type() {
        // A committed knight lane is a whole chain, not just its next step —
        // the exemption `passes_star_gate` already grants it.
        let knight_lane = aux.overlays.knight_commit
            && matches!(
                tech,
                TechnologyType::Riding | TechnologyType::FreeSpirit | TechnologyType::Chivalry
            );
        if !knight_lane {
            // EXP_ELO_052: the COMMITTED LANE'S OWN TECHS COME FIRST. Before
            // this, a RiderRoads seat could have Riding gated out by a
            // Windmill batch — the lane-discipline machinery forbidding the
            // lane's opening move. Measured: `Research Riding` reached the
            // ballot in 8 of 764 plies over t0-t5 on Imperius, and the first
            // Rider landed t11.6 against a target of t3.
            // The lane's own next tech is always permitted — never gated by
            // a hub batch, which is the bug that kept Riding off a
            // RiderRoads ballot. It is NOT exclusive: iteration 2 made it so
            // and starved the economy (Imperius hubs@t15 0.94 -> 0.09,
            // giants 1.12 -> 0.38), because no eco tech could be bought while
            // the lane was unfinished.
            if aux.lane_next_tech == Some(tech) {
                // allowed
            } else if let Some(next) = aux.save_next_tech {
                // Lane complete: bank for the hub, and only its next step is
                // not a delay. `save_batch_plan` self-terminates once the
                // batch is affordable, so this never freezes research.
                if tech != next {
                    return false;
                }
            } else if !aux.recommended_techs.is_empty()
                && !aux.recommended_techs.contains(&tech)
            {
                return false;
            }
        }
    }
    // v7.1 (Verdi, Aug 2026): on a map with no water the whole naval lane —
    // Fishing/Sailing/Ramming/Navigation/Aquatism — unlocks nothing buildable
    // and nothing reachable behind it. A mask, not a price: this is the
    // never-do case masks exist for.
    if aux.water_dead {
        if let Ok(tech) = m.tech_type() {
            if crate::settings::technology::is_water_dead_end(tech) {
                return false;
            }
        }
    }
    // v3 stepping-stone rule: FreeSpirit has no standalone value — it (and
    // Chivalry behind it) is buyable only under an active knight commitment.
    if !aux.overlays.knight_commit {
        if let Ok(t) = m.tech_type() {
            if t == TechnologyType::FreeSpirit || t == TechnologyType::Chivalry {
                return false;
            }
        }
    }
    // v7 economy-first ordering: a combat tier-3 (Chivalry/Navigation/
    // Diplomacy — the ones unlocking no yielding structure) waits until an
    // economic tier-3 is owned. Real games almost never take knights before
    // the level-3 pop buildings, because those are what lead to giants; the
    // exception (a lucky free-spirit ruin) is covered by reading OWNERSHIP,
    // so a free economy tier-3 unblocks the lane immediately.
    if let Ok(tech) = m.tech_type() {
        let setting = crate::settings::technology::get_technology_setting(tech);
        if setting.tier == Some(3)
            && !crate::settings::technology::is_eco_tier3(tech)
            && !aux.eco_tier3_owned
        {
            return false;
        }
    }
    if aux.tier3_bought >= TIER3_CAP_PER_GAME {
        if let Ok(tech) = m.tech_type() {
            if crate::settings::technology::get_technology_setting(tech).tier == Some(3) {
                // v6: per-overlay tier-3 exemptions, not a global cap raise —
                // an ACTIVE knight commitment exempts Chivalry, an active
                // market push exempts Trade (both still count toward the
                // whole-game cap). Otherwise the terrain lane's tier-3,
                // bought by ~t12, permanently locked these lanes out.
                let exempt = (aux.overlays.knight_commit && tech == TechnologyType::Chivalry)
                    || (aux.market_push && tech == TechnologyType::Trade);
                if !exempt {
                    return false;
                }
            }
        }
    }
    true
}

/// Root-only ability gate — applied whenever a `GoalAux` is set, like the
/// tech caps. Destroy (demolish own structure) is masked out entirely: the
/// Jul 30 2026 gauge audit measured ~9 destroys/game of pure churn, and the
/// rare strategic rebuild is deferred until the net owns the basics.
///
/// v8: Clear/Burn Forest on a tile carrying a resource joins it. The ability is
/// free and pays a star, but `consume_resource` DELETES the Game sitting there
/// — trading a harvestable pop source for one star is dominated at every star
/// price, so it is a mask rather than a price. Clearing bare forest stays
/// legal and is priced by `SHAPE_GOAL_FOREST_STANDING`.
pub fn passes_ability_gate(state: &GameState, m: &dyn Move) -> bool {
    if m.move_type() != MoveType::Ability {
        return true;
    }
    let Ok(ability) = m.ability_type() else {
        return true;
    };
    if ability == AbilityType::Destroy {
        return false;
    }
    if matches!(ability, AbilityType::ClearForest | AbilityType::BurnForest) {
        if let Ok(idx) = m.target_idx() {
            if matches!(state.resources.get(&(idx as i32)), Some(Some(_))) {
                return false;
            }
        }
    }
    true
}

/// Root-only capture-first gate (v6): a unit standing on a capturable
/// village/ruin never attacks — capture is strictly better (city defense
/// bonus, unit production; attacking forfeits the capture turn and eats
/// free retaliation). Applies even when Capture isn't legal this ply (the
/// unit stepped on this turn): it idles and captures next turn.
pub fn passes_capture_first(state: &GameState, m: &dyn Move) -> bool {
    if m.move_type() != MoveType::Attack {
        return true;
    }
    let Ok(src) = m.source_idx() else {
        return true;
    };
    let src = src as i32;
    let structure = state
        .structures
        .get(&src)
        .and_then(|s| s.as_ref())
        .map(|s| s.structure_type);
    match structure {
        Some(StructureType::Ruin) => false,
        Some(StructureType::Village) => {
            let owner = state.tiles.get(&src).map_or(0, |t| t.owner);
            owner == state.settings.current_player_turn_id
        }
        _ => true,
    }
}

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

/// Greedy unique unit→EXPAND-target assignment, nearest pair first. Each
/// target's approach term pays only its assigned unit, so two scouts never
/// bank progress on the same fog target (audit: 89% duplicate-sector
/// scouting). Deterministic: ties break on (unit idx, target idx).
pub fn assign_expand_targets(
    state: &GameState,
    player: PlayerId,
    targets: &[i32],
) -> Vec<(i32, i32)> {
    let size = state.settings.size as i32;
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    if size <= 0 {
        return Vec::new();
    }
    let cheb =
        |a: i32, b: i32| crate::functions::get_chebyshev_distance(a, b, size);
    // v6: real (explored) targets outrank fog guesses in pairing — a scarce
    // unit must never be pinned to a guess while a discovered village waits.
    let is_guess = |t: i32| {
        !state
            .tiles
            .get(&t)
            .map_or(false, |tile| tile.explorers.contains(&player))
    };
    let mut pairs: Vec<(bool, i32, i32, i32)> = Vec::new();
    for u in &tribe.units {
        for &t in targets {
            pairs.push((is_guess(t), cheb(u.coords.idx, t), u.coords.idx, t));
        }
    }
    pairs.sort_unstable();
    let mut used_u = std::collections::HashSet::new();
    let mut used_t = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (_, _, u, t) in pairs {
        if used_u.contains(&u) || used_t.contains(&t) {
            continue;
        }
        used_u.insert(u);
        used_t.insert(t);
        out.push((u, t));
    }
    out
}

// ========================= Archetype layer (v3) =========================
// Doctrine chosen from ground-truth predicates, sticky with hysteresis,
// expressed through tech recommendations, unit pricing, and the
// stepping-stone tech gate. No new input channels: every predicate is a
// function of state the net already sees (terrain, ghost units, economy).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Archetype {
    /// Open map + live expansion race + real route advantage, enemy not
    /// heavy-dominant. Buys Riding→Roads ONLY (FreeSpirit is a stepping
    /// stone redeemed solely by a knight commitment).
    RiderRoads,
    /// Anti-heavy/siege AND push support: range beats high defense, and a
    /// backline wears targets down while warriors advance.
    ArcherLine,
    /// Metal-rich explored map, no immediate threat: Mining→Smithery,
    /// forge economy into giants/swordsmen.
    ForgeGiants,
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

/// Per-seat persistent archetype state, threaded through the play loop like
/// the tech counters. Peak counts approximate observation memory: a unit
/// seen once stays counted after it retreats into fog (the net's ghost
/// channels carry the same information).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArchetypeState {
    pub archetype: Option<Archetype>,
    pub overlays: Overlays,
    pub seen_squishy: u32,
    pub seen_heavy: u32,
    pub seen_cavalry: u32,
    pub seen_ranged: u32,
    challenger: Option<Archetype>,
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
pub const LANE_ORDER: [Archetype; LANES] =
    [Archetype::RiderRoads, Archetype::ArcherLine, Archetype::ForgeGiants];

/// The tech chain that *is* the lane — single source of truth, consumed by
/// `scripted_goal_aux`'s recommendations and by the spawn tribe prior.
pub fn lane_techs(a: Archetype) -> &'static [TechnologyType] {
    use TechnologyType as T;
    match a {
        Archetype::RiderRoads => &[T::Riding, T::Roads],
        Archetype::ArcherLine => &[T::Hunting, T::Archery],
        Archetype::ForgeGiants => &[T::Climbing, T::Mining, T::Smithery],
    }
}

/// Lane the tribe is born into: mapgen stamps one tribe tech at turn 0
/// (`mapgen.rs`), and if it opens a lane's chain that lane starts ahead —
/// "your tribe sets the tone" before any terrain is explored. Derived from
/// `lane_techs`, so the two can never drift apart.
pub fn tribe_lane_prior(state: &GameState, player: PlayerId) -> Option<Archetype> {
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
pub const ARCH_ENTRY_MIN: i32 = 3;
/// A challenger must outscore the incumbent by this margin…
pub const ARCH_SWITCH_MARGIN: i32 = 2;
/// …for this many distinct turns before a soft switch (hysteresis).
pub const ARCH_SWITCH_TURNS: u8 = 3;
/// Explored-land open-field share for rider terrain.
pub const OPEN_FRAC_RIDER: f32 = 0.45;
/// Explored-land rough share for archer terrain.
pub const ROUGH_FRAC_ARCHER: f32 = 0.30;
/// Explored metal resources for the forge line to be worth committing.
pub const METAL_FORGE_MIN: i32 = 2;
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
    metal: i32,
}

fn read_map(state: &GameState, player: PlayerId) -> MapRead {
    use crate::types::{ResourceType as R, TerrainType as T};
    let (mut open, mut rough, mut land, mut metal) = (0i32, 0i32, 0i32, 0i32);
    for (idx, tile) in state.tiles.iter() {
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
        if let Some(Some(r)) = state.resources.get(idx) {
            if r.resource_type == R::Metal {
                metal += 1;
            }
        }
    }
    let denom = land.max(1) as f32;
    MapRead { open_frac: open as f32 / denom, rough_frac: rough as f32 / denom, metal }
}

/// Update peak seen-counts from enemy units standing on tiles this player
/// has explored — the script-side proxy for the ghost channels.
fn observe_enemies(state: &GameState, player: PlayerId, st: &mut ArchetypeState) {
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
fn archetype_scores(
    state: &GameState,
    player: PlayerId,
    goal: &MacroGoal,
    st: &ArchetypeState,
    map: &MapRead,
) -> [(Archetype, i32); 3] {
    let tribe_cities =
        state.tribes.get(&player).map_or(0, |t| t.cities.len());
    let race_live = tribe_cities < COMMIT_CITY_TARGET
        || goal.orders.iter().any(|(k, _)| *k == OrderKind::Expand);
    let expand_targets: Vec<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Expand)
        .map(|(_, i)| *i)
        .collect();
    let mobility = !expand_targets.is_empty()
        && rider_turns_saved(state, player, &expand_targets) >= RIDER_PUSH_MIN_TURNS_SAVED;

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
    let has_defend = goal.orders.iter().any(|(k, _)| *k == OrderKind::Defend);
    let forge = 2 * (map.metal >= METAL_FORGE_MIN) as i32
        + (map.rough_frac >= ROUGH_FRAC_ARCHER) as i32
        + (!has_defend) as i32;
    [
        (Archetype::RiderRoads, rider),
        (Archetype::ArcherLine, archer),
        (Archetype::ForgeGiants, forge),
    ]
}

/// Per-ply archetype update: observe enemies (peaks), refresh overlays,
/// then enter/hold/switch the base doctrine with hysteresis — hard exits
/// fire immediately (score drops to 0), soft switches need the challenger
/// to outscore by `ARCH_SWITCH_MARGIN` for `ARCH_SWITCH_TURNS` turns.
/// Per-ply entry point for the script paths: observe every ply, but run the
/// Tier-1 selector only at a turn boundary (or to make the very first
/// commit). Callers that already sit on a turn boundary — the macro agent's
/// replan branch — call `observe_archetype` + `select_playstyle` directly.
pub fn update_archetype(
    state: &GameState,
    player: PlayerId,
    goal: &MacroGoal,
    st: &mut ArchetypeState,
) {
    let before = st.overlays;
    observe_archetype(state, player, st);
    // Refutation bypasses the turn boundary, mirroring the stance layer's
    // urgent-threat path (`update_goal`): new counter-evidence — the
    // sighting that flips an overlay is exactly what zeroes a lane's score —
    // must not wait a turn to be acted on. Discretionary switches still wait,
    // and the pivot budget still binds either way.
    let refuted = st.overlays != before;
    if state.settings.turn != st.last_turn || st.archetype.is_none() || refuted {
        select_playstyle(state, player, goal, st, None);
    }
}

/// Per-ply half: peak enemy-type counts and the reactive overlays. Cheap,
/// runs on every executor ply. Selection deliberately does NOT happen here —
/// a lane recomputed 20x a turn is a running average, not an identity.
pub fn observe_archetype(state: &GameState, player: PlayerId, st: &mut ArchetypeState) {
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
pub fn select_playstyle(
    state: &GameState,
    player: PlayerId,
    goal: &MacroGoal,
    st: &mut ArchetypeState,
    head: Option<&[f32; LANES]>,
) -> Option<Archetype> {
    let map = read_map(state, player);
    let turn = state.settings.turn;
    let prior = tribe_lane_prior(state, player);

    let mut scores = archetype_scores(state, player, goal, st, &map);
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
    let score_of = |a: Archetype| {
        LANE_ORDER.iter().position(|k| *k == a).map_or(0.0, |i| st.last_scores[i])
    };
    let (best, best_score) = LANE_ORDER
        .iter()
        .map(|a| (*a, score_of(*a)))
        .fold((Archetype::RiderRoads, f32::NEG_INFINITY), |acc, x| {
            if x.1 > acc.1 { x } else { acc }
        });

    let new_turn = turn != st.last_turn;
    st.last_turn = turn;
    let entry_min = ARCH_ENTRY_MIN as f32;

    match st.archetype {
        None => {
            let pick = if best_score >= entry_min { Some(best) } else { prior };
            if pick.is_some() {
                st.archetype = pick;
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
                st.archetype = (best_score >= entry_min).then_some(best);
                if st.archetype.is_some() {
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
                && best_score >= score_of(cur) + ARCH_SWITCH_MARGIN as f32
            {
                if st.challenger == Some(best) {
                    if new_turn {
                        st.streak += 1;
                    }
                } else {
                    st.challenger = Some(best);
                    st.streak = 1;
                }
                if st.streak >= ARCH_SWITCH_TURNS {
                    st.archetype = Some(best);
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
    st.archetype
}


/// Root-only research gate (v9, dual-exempt). Every non-Research move passes.
/// A gated Research move is dropped outright — the old "unless you keep
/// `STAR_GATE_RESERVE` stars" escape is gone. It read as a soft price but the
/// measured policy is hand-to-mouth (median 1 star at EndTurn), so it opened on
/// 0.5% of gated GROW plies: a constant pretending to be a decision.
///
/// A stance gates only the tech that is PURELY the other class. Dual-class tech
/// serves both and is never dropped — Smithery fields a Swordsman AND opens the
/// Forge, so gating it under GROW while ARM lets it through made the Forge lane
/// hostage to a stance that is ARM 85% of the time after turn 10.
/// - `Some(Grow)` / `Some(Save)`: pure-combat tech (no eco effect). Both are
///   economy stances in `reward::goal_potential`; SAVE additionally cannot gate
///   its own batch, whose cost includes an unowned tech chain.
/// - `Some(Arm)`: pure-eco tech (fields no combat unit).
/// - `Some(Unlock)`: nothing gated (no unlock policy yet).
/// - `None`: every tech (the EXP_ELO_026 legacy gate, kept reproducible for
///   arena `--macro-star-gate`; now a hard drop rather than a reserve test).
pub fn passes_star_gate(
    _state: &GameState,
    m: &dyn Move,
    stance: Option<Stance>,
    aux: Option<&GoalAux>,
) -> bool {
    if m.move_type() != MoveType::Research {
        return true;
    }
    let Ok(tech) = m.tech_type() else {
        return true;
    };
    // v6: an active knight commitment makes its lane stance-coherent — the
    // stance-class gating no longer applies (GROW gated Chivalry as combat
    // tech while ARM gated FreeSpirit as eco tech, blocking the lane from
    // both sides). passes_tech_caps already restricts the lane to commits.
    if (tech == TechnologyType::FreeSpirit || tech == TechnologyType::Chivalry)
        && aux.map_or(false, |a| a.overlays.knight_commit)
    {
        return true;
    }
    // EXP_ELO_051: a live savings plan outranks the stance classes. While we
    // are banking for a named lane, a tech that does not advance it is the
    // purchase that delays it — which is exactly the "random tech" Verdi
    // flagged (Organization on t4-t5 of every fixture seed). Bounded by the
    // plan's own lifetime: `save_batch_plan` self-terminates once the batch
    // is affordable, so this never becomes an open-ended research freeze.
    let effects = crate::settings::technology::get_tech_effects(tech);
    let arms = !effects.combat_units.is_empty();
    let grows = crate::settings::technology::is_eco_tech(tech);
    let gated = match stance {
        None => true,
        Some(Stance::Grow) | Some(Stance::Save) => arms && !grows,
        // Verdi (Aug 14): masking is legitimate only at near-certain need —
        // below that, a covered skirmish must not lock the eco lanes.
        Some(Stance::Arm) => grows && !arms && aux.map_or(true, |a| a.arm_strength >= 0.98),
        Some(Stance::Unlock) => false,
    };
    !gated
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::EndTurnMove;
    use crate::moves::research::ResearchMove;
    use crate::Coords;
    use crate::states::{StructureState, TechnologyState, TileState, TribeState, UnitState};
    use crate::types::{TechnologyType, TribeType, UnitType};

    /// End-to-end: a real generated Drylands game must report `water_dead`
    /// through the same path self_play uses, and mask the naval lane there.
    #[test]
    fn a_generated_drylands_game_masks_the_water_lane() {
        let mut game = crate::game::Game::new();
        for seed in 0..8 {
            game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
                size: crate::types::MapSize::Tiny,
                map_type: crate::types::MapType::Drylands,
                tribes: vec![crate::types::TribeType::Imperius, crate::types::TribeType::Bardur],
                seed,
                version: 115,
            });
            game.post_load();
            let goal = MacroGoal::default();
            let aux = scripted_goal_aux(&game.state, 1, &goal, 0, 0, None);
            assert!(aux.water_dead, "seed {seed}: generated Drylands still reads wet");
            assert!(
                !passes_tech_caps(&ResearchMove::new(TechnologyType::Fishing), &aux),
                "seed {seed}: Fishing survived the mask"
            );
        }
    }

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

    /// A bare city with nothing happening: no military pressure either way.
    #[test]
    fn stance_strength_is_zero_arm_in_a_quiet_position() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        state.tribes.insert(1, t1);
        let s = stance_strength(&state, 1);
        assert_eq!(s.arm, 0.0);
        assert_eq!(s.cause, ArmCause::None);
    }

    /// The distinction the categorical stance throws away: one enemy scout near
    /// one of three cities is a weak signal; a stack pressing the only city I
    /// have, with nothing defending, is near-maximal. Both are `Stance::Arm`.
    #[test]
    fn threat_strength_scales_with_how_much_is_pressed() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        for idx in [12, 60, 108] {
            t1.cities.push(crate::states::CityState { idx, ..Default::default() });
        }
        // Own defenders sitting on each city.
        for idx in [12, 60, 108] {
            t1.units.push(unit_at(idx));
        }
        state.tribes.insert(1, t1);
        let mut t2 = TribeState::default();
        t2.units.push(unit_at(61)); // adjacent to city 60 only
        state.tribes.insert(2, t2);

        let weak = stance_strength(&state, 1);
        assert_eq!(weak.cause, ArmCause::Threat);
        assert!(weak.arm > 0.0 && weak.arm < 0.25, "one of three cities, defended: {}", weak.arm);

        // Now: a single undefended city with three enemies on it.
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        state.tribes.insert(1, t1);
        let mut t2 = TribeState::default();
        for idx in [59, 61, 71] {
            t2.units.push(unit_at(idx));
        }
        state.tribes.insert(2, t2);

        let dire = stance_strength(&state, 1);
        assert_eq!(dire.cause, ArmCause::Threat);
        assert!(dire.arm > 0.9, "sole city, undefended, surrounded: {}", dire.arm);
        assert!(dire.arm > weak.arm * 3.0);
    }

    /// The other route to a high ARM: overwhelming force with somewhere to put
    /// it. Reported as MOMENTUM, not THREAT — they want opposite economies.
    #[test]
    fn army_dominance_reads_as_momentum_not_threat() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 0, ..Default::default() });
        // Six attackers massed on the enemy city at 60, far from home.
        for idx in [48, 49, 50, 59, 61, 70] {
            t1.units.push(unit_at(idx));
        }
        state.tribes.insert(1, t1);
        let mut t2 = TribeState::default();
        t2.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        state.tribes.insert(2, t2);

        let s = stance_strength(&state, 1);
        assert_eq!(s.cause, ArmCause::Momentum);
        assert!(s.arm > 0.9, "total army dominance with a target: {}", s.arm);
    }

    /// Parity is not momentum, however many units are on the board.
    #[test]
    fn even_armies_produce_no_momentum() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 0, ..Default::default() });
        let mut t2 = TribeState::default();
        t2.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        for idx in [30, 31] {
            t1.units.push(unit_at(idx));
            t2.units.push(unit_at(idx + 50));
        }
        state.tribes.insert(1, t1);
        state.tribes.insert(2, t2);
        let s = stance_strength(&state, 1);
        assert_eq!(s.arm, 0.0, "parity must not read as momentum");
    }

    /// GROW tracks available economy: open villages to take, or stars that
    /// could already be converted into population.
    #[test]
    fn grow_strength_rises_with_capturable_villages() {
        let quiet = {
            let mut state = GameState::default();
            let mut t1 = TribeState::default();
            t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
            state.tribes.insert(1, t1);
            stance_strength(&state, 1).grow
        };
        let mut state = state_with_villages(0, &[3, 5, 7]);
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 60,
            ..Default::default()
        });
        let rich = stance_strength(&state, 1).grow;
        assert!(rich > quiet, "three open villages must beat none: {rich} vs {quiet}");
        assert!(rich >= 1.0);
    }

    #[test]
    fn capture_first_gate_blocks_attacks_from_capturable_tiles() {
        use crate::moves::attack::AttackMove;
        let mut state = state_with_villages(10, &[10]);
        state.settings.current_player_turn_id = 1;
        let attack = AttackMove::new(10, 11);

        // Standing on a neutral village: attack blocked.
        assert!(!passes_capture_first(&state, &attack));

        // Enemy-owned village (their city): still blocked — recapture instead.
        state.tiles.get_mut(&10).unwrap().owner = 2;
        assert!(!passes_capture_first(&state, &attack));

        // Own city tile: attack allowed.
        state.tiles.get_mut(&10).unwrap().owner = 1;
        assert!(passes_capture_first(&state, &attack));

        // Ruin: blocked.
        state.structures.insert(
            10,
            Some(StructureState {
                structure_type: StructureType::Ruin,
                level: 0,
                founded: 0,
            }),
        );
        assert!(!passes_capture_first(&state, &attack));

        // Plain tile: allowed; non-attack moves always pass.
        state.structures.shift_remove(&10);
        assert!(passes_capture_first(&state, &attack));
        assert!(passes_capture_first(&state, &EndTurnMove));
    }

    #[test]
    fn retakeable_village_predicate_and_radius() {
        let mut state = state_with_villages(13, &[12]);
        state.settings.size = 11;
        // Neutral: not retakeable (still_capturable covers it).
        assert!(!retakeable_village(&state, 12, 1));
        // Enemy-owned, explored, within radius of our unit at 10: retakeable.
        state.tiles.get_mut(&12).unwrap().owner = 2;
        assert!(retakeable_village(&state, 12, 1));
        // Enemy capital: never painted.
        state.tiles.get_mut(&12).unwrap().capital_of = 2;
        assert!(!retakeable_village(&state, 12, 1));
        state.tiles.get_mut(&12).unwrap().capital_of = 0;
        // Beyond RETAKE_PAINT_RADIUS: not painted (move our unit far away).
        state.tribes.get_mut(&1).unwrap().units[0] = unit_at(120);
        assert!(!retakeable_village(&state, 12, 1));
    }

    #[test]
    fn real_target_outranks_fog_guess_in_assignment() {
        // One unit, two targets: a fog guess NEARBY and a real explored
        // village further out — the unit must pair with the real one.
        let mut state = state_with_villages(0, &[5]);
        state.settings.size = 11;
        // Fog guess at 2 (no tile entry → unexplored), real village at 5.
        let pairs = assign_expand_targets(&state, 1, &[2, 5]);
        assert_eq!(pairs, vec![(0, 5)]);
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

    /// EXP_ELO_051 — Verdi: "we should be saving towards a lane if that is
    /// what T1 says … the best computed path for that giant spam is forges."
    /// A tribe holding Climbing+Mining is walking the Forge lane even when a
    /// Windmill is cheaper, and a mountain that could take a Mine counts as a
    /// Forge partner before the Mine is standing — otherwise the plan waits
    /// on builds that nothing is planning.
    #[test]
    fn the_invested_lane_wins_and_future_mines_count_as_partners() {
        use crate::types::{TechnologyType as T, TerrainType};
        let mut state = state_with_villages(0, &[3, 5]);
        {
            let t1 = state.tribes.get_mut(&1).unwrap();
            for tech in [T::Climbing, T::Mining, T::Organization, T::Farming] {
                t1.tech_vanilla.push(crate::states::TechnologyState {
                    tech_type: tech,
                    discovered: true,
                    discovered_turn: 0,
                });
            }
            t1.stars = 30;
            t1.cities.push(crate::states::CityState {
                idx: 60,
                owner: 1,
                _territory: vec![60, 61, 50, 72],
                production: 3,
                ..Default::default()
            });
        }
        // 61 is bare field (a Forge site); 50 and 72 are ore mountains with no
        // mine on them yet — the exact board that used to price zero.
        for idx in [60, 61, 50, 72] {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.owner = 1;
            // Resource visibility is FOW-honest: unexplored ground is not a plan.
            tile.explorers.insert(1);
            tile.terrain_type = if idx == 50 || idx == 72 {
                TerrainType::Mountain
            } else {
                TerrainType::Field
            };
        }
        for idx in [50, 72] {
            state.resources.insert(
                idx,
                Some(crate::states::ResourceState {
                    resource_type: crate::types::ResourceType::Metal,
                }),
            );
        }
        let plan = save_batch_plan(&state, 1, 0, None).expect("an unbuilt mine still makes a site");
        assert_eq!(
            plan.structure,
            crate::types::StructureType::Forge,
            "the invested lane must win, got {:?}",
            plan.structure
        );
        assert_eq!(plan.tech, T::Smithery);

        // …and the batch never grows past the next two placements.
        assert!(
            plan.structure_cost <= plan.structure_unit_cost * SAVE_MAX_PLACEMENTS,
            "structure_cost {} exceeds two placements",
            plan.structure_cost
        );
    }

    /// While banking, research that is not the batch's own next step is the
    /// purchase that delays it — the Organization buy Verdi flagged.
    #[test]
    fn banking_gates_research_that_is_not_the_plan() {
        use crate::types::TechnologyType as T;
        let mut aux = GoalAux::default();
        aux.save_next_tech = Some(T::Mining);
        aux.recommended_techs = vec![T::Mining];
        assert!(passes_tech_caps(&ResearchMove::new(T::Mining), &aux));
        assert!(!passes_tech_caps(&ResearchMove::new(T::Organization), &aux));
        // No batch: the committed lane's recommendations are the whitelist.
        aux.save_next_tech = None;
        assert!(passes_tech_caps(&ResearchMove::new(T::Mining), &aux));
        assert!(!passes_tech_caps(&ResearchMove::new(T::Organization), &aux));
        // No opinion at all: nothing is gated on lane grounds.
        aux.recommended_techs.clear();
        assert!(passes_tech_caps(&ResearchMove::new(T::Organization), &aux));
    }

    /// v7: SAVE fires only for a batch that is out of pocket now but inside
    /// SAVE_MAX_TURNS of income, so it self-terminates instead of becoming an
    /// open-ended hoard — the failure mode a savings reward invites.
    #[test]
    fn save_stance_targets_a_reachable_batch_and_self_terminates() {
        use crate::types::{StructureType, TechnologyType};
        let mut state = state_with_villages(0, &[3, 5]);
        {
            let t1 = state.tribes.get_mut(&1).unwrap();
            t1.stars = 40;
            for tech in [
                TechnologyType::Organization,
                TechnologyType::Farming,
                TechnologyType::Construction,
            ] {
                t1.tech_vanilla.push(crate::states::TechnologyState {
                    tech_type: tech,
                    discovered: true,
                    discovered_turn: 0,
                });
            }
            t1.cities.push(crate::states::CityState {
                idx: 60,
                owner: 1,
                _territory: vec![60, 61, 50, 72],
                production: 2, // income, so the batch is reachable at all
                ..Default::default()
            });
        }
        // Two standing Farms around the empty field at 61 → a Windmill worth
        // banking for (2 partners clears SAVE_MIN_PARTNERS).
        for idx in [50, 72] {
            state.structures.insert(
                idx,
                Some(crate::states::StructureState {
                    structure_type: StructureType::Farm,
                    ..Default::default()
                }),
            );
        }
        for idx in [60, 61, 50, 72] {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.owner = 1;
            tile.terrain_type = crate::types::TerrainType::Field;
        }
        assert_eq!(save_batch_plan(&state, 1, 0, None).map(|l| l.cost), Some(5),
            "one 5-star windmill, tech owned");

        // The lane is what costs: drop Construction and the batch must absorb
        // the tier-3 tech price, which is the thing actually worth banking for.
        {
            state
                .tribes
                .get_mut(&1)
                .unwrap()
                .tech_vanilla
                .retain(|t| t.tech_type != TechnologyType::Construction);
            let tech_cost = crate::functions::get_tech_cost(
                &state.tribes[&1],
                TechnologyType::Construction,
            );
            assert!(tech_cost > 0);
            assert_eq!(save_batch_plan(&state, 1, 0, None).map(|l| l.cost), Some(5 + tech_cost));
            state.tribes.get_mut(&1).unwrap().tech_vanilla.push(
                crate::states::TechnologyState {
                    tech_type: TechnologyType::Construction,
                    discovered: true,
                    discovered_turn: 0,
                },
            );
        }

        // Broke but within reach → SAVE with the batch named.
        state.tribes.get_mut(&1).unwrap().stars = 1;
        let g = scripted_goal(&state, 1, 0, None);
        assert_eq!(g.stance, Stance::Save);
        assert_eq!(g.save_target.as_ref().map(|l| l.cost), Some(5));

        // Already affordable → nothing to save for, back to GROW.
        state.tribes.get_mut(&1).unwrap().stars = 5;
        let g = scripted_goal(&state, 1, 0, None);
        assert_eq!(g.stance, Stance::Grow);
        assert_eq!(g.save_target, None);

        // Out of reach (no income, batch unaffordable for SAVE_MAX_TURNS) →
        // GROW rather than an indefinite hoard.
        state.tribes.get_mut(&1).unwrap().stars = 0;
        state.tribes.get_mut(&1).unwrap().cities[0].production = 0;
        let far = scripted_goal(&state, 1, 0, None);
        assert!(
            far.stance != Stance::Save || far.save_target.is_some(),
            "SAVE is only ever set together with a named target"
        );
    }

    /// v7: a discretionary stance swing must hold for STANCE_SWITCH_TURNS
    /// turns, and re-running the same turn's plies must not advance the streak
    /// (the goal-setter runs every ply, the commitment counts turns).
    #[test]
    fn stance_commitment_damps_discretionary_swings_across_turns() {
        let mut st = StanceCommit::default();
        let mut state = state_with_villages(0, &[3, 5]);
        state.settings.turn = 1;

        // First read commits immediately — nothing to be loyal to yet.
        assert_eq!(update_goal(&state, 1, &mut st, 0).stance, Stance::Grow);
        assert_eq!(st.stance, Some(Stance::Grow));

        // Force the script to want ARM: post-expansion "prepare" phase — an
        // explored enemy city we outweigh but cannot yet storm, at 3+ cities.
        let mut t2 = TribeState::default();
        t2.cities.push(crate::states::CityState { idx: 40, ..Default::default() });
        t2.units.push(unit_at(41));
        state.tribes.insert(2, t2);
        explore_tile(&mut state, 40);
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(29));
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        assert_eq!(
            scripted_goal(&state, 1, 0, None).stance,
            Stance::Arm,
            "precondition: script wants ARM here"
        );

        // Same turn, several plies: the challenger must not accrue a streak.
        for _ in 0..4 {
            assert_eq!(
                update_goal(&state, 1, &mut st, 0).stance,
                Stance::Grow,
                "extra plies of one turn must not buy a stance switch"
            );
        }
        // Next turn: streak reaches STANCE_SWITCH_TURNS and the switch lands.
        state.settings.turn = 2;
        assert_eq!(update_goal(&state, 1, &mut st, 0).stance, Stance::Arm);
        assert_eq!(st.stance_flips, 1);
    }

    /// Threat responses bypass the hysteresis — a DEFEND order means an enemy
    /// is already inside the threat radius, and arriving two turns late is the
    /// same as not arriving.
    #[test]
    fn stance_commitment_lets_threat_response_switch_immediately() {
        let mut st = StanceCommit::default();
        let mut state = state_with_villages(0, &[3, 5]);
        state.settings.turn = 1;
        assert_eq!(update_goal(&state, 1, &mut st, 0).stance, Stance::Grow);

        // Visible deliverable strike on an own city → DEFEND → ARM (040:
        // threat math, not the old position count — stats must be real).
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            ..Default::default()
        });
        let t2 = state.tribes.entry(2).or_insert_with(TribeState::default);
        t2.id = 2;
        t2.units.push(unit_at(1));
        t2.units.push(unit_at(11));
        explore_tile(&mut state, 0);
        explore_tile(&mut state, 1);
        explore_tile(&mut state, 11);
        {
            let t1 = state.tribes.get_mut(&1).unwrap();
            t1.units[0].owner = 1;
            t1.units[0].health = 10.0;
        }
        for u in state.tribes.get_mut(&2).unwrap().units.iter_mut() {
            u.owner = 2;
            u.health = 10.0;
        }
        let g = update_goal(&state, 1, &mut st, 0);
        assert!(g.orders.iter().any(|(k, _)| *k == OrderKind::Defend));
        assert_eq!(g.stance, Stance::Arm, "threat response must not wait");
        assert_eq!(st.stance_flips, 1);
    }

    #[test]
    fn scripted_goal_paints_expand_attack_defend_and_sets_stance() {
        let mut state = state_with_villages(0, &[3, 5]);
        // Under 3 cities with two capturable villages → two EXPAND orders,
        // sorted, GROW stance, star gate active.
        let g = scripted_goal(&state, 1, 0, None);
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
        let g = scripted_goal(&state, 1, 0, None);
        assert!(g.orders.contains(&(OrderKind::Attack, 40)));
        assert_eq!(g.stance, Stance::Grow);

        // Threatened own city → DEFEND + ARM stance. 040 contract: a single
        // VISIBLE enemy that can reach the unguarded city suffices (the old
        // `near >= 2` proxy is gone, and hidden units never count).
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            ..Default::default()
        });
        let t2 = state.tribes.get_mut(&2).unwrap();
        t2.units.push(unit_at(1));
        t2.units.push(unit_at(12));
        explore_tile(&mut state, 0);
        explore_tile(&mut state, 1);
        explore_tile(&mut state, 12);
        // 040 threat math reads real stats: garrison + attackers need owner
        // and HP (the old proxy counted bare positions).
        {
            let t1 = state.tribes.get_mut(&1).unwrap();
            t1.units[0].owner = 1;
            t1.units[0].health = 10.0;
        }
        for u in state.tribes.get_mut(&2).unwrap().units.iter_mut() {
            u.owner = 2;
            u.health = 10.0;
        }
        let g = scripted_goal(&state, 1, 0, None);
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
        let g = scripted_goal(&state, 1, 0, None);
        assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));

        // A third attacker reaches parity-plus but not the 1.5x margin.
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(30));
        let g = scripted_goal(&state, 1, 0, None);
        assert!(!g.orders.iter().any(|(k, _)| *k == OrderKind::Attack));

        // A fourth clears the margin → ATTACK.
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(20));
        let g = scripted_goal(&state, 1, 0, None);
        assert!(g.orders.contains(&(OrderKind::Attack, 40)));

        // Unexplored enemy city never draws an order.
        state.tiles.get_mut(&40).unwrap().explorers.clear();
        let g = scripted_goal(&state, 1, 0, None);
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
        let g = scripted_goal(&state, 1, 0, None);
        assert_eq!(g.stance, Stance::Grow);

        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        let g = scripted_goal(&state, 1, 0, None);
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
        let g = scripted_goal(&state, 1, 0, None);
        assert!(g.orders.contains(&(OrderKind::Expand, 3)));
        assert_eq!(g.stance, Stance::Grow);
        assert!(!goal_star_gate(&state, 1, &g));
    }

    #[test]
    fn legacy_star_gate_blocks_research_at_any_star_count() {
        // Legacy (stance-less, EXP_ELO_026) arm: every tech is gated, and v9
        // removed the reserve escape — being rich no longer lifts it.
        let mut state = state_with_villages(0, &[3]);
        state.settings.current_player_turn_id = 1;
        let research = ResearchMove::new(TechnologyType::Organization);

        for stars in [0, 5, 50, 500] {
            state.tribes.get_mut(&1).unwrap().stars = stars;
            assert!(!passes_star_gate(&state, &research, None, None));
        }

        // Non-research moves always pass, regardless of stars.
        state.tribes.get_mut(&1).unwrap().stars = 0;
        assert!(passes_star_gate(&state, &EndTurnMove, None, None));
    }

    /// v9: the whole point of the dual-class exemption — Smithery opens the
    /// Forge (giants) and fields a Swordsman, so no economy-or-army stance may
    /// drop it. Same for Mathematics (Sawmill + Catapult).
    #[test]
    fn dual_class_tech_is_never_stance_gated() {
        let mut state = state_with_villages(0, &[3]);
        state.settings.current_player_turn_id = 1;
        state.tribes.get_mut(&1).unwrap().stars = 0;
        for tech in [TechnologyType::Smithery, TechnologyType::Mathematics] {
            let m = ResearchMove::new(tech);
            for stance in [Stance::Grow, Stance::Arm, Stance::Save] {
                assert!(
                    passes_star_gate(&state, &m, Some(stance), None),
                    "{tech:?} gated under {stance:?}"
                );
            }
        }
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

        // GROW gates PURE-combat tech; eco, passage and dual-class flow freely
        // (Climbing carries a defense bonus but fields no unit).
        let grow = Some(Stance::Grow);
        assert!(passes_star_gate(&state, &eco, grow, None));
        assert!(passes_star_gate(&state, &passage, grow, None));
        assert!(!passes_star_gate(&state, &combat, grow, None));
        assert!(passes_star_gate(&state, &mixed, grow, None));

        // ARM flips it: pure-eco tech gated, unit tech (incl. mixed) free.
        let arm = Some(Stance::Arm);
        assert!(!passes_star_gate(&state, &eco, arm, None));
        assert!(passes_star_gate(&state, &combat, arm, None));
        assert!(passes_star_gate(&state, &mixed, arm, None));

        // SAVE is an economy stance and gates the same class GROW does — it
        // must not block the tech chain its own batch is priced to buy.
        let save = Some(Stance::Save);
        assert!(passes_star_gate(&state, &eco, save, None));
        assert!(passes_star_gate(&state, &mixed, save, None));
        assert!(!passes_star_gate(&state, &combat, save, None));

        // v9: no reserve — being rich no longer lifts a gated class.
        state.tribes.get_mut(&1).unwrap().stars = 500;
        assert!(!passes_star_gate(&state, &combat, grow, None));

        // UNLOCK gates nothing (no unlock policy yet).
        state.tribes.get_mut(&1).unwrap().stars = 0;
        assert!(passes_star_gate(&state, &combat, Some(Stance::Unlock), None));

        // v6: an active knight commitment makes its lane stance-coherent —
        // FreeSpirit passes under ARM and Chivalry under GROW, even broke;
        // without the commit both stay gated by their stance class.
        let free_spirit = ResearchMove::new(TechnologyType::FreeSpirit);
        let chivalry = ResearchMove::new(TechnologyType::Chivalry);
        let mut committed = GoalAux::default();
        committed.overlays.knight_commit = true;
        // Aug 14: the ARM eco-mask is intensity-conditional — it fires only
        // at near-certain pressure (arm_strength >= 0.98). A covered
        // skirmish (low strength) must NOT lock the eco lanes.
        let mut uncommitted = GoalAux::default();
        uncommitted.arm_strength = 1.0;
        assert!(passes_star_gate(&state, &chivalry, grow, Some(&committed)));
        assert!(passes_star_gate(&state, &free_spirit, arm, Some(&committed)));
        assert!(!passes_star_gate(&state, &chivalry, grow, Some(&uncommitted)));
        assert!(!passes_star_gate(&state, &free_spirit, arm, Some(&uncommitted)));
        let mut covered = GoalAux::default();
        covered.arm_strength = 0.3;
        assert!(
            passes_star_gate(&state, &free_spirit, arm, Some(&covered)),
            "low-intensity ARM must not mask eco tech"
        );
    }

    #[test]
    fn market_ready_needs_three_cities_and_a_hub() {
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        for i in 0..3 {
            t1.cities.push(crate::states::CityState { idx: i, owner: 1, ..Default::default() });
        }
        state.tribes.insert(1, t1);
        // Three cities but no hub yet.
        assert!(!market_ready(&state, 1));
        // A windmill on own territory opens the lane.
        state.structures.insert(
            40,
            Some(StructureState {
                structure_type: StructureType::Windmill,
                level: 0,
                founded: 0,
            }),
        );
        state.tiles.entry(40).or_insert_with(TileState::default).owner = 1;
        assert!(market_ready(&state, 1));
        // Two cities: not ready even with the hub.
        state.tribes.get_mut(&1).unwrap().cities.pop();
        assert!(!market_ready(&state, 1));
    }

    #[test]
    fn tier3_cap_exempts_chivalry_under_knight_commit() {
        let chivalry = ResearchMove::new(TechnologyType::Chivalry);
        let math = ResearchMove::new(TechnologyType::Mathematics);
        let mut aux = GoalAux::default();
        aux.tier3_bought = TIER3_CAP_PER_GAME;
        aux.overlays.knight_commit = true;
        aux.eco_tier3_owned = true; // v7: economy first, then the combat lane
        // Cap spent: Chivalry still passes under the commit; other tier-3s
        // stay blocked; without the commit Chivalry is blocked too (by the
        // stepping-stone rule AND the cap).
        assert!(passes_tech_caps(&chivalry, &aux));
        assert!(!passes_tech_caps(&math, &aux));
        aux.overlays.knight_commit = false;
        assert!(!passes_tech_caps(&chivalry, &aux));
    }

    /// v7 (Verdi): players almost never take knights before the level-3 pop
    /// buildings, because those are what lead to giants. A combat tier-3 waits
    /// for an economic one — and OWNERSHIP is the predicate, so a free
    /// economy tier-3 out of a ruin unblocks it immediately.
    #[test]
    fn combat_tier3_waits_for_an_economic_tier3() {
        let chivalry = ResearchMove::new(TechnologyType::Chivalry);
        let construction = ResearchMove::new(TechnologyType::Construction);
        let mut aux = GoalAux::default();
        aux.overlays.knight_commit = true; // clears the stepping-stone rule
        aux.tier3_bought = 0; // budget available

        assert!(
            !passes_tech_caps(&chivalry, &aux),
            "combat tier-3 blocked while no economic tier-3 is owned"
        );
        assert!(
            passes_tech_caps(&construction, &aux),
            "the economic tier-3 itself is never blocked by the ordering rule"
        );
        aux.eco_tier3_owned = true;
        assert!(passes_tech_caps(&chivalry, &aux), "economy first, then knights");

        // Two slots now, so economy + combat both fit in one game.
        assert_eq!(TIER3_CAP_PER_GAME, 2);
        aux.tier3_bought = 1;
        assert!(passes_tech_caps(&chivalry, &aux));
        aux.tier3_bought = 2;
        assert!(!passes_tech_caps(&construction, &aux), "cap still binds at 2");
    }

    /// The economic/combat split must come from the settings tables, not a
    /// hand list — the exact discipline `max_affordable_pop` failed at.
    #[test]
    fn eco_tier3_classification_is_table_derived() {
        use crate::settings::technology::is_eco_tier3;
        for t in [
            TechnologyType::Construction,
            TechnologyType::Mathematics,
            TechnologyType::Smithery,
            TechnologyType::Trade,
            TechnologyType::Philosophy,
        ] {
            assert!(is_eco_tier3(t), "{t:?} unlocks a yielding structure");
        }
        for t in [TechnologyType::Chivalry, TechnologyType::Navigation] {
            assert!(!is_eco_tier3(t), "{t:?} unlocks no yielding structure");
        }
        // Not tier 3 at all.
        assert!(!is_eco_tier3(TechnologyType::Farming));
    }

    /// On a dry map the naval lane unlocks nothing, so it is masked at the
    /// root — but only the lane that dead-ends, and only when the map is dry.
    #[test]
    fn water_techs_are_masked_only_on_a_map_without_water() {
        let mut aux = GoalAux::default();
        aux.overlays.knight_commit = true;
        aux.eco_tier3_owned = true;

        let wet = [
            TechnologyType::Fishing,
            TechnologyType::Sailing,
            TechnologyType::Ramming,
            TechnologyType::Aquatism,
            TechnologyType::Navigation,
        ];
        aux.water_dead = false;
        for t in wet {
            assert!(passes_tech_caps(&ResearchMove::new(t), &aux), "{t:?} legal with water");
        }
        aux.water_dead = true;
        for t in wet {
            assert!(!passes_tech_caps(&ResearchMove::new(t), &aux), "{t:?} dead without water");
        }
        // Land techs are untouched, and a non-Research move never sees the gate.
        for t in [TechnologyType::Construction, TechnologyType::Chivalry, TechnologyType::Riding] {
            assert!(passes_tech_caps(&ResearchMove::new(t), &aux), "{t:?} unaffected");
        }
    }

    /// Aquatism yields population, so the table calls it an economic tier-3 —
    /// but a WaterTemple can never be built on a dry map, and letting it pass
    /// the economy-first rule would hand the combat lane a free unlock.
    #[test]
    fn a_water_tier3_does_not_satisfy_the_economy_first_rule_when_dry() {
        use crate::states::TechnologyState;
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Aquatism,
            discovered: true,
            discovered_turn: 3,
        });
        state.tribes.insert(1, t1);
        // No tiles at all -> no water.
        let goal = MacroGoal::default();
        let dry = scripted_goal_aux(&state, 1, &goal, 0, 0, None);
        assert!(dry.water_dead);
        assert!(!dry.eco_tier3_owned, "a dead water temple is not an economy");

        // Same tech, same seat, on a map that has water: it counts.
        let mut wet_tile = TileState::default();
        wet_tile.terrain_type = TerrainType::Water;
        state.tiles.insert(0, wet_tile);
        let wet = scripted_goal_aux(&state, 1, &goal, 0, 0, None);
        assert!(!wet.water_dead);
        assert!(wet.eco_tier3_owned);
    }

    /// A lane's price is the whole path to it, not just the last tech.
    #[test]
    fn tech_chain_cost_prices_undiscovered_prerequisites() {
        use crate::settings::technology::get_technology_setting;
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState::default());
        let direct = crate::functions::get_tech_cost(&t1, TechnologyType::Trade);
        let chain = tech_chain_cost(&t1, TechnologyType::Trade);
        assert!(
            chain > direct,
            "Trade sits behind Roads behind Riding — the chain must cost more \
             than the tech alone ({chain} vs {direct})"
        );
        // Owning the prerequisite removes its cost from the chain.
        let req = get_technology_setting(TechnologyType::Trade).requires.unwrap();
        t1.tech_vanilla.push(crate::states::TechnologyState {
            tech_type: req,
            discovered: true,
            discovered_turn: 0,
        });
        assert_eq!(tech_chain_cost(&t1, TechnologyType::Trade), direct);
    }

    /// A lane the tier-3 cap will refuse is not a plan — it is a hoard with no
    /// exit. v7 shipped priced-but-unbuyable lanes; this pins the fix.
    #[test]
    fn save_batch_skips_lanes_the_tier3_cap_will_refuse() {
        use crate::types::{StructureType, TechnologyType, TerrainType};
        let mut state = state_with_villages(0, &[3, 5]);
        {
            let t1 = state.tribes.get_mut(&1).unwrap();
            t1.stars = 40;
            for tech in [TechnologyType::Organization, TechnologyType::Farming] {
                t1.tech_vanilla.push(crate::states::TechnologyState {
                    tech_type: tech,
                    discovered: true,
                    discovered_turn: 0,
                });
            }
            t1.cities.push(crate::states::CityState {
                idx: 60,
                owner: 1,
                _territory: vec![60, 61, 50, 72],
                production: 2,
                ..Default::default()
            });
        }
        for idx in [50, 72] {
            state.structures.insert(
                idx,
                Some(crate::states::StructureState {
                    structure_type: StructureType::Farm,
                    ..Default::default()
                }),
            );
        }
        for idx in [60, 61, 50, 72] {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.owner = 1;
            tile.terrain_type = TerrainType::Field;
        }
        // Construction unowned: the lane is priced with its full chain.
        let with_budget = save_batch_plan(&state, 1, 0, None).expect("lane priced").cost;
        assert!(with_budget > 5, "chain cost must be included, got {with_budget}");
        // Tier-3 budget spent: the same lane is unreachable and must vanish.
        assert!(save_batch_plan(&state, 1, TIER3_CAP_PER_GAME, None).is_none());
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
        let goal = scripted_goal(&state, 1, 0, None);
        assert!(scripted_goal_aux(&state, 1, &goal, 0, 0, None).rider_push);
        assert!(rider_turns_saved(&state, 1, &[44]) >= 2);

        // A thin band is NOT enough: a rider weaves open-step + forest-step
        // (2 tiles/turn, real rider mechanics) and still saves a turn.
        for r in 1..4 {
            for c in 0..3 {
                state.tiles.insert(r * 11 + c, terrain_tile(TerrainType::Forest));
            }
        }
        let goal = scripted_goal(&state, 1, 0, None);
        assert!(scripted_goal_aux(&state, 1, &goal, 0, 0, None).rider_push);

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
        let g = scripted_goal(&state, 1, 0, None);
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
        let goal = scripted_goal(&state, 1, 0, None); // EXPAND on village 3
        let aux = scripted_goal_aux(&state, 1, &goal, 0, 0, None);
        assert!(aux.rider_push);
        assert_eq!(aux.recommended_techs.first(), Some(&TechnologyType::Riding));

        // Without an EXPAND order there is no rider push.
        let quiet = MacroGoal::default();
        assert!(!scripted_goal_aux(&state, 1, &quiet, 0, 0, None).rider_push);

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

    /// Explored open fields at 22..42 for archetype map reads.
    fn explore_open_fields(state: &mut GameState) {
        for idx in 22..42 {
            let mut tile = TileState::default();
            tile.terrain_type = crate::types::TerrainType::Field;
            tile.explorers.insert(1);
            state.tiles.insert(idx, tile);
        }
    }

    #[test]
    fn archetype_rider_enters_on_open_map_and_hard_exits_on_heavy() {
        let mut state = state_with_villages(0, &[24]);
        state.settings.current_player_turn_id = 1;
        explore_open_fields(&mut state);
        let goal = scripted_goal(&state, 1, 0, None);
        let mut st = ArchetypeState::default();
        update_archetype(&state, 1, &goal, &mut st);
        assert_eq!(st.archetype, Some(Archetype::RiderRoads));

        // Lane expression: Riding recommended, Rider preferred; FreeSpirit
        // stays blocked without a knight commitment.
        let aux = scripted_goal_aux(&state, 1, &goal, 0, 0, Some(&st));
        assert!(aux.recommended_techs.contains(&TechnologyType::Riding));
        assert!(aux.preferred_units.contains(&UnitType::Rider));
        assert!(!passes_tech_caps(&ResearchMove::new(TechnologyType::FreeSpirit), &aux));

        // Two giants observed → catapult overlay fires and riders hard-exit.
        let mut t2 = TribeState::default();
        for &i in &[23, 25] {
            let mut g = unit_at(i);
            g.unit_type = UnitType::Giant;
            t2.units.push(g);
        }
        state.tribes.insert(2, t2);
        let goal2 = scripted_goal(&state, 1, 0, None);
        // Tier 1: discretionary switches wait for the turn boundary, but
        // REFUTATION does not — the sighting that flips an overlay is what
        // zeroes the lane's score, so it re-selects immediately (same
        // precedent as the stance layer's urgent-threat bypass).
        update_archetype(&state, 1, &goal2, &mut st);
        assert!(st.overlays.catapult_counter);
        assert_eq!(st.archetype, Some(Archetype::ArcherLine), "refutation is immediate");
        assert_eq!(st.pivots_used, 1, "a refuted lane costs budget");
        let aux2 = scripted_goal_aux(&state, 1, &goal2, 0, 0, Some(&st));
        assert!(aux2.preferred_units.contains(&UnitType::Catapult));
        assert!(aux2.preferred_units.contains(&UnitType::Archer));
    }

    /// Tier 1: the tribe's birth tech commits a lane before any terrain is
    /// explored, and the mapping is derived from `lane_techs` (not a second
    /// table that can drift from mapgen). `select_playstyle` only falls back
    /// to the prior when the census has nothing to say at all — a visible
    /// village alone already gives the rider lane a real (non-terrain)
    /// mobility signal, so this isolates the true information vacuum with
    /// no villages and no cities.
    #[test]
    fn tribe_prior_commits_a_lane_before_the_map_speaks() {
        for (tribe, tech, lane) in [
            (TribeType::Oumaji, TechnologyType::Riding, Archetype::RiderRoads),
            (TribeType::Hoodrick, TechnologyType::Archery, Archetype::ArcherLine),
            (TribeType::XinXi, TechnologyType::Climbing, Archetype::ForgeGiants),
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
            // A bare goal with no painted orders isolates the prior-fallback
            // mechanism from `scripted_goal`'s own guessed-village Expand
            // orders, which (via rider mobility) carry a real, non-terrain
            // signal of their own and are essentially always present once
            // `guessed_village_sites` kicks in below `COMMIT_CITY_TARGET`
            // cities — even the census "has nothing to say" fixture is not
            // actually reachable through the production goal-setter.
            let goal = MacroGoal::default();
            let mut st = ArchetypeState::default();
            assert_eq!(select_playstyle(&state, 1, &goal, &mut st, None), Some(lane));
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
        let goal = scripted_goal(&state, 1, 0, None);
        let mut st = ArchetypeState::default();
        select_playstyle(&state, 1, &goal, &mut st, None);
        let first = st.archetype.expect("a lane is committed on an explored map");
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
            let g = scripted_goal(&state, 1, 0, None);
            select_playstyle(&state, 1, &g, &mut st, None);
        }
        assert!(st.pivots_used <= MAX_PIVOTS, "budget must cap at {MAX_PIVOTS}");
        let frozen = st.archetype;
        st.pivots_used = MAX_PIVOTS;
        for turn in 13..=20 {
            state.settings.turn = turn;
            let g = scripted_goal(&state, 1, 0, None);
            select_playstyle(&state, 1, &g, &mut st, None);
        }
        assert_eq!(st.archetype, frozen, "no lane change once the budget is spent");
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
        let goal = scripted_goal(&state, 1, 0, None);
        let mut st = ArchetypeState::default();
        select_playstyle(&state, 1, &goal, &mut st, None);
        st.pivots_used = MAX_PIVOTS;
        let committed = st.archetype;

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
            let g = scripted_goal(&state, 1, 0, None);
            update_archetype(&state, 1, &g, &mut st);
        }
        assert_eq!(st.archetype, committed, "the cap binds even under refutation");
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
        let goal = scripted_goal(&state, 1, 0, None);

        let mut algo_only = ArchetypeState::default();
        select_playstyle(&state, 1, &goal, &mut algo_only, None);
        let census_pick = algo_only.archetype.unwrap();

        let idx = LANE_ORDER.iter().position(|a| *a == census_pick).unwrap();
        let mut head = [0.0f32; LANES];
        head[(idx + 1) % LANES] = 50.0; // overwhelming opinion for another lane
        let mut with_head = ArchetypeState::default();
        let pick = select_playstyle(&state, 1, &goal, &mut with_head, Some(&head)).unwrap();
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
        let goal = scripted_goal(&state, 1, 0, None);
        let mut st = ArchetypeState::default();
        update_archetype(&state, 1, &goal, &mut st);
        assert!(st.overlays.knight_commit);
        assert!(st.overlays.defender_screen);
        let aux = scripted_goal_aux(&state, 1, &goal, 0, 0, Some(&st));
        assert!(passes_tech_caps(&ResearchMove::new(TechnologyType::FreeSpirit), &aux));
        assert!(aux.preferred_units.contains(&UnitType::Knight));
        assert!(aux.preferred_units.contains(&UnitType::Defender));
    }

    #[test]
    fn expand_assignment_is_unique_and_nearest_first() {
        let mut state = state_with_villages(0, &[4, 44]);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(40));
        // Two units, two targets: greedy nearest-pair-first must cover both
        // targets with distinct units — never two scouts on one target.
        let pairs = assign_expand_targets(&state, 1, &[4, 44]);
        assert_eq!(pairs.len(), 2);
        let units: std::collections::HashSet<i32> = pairs.iter().map(|(u, _)| *u).collect();
        let targets: std::collections::HashSet<i32> = pairs.iter().map(|(_, t)| *t).collect();
        assert_eq!(units.len(), 2, "each unit assigned at most once");
        assert_eq!(targets.len(), 2, "each target assigned at most once");
    }

    #[test]
    fn guessed_sites_spread_across_quadrants() {
        // Anchor in the center; legal spots exist in multiple quadrants.
        let state = state_with_villages(60, &[]);
        let picks = guessed_village_sites(&state, 1, 2);
        assert_eq!(picks.len(), 2);
        let size = 11;
        let q = |idx: i32| ((idx % size > 5) as u8) * 2 + ((idx / size > 5) as u8);
        assert_ne!(q(picks[0]), q(picks[1]), "guesses should span distinct quadrants");
    }

    #[test]
    fn ability_gate_blocks_destroy_and_resource_clearing() {
        use crate::moves::abilities::{BurnForestMove, ClearForestMove, DestroyMove};
        let mut state = GameState::default();
        assert!(!passes_ability_gate(&state, &DestroyMove::new(5)));
        assert!(passes_ability_gate(&state, &EndTurnMove));
        assert!(passes_ability_gate(&state, &ResearchMove::new(TechnologyType::Organization)));
        // Bare forest may still be cleared — that trade is priced, not masked.
        assert!(passes_ability_gate(&state, &ClearForestMove::new(5)));
        // v8: a forest carrying a resource may not be — clearing DELETES the
        // Game sitting on it for one star.
        state.resources.insert(
            5,
            Some(crate::states::ResourceState { resource_type: crate::types::ResourceType::Game }),
        );
        assert!(!passes_ability_gate(&state, &ClearForestMove::new(5)));
        assert!(!passes_ability_gate(&state, &BurnForestMove::new(5)));
        assert!(passes_ability_gate(&state, &ClearForestMove::new(6)));
    }


    #[test]
    fn goal_star_gate_is_stance_aware() {
        let mut state = state_with_villages(0, &[3]);
        // ARM gates regardless of expansion state.
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
        assert!(goal_star_gate(&state, 1, &arm));
        // GROW gates only inside the expansion window.
        let grow = MacroGoal {
            orders: vec![(OrderKind::Expand, 3)],
            stance: Stance::Grow,
            save_target: None,
        };
        assert!(goal_star_gate(&state, 1, &grow));
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        assert!(!goal_star_gate(&state, 1, &grow));
    }
}
