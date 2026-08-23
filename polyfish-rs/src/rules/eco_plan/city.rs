//! Per-city, per-tile planning: what each tile could become, what a hub
//! site is worth, and one city's best build-out under a scenario. The
//! biggest piece of the planner — everything here is independent once
//! hub sites are fixed, which is what makes the empire module's joint
//! search tractable.

use super::*;
use super::tech::*;
use crate::functions::{get_adjacent_indices, get_chebyshev_distance, MARKET_MAX_LEVEL};
use crate::rules::economy::{level_at_pop, super_units_at_level as giants_at_level};
use crate::settings::resources::get_resource_setting;
use crate::settings::structures::get_structure_setting;
use crate::states::GameState;
use crate::types::*;
use std::collections::{HashMap, HashSet};

/// Every tile a city could develop, after any terrain conversion the scenario
/// allows. Returns (buys, hub_sites, conversion_cost_by_tile).
pub fn tile_options(
    state: &GameState,
    territory: &[i32],
    sc: Scenario,
) -> (Vec<Buy>, Vec<i32>, HashMap<i32, i32>) {
    let mut buys = Vec::new();
    let mut hub_sites = Vec::new();
    let mut convert_cost = HashMap::new();

    let burn = crate::version_sync::get_burn_forest_cost(state);
    const GROW_FOREST_COST: i32 = 5;

    for &idx in territory {
        let Some(tile) = state.tiles.get(&idx) else { continue };
        if crate::functions::get_structure_at(state, idx).is_some() {
            continue; // already built (city centre, village, ruin)
        }
        let res = state
            .resources
            .get(&idx)
            .and_then(|r| r.as_ref())
            .map(|r| r.resource_type);
        let terrain = tile.terrain_type;
        let harvest = |r: ResourceType| -> Option<Buy> {
            let rs = get_resource_setting(r);
            (rs.reward_pop > 0 && rs.struct_required.is_none()).then(|| Buy {
                idx,
                what: "harvest",
                cost: rs.cost.unwrap_or(0),
                pop: rs.reward_pop,
                occupies: false,
                techs: vec![rs.tech_required],
            })
        };

        match terrain {
            TerrainType::Field => {
                hub_sites.push(idx);
                match res {
                    Some(ResourceType::Crop) => buys.push(Buy {
                        idx,
                        what: "Farm",
                        cost: get_structure_setting(StructureType::Farm).cost.unwrap_or(5),
                        pop: get_resource_setting(ResourceType::Crop).reward_pop,
                        occupies: true,
                        techs: structure_techs(StructureType::Farm),
                    }),
                    Some(r) => buys.extend(harvest(r)),
                    None => {}
                }
                if sc.lane == Lane::Forest && sc.convert {
                    convert_cost.insert(idx, GROW_FOREST_COST);
                    let mut techs = structure_techs(StructureType::LumberHut);
                    // GrowForest (Field -> Forest) is its own ability-gated
                    // tech (Spiritualism), separate from the tech that
                    // unlocks the LumberHut structure itself -- missing
                    // this silently undercharged every convert scenario by
                    // Spiritualism's full price. Same SSOT lookup lane_chain
                    // uses, so this can't drift from it again.
                    techs.extend(ability_techs(AbilityType::GrowForest));
                    buys.push(Buy {
                        idx,
                        // Named for the structure it ends up placing: the hub's
                        // partner match is on the suffix, and "grow+Hut" missed
                        // it, so grown huts fed no Sawmill.
                        what: "grow+LumberHut",
                        cost: GROW_FOREST_COST + 3,
                        pop: 1,
                        occupies: true,
                        techs,
                    });
                }
            }
            TerrainType::Forest => {
                // Forestry unlocks ClearForest: Forest -> Field, and it PAYS a
                // star. So any forest is a candidate hub site for the forest
                // lane -- which is how a Sawmill ends up in the middle of a
                // forest cluster rather than on its edge.
                if matches!(sc.lane, Lane::Forest | Lane::Mine) {
                    hub_sites.push(idx);
                }
                // Hunting the Game leaves the forest standing, so the tile can
                // pay twice: harvest now, LumberHut after.
                if let Some(r) = res {
                    buys.extend(harvest(r));
                }
                if sc.lane == Lane::Forest || !sc.convert {
                    buys.push(Buy {
                        idx,
                        what: "LumberHut",
                        cost: 3,
                        pop: 1,
                        occupies: true,
                        techs: structure_techs(StructureType::LumberHut),
                    });
                } else {
                    convert_cost.insert(idx, burn);
                    buys.push(Buy {
                        idx,
                        what: "burn+Farm",
                        cost: burn + get_structure_setting(StructureType::Farm).cost.unwrap_or(5),
                        pop: get_resource_setting(ResourceType::Crop).reward_pop,
                        occupies: true,
                        techs: structure_techs(StructureType::Farm),
                    });
                }
            }
            TerrainType::Mountain => {
                if res == Some(ResourceType::Metal) {
                    buys.push(Buy {
                        idx,
                        what: "Mine",
                        cost: get_structure_setting(StructureType::Mine).cost.unwrap_or(5),
                        pop: get_resource_setting(ResourceType::Metal).reward_pop,
                        occupies: true,
                        techs: structure_techs(StructureType::Mine),
                    });
                }
            }
            TerrainType::Water | TerrainType::Ocean => {
                if let Some(r) = res {
                    buys.extend(harvest(r));
                }
            }
            _ => {}
        }
    }
    (buys, hub_sites, convert_cost)
}

pub struct CityPlan {
    pub scenario: &'static str,
    pub territory: usize,
    pub max_pop: i32,
    pub stars: i32,
    pub level: i32,
    pub giants: i32,
    pub spt: i32,
    pub cost_per_giant: f64,
    pub hub_site: Option<i32>,
    pub hub_level: i32,
    pub market_site: Option<i32>,
    pub feasible: bool,
}

/// What a city can build out to on a given tile set, under one scenario.
/// Shared by the real plan and the border-growth reachability check so the two
/// can never disagree about how much pop a tile set yields.
pub struct BuildOut {
    pub pop: i32,
    pub stars: i32,
    /// Techs the taken buys require — billed by `plan_city`, so a plan pays for
    /// the Mining or Fishing it leans on instead of assuming it for free.
    pub techs: HashSet<TechnologyType>,
    pub hub_site: Option<i32>,
    pub partners: i32,
    pub market_site: Option<i32>,
    pub market_spt: i32,
    pub monuments: i32,
}

/// Monuments are free and pay 3 pop, but occupy a tile. Placing one costs
/// whatever that tile would have produced — 0 on a bare Field, 1 on a Forest
/// that would have been a LumberHut, 2 more if it was feeding the hub. Returns
/// (tiles used, net pop gained).
pub fn place_monuments(
    state: &GameState,
    territory: &[i32],
    buys: &[Buy],
    partner_tiles: &HashSet<i32>,
    adj_to_hub: &[i32],
    hub_site: Option<i32>,
    market_site: Option<i32>,
    budget: i32,
) -> (HashSet<i32>, i32) {
    const MONUMENT_POP: i32 = 3;
    // `take(0)` already made the loop below a no-op at zero budget, but the
    // scan, the two maps and the sort all ran first — on every priced plan, and
    // no monuments is the default.
    if budget <= 0 {
        return (HashSet::new(), 0);
    }
    let occupied: HashSet<i32> = buys.iter().filter(|b| b.occupies).map(|b| b.idx).collect();
    let pop_at: HashMap<i32, i32> = buys.iter().map(|b| (b.idx, b.pop)).collect();

    let mut cands: Vec<(i32, i32)> = territory
        .iter()
        .copied()
        .filter(|i| Some(*i) != hub_site && Some(*i) != market_site)
        .filter(|i| {
            state.tiles.get(i).is_some_and(|t| {
                matches!(
                    t.terrain_type,
                    TerrainType::Field | TerrainType::Forest | TerrainType::Water
                )
            }) && crate::functions::get_structure_at(state, *i).is_none()
        })
        .map(|i| {
            // Pop forgone: the tile's own yield, plus the hub partner it was.
            let mut loss = if occupied.contains(&i) { *pop_at.get(&i).unwrap_or(&0) } else { 0 };
            if partner_tiles.contains(&i) && adj_to_hub.contains(&i) {
                loss += 1;
            }
            (loss, i)
        })
        .collect();
    cands.sort();

    let mut used = HashSet::new();
    let mut gained = 0;
    for (loss, idx) in cands.into_iter().take(budget.max(0) as usize) {
        if MONUMENT_POP - loss <= 0 {
            break; // never worth displacing more than it pays
        }
        used.insert(idx);
        gained += MONUMENT_POP - loss;
    }
    (used, gained)
}

/// Could this lane's hub ever stand here? A hub needs two partner tiles
/// beside one tile, which most maps never offer. Scoring the lane anyway
/// costs a full allocation pass per city and returns the same hub-less
/// build every time, so the lane is gated on the terrain rather than
/// evaluated and discarded. Natural placement only -- Forest's `convert`
/// scenario (grow a forest first) is handled downstream by the exhaustive
/// per-scenario search, not this cheap pre-check.
pub fn lane_can_place_hub(state: &GameState, territory: &[i32], lane: Lane) -> bool {
    let partner_tiles: HashSet<i32> = territory
        .iter()
        .copied()
        .filter(|&i| match lane {
            Lane::Mine => state
                .resources
                .get(&i)
                .and_then(|r| r.as_ref())
                .is_some_and(|r| r.resource_type == ResourceType::Metal),
            Lane::Farm => state
                .resources
                .get(&i)
                .and_then(|r| r.as_ref())
                .is_some_and(|r| r.resource_type == ResourceType::Crop),
            Lane::Forest => state
                .tiles
                .get(&i)
                .is_some_and(|t| t.terrain_type == TerrainType::Forest),
        })
        .collect();
    territory.iter().any(|&t| {
        get_adjacent_indices(state, t, 1)
            .into_iter()
            .filter(|a| partner_tiles.contains(a))
            .count()
            >= 2
    })
}

/// The radius-2 square a city could ever rule, for cheap pre-checks that must
/// not depend on an allocation that has not happened yet.
pub fn city_square(state: &GameState, city: i32) -> Vec<i32> {
    get_adjacent_indices(state, city, 2)
        .into_iter()
        .chain([city])
        .filter(|i| state.tiles.contains_key(i))
        .collect()
}

/// Fewest monuments this city must be GIVEN before its border is reachable at
/// all, or None if nothing within `budget` does it. Monuments are the only
/// lever here that is not bought with stars, so a plan that needs one is
/// spending a scarce earned resource rather than paying a price.
pub fn monuments_to_reach(
    state: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    hub: Option<i32>,
    budget: i32,
) -> Option<i32> {
    if !sc.border_growth {
        return Some(0);
    }
    (0..=budget).find(|&m| match hub {
        Some(h) => site_reachable(state, city_idx, territory, sc, m, h),
        None => {
            let inner = inner_ring(state, city_idx, territory);
            city_build(state, &inner, sc, m, None, None, None).pop >= POP_FOR_LEVEL_4
        }
    })
}

/// Hub sites in a territory, best first by buildable partner count. A site
/// paying only 1 partner is never worth 5* (worse than the partner feeding it),
/// and a 0-partner hub is not even legal (`build.rs` requires an adjacent
/// partner), so both are excluded.
pub fn hub_candidates(state: &GameState, territory: &[i32], sc: Scenario, top_k: usize) -> Vec<i32> {
    let plot = Plot::new(state, territory, sc);
    hub_candidates_on(state, &plot, top_k, &plot.partner_tiles)
}

/// Hub sites for one city, best first, capped at `top_k`.
///
/// Core form: the caller supplies the partner set, the way
/// `rules::economy::partner_count_with` does. The joint frontier scores against
/// the EMPIRE's partners because it knows every city's lane and a hub on a
/// border collects across it; a single city can only see its own ground. That
/// is the one thing the two callers are allowed to disagree about — the site
/// space and the floor are shared, and used not to be: `enumerate_empire` kept
/// its own copy that demanded two partners and refused Forest tiles to the
/// Forge lane, so the frontier never saw sites the per-city ranker priced.
pub fn hub_candidates_on(
    state: &GameState,
    plot: &Plot,
    top_k: usize,
    partners: &HashSet<i32>,
) -> Vec<i32> {
    let mut scored: Vec<(i32, i32)> = plot
        .hub_sites
        .iter()
        .map(|&s| {
            let n = get_adjacent_indices(state, s, 1)
                .into_iter()
                .filter(|a| partners.contains(a))
                .count() as i32;
            (-n, s)
        })
        // >= 1, the engine's own legality floor: `moves/build.rs` requires one
        // friendly partner to build a hub at all. This used to demand >= 2 as a
        // proxy for a good site, which was defensible while ranking was on pop
        // -- but a single-partner tile can win on SPT by being cheap or by
        // enabling a better Market, and the verify goal-sweep caught exactly
        // that: --goal army picking a dominated site because the site that
        // dominated it had been filtered out before ranking ever saw it.
        .filter(|&(negn, _)| -negn >= 1)
        .collect();
    scored.sort();
    scored.into_iter().take(top_k).map(|(_, s)| s).collect()
}

/// One city's build-out with the hub and market sites SUPPLIED. Market income
/// is not computed here — it depends on every city's hub, so the caller does it.
/// A (territory, scenario) pair with its tile scan already done.
///
/// `city_build` used to run `tile_options` — a full territory scan — on every
/// call, and the joint frontier calls it about `2n` times per hub combination,
/// of which there are `(top_k+1)^n`. Scanning once per (city, scenario) is what
/// makes the frontier's cost track the number of PLANS rather than the number
/// of tiles times the number of plans.
///
/// Build a Plot against the same state it will be priced on: `materialize`
/// works on an owned board whose tiles differ from the base.
pub struct Plot<'a> {
    pub territory: &'a [i32],
    pub buys: Vec<Buy>,
    pub hub_sites: Vec<i32>,
    pub standing_partners: Vec<i32>,
    /// Tiles this lane will hold a partner on: the partner buys plus whatever
    /// already stands. Independent of where the hub goes, so it is built once
    /// instead of once per hub combination.
    pub partner_tiles: HashSet<i32>,
}

impl<'a> Plot<'a> {
    pub fn new(state: &GameState, territory: &'a [i32], sc: Scenario) -> Self {
        let (buys, hub_sites, _conv) = tile_options(state, territory, sc);
        let standing_partners = standing(state, territory, lane_partner_type(sc.lane));
        let (_, partner_name) = lane_hub(sc.lane);
        let partner_tiles: HashSet<i32> = buys
            .iter()
            .filter(|b| is_partner_buy(b, partner_name))
            .map(|b| b.idx)
            .chain(standing_partners.iter().copied())
            .collect();
        Plot { territory, buys, hub_sites, standing_partners, partner_tiles }
    }
}

pub fn city_build(
    state: &GameState,
    territory: &[i32],
    sc: Scenario,
    monuments: i32,
    hub: Option<i32>,
    market: Option<i32>,
    empire_partners: Option<&HashSet<i32>>,
) -> BuildOut {
    city_build_on(
        state,
        &Plot::new(state, territory, sc),
        sc,
        monuments,
        hub,
        market,
        empire_partners,
        &[],
    )
}

pub fn city_build_on(
    state: &GameState,
    plot: &Plot,
    sc: Scenario,
    monuments: i32,
    hub: Option<i32>,
    market: Option<i32>,
    // Partner tiles across the WHOLE empire. Adjacency pay is player-scoped
    // (`build_structure` tests `t.owner == pov_id`), so a hub sited near a
    // border collects partners on the far side too. Counting only this city's
    // tiles undervalues exactly the border hubs that feed a shared Market.
    empire_partners: Option<&HashSet<i32>>,
    // Tiles other hubs stand on. A tile holds ONE structure, so a hub sited next
    // to a neighbour's hub must not count it as a partner — the empire partner
    // set offers every city's candidate tiles and cannot know which were spent
    // on hubs in this combination.
    taken: &[i32],
) -> BuildOut {
    let territory = plot.territory;
    let buys = &plot.buys;
    let (hub_type, _) = lane_hub(sc.lane);
    let hub_cost = get_structure_setting(hub_type).cost.unwrap_or(5);
    let market_cost = get_structure_setting(StructureType::Market).cost.unwrap_or(5);

    let partner_tiles = &plot.partner_tiles;
    // Neighbours of the hub, in a fixed buffer. This used to be a HashSet built
    // per call, and the frontier makes millions of these calls.
    let adj_to_hub: Vec<i32> = hub
        .map(|s| get_adjacent_indices(state, s, 1))
        .unwrap_or_default();

    // A structure-placing buy is displaced by a hub or market on its tile;
    // harvests are not, since they leave the tile empty.
    let displaced = |b: &Buy| b.occupies && (Some(b.idx) == hub || Some(b.idx) == market);
    let mut pop = 0;
    let mut stars = 0;
    let mut techs: HashSet<TechnologyType> = HashSet::new();
    for b in buys.iter().filter(|b| !displaced(b)) {
        stars += b.cost;
        pop += b.pop;
        techs.extend(b.techs.iter().copied());
    }
    // Walk the hub's <=8 neighbours and test each, rather than walking the
    // partner set and testing adjacency: the neighbours are already unique, so
    // no set is needed to dedupe and none is allocated.
    // A neighbour feeds the hub if the plan will put a partner there, or if one
    // already stands there on ground the PLAYER owns. The second half matters
    // because adjacency pay is player-scoped (`rules::economy::partner_count`)
    // while `partner_tiles` only ever saw this city's ground, so every per-city
    // consumer underpriced exactly the border sites `enumerate_empire` prices
    // correctly — claimed 1 where the engine paid 4 (Aug 2026).
    let counted = empire_partners.unwrap_or(partner_tiles);
    let pov = pov_of(state);
    let feeds = &get_structure_setting(hub_type).adjacent_types;
    let partners = adj_to_hub
        .iter()
        .filter(|&&t| {
            (counted.contains(&t)
                && Some(t) != hub
                && Some(t) != market
                && !taken.contains(&t))
                || (state.tiles.get(&t).is_some_and(|x| x.owner == pov)
                    && crate::functions::get_structure_at(state, t)
                        .is_some_and(|st| feeds.contains(&st.structure_type)))
        })
        .count() as i32;
    pop += partners;

    // You do not pay for what is already standing. On a generated map nothing
    // is, and this reduces to the old unconditional charge; on a live state it
    // is what makes the plan "what to do next" rather than "what the ideal
    // would have been".
    let already = |idx: i32, kind: StructureType| {
        crate::functions::get_structure_type_at(state, idx) == Some(kind)
    };
    if let Some(site) = hub {
        if !already(site, hub_type) {
            stars += hub_cost;
            // Siting the hub on a forest means clearing it, which pays out.
            if state.tiles.get(&site).map(|t| t.terrain_type) == Some(TerrainType::Forest) {
                stars -= crate::version_sync::get_clear_forest_stars(state);
            }
        }
    }
    if let Some(m) = market {
        if !already(m, StructureType::Market) {
            stars += market_cost;
        }
    }

    let (mon_tiles, mon_pop) = place_monuments(
        state, territory, &buys, &partner_tiles, &adj_to_hub, hub, market, monuments,
    );
    pop += mon_pop;
    for m in &mon_tiles {
        if let Some(b) = buys.iter().find(|b| b.idx == *m && b.occupies && !displaced(b)) {
            stars -= b.cost;
        }
    }
    // A tech is still owed if any surviving buy needs it.
    let kept: HashSet<TechnologyType> = buys
        .iter()
        .filter(|b| !displaced(b) && !mon_tiles.contains(&b.idx))
        .flat_map(|b| b.techs.iter().copied())
        .collect();
    techs.retain(|t| kept.contains(t));

    BuildOut {
        pop,
        stars,
        techs,
        hub_site: hub,
        partners,
        market_site: market,
        market_spt: 0,
        monuments: mon_tiles.len() as i32,
    }
}

/// How many hub sites to price before choosing. A Tiny map's territory is at
/// most 25 tiles, so in practice every legal candidate is priced and the cap
/// only guards a pathologically large territory.
pub const HUB_TOP_K: usize = 32;

/// The tiles a city rules BEFORE BorderGrowth — the inner 3x3.
pub fn inner_ring(state: &GameState, city_idx: i32, territory: &[i32]) -> Vec<i32> {
    territory
        .iter()
        .copied()
        .filter(|&i| get_chebyshev_distance(i, city_idx, state.settings.size) <= 1)
        .collect()
}

/// Can this city actually end up with its hub on `site`?
///
/// The outer ring is the level-4 reward, so the inner 3x3 must produce
/// `POP_FOR_LEVEL_4` on its own before any outer tile is ownable. Hubs are
/// `limited_per_city`, so the single hub is spent EITHER on that climb or on an
/// outer-ring site — never both. An outer site therefore requires the inner
/// ring to reach level 4 with no hub at all, which is a much harder ask and is
/// what the old gate silently granted for free.
pub fn site_reachable(
    state: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    monuments: i32,
    site: i32,
) -> bool {
    if !sc.border_growth {
        return true;
    }
    let inner = inner_ring(state, city_idx, territory);
    let outer = get_chebyshev_distance(site, city_idx, state.settings.size) > 1;
    let climb_hub = if outer { None } else { Some(site) };
    city_build(state, &inner, sc, monuments, climb_hub, None, None).pop >= POP_FOR_LEVEL_4
}

/// What a hub site is worth to its city, as the SPT the city ends up producing
/// with the hub there and the best Market it enables.
///
/// One value function for every consumer that ranks sites. `build_out` used to
/// rank on delivered POP and site a Market afterwards, while the joint frontier
/// scores plans on SPT — so the two disagreed about the same city on the same
/// state: on an L of four crops, ranking by pop declines to crush the middle
/// one (10 pop vs 9) while SPT prefers it, because the crush buys a level-3 hub
/// and a Market reads hub level at a star a turn, forever.
///
/// Cross-city Market coupling still belongs to the frontier alone — a Market
/// can read hubs from two cities and a per-city view cannot see it. That is a
/// legitimate reason for the two to differ; the OBJECTIVE is not.
pub fn site_value(
    state: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    monuments: i32,
    site: Option<i32>,
) -> (i32, i32, i32, i32) {
    let plot = Plot::new(state, territory, sc);
    let mut b = city_build_on(state, &plot, sc, monuments, site, None, None, &[]);
    if let Some(h) = site {
        if b.partners > 0 {
            let partners = partner_tiles_of(state, territory, sc);
            if let Some((_, _, neg)) = market_sites(state, territory, h, &partners)
                .into_iter()
                .map(|m| {
                    let cb = city_build_on(state, &plot, sc, monuments, site, Some(m), None, &[]);
                    (cb.pop, -cb.stars, -m)
                })
                .max()
            {
                b = city_build_on(state, &plot, sc, monuments, site, Some(-neg), None, &[]);
                b.market_spt = b.partners.min(MARKET_MAX_LEVEL);
            }
        }
    }
    let mut pop = b.pop;
    if !sc.border_growth && pop >= POP_FOR_LEVEL_4 {
        pop += 3;
    }
    let level = level_at_pop(pop);
    let is_capital = state.tiles.get(&city_idx).is_some_and(|t| t.capital_of != 0);
    let spt = level + i32::from(is_capital) + i32::from(level >= 2) + b.market_spt;
    // The frontier's own axes, in its own order: more SPT, more super units,
    // fewer stars. Pop is deliberately NOT one of them -- at equal SPT the
    // level is equal, so extra pop inside a level buys nothing and ranking on
    // it picks needlessly expensive sites. Returned last, for display only.
    (spt, giants_at_level(level), b.stars, pop)
}

/// Single-city build: the best hub by what it actually delivers, market beside
/// it. Used by the allocator and the per-city table; the empire frontier
/// enumerates hubs jointly instead, because Market income couples the cities.
pub fn build_out(
    state: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    monuments: i32,
    goal: Goal,
) -> BuildOut {
    // Rank by delivered pop then cost, not by partner count alone. Sites with
    // equal partner counts still differ in stars — siting on Forest refunds the
    // ClearForest star, and the hub displaces whatever that tile would have
    // built — so the old lowest-tile-index tie-break could take a site strictly
    // dominated by another paying the same pop for fewer stars. Unreachable
    // sites are dropped BEFORE ranking, or the cheapest plan is one the city
    // can never actually arrive at.
    // A hub already standing IS the hub: `limited_per_city` means the slot is
    // spent, so there is nothing to choose and nothing to re-cost.
    let built = standing(state, territory, lane_hub(sc.lane).0);
    let sites: Vec<i32> = if built.is_empty() {
        hub_candidates(state, territory, sc, HUB_TOP_K)
    } else {
        built
    };
    let mut ranked: Vec<(i32, i32, i32, i32)> = sites
        .into_iter()
        .filter(|&h| site_reachable(state, city_idx, territory, sc, monuments, h))
        .map(|h| {
            let (spt, giants, stars, _pop) =
                site_value(state, city_idx, territory, sc, monuments, Some(h));
            (spt, giants, stars, h)
        })
        .collect();
    // Ordered for the STATED goal, through the same comparator `pick_for_goal`
    // applies to whole plans. `site_value` above is the measurement and stays
    // goal-agnostic; only the ordering is allowed to depend on intent.
    let m = site_maxima(&ranked.iter().map(|&(a, b, c, _)| (a, b, c)).collect::<Vec<_>>());
    ranked.sort_by(|a, b| {
        site_order_key(b.0, b.1, b.2, goal, m)
            .cmp(&site_order_key(a.0, a.1, a.2, goal, m))
            .then(a.3.cmp(&b.3))
    });
    let hub = ranked.first().map(|&(_, _, _, h)| h);
    let plot = Plot::new(state, territory, sc);
    let mut b = city_build_on(state, &plot, sc, monuments, hub, None, None, &[]);
    if let Some(h) = hub {
        if b.partners > 0 {
            // Sites stopped being interchangeable once a Market may sit on a
            // tile the plan would otherwise harvest or farm, so price each one
            // instead of taking the lowest index.
            let partners = partner_tiles_of(state, territory, sc);
            let best = market_sites(state, territory, h, &partners)
                .into_iter()
                .map(|m| {
                    let cb = city_build_on(state, &plot, sc, monuments, hub, Some(m), None, &[]);
                    (cb.pop, -cb.stars, -m)
                })
                .max();
            if let Some((_, _, neg)) = best {
                b = city_build_on(state, &plot, sc, monuments, hub, Some(-neg), None, &[]);
                b.market_spt = b.partners.min(MARKET_MAX_LEVEL);
            }
        }
    }
    b
}

/// Is a Market legal on `site`?
///
/// A resource on the tile is NOT disqualifying. The engine's terrain-structure
/// branch never looks at one (`moves/build.rs`), and a harvest CONSUMES it —
/// so a Fruit tile the plan harvests anyway is a free Market site. The tile a
/// Market must not take is a `is_hub_partner` one: displacing a partner costs
/// its hub a level, and with it every Market touching that hub.
pub fn market_site_legal(state: &GameState, site: i32, is_hub_partner: bool) -> bool {
    !is_hub_partner
        && state.tiles.get(&site).map(|t| t.terrain_type) == Some(TerrainType::Field)
        && crate::functions::get_structure_at(state, site).is_none()
}

/// Tiles already holding `kind` inside this territory. Empty on a generated
/// map; on a live state this is what the plan must build around rather than
/// re-propose.
pub fn standing(state: &GameState, territory: &[i32], kind: StructureType) -> Vec<i32> {
    territory
        .iter()
        .copied()
        .filter(|&t| crate::functions::get_structure_type_at(state, t) == Some(kind))
        .collect()
}

/// Tiles that will hold the hub's partner structure — the ones this lane plans
/// to build, plus the ones already standing. `tile_options` skips occupied
/// tiles, so without the second half a real Sawmill would count none of the
/// LumberHuts already feeding it.
pub fn partner_tiles_of(state: &GameState, territory: &[i32], sc: Scenario) -> HashSet<i32> {
    let (_, partner_name) = lane_hub(sc.lane);
    let (buys, _, _) = tile_options(state, territory, sc);
    buys.iter()
        .filter(|b| is_partner_buy(b, partner_name))
        .map(|b| b.idx)
        .chain(standing(state, territory, lane_partner_type(sc.lane)))
        .collect()
}

/// Tiles in `territory` where a Market is legal and useful — adjacent to `hub`,
/// and not one of the partners feeding it.
pub fn market_sites(
    state: &GameState,
    territory: &[i32],
    hub: i32,
    partners: &HashSet<i32>,
) -> Vec<i32> {
    let mut v: Vec<i32> = territory
        .iter()
        .copied()
        .filter(|a| {
            *a != hub
                && market_site_legal(state, *a, partners.contains(a))
                && get_adjacent_indices(state, hub, 1).contains(a)
        })
        .collect();
    v.sort();
    v
}

pub fn plan_city(
    state: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    owned: &HashSet<TechnologyType>,
    num_cities: i32,
    monuments: i32,
) -> CityPlan {
    // BorderGrowth IS the level-4 reward, so the outer ring only arrives after
    // the inner 3x3 has produced 9 pop by itself. Checked with the same
    // build_out the plan uses, or the two would disagree.
    if sc.border_growth {
        let inner = inner_ring(state, city_idx, territory);
        // Best the inner ring can do with the hub allowed — if even that misses
        // level 4 the city never earns its outer ring at all. Per-SITE
        // reachability is a stricter test applied inside `build_out`; this is
        // only the "can this city ever grow" precondition.
        let no_hub = city_build(state, &inner, sc, monuments, None, None, None).pop;
        let best_inner = hub_candidates(state, &inner, sc, HUB_TOP_K)
            .into_iter()
            .map(|h| city_build(state, &inner, sc, monuments, Some(h), None, None).pop)
            .chain(std::iter::once(no_hub))
            .max()
            .unwrap_or(no_hub);
        if best_inner < POP_FOR_LEVEL_4 {
            return CityPlan {
                scenario: sc.name,
                territory: inner.len(),
                max_pop: best_inner,
                stars: 0,
                level: level_at_pop(best_inner),
                giants: 0,
                spt: 0,
                cost_per_giant: f64::INFINITY,
                hub_site: None,
                hub_level: 0,
                market_site: None,
                feasible: false,
            };
        }
    }

    let b = build_out(state, city_idx, territory, sc, monuments, Goal::Balanced);
    let mut pop = b.pop;
    // PopGrowth (+3) is the alternative to BorderGrowth in the same slot, so it
    // pays only when the city reaches level 4 and did not take the border.
    if !sc.border_growth && pop >= POP_FOR_LEVEL_4 {
        pop += 3;
    }
    // Every tech the plan leans on, not just the lane's. A build-out that mines
    // or fishes pays for Mining or Fishing like any other cost; it used to get
    // them free, which flattered exactly the greediest plans.
    let mut chain = lane_chain(sc.lane, sc.convert);
    let mut extra: Vec<TechnologyType> = b.techs.iter().copied().collect();
    extra.sort_by_key(|t| *t as i32); // HashSet order is not stable; the bill must be
    chain.extend(extra);
    let mut stars = b.stars + tech_bill(&chain, owned, num_cities);
    if b.market_site.is_some() {
        stars += tech_bill(&market_chain(), owned, num_cities);
    }
    let level = level_at_pop(pop);
    let giants = giants_at_level(level);
    // Base production + Workshop (slot 1) + Market. Park is the alternative to a
    // giant at every slot >=4, so it is deliberately not added -- the frontier
    // shows giants and SPT as competing uses of the same slots.
    let is_capital = state.tiles.get(&city_idx).map_or(false, |t| t.capital_of != 0);
    // Base production tracks the city's LEVEL (actions/city.rs increments it per
    // level-up), so a level-5 city yields 5 — not the flat 1 this used to assume.
    let spt = level + i32::from(is_capital) + i32::from(level >= 2) + b.market_spt;

    CityPlan {
        scenario: sc.name,
        territory: territory.len(),
        max_pop: pop,
        stars,
        level,
        giants,
        spt,
        cost_per_giant: if giants > 0 { stars as f64 / giants as f64 } else { f64::INFINITY },
        hub_site: b.hub_site,
        hub_level: b.partners,
        market_site: b.market_site,
        feasible: true,
    }
}

/// The tiles each planned city actually rules, or `None` if any of them is not
/// a real city with territory — which is the case for every generated map,
/// where the "cities" past the capital are villages nobody has captured yet.
///
/// A tile a city rules is a tile its owner has explored, so planning over this
/// cannot leak anything the seat has not seen.
pub fn engine_territory(state: &GameState, cities: &[i32]) -> Option<Vec<Vec<i32>>> {
    let mut out = Vec::with_capacity(cities.len());
    for &idx in cities {
        let city = state
            .tribes
            .values()
            .flat_map(|t| t.cities.iter())
            .find(|c| c.idx == idx)
            .filter(|c| !c._territory.is_empty())?;
        out.push(crate::rules::economy::territory_tiles(state, city).collect());
    }
    Some(out)
}

/// Own `territory` outright for player 1 and hand them stars, so the engine's
/// own build path can be run against the planner's tile set.
/// The seat being planned for. On a generated map that is always player 1; on a
/// real board it is whoever `--state` was pointed at, and renumbering it to 1
/// would merge the OPPONENT's tiles into the POV's — their LumberHut next to
/// your Sawmill would count as a partner (Aug 2026).
pub fn pov_of(state: &GameState) -> crate::states::PlayerId {
    state.settings.current_player_turn_id
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Farm and Forest must get the same real adjacency check Mine always
    /// had -- previously both unconditionally returned true regardless of
    /// whether any Crop/Forest tile existed at all.
    #[test]
    fn lane_can_place_hub_checks_farm_and_forest_adjacency_for_real() {
        let mut state = GameState::default();
        state.settings.size = 11;
        // A lone Field tile: no Crop resource, no Forest neighbor anywhere.
        let idx = 5 * 11 + 5;
        let mut tile = crate::states::TileState::default();
        tile.terrain_type = TerrainType::Field;
        state.tiles.insert(idx, tile);
        let territory = vec![idx];

        assert!(!lane_can_place_hub(&state, &territory, Lane::Farm));
        assert!(!lane_can_place_hub(&state, &territory, Lane::Forest));
        assert!(!lane_can_place_hub(&state, &territory, Lane::Mine));

        // Two Crop tiles adjacent to the same center tile: Farm goes true,
        // Forest/Mine stay false.
        let center = 5 * 11 + 5;
        let crop_a = 4 * 11 + 5;
        let crop_b = 5 * 11 + 4;
        for &i in &[center, crop_a, crop_b] {
            let mut t = crate::states::TileState::default();
            t.terrain_type = TerrainType::Field;
            state.tiles.insert(i, t);
        }
        for &i in &[crop_a, crop_b] {
            state.resources.insert(
                i,
                Some(crate::states::ResourceState { resource_type: ResourceType::Crop }),
            );
        }
        let territory = vec![center, crop_a, crop_b];
        assert!(lane_can_place_hub(&state, &territory, Lane::Farm));
        assert!(!lane_can_place_hub(&state, &territory, Lane::Forest));
        assert!(!lane_can_place_hub(&state, &territory, Lane::Mine));
    }

    /// Regression for the tech-cost undercounting bug: the grow+LumberHut
    /// Buy (Field -> Forest via GrowForest, then build) must price
    /// Spiritualism, not just Forestry -- GrowForestMove is illegal without
    /// it (`moves/abilities/forest.rs`'s `has_grow` check), so a plan that
    /// didn't charge for it looked cheaper than it could ever actually be.
    #[test]
    fn grow_lumber_hut_buy_prices_spiritualism_not_just_forestry() {
        let mut state = GameState::default();
        state.settings.size = 11;
        let idx = 5 * 11 + 5;
        let mut tile = crate::states::TileState::default();
        tile.terrain_type = TerrainType::Field;
        tile.owner = 1;
        state.tiles.insert(idx, tile);

        // SCENARIOS[2] = "sawmill max greed": Lane::Forest, convert: true --
        // the scenario that offers the grow+LumberHut conversion at all.
        let sc = super::super::SCENARIOS[2];
        assert!(sc.lane == Lane::Forest);
        assert!(sc.convert);

        let (buys, _hub_sites, _convert_cost) = tile_options(&state, &[idx], sc);
        let grow = buys
            .iter()
            .find(|b| b.what == "grow+LumberHut")
            .expect("grow+LumberHut option must be offered on a bare Field tile");

        assert!(
            grow.techs.contains(&TechnologyType::Spiritualism),
            "grow+LumberHut must price Spiritualism (gates GrowForestMove itself), got {:?}",
            grow.techs
        );
        assert!(
            grow.techs.contains(&TechnologyType::Forestry),
            "grow+LumberHut must still price Forestry (unlocks the LumberHut structure), got {:?}",
            grow.techs
        );
    }

    /// Same fact, the other call site: `lane_chain`'s forest-convert case
    /// must price Spiritualism through the same SSOT lookup, not a
    /// hand-written literal that could silently drift from it.
    #[test]
    fn lane_chain_forest_convert_prices_spiritualism() {
        let chain = lane_chain(Lane::Forest, true);
        assert!(chain.contains(&TechnologyType::Spiritualism));
        assert!(!lane_chain(Lane::Forest, false).contains(&TechnologyType::Spiritualism));
    }
}

