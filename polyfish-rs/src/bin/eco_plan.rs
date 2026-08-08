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

use polyfish::functions::{get_adjacent_indices, get_chebyshev_distance, MARKET_MAX_LEVEL};
use polyfish::settings::resources::get_resource_setting;
use polyfish::settings::structures::get_structure_setting;
use polyfish::settings::technology::{get_tech_cost, get_technology_setting};
use polyfish::states::GameState;
use polyfish::types::*;
use std::collections::{HashMap, HashSet};

/// Pop needed to take a city from `from` to `to` (level L->L+1 costs L+1).
fn pop_for_levels(from: i32, to: i32) -> i32 {
    (from..to).map(|l| l + 1).sum()
}

/// Pop to reach the level-4 reward slot, where BorderGrowth/PopGrowth is offered.
const POP_FOR_LEVEL_4: i32 = 9;

/// Highest level reachable with `pop`, starting from level 1.
fn level_at_pop(pop: i32) -> i32 {
    let mut level = 1;
    while pop_for_levels(1, level + 1) <= pop {
        level += 1;
    }
    level
}

/// Giants available at a level: every reward slot from 4 up offers SuperUnit.
fn giants_at_level(level: i32) -> i32 {
    (level - 4).max(0)
}

/// One buyable thing on one tile.
#[derive(Clone, Debug)]
struct Buy {
    idx: i32,
    what: &'static str,
    cost: i32,
    pop: i32,
    /// Places a structure, so it competes with a hub for the tile. Harvests
    /// (Fruit/Game/Fish) leave the tile empty and do not.
    occupies: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Lane {
    Forest,
    Farm,
}

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    lane: Lane,
    border_growth: bool,
    /// Forest->Field+Crop (BurnForest, with Construction) for the farm lane, or
    /// Field->Forest (GrowForest, needs Spiritualism) for the forest lane.
    convert: bool,
}

const SCENARIOS: [Scenario; 6] = [
    Scenario { name: "sawmill natural",      lane: Lane::Forest, border_growth: false, convert: false },
    Scenario { name: "sawmill +border",      lane: Lane::Forest, border_growth: true,  convert: false },
    Scenario { name: "sawmill max greed",    lane: Lane::Forest, border_growth: true,  convert: true  },
    Scenario { name: "windmill natural",     lane: Lane::Farm,   border_growth: false, convert: false },
    Scenario { name: "windmill +border",     lane: Lane::Farm,   border_growth: true,  convert: false },
    Scenario { name: "windmill max greed",   lane: Lane::Farm,   border_growth: true,  convert: true  },
];

/// Techs a lane needs, in dependency order. Returned regardless of endowment;
/// the caller prices only the ones not already owned.
fn lane_chain(lane: Lane, convert: bool) -> Vec<TechnologyType> {
    let mut v = match lane {
        Lane::Forest => vec![
            TechnologyType::Hunting,
            TechnologyType::Forestry,
            TechnologyType::Mathematics,
        ],
        Lane::Farm => vec![TechnologyType::Farming, TechnologyType::Construction],
    };
    if convert {
        match lane {
            // GrowForest: Field -> Forest.
            Lane::Forest => v.extend([
                TechnologyType::Archery,
                TechnologyType::Spiritualism,
            ]),
            // BurnForest rides along with Construction, already in the chain.
            Lane::Farm => {}
        }
    }
    v
}

/// Market needs Trade, which is three techs deep off nothing the economy lanes
/// buy. Charged once empire-wide, and only when a Market is actually placed.
fn market_chain() -> [TechnologyType; 3] {
    [
        TechnologyType::Riding,
        TechnologyType::Roads,
        TechnologyType::Trade,
    ]
}

fn tech_bill(chain: &[TechnologyType], owned: &HashSet<TechnologyType>, cities: i32) -> i32 {
    let mut bill = 0;
    let mut have = owned.clone();
    for t in chain {
        if have.contains(t) {
            continue;
        }
        // Pull in any unowned prerequisite too.
        let mut stack = vec![*t];
        while let Some(cur) = stack.pop() {
            if have.contains(&cur) {
                continue;
            }
            let s = get_technology_setting(cur);
            if let Some(req) = s.requires {
                if !have.contains(&req) {
                    stack.push(cur);
                    stack.push(req);
                    continue;
                }
            }
            bill += get_tech_cost(cities, s.tier.unwrap_or(1), false);
            have.insert(cur);
        }
    }
    bill
}

/// Every tile a city could develop, after any terrain conversion the scenario
/// allows. Returns (buys, hub_sites, conversion_cost_by_tile).
fn tile_options(
    state: &GameState,
    territory: &[i32],
    sc: Scenario,
) -> (Vec<Buy>, Vec<i32>, HashMap<i32, i32>) {
    let mut buys = Vec::new();
    let mut hub_sites = Vec::new();
    let mut convert_cost = HashMap::new();

    let burn = polyfish::version_sync::get_burn_forest_cost(state);
    const GROW_FOREST_COST: i32 = 5;

    for &idx in territory {
        let Some(tile) = state.tiles.get(&idx) else { continue };
        if polyfish::functions::get_structure_at(state, idx).is_some() {
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
                    }),
                    Some(r) => buys.extend(harvest(r)),
                    None => {}
                }
                if sc.lane == Lane::Forest && sc.convert {
                    convert_cost.insert(idx, GROW_FOREST_COST);
                    buys.push(Buy {
                        idx,
                        what: "grow+Hut",
                        cost: GROW_FOREST_COST + 3,
                        pop: 1,
                        occupies: true,
                    });
                }
            }
            TerrainType::Forest => {
                // Forestry unlocks ClearForest: Forest -> Field, and it PAYS a
                // star. So any forest is a candidate hub site for the forest
                // lane -- which is how a Sawmill ends up in the middle of a
                // forest cluster rather than on its edge.
                if sc.lane == Lane::Forest {
                    hub_sites.push(idx);
                }
                // Hunting the Game leaves the forest standing, so the tile can
                // pay twice: harvest now, LumberHut after.
                if let Some(r) = res {
                    buys.extend(harvest(r));
                }
                if sc.lane == Lane::Forest || !sc.convert {
                    buys.push(Buy { idx, what: "LumberHut", cost: 3, pop: 1, occupies: true });
                } else {
                    convert_cost.insert(idx, burn);
                    buys.push(Buy {
                        idx,
                        what: "burn+Farm",
                        cost: burn + get_structure_setting(StructureType::Farm).cost.unwrap_or(5),
                        pop: get_resource_setting(ResourceType::Crop).reward_pop,
                        occupies: true,
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

struct CityPlan {
    scenario: &'static str,
    territory: usize,
    max_pop: i32,
    stars: i32,
    level: i32,
    giants: i32,
    spt: i32,
    cost_per_giant: f64,
    hub_site: Option<i32>,
    hub_level: i32,
    market_site: Option<i32>,
    feasible: bool,
}

/// The partner structure a lane's hub eats, and the hub itself.
fn lane_hub(lane: Lane) -> (StructureType, &'static str) {
    match lane {
        Lane::Forest => (StructureType::Sawmill, "LumberHut"),
        Lane::Farm => (StructureType::Windmill, "Farm"),
    }
}

/// What a city can build out to on a given tile set, under one scenario.
/// Shared by the real plan and the border-growth reachability check so the two
/// can never disagree about how much pop a tile set yields.
struct BuildOut {
    pop: i32,
    stars: i32,
    hub_site: Option<i32>,
    partners: i32,
    market_site: Option<i32>,
    market_spt: i32,
    monuments: i32,
}

/// Monuments are free and pay 3 pop, but occupy a tile. Placing one costs
/// whatever that tile would have produced — 0 on a bare Field, 1 on a Forest
/// that would have been a LumberHut, 2 more if it was feeding the hub. Returns
/// (tiles used, net pop gained).
fn place_monuments(
    state: &GameState,
    territory: &[i32],
    buys: &[Buy],
    partner_tiles: &HashSet<i32>,
    adj_to_hub: &HashSet<i32>,
    hub_site: Option<i32>,
    market_site: Option<i32>,
    budget: i32,
) -> (HashSet<i32>, i32) {
    const MONUMENT_POP: i32 = 3;
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
            }) && polyfish::functions::get_structure_at(state, *i).is_none()
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

/// Hub sites in a territory, best first by buildable partner count. A site
/// paying only 1 partner is never worth 5* (worse than the partner feeding it),
/// and a 0-partner hub is not even legal (`build.rs` requires an adjacent
/// partner), so both are excluded.
fn hub_candidates(state: &GameState, territory: &[i32], sc: Scenario, top_k: usize) -> Vec<i32> {
    let (buys, hub_sites, _) = tile_options(state, territory, sc);
    let (_, partner_name) = lane_hub(sc.lane);
    let partner_tiles: HashSet<i32> = buys
        .iter()
        .filter(|b| b.what == partner_name || b.what.ends_with(partner_name))
        .map(|b| b.idx)
        .collect();
    let mut scored: Vec<(i32, i32)> = hub_sites
        .iter()
        .map(|&s| {
            let n = get_adjacent_indices(state, s, 1)
                .into_iter()
                .filter(|a| partner_tiles.contains(a) && *a != s)
                .count() as i32;
            (-n, s)
        })
        .filter(|&(negn, _)| -negn >= 2)
        .collect();
    scored.sort();
    scored.into_iter().take(top_k).map(|(_, s)| s).collect()
}

/// One city's build-out with the hub and market sites SUPPLIED. Market income
/// is not computed here — it depends on every city's hub, so the caller does it.
fn city_build(
    state: &GameState,
    territory: &[i32],
    sc: Scenario,
    monuments: i32,
    hub: Option<i32>,
    market: Option<i32>,
    // Partner tiles across the WHOLE empire. Adjacency pay is player-scoped
    // (`build_structure` tests `t.owner == pov_id`), so a hub sited near a
    // border collects partners on the far side too. Counting only this city's
    // tiles undervalues exactly the border hubs that feed a shared Market.
    empire_partners: Option<&HashSet<i32>>,
) -> BuildOut {
    let (buys, _hub_sites, _conv) = tile_options(state, territory, sc);
    let (hub_type, partner_name) = lane_hub(sc.lane);
    let hub_cost = get_structure_setting(hub_type).cost.unwrap_or(5);
    let market_cost = get_structure_setting(StructureType::Market).cost.unwrap_or(5);

    let partner_tiles: HashSet<i32> = buys
        .iter()
        .filter(|b| b.what == partner_name || b.what.ends_with(partner_name))
        .map(|b| b.idx)
        .collect();
    let adj_to_hub: HashSet<i32> = hub
        .map(|s| get_adjacent_indices(state, s, 1).into_iter().collect())
        .unwrap_or_default();

    // A structure-placing buy is displaced by a hub or market on its tile;
    // harvests are not, since they leave the tile empty.
    let displaced = |b: &Buy| b.occupies && (Some(b.idx) == hub || Some(b.idx) == market);
    let mut pop = 0;
    let mut stars = 0;
    for b in buys.iter().filter(|b| !displaced(b)) {
        stars += b.cost;
        pop += b.pop;
    }
    let counted = empire_partners.unwrap_or(&partner_tiles);
    let partners = counted
        .iter()
        .filter(|t| adj_to_hub.contains(t) && Some(**t) != hub && Some(**t) != market)
        .count() as i32;
    pop += partners;

    if let Some(site) = hub {
        stars += hub_cost;
        // Siting the hub on a forest means clearing it, which pays out.
        if state.tiles.get(&site).map(|t| t.terrain_type) == Some(TerrainType::Forest) {
            stars -= polyfish::version_sync::get_clear_forest_stars(state);
        }
    }
    if market.is_some() {
        stars += market_cost;
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

    BuildOut {
        pop,
        stars,
        hub_site: hub,
        partners,
        market_site: market,
        market_spt: 0,
        monuments: mon_tiles.len() as i32,
    }
}

/// Greedy single-city build: best hub by partner count, market beside it.
/// Used by the allocator and the per-city table; the empire frontier enumerates
/// hubs jointly instead, because Market income couples the cities.
fn build_out(state: &GameState, territory: &[i32], sc: Scenario, monuments: i32) -> BuildOut {
    let hub = hub_candidates(state, territory, sc, 1).into_iter().next();
    let mut b = city_build(state, territory, sc, monuments, hub, None, None);
    if let Some(h) = hub {
        if b.partners > 0 {
            let market = market_sites(state, territory, h)
                .into_iter()
                .next();
            if let Some(m) = market {
                b = city_build(state, territory, sc, monuments, hub, Some(m), None);
                b.market_spt = b.partners.min(MARKET_MAX_LEVEL);
            }
        }
    }
    b
}

/// Tiles in `territory` where a Market is legal and useful: Field terrain, empty,
/// carrying no resource worth harvesting, and adjacent to `hub`.
fn market_sites(state: &GameState, territory: &[i32], hub: i32) -> Vec<i32> {
    let mut v: Vec<i32> = territory
        .iter()
        .copied()
        .filter(|a| {
            *a != hub
                && state.tiles.get(a).map(|t| t.terrain_type) == Some(TerrainType::Field)
                && state.resources.get(a).map_or(true, |r| r.is_none())
                && polyfish::functions::get_structure_at(state, *a).is_none()
                && get_adjacent_indices(state, hub, 1).contains(a)
        })
        .collect();
    v.sort();
    v
}

fn plan_city(
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
        let inner: Vec<i32> = territory
            .iter()
            .copied()
            .filter(|&i| get_chebyshev_distance(i, city_idx, state.settings.size) <= 1)
            .collect();
        let ib = build_out(state, &inner, sc, monuments);
        if ib.pop < POP_FOR_LEVEL_4 {
            return CityPlan {
                scenario: sc.name,
                territory: inner.len(),
                max_pop: ib.pop,
                stars: 0,
                level: level_at_pop(ib.pop),
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

    let b = build_out(state, territory, sc, monuments);
    let mut pop = b.pop;
    // PopGrowth (+3) is the alternative to BorderGrowth in the same slot, so it
    // pays only when the city reaches level 4 and did not take the border.
    if !sc.border_growth && pop >= POP_FOR_LEVEL_4 {
        pop += 3;
    }
    let mut stars = b.stars + tech_bill(&lane_chain(sc.lane, sc.convert), owned, num_cities);
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

/// Assign every tile within radius 2 of any planned city to exactly one city —
/// nearest wins, ties to the earlier city — so overlapping 5x5s are not
/// double-counted.
/// Joint allocation by MARGINAL VALUE: a contested tile goes to whichever
/// claimant's plan gains more pop from it. Nearest-city was the first rule and
/// it was wrong — it handed the capital tiles it had no use for while capping a
/// neighbour's Sawmill a level below what the map supports.
fn allocate_value(
    state: &GameState,
    cities: &[i32],
    sc: Scenario,
    monuments: i32,
) -> Vec<Vec<i32>> {
    let radius = if sc.border_growth { 2 } else { 1 };
    let mut claimants: HashMap<i32, Vec<usize>> = HashMap::new();
    for (ci, &c) in cities.iter().enumerate() {
        for idx in get_adjacent_indices(state, c, radius).into_iter().chain([c]) {
            if state.tiles.contains_key(&idx) {
                claimants.entry(idx).or_default().push(ci);
            }
        }
    }

    let mut terr: Vec<Vec<i32>> = vec![Vec::new(); cities.len()];
    let mut contested: Vec<(i32, Vec<usize>)> = Vec::new();
    for (idx, who) in claimants {
        if who.len() == 1 {
            terr[who[0]].push(idx);
        } else {
            contested.push((idx, who));
        }
    }
    // A city centre always belongs to itself.
    for (ci, &c) in cities.iter().enumerate() {
        contested.retain(|(idx, _)| {
            if *idx == c {
                terr[ci].push(c);
                false
            } else {
                true
            }
        });
    }
    contested.sort_by_key(|(idx, _)| *idx);

    for (idx, who) in contested {
        // Rank by (pop gained, nearer city, lower index) — every term
        // deterministic, so the allocation is reproducible.
        let mut ranked: Vec<(i32, i32, usize)> = who
            .iter()
            .map(|&ci| {
                let mut with = terr[ci].clone();
                let base = build_out(state, &with, sc, monuments).pop;
                with.push(idx);
                with.sort();
                let gain = build_out(state, &with, sc, monuments).pop - base;
                (
                    -gain,
                    get_chebyshev_distance(idx, cities[ci], state.settings.size),
                    ci,
                )
            })
            .collect();
        ranked.sort();
        terr[ranked[0].2].push(idx);
    }
    for v in terr.iter_mut() {
        v.sort();
    }
    terr
}

fn allocate(state: &GameState, cities: &[i32], border_growth: bool) -> Vec<Vec<i32>> {
    allocate_mode(state, cities, border_growth, false)
}

/// `standalone` scores each city on its full square, ignoring that neighbours
/// would claim the shared tiles — the per-city CEILING, not a joint plan.
fn allocate_mode(
    state: &GameState,
    cities: &[i32],
    border_growth: bool,
    standalone: bool,
) -> Vec<Vec<i32>> {
    let radius = if border_growth { 2 } else { 1 };
    if standalone {
        return cities
            .iter()
            .map(|&c| {
                let mut v: Vec<i32> = get_adjacent_indices(state, c, radius)
                    .into_iter()
                    .chain([c])
                    .filter(|i| state.tiles.contains_key(i))
                    .collect();
                v.sort();
                v
            })
            .collect();
    }
    let mut owner: HashMap<i32, usize> = HashMap::new();
    for (ci, &c) in cities.iter().enumerate() {
        for idx in get_adjacent_indices(state, c, radius).into_iter().chain([c]) {
            let d = get_chebyshev_distance(idx, c, state.settings.size);
            match owner.get(&idx) {
                Some(&prev) => {
                    let pd = get_chebyshev_distance(idx, cities[prev], state.settings.size);
                    if d < pd {
                        owner.insert(idx, ci);
                    }
                }
                None => {
                    owner.insert(idx, ci);
                }
            }
        }
    }
    let mut out = vec![Vec::new(); cities.len()];
    for (idx, ci) in owner {
        out[ci].push(idx);
    }
    for v in out.iter_mut() {
        v.sort();
    }
    out
}

/// One whole-empire plan: a hub choice per city, the markets that follow, and
/// what it costs and yields.
#[derive(Clone)]
struct EmpirePlan {
    scenario: &'static str,
    stars: i32,
    pop: i32,
    giants: i32,
    spt: i32,
    hubs: Vec<Option<i32>>,
    levels: Vec<i32>,
    markets: Vec<Option<i32>>,
    market_income: Vec<i32>,
}

/// Enumerate hub sites JOINTLY across cities. Markets are one per city but a
/// market earns the summed LEVEL of every friendly hub it touches, including
/// hubs belonging to other cities — so two sawmills sited near a shared border
/// can feed two markets at full value. A per-city greedy picker cannot see
/// that; this is why the enumeration has to be joint.
fn enumerate_empire(
    state: &GameState,
    cities: &[i32],
    terr: &[Vec<i32>],
    sc: Scenario,
    owned: &HashSet<TechnologyType>,
    monuments: i32,
    top_k: usize,
    with_markets: bool,
) -> Vec<EmpirePlan> {
    let n = cities.len();
    let num_cities = n as i32;

    // STAGE 1 — hubs and markets over PLAYER-owned tiles, city boundaries
    // ignored. Sound because adjacency is player-scoped throughout the engine:
    // `build_structure` tests `t.owner == pov_id` and the Market branch tests
    // `t.owner == city.owner`, both player ids. A hub's level and a Market's
    // income therefore do not depend on how tiles are split between cities.
    let union: Vec<i32> = {
        let mut u: HashSet<i32> = HashSet::new();
        for t in terr {
            u.extend(t.iter().copied());
        }
        let mut v: Vec<i32> = u.into_iter().collect();
        v.sort();
        v
    };
    // Empire-wide partner tiles: what any hub can collect, wherever it sits.
    let empire_partners: HashSet<i32> = {
        let (buys, _, _) = tile_options(state, &union, sc);
        let (_, partner_name) = lane_hub(sc.lane);
        buys.iter()
            .filter(|b| b.what == partner_name || b.what.ends_with(partner_name))
            .map(|b| b.idx)
            .collect()
    };

    // Candidates PER CITY, scored on empire-wide partners but sited only on
    // tiles that city owns — so `limited_per_city` holds by construction and
    // nothing is generated just to be filtered out.
    let cands: Vec<Vec<Option<i32>>> = (0..n)
        .map(|ci| {
            let mut scored: Vec<(i32, i32)> = terr[ci]
                .iter()
                .copied()
                .filter(|&t| {
                    let Some(tile) = state.tiles.get(&t) else { return false };
                    polyfish::functions::get_structure_at(state, t).is_none()
                        && (tile.terrain_type == TerrainType::Field
                            || (sc.lane == Lane::Forest
                                && tile.terrain_type == TerrainType::Forest))
                })
                .map(|t| {
                    let n = get_adjacent_indices(state, t, 1)
                        .into_iter()
                        .filter(|a| empire_partners.contains(a) && *a != t)
                        .count() as i32;
                    (-n, t)
                })
                .filter(|&(negn, _)| -negn >= 2)
                .collect();
            scored.sort();
            let mut v: Vec<Option<i32>> =
                scored.into_iter().take(top_k).map(|(_, t)| Some(t)).collect();
            v.push(None);
            v
        })
        .collect();

    let lane_tech = tech_bill(&lane_chain(sc.lane, sc.convert), owned, num_cities);
    let market_tech = tech_bill(&market_chain(), owned, num_cities);

    let mut combos: Vec<Vec<Option<i32>>> = Vec::new();
    let total: usize = cands.iter().map(|c| c.len()).product();
    for mut code in 0..total {
        let mut pick = Vec::with_capacity(n);
        for c in &cands {
            pick.push(c[code % c.len()]);
            code /= c.len();
        }
        combos.push(pick);
    }

    let mut out: Vec<EmpirePlan> = Vec::new();
    // Many hub combinations score identically; keep one representative each.
    let mut seen_score: HashSet<(i32, i32, i32)> = HashSet::new();
    for hubs in combos {
        let base: Vec<BuildOut> = (0..n)
            .map(|ci| {
                city_build(
                    state, &terr[ci], sc, monuments, hubs[ci], None,
                    Some(&empire_partners),
                )
            })
            .collect();
        let levels: Vec<i32> = base.iter().map(|b| b.partners).collect();
        let placed: Vec<(i32, i32)> = (0..n)
            .filter_map(|ci| hubs[ci].map(|h| (h, levels[ci])))
            .filter(|&(_, l)| l > 0)
            .collect();

        // STAGE 2 — one Market per city, sited to collect the summed level of
        // every hub it touches, whichever city that hub belongs to.
        let mut markets = vec![None; n];
        let mut income = vec![0; n];
        if with_markets && !placed.is_empty() {
            for ci in 0..n {
                let best = terr[ci]
                    .iter()
                    .copied()
                    .filter(|a| {
                        state.tiles.get(a).map(|t| t.terrain_type) == Some(TerrainType::Field)
                            && state.resources.get(a).map_or(true, |r| r.is_none())
                            && polyfish::functions::get_structure_at(state, *a).is_none()
                            && !hubs.iter().any(|h| *h == Some(*a))
                    })
                    .map(|a| {
                        let adj = get_adjacent_indices(state, a, 1);
                        let sum: i32 = placed
                            .iter()
                            .filter(|(h, _)| adj.contains(h))
                            .map(|(_, l)| *l)
                            .sum();
                        (sum.min(MARKET_MAX_LEVEL), -a)
                    })
                    .filter(|&(sum, _)| sum > 0) // a Market is only legal beside a hub
                    .max();
                if let Some((sum, neg)) = best {
                    markets[ci] = Some(-neg);
                    income[ci] = sum;
                }
            }
        }

        // STAGE 3 — cost it, route pop to cities, score.
        let mut stars = lane_tech;
        if markets.iter().any(|m| m.is_some()) {
            stars += market_tech;
        }
        let (mut pop, mut giants, mut spt) = (0, 0, 0);
        let mut feasible = true;
        for ci in 0..n {
            if sc.border_growth {
                let inner: Vec<i32> = terr[ci]
                    .iter()
                    .copied()
                    .filter(|&i| {
                        get_chebyshev_distance(i, cities[ci], state.settings.size) <= 1
                    })
                    .collect();
                if city_build(state, &inner, sc, monuments, None, None, None).pop
                    < POP_FOR_LEVEL_4
                {
                    feasible = false;
                    break;
                }
            }
            let b = city_build(
                state, &terr[ci], sc, monuments, hubs[ci], markets[ci],
                Some(&empire_partners),
            );
            let mut city_pop = b.pop;
            if !sc.border_growth && city_pop >= POP_FOR_LEVEL_4 {
                city_pop += 3; // PopGrowth, the alternative to the border
            }
            let level = level_at_pop(city_pop);
            let is_capital =
                state.tiles.get(&cities[ci]).map_or(false, |t| t.capital_of != 0);
            stars += b.stars;
            pop += city_pop;
            giants += giants_at_level(level);
            spt += level + i32::from(is_capital) + i32::from(level >= 2) + income[ci];
        }
        if !feasible {
            continue;
        }
        if !seen_score.insert((stars, giants, spt)) {
            continue;
        }
        out.push(EmpirePlan {
            scenario: sc.name,
            stars,
            pop,
            giants,
            spt,
            hubs,
            levels,
            markets,
            market_income: income,
        });
    }
    out
}
/// Non-dominated plans: cheaper is better, more giants and more SPT are better.
fn pareto(plans: &[EmpirePlan]) -> Vec<EmpirePlan> {
    plans
        .iter()
        .filter(|p| {
            !plans.iter().any(|q| {
                q.stars <= p.stars
                    && q.giants >= p.giants
                    && q.spt >= p.spt
                    && (q.stars < p.stars || q.giants > p.giants || q.spt > p.spt)
            })
        })
        .cloned()
        .collect()
}

/// Tile-by-tile dump for one city+scenario, so a number in the table can be
/// traced back to the map.
fn explain(state: &GameState, city_idx: i32, territory: &[i32], sc: Scenario) {
    let (buys, hub_sites, _) = tile_options(state, territory, sc);
    let (_, partner_name) = lane_hub(sc.lane);
    let partner_tiles: HashSet<i32> = buys
        .iter()
        .filter(|b| b.what == partner_name || b.what.ends_with(partner_name))
        .map(|b| b.idx)
        .collect();
    let b = build_out(state, territory, sc, 0);
    println!("\n  --- explain: city {city_idx}, {} ---", sc.name);
    println!("  territory ({} tiles): {:?}", territory.len(), territory);
    let mut sites: Vec<(i32, i32)> = hub_sites
        .iter()
        .map(|&s| {
            (
                s,
                get_adjacent_indices(state, s, 1)
                    .into_iter()
                    .filter(|a| partner_tiles.contains(a))
                    .count() as i32,
            )
        })
        .collect();
    sites.sort_by_key(|&(s, n)| (-n, s));
    println!("  hub sites by partner count: {:?}", &sites[..sites.len().min(8)]);
    println!("  chosen hub {:?} at level {}", b.hub_site, b.partners);
    let mut by_kind: std::collections::BTreeMap<&str, Vec<i32>> = Default::default();
    for x in buys.iter().filter(|x| Some(x.idx) != b.hub_site) {
        by_kind.entry(x.what).or_default().push(x.idx);
    }
    for (k, v) in by_kind {
        println!("    {k:<12} x{:<3} {:?}", v.len(), v);
    }
    println!("  pop {}  stars(structures) {}  market {:?} (+{} SPT)",
             b.pop, b.stars, b.market_site, b.market_spt);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
    };
    let seed: i64 = get("--seed").and_then(|s| s.parse().ok()).unwrap_or(4102);
    let owned: HashSet<TechnologyType> = get("--techs")
        .map(|s| {
            s.split(',')
                .filter_map(|t| match t.trim().to_lowercase().as_str() {
                    "organization" => Some(TechnologyType::Organization),
                    "hunting" => Some(TechnologyType::Hunting),
                    "forestry" => Some(TechnologyType::Forestry),
                    "mathematics" => Some(TechnologyType::Mathematics),
                    "farming" => Some(TechnologyType::Farming),
                    "construction" => Some(TechnologyType::Construction),
                    "archery" => Some(TechnologyType::Archery),
                    "spiritualism" => Some(TechnologyType::Spiritualism),
                    "climbing" => Some(TechnologyType::Climbing),
                    "mining" => Some(TechnologyType::Mining),
                    "fishing" => Some(TechnologyType::Fishing),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_else(|| [TechnologyType::Organization].into_iter().collect());

    let state = polyfish::mapgen::generate(polyfish::mapgen::MapGenSettings {
        size: MapSize::Tiny,
        map_type: MapType::Drylands,
        tribes: vec![TribeType::Imperius, TribeType::Imperius],
        seed,
        ..Default::default()
    });

    // Capital of player 1, then the nearest villages.
    let capital = state
        .tribes
        .get(&1)
        .and_then(|t| t.cities.first().map(|c| c.idx))
        .expect("player 1 has no city");
    let mut villages: Vec<i32> = state
        .structures
        .iter()
        .filter_map(|(&i, s)| {
            s.as_ref()
                .filter(|s| s.structure_type == StructureType::Village)
                .filter(|_| state.tiles.get(&i).map_or(false, |t| t.owner == 0))
                .map(|_| i)
        })
        .collect();
    villages.sort_by_key(|&v| get_chebyshev_distance(v, capital, state.settings.size));
    let n_extra: usize = get("--cities").and_then(|s| s.parse().ok()).unwrap_or(3) - 1;
    let mut cities = vec![capital];
    cities.extend(villages.into_iter().take(n_extra));

    println!("seed {seed} | capital {capital} | cities {cities:?}");
    let monuments: i32 = get("--monuments").and_then(|s| s.parse().ok()).unwrap_or(1);
    let standalone = args.iter().any(|a| a == "--standalone");
    let with_markets = !args.iter().any(|a| a == "--no-markets");
    if standalone {
        println!("STANDALONE: each city scored on its full square (tiles double-counted)");
    }
    if let Some(which) = get("--explain") {
        let ci: usize = which.parse().unwrap_or(0);
        for sc in SCENARIOS {
            let terr = if standalone {
                allocate_mode(&state, &cities, sc.border_growth, true)
            } else {
                allocate_value(&state, &cities, sc, monuments)
            };
            explain(&state, cities[ci], &terr[ci], sc);
        }
        return;
    }
    println!(
        "owned techs: {:?}",
        owned.iter().map(|t| format!("{t:?}")).collect::<Vec<_>>()
    );
    let num_cities = cities.len() as i32;
    println!("monuments allowed per city: {monuments}  (free, 3 pop, task-gated in play)");

    let mut all: Vec<(i32, CityPlan)> = Vec::new();
    for sc in SCENARIOS {
        let terr = if standalone {
            allocate_mode(&state, &cities, sc.border_growth, true)
        } else {
            allocate_value(&state, &cities, sc, monuments)
        };
        for (ci, &c) in cities.iter().enumerate() {
            all.push((c, plan_city(&state, c, &terr[ci], sc, &owned, num_cities, monuments)));
        }
    }

    for &c in &cities {
        println!("\n{}", "=".repeat(104));
        let role = if c == capital { "CAPITAL" } else { "city" };
        println!("  {role} @ tile {c}");
        {
            let mut census: std::collections::BTreeMap<String, i32> = Default::default();
            for idx in get_adjacent_indices(&state, c, 2).into_iter().chain([c]) {
                let Some(t) = state.tiles.get(&idx) else { continue };
                let r = state
                    .resources
                    .get(&idx)
                    .and_then(|r| r.as_ref())
                    .map(|r| format!("+{:?}", r.resource_type))
                    .unwrap_or_default();
                *census.entry(format!("{:?}{}", t.terrain_type, r)).or_default() += 1;
            }
            let line: Vec<String> = census.iter().map(|(k, v)| format!("{k} {v}")).collect();
            println!("  5x5 terrain: {}", line.join(", "));
        }
        println!("{}", "=".repeat(104));
        println!(
            "  {:<20}{:>6}{:>7}{:>8}{:>7}{:>8}{:>7}{:>12}{:>10}{:>8}",
            "scenario", "tiles", "pop", "stars", "level", "giants", "SPT", "★/giant", "hub@lvl", "market"
        );
        println!("  {}", "-".repeat(100));
        for (city, p) in all.iter().filter(|(city, _)| *city == c) {
            let _ = city;
            let cpg = if p.cost_per_giant.is_finite() {
                format!("{:.1}", p.cost_per_giant)
            } else {
                "—".into()
            };
            if !p.feasible {
                println!(
                    "  {:<20}{:>6}{:>7}{:>8}{:>7}{:>8}{:>7}{:>12}{:>10}{:>8}",
                    p.scenario, p.territory, p.max_pop, "—", p.level, "—", "—",
                    "unreachable", "—", "—"
                );
                continue;
            }
            let hub = match p.hub_site {
                Some(s) => format!("{}@{}", s, p.hub_level),
                None => "—".into(),
            };
            println!(
                "  {:<20}{:>6}{:>7}{:>8}{:>7}{:>8}{:>7}{:>12}{:>10}{:>8}",
                p.scenario,
                p.territory,
                p.max_pop,
                p.stars,
                p.level,
                p.giants,
                p.spt,
                cpg,
                hub,
                p.market_site.map(|s| s.to_string()).unwrap_or("—".into()),
            );
        }
    }

    // ---- Joint frontier: hubs enumerated across cities, markets placed to
    // ---- touch every hub they can reach.
    let mut all_plans: Vec<EmpirePlan> = Vec::new();
    for sc in SCENARIOS {
        let terr = if standalone {
            allocate_mode(&state, &cities, sc.border_growth, true)
        } else {
            allocate_value(&state, &cities, sc, monuments)
        };
        all_plans.extend(enumerate_empire(
            &state, &cities, &terr, sc, &owned, monuments, 8, with_markets,
        ));
    }
    let front = pareto(&all_plans);

    println!("\n{}", "=".repeat(112));
    println!(
        "  JOINT FRONTIER — {} plans enumerated, {} non-dominated  (min stars, max giants, max SPT)",
        all_plans.len(),
        front.len()
    );
    println!("{}", "=".repeat(112));

    // Knee: on the giants/SPT plane, the point furthest from the chord joining
    // the two extremes. No hand-tuned exchange rate, and unlike a weighted sum
    // it can select points in a concave stretch of the frontier — which is
    // where "slightly worse hub, much better market" plans live.
    let (gmin, gmax) = (
        front.iter().map(|p| p.giants).min().unwrap_or(0) as f32,
        front.iter().map(|p| p.giants).max().unwrap_or(1) as f32,
    );
    let (smin, smax) = (
        front.iter().map(|p| p.spt).min().unwrap_or(0) as f32,
        front.iter().map(|p| p.spt).max().unwrap_or(1) as f32,
    );
    let norm = |p: &EmpirePlan| {
        (
            if gmax > gmin { (p.giants as f32 - gmin) / (gmax - gmin) } else { 0.0 },
            if smax > smin { (p.spt as f32 - smin) / (smax - smin) } else { 0.0 },
        )
    };
    let knee = front
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let (ag, as_) = norm(a);
            let (bg, bs) = norm(b);
            // Distance from the x+y=1 chord; ties to the cheaper plan.
            ((ag + as_ - 1.0).abs() * -1.0 + ag + as_)
                .partial_cmp(&((bg + bs - 1.0).abs() * -1.0 + bg + bs))
                .unwrap()
                .then(b.stars.cmp(&a.stars))
        })
        .map(|(i, _)| i);
    let best_giants = front
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.giants.cmp(&b.giants).then(b.stars.cmp(&a.stars)))
        .map(|(i, _)| i);
    let best_spt = front
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.spt.cmp(&b.spt).then(b.stars.cmp(&a.stars)))
        .map(|(i, _)| i);

    println!(
        "  {:<20}{:>8}{:>7}{:>8}{:>7}{:>10}  {:<26}{:<22}{}",
        "scenario", "stars", "pop", "giants", "SPT", "★/giant", "hubs @ level", "markets (+income)", ""
    );
    println!("  {}", "-".repeat(108));
    let mut rows: Vec<(usize, &EmpirePlan)> = front.iter().enumerate().collect();
    rows.sort_by_key(|(_, p)| (p.stars, -p.giants));
    for (i, p) in rows {
        let hubs: Vec<String> = p
            .hubs
            .iter()
            .zip(&p.levels)
            .map(|(h, l)| match h {
                Some(h) => format!("{h}@{l}"),
                None => "—".into(),
            })
            .collect();
        let mkts: Vec<String> = p
            .markets
            .iter()
            .zip(&p.market_income)
            .map(|(m, inc)| match m {
                Some(m) => format!("{m}+{inc}"),
                None => "—".into(),
            })
            .collect();
        let mut tag = String::new();
        if Some(i) == knee {
            tag.push_str(" KNEE");
        }
        if Some(i) == best_giants {
            tag.push_str(" MAX-GIANTS");
        }
        if Some(i) == best_spt {
            tag.push_str(" MAX-SPT");
        }
        let cpg = if p.giants > 0 {
            format!("{:.1}", p.stars as f64 / p.giants as f64)
        } else {
            "—".into()
        };
        println!(
            "  {:<20}{:>8}{:>7}{:>8}{:>7}{:>10}  {:<26}{:<22}{}",
            p.scenario,
            p.stars,
            p.pop,
            p.giants,
            p.spt,
            cpg,
            hubs.join(" "),
            mkts.join(" "),
            tag
        );
    }
}
