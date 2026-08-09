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

/// Does this buy place the structure that feeds a hub? Conversions prefix the
/// name (`burn+Farm`), so the match is on the suffix.
fn is_partner_buy(b: &Buy, partner_name: &str) -> bool {
    b.what == partner_name || b.what.ends_with(partner_name)
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
    /// Techs this buy needs. Charged only when the buy is actually taken, so a
    /// plan that skips the Mountains never pays for Mining.
    techs: Vec<TechnologyType>,
}

#[derive(Clone, Copy, PartialEq)]
enum Lane {
    Forest,
    Farm,
    /// Forge eats Mines and pays 2 pop per partner — double the other hubs —
    /// but only a mountainous city can feed it.
    Mine,
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

const SCENARIOS: [Scenario; 8] = [
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

/// Techs a lane needs, in dependency order. Returned regardless of endowment;
/// the caller prices only the ones not already owned.
fn lane_chain(lane: Lane, convert: bool) -> Vec<TechnologyType> {
    // Derived from the structures the lane actually places, so adding a lane
    // cannot forget to price its techs. Everything else a build-out touches is
    // billed per-buy; only the terrain conversions are extra.
    let mut v = structure_techs(lane_partner_type(lane));
    v.extend(structure_techs(lane_hub(lane).0));
    if convert {
        match lane {
            // GrowForest: Field -> Forest. Its prerequisite is pulled in by
            // `tech_bill`, so only the ability tech is named.
            Lane::Forest => v.push(TechnologyType::Spiritualism),
            // BurnForest rides along with Construction, already in the chain.
            Lane::Farm | Lane::Mine => {}
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
    tech_bill_itemised(chain, owned, cities).0
}

/// The bill, plus the techs actually charged for — the build card needs to name
/// them, and deriving the list separately would let it drift from the price.
fn tech_bill_itemised(
    chain: &[TechnologyType],
    owned: &HashSet<TechnologyType>,
    cities: i32,
) -> (i32, Vec<TechnologyType>) {
    let mut bought = Vec::new();
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
            // Price each tech at ITS OWN tier. Billing prerequisites at the
            // target's tier over-charged every deep chain.
            bill += get_tech_cost(
                cities,
                polyfish::settings::technology::tech_tier(cur),
                false,
            );
            bought.push(cur);
            have.insert(cur);
        }
    }
    (bill, bought)
}

/// The tech a structure needs, read off the tech table via the engine's own
/// reverse lookup so this cannot drift from it.
fn structure_techs(s: StructureType) -> Vec<TechnologyType> {
    polyfish::settings::technology::get_tech_unlocking_structure(s)
        .into_iter()
        .collect()
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
                    buys.push(Buy {
                        idx,
                        // Named for the structure it ends up placing: the hub's
                        // partner match is on the suffix, and "grow+Hut" missed
                        // it, so grown huts fed no Sawmill.
                        what: "grow+LumberHut",
                        cost: GROW_FOREST_COST + 3,
                        pop: 1,
                        occupies: true,
                        techs: structure_techs(StructureType::LumberHut),
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
        Lane::Mine => (StructureType::Forge, "Mine"),
    }
}

/// What a city can build out to on a given tile set, under one scenario.
/// Shared by the real plan and the border-growth reachability check so the two
/// can never disagree about how much pop a tile set yields.
struct BuildOut {
    pop: i32,
    stars: i32,
    /// Techs the taken buys require — billed by `plan_city`, so a plan pays for
    /// the Mining or Fishing it leans on instead of assuming it for free.
    techs: HashSet<TechnologyType>,
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

/// Could this lane's hub ever stand here? A Forge needs two Mines beside one
/// tile, which most maps never offer. Scoring the lane anyway costs a full
/// allocation pass per city and returns the same hub-less build every time, so
/// the lane is gated on the terrain rather than evaluated and discarded.
fn lane_can_place_hub(state: &GameState, territory: &[i32], lane: Lane) -> bool {
    if lane != Lane::Mine {
        return true;
    }
    let metal: HashSet<i32> = territory
        .iter()
        .copied()
        .filter(|i| {
            state
                .resources
                .get(i)
                .and_then(|r| r.as_ref())
                .is_some_and(|r| r.resource_type == ResourceType::Metal)
        })
        .collect();
    territory.iter().any(|&t| {
        get_adjacent_indices(state, t, 1)
            .into_iter()
            .filter(|a| metal.contains(a))
            .count()
            >= 2
    })
}

/// The radius-2 square a city could ever rule, for cheap pre-checks that must
/// not depend on an allocation that has not happened yet.
fn city_square(state: &GameState, city: i32) -> Vec<i32> {
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
fn monuments_to_reach(
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
fn hub_candidates(state: &GameState, territory: &[i32], sc: Scenario, top_k: usize) -> Vec<i32> {
    let (buys, hub_sites, _) = tile_options(state, territory, sc);
    let (_, partner_name) = lane_hub(sc.lane);
    let partner_tiles: HashSet<i32> = buys
        .iter()
        .filter(|b| is_partner_buy(b, partner_name))
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
        .filter(|b| is_partner_buy(b, partner_name))
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
    let mut techs: HashSet<TechnologyType> = HashSet::new();
    for b in buys.iter().filter(|b| !displaced(b)) {
        stars += b.cost;
        pop += b.pop;
        techs.extend(b.techs.iter().copied());
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

/// How many hub sites to price before choosing. The `>= 2 partners` filter in
/// `hub_candidates` already bounds the list well under this on Tiny maps, so in
/// practice every surviving candidate is priced and the cap only guards a
/// pathologically large territory.
const HUB_TOP_K: usize = 32;

/// The tiles a city rules BEFORE BorderGrowth — the inner 3x3.
fn inner_ring(state: &GameState, city_idx: i32, territory: &[i32]) -> Vec<i32> {
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
fn site_reachable(
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

/// Single-city build: the best hub by what it actually delivers, market beside
/// it. Used by the allocator and the per-city table; the empire frontier
/// enumerates hubs jointly instead, because Market income couples the cities.
fn build_out(
    state: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    monuments: i32,
) -> BuildOut {
    // Rank by delivered pop then cost, not by partner count alone. Sites with
    // equal partner counts still differ in stars — siting on Forest refunds the
    // ClearForest star, and the hub displaces whatever that tile would have
    // built — so the old lowest-tile-index tie-break could take a site strictly
    // dominated by another paying the same pop for fewer stars. Unreachable
    // sites are dropped BEFORE ranking, or the cheapest plan is one the city
    // can never actually arrive at.
    let mut ranked: Vec<(i32, i32, i32)> = hub_candidates(state, territory, sc, HUB_TOP_K)
        .into_iter()
        .filter(|&h| site_reachable(state, city_idx, territory, sc, monuments, h))
        .map(|h| {
            let b = city_build(state, territory, sc, monuments, Some(h), None, None);
            (b.pop, b.stars, h)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let hub = ranked.first().map(|&(_, _, h)| h);
    let mut b = city_build(state, territory, sc, monuments, hub, None, None);
    if let Some(h) = hub {
        if b.partners > 0 {
            // Sites stopped being interchangeable once a Market may sit on a
            // tile the plan would otherwise harvest or farm, so price each one
            // instead of taking the lowest index.
            let partners = partner_tiles_of(state, territory, sc);
            let best = market_sites(state, territory, h, &partners)
                .into_iter()
                .map(|m| {
                    let cb = city_build(state, territory, sc, monuments, hub, Some(m), None);
                    (cb.pop, -cb.stars, -m)
                })
                .max();
            if let Some((_, _, neg)) = best {
                b = city_build(state, territory, sc, monuments, hub, Some(-neg), None);
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
fn market_site_legal(state: &GameState, site: i32, is_hub_partner: bool) -> bool {
    !is_hub_partner
        && state.tiles.get(&site).map(|t| t.terrain_type) == Some(TerrainType::Field)
        && polyfish::functions::get_structure_at(state, site).is_none()
}

/// Tiles this city's lane would cover with the hub's partner structure.
fn partner_tiles_of(state: &GameState, territory: &[i32], sc: Scenario) -> HashSet<i32> {
    let (_, partner_name) = lane_hub(sc.lane);
    let (buys, _, _) = tile_options(state, territory, sc);
    buys.iter()
        .filter(|b| is_partner_buy(b, partner_name))
        .map(|b| b.idx)
        .collect()
}

/// Tiles in `territory` where a Market is legal and useful — adjacent to `hub`,
/// and not one of the partners feeding it.
fn market_sites(
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

    let b = build_out(state, city_idx, territory, sc, monuments);
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

/// Assign every tile within radius 2 of any planned city to exactly one city —
/// nearest wins, ties to the earlier city — so overlapping 5x5s are not
/// double-counted.
/// Joint allocation by MARGINAL VALUE: a contested tile goes to whichever
/// claimant's plan gains more pop from it. Nearest-city was the first rule and
/// it was wrong — it handed the capital tiles it had no use for while capping a
/// neighbour's Sawmill a level below what the map supports.
/// The same scenario for every city — the pure (unmixed) empire.
fn uniform(sc: Scenario, n: usize) -> Vec<Scenario> {
    vec![sc; n]
}

fn allocate_value(
    state: &GameState,
    cities: &[i32],
    scs: &[Scenario],
    monuments: i32,
) -> Vec<Vec<i32>> {
    let mut claimants: HashMap<i32, Vec<usize>> = HashMap::new();
    for (ci, &c) in cities.iter().enumerate() {
        let radius = if scs[ci].border_growth { 2 } else { 1 };
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

    // An inner-ring tile is not contestable. `claim_territory` takes only
    // UNOWNED tiles, so a city's own 3x3 is its own from the moment it is
    // founded or captured, and a neighbour's BorderGrowth — which is a
    // level-4 reward, and so always later — can never take it back. Where two
    // inner rings overlap the earlier city wins, and `cities` is already in
    // capture order: capital first, then villages by distance.
    let mut open: Vec<(i32, Vec<usize>)> = Vec::new();
    for (idx, who) in contested {
        let inner = who
            .iter()
            .copied()
            .find(|&ci| get_chebyshev_distance(idx, cities[ci], state.settings.size) <= 1);
        match inner {
            Some(ci) => terr[ci].push(idx),
            None => open.push((idx, who)),
        }
    }

    for (idx, who) in open {
        // Rank by (pop gained, nearer city, lower index) — every term
        // deterministic, so the allocation is reproducible.
        let mut ranked: Vec<(i32, i32, usize)> = who
            .iter()
            .map(|&ci| {
                let mut with = terr[ci].clone();
                let base = build_out(state, cities[ci], &with, scs[ci], monuments).pop;
                with.push(idx);
                with.sort();
                let gain =
                    build_out(state, cities[ci], &with, scs[ci], monuments).pop - base;
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
    /// One scenario per city: a mixed empire can run the forest lane in a
    /// wooded city and the farm lane next door.
    scenarios: Vec<Scenario>,
    stars: i32,
    pop: i32,
    giants: i32,
    spt: i32,
    /// Per-city pop and level, and the techs the plan pays for — carried so a
    /// build card can be printed without re-deriving (and mis-deriving) them.
    city_pop: Vec<i32>,
    city_level: Vec<i32>,
    techs: Vec<TechnologyType>,
    /// Monuments this plan consumes, per city. Empire-wide and earned, so this
    /// is a scarcity cost the star total cannot express.
    monuments: Vec<i32>,
    hubs: Vec<Option<i32>>,
    levels: Vec<i32>,
    markets: Vec<Option<i32>>,
    market_income: Vec<i32>,
}

impl EmpirePlan {
    fn monuments_used(&self) -> i32 {
        self.monuments.iter().sum()
    }

    /// The scenario name when every city agrees, otherwise a per-city sketch.
    fn label(&self) -> String {
        let first = self.scenarios[0].name;
        if self.scenarios.iter().all(|s| s.name == first) {
            return first.to_string();
        }
        let parts: Vec<&str> = self.scenarios.iter().map(|s| sc_short(*s)).collect();
        format!("mixed {}", parts.join("/"))
    }
}

/// Compact scenario tag for mixed-empire rows: lane initial, +b for border,
/// +bc when it also converts terrain.
fn sc_short(sc: Scenario) -> &'static str {
    match (sc.lane, sc.border_growth, sc.convert) {
        (Lane::Forest, false, _) => "S",
        (Lane::Forest, true, false) => "S+b",
        (Lane::Forest, true, true) => "S+bc",
        (Lane::Farm, false, _) => "W",
        (Lane::Farm, true, false) => "W+b",
        (Lane::Farm, true, true) => "W+bc",
        (Lane::Mine, false, _) => "F",
        (Lane::Mine, true, _) => "F+b",
    }
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
    scs: &[Scenario],
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
    // Empire-wide partner tiles, keyed by the structure that will actually stand
    // there. With a mixed empire a tile holds whatever ITS OWN city's lane
    // builds, so a Sawmill on a border collects the neighbour's LumberHuts only
    // if that neighbour is running the forest lane.
    let mut partners_by_type: HashMap<StructureType, HashSet<i32>> = HashMap::new();
    for ci in 0..n {
        let (buys, _, _) = tile_options(state, &terr[ci], scs[ci]);
        let (_, partner_name) = lane_hub(scs[ci].lane);
        let ptype = lane_partner_type(scs[ci].lane);
        let e = partners_by_type.entry(ptype).or_default();
        for b in buys.iter().filter(|b| is_partner_buy(b, partner_name)) {
            e.insert(b.idx);
        }
    }
    let empty_partners: HashSet<i32> = HashSet::new();
    let partners_for = |ci: usize| -> &HashSet<i32> {
        partners_by_type
            .get(&lane_partner_type(scs[ci].lane))
            .unwrap_or(&empty_partners)
    };
    let _ = &union;

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
                            || (scs[ci].lane == Lane::Forest
                                && tile.terrain_type == TerrainType::Forest))
                })
                .map(|t| {
                    let n = get_adjacent_indices(state, t, 1)
                        .into_iter()
                        .filter(|a| partners_for(ci).contains(a) && *a != t)
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
    let mut seen_score: HashSet<(i32, i32, i32, i32)> = HashSet::new();
    for hubs in combos {
        // A tile holds ONE structure. `city_build` drops this city's own hub and
        // market from its partner count, but the empire-wide set still offered
        // every OTHER city's hub tile — so a Sawmill next to a neighbour's
        // Sawmill counted it as a LumberHut, inflating both hub levels and the
        // Market income read off them.
        let hub_tiles: HashSet<i32> = hubs.iter().flatten().copied().collect();
        let counted: Vec<HashSet<i32>> = (0..n)
            .map(|ci| partners_for(ci).difference(&hub_tiles).copied().collect())
            .collect();
        let base: Vec<BuildOut> = (0..n)
            .map(|ci| {
                city_build(
                    state, &terr[ci], scs[ci], monuments, hubs[ci], None,
                    Some(&counted[ci]),
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
                        let adj = get_adjacent_indices(state, *a, 1);
                        // Feeding ANY adjacent hub disqualifies the tile, whichever
                        // city that hub belongs to — taking it would drop the level
                        // stage 1 already priced, and every Market reading it.
                        let feeds = (0..n).any(|cj| {
                            hubs[cj].is_some_and(|h| adj.contains(&h))
                                && partners_for(cj).contains(a)
                        });
                        !hubs.iter().any(|h| *h == Some(*a))
                            && market_site_legal(state, *a, feeds)
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

        // STAGE 3 — allocate monuments, cost it, route pop to cities, score.
        //
        // Monuments are empire-wide, scarce and earned — none exist at turn 0 —
        // so they are allocated from one budget rather than handed to every
        // city. Reachability comes first: a city whose border is otherwise
        // unreachable must be given its minimum or the whole combo dies. The
        // remainder goes greedily to the best marginal pop, which is optimal
        // because `place_monuments` takes lowest-loss tiles first, making the
        // gain per extra monument non-increasing.
        let mut floor = vec![0i32; n];
        let mut feasible = true;
        for ci in 0..n {
            match monuments_to_reach(state, cities[ci], &terr[ci], scs[ci], hubs[ci], monuments)
            {
                Some(m) => floor[ci] = m,
                None => {
                    feasible = false;
                    break;
                }
            }
        }
        let need: i32 = floor.iter().sum();
        if !feasible || need > monuments {
            continue;
        }

        // Emit a plan at EVERY affordable monument count, not just the full
        // budget. Spending is always weakly pop-positive, so a greedy that
        // always drains the budget would stamp the same count on every plan and
        // the scarcity axis would compare nothing. Letting the count vary is
        // what lets a 0-monument plan dominate a 3-monument one that bought
        // nothing with them.
        for m_total in need..=monuments {
        let mut alloc = floor.clone();
        let mut left = m_total - need;
        while left > 0 {
            let best = (0..n)
                .filter_map(|ci| {
                    let at = |m: i32| {
                        city_build(
                            state, &terr[ci], scs[ci], m, hubs[ci], markets[ci],
                            Some(&counted[ci]),
                        )
                        .pop
                    };
                    let gain = at(alloc[ci] + 1) - at(alloc[ci]);
                    (gain > 0).then_some((gain, ci))
                })
                .max();
            match best {
                Some((_, ci)) => {
                    alloc[ci] += 1;
                    left -= 1;
                }
                None => break,
            }
        }
        // A count the greedy could not actually place is the same plan as the
        // one below it; skip so the dedupe key does not fragment.
        if alloc.iter().sum::<i32>() != m_total {
            continue;
        }

        let mut stars = 0;
        let mut plan_techs: HashSet<TechnologyType> = HashSet::new();
        let (mut pop, mut giants, mut spt) = (0, 0, 0);
        let mut city_pop_v = vec![0; n];
        let mut city_level_v = vec![0; n];
        for ci in 0..n {
            let b = city_build(
                state, &terr[ci], scs[ci], alloc[ci], hubs[ci], markets[ci],
                Some(&counted[ci]),
            );
            let mut city_pop = b.pop;
            if !scs[ci].border_growth && city_pop >= POP_FOR_LEVEL_4 {
                city_pop += 3; // PopGrowth, the alternative to the border
            }
            let level = level_at_pop(city_pop);
            city_pop_v[ci] = city_pop;
            city_level_v[ci] = level;
            let is_capital =
                state.tiles.get(&cities[ci]).map_or(false, |t| t.capital_of != 0);
            stars += b.stars;
            plan_techs.extend(b.techs.iter().copied());
            pop += city_pop;
            giants += giants_at_level(level);
            spt += level + i32::from(is_capital) + i32::from(level >= 2) + income[ci];
        }
        // Techs are empire-wide, so the union across cities is billed once.
        let mut chain: Vec<TechnologyType> = (0..n)
            .flat_map(|ci| lane_chain(scs[ci].lane, scs[ci].convert))
            .collect();
        let mut extra: Vec<TechnologyType> = plan_techs.into_iter().collect();
        extra.sort_by_key(|t| *t as i32);
        chain.extend(extra);
        if markets.iter().any(|m| m.is_some()) {
            chain.extend(market_chain());
        }
        let (bill, bought) = tech_bill_itemised(&chain, owned, num_cities);
        stars += bill;
        if !seen_score.insert((stars, giants, spt, alloc.iter().sum::<i32>())) {
            continue;
        }
        out.push(EmpirePlan {
            scenarios: scs.to_vec(),
            stars,
            pop,
            giants,
            spt,
            city_pop: city_pop_v,
            city_level: city_level_v,
            techs: bought,
            monuments: alloc,
            hubs: hubs.clone(),
            levels: levels.clone(),
            markets: markets.clone(),
            market_income: income.clone(),
        });
        }
    }
    out
}
/// Does `q` dominate `p`? Cheaper is better, more giants and more SPT are
/// better, and FEWER monuments is better — they are earned, not bought, so a
/// plan reaching the same output without burning one is strictly preferable.
fn dominates(q: &EmpirePlan, p: &EmpirePlan) -> bool {
    q.stars <= p.stars
        && q.giants >= p.giants
        && q.spt >= p.spt
        && q.monuments_used() <= p.monuments_used()
        && (q.stars < p.stars
            || q.giants > p.giants
            || q.spt > p.spt
            || q.monuments_used() < p.monuments_used())
}

/// Add a plan to a running frontier, dropping whatever it dominates.
///
/// Incremental rather than a final O(n^2) sweep: widening the search to
/// per-city lanes multiplies the candidate count by orders of magnitude, and
/// the frontier itself stays small (tens of plans), so this is the difference
/// between linear and quadratic in the thing that grows.
fn frontier_insert(front: &mut Vec<EmpirePlan>, p: EmpirePlan) {
    if front.iter().any(|q| dominates(q, &p)) {
        return;
    }
    front.retain(|q| !dominates(&p, q));
    front.push(p);
}

fn pareto(plans: &[EmpirePlan]) -> Vec<EmpirePlan> {
    let mut front: Vec<EmpirePlan> = Vec::new();
    for p in plans {
        frontier_insert(&mut front, p.clone());
    }
    front
}

/// Tile-by-tile dump for one city+scenario, so a number in the table can be
/// traced back to the map.
fn explain(state: &GameState, city_idx: i32, territory: &[i32], sc: Scenario) {
    let (buys, hub_sites, _) = tile_options(state, territory, sc);
    let (_, partner_name) = lane_hub(sc.lane);
    let partner_tiles: HashSet<i32> = buys
        .iter()
        .filter(|b| is_partner_buy(b, partner_name))
        .map(|b| b.idx)
        .collect();
    let b = build_out(state, city_idx, territory, sc, 0);
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

/// The partner structure each lane actually builds on a tile.
fn lane_partner_type(lane: Lane) -> StructureType {
    match lane {
        Lane::Forest => StructureType::LumberHut,
        Lane::Farm => StructureType::Farm,
        Lane::Mine => StructureType::Mine,
    }
}

/// Own `territory` outright for player 1 and hand them stars, so the engine's
/// own build path can be run against the planner's tile set.
fn owned_board(base: &GameState, city_idx: i32, territory: &[i32]) -> GameState {
    let mut s = base.clone();
    let size = s.settings.size;
    s.settings.current_player_turn_id = 1;
    if let Some(t) = s.tribes.get_mut(&1) {
        t.stars = 100_000;
    }
    for &i in territory {
        if let Some(t) = s.tiles.get_mut(&i) {
            t.owner = 1;
            t.ruling_city_coords = Some(polyfish::coords::Coords::from_index(city_idx, size));
        }
    }
    s
}

/// Run the plan's builds through the engine in `order`, returning the finished
/// board and the population the city actually gained.
fn materialize(
    base: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    hub: Option<i32>,
    order: &[usize],
) -> (GameState, i32) {
    let mut s = owned_board(base, city_idx, territory);
    let (buys, _, _) = tile_options(&s, territory, sc);
    let (hub_type, partner_name) = lane_hub(sc.lane);

    let mut plan: Vec<(i32, StructureType)> = buys
        .iter()
        .filter(|b| is_partner_buy(b, partner_name))
        .map(|b| b.idx)
        .filter(|i| Some(*i) != hub)
        .map(|i| (i, lane_partner_type(sc.lane)))
        .collect();
    if let Some(h) = hub {
        plan.push((h, hub_type));
    }

    let pop_of = |s: &GameState| -> i32 {
        s.tribes
            .get(&1)
            .and_then(|t| t.cities.iter().find(|c| c.idx == city_idx))
            .map_or(0, |c| c.population)
    };
    let before = pop_of(&s);
    for &k in order {
        let (idx, st) = plan[k];
        let _ = polyfish::actions::structure::build_structure(&mut s, idx, st);
    }
    let gained = pop_of(&s) - before;
    (s, gained)
}

/// Number of builds in a plan, so orders can be generated without materializing.
fn plan_len(state: &GameState, territory: &[i32], sc: Scenario, hub: Option<i32>) -> usize {
    let (buys, _, _) = tile_options(state, territory, sc);
    let (_, partner_name) = lane_hub(sc.lane);
    buys.iter()
        .filter(|b| is_partner_buy(b, partner_name))
        .map(|b| b.idx)
        .filter(|i| Some(*i) != hub)
        .count()
        + usize::from(hub.is_some())
}

/// Check the planner's placement claim against the engine rather than against
/// a second copy of the planner. Per city and scenario:
///   1. the hub's engine partner count equals the claimed level;
///   2. the hub's stored level equals it too (the UI and Market read that);
///   3. pop is a function of the built SET — build order changes nothing;
///   4. the chosen site is argmax of engine partner count over every empty
///      owned alternative on the same finished board.
/// Returns false if any check failed.
fn verify(state: &GameState, cities: &[i32], monuments: i32) -> bool {
    let mut failures = 0;
    let capital = cities[0];

    println!("\n{}", "=".repeat(96));
    println!("  VERIFY — planner claims vs engine, seed fixture");
    println!("{}", "=".repeat(96));
    println!(
        "  {:<20} {:>5} {:>6} {:>8} {:>8} {:>9} {:>9}",
        "scenario", "city", "hub", "claimed", "engine", "st.level", "argmax"
    );
    println!("  (pop = capital population delivered by the engine, identical across 3 build orders)");
    println!("  {}", "-".repeat(92));

    for sc in SCENARIOS {
        let terr = allocate_value(state, cities, &uniform(sc, cities.len()), monuments);
        for (ci, &city_idx) in cities.iter().enumerate() {
            // Monuments are excluded: they displace tiles but are not the
            // placement decision under test.
            let b = build_out(state, city_idx, &terr[ci], sc, 0);
            let Some(hub) = b.hub_site else {
                println!(
                    "  {:<20} {:>5} {:>6} {:>8} {:>8} {:>9} {:>9}",
                    sc.name, city_idx, "—", "—", "—", "—", "—"
                );
                continue;
            };
            let (hub_type, _) = lane_hub(sc.lane);
            let n = plan_len(state, &terr[ci], sc, Some(hub));
            let ident: Vec<usize> = (0..n).collect();
            let (board, pop_ident) = materialize(state, city_idx, &terr[ci], sc, Some(hub), &ident);

            let engine_partners =
                polyfish::rules::economy::partner_count(&board, hub, hub_type, 1);
            let stored_level = polyfish::functions::get_structure_at(&board, hub)
                .map_or(-1, |s| s.level);

            // 3. Build-order invariance, on the capital only — the other cities
            //    are villages here, so no CityState collects their pop.
            let mut order_ok = true;
            if city_idx == capital {
                let mut rev = ident.clone();
                rev.reverse();
                // Hub first, then partners in order: the retroactive-pay path.
                let mut hub_first = vec![n - 1];
                hub_first.extend(0..n - 1);
                for alt in [rev, hub_first] {
                    let (_, pop_alt) =
                        materialize(state, city_idx, &terr[ci], sc, Some(hub), &alt);
                    if pop_alt != pop_ident {
                        order_ok = false;
                        println!(
                            "    ORDER-DEPENDENT: {} city {city_idx} pop {pop_ident} -> {pop_alt}",
                            sc.name
                        );
                    }
                }
            }

            // 4. Argmax over alternatives on the finished board. Candidates are
            //    the planner's own site space (Field, plus Forest for the forest
            //    lane, which ClearForest turns into Field) narrowed to tiles
            //    still EMPTY once the plan is built — moving the hub onto an
            //    occupied tile would displace a partner and change the set.
            let (_, site_space, _) = tile_options(state, &terr[ci], sc);
            // The site space is the planner's own, so guard it against the
            // engine's terrain whitelist — otherwise argmax could pass by
            // simply never offering the better tile.
            let legal_terrain = &get_structure_setting(hub_type).terrain_types;
            let uncovered: Vec<i32> = terr[ci]
                .iter()
                .copied()
                .filter(|i| {
                    polyfish::functions::get_structure_at(state, *i).is_none()
                        && state
                            .tiles
                            .get(i)
                            .is_some_and(|t| legal_terrain.contains(&t.terrain_type))
                        && !site_space.contains(i)
                })
                .collect();
            if !uncovered.is_empty() {
                failures += 1;
                println!(
                    "    SITE-SPACE GAP: {} city {city_idx} omits legal tiles {uncovered:?}",
                    sc.name
                );
            }
            // Optimality is judged on the objective the planner actually
            // optimizes — pop against stars — not on partner count, which is
            // only a proxy and one the planner deliberately trades away.
            let doms = dominators_of(state, city_idx, &terr[ci], sc, hub);
            let argmax_ok = doms.is_empty();
            let (best_site, best_n) = doms.first().map_or((hub, engine_partners), |&(p, _, s)| (s, p));

            let level_ok = stored_level == b.partners;
            let count_ok = engine_partners == b.partners;
            let ok = level_ok && count_ok && argmax_ok && order_ok;
            if !ok {
                failures += 1;
            }
            println!(
                "  {:<20} {:>5} {:>6} {:>8} {:>8} {:>9} {:>9}  {}",
                sc.name,
                city_idx,
                hub,
                b.partners,
                engine_partners,
                stored_level,
                if argmax_ok { "yes".to_string() } else { format!("{best_site}@{best_n}") },
                if ok { "ok" } else { "FAIL" }
            );
            if city_idx == capital {
                println!("      builds {n}, pop delivered {pop_ident}, order-invariant {order_ok}");
            }
        }
    }

    println!("\n  {} failing (city, scenario) pairs", failures);
    failures == 0
}

/// Sites that DOMINATE `chosen`: at least as much pop for no more stars, and
/// strictly better on one. More pop for more stars is a different point on the
/// frontier, not a mistake, so it is not a dominator.
fn dominators_of(
    state: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    chosen: i32,
) -> Vec<(i32, i32, i32)> {
    let (_, site_space, _) = tile_options(state, territory, sc);
    let score = |s: i32| {
        let b = city_build(state, territory, sc, 0, Some(s), None, None);
        (b.pop, b.stars)
    };
    let (cp, cs) = score(chosen);
    let mut out: Vec<(i32, i32, i32)> = site_space
        .iter()
        .copied()
        .filter(|&s| s != chosen)
        // An unreachable site cannot dominate anything — the city never gets there.
        .filter(|&s| site_reachable(state, city_idx, territory, sc, 0, s))
        .filter_map(|s| {
            let (p, st) = score(s);
            (p >= cp && st <= cs && (p > cp || st < cs)).then_some((p, st, s))
        })
        .collect();
    out.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    out
}

/// Is the greedy hub pick actually the best site available?
///
/// `hub_candidates` ranks sites by POTENTIAL partner count and `build_out` takes
/// the top one. That is not the same objective as population: a site can touch
/// more partners yet displace a better tile, and siting on Forest refunds a star
/// via ClearForest. So this scores EVERY candidate with the planner's own
/// `city_build` and ranks, to see whether greedy and exhaustive agree.
fn optimal_report(state: &GameState, cities: &[i32], monuments: i32, lane: Lane) {
    let hub_type = lane_hub(lane).0;
    for sc in SCENARIOS.iter().filter(|s| s.lane == lane) {
        let terr = allocate_value(state, cities, &uniform(*sc, cities.len()), monuments);
        for (ci, &city_idx) in cities.iter().enumerate() {
            let chosen = build_out(state, city_idx, &terr[ci], *sc, 0).hub_site;
            let (_, site_space, _) = tile_options(state, &terr[ci], *sc);

            // (pop, -stars, site) so the sort puts most pop, then cheapest, first.
            let mut ranked: Vec<(i32, i32, i32, i32)> = site_space
                .iter()
                .filter(|&&s| site_reachable(state, city_idx, &terr[ci], *sc, 0, s))
                .map(|&s| {
                    let b = city_build(state, &terr[ci], *sc, 0, Some(s), None, None);
                    (b.pop, -b.stars, s, b.partners)
                })
                .collect();
            ranked.sort_by(|a, b| b.cmp(a));

            let blocked: Vec<i32> = site_space
                .iter()
                .copied()
                .filter(|&s| !site_reachable(state, city_idx, &terr[ci], *sc, 0, s))
                .collect();

            let no_hub = city_build(state, &terr[ci], *sc, 0, None, None, None);
            let best = ranked.first().copied();

            println!("\n  {} — city {city_idx} ({} candidate sites)", sc.name, site_space.len());
            println!("      no hub at all:            pop {:>3}  stars {:>4}", no_hub.pop, no_hub.stars);
            for (pop, negstars, site, partners) in ranked.iter().take(5) {
                let mark = if Some(*site) == chosen { "  <- GREEDY PICK" } else { "" };
                let ring = if get_chebyshev_distance(*site, city_idx, state.settings.size) > 1 {
                    "outer"
                } else {
                    "inner"
                };
                println!(
                    "      site {site:>3} ({partners} partners, {ring}):  pop {pop:>3}  stars {:>4}{mark}",
                    -negstars
                );
            }
            if !blocked.is_empty() {
                println!(
                    "      unreachable ({} sites): {blocked:?} — inner ring cannot hit {POP_FOR_LEVEL_4} pop without the hub",
                    blocked.len()
                );
            }
            // A site only counts as better if it DOMINATES: at least as much pop
            // for no more stars, strictly better on one. More pop for more stars
            // is a different point on the frontier, not a mistake.
            match chosen {
                Some(c) => match ranked.iter().find(|r| r.2 == c).copied() {
                    None => println!("      VERDICT: chosen site {c} is not in the candidate space"),
                    Some((cp, cs, _, _)) => {
                        let dominators: Vec<(i32, i32, i32)> = ranked
                            .iter()
                            .filter(|&&(p, s, site, _)| {
                                site != c && p >= cp && s >= cs && (p > cp || s > cs)
                            })
                            .map(|&(p, s, site, _)| (p, -s, site))
                            .collect();
                        if dominators.is_empty() {
                            println!("      VERDICT: greedy pick is on the frontier (not dominated)");
                        } else {
                            for (p, s, site) in &dominators {
                                println!(
                                    "      VERDICT: DOMINATED — site {site} gives pop {p} for {s} stars; chosen {c} gives pop {cp} for {} stars",
                                    -cs
                                );
                            }
                        }
                    }
                },
                None if best.is_some_and(|(bp, _, _, _)| bp > no_hub.pop) => {
                    let (bp, _, bsite, _) = best.unwrap();
                    println!("      VERDICT: NO HUB CHOSEN but site {bsite} would pay pop {bp}");
                }
                None => println!("      VERDICT: no hub, none worthwhile"),
            }

            // Confirm the winner against the engine, not just the planner's model.
            if let Some((_, _, bsite, bpartners)) = best {
                let n = plan_len(state, &terr[ci], *sc, Some(bsite));
                let order: Vec<usize> = (0..n).collect();
                let (board, _) = materialize(state, city_idx, &terr[ci], *sc, Some(bsite), &order);
                let engine = polyfish::rules::economy::partner_count(&board, bsite, hub_type, 1);
                if engine != bpartners {
                    println!("      ENGINE DISAGREES on site {bsite}: planner {bpartners}, engine {engine}");
                }
            }
        }
    }
}


/// What a plan is optimised FOR. The frontier answers "what are the options";
/// these answer "give me the build for what I need right now".
/// Two ceilings and three knees. The ceilings say what the map can do at any
/// price; the knees say what it is worth paying for, on income, on army, and on
/// the two together.
#[derive(Clone, Copy, PartialEq)]
enum Goal {
    Spt,
    Eco,
    Balanced,
    Army,
    Giants,
}

/// Frontier maxima for SPT, super units, stars and monuments — the scale every
/// knee is measured against. Never zero, so they are safe denominators.
fn frontier_maxima(front: &[EmpirePlan]) -> (i64, i64, i64, i64) {
    let m = |v: i64| v.max(1);
    (
        m(front.iter().map(|p| p.spt).max().unwrap_or(1) as i64),
        m(front.iter().map(|p| p.giants).max().unwrap_or(1) as i64),
        m(front.iter().map(|p| p.stars).max().unwrap_or(1) as i64),
        m(front.iter().map(|p| p.monuments_used()).max().unwrap_or(1) as i64),
    )
}

/// The knee: value minus cost, with every axis normalised to [0,1] across the
/// frontier.
///
/// On normalised axes the chord from the cheapest plan to the richest IS the
/// diagonal, so the maximum of (value - cost) is the point furthest above that
/// chord — the elbow, with no hand-tuned exchange rate. `w_spt`/`w_su` choose
/// what "value" means: (1,0) is income alone, (0,1) army alone, (1,1) both.
///
/// COST IS BOTH CURRENCIES. Monuments were only a tiebreak, so a picker handed
/// a budget of three always spent three if it gained anything at all, however
/// marginal — it would take the border for one city over a plan four stars
/// cheaper that kept all three in the bank. Monuments are earned, not bought,
/// so they carry a full axis of cost: spend one only when it buys a lot.
/// Everything is scaled by the common denominator so ties stay exact.
fn knee_score(p: &EmpirePlan, w_spt: i64, w_su: i64, (ms, mg, mc, mm): (i64, i64, i64, i64)) -> i64 {
    let w = w_spt + w_su;
    let value = mm * (w_spt * p.spt as i64 * mg * mc + w_su * p.giants as i64 * ms * mc);
    let star_cost = mm * w * p.stars as i64 * ms * mg;
    let monument_cost = w * p.monuments_used() as i64 * ms * mg * mc;
    value - star_cost - monument_cost
}

fn parse_goal(s: &str) -> Option<Goal> {
    match s.trim().to_lowercase().as_str() {
        "spt" | "greed" => Some(Goal::Spt),
        "army" | "military" | "war" => Some(Goal::Army),
        "giants" | "su" => Some(Goal::Giants),
        "balanced" | "middle" | "mid" => Some(Goal::Balanced),
        "eco" | "efficiency" => Some(Goal::Eco),
        _ => None,
    }
}

fn goal_name(g: Goal) -> &'static str {
    match g {
        Goal::Spt => "MAX SPT (ceiling on income, cost no object)",
        Goal::Eco => "ECONOMY (knee: the most income the stars are worth)",
        Goal::Balanced => "BALANCED (knee: income and army weighted evenly)",
        Goal::Army => "ARMY (knee: the most super units the stars are worth)",
        Goal::Giants => "MAX SUPER UNITS (ceiling, cost no object)",
    }
}

/// Short tag for the frontier listing, so a row says which goal claims it.
fn goal_tag(g: Goal) -> &'static str {
    match g {
        Goal::Spt => " MAX-SPT",
        Goal::Eco => " ECO",
        Goal::Balanced => " BALANCED",
        Goal::Army => " ARMY",
        Goal::Giants => " MAX-GIANTS",
    }
}

const GOALS: [Goal; 5] = [Goal::Spt, Goal::Eco, Goal::Balanced, Goal::Army, Goal::Giants];

/// Pick the plan that best serves `g`. Every ordering ends with fewer stars and
/// then fewer monuments, so ties resolve toward the plan that spends least.
fn pick_for_goal<'a>(front: &'a [EmpirePlan], g: Goal) -> Option<&'a EmpirePlan> {
    let thrift = |p: &EmpirePlan| (-p.stars, -p.monuments_used());
    let m = frontier_maxima(front);
    front.iter().max_by(|a, b| {
        let key = |p: &EmpirePlan| match g {
            Goal::Spt => (p.spt as i64, p.giants as i64),
            Goal::Giants => (p.giants as i64, p.spt as i64),
            Goal::Eco => (knee_score(p, 1, 0, m), p.spt as i64),
            Goal::Balanced => (knee_score(p, 1, 1, m), p.spt as i64),
            Goal::Army => (knee_score(p, 0, 1, m), p.giants as i64),
        };
        key(a).cmp(&key(b)).then(thrift(a).cmp(&thrift(b)))
    })
}

/// Terrain, resource and territory owner per tile — the ground truth every
/// claim about "tile 14 is a Fruit field" has to be checked against.
fn print_map(state: &GameState, cities: &[i32], terr: &[Vec<i32>], sc: Scenario) {
    let size = state.settings.size;
    println!("\n  MAP — terrain/resource, and the city each tile is allocated to");
    println!("  terrain F=field f=forest M=mountain W=water    resource C=crop R=fruit G=game E=metal");
    println!("  allocation is for '{}', which is what sets each city's territory\n", sc.name);
    print!("        ");
    for x in 0..size {
        print!("{:>10}", format!("x{x}"));
    }
    println!();
    for y in 0..size {
        print!("   y{y:<3} ");
        for x in 0..size {
            let idx = y * size + x;
            let t = match state.tiles.get(&idx).map(|t| t.terrain_type) {
                Some(TerrainType::Field) => "F",
                Some(TerrainType::Forest) => "f",
                Some(TerrainType::Mountain) => "M",
                Some(TerrainType::Water) | Some(TerrainType::Ocean) => "W",
                _ => "?",
            };
            let r = match state
                .resources
                .get(&idx)
                .and_then(|r| r.as_ref())
                .map(|r| r.resource_type)
            {
                Some(ResourceType::Crop) => "C",
                Some(ResourceType::Fruit) => "R",
                Some(ResourceType::Game) => "G",
                Some(ResourceType::Metal) => "E",
                Some(_) => "x",
                None => ".",
            };
            let s = polyfish::functions::get_structure_at(state, idx)
                .map(|s| if s.structure_type == StructureType::Village { "v" } else { "s" })
                .unwrap_or(" ");
            let owner = terr
                .iter()
                .position(|t| t.contains(&idx))
                .map(|ci| format!("c{}", cities[ci]))
                .unwrap_or_else(|| "  ".into());
            print!("{:>10}", format!("{idx}:{t}{r}{s}{owner}"));
        }
        println!();
    }
    println!("\n  cities {cities:?}; a tile with no cN is outside every planned territory");
}

/// The full build for one plan: what to put where, what it costs, what it pays.
fn print_build_card(
    state: &GameState,
    cities: &[i32],
    plan: &EmpirePlan,
    monuments: i32,
    g: Goal,
) {
    let terr = allocate_value(state, cities, &plan.scenarios, monuments);

    println!("\n{}", "=".repeat(112));
    println!("  BUILD FOR: {}", goal_name(g));
    println!("{}", "=".repeat(112));
    println!(
        "  strategy {} | {} stars | pop {} | level-driven giants {} | {} SPT | {} monument(s)",
        plan.label(),
        plan.stars,
        plan.pop,
        plan.giants,
        plan.spt,
        plan.monuments_used()
    );
    if plan.techs.is_empty() {
        println!("  techs to buy: none (already owned)");
    } else {
        let names: Vec<String> = plan.techs.iter().map(|t| format!("{t:?}")).collect();
        println!("  techs to buy: {}", names.join(", "));
    }

    for (ci, &c) in cities.iter().enumerate() {
        let sc = plan.scenarios[ci];
        let (_, partner_name) = lane_hub(sc.lane);
        let hub_type = lane_hub(sc.lane).0;
        let (buys, _, _) = tile_options(state, &terr[ci], sc);
        let hub = plan.hubs[ci];
        let market = plan.markets[ci];
        let displaced = |b: &Buy| b.occupies && (Some(b.idx) == hub || Some(b.idx) == market);
        let mut by_kind: std::collections::BTreeMap<&str, Vec<i32>> = Default::default();
        for b in buys.iter().filter(|b| !displaced(b)) {
            by_kind.entry(b.what).or_default().push(b.idx);
        }
        println!(
            "\n  city {c} — {} — pop {} (level {}), {} monument(s)",
            sc.name, plan.city_pop[ci], plan.city_level[ci], plan.monuments[ci]
        );
        match hub {
            Some(h) => {
                let ring = if get_chebyshev_distance(h, c, state.settings.size) > 1 {
                    "outer ring — needs BorderGrowth first"
                } else {
                    "inner ring"
                };
                println!(
                    "    {:?} at {h}, level {} ({ring})",
                    hub_type, plan.levels[ci]
                );
            }
            None => println!("    no {hub_type:?}"),
        }
        if let Some(m) = market {
            println!("    Market at {m}, +{} SPT", plan.market_income[ci]);
        }
        for (kind, tiles) in by_kind {
            let tag = if kind.ends_with(partner_name) { " (feeds the hub)" } else { "" };
            println!("    {kind:<12} x{:<3} {tiles:?}{tag}", tiles.len());
        }
    }
    println!(
        "\n  Note: \"giants\" counts super-unit reward slots (every city level >= 4), the only\n  military output this planner models. It prices the ECONOMY that funds an army,\n  not the army itself — read SPT as the sustain rate for unit production."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned()
    };
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            r#"eco_plan — deterministic economy planner and its own ground truth.

  --seed N            map seed (default 4102)
  --cities N          cities to plan for, capital first (default 3)
  --techs a,b,c       techs already owned (default organization)
  --monuments N       monuments the EMPIRE ever earns (default 0 — you hold
                      none at turn 0, and each one costs a full axis in the
                      knees, so a plan has to earn the ones it spends)
  --no-markets        plan without Markets
  --standalone        score each city on its full square, ignoring neighbours
  --no-mix            one lane for the whole empire (default mixes per city)

  --goal WHICH        print the full build for a stated need. Two ceilings and
                      three knees; a knee is value minus cost with SPT, super
                      units and stars each normalised over the frontier, so it
                      is the elbow with no hand-tuned exchange rate:
                        spt       ceiling on stars/turn, cost no object
                        eco       knee on income alone
                        balanced  knee on income and army weighted evenly
                        army      knee on super units alone
                        giants    ceiling on super units, cost no object

  --explain CITY      per-scenario reasoning for one city (0-based)
  --map               print terrain, resources and the territory split
  --verify            check placements against the engine (exit 1 on failure)
  --optimal           rank every hub site; --windmill / --forge for other lanes
"#
        );
        return;
    }
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
    // Empire-wide, not per city: monuments are earned from tasks, so a tribe
    // holds none at turn 0 and only ever has a handful. The frontier decides
    // which city each one goes to.
    // Default NONE: a tribe holds no monument at turn 0, so the turn-0 truth is
    // the honest headline and monument-funded plans are opted into.
    let monuments: i32 = get("--monuments").and_then(|s| s.parse().ok()).unwrap_or(0);
    let standalone = args.iter().any(|a| a == "--standalone");
    let goal = get("--goal").and_then(|g| {
        let parsed = parse_goal(&g);
        if parsed.is_none() {
            eprintln!(
                "unknown --goal '{g}'; use spt | eco | balanced | army | giants"
            );
        }
        parsed
    });
    let with_markets = !args.iter().any(|a| a == "--no-markets");
    let mix = !args.iter().any(|a| a == "--no-mix");
    if standalone {
        println!("STANDALONE: each city scored on its full square (tiles double-counted)");
    }
    if args.iter().any(|a| a == "--map") {
        let sc = SCENARIOS[1]; // sawmill +border: full radius-2 territory
        let terr = allocate_value(&state, &cities, &uniform(sc, cities.len()), monuments);
        print_map(&state, &cities, &terr, sc);
        return;
    }
    if args.iter().any(|a| a == "--optimal") {
        let lane = if args.iter().any(|a| a == "--windmill") {
            Lane::Farm
        } else if args.iter().any(|a| a == "--forge") {
            Lane::Mine
        } else {
            Lane::Forest
        };
        optimal_report(&state, &cities, monuments, lane);
        return;
    }
    if args.iter().any(|a| a == "--verify") {
        std::process::exit(i32::from(!verify(&state, &cities, monuments)));
    }
    if let Some(which) = get("--explain") {
        let ci: usize = which.parse().unwrap_or(0);
        for sc in SCENARIOS {
            let terr = if standalone {
                allocate_mode(&state, &cities, sc.border_growth, true)
            } else {
                allocate_value(&state, &cities, &uniform(sc, cities.len()), monuments)
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
    println!(
        "monuments available to the empire: {monuments}  (3 pop each, earned from tasks — none at turn 0)"
    );

    let mut all: Vec<(i32, CityPlan)> = Vec::new();
    for sc in SCENARIOS {
        let terr = if standalone {
            allocate_mode(&state, &cities, sc.border_growth, true)
        } else {
            allocate_value(&state, &cities, &uniform(sc, cities.len()), monuments)
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
        if !cities
            .iter()
            .any(|&c| lane_can_place_hub(&state, &city_square(&state, c), sc.lane))
        {
            continue;
        }
        let terr = if standalone {
            allocate_mode(&state, &cities, sc.border_growth, true)
        } else {
            allocate_value(&state, &cities, &uniform(sc, cities.len()), monuments)
        };
        all_plans.extend(enumerate_empire(
            &state,
            &cities,
            &terr,
            &uniform(sc, cities.len()),
            &owned,
            monuments,
            8,
            with_markets,
        ));
    }

    // ---- Mixed empires: each city picks its own lane. A wooded city should
    // ---- sawmill while its neighbour farms, and the pure-scenario sweep above
    // ---- can never express that. Mixed plans are ADDED to the pure ones, so
    // ---- widening the search can only find better plans, never lose known ones.
    if mix {
        // Trim first: for each city, drop any scenario another scenario beats on
        // every axis that matters in isolation — no more stars, at least as much
        // pop, at least as good a hub. Only market coupling can rescue such a
        // scenario, and that needs hub level, which is one of the axes kept.
        let mut per_city: Vec<Vec<Scenario>> = Vec::new();
        let mut dropped = 0usize;
        for (ci, &c) in cities.iter().enumerate() {
            let square = city_square(&state, c);
            let scored: Vec<(Scenario, i32, i32, i32)> = SCENARIOS
                .iter()
                .filter(|sc| lane_can_place_hub(&state, &square, sc.lane))
                .map(|&sc| {
                    let terr = allocate_value(&state, &cities, &uniform(sc, cities.len()), monuments);
                    let b = build_out(&state, c, &terr[ci], sc, monuments);
                    (sc, b.stars, b.pop, b.partners)
                })
                .collect();
            // Collapse exact duplicates first. A lane that cannot place its hub
            // here — Forge with fewer than two adjacent Mines, say — produces
            // the identical hub-less build as every other inert lane, and
            // dominance never separates equals, so all of them would survive
            // and multiply the assignment space for nothing.
            let mut seen: HashSet<(i32, i32, i32)> = HashSet::new();
            let unique: Vec<(Scenario, i32, i32, i32)> = scored
                .iter()
                .filter(|&&(_, st, pop, lvl)| seen.insert((st, pop, lvl)))
                .copied()
                .collect();
            let keep: Vec<Scenario> = unique
                .iter()
                .filter(|&&(_, st, pop, lvl)| {
                    !unique.iter().any(|&(_, st2, pop2, lvl2)| {
                        st2 <= st
                            && pop2 >= pop
                            && lvl2 >= lvl
                            && (st2 < st || pop2 > pop || lvl2 > lvl)
                    })
                })
                .map(|&(sc, ..)| sc)
                .collect();
            dropped += SCENARIOS.len() - keep.len();
            per_city.push(if keep.is_empty() { SCENARIOS.to_vec() } else { keep });
        }

        let combos: usize = per_city.iter().map(|v| v.len()).product();
        println!(
            "  mixed-lane search: {combos} assignments ({dropped} per-city scenarios trimmed as dominated)"
        );

        let mut idx = vec![0usize; cities.len()];
        loop {
            let scs: Vec<Scenario> = (0..cities.len()).map(|ci| per_city[ci][idx[ci]]).collect();
            // Uniform assignments were already enumerated above.
            if !scs.iter().all(|x| x.name == scs[0].name) {
                let terr = allocate_value(&state, &cities, &scs, monuments);
                all_plans.extend(enumerate_empire(
                    &state, &cities, &terr, &scs, &owned, monuments, 5, with_markets,
                ));
            }
            // Odometer over the per-city scenario lists.
            let mut k = 0;
            loop {
                if k == cities.len() {
                    break;
                }
                idx[k] += 1;
                if idx[k] < per_city[k].len() {
                    break;
                }
                idx[k] = 0;
                k += 1;
            }
            if k == cities.len() {
                break;
            }
        }
    }
    let front = pareto(&all_plans);

    println!("\n{}", "=".repeat(112));
    println!(
        "  JOINT FRONTIER — {} plans enumerated, {} non-dominated  (min stars, max giants, max SPT)",
        all_plans.len(),
        front.len()
    );
    println!("{}", "=".repeat(112));

    // Tag each row with the goal that claims it, resolved through the SAME
    // `pick_for_goal` the `--goal` flag uses — the listing and the build card
    // cannot disagree about which plan is the knee.
    let claimed: Vec<(usize, Goal)> = GOALS
        .iter()
        .filter_map(|&g| {
            let p = pick_for_goal(&front[..], g)?;
            front.iter().position(|q| std::ptr::eq(q, p)).map(|i| (i, g))
        })
        .collect();

    println!(
        "  {:<20}{:>8}{:>7}{:>8}{:>7}{:>5}{:>10}  {:<26}{:<22}{}",
        "scenario", "stars", "pop", "giants", "SPT", "mon", "★/giant", "hubs @ level",
        "markets (+income)", ""
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
        for &(idx, g) in &claimed {
            if idx == i {
                tag.push_str(goal_tag(g));
            }
        }
        let cpg = if p.giants > 0 {
            format!("{:.1}", p.stars as f64 / p.giants as f64)
        } else {
            "—".into()
        };
        println!(
            "  {:<20}{:>8}{:>7}{:>8}{:>7}{:>5}{:>10}  {:<26}{:<22}{}",
            p.label(),
            p.stars,
            p.pop,
            p.giants,
            p.spt,
            p.monuments_used(),
            cpg,
            hubs.join(" "),
            mkts.join(" "),
            tag
        );
    }

    // The frontier says what the options are; a goal says which one to build.
    let goals: Vec<Goal> = match goal {
        Some(g) => vec![g],
        None => Vec::new(),
    };
    for g in goals {
        match pick_for_goal(&front, g) {
            Some(p) => print_build_card(&state, &cities, p, monuments, g),
            None => println!("\n  no plan satisfies {}", goal_name(g)),
        }
    }
}
