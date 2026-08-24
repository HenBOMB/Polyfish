//! Empire-level territory allocation and the joint Pareto frontier across
//! cities. Hub sites are the one coupled choice (a Market reads every
//! friendly hub it touches, including a neighbour city's), so this is the
//! only module that enumerates combinations rather than planning cities
//! independently.

use super::*;
use super::city::*;
use super::tech::*;
use crate::functions::{get_adjacent_indices, get_chebyshev_distance, MARKET_MAX_LEVEL};
use crate::rules::economy::{level_at_pop, super_units_at_level as giants_at_level};
use crate::states::GameState;
use crate::types::*;
use std::collections::{HashMap, HashSet};

/// Assign every tile within radius 2 of any planned city to exactly one city —
/// nearest wins, ties to the earlier city — so overlapping 5x5s are not
/// double-counted.
/// Joint allocation by MARGINAL VALUE: a contested tile goes to whichever
/// claimant's plan gains more pop from it. Nearest-city was the first rule and
/// it was wrong — it handed the capital tiles it had no use for while capping a
/// neighbour's Sawmill a level below what the map supports.
/// The same scenario for every city — the pure (unmixed) empire.
pub fn uniform(sc: Scenario, n: usize) -> Vec<Scenario> {
    vec![sc; n]
}

pub fn allocate_value(
    state: &GameState,
    cities: &[i32],
    scs: &[Scenario],
    monuments: i32,
) -> Vec<Vec<i32>> {
    // Believe the state when it knows. The radius model below exists only
    // because a generated map has no ownership yet; where every planned city is
    // a real city that already rules tiles, the engine's own answer is not an
    // approximation of the truth, it IS the truth, and modelling it can only
    // disagree. All-or-nothing: a half-real allocation would mix the two rules.
    // `engine_territory` alone only ever returns what a city rules TODAY, so a
    // BorderGrowth scenario planned from a real state used to get the exact
    // same tiles as its natural counterpart -- every "+border" row silently
    // reduced to "natural minus the PopGrowth bonus" (found Aug 2026: the
    // +border territory count matched +natural's exactly, tile for tile).
    // `extend_for_border_growth` adds the ring such a city would actually grow
    // into, without touching the "believe reality" rule for tiles it already
    // rules.
    if let Some(real) = engine_territory(state, cities) {
        return extend_for_border_growth(state, cities, scs, real);
    }

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
        // Rank by what the tile is worth to each claimant, on the same axes as
        // every other consumer: SPT first, then super units. This used to rank
        // on marginal POP, the last place still using the old yardstick -- and
        // the most consequential, since the split it produces is the ground the
        // frontier then plans on. A tile that lifts a hub a level and feeds a
        // Market is worth more than one that adds raw population, and pop could
        // not see the difference. Distance and index break ties, so the
        // allocation stays deterministic and reproducible.
        let value_of = |ci: usize, tiles: &[i32]| -> (i32, i32) {
            let hub = build_out(state, cities[ci], tiles, scs[ci], monuments, Goal::Balanced)
                .hub_site;
            let (spt, giants, _stars, _pop) =
                site_value(state, cities[ci], tiles, scs[ci], monuments, hub);
            (spt, giants)
        };
        let mut ranked: Vec<(i32, i32, i32, usize)> = who
            .iter()
            .map(|&ci| {
                let (base_spt, base_g) = value_of(ci, &terr[ci]);
                let mut with = terr[ci].clone();
                with.push(idx);
                with.sort();
                let (spt, g) = value_of(ci, &with);
                (
                    -(spt - base_spt),
                    -(g - base_g),
                    get_chebyshev_distance(idx, cities[ci], state.settings.size),
                    ci,
                )
            })
            .collect();
        ranked.sort();
        terr[ranked[0].3].push(idx);
    }
    for v in terr.iter_mut() {
        v.sort();
    }
    terr
}

/// Adds the not-yet-owned radius-2 ring to any city whose scenario plans to
/// take BorderGrowth. Only unclaimed tiles (`owner == 0`) are eligible --
/// real Polytopia border growth claims neutral ground, never a tile anyone
/// (friend or foe) already rules, so a tile another real city of ours holds
/// is left alone rather than reassigned. Contested rings between two of
/// OUR OWN growing cities resolve to the nearer one, same tie-break as the
/// synthetic radius model below.
fn extend_for_border_growth(
    state: &GameState,
    cities: &[i32],
    scs: &[Scenario],
    mut real: Vec<Vec<i32>>,
) -> Vec<Vec<i32>> {
    if !scs.iter().any(|s| s.border_growth) {
        return real;
    }
    let owned: HashSet<i32> = real.iter().flatten().copied().collect();
    let mut candidates: HashMap<i32, Vec<usize>> = HashMap::new();
    for (ci, &c) in cities.iter().enumerate() {
        if !scs[ci].border_growth {
            continue;
        }
        for idx in get_adjacent_indices(state, c, 2) {
            if owned.contains(&idx) {
                continue;
            }
            if state.tiles.get(&idx).is_none_or(|t| t.owner != 0) {
                continue;
            }
            candidates.entry(idx).or_default().push(ci);
        }
    }
    for (idx, who) in candidates {
        let winner = who
            .iter()
            .copied()
            .min_by_key(|&ci| (get_chebyshev_distance(idx, cities[ci], state.settings.size), ci))
            .unwrap();
        real[winner].push(idx);
    }
    for v in real.iter_mut() {
        v.sort();
        v.dedup();
    }
    real
}

pub fn allocate(state: &GameState, cities: &[i32], border_growth: bool) -> Vec<Vec<i32>> {
    allocate_mode(state, cities, border_growth, false)
}

/// `standalone` scores each city on its full square, ignoring that neighbours
/// would claim the shared tiles — the per-city CEILING, not a joint plan.
pub fn allocate_mode(
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
pub struct EmpirePlan {
    /// One scenario per city: a mixed empire can run the forest lane in a
    /// wooded city and the farm lane next door.
    pub scenarios: Vec<Scenario>,
    pub stars: i32,
    pub pop: i32,
    pub giants: i32,
    pub spt: i32,
    /// Per-city pop and level, and the techs the plan pays for — carried so a
    /// build card can be printed without re-deriving (and mis-deriving) them.
    pub city_pop: Vec<i32>,
    pub city_level: Vec<i32>,
    pub techs: Vec<TechnologyType>,
    /// Monuments this plan consumes, per city. Empire-wide and earned, so this
    /// is a scarcity cost the star total cannot express.
    pub monuments: Vec<i32>,
    pub hubs: Vec<Option<i32>>,
    pub levels: Vec<i32>,
    pub markets: Vec<Option<i32>>,
    pub market_income: Vec<i32>,
}

impl EmpirePlan {
    pub fn monuments_used(&self) -> i32 {
        self.monuments.iter().sum()
    }

    /// The scenario name when every city agrees, otherwise a per-city sketch.
    pub fn label(&self) -> String {
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
pub fn sc_short(sc: Scenario) -> &'static str {
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
/// How many hub sites per city the frontier may consider.
///
/// The frontier enumerates `(k+1)^n` combinations, so the shortlist is the
/// exponent's base and the only lever that does not change what a plan means.
/// Past a few cities the full shortlist runs to millions of combinations, so it
/// shrinks to hold the product under `COMBO_BUDGET`. The caller announces it
/// when it bites: a search that truncates without saying so reads as an
/// exhaustive one.
pub fn shortlist(top_k: usize, n: usize, rounds: usize) -> usize {
    // Inner iterations the whole frontier may spend. Calibrated on a 6-city
    // board at roughly 4us per combination, so this is about 25 seconds.
    const WORK_BUDGET: usize = 6_000_000;
    if n == 0 || rounds == 0 {
        return top_k;
    }
    let budget = (WORK_BUDGET / rounds).max(1);
    let mut k = top_k;
    while k > 1 && (k + 1).checked_pow(n as u32).is_none_or(|c| c > budget) {
        k -= 1;
    }
    k
}

pub fn enumerate_empire(
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
    // One tile scan per city, reused for every hub combination below.
    let plots: Vec<Plot> = (0..n).map(|ci| Plot::new(state, &terr[ci], scs[ci])).collect();
    for ci in 0..n {
        let (_, partner_name) = lane_hub(scs[ci].lane);
        let ptype = lane_partner_type(scs[ci].lane);
        let e = partners_by_type.entry(ptype).or_default();
        for b in plots[ci].buys.iter().filter(|b| is_partner_buy(b, partner_name)) {
            e.insert(b.idx);
        }
        // Partners already standing count too — `tile_options` skips occupied
        // tiles, so on a live state these would otherwise feed nothing.
        e.extend(plots[ci].standing_partners.iter().copied());
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
            // A hub already standing leaves nothing to enumerate.
            let built = standing(state, &terr[ci], lane_hub(scs[ci].lane).0);
            if let Some(&h) = built.first() {
                return vec![Some(h)];
            }
            let mut v: Vec<Option<i32>> =
                hub_candidates_on(state, &plots[ci], top_k, partners_for(ci))
                    .into_iter()
                    .map(Some)
                    .collect();
            v.push(None);
            v
        })
        .collect();


    // Adjacency and Market-site terrain do not depend on where the hubs go, but
    // the scan below runs once per city per combination — the top of the
    // profile before this was `get_adjacent_indices` handing back a fresh Vec.
    let tile_count = (state.settings.size * state.settings.size) as usize;
    let adj_tbl: Vec<Vec<i32>> = (0..tile_count)
        .map(|i| get_adjacent_indices(state, i as i32, 1))
        .collect();
    let market_cands: Vec<Vec<i32>> = (0..n)
        .map(|ci| {
            terr[ci]
                .iter()
                .copied()
                .filter(|&a| market_site_legal(state, a, false))
                .collect()
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
        // `hub_tiles` used to be a HashSet and `counted` a fresh set per city,
        // rebuilt for every combination. Both are at most `n` entries wide, so
        // a slice the callee scans is cheaper than the sets were to allocate.
        let hub_tiles: Vec<i32> = hubs.iter().flatten().copied().collect();
        let base: Vec<BuildOut> = (0..n)
            .map(|ci| {
                city_build_on(
                    state, &plots[ci], scs[ci], monuments, hubs[ci], None,
                    Some(partners_for(ci)), &hub_tiles,
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
                let best = market_cands[ci]
                    .iter()
                    .copied()
                    .filter(|a| {
                        let adj = &adj_tbl[*a as usize];
                        // Feeding ANY adjacent hub disqualifies the tile, whichever
                        // city that hub belongs to — taking it would drop the level
                        // stage 1 already priced, and every Market reading it.
                        let feeds = (0..n).any(|cj| {
                            hubs[cj].is_some_and(|h| adj.contains(&h))
                                && partners_for(cj).contains(a)
                        });
                        !feeds && !hubs.iter().any(|h| *h == Some(*a))
                    })
                    .map(|a| {
                        let adj = &adj_tbl[a as usize];
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
                        city_build_on(
                            state, &plots[ci], scs[ci], m, hubs[ci], markets[ci],
                            Some(partners_for(ci)), &hub_tiles,
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
            let b = city_build_on(
                state, &plots[ci], scs[ci], alloc[ci], hubs[ci], markets[ci],
                Some(partners_for(ci)), &hub_tiles,
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
pub fn dominates(q: &EmpirePlan, p: &EmpirePlan) -> bool {
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
pub fn frontier_insert(front: &mut Vec<EmpirePlan>, p: EmpirePlan) {
    if front.iter().any(|q| dominates(q, &p)) {
        return;
    }
    front.retain(|q| !dominates(&p, q));
    front.push(p);
}

pub fn pareto(plans: &[EmpirePlan]) -> Vec<EmpirePlan> {
    let mut front: Vec<EmpirePlan> = Vec::new();
    for p in plans {
        frontier_insert(&mut front, p.clone());
    }
    front
}

/// Tile-by-tile dump for one city+scenario, so a number in the table can be
/// traced back to the map.

/// Frontier maxima for SPT, super units and stars — the scale every knee is
/// measured against. Never zero, so they are safe denominators.
pub fn frontier_maxima(front: &[EmpirePlan]) -> (i64, i64, i64) {
    let m = |v: i64| v.max(1);
    (
        m(front.iter().map(|p| p.spt).max().unwrap_or(1) as i64),
        m(front.iter().map(|p| p.giants).max().unwrap_or(1) as i64),
        m(front.iter().map(|p| p.stars).max().unwrap_or(1) as i64),
    )
}

/// The knee: value minus cost, with SPT, super units and stars each normalised
/// to [0,1] across the frontier.
///
/// On normalised axes the chord from the cheapest plan to the richest IS the
/// diagonal, so the maximum of (value - cost) is the point furthest above that
/// chord — the elbow, with no hand-tuned exchange rate. `w_spt`/`w_su` choose
/// what "value" means: (1,0) is income alone, (0,1) army alone, (1,1) both.
/// Scaled to integers by the common denominator so ties are exact.
///
/// MONUMENTS ARE NOT PRICED HERE. A monument is 3 pop, and 3 pop is worth the
/// reward slot it completes and nothing otherwise — threshold-shaped, not
/// linear, so no single exchange rate against stars is right. Charging them a
/// full axis made every knee refuse to spend one across 11 seeds; leaving them
/// free made a picker drain its budget for a rounding error. The ladder shows
/// the marginal value instead and the caller decides.
pub fn knee_score(p: &EmpirePlan, w_spt: i64, w_su: i64, m: (i64, i64, i64)) -> i64 {
    knee_raw(p.spt as i64, p.giants as i64, p.stars as i64, w_spt, w_su, m)
}

/// The knee on bare numbers, so a whole plan and a single hub site are scored
/// by one implementation rather than two that can drift.
pub fn knee_raw(
    spt: i64,
    giants: i64,
    stars: i64,
    w_spt: i64,
    w_su: i64,
    (ms, mg, mc): (i64, i64, i64),
) -> i64 {
    let value = w_spt * spt * mg * mc + w_su * giants * ms * mc;
    let cost = (w_spt + w_su) * stars * ms * mg;
    value - cost
}

/// Rank one hub site FOR A STATED GOAL, mirroring `pick_for_goal` exactly.
///
/// `site_value` measures a site — that is a fact and stays goal-agnostic. This
/// is the ordering, and the ordering is the part that must follow intent: an
/// army build and an income build should not be handed the same tile just
/// because one comparator was hardcoded. Ties break toward fewer stars.
pub fn site_order_key(
    spt: i32,
    giants: i32,
    stars: i32,
    g: Goal,
    m: (i64, i64, i64),
) -> (i64, i64, i64, i64) {
    let (spt, giants, stars) = (spt as i64, giants as i64, stars as i64);
    // Every ordering ends with the axes its objective does NOT weight, so a
    // goal can never throw away value it is merely indifferent to. Without
    // this, --goal army (knee on super units alone) happily takes a site beaten
    // outright on SPT at the same cost, which the verify goal-sweep catches as
    // picking a Pareto-dominated site. Each key is now a refinement of Pareto.
    let (a, b, c) = match g {
        Goal::Spt => (spt, giants, -stars),
        Goal::Giants => (giants, spt, -stars),
        Goal::Eco => (knee_raw(spt, giants, stars, 1, 0, m), spt, giants),
        Goal::Balanced => (knee_raw(spt, giants, stars, 1, 1, m), spt, giants),
        Goal::Army => (knee_raw(spt, giants, stars, 0, 1, m), giants, spt),
    };
    (a, b, c, -stars)
}

/// Frontier maxima over a set of scored sites, for the knee's normalisation.
pub fn site_maxima(v: &[(i32, i32, i32)]) -> (i64, i64, i64) {
    let m = |x: i64| x.max(1);
    (
        m(v.iter().map(|t| t.0).max().unwrap_or(1) as i64),
        m(v.iter().map(|t| t.1).max().unwrap_or(1) as i64),
        m(v.iter().map(|t| t.2).max().unwrap_or(1) as i64),
    )
}


/// Pick the plan that best serves `g`. Every ordering ends with fewer stars and
/// then fewer monuments, so ties resolve toward the plan that spends least.
pub fn pick_for_goal<'a>(front: &'a [EmpirePlan], g: Goal) -> Option<&'a EmpirePlan> {
    let thrift = |p: &EmpirePlan| (-p.stars, -p.monuments_used());
    let m = frontier_maxima(front);
    front.iter().max_by(|a, b| {
        let key = |p: &EmpirePlan| match g {
            Goal::Spt => (p.spt as i64, p.giants as i64, 0),
            Goal::Giants => (p.giants as i64, p.spt as i64, 0),
            Goal::Eco => (knee_score(p, 1, 0, m), p.spt as i64, p.giants as i64),
            Goal::Balanced => (knee_score(p, 1, 1, m), p.spt as i64, p.giants as i64),
            Goal::Army => (knee_score(p, 0, 1, m), p.giants as i64, p.spt as i64),
        };
        key(a).cmp(&key(b)).then(thrift(a).cmp(&thrift(b)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::{CityState, TileState, TribeState};

    /// Regression for the Aug 2026 bug: planning a BorderGrowth scenario from
    /// a REAL state (`engine_territory` succeeds) used to hand back the exact
    /// same tiles as the natural scenario, because `engine_territory` only
    /// ever reports what a city rules TODAY. Every "+border" row silently
    /// reduced to "natural minus the PopGrowth bonus" instead of the radius-2
    /// ring it should plan into.
    #[test]
    fn border_growth_on_a_real_state_actually_grows_the_territory() {
        let mut state = GameState::default();
        state.settings.size = 11;
        state.settings.current_player_turn_id = 1;
        let center = 5 * 11 + 5;
        // A real 3x3 inner territory, same shape `still_capturable`-driven
        // play would leave a level-1 city holding.
        let inner: Vec<i32> = get_adjacent_indices(&state, center, 1).into_iter().chain([center]).collect();
        for &idx in &inner {
            let mut t = TileState::default();
            t.terrain_type = TerrainType::Field;
            t.owner = 1;
            t.ruling_city_coords = Some(crate::coords::Coords::from_index(center, 11));
            state.tiles.insert(idx, t);
        }
        // The rest of the 5x5 ring: unclaimed ground a BorderGrowth could
        // actually take.
        for idx in get_adjacent_indices(&state, center, 2) {
            state.tiles.entry(idx).or_insert_with(|| {
                let mut t = TileState::default();
                t.terrain_type = TerrainType::Field;
                t
            });
        }
        let mut tribe = TribeState::default();
        tribe.id = 1;
        let mut city = CityState { idx: center, owner: 1, ..Default::default() };
        city._territory = inner.clone();
        tribe.cities.push(city);
        state.tribes.insert(1, tribe);

        let cities = [center];
        let natural = allocate_value(&state, &cities, &[SCENARIOS[0]], 0); // sawmill natural
        let border = allocate_value(&state, &cities, &[SCENARIOS[1]], 0); // sawmill +border

        assert_eq!(natural[0].len(), inner.len(), "natural must stay exactly the real territory");
        assert!(
            border[0].len() > natural[0].len(),
            "+border must grow past the real territory, got {} vs natural's {}",
            border[0].len(),
            natural[0].len()
        );
        // Every added tile must be within the radius-2 ring and previously unclaimed.
        for &idx in &border[0] {
            assert!(
                get_chebyshev_distance(idx, center, 11) <= 2,
                "tile {idx} is outside the radius-2 growth ring"
            );
        }
    }

    /// A scenario with no BorderGrowth must never touch tiles outside what
    /// the city already really rules -- the "believe the state" rule stays
    /// absolute for non-growing scenarios.
    #[test]
    fn natural_scenario_never_grows_a_real_territory() {
        let mut state = GameState::default();
        state.settings.size = 11;
        state.settings.current_player_turn_id = 1;
        let center = 5 * 11 + 5;
        let inner: Vec<i32> = get_adjacent_indices(&state, center, 1).into_iter().chain([center]).collect();
        for &idx in &inner {
            let mut t = TileState::default();
            t.terrain_type = TerrainType::Field;
            t.owner = 1;
            t.ruling_city_coords = Some(crate::coords::Coords::from_index(center, 11));
            state.tiles.insert(idx, t);
        }
        let mut tribe = TribeState::default();
        tribe.id = 1;
        let mut city = CityState { idx: center, owner: 1, ..Default::default() };
        city._territory = inner.clone();
        tribe.cities.push(city);
        state.tribes.insert(1, tribe);

        let cities = [center];
        let mut natural = allocate_value(&state, &cities, &[SCENARIOS[0]], 0);
        natural[0].sort();
        let mut expected = inner.clone();
        expected.sort();
        assert_eq!(natural[0], expected);
    }
}
