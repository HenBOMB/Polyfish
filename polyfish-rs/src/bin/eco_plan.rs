//! CLI + verification/reporting harness for `polyfish::rules::eco_plan`.
//!
//! Checks the planner's placement claims against the engine rather than
//! against a second copy of the planner (--verify), explains one city's
//! build tile-by-tile (--explain), reports the star-efficient frontier
//! (--optimal), and prints build cards. No planning logic lives here —
//! see `polyfish::rules::eco_plan` for that.

use polyfish::functions::{get_adjacent_indices, get_chebyshev_distance};
use polyfish::rules::eco_plan::*;
use polyfish::settings::structures::get_structure_setting;
use polyfish::states::GameState;
use polyfish::types::*;
use std::collections::HashSet;

fn explain(state: &GameState, city_idx: i32, territory: &[i32], sc: Scenario) {
    let (buys, hub_sites, _) = tile_options(state, territory, sc);
    let (_, partner_name) = lane_hub(sc.lane);
    let partner_tiles: HashSet<i32> = buys
        .iter()
        .filter(|b| is_partner_buy(b, partner_name))
        .map(|b| b.idx)
        .collect();
    let b = build_out(state, city_idx, territory, sc, 0, Goal::Balanced);
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
    println!(
        "  hub sites by partner count: {:?}",
        &sites[..sites.len().min(8)]
    );
    println!("  chosen hub {:?} at level {}", b.hub_site, b.partners);
    let mut by_kind: std::collections::BTreeMap<&str, Vec<i32>> = Default::default();
    for x in buys.iter().filter(|x| Some(x.idx) != b.hub_site) {
        by_kind.entry(x.what).or_default().push(x.idx);
    }
    for (k, v) in by_kind {
        println!("    {k:<12} x{:<3} {:?}", v.len(), v);
    }
    println!(
        "  pop {}  stars(structures) {}  market {:?} (+{} SPT)",
        b.pop, b.stars, b.market_site, b.market_spt
    );
}

/// The partner structure each lane actually builds on a tile.

fn owned_board(base: &GameState, city_idx: i32, territory: &[i32]) -> GameState {
    let mut s = base.clone();
    let size = s.settings.size;
    let pov = pov_of(base);
    if let Some(t) = s.tribes.get_mut(&pov) {
        t.stars = 100_000;
    }
    for &i in territory {
        if let Some(t) = s.tiles.get_mut(&i) {
            t.owner = pov;
            t.ruling_city_coords = Some(polyfish::coords::Coords::from_index(city_idx, size));
        }
    }
    s
}

/// The structures a plan puts on the board, partners first and the hub last.
/// One definition, so `materialize` and `plan_len` cannot drift apart.
fn plan_of(
    state: &GameState,
    territory: &[i32],
    sc: Scenario,
    hub: Option<i32>,
) -> Vec<(i32, StructureType)> {
    let (buys, _, _) = tile_options(state, territory, sc);
    let (hub_type, partner_name) = lane_hub(sc.lane);
    let mut plan: Vec<(i32, StructureType)> = buys
        .iter()
        .filter(|b| is_partner_buy(b, partner_name))
        .map(|b| b.idx)
        .filter(|i| Some(*i) != hub)
        .map(|i| (i, lane_partner_type(sc.lane)))
        .collect();
    // A hub already standing is not a build. Re-creating it overwrites the
    // structure and pays its adjacency bonus a SECOND time, which made the
    // total depend on where the redundant build landed in the order: partners
    // first paid 1x2, hub first paid 1x0 (Aug 2026). `city_build` already
    // declines to re-cost a standing hub; this is the same fact, executed.
    if let Some(h) = hub {
        if polyfish::functions::get_structure_type_at(state, h) != Some(hub_type) {
            plan.push((h, hub_type));
        }
    }
    plan
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
    let plan = plan_of(&s, territory, sc, hub);

    let pov = pov_of(base);
    let pop_of = |s: &GameState| -> i32 {
        s.tribes
            .get(&pov)
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
    plan_of(state, territory, sc, hub).len()
}

/// Replay a plan one build at a time, reporting what each step paid.
/// Printed only when `verify` catches an order-dependent plan — the step where
/// two orders stop agreeing names the branch that lost the population.
fn trace_order(
    base: &GameState,
    city_idx: i32,
    territory: &[i32],
    sc: Scenario,
    hub: Option<i32>,
    order: &[usize],
    label: &str,
) {
    let mut s = owned_board(base, city_idx, territory);
    let plan = plan_of(&s, territory, sc, hub);
    let pov = pov_of(base);
    let pop_of = |s: &GameState| -> i32 {
        s.tribes
            .get(&pov)
            .and_then(|t| t.cities.iter().find(|c| c.idx == city_idx))
            .map_or(0, |c| c.population)
    };
    print!("      {label:<14}");
    let mut prev = pop_of(&s);
    for &k in order {
        let (idx, st) = plan[k];
        let ruled = polyfish::functions::get_city_owning_tile(&s, idx).map(|c| c.idx);
        let _ = polyfish::actions::structure::build_structure(&mut s, idx, st);
        let now = pop_of(&s);
        print!(" | {st:?}@{idx} ruled_by={ruled:?} +{}", now - prev);
        prev = now;
    }
    println!(" = {prev}");
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
    println!(
        "  (pop = capital population delivered by the engine, identical across 3 build orders)"
    );
    println!("  {}", "-".repeat(92));

    for sc in SCENARIOS {
        let terr = allocate_value(state, cities, &uniform(sc, cities.len()), monuments);
        for (ci, &city_idx) in cities.iter().enumerate() {
            // Monuments are excluded: they displace tiles but are not the
            // placement decision under test.
            let b = build_out(state, city_idx, &terr[ci], sc, 0, Goal::Balanced);
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
                polyfish::rules::economy::partner_count(&board, hub, hub_type, pov_of(state));
            let stored_level =
                polyfish::functions::get_structure_at(&board, hub).map_or(-1, |s| s.level);

            // 3. Build-order invariance, on the capital only — the other cities
            //    are villages here, so no CityState collects their pop.
            // Fewer than two builds has only one order. On a live board a city
            // whose hub already stands and whose partners are all built plans
            // nothing, so `n` can now be 0 and `n - 1` would underflow.
            let mut order_ok = true;
            if city_idx == capital && n >= 2 {
                let mut rev = ident.clone();
                rev.reverse();
                // Hub first, then partners in order: the retroactive-pay path.
                let mut hub_first = vec![n - 1];
                hub_first.extend(0..n - 1);
                for alt in [rev, hub_first] {
                    let (_, pop_alt) = materialize(state, city_idx, &terr[ci], sc, Some(hub), &alt);
                    if pop_alt != pop_ident {
                        order_ok = false;
                        println!(
                            "    ORDER-DEPENDENT: {} city {city_idx} pop {pop_ident} -> {pop_alt}",
                            sc.name
                        );
                        trace_order(
                            state,
                            city_idx,
                            &terr[ci],
                            sc,
                            Some(hub),
                            &ident,
                            "plan order",
                        );
                        trace_order(
                            state,
                            city_idx,
                            &terr[ci],
                            sc,
                            Some(hub),
                            &alt,
                            "this order",
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
            let (best_site, best_n) = doms
                .first()
                .map_or((hub, engine_partners), |&(p, _, s)| (s, p));

            // A hub the plan did not build carries a level from its own past:
            // set when it was built, bumped for each partner added since, and
            // never decremented when one is razed or its tile captured. On
            // state_4114_a a Windmill stands at level 1 with no partner left
            // alive. Only a hub this plan actually builds makes a claim about
            // its stored level.
            let plan_builds_hub =
                polyfish::functions::get_structure_type_at(state, hub) != Some(hub_type);
            let level_ok = !plan_builds_hub || stored_level == b.partners;
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
                if argmax_ok {
                    "yes".to_string()
                } else {
                    format!("{best_site}@{best_n}")
                },
                if ok { "ok" } else { "FAIL" }
            );
            if city_idx == capital {
                println!("      builds {n}, pop delivered {pop_ident}, order-invariant {order_ok}");
            }
        }
    }

    // Every goal, not just the one the table above audits. Pareto dominance is
    // goal-INDEPENDENT — a site beaten on SPT, super units and stars at once is
    // worse under any objective — so no goal may ever pick a dominated site.
    // The table checks Balanced; this sweeps the rest and stays silent unless
    // something breaks, which is what makes it useful as a regression net when
    // a comparator changes.
    let mut goal_failures = 0;
    for &g in &GOALS {
        for (ci, &city_idx) in cities.iter().enumerate() {
            for sc in SCENARIOS.iter() {
                let terr = allocate_value(state, cities, &uniform(*sc, cities.len()), monuments);
                if !lane_can_place_hub(state, &terr[ci], sc.lane) {
                    continue;
                }
                let Some(hub) = build_out(state, city_idx, &terr[ci], *sc, 0, g).hub_site else {
                    continue;
                };
                let doms = dominators_of(state, city_idx, &terr[ci], *sc, hub);
                if let Some(&(spt, st, site)) = doms.first() {
                    goal_failures += 1;
                    println!(
                        "  GOAL SWEEP FAIL: --goal {} city {city_idx} {} picked site {hub}, \
                         but site {site} gives {spt} SPT for {st} stars and dominates it",
                        goal_tag(g).trim().to_lowercase(),
                        sc.name
                    );
                }
            }
        }
    }
    if goal_failures == 0 {
        println!("  goal sweep: every goal x scenario x city picks an undominated site");
    }
    failures += goal_failures;

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
    // The SAME space build_out chooses from -- `hub_candidates` applies the
    // engine's one-partner legality floor, and a tile that cannot host a hub
    // cannot dominate one that can. Scanning raw `tile_options` site space
    // reported zero-partner tiles as dominators of legal picks.
    let site_space = hub_candidates(state, territory, sc, usize::MAX);
    // Through `site_value`, the one place a site is scored. Verify used to
    // judge on pop against stars while `build_out` chose on the same pair, and
    // both were blind to the Market income a site enables -- so when the
    // ranking moved to the frontier's axes this check kept failing correct
    // picks. Three consumers, one objective.
    let score = |s: i32| {
        let (spt, giants, stars, _pop) = site_value(state, city_idx, territory, sc, 0, Some(s));
        (spt, giants, stars)
    };
    let (cspt, cg, cs) = score(chosen);
    let mut out: Vec<(i32, i32, i32)> = site_space
        .iter()
        .copied()
        .filter(|&s| s != chosen)
        // An unreachable site cannot dominate anything — the city never gets there.
        .filter(|&s| site_reachable(state, city_idx, territory, sc, 0, s))
        .filter_map(|s| {
            let (spt, g, st) = score(s);
            (spt >= cspt && g >= cg && st <= cs && (spt > cspt || g > cg || st < cs))
                .then_some((spt, st, s))
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
fn optimal_report(state: &GameState, cities: &[i32], monuments: i32, lane: Lane, goal: Goal) {
    let hub_type = lane_hub(lane).0;
    for sc in SCENARIOS.iter().filter(|s| s.lane == lane) {
        let terr = allocate_value(state, cities, &uniform(*sc, cities.len()), monuments);
        for (ci, &city_idx) in cities.iter().enumerate() {
            let chosen = build_out(state, city_idx, &terr[ci], *sc, 0, goal).hub_site;
            // The space `build_out` actually chose from, so the ranking shown
            // and the pick shown answer over the same set. Reading raw
            // `tile_options` here listed tiles with no partner -- illegal hub
            // sites -- and then reported them as dominating a legal pick.
            let site_space = hub_candidates(state, &terr[ci], *sc, usize::MAX);

            let mut ranked: Vec<(i32, i32, i32, i32, i32, i32)> = site_space
                .iter()
                .filter(|&&s| site_reachable(state, city_idx, &terr[ci], *sc, 0, s))
                .map(|&s| {
                    let (spt, giants, stars, pop) =
                        site_value(state, cities[ci], &terr[ci], *sc, 0, Some(s));
                    let b = city_build(state, &terr[ci], *sc, 0, Some(s), None, None);
                    (spt, giants, -stars, s, b.partners, pop)
                })
                .collect();
            let m = site_maxima(
                &ranked
                    .iter()
                    .map(|&(a, b, c, _, _, _)| (a, b, -c))
                    .collect::<Vec<_>>(),
            );
            ranked.sort_by(|x, y| {
                site_order_key(y.0, y.1, -y.2, goal, m)
                    .cmp(&site_order_key(x.0, x.1, -x.2, goal, m))
                    .then(x.3.cmp(&y.3))
            });

            let blocked: Vec<i32> = site_space
                .iter()
                .copied()
                .filter(|&s| !site_reachable(state, city_idx, &terr[ci], *sc, 0, s))
                .collect();

            let no_hub = city_build(state, &terr[ci], *sc, 0, None, None, None);
            let best = ranked.first().copied();

            println!(
                "\n  {} — city {city_idx} ({} candidate sites)",
                sc.name,
                site_space.len()
            );
            println!(
                "      no hub at all:            pop {:>3}  stars {:>4}",
                no_hub.pop, no_hub.stars
            );
            for (_spt, _g, negstars, site, partners, pop) in ranked.iter().take(5) {
                let mark = if Some(*site) == chosen {
                    "  <- GREEDY PICK"
                } else {
                    ""
                };
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
                Some(c) => match ranked.iter().find(|r| r.3 == c).copied() {
                    None => {
                        println!("      VERDICT: chosen site {c} is not in the candidate space")
                    }
                    Some((cspt, cg, cs, _, _, cp)) => {
                        // Dominance on the SAME axes the planner optimises:
                        // SPT (which carries pop through level, plus Market
                        // income) at no greater cost.
                        let dominators: Vec<(i32, i32, i32)> = ranked
                            .iter()
                            .filter(|&&(spt, g, st, site, _, _)| {
                                site != c
                                    && spt >= cspt
                                    && g >= cg
                                    && st >= cs
                                    && (spt > cspt || g > cg || st > cs)
                            })
                            .map(|&(spt, _, st, site, _, _)| (spt, -st, site))
                            .collect();
                        if dominators.is_empty() {
                            println!(
                                "      VERDICT: greedy pick is on the frontier (not dominated)"
                            );
                        } else {
                            for (spt, st, site) in &dominators {
                                println!(
                                    "      VERDICT: DOMINATED — site {site} gives {spt} SPT for {st} stars; chosen {c} gives {cspt} SPT for {} stars (pop {cp})",
                                    -cs
                                );
                            }
                        }
                    }
                },
                None if best.is_some_and(|(_, _, _, _, _, bp)| bp > no_hub.pop) => {
                    let (bspt, _, _, bsite, _, bp) = best.unwrap();
                    println!(
                        "      VERDICT: NO HUB CHOSEN but site {bsite} would pay pop {bp} / {bspt} SPT"
                    );
                }
                None => println!("      VERDICT: no hub, none worthwhile"),
            }

            // Confirm the winner against the engine, not just the planner's model.
            if let Some((_, _, _, bsite, bpartners, _)) = best {
                let n = plan_len(state, &terr[ci], *sc, Some(bsite));
                let order: Vec<usize> = (0..n).collect();
                let (board, _) = materialize(state, city_idx, &terr[ci], *sc, Some(bsite), &order);
                let engine =
                    polyfish::rules::economy::partner_count(&board, bsite, hub_type, pov_of(state));
                if engine != bpartners {
                    println!(
                        "      ENGINE DISAGREES on site {bsite}: planner {bpartners}, engine {engine}"
                    );
                }
            }
        }
    }
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

const GOALS: [Goal; 5] = [
    Goal::Spt,
    Goal::Eco,
    Goal::Balanced,
    Goal::Army,
    Goal::Giants,
];

/// What each successive monument actually buys, and where it goes.
///
/// A monument's worth is threshold-shaped — it completes a reward slot or it
/// does nothing — so pricing it against stars needs an exchange rate nobody can
/// justify. This shows the curve instead: the best plan at each affordable
/// count, and the margin over the count below. Read it against the reasons the
/// planner cannot see (denying a frontier city space, answering a giant,
/// breaking a siege) and spend accordingly.
fn print_monument_ladder(front: &[EmpirePlan], cities: &[i32], g: Goal, budget: i32) {
    // Owned, because each count's pick borrows a temporary filtered Vec.
    let rows: Vec<(i32, EmpirePlan)> = (0..=budget)
        .filter_map(|m| {
            let at: Vec<EmpirePlan> = front
                .iter()
                .filter(|p| p.monuments_used() == m)
                .cloned()
                .collect();
            pick_for_goal(&at[..], g).map(|p| (m, p.clone()))
        })
        .collect();

    println!(
        "\n  MONUMENT LADDER — best plan at each count, for {}",
        goal_name(g)
    );
    println!(
        "  {:<5}{:>8}{:>7}{:>8}   {:<22}{:<24}{}",
        "mon", "stars", "SPT", "giants", "margin over previous", "placed at", "strategy"
    );
    for (i, (m, p)) in rows.iter().enumerate() {
        let margin = match i.checked_sub(1).and_then(|j| rows.get(j)) {
            Some((_, q)) => format!(
                "{:+}★  {:+} SPT  {:+} SU",
                p.stars - q.stars,
                p.spt - q.spt,
                p.giants - q.giants
            ),
            None => "—".to_string(),
        };
        let placed: Vec<String> = cities
            .iter()
            .zip(&p.monuments)
            .filter(|(_, n)| **n > 0)
            .map(|(c, n)| format!("city {c} x{n}"))
            .collect();
        println!(
            "  {:<5}{:>8}{:>7}{:>8}   {:<22}{:<24}{}",
            m,
            p.stars,
            p.spt,
            p.giants,
            margin,
            if placed.is_empty() {
                "—".into()
            } else {
                placed.join(", ")
            },
            p.label(),
        );
    }
    println!(
        "\n  A monument is 3 pop, so it is worth the reward slot it completes and\n  \
         nothing otherwise. This prices only what the planner can see: it cannot\n  \
         value denying a frontier city space, answering a giant, or breaking a siege."
    );
}

/// Terrain, resource and territory owner per tile — the ground truth every
/// claim about "tile 14 is a Fruit field" has to be checked against.
fn print_map(state: &GameState, cities: &[i32], terr: &[Vec<i32>], sc: Scenario) {
    let size = state.settings.size;
    println!("\n  MAP — terrain/resource, and the city each tile is allocated to");
    println!(
        "  terrain F=field f=forest M=mountain W=water    resource C=crop R=fruit G=game E=metal"
    );
    println!(
        "  allocation is for '{}', which is what sets each city's territory\n",
        sc.name
    );
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
                .map(|s| {
                    if s.structure_type == StructureType::Village {
                        "v"
                    } else {
                        "s"
                    }
                })
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
fn print_build_card(state: &GameState, cities: &[i32], plan: &EmpirePlan, monuments: i32, g: Goal) {
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
            let tag = if kind.ends_with(partner_name) {
                " (feeds the hub)"
            } else {
                ""
            };
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
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
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
  --state FILE.json   plan from a real position instead of a generated map:
                      your cities, the ground they rule, and whatever is
                      already built. Prices only what is left to do.
  --ladder            what each successive monument buys, and where it goes
                      (printed automatically whenever --monuments > 0)
  --map               print terrain, resources and the territory split
  --verify            check placements against the engine (exit 1 on failure)
  --optimal           rank every hub site for --goal (balanced if unstated);
                      --windmill / --forge for other lanes
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

    // A real position beats a generated one: `--state` plans for the cities you
    // actually hold, on the ground they actually rule, around what is already
    // built. `--seed` stays for the offline reference the evaluator is checked
    // against.
    let (state, cities) = match get("--state") {
        Some(path) => {
            let json = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
            // Two producers, two shapes: the mod writes a bare GameState
            // (live_game.json), the recorder wraps one under `gameState`.
            let loaded: GameState = serde_json::from_str(&json)
                .or_else(|_| {
                    serde_json::from_str::<serde_json::Value>(&json)
                        .and_then(|v| serde_json::from_value(v["gameState"].clone()))
                })
                .unwrap_or_else(|e| panic!("{path} holds no GameState: {e}"));
            let mut game = polyfish::game::Game::new();
            game.state = loaded;
            game.post_load(); // rebuilds tile indices, visibility and territory
            let pov = game.state.settings.current_player_turn_id;
            let cities: Vec<i32> = game
                .state
                .tribes
                .get(&pov)
                .map(|t| t.cities.iter().map(|c| c.idx).collect())
                .unwrap_or_default();
            assert!(!cities.is_empty(), "player {pov} holds no city in {path}");
            println!("state {path} | player {pov} | cities {cities:?}");
            (game.state, cities)
        }
        None => {
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
            (state, cities)
        }
    };
    // Empire-wide, not per city: monuments are earned from tasks, so a tribe
    // holds none at turn 0 and only ever has a handful. The frontier decides
    // which city each one goes to.
    // Default NONE: a tribe holds no monument at turn 0, so the turn-0 truth is
    // the honest headline and monument-funded plans are opted into.
    let monuments: i32 = get("--monuments").and_then(|s| s.parse().ok()).unwrap_or(0);
    let standalone = args.iter().any(|a| a == "--standalone");
    let ladder = args.iter().any(|a| a == "--ladder");
    let goal = get("--goal").and_then(|g| {
        let parsed = parse_goal(&g);
        if parsed.is_none() {
            eprintln!("unknown --goal '{g}'; use spt | eco | balanced | army | giants");
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
        // Audit against the objective you asked about; balanced when unstated.
        optimal_report(
            &state,
            &cities,
            monuments,
            lane,
            goal.unwrap_or(Goal::Balanced),
        );
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
            all.push((
                c,
                plan_city(&state, c, &terr[ci], sc, &owned, num_cities, monuments),
            ));
        }
    }

    for &c in &cities {
        println!("\n{}", "=".repeat(104));
        let is_capital = state.tiles.get(&c).is_some_and(|t| t.capital_of != 0);
        let role = if is_capital { "CAPITAL" } else { "city" };
        println!("  {role} @ tile {c}");
        {
            let mut census: std::collections::BTreeMap<String, i32> = Default::default();
            for idx in get_adjacent_indices(&state, c, 2).into_iter().chain([c]) {
                let Some(t) = state.tiles.get(&idx) else {
                    continue;
                };
                let r = state
                    .resources
                    .get(&idx)
                    .and_then(|r| r.as_ref())
                    .map(|r| format!("+{:?}", r.resource_type))
                    .unwrap_or_default();
                *census
                    .entry(format!("{:?}{}", t.terrain_type, r))
                    .or_default() += 1;
            }
            let line: Vec<String> = census.iter().map(|(k, v)| format!("{k} {v}")).collect();
            println!("  5x5 terrain: {}", line.join(", "));
        }
        println!("{}", "=".repeat(104));
        println!(
            "  {:<20}{:>6}{:>7}{:>8}{:>7}{:>8}{:>7}{:>12}{:>10}{:>8}",
            "scenario",
            "tiles",
            "pop",
            "stars",
            "level",
            "giants",
            "SPT",
            "★/giant",
            "hub@lvl",
            "market"
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
                    p.scenario,
                    p.territory,
                    p.max_pop,
                    "—",
                    p.level,
                    "—",
                    "—",
                    "unreachable",
                    "—",
                    "—"
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
    const K_UNIFORM: usize = 8;
    const K_MIXED: usize = 5;
    let k_uniform = shortlist(K_UNIFORM, cities.len(), SCENARIOS.len());
    if k_uniform < K_UNIFORM {
        println!(
            "  NOTE: {} cities — uniform-lane hub shortlist trimmed to {k_uniform} sites per \
             city (from {K_UNIFORM}) to bound the search. Sites past the trim rank lower on \
             partner count and were not priced.",
            cities.len()
        );
    }
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
            k_uniform,
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
                    let terr =
                        allocate_value(&state, &cities, &uniform(sc, cities.len()), monuments);
                    let b = build_out(&state, c, &terr[ci], sc, monuments, Goal::Balanced);
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
            per_city.push(if keep.is_empty() {
                SCENARIOS.to_vec()
            } else {
                keep
            });
        }

        let combos: usize = per_city.iter().map(|v| v.len()).product();
        // Total work is assignments x hub combinations, so the shortlist has to
        // be set against the assignment count, not independently of it.
        let k_mixed = shortlist(K_MIXED, cities.len(), combos);
        println!(
            "  mixed-lane search: {combos} assignments ({dropped} per-city scenarios trimmed as dominated)"
        );
        if k_mixed < K_MIXED {
            println!(
                "  NOTE: mixed-lane hub shortlist trimmed to {k_mixed} sites per city (from \
                 {K_MIXED}) — {combos} assignments x {}^{} combinations would not have \
                 finished. Sites past the trim rank lower on partner count and were not priced.",
                K_MIXED + 1,
                cities.len()
            );
        }

        let mut idx = vec![0usize; cities.len()];
        loop {
            let scs: Vec<Scenario> = (0..cities.len()).map(|ci| per_city[ci][idx[ci]]).collect();
            // Uniform assignments were already enumerated above.
            if !scs.iter().all(|x| x.name == scs[0].name) {
                let terr = allocate_value(&state, &cities, &scs, monuments);
                all_plans.extend(enumerate_empire(
                    &state,
                    &cities,
                    &terr,
                    &scs,
                    &owned,
                    monuments,
                    k_mixed,
                    with_markets,
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
            front
                .iter()
                .position(|q| std::ptr::eq(q, p))
                .map(|i| (i, g))
        })
        .collect();

    println!(
        "  {:<20}{:>8}{:>7}{:>8}{:>7}{:>5}{:>10}  {:<26}{:<22}{}",
        "scenario",
        "stars",
        "pop",
        "giants",
        "SPT",
        "mon",
        "★/giant",
        "hubs @ level",
        "markets (+income)",
        ""
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
        if monuments > 0 || ladder {
            print_monument_ladder(&front, &cities, g, monuments);
        }
    }
    if (ladder || monuments > 0) && goal.is_none() {
        print_monument_ladder(&front, &cities, Goal::Balanced, monuments);
    }
}
