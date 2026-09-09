//! Per-turn cached economic plan: which Mine-lane tiles currently partner a
//! committed Forge hub, so real building can follow the SAME plan
//! `bin/eco_plan`'s joint frontier (`rules::eco_plan::enumerate_empire`)
//! already computes correctly, instead of `scoring.rs`'s old reactive
//! "cluster near whatever mine happened to land first" heuristic (the
//! original "gravity" complaint this whole investigation started from).
//!
//! Mine lane only this round (EXP_ELO_100) — the other lanes (Sawmill,
//! Windmill) can follow the same pattern once this one is measured.
//!
//! # Momentum, not a fresh answer every turn
//!
//! Credit accumulates turn to turn instead of being overwritten: once a
//! tile earns credit for partnering a committed hub, it keeps that credit
//! even if a later, more-informed recompute's specific hub choice shifts.
//! `EXP_ELO_100`'s own turn4->turn7 convergence check is the reason this
//! matters — a rough early estimate (planning around a village we don't
//! hold yet) and a precise later one (once it's real) can legitimately
//! disagree on the exact best hub tile without either being wrong, and
//! erasing the early credit on every refinement would waste the very
//! investment this feature exists to protect. Credit is hard-invalidated
//! only when the tile's OWNING city is actually lost.
//!
//! # Prospective cities
//!
//! A village we don't hold yet is planned around (as if already captured,
//! seeded with the same radius-1 square a real capture claims — see
//! `rules::eco_plan::allocate_value_with_prospective`) only when
//! `ai::movement::village_race_confidence` clears `PROSPECTIVE_CONFIDENCE_MIN`
//! — Verdi's own framing: "confidence needs to be high enough for the
//! assumption to hold." A prospective city's own territory is never
//! credited directly (we can't build there yet); only HELD cities' tiles
//! are, against the full committed-hub set (real + prospective), since a
//! currently-buildable mine that will end up adjacent to a not-yet-real
//! neighbour's future hub is still the right move now — the real engine's
//! cross-city partner credit (`city_build_on`) doesn't care about build
//! order, only final adjacency and ownership.

use crate::ai::movement::village_race_confidence;
use crate::ai::oracle_macro::{retakeable_village, still_capturable};
use crate::functions::get_chebyshev_distance;
use crate::rules::eco_plan::{
    allocate_value, allocate_value_with_prospective, enumerate_empire, pareto, pick_for_goal,
    tech_bill, EmpirePlan, Goal, Scenario, SCENARIOS,
};
use crate::states::{GameState, PlayerId};
use crate::types::{ResourceType, TechnologyType};
use std::collections::{HashMap, HashSet};

/// Minimum `village_race_confidence` for a not-yet-captured village to be
/// planned around as if it were already ours.
pub const PROSPECTIVE_CONFIDENCE_MIN: f32 = 0.75;

/// splitmix64 finalizer, mirroring `belief::map::cache::mix` — combined via
/// wrapping addition so the fingerprint below is independent of set
/// iteration order.
fn mix(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Cache fingerprint for `update`'s recompute gate. Held and prospective
/// city sets are not monotonic (capture AND loss), so a `.len()` proxy
/// would be unsafe — see `belief::map::cache::BeliefKey`'s doc comment for
/// the same reasoning applied to a different non-monotonic set. Owned tech
/// is included too: a newly-discovered tech can change which hub prices out
/// as best even with the same cities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct EcoPlanFingerprint {
    n_held: u32,
    n_prospective: u32,
    n_tech: u32,
    hash: u64,
}

impl EcoPlanFingerprint {
    fn compute(held: &[i32], prospective: &HashSet<i32>, owned: &HashSet<TechnologyType>) -> Self {
        let mut hash = 0u64;
        for &c in held {
            hash = hash.wrapping_add(mix(c as u64));
        }
        for &p in prospective {
            hash = hash.wrapping_add(mix(p as u64 ^ 0xA5A5_0000_0000_0001));
        }
        for &t in owned {
            hash = hash.wrapping_add(mix(t as u64 ^ 0x5A5A_0000_0000_0002));
        }
        Self {
            n_held: held.len() as u32,
            n_prospective: prospective.len() as u32,
            n_tech: owned.len() as u32,
            hash,
        }
    }
}

/// Cities farther apart than this can never interact through
/// `enumerate_empire`'s mechanics (territory contest, or a market reading a
/// neighbour's hub): each city's own territory claim reaches at most radius
/// 2 (`allocate_value`'s BorderGrowth radius, matching `city::city_square`),
/// and a market/hub "touch" is radius-1 adjacency on top of that —
/// 2 + 1 + 2 = 5. Over-grouping two cities that turn out not to interact
/// only costs performance, never correctness, so this is a safe upper
/// bound, not an exact threshold.
const ECO_PLAN_INTERACTION_RADIUS: i32 = 5;

/// Connected components of `cities` under "could ever interact" (Chebyshev
/// distance <= `ECO_PLAN_INTERACTION_RADIUS`), as groups of original
/// indices in ascending order. A single component (the common case) means
/// no decomposition is possible or needed.
fn connectivity_groups(state: &GameState, cities: &[i32]) -> Vec<Vec<usize>> {
    let n = cities.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if get_chebyshev_distance(cities[i], cities[j], state.settings.size)
                <= ECO_PLAN_INTERACTION_RADIUS
            {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    let mut result: Vec<Vec<usize>> = groups.into_values().collect();
    result.sort_by_key(|g| g[0]);
    result
}

/// The index cross product of `sizes` (one axis per group's own plan
/// list), as `Vec<Vec<usize>>` — each inner Vec picks one plan index per
/// group, in group order. Empty whenever any `size` is 0, propagating
/// naturally through the inner loop.
fn cross_product_indices(sizes: &[usize]) -> Vec<Vec<usize>> {
    let mut result: Vec<Vec<usize>> = vec![Vec::new()];
    for &size in sizes {
        let mut next = Vec::with_capacity(result.len() * size.max(1));
        for combo in &result {
            for i in 0..size {
                let mut c = combo.clone();
                c.push(i);
                next.push(c);
            }
        }
        result = next;
    }
    result
}

/// `enumerate_empire`, decomposed by spatial connectivity when `cities`
/// splits into non-interacting groups: solving each group's joint
/// hub/market search independently instead of the full `top_k^n_cities`
/// cross product turns the expensive tail (5-6 cities, ~91% of this
/// routine's measured self-play CPU cost) into a handful of small
/// per-group searches. Scoped to `monuments=0` callers only (this file's
/// own call site) — `enumerate_empire`'s general empire-wide monument-
/// budget coupling is out of scope here, and is a proven no-op at budget 0
/// (every surviving plan already has an all-zero `monuments` Vec, since any
/// combo needing so much as one monument is dropped before it's returned).
///
/// The one coupling that DOES survive at `monuments=0`: tech cost scales
/// with the TRUE total city count (`tech::tech_bill`), not a group's own
/// smaller count, and a tech needed by two different groups' chosen plans
/// must be billed once for the merged empire, not twice. Both are
/// corrected explicitly below by re-deriving each plan's tech bill with the
/// existing public `tech_bill` helper — `enumerate_empire`, `pick_for_goal`
/// and `pareto` are all called unmodified.
fn enumerate_empire_decomposed(
    state: &GameState,
    cities: &[i32],
    terr: &[Vec<i32>],
    scs: &[Scenario],
    owned: &HashSet<TechnologyType>,
    top_k: usize,
) -> Vec<EmpirePlan> {
    let groups = connectivity_groups(state, cities);
    if groups.len() <= 1 {
        return enumerate_empire(state, cities, terr, scs, owned, 0, top_k, true);
    }

    let true_n = cities.len() as i32;
    // original city index -> (group id, position within that group's cities)
    let mut reverse: Vec<(usize, usize)> = vec![(0, 0); cities.len()];
    for (gid, group) in groups.iter().enumerate() {
        for (pos, &orig) in group.iter().enumerate() {
            reverse[orig] = (gid, pos);
        }
    }

    let mut group_plans: Vec<Vec<EmpirePlan>> = Vec::with_capacity(groups.len());
    for group in &groups {
        let g_cities: Vec<i32> = group.iter().map(|&i| cities[i]).collect();
        let g_terr: Vec<Vec<i32>> = group.iter().map(|&i| terr[i].clone()).collect();
        let g_scs: Vec<Scenario> = group.iter().map(|&i| scs[i]).collect();
        let component_n = g_cities.len() as i32;
        let mut plans = enumerate_empire(state, &g_cities, &g_terr, &g_scs, owned, 0, top_k, true);
        // Re-price each plan's tech bill at the TRUE total city count before
        // this group's own Pareto reduction, so a plan's own standalone
        // cost is right even though it doesn't yet know what other groups
        // will also need to buy (that cross-group double-count is
        // corrected once at merge time below).
        for p in &mut plans {
            let wrong_bill = tech_bill(&p.techs, owned, component_n);
            let right_bill = tech_bill(&p.techs, owned, true_n);
            p.stars = p.stars - wrong_bill + right_bill;
        }
        group_plans.push(pareto(&plans));
    }

    let sizes: Vec<usize> = group_plans.iter().map(|g| g.len()).collect();
    let n = cities.len();
    let mut merged: Vec<EmpirePlan> = Vec::new();
    for combo in cross_product_indices(&sizes) {
        let chosen: Vec<&EmpirePlan> = combo
            .iter()
            .enumerate()
            .map(|(gid, &pi)| &group_plans[gid][pi])
            .collect();

        let mut techs_union: HashSet<TechnologyType> = HashSet::new();
        let mut sum_own_bills = 0;
        let mut pop = 0;
        let mut giants = 0;
        let mut spt = 0;
        for p in &chosen {
            techs_union.extend(p.techs.iter().copied());
            sum_own_bills += tech_bill(&p.techs, owned, true_n);
            pop += p.pop;
            giants += p.giants;
            spt += p.spt;
        }
        // Sorted, not raw HashSet iteration -- see the comment on
        // `prospective_sorted` below for the exact class of nondeterminism
        // bug this avoids.
        let mut techs_union_vec: Vec<TechnologyType> = techs_union.into_iter().collect();
        techs_union_vec.sort_by_key(|t| *t as i8);
        let joint_bill = tech_bill(&techs_union_vec, owned, true_n);
        let stars = chosen.iter().map(|p| p.stars).sum::<i32>() - sum_own_bills + joint_bill;

        let mut scenarios = Vec::with_capacity(n);
        let mut city_pop = vec![0; n];
        let mut city_level = vec![0; n];
        let mut monuments = vec![0; n];
        let mut hubs = vec![None; n];
        let mut levels = vec![0; n];
        let mut markets = vec![None; n];
        let mut market_income = vec![0; n];
        // Safety: `scs[0]` exists because `held.is_empty()` already
        // returned before this is ever called (see `update`).
        scenarios.resize(n, scs[0]);
        for orig in 0..n {
            let (gid, pos) = reverse[orig];
            let p = chosen[gid];
            scenarios[orig] = p.scenarios[pos];
            city_pop[orig] = p.city_pop[pos];
            city_level[orig] = p.city_level[pos];
            monuments[orig] = p.monuments[pos];
            hubs[orig] = p.hubs[pos];
            levels[orig] = p.levels[pos];
            markets[orig] = p.markets[pos];
            market_income[orig] = p.market_income[pos];
        }

        merged.push(EmpirePlan {
            scenarios,
            stars,
            pop,
            giants,
            spt,
            city_pop,
            city_level,
            techs: techs_union_vec,
            monuments,
            hubs,
            levels,
            markets,
            market_income,
        });
    }
    merged
}

/// Forge lane, BorderGrowth on — the scenario `EXP_ELO_099` established as
/// the only one whose territory can even see a neighbour's planned partner
/// (natural scenarios never reach not-yet-owned ground at all).
fn plan_scenario() -> Scenario {
    SCENARIOS[7]
}

fn is_metal_mountain(state: &GameState, idx: i32) -> bool {
    state.resources.get(&idx).and_then(|r| r.as_ref()).is_some_and(|r| r.resource_type == ResourceType::Metal)
}

#[derive(Clone, Debug, Default)]
pub struct EcoPlanCommit {
    key: Option<(PlayerId, EcoPlanFingerprint)>,
    /// Metal tiles that partner some committed Forge hub, accumulated
    /// across turns (see module doc — momentum, not overwritten).
    mine_partners: HashSet<i32>,
    /// Which currently-held city a credited tile's credit is tied to, so
    /// losing that city can drop it again.
    owning_city_of: HashMap<i32, i32>,
}

impl EcoPlanCommit {
    /// Does `tile` currently partner a committed Forge hub?
    pub fn is_mine_partner(&self, tile: i32) -> bool {
        self.mine_partners.contains(&tile)
    }

    /// Every currently-credited tile, for diagnostics (fire-rate/scope
    /// checks) — not consumed by scoring, which only ever needs
    /// membership.
    pub fn mine_partner_tiles(&self) -> impl Iterator<Item = i32> + '_ {
        self.mine_partners.iter().copied()
    }

    /// Recompute if the plan-relevant inputs (held cities, prospective
    /// cities, owned tech) have moved on since the last call; no-op
    /// otherwise. Call once per ply from the same place `LaneState`/
    /// `TurnCounters` already get updated, never per candidate move:
    /// `rules::eco_plan::enumerate_empire` costs single-digit milliseconds
    /// even on a "+border" scenario, which is fine once a turn and ruinous
    /// evaluated per candidate (rank_plies scores dozens of candidates,
    /// tens of times per real move decision).
    ///
    /// The cheap prefix below (held/hard-invalidation/prospective/tech) runs
    /// every call, same as `MapBelief::key_of`'s "never allocates the
    /// [expensive] grids" fingerprint — only the expensive tail
    /// (`enumerate_empire_decomposed` and the credit-assignment loop) is
    /// gated on the fingerprint actually changing.
    pub fn update(&mut self, state: &GameState, player: PlayerId) {
        let Some(tribe) = state.tribes.get(&player) else {
            return;
        };
        let held: Vec<i32> = tribe.cities.iter().map(|c| c.idx).collect();
        let held_set: HashSet<i32> = held.iter().copied().collect();

        // Hard invalidation: a tile whose owning city is no longer ours
        // loses its credit — that investment is genuinely gone, not just
        // stale. Runs every call (cheap — a small HashMap retain) rather
        // than only when the fingerprint gate below trips, so a same-turn
        // loss is never missed.
        let owning_city_of = &mut self.owning_city_of;
        let mine_partners = &mut self.mine_partners;
        owning_city_of.retain(|tile, city| {
            let keep = held_set.contains(city);
            if !keep {
                mine_partners.remove(tile);
            }
            keep
        });

        if held.is_empty() {
            return;
        }

        // REAL, confirmed VILLAGES only -- `still_capturable`/
        // `retakeable_village`, not the broader `expand_target_valid` (which
        // also matches capturable Ruins: a one-time reward, not a
        // settlement, with no territory or hub potential — including one
        // here starved `enumerate_empire` of every valid combo, zero plans).
        // Deliberately not the fog-guessed sites `oracle_macro::
        // expand_targets` tops up with below `COMMIT_CITY_TARGET` cities,
        // either: a guess already carries its own, separate "does a village
        // even exist here" confidence (`VillageGuess::confidence`) that
        // `village_race_confidence` was never designed to compose with --
        // Verdi's own example was a village we could already see.
        let prospective: HashSet<i32> = state
            .structures
            .keys()
            .copied()
            .filter(|&idx| still_capturable(state, idx, player) || retakeable_village(state, idx, player))
            .filter(|&t| village_race_confidence(state, player, t) >= PROSPECTIVE_CONFIDENCE_MIN)
            .collect();

        let owned: HashSet<TechnologyType> = tribe
            .tech_vanilla
            .iter()
            .filter(|t| t.discovered)
            .map(|t| t.tech_type)
            .collect();

        // `turn` is deliberately NOT part of the gate key -- mirrors
        // `belief::map::cache::BeliefKey`, which excludes it for the same
        // reason: the plan should stay stable across a turn advance that
        // changed nothing relevant, not just across a repeat call within
        // one turn.
        let fingerprint = EcoPlanFingerprint::compute(&held, &prospective, &owned);
        let key = (player, fingerprint);
        if self.key == Some(key) {
            return;
        }
        self.key = Some(key);

        // Sorted, not raw HashSet iteration: `cities`' ORDER feeds
        // `extend_for_border_growth`'s tiebreak and `enumerate_empire`'s
        // combination indexing, and `HashSet`'s iteration order is
        // randomized per-process (`RandomState`) -- exactly the class of
        // bug EXP_ELO_091 already found and fixed elsewhere in move
        // generation. Confirmed by measurement: a paired-gauge rerun of
        // this exact code produced 39/128 then 38/128 anchor wins from the
        // identical model and seed, while the (HashSet-free) baseline
        // reproduced bit-for-bit.
        let mut prospective_sorted: Vec<i32> = prospective.iter().copied().collect();
        prospective_sorted.sort_unstable();
        let mut cities = held.clone();
        cities.extend(prospective_sorted);
        let sc = plan_scenario();
        let scs: Vec<Scenario> = cities.iter().map(|_| sc).collect();
        let terr = if prospective.is_empty() {
            allocate_value(state, &cities, &scs, 0)
        } else {
            allocate_value_with_prospective(state, &cities, &prospective, &scs, 0)
        };

        let plans = enumerate_empire_decomposed(state, &cities, &terr, &scs, &owned, 8);
        let Some(best) = pick_for_goal(&plans, Goal::Balanced) else {
            return;
        };

        let hub_sites: Vec<i32> = best.hubs.iter().flatten().copied().collect();
        if hub_sites.is_empty() {
            return;
        }
        let size = state.settings.size;
        for (ci, &city) in cities.iter().enumerate() {
            if !held_set.contains(&city) {
                continue; // only currently-buildable (held) ground can be credited now
            }
            for &t in &terr[ci] {
                if is_metal_mountain(state, t)
                    && hub_sites.iter().any(|&h| get_chebyshev_distance(t, h, size) <= 1)
                {
                    self.mine_partners.insert(t);
                    self.owning_city_of.insert(t, city);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Coords;
    use crate::states::{CityState, ResourceState, StructureState, TileState, TribeState};
    use crate::types::{StructureType, TerrainType};

    fn owned_field(owner: PlayerId, ruling: i32, size: i32) -> TileState {
        let mut t = TileState::default();
        t.terrain_type = TerrainType::Field;
        t.owner = owner;
        t.ruling_city_coords = Some(Coords::from_index(ruling, size));
        t
    }

    /// Minimal real state: one held city with a standing Mine adjacent to a
    /// Field tile that can become its Forge, and enough Crop tiles to clear
    /// the BorderGrowth reachability floor (`POP_FOR_LEVEL_4`).
    fn one_city_state() -> (GameState, PlayerId, i32, i32, i32) {
        let size = 11;
        let mut state = GameState::default();
        state.settings.size = size;
        state.settings.current_player_turn_id = 1;
        let center = 5 * size + 5;
        let hub = 5 * size + 6;
        let mine = 4 * size + 6;
        let crops = [4 * size + 4, 4 * size + 5, 5 * size + 4, 6 * size + 4];

        state.tiles.insert(center, owned_field(1, center, size));
        // A city's own tile keeps `StructureType::Village` in `structures`
        // even long after capture (confirmed against a real captured city's
        // dumped state) -- without this, `tile_options` treats the centre as
        // an ordinary empty Field and a hub can land ON the city itself.
        state.structures.insert(
            center,
            Some(StructureState { structure_type: StructureType::Village, level: 1, founded: 0 }),
        );
        state.tiles.insert(hub, owned_field(1, center, size));
        let mut mine_tile = owned_field(1, center, size);
        mine_tile.terrain_type = TerrainType::Mountain;
        state.tiles.insert(mine, mine_tile);
        state.resources.insert(mine, Some(ResourceState { resource_type: ResourceType::Metal, ..Default::default() }));
        state.structures.insert(mine, Some(StructureState { structure_type: StructureType::Mine, level: 1, founded: 0 }));
        for &c in &crops {
            state.tiles.insert(c, owned_field(1, center, size));
            state.resources.insert(c, Some(ResourceState { resource_type: ResourceType::Crop, ..Default::default() }));
        }

        let mut tribe = TribeState { id: 1, ..Default::default() };
        let mut territory: Vec<i32> = vec![center, hub, mine];
        territory.extend(crops.iter().copied());
        tribe.cities.push(CityState { idx: center, owner: 1, _territory: territory, ..Default::default() });
        state.tribes.insert(1, tribe);
        state.tribes.insert(2, TribeState { id: 2, ..Default::default() });
        (state, 1, center, hub, mine)
    }

    #[test]
    fn credits_a_mine_that_partners_the_committed_hub() {
        let (state, player, ..) = one_city_state();
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, player);
        // The standing mine feeds SOME hub in a 9-tile-plus inner ring; the
        // specific site is `build_out`'s call, this only asserts the wiring
        // (state -> plan -> credited partner set) actually produces credit.
        assert!(!commit.mine_partners.is_empty(), "expected at least one credited partner tile");
    }

    #[test]
    fn recompute_is_gated_on_the_fingerprint_not_just_the_turn() {
        let (mut state, player, ..) = one_city_state();
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, player);
        let first_key = commit.key;
        commit.update(&state, player); // same turn, nothing changed -- must be a no-op
        assert_eq!(commit.key, first_key);

        state.settings.turn += 1;
        commit.update(&state, player); // new turn, nothing relevant changed -- STILL a no-op
        assert_eq!(commit.key, first_key, "turn advancing alone must not force a recompute");
    }

    #[test]
    fn a_held_city_change_forces_recompute_even_within_the_same_turn() {
        let (mut state, player, ..) = one_city_state();
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, player);
        let first_key = commit.key;

        // Capture a second city, same turn -- a (turn, player)-only gate
        // would treat this as an unchanged key and never notice.
        let size = state.settings.size;
        let second = 8 * size + 8;
        state.tiles.insert(second, owned_field(1, second, size));
        state.structures.insert(
            second,
            Some(StructureState { structure_type: StructureType::Village, level: 1, founded: 0 }),
        );
        state.tribes.get_mut(&player).unwrap().cities.push(CityState {
            idx: second,
            owner: 1,
            _territory: vec![second],
            ..Default::default()
        });

        commit.update(&state, player);
        assert_ne!(commit.key, first_key, "a same-turn held-city change must still trigger recompute");
    }

    #[test]
    fn losing_a_city_drops_its_credited_tiles() {
        let (mut state, player, center, _hub, mine) = one_city_state();
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, player);
        assert!(commit.owning_city_of.values().any(|&c| c == center));

        state.settings.turn += 1;
        state.tribes.get_mut(&player).unwrap().cities.clear();
        commit.update(&state, player);
        assert!(!commit.is_mine_partner(mine), "credit must not survive losing the owning city");
    }

    #[test]
    fn no_tribe_or_no_cities_does_not_panic_and_credits_nothing() {
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut commit = EcoPlanCommit::default();
        commit.update(&state, 1); // no tribe at all
        assert!(commit.mine_partners.is_empty());
        state.tribes.insert(1, TribeState { id: 1, ..Default::default() });
        commit.update(&state, 1); // tribe with zero cities
        assert!(commit.mine_partners.is_empty());
    }

    /// Places the same "one city" kit `one_city_state` uses (a standing
    /// Mine feeding a hub site, plus crops clearing the BorderGrowth
    /// floor) at each anchor -- so every city is a genuine, independently
    /// plannable economy, not a placeholder, and connectivity grouping is
    /// tested against real terrain rather than bare tile indices.
    fn multi_city_state(anchors: &[(i32, i32)]) -> (GameState, Vec<i32>) {
        let size = 11;
        let mut state = GameState::default();
        state.settings.size = size;
        state.settings.current_player_turn_id = 1;
        let mut tribe = TribeState { id: 1, ..Default::default() };
        let mut centers = Vec::with_capacity(anchors.len());

        for &(r, c) in anchors {
            let center = r * size + c;
            let hub = r * size + (c + 1);
            let mine = (r - 1) * size + (c + 1);
            let crops = [
                (r - 1) * size + (c - 1),
                (r - 1) * size + c,
                r * size + (c - 1),
                (r + 1) * size + (c - 1),
            ];

            state.tiles.insert(center, owned_field(1, center, size));
            state.structures.insert(
                center,
                Some(StructureState { structure_type: StructureType::Village, level: 1, founded: 0 }),
            );
            state.tiles.insert(hub, owned_field(1, center, size));
            let mut mine_tile = owned_field(1, center, size);
            mine_tile.terrain_type = TerrainType::Mountain;
            state.tiles.insert(mine, mine_tile);
            state.resources.insert(mine, Some(ResourceState { resource_type: ResourceType::Metal, ..Default::default() }));
            state.structures.insert(mine, Some(StructureState { structure_type: StructureType::Mine, level: 1, founded: 0 }));
            for &cr in &crops {
                state.tiles.insert(cr, owned_field(1, center, size));
                state.resources.insert(cr, Some(ResourceState { resource_type: ResourceType::Crop, ..Default::default() }));
            }

            let mut territory: Vec<i32> = vec![center, hub, mine];
            territory.extend(crops.iter().copied());
            tribe.cities.push(CityState { idx: center, owner: 1, _territory: territory, ..Default::default() });
            centers.push(center);
        }

        state.tribes.insert(1, tribe);
        state.tribes.insert(2, TribeState { id: 2, ..Default::default() });
        (state, centers)
    }

    /// Comparable signature for an `EmpirePlan` -- everything `pareto`'s
    /// `dominates` reads, plus the actual hub/market siting, so two plan
    /// sets can be compared as sorted multisets without relying on
    /// `pick_for_goal`'s last-wins tie-break order (which two structurally
    /// identical "kit" cities, as these fixtures use, are prone to tie on).
    fn plan_signature(p: &EmpirePlan) -> (i32, i32, i32, i32, i32, Vec<Option<i32>>, Vec<Option<i32>>) {
        (p.stars, p.pop, p.giants, p.spt, p.monuments_used(), p.hubs.clone(), p.markets.clone())
    }

    fn assert_decomposed_matches_joint(state: &GameState, cities: &[i32]) {
        let sc = plan_scenario();
        let scs: Vec<Scenario> = cities.iter().map(|_| sc).collect();
        let terr = allocate_value(state, cities, &scs, 0);
        let owned: HashSet<TechnologyType> = HashSet::new();

        let mut joint: Vec<_> = pareto(&enumerate_empire(state, cities, &terr, &scs, &owned, 0, 8, true))
            .iter()
            .map(plan_signature)
            .collect();
        let mut decomposed: Vec<_> = pareto(&enumerate_empire_decomposed(state, cities, &terr, &scs, &owned, 8))
            .iter()
            .map(plan_signature)
            .collect();
        joint.sort();
        decomposed.sort();
        assert_eq!(decomposed, joint, "decomposed Pareto frontier must match the joint search exactly");
    }

    #[test]
    fn decomposed_matches_joint_search_for_a_single_tight_cluster() {
        let (state, cities) = multi_city_state(&[(2, 2), (2, 5)]);
        assert_eq!(connectivity_groups(&state, &cities).len(), 1, "fixture must NOT trigger decomposition");
        assert_decomposed_matches_joint(&state, &cities);
    }

    #[test]
    fn decomposed_matches_joint_search_for_a_cluster_plus_a_far_outlier() {
        let (state, cities) = multi_city_state(&[(2, 2), (2, 5), (8, 8)]);
        assert_eq!(connectivity_groups(&state, &cities).len(), 2, "fixture must actually exercise decomposition");
        assert_decomposed_matches_joint(&state, &cities);
    }

    #[test]
    fn decomposed_matches_joint_search_for_two_separate_pairs() {
        let (state, cities) = multi_city_state(&[(2, 2), (2, 5), (8, 2), (8, 5)]);
        assert_eq!(connectivity_groups(&state, &cities).len(), 2, "fixture must actually exercise decomposition");
        assert_decomposed_matches_joint(&state, &cities);
    }
}
