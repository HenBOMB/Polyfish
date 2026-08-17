//! Economic lane planning: what a Tier-1 playstyle banks toward, the star
//! cost of reaching it, and the environment-fit tech recommendations that
//! feed T2's ballot. Consumed by `oracle_macro`'s orchestration (which
//! re-exports the public items below so existing `crate::ai::oracle_macro::X`
//! call sites keep resolving) and by `reward.rs`'s savings-ramp pricing.

use crate::ai::oracle_macro::{tribe_lane_prior, Lane, TIER3_CAP_PER_GAME};
use crate::moves::Move;
use crate::states::{GameState, PlayerId};
use crate::types::{MoveType, TechnologyType};

/// The economy batch a SAVE stance is banking for, with the lane it belongs
/// to.
///
/// v10: this used to be a bare `Option<i32>` — `pick_save_lane` identified
/// the lane and then discarded everything but the price, so nothing
/// downstream could tell "saving for a Forge" from "saving for 21 stars".
/// Search could not boost the very move the plan existed to reach.
#[derive(Clone, Debug, PartialEq)]
pub struct SaveTarget {
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
///
/// Superseded by `lane_save_structure` (EXP_ELO_052, the lane is now stated
/// directly instead of inferred) but kept — some callers may still want the
/// raw investment count rather than the direct mapping.
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
pub fn lane_save_structure(a: Lane) -> crate::types::StructureType {
    use crate::types::StructureType as S;
    match a {
        Lane::RiderRoads => S::Market,
        Lane::ArcherLine => S::Sawmill,
        Lane::ForgeGiants => S::Forge,
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
pub fn pick_save_lane(
    state: &GameState,
    player: PlayerId,
    tier3_bought: u32,
    committed: Option<Lane>,
) -> Option<SaveTarget> {
    // Before the selector has committed, the spawn tribe tech already says
    // which lane this tribe is born into — so the plan is right from ply one
    // and every caller resolves it identically.
    let committed = committed.or_else(|| tribe_lane_prior(state, player));
    use crate::settings::structures::get_structure_setting;
    use crate::settings::technology::has_technology;
    use crate::types::StructureType;
    let tribe = state.tribes.get(&player)?;
    let mut best: Option<(SaveTarget, i32)> = None;
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
        let plan = SaveTarget {
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
        // in `passes_tech_purchase_limits`.
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
        let better = best.as_ref().map_or(true, |(b, bi): &(SaveTarget, i32)| {
            rank > *bi || (rank == *bi && plan.cost < b.cost)
        });
        if better {
            best = Some((plan, rank));
        }
    }
    best.map(|(p, _)| p)
}

/// Does `m` advance the banked plan? The whole undiscovered `requires` chain
/// counts, not just the final tech — `pick_save_lane` prices the chain
/// (Market sits behind Roads behind Riding), so boosting only the last step
/// would leave every multi-step lane exactly as stuck as it is today.
///
/// Structurally inert while banking: a Research/Build move is only generated
/// once it is affordable, so this fires exactly when the purchase goes live.
pub fn advances_save_plan(m: &dyn Move, lane: &SaveTarget, tribe: &crate::states::TribeState) -> bool {
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

/// Environment-fit tech lines. Returns the next unowned tech of the top two
/// lines, ranked by `evaluator::research::evaluate_tech_utility` — the same
/// per-tech ROI score `scoring.rs`'s Research-move heuristic and the
/// evaluator's unused-tech penalty already use, so a line's rank and its
/// move-scoring priority can no longer disagree (EXP_ELO_055; before this
/// the rank came from a separately hand-tuned terrain/resource census).
///
/// Chain semantics are preserved deliberately: each line is still scored as
/// a WHOLE (via its own next-unowned tech), never by ranking every tech in
/// the game individually — that would recommend a downstream tech (e.g.
/// Mathematics) ahead of its own line's earlier, unowned prerequisite.
///
/// EXP_ELO_055 follow-up: `evaluate_tech_utility` counts resources/terrain
/// from CITY TERRITORY, which is too little ground to rank reliably before a
/// second city exists — measured cost was +10 off-lane techs across 48
/// games (sign test p≈0.011), from the committed lane's own next tech
/// sometimes losing to an off-lane pickup in exactly this window. Below two
/// cities, fall back to the old map-wide explored-tile census (the same
/// scoring this function used before EXP_ELO_055); switch to the
/// territory-scoped ROI signal once there's a second city's worth of ground
/// to score it against.
pub fn recommended_techs(state: &GameState, player: PlayerId) -> Vec<TechnologyType> {
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    use TechnologyType as Tech;
    let forest_line: &[Tech] = &[Tech::Hunting, Tech::Forestry, Tech::Mathematics];
    let mountain_line: &[Tech] = &[Tech::Climbing, Tech::Mining, Tech::Smithery];
    let farm_line: &[Tech] = &[Tech::Organization, Tech::Farming, Tech::Construction];
    let water_line: &[Tech] = &[Tech::Fishing];
    let lines: [&[Tech]; 4] = [forest_line, mountain_line, farm_line, water_line];
    let next_unowned = |line: &[Tech]| -> Option<Tech> {
        line.iter()
            .copied()
            .find(|t| !crate::settings::technology::is_tech_unlocked(&tribe.tech_vanilla, *t))
    };

    if tribe.cities.len() < 2 {
        use crate::types::{ResourceType as R, TerrainType as T};
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
        let census: [i32; 4] = [
            forest + 2 * game_r,
            mountain + 2 * metal,
            field / 2 + 2 * (fruit + crop),
            water / 2 + 2 * fish,
        ];
        let mut ranked: Vec<(i32, usize)> = census.into_iter().zip(0..4).collect();
        ranked.sort_by_key(|(score, _)| -*score);
        return ranked
            .into_iter()
            .take(2)
            .filter(|(score, _)| *score > 0)
            .filter_map(|(_, i)| next_unowned(lines[i]))
            .collect();
    }

    let mut scored: Vec<(f32, Tech)> = lines
        .iter()
        .filter_map(|line| {
            let next = next_unowned(line)?;
            let score = crate::ai::evaluator::research::evaluate_tech_utility(state, player, next);
            Some((score, next))
        })
        .collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    scored
        .into_iter()
        .take(2)
        .filter(|(score, _)| *score >= 0.0)
        .map(|(_, t)| t)
        .collect()
}
