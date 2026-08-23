//! Economic lane planning: what a Tier-1 playstyle banks toward, the star
//! cost of reaching it, and the environment-fit tech recommendations that
//! feed T2's ballot. Consumed by `oracle_macro`'s orchestration (which
//! re-exports the public items below so existing `crate::ai::oracle_macro::X`
//! call sites keep resolving) and by `reward.rs`'s savings-ramp pricing.

use crate::ai::oracle_macro::TIER3_CAP_PER_GAME;
use crate::moves::Move;
use crate::states::{GameState, PlayerId, TribeState};
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

/// The `rules::eco_plan` lane a save-lane structure belongs to, or `None` for
/// Market — eco_plan has no Trade/Market lane (its 3 lanes are Forest/Farm/
/// Mine, keyed by which resource-hub they build), so Market keeps its own
/// path in `pick_save_lane` rather than a fabricated mapping.
fn eco_plan_lane_for_structure(s: crate::types::StructureType) -> Option<crate::rules::eco_plan::Lane> {
    use crate::rules::eco_plan::Lane as EcoLane;
    use crate::types::StructureType as S;
    match s {
        S::Sawmill => Some(EcoLane::Forest),
        S::Windmill => Some(EcoLane::Farm),
        S::Forge => Some(EcoLane::Mine),
        _ => None,
    }
}

/// EXP_ELO_057: the tribe's best-city reading for one `rules::eco_plan` hub
/// lane — `plan_city`, the same single source of truth `bin/eco_plan --verify`
/// checks against, replacing hand-rolled terrain/yield estimates that used to
/// answer this question independently in two places (`lane_scores`'s giants
/// term and this file's old `lane_yield_per_star`). The "natural" scenario
/// (no BorderGrowth speculation, no terrain conversion) matches what
/// `pick_save_lane` already scopes to — ground the tribe controls right now.
/// `None` with no cities yet: nothing to plan against.
pub fn eco_plan_best_city(
    state: &GameState,
    player: PlayerId,
    lane: crate::rules::eco_plan::Lane,
) -> Option<crate::rules::eco_plan::CityPlan> {
    use crate::rules::eco_plan::{plan_city, SCENARIOS};
    let tribe = state.tribes.get(&player)?;
    if tribe.cities.is_empty() {
        return None;
    }
    let sc = *SCENARIOS
        .iter()
        .find(|s| s.lane == lane && !s.border_growth && !s.convert)?;
    let owned: std::collections::HashSet<TechnologyType> = tribe
        .tech_vanilla
        .iter()
        .filter(|t| t.discovered)
        .map(|t| t.tech_type)
        .collect();
    let num_cities = tribe.cities.len() as i32;
    tribe
        .cities
        .iter()
        .map(|c| plan_city(state, c.idx, &c._territory, sc, &owned, num_cities, 0))
        .max_by(|a, b| a.spt.cmp(&b.spt).then(a.giants.cmp(&b.giants)))
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
) -> Option<SaveTarget> {
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
        // Pareto: star-per-turn yield ON THIS MAP, from `rules::eco_plan`'s
        // own best-city reading — Market has no eco_plan lane and keeps its
        // old rank of 0 (its hub pays stars, not `reward_pop`, so `pop` was
        // always 0 under the pre-eco_plan formula too; this is not a
        // behavior change for Market). Scaled to an integer so the existing
        // tie-break on price still applies between equals.
        let rank = match eco_plan_lane_for_structure(s_type) {
            Some(eco_lane) => eco_plan_best_city(state, player, eco_lane)
                .filter(|p| p.stars > 0)
                .map_or(0, |p| ((p.spt as f32 / p.stars as f32) * 1000.0) as i32),
            None => 0,
        };
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
/// Line index -> the `eco_plan::Lane` it maps to, where eco_plan models it.
/// Water isn't modeled (no `Lane` variant for it), so `lane_confirmed_placeable`
/// is never consulted for it and it keeps its pre-fix behavior unchanged.
fn eco_plan_lane_for_line(line_idx: usize) -> Option<crate::rules::eco_plan::Lane> {
    use crate::rules::eco_plan::Lane;
    match line_idx {
        0 => Some(Lane::Forest),
        1 => Some(Lane::Mine),
        2 => Some(Lane::Farm),
        _ => None, // water_line
    }
}

/// Discount applied to a tech line's recommendation score when `eco_plan`
/// can't confirm a hub is placeable anywhere within reach yet -- Verdi's
/// critique (Aug 23, 2026): a raw terrain census scores tile density the
/// same whether or not a real Windmill/Sawmill/Forge site actually exists,
/// so a line whose payoff is still speculative (buy the tech, THEN find out
/// if a hub fits) can out-rank one whose payoff is already verifiable from
/// the current map. Halved, not zeroed -- an unconfirmed line still has
/// standalone value (a bare Farm still pays population with no Windmill).
const UNVERIFIED_LANE_DISCOUNT: f32 = 0.5;

fn lane_confirmed_placeable(
    state: &GameState,
    tribe: &TribeState,
    lane: crate::rules::eco_plan::Lane,
) -> bool {
    use crate::rules::eco_plan::city::{city_square, lane_can_place_hub};
    tribe
        .cities
        .iter()
        .any(|c| lane_can_place_hub(state, &city_square(state, c.idx), lane))
}

/// Line score after the eco_plan feasibility discount, keyed by line index
/// so both branches below (census and utility-scored) apply it identically.
fn discount_unverified(state: &GameState, tribe: &TribeState, line_idx: usize, score: f32) -> f32 {
    match eco_plan_lane_for_line(line_idx) {
        Some(lane) if !lane_confirmed_placeable(state, tribe, lane) => score * UNVERIFIED_LANE_DISCOUNT,
        _ => score,
    }
}

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
        let mut ranked: Vec<(f32, usize)> = census
            .into_iter()
            .zip(0..4)
            .map(|(score, i)| (discount_unverified(state, tribe, i, score as f32), i))
            .collect();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
        return ranked
            .into_iter()
            .take(2)
            .filter(|(score, _)| *score > 0.0)
            .filter_map(|(_, i)| next_unowned(lines[i]))
            .collect();
    }

    let mut scored: Vec<(f32, Tech)> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let next = next_unowned(line)?;
            let score = crate::ai::evaluator::research::evaluate_tech_utility(state, player, next);
            Some((discount_unverified(state, tribe, i, score), next))
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

#[cfg(test)]
mod recommended_techs_eco_plan_tests {
    use super::*;
    use crate::coords::Coords;
    use crate::rules::eco_plan::Lane;
    use crate::states::{CityState, ResourceState, TileState};
    use crate::types::{ResourceType, TerrainType};

    fn base_state() -> GameState {
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut tribe = TribeState::default();
        tribe.cities.push(CityState { idx: 5 * 11 + 5, owner: 1, ..Default::default() });
        state.tribes.insert(1, tribe);
        state
    }

    fn field_tile(player: i32) -> TileState {
        let mut t = TileState::default();
        t.terrain_type = TerrainType::Field;
        t.explorers.insert(player);
        t
    }

    /// Unit-level check on the new helper directly: two adjacent Metal
    /// tiles confirm Mine; two isolated Crop tiles (no shared neighbor)
    /// do not confirm Farm.
    #[test]
    fn lane_confirmed_placeable_matches_lane_can_place_hub() {
        let mut state = base_state();
        // Metal cluster within city_square (radius 2 of idx 60).
        for idx in [4 * 11 + 4, 4 * 11 + 5, 4 * 11 + 6] {
            state.tiles.insert(idx, field_tile(1));
        }
        state.resources.insert(4 * 11 + 4, Some(ResourceState { resource_type: ResourceType::Metal }));
        state.resources.insert(4 * 11 + 6, Some(ResourceState { resource_type: ResourceType::Metal }));
        // Two isolated Crop tiles, far enough apart to share no neighbor.
        for idx in [3 * 11 + 3, 7 * 11 + 7] {
            state.tiles.insert(idx, field_tile(1));
        }
        state.resources.insert(3 * 11 + 3, Some(ResourceState { resource_type: ResourceType::Crop }));
        state.resources.insert(7 * 11 + 7, Some(ResourceState { resource_type: ResourceType::Crop }));

        let tribe = state.tribes.get(&1).unwrap();
        assert!(lane_confirmed_placeable(&state, tribe, Lane::Mine));
        assert!(!lane_confirmed_placeable(&state, tribe, Lane::Farm));
    }

    /// Integration-level check: a speculative line that out-scores TWO
    /// verified ones on raw census must not silently monopolize a
    /// recommendation slot -- this is the exact shape of the
    /// Organization-vs-Smithery incident (turn 7-8, seed 1787434721) that
    /// motivated the fix. `recommended_techs`'s only consumers
    /// (`goal_potential`'s tech-fit term, `passes_stance_tech_mask`) both
    /// test *set membership*, never order, so a fixture with only two
    /// nonzero lines can't actually exercise a behavior change (whichever
    /// order they come back in, both still make the top-2 cut). Three
    /// lines here, engineered so Farm's raw census beats Mountain AND
    /// Forest, but only Mountain and Forest have a confirmed hub site --
    /// the discount must swap Farm out of the top 2 for Forest entirely,
    /// not just reorder it.
    #[test]
    fn unverified_line_can_be_displaced_out_of_the_top_two() {
        let mut state = base_state();

        // Mountain: 2 metal tiles adjacent to a shared tile -- confirmed.
        // Census: 3 terrain + 2*2 metal = 7.
        for idx in [4 * 11 + 4, 4 * 11 + 5, 4 * 11 + 6] {
            let mut t = TileState::default();
            t.terrain_type = TerrainType::Mountain;
            t.explorers.insert(1);
            state.tiles.insert(idx, t);
        }
        state.resources.insert(4 * 11 + 4, Some(ResourceState { resource_type: ResourceType::Metal }));
        state.resources.insert(4 * 11 + 6, Some(ResourceState { resource_type: ResourceType::Metal }));

        // Forest: 2 forest tiles adjacent to a shared tile -- confirmed.
        // Census: 3 terrain + 2*1 game = 5.
        for idx in [7 * 11 + 4, 7 * 11 + 5, 7 * 11 + 6] {
            let mut t = TileState::default();
            t.terrain_type = TerrainType::Forest;
            t.explorers.insert(1);
            state.tiles.insert(idx, t);
        }
        state.resources.insert(7 * 11 + 4, Some(ResourceState { resource_type: ResourceType::Game }));

        // Farm: three ISOLATED crop tiles -- the three far corners of the
        // radius-2 city_square, each a Chebyshev distance of 4 from the
        // other two, so no single tile (set or not) is adjacent to more
        // than one of them -- plus a bare field tile. Highest raw census
        // (8), but no hub site anywhere.
        for idx in [3 * 11 + 3, 3 * 11 + 7, 7 * 11 + 3, 5 * 11 + 3] {
            state.tiles.insert(idx, field_tile(1));
        }
        for idx in [3 * 11 + 3, 3 * 11 + 7, 7 * 11 + 3] {
            state.resources.insert(idx, Some(ResourceState { resource_type: ResourceType::Crop }));
        }

        let tribe = state.tribes.get(&1).unwrap();
        assert!(lane_confirmed_placeable(&state, tribe, Lane::Mine));
        assert!(lane_confirmed_placeable(&state, tribe, Lane::Forest));
        assert!(!lane_confirmed_placeable(&state, tribe, Lane::Farm));

        let recs = recommended_techs(&state, 1);
        assert_eq!(
            recs,
            vec![TechnologyType::Climbing, TechnologyType::Hunting],
            "undiscounted census ranks Farm(8) > Mountain(7) > Forest(5) -- Farm would win \
             a slot outright without the fix. Discounted: Mountain(7) > Forest(5) > \
             Farm(4), so Farm must be fully displaced by Forest, not just reordered: {recs:?}"
        );
    }
}
