//! City growth-capacity helpers (Aug 2026 taxonomy split out of reward.rs to
//! keep every file under ~1000 lines): how much population a city could still
//! buy from remaining territory routes (`max_affordable_pop`), whether its
//! next level is structurally reachable at all (`city_completable*`), the
//! stranded-progress penalty and completion-progress bonus `goal_potential`
//! prices, and the threatened-city predicate the defend terms share.
//! Re-exported through `reward` so existing `crate::ai::reward::X` call
//! sites keep resolving.

use super::cheb;
use crate::states::GameState;

/// Adjacency-multiplier tier: each pays `reward_pop × friendly adjacent
/// partners` (see `actions::structure::build_structure`), one per city.
const MULTIPLIER_TIER: [crate::types::StructureType; 3] = [
    crate::types::StructureType::Windmill,
    crate::types::StructureType::Sawmill,
    crate::types::StructureType::Forge,
];

/// Pop-bearing terrain structures that need neither a resource nor a partner.
const TERRAIN_POP_TIER: [crate::types::StructureType; 5] = [
    crate::types::StructureType::Temple,
    crate::types::StructureType::MountainTemple,
    crate::types::StructureType::WaterTemple,
    crate::types::StructureType::ForestTemple,
    crate::types::StructureType::Port,
];

/// v7: max population this city could still buy, derived from the settings
/// tables — every pop route the engine actually offers: resource harvests and
/// their structures, LumberHuts on owned forest, the adjacency-multiplier tier
/// (Windmill/Sawmill/Forge, whose yield scales with friendly partners the city
/// could still build), and pop-bearing terrain structures. Greedy
/// cheapest-per-pop knapsack under `stars`.
///
/// v6 counted resource tiles only, which mislabelled most level-4+ cities as
/// dead ends — the level threshold grows as N+1 while resource tiles are finite
/// and get consumed, so the multiplier tier IS the late-game pop engine. That
/// omission taxed exactly the cities that make SPT (see the v6 regression
/// diagnosis in the ledger).
///
/// Deliberately excludes pending monuments: free and worth 3 pop, but
/// `check_task` over every TaskType is too costly for this leaf-path helper.
/// The omission only ever makes the predicate more pessimistic.
pub fn max_affordable_pop(state: &GameState, player: i32, city: &crate::states::CityState, stars: i32) -> i32 {
    use crate::functions::{get_adjacent_indices, get_structure_at, get_structure_type_at};
    use crate::settings::structures::get_structure_setting;
    let Some(tribe) = state.tribes.get(&player) else {
        return 0;
    };
    let owned = |tech: crate::types::TechnologyType| {
        tribe
            .tech_vanilla
            .iter()
            .any(|t| t.tech_type == tech && t.discovered)
    };
    // Mirrors the build-legality gate: empty, and not sat on by an enemy.
    let open = |idx: i32| {
        get_structure_at(state, idx).is_none()
            && crate::functions::get_enemy_at(state, idx, player).is_none()
    };
    let mut options: Vec<(i32, i32)> = Vec::new(); // (cost, pop)
    // Tiles a cheaper pass already spoke for, and what would stand there — the
    // multiplier tier counts partners this city could still build, not just
    // the ones standing today.
    let mut claimed: std::collections::HashMap<i32, Option<crate::types::StructureType>> =
        std::collections::HashMap::new();

    for idx in crate::rules::economy::territory_tiles(state, city) {
        if !open(idx) {
            continue;
        }
        if let Some(Some(res)) = state.resources.get(&idx) {
            let setting = crate::settings::resources::get_resource_setting(res.resource_type);
            if setting.reward_pop <= 0 || setting.requires_capture || !owned(setting.tech_required)
            {
                continue;
            }
            match setting.struct_required {
                None => {
                    if let Some(cost) = setting.cost {
                        options.push((cost, setting.reward_pop));
                        claimed.insert(idx, None);
                    }
                }
                Some(s_type) => {
                    let s = get_structure_setting(s_type);
                    if let Some(cost) = s.cost {
                        options.push((cost, s.reward_pop));
                        claimed.insert(idx, Some(s_type));
                    }
                }
            }
        } else if owned(crate::types::TechnologyType::Forestry) {
            let is_forest = state
                .tiles
                .get(&idx)
                .map_or(false, |t| t.terrain_type == crate::types::TerrainType::Forest);
            if is_forest {
                let hut = get_structure_setting(crate::types::StructureType::LumberHut);
                if let Some(cost) = hut.cost {
                    options.push((cost, hut.reward_pop));
                    claimed.insert(idx, Some(crate::types::StructureType::LumberHut));
                }
            }
        }
    }

    for m_type in MULTIPLIER_TIER {
        if !crate::moves::build::is_structure_unlocked(tribe, m_type) {
            continue;
        }
        let s = get_structure_setting(m_type);
        let Some(cost) = s.cost else { continue };
        if s.limited_per_city
            && city
                ._territory
                .iter()
                .any(|&t| get_structure_type_at(state, t) == Some(m_type))
        {
            continue;
        }
        // Best placement in this city — yield scales with partner count.
        let mut best = 0;
        for idx in crate::rules::economy::territory_tiles(state, city) {
            if claimed.contains_key(&idx) || !open(idx) {
                continue;
            }
            let Some(tile) = state.tiles.get(&idx) else { continue };
            if !s.terrain_types.contains(&tile.terrain_type) || tile.is_algae() {
                continue;
            }
            let partners = get_adjacent_indices(state, idx, 1)
                .iter()
                .filter(|&&n| {
                    state.tiles.get(&n).map_or(false, |t| t.owner == player)
                        && match claimed.get(&n) {
                            Some(Some(planned)) => s.adjacent_types.contains(planned),
                            _ => get_structure_type_at(state, n)
                                .map_or(false, |st| s.adjacent_types.contains(&st)),
                        }
                })
                .count() as i32;
            best = best.max(partners);
        }
        if best > 0 {
            options.push((cost, s.reward_pop * best));
        }
    }

    for t_type in TERRAIN_POP_TIER {
        if !crate::moves::build::is_structure_unlocked(tribe, t_type) {
            continue;
        }
        let s = get_structure_setting(t_type);
        let Some(cost) = s.cost else { continue };
        if s.reward_pop <= 0 {
            continue;
        }
        for idx in crate::rules::economy::territory_tiles(state, city) {
            if claimed.contains_key(&idx) || !open(idx) {
                continue;
            }
            let Some(tile) = state.tiles.get(&idx) else { continue };
            if s.terrain_types.contains(&tile.terrain_type) && !tile.is_algae() {
                options.push((cost, s.reward_pop));
                claimed.insert(idx, Some(t_type));
            }
        }
    }

    // Cheapest stars-per-pop first.
    options.sort_by(|a, b| (a.0 * b.1).cmp(&(b.0 * a.1)));
    let mut budget = stars;
    let mut pop = 0;
    for (cost, p) in options {
        if cost <= budget {
            budget -= cost;
            pop += p;
        }
    }
    pop
}

/// A city can still finish its next level from remaining territory routes at
/// any star budget. Stars are deliberately excluded — a stars-dependent
/// predicate turns every purchase into a potential flip and taxes the whole
/// economy (v6 first-fit, −12.5pp win rate).
fn city_completable(state: &GameState, player: i32, city: &crate::states::CityState) -> bool {
    city.progress + max_affordable_pop(state, player, city, i32::MAX) >= city.level + 1
}

/// v8: can this city reach its next level THIS TURN, out of the stars in hand?
/// Distinct from `city_completable`, which asks whether the level is reachable
/// at ANY star budget — that is the structural stranding question, this is the
/// spend-timing one the pop-discipline gate acts on.
pub fn city_completable_now(
    state: &GameState,
    player: i32,
    city: &crate::states::CityState,
    stars: i32,
) -> bool {
    city.progress + max_affordable_pop(state, player, city, stars) >= city.level + 1
}

/// Limits the stranded-growth penalty above to one point per city, no
/// matter how much progress is stuck there.
pub const STRANDED_PER_CITY_CAP: i32 = 1;

/// v6: STRANDED PROGRESS — cities whose next level cannot complete from
/// remaining territory routes. Threatened cities exempt (Verdi's
/// harvest-under-threat case).
pub fn completion_stranded(state: &GameState, player: i32) -> i32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0;
    };
    tribe
        .cities
        .iter()
        .filter(|c| {
            c.progress > 0
                && !city_completable(state, player, c)
                && !city_threatened(state, player, c.idx)
        })
        .map(|c| c.progress.min(STRANDED_PER_CITY_CAP))
        .sum()
}

/// v7 completion BONUS (the registered replacement for a deeper penalty —
/// end-stranded sat at 70% after v6, above the 60% trigger). Progress toward a
/// REACHABLE level pays a fraction of the way there, so each pop point into a
/// completable city is a gain rather than a neutral step.
///
/// Fractional by design: the term is worth at most `progress/(level+1) < 1`
/// unit, always less than the `SHAPE_GOAL_SPT` jump a level-up banks, so
/// levelling up (which resets progress to overflow) is never self-defeating.
/// Paying a flat amount per pop point would recreate exactly that trap.
pub fn completion_progress(
    state: &GameState,
    player: i32,
    belief: Option<&crate::ai::belief::map::MapBelief>,
) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    tribe
        .cities
        .iter()
        .filter(|c| c.progress > 0 && city_completable(state, player, c))
        .map(|c| {
            let base = c.progress as f32 / (c.level + 1).max(1) as f32;
            base * city_completion_weight(state, belief, c)
        })
        .sum()
}

/// How much finishing THIS city's next level is worth, beyond raw progress:
/// a frontier-facing city's reward pick is worth more than a backline
/// city's, and that gap matters most while the reward menu is still ahead
/// of it (low level), not already spent. 1.0 (neutral, legacy) without a
/// belief to weigh against.
fn city_completion_weight(
    state: &GameState,
    belief: Option<&crate::ai::belief::map::MapBelief>,
    city: &crate::states::CityState,
) -> f32 {
    let Some(belief) = belief else {
        return 1.0;
    };
    let avg = super::goal_shape_consts::avg_frontier_in_reach(
        state,
        belief,
        city.idx,
        super::goal_shape_consts::COMPLETION_FRONTIER_RANGE,
    );
    let frontier_factor = 1.0
        + super::goal_shape_consts::COMPLETION_FRONTIER_W
            * (avg - super::goal_shape_consts::FRONTIER_W_FOG).max(0.0);
    let level_factor = super::goal_shape_consts::COMPLETION_LEVEL_DECAY.powi(city.level.max(0));
    frontier_factor * level_factor
}

/// v6: a city is threatened when a visible enemy unit stands within
/// Chebyshev 2 of it — same radius as the Defend-order predicate. Used by
/// the level-completion discipline (harvest-under-threat is exempt) and
/// the level-completion dump.
pub fn city_threatened(state: &GameState, player: i32, city_idx: i32) -> bool {
    let width = state.settings.size as i32;
    if width == 0 {
        return false;
    }
    state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .flat_map(|(_, t)| t.units.iter())
        .any(|u| {
            let idx = u.coords.idx;
            cheb(idx, city_idx, width) <= 2
                && state
                    .tiles
                    .get(&idx)
                    .map_or(false, |t| t.explorers.contains(&player))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::goal_potential::goal_potential;
    use super::super::goal_shape_consts::{SHAPE_GOAL_COMPLETION, SHAPE_GOAL_SPT, SHAPE_GOAL_STRANDED};
    use super::super::test_support::*;
    use crate::states::{TileState, TribeState};
    use crate::types::UnitType;

    /// Builds a 1-city tribe holding `techs`, with `territory` owned by it.
    fn city_with(techs: &[crate::types::TechnologyType], territory: Vec<i32>) -> GameState {
        use crate::states::TechnologyState;
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.id = 1;
        for &tech in techs {
            t1.tech_vanilla.push(TechnologyState {
                tech_type: tech,
                discovered: true,
                discovered_turn: 0,
            });
        }
        let mut city = crate::states::CityState { idx: 60, owner: 1, ..Default::default() };
        city.level = 4;
        city._territory = territory.clone();
        t1.cities.push(city);
        state.tribes.insert(1, t1);
        for idx in territory {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.owner = 1;
            tile.terrain_type = crate::types::TerrainType::Field;
        }
        state
    }
    #[test]
    fn stranded_progress_penalizes_only_uncompletable_unthreatened_cities() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::states::{ResourceState, TechnologyState};
        use crate::types::{ResourceType, TechnologyType};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.stars = 0;
        let mut city = crate::states::CityState { idx: 60, owner: 1, ..Default::default() };
        city.level = 2;
        city.progress = 1; // needs 3 to level; nothing affordable at 0 stars
        city._territory = vec![60, 61];
        t1.cities.push(city);
        state.tribes.insert(1, t1);
        // Explored up front so later phases don't shift the scout term.
        state.tiles.entry(61).or_insert_with(TileState::default).explorers.insert(1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };

        // No resources anywhere: progress 1 is structurally stranded.
        let stranded = goal_potential(&state, 1, &grow, None);
        // ARM is exempt from the discipline (and from GROW-only terms).
        assert!(goal_potential(&state, 1, &arm, None).abs() < 1e-4);

        // v7: FLAGGED, not billed by depth — deeper sunk progress costs no
        // more, so a level-up landing in overflow cannot out-cost its own SPT.
        state.tribes.get_mut(&1).unwrap().cities[0].progress = 2;
        let deeper = goal_potential(&state, 1, &grow, None);
        assert!(
            (stranded - deeper).abs() < 1e-3,
            "stranded penalty is capped per city, not summed over sunk progress"
        );
        state.tribes.get_mut(&1).unwrap().cities[0].progress = 1;

        // Remaining territory resources covering the need lift the penalty
        // REGARDLESS of stars (structural predicate): 2 fruit → 1+2 >= 3.
        state.tribes.get_mut(&1).unwrap().tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Organization,
            discovered: true,
            discovered_turn: 0,
        });
        state.resources.insert(61, Some(ResourceState { resource_type: ResourceType::Fruit }));
        state.tribes.get_mut(&1).unwrap().cities[0]._territory.push(62);
        state.resources.insert(62, Some(ResourceState { resource_type: ResourceType::Fruit }));
        let completable = goal_potential(&state, 1, &grow, None);
        assert!(
            (completable - stranded - SHAPE_GOAL_STRANDED - SHAPE_GOAL_COMPLETION / 3.0).abs()
                < 1e-3,
            "completable progress drops the penalty AND earns the v7 bonus (1 of 3 pop)"
        );

        // Threatened city (enemy adjacent) is exempt even when stranded.
        state.resources.insert(61, None);
        state.resources.insert(62, None);
        let restranded = goal_potential(&state, 1, &grow, None);
        assert!((restranded - stranded).abs() < 1e-3);
        // EXP_ELO_114: garrison the city before the enemy shows up, and
        // re-baseline against a garrisoned reading -- an ungarrisoned city
        // with a visible adjacent enemy also trips the new
        // `city_open_exposed` term, which is a real, separate signal this
        // test isn't the place to isolate. Garrisoning keeps `open=false`
        // throughout so that term stays at zero on both sides of the
        // comparison, leaving only the threat-exemption delta this test
        // actually checks.
        state.tiles.entry(60).or_insert_with(TileState::default);
        let mut garrison = unit_at(60, UnitType::Warrior);
        garrison.owner = 1;
        state.tribes.get_mut(&1).unwrap().units.push(garrison);
        let restranded_garrisoned = goal_potential(&state, 1, &grow, None);
        let mut t2 = TribeState::default();
        t2.id = 2;
        t2.units.push(unit_at(61, UnitType::Warrior));
        state.tribes.insert(2, t2);
        let threatened = goal_potential(&state, 1, &grow, None);
        assert!(
            (threatened - restranded_garrisoned - SHAPE_GOAL_STRANDED).abs() < 1e-3,
            "threat exemption must lift the penalty"
        );
    }
    /// The v6 trap, guarded: a level-up zeroes progress (or leaves overflow),
    /// so if banked progress were worth more than the SPT jump, growing would
    /// be self-defeating. Both v7 progress terms must stay under one level.
    #[test]
    fn completion_terms_never_outweigh_the_level_up_they_pay_for() {
        // Worst case for the bonus: progress one short of the threshold.
        for level in 1..8 {
            let held = level as f32 / (level + 1) as f32;
            assert!(
                SHAPE_GOAL_COMPLETION * held < SHAPE_GOAL_SPT,
                "level {level}: banked completion bonus must not exceed the +1 SPT a level-up banks"
            );
        }
        // Worst case for the penalty: a level-up landing in stranded overflow.
        assert!(
            SHAPE_GOAL_STRANDED * (STRANDED_PER_CITY_CAP as f32) < SHAPE_GOAL_SPT,
            "a stranded landing must not out-cost the level-up that caused it"
        );
    }
    /// v7 regression: v6 saw only resource tiles, so a city whose crops were
    /// already farmed read as a dead end even with a Windmill available.
    #[test]
    fn max_affordable_pop_counts_multiplier_tier_by_partner_count() {
        use crate::types::{StructureType, TechnologyType, TerrainType};
        let techs = [TechnologyType::Organization, TechnologyType::Farming, TechnologyType::Construction];
        // Three standing Farms around 61, which is an empty Field.
        let mut state = city_with(&techs, vec![60, 61, 50, 72, 62]);
        for idx in [50, 72, 62] {
            state.structures.insert(
                idx,
                Some(crate::states::StructureState {
                    structure_type: StructureType::Farm,
                    ..Default::default()
                }),
            );
        }
        let city = state.tribes[&1].cities[0].clone();
        // Windmill on 61 pays 1 x 3 friendly adjacent Farms.
        assert_eq!(
            max_affordable_pop(&state, 1, &city, i32::MAX),
            3,
            "windmill yield must scale with adjacent friendly farms"
        );

        // Without Construction the windmill is not unlocked and nothing is left.
        let bare = city_with(
            &[TechnologyType::Organization, TechnologyType::Farming],
            vec![60, 61, 50, 72, 62],
        );
        let mut bare = bare;
        for idx in [50, 72, 62] {
            bare.structures.insert(
                idx,
                Some(crate::states::StructureState {
                    structure_type: StructureType::Farm,
                    ..Default::default()
                }),
            );
        }
        let bare_city = bare.tribes[&1].cities[0].clone();
        assert_eq!(max_affordable_pop(&bare, 1, &bare_city, i32::MAX), 0);

        // A Mountain tile with Meditation adds a 20-star temple pop.
        let mut with_temple = city_with(
            &[TechnologyType::Climbing, TechnologyType::Philosophy],
            vec![60, 40],
        );
        with_temple.tiles.get_mut(&40).unwrap().terrain_type = TerrainType::Mountain;
        let tc = with_temple.tribes[&1].cities[0].clone();
        assert_eq!(
            max_affordable_pop(&with_temple, 1, &tc, i32::MAX),
            1,
            "mountain temple is a legal pop route"
        );
        // …and it is priced: a 19-star budget cannot buy it.
        assert_eq!(max_affordable_pop(&with_temple, 1, &tc, 19), 0);
    }
    /// The multiplier tier counts partners the city could still build, not
    /// only the ones standing today — otherwise a city with unfarmed crops
    /// still reads as a dead end.
    #[test]
    fn max_affordable_pop_counts_buildable_partners() {
        use crate::states::ResourceState;
        use crate::types::{ResourceType, TechnologyType};
        let techs = [TechnologyType::Organization, TechnologyType::Farming, TechnologyType::Construction];
        let mut state = city_with(&techs, vec![60, 61, 50, 72]);
        // Two unfarmed crops adjacent to the empty field at 61.
        for idx in [50, 72] {
            state.resources.insert(idx, Some(ResourceState { resource_type: ResourceType::Crop }));
        }
        let city = state.tribes[&1].cities[0].clone();
        // 2 farms (2 pop each) + a windmill worth 1 x 2 planned partners.
        assert_eq!(max_affordable_pop(&state, 1, &city, i32::MAX), 6);
    }
}
