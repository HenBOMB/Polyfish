//! Shared per-move reward definition for TD value labels (self_play) and
//! reward-aware MCTS backup (gumbel_mcts). One source of truth so a move's
//! score gain is normalized identically whether it's being summed into a
//! training label or backed up through the search tree.

use crate::states::GameState;

/// Turn-boundary discount for TD backup/labels: `γ^Δturn` applied only when
/// an edge crosses into a new game turn (within-turn moves are undiscounted).
/// ~10-turn effective horizon; gives a strict banked-now > pending-later
/// ordering independent of noise, unlike the old fixed forward-window MC
/// label it replaces. See notes.md, decision-trace section.
pub const GAMMA_TURN: f32 = 0.9;

/// Weight of the relative (vs opponent) component within a reward. Abs-
/// dominant: in mirror self-play both copies gain roughly in lockstep, so a
/// capture's relative swing nets to ~0 and teaches nothing; an absolute
/// anchor on my own score progress rewards it regardless of the opponent.
/// EXP_ELO_005: raising this to 0.7 broke SEARCH before it could test the
/// label hypothesis (instant hoarding/passivity in self-play) — a label-only
/// rel weight must be threaded separately, not changed here.
pub const REL_W: f32 = 0.4;

/// Reward normalization scales with the game's economy: a saturating swing
/// is ~15% of combined score, floored for the small opening turns.
pub const NORM_FRAC: f32 = 0.15;
pub const NORM_FLOOR: f32 = 600.0;

/// Normalization denominator for a reward measured from a state where `my`/
/// `opp` are the pre-transition scores.
pub fn score_norm(my: i32, opp: i32) -> f32 {
    (NORM_FRAC * (my + opp) as f32).max(NORM_FLOOR)
}

/// Normalized reward for a transition `(my_pre, opp_pre) -> (my_post,
/// opp_post)`, blending absolute (my own score gain) and relative (my gain
/// vs the opponent's) progress. Not clamped — callers accumulate/discount
/// multiple rewards before clamping the final label.
pub fn normalized_reward(my_pre: i32, opp_pre: i32, my_post: i32, opp_post: i32) -> f32 {
    normalized_reward_w(my_pre, opp_pre, my_post, opp_post, REL_W)
}

/// `normalized_reward` with an explicit relative weight — lets TD labels
/// price windows independently of the in-tree backup (EXP_ELO_006).
pub fn normalized_reward_w(
    my_pre: i32,
    opp_pre: i32,
    my_post: i32,
    opp_post: i32,
    rel_w: f32,
) -> f32 {
    let norm = score_norm(my_pre, opp_pre);
    let delta_abs = (my_post - my_pre) as f32 / norm;
    let delta_rel = ((my_post - opp_post) - (my_pre - opp_pre)) as f32 / norm;
    rel_w * delta_rel + (1.0 - rel_w) * delta_abs
}

/// `normalized_reward_w` over f32 snapshots — the shaped-potential path
/// (EXP_ELO_016) produces fractional augmented scores.
pub fn normalized_reward_wf(
    my_pre: f32,
    opp_pre: f32,
    my_post: f32,
    opp_post: f32,
    rel_w: f32,
) -> f32 {
    let norm = (NORM_FRAC * (my_pre + opp_pre)).max(NORM_FLOOR);
    let delta_abs = (my_post - my_pre) / norm;
    let delta_rel = ((my_post - opp_post) - (my_pre - opp_pre)) / norm;
    rel_w * delta_rel + (1.0 - rel_w) * delta_abs
}

/// `(my_score, best_opponent_score)` for `player` in `state`. Shared snapshot
/// helper for reward computation at both a tree edge (gumbel_mcts) and a
/// self-play history step.
pub fn score_snapshot(state: &GameState, player: i32) -> (i32, i32) {
    let my = state.tribes.get(&player).map(|t| t.score).unwrap_or(0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(_, t)| t.score)
        .max()
        .unwrap_or(0);
    (my, opp)
}

// ---- EXP_ELO_016: development-potential shaping ------------------------
// The raw score prices tech at 100·tier (instant, riskless) vs units at
// 5·cost (clawed back on death), making tech-towering approximately the
// greedy-optimal policy under score-delta labels. `dev_potential` reprices
// the label: de-weight tech, pay income/army/village-proximity densely so
// delay itself costs γ (the "pull future credit forward" mechanism). Applied
// as `score + w·Φ` deltas under the existing discounted-delta convention.

/// Fraction of tech's 100·tier score removed from the shaped label.
pub const SHAPE_TECH_DEWEIGHT: f32 = 0.75;
/// Score-equivalents per star-per-turn of income.
pub const SHAPE_SPT: f32 = 20.0;
/// Extra score-equivalents per star of living units (on top of the game's
/// 5·cost) — kept below tech parity so army pays through captures, not count.
pub const SHAPE_ARMY_PER_COST: f32 = 5.0;
/// Score-equivalents per tile of closed distance toward the nearest
/// FOW-visible uncaptured village.
pub const SHAPE_PROX_PER_TILE: f32 = 12.0;
/// Proximity credit saturates beyond this distance.
pub const SHAPE_PROX_CAP: i32 = 7;

/// EXP_ELO_018: score-equivalents per tile of closed distance toward the
/// nearest visible uncaptured village, for the *isolated* pursuit-progress
/// reward (independent weight, see `pursuit_potential`). Sized from the
/// measured chosen−toward Q gap on wrong-move pursuer-turns (median 0.19 /
/// p75 0.42 normalized, ≈150–350 score-equiv through `score_norm≈700`) —
/// ~15× EXP_ELO_016's `SHAPE_PROX_PER_TILE`, which was too weak to flip the
/// decision (FM-3 pursuit metric — see the notes.md pursuit diagnosis;
/// current status in current_understanding.md).
pub const SHAPE_PURSUIT_PER_TILE: f32 = 200.0;

/// Chebyshev distance between two row-major tile indices.
fn cheb(a: i32, b: i32, width: i32) -> i32 {
    let (ra, ca) = (a / width, a % width);
    let (rb, cb) = (b / width, b % width);
    (ra - rb).abs().max((ca - cb).abs())
}

/// `max(0, CAP − min dist(own units → nearest visible uncaptured village))`
/// in TILES, 0 with no units or no visible village — the raw proximity
/// gradient shared by both the EXP_ELO_016 `village_proximity` term and the
/// EXP_ELO_018 `pursuit_potential` term (each applies its own per-tile
/// weight). A step toward the village banks potential now; hovering banks
/// nothing further (potential-based).
fn village_proximity_tiles(state: &GameState, player: i32) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    if tribe.units.is_empty() {
        return 0.0;
    }
    let width = state.settings.size as i32;
    if width <= 0 {
        return 0.0;
    }
    let mut best: Option<i32> = None;
    for (&idx, s) in state.structures.iter() {
        let Some(s) = s else { continue };
        if s.structure_type != crate::types::StructureType::Village {
            continue;
        }
        let Some(tile) = state.tiles.get(&idx) else {
            continue;
        };
        if tile.owner != 0 || !tile.explorers.contains(&player) {
            continue;
        }
        for u in &tribe.units {
            let d = cheb(u.coords.idx, idx, width);
            if best.map_or(true, |b| d < b) {
                best = Some(d);
            }
        }
    }
    match best {
        Some(d) => (SHAPE_PROX_CAP - d).max(0) as f32,
        None => 0.0,
    }
}

/// EXP_ELO_016 proximity term (score-equivalent units).
fn village_proximity(state: &GameState, player: i32) -> f32 {
    SHAPE_PROX_PER_TILE * village_proximity_tiles(state, player)
}

/// EXP_ELO_018 isolated pursuit-progress potential Φ (score-equivalent
/// units): the same tile gradient as `village_proximity`, weighted at the
/// data-sized `SHAPE_PURSUIT_PER_TILE` so a step that closes distance to the
/// nearest visible uncaptured village banks enough reward to flip the
/// measured chosen−toward Q gap. Threaded on its own weight, so this arm can
/// run with the tech/SPT/army repricing off (`dev_w = 0`).
pub fn pursuit_potential(state: &GameState, player: i32) -> f32 {
    SHAPE_PURSUIT_PER_TILE * village_proximity_tiles(state, player)
}

// ---- EXP_ELO_028 Phase 1c: goal-priced in-tree shaping ------------------
// The painted macro goal gets an actuator: each stance/order prices the
// resource conversion it names, as a potential on the goal-holder's side
// only (the opponent's goal is unknown at search time). Sized like
// SHAPE_PURSUIT_PER_TILE — large enough to flip a decisive Q gap (~0.15-0.2
// normalized through score_norm≈600-700).

/// Score-equivalents per star-per-turn of income while stance is GROW.
pub const SHAPE_GOAL_SPT: f32 = 150.0;
/// Extra score-equivalents per star of living army while stance is ARM
/// (on top of the game score's 5·cost).
pub const SHAPE_GOAL_ARM_PER_COST: f32 = 50.0;
/// Score-equivalents per tile of closed distance toward a painted EXPAND
/// target. Summed over targets; an achieved (self-owned) target holds the
/// full cap so the final capture banks its step instead of cliffing -CAP.
pub const SHAPE_GOAL_EXPAND_PER_TILE: f32 = 200.0;
/// Score-equivalents per OWNED environment-recommended tech (GoalAux) —
/// buying the map-fit tech banks this in-tree; off-fit tech banks nothing.
pub const SHAPE_GOAL_TECH_FIT: f32 = 150.0;
/// v2.4 scout term: score-equivalents per explored tile while GROW holds,
/// no EXPAND target is known, and expansion is unfinished — pays the
/// "find your village" step the audit showed nothing was paying for.
pub const SHAPE_GOAL_SCOUT: f32 = 25.0;
/// v2.4: extra proximity-tiles banked when an EXPAND target is achieved
/// (on top of the held cap) — makes the final Capture-vs-Step choice a
/// ~0.4-normalized landslide instead of one more step of gradient.
pub const SHAPE_GOAL_EXPAND_DONE: f32 = 2.0;
/// Score-equivalents per living Rider while the rider push is on (open
/// terrain + active EXPAND).
pub const SHAPE_GOAL_RIDER: f32 = 100.0;
/// v3→v6: score-equivalents per STAR OF COST of living archetype/overlay-
/// preferred units (GoalAux.preferred_units). Was 100 flat per head, which
/// made a Knight (8★) worth 17.5 Φ/star against a Defender's 38 from the
/// SAME overlay — the measured reason the knight lane researched but never
/// converted. 33/cost keeps the cost-3 units (Rider/Archer/Defender)
/// numerically unchanged (99 ≈ 100) while heavies price per star invested
/// (Knight 264, Catapult 264, Giant 330). Stacks with SHAPE_GOAL_RIDER.
pub const SHAPE_GOAL_ARCHETYPE_PER_COST: f32 = 33.0;
/// v4: per-quadrant cap on scout-term tiles — tiles in a fresh quadrant keep
/// paying after a covered quadrant has gone flat (audit: half of games left
/// a quadrant unvisited).
pub const SCOUT_QUADRANT_CAP: i32 = 20;
/// v4: one-time score-equivalents per map corner explored (lighthouse →
/// monument progress; audit: 29/64 games touched zero corners).
pub const SHAPE_GOAL_LIGHTHOUSE: f32 = 120.0;
/// v4 (amended per Verdi — encourage, never hard-gate): score-equivalents per
/// Explorer reward taken, scaled by hidden_frac² (see the term). Re-sized
/// Jul 31 from the measured root Q gap (--dump-reward-choices, 166 modal
/// Explorer/Workshop plies): Workshop led by median +0.26 normalized at
/// hidden≥0.5 and the old 150·h term (≈+0.12 effective at the horizon) lost
/// 86% of those plies. Iterated against the measured dq shift (~0.085 per
/// 100): 1000 overshot (83% take, SPT@t20 −5, wr −14pp matched); 600 sat at
/// the exact flip point (dq −0.004, 52% take). 700 lands the dark-map take
/// in the registered ≥60% band at roughly half of 1000's economic cost.
pub const SHAPE_GOAL_EXPLORER: f32 = 700.0;
/// v4: extra Explorer lift when an UNREVEALED map corner sits within the
/// walk's plausible reach of the city. The 12-step fog-biased walk is
/// deterministic — an exact corner hit is already priced through
/// SHAPE_GOAL_LIGHTHOUSE in the simulated child — so this credits the
/// near-miss chance the simulation can't see. Scaled with the main term
/// (Jul 31 re-size, ~same ratio as the old 150:60).
pub const SHAPE_GOAL_EXPLORER_LIGHTHOUSE: f32 = 230.0;
/// Chebyshev radius for "corner within explorer reach". Verdi: a centrally
/// located explorer reliably reaches one, sometimes two lighthouses — so 5
/// (center-to-corner on 11x11) is in range, and the lift scales per
/// reachable dark corner, capped at 2.
pub const EXPLORER_WALK_RANGE: i32 = 5;
/// Max dark corners credited per explorer (one, sometimes two).
pub const EXPLORER_CORNER_CAP: usize = 2;
/// v5: score-equivalents per unit of reward_pop per adjacency partner BEYOND
/// the first, for owned yield structures (Windmill/Sawmill/Forge — anything
/// in structures.rs with reward_pop > 0 and adjacent_types). Steers the tile
/// choice toward multi-partner spots (audit: 52% of windmills sat next to a
/// single farm). The first partner is the structure paying for itself — no
/// bonus; a partner-less build stays unpriced rather than penalized.
pub const SHAPE_GOAL_YIELD_ADJ: f32 = 100.0;
/// v6: star-yield analog for Market (reward_stars > 0 + adjacent_types) —
/// HALF the pop analog, deliberately: each partner's +1 SPT is already paid
/// at SHAPE_GOAL_SPT through get_tribe_spt; this only sharpens the 2-3-hub
/// placement choice.
pub const SHAPE_GOAL_YIELD_ADJ_STARS: f32 = 50.0;
/// v5: option value of a standing forest in own territory (future lumber
/// hut / sawmill feed / grow line). Clearing or burning drops it, making
/// the ~1/game follow-through-free clear (audit: 10% of clears fed no
/// build, level-up, or sawmill) net-negative in-tree, while justified
/// clears still win on their follow-up's much larger payoff.
pub const SHAPE_GOAL_FOREST_STANDING: f32 = 50.0;


/// v8: multiplier on the Explorer term while the tribe still holds only its
/// capital. Measured Aug 1: the capital's first city reward was Explorer in
/// **12/12 seats** — at t0 the map is maximally dark, so `hidden²` ≈ 1 and the
/// term pays ~700-930 against Workshop's 150. That is not a distribution, it is
/// a constant. Workshop's +1 SPT is worth most at t0 (longest compounding
/// horizon), so the explorer edge is discounted for exactly the first reward
/// and left untouched afterwards — the dark-map dial Verdi measured in July
/// (1000 → wr −14pp, 600 → the flip point) is deliberately NOT reopened.
pub const SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE: f32 = 0.15;

/// v6: score-equivalents of penalty PER POINT of structurally stranded
/// progress — pop sunk into a city whose next level can no longer be
/// completed from its remaining territory resources/pop structures at ANY
/// star budget. First fit (150/city, stars-dependent) fired both
/// falsifiers: star-dependence taxed all spending (wr −12pp) while
/// depth-blindness left extra pop into stranded cities free. Depth ×
/// resource-structural has no spend-tax and prices exactly the
/// harvest-into-a-dead-end Verdi flagged. Threatened cities exempt.
pub const SHAPE_GOAL_STRANDED: f32 = 75.0;

/// v7: savings ramp — Phi paid for holding `stars/cost` of the SAVE batch.
/// Deliberately below `SHAPE_GOAL_SPT`: reaching the batch and spending it must
/// beat sitting on a full bank, or the ramp becomes a reason to hoard forever.
/// First fit — dial against the measured carried-balance and purchase rates
/// per the q-gap method (first fits have overshot ~2x every time).
pub const SHAPE_GOAL_SAVE: f32 = 100.0;

/// v7: completion BONUS — pays progress toward a REACHABLE level (see
/// `completion_progress`). Held below `SHAPE_GOAL_SPT` so a level-up, which
/// zeroes the bonus and banks the SPT jump, is always the better move.
pub const SHAPE_GOAL_COMPLETION: f32 = 75.0;

/// v6: approach weight for an enemy-taken village painted as a retake
/// target — pays slightly under a free village (the defender-adjusted
/// odds); the recapture itself banks the full DONE bonus.
pub const SHAPE_GOAL_RETAKE_W: f32 = 0.75;
/// v6: a contested EXPAND target (visible enemy standing on it) pays the
/// nearest second unit this fraction of the approach gradient — exactly
/// one converger, so a squatter can be killed and the tile still taken.
pub const SHAPE_GOAL_CONTEST_SECOND: f32 = 0.5;
/// v6: score-equivalents per living unit up to a state-derived cap
/// (min(cities+1, BODY_CAP_MAX)) while GROW holds and there is still map
/// to take — the early bodies that scout and grab villages. Flat per head:
/// warriors ARE the desired bodies; the cap kills summon spam. 75 clears
/// the measured −0.118 summon Q deficit without doubling it (dial lesson).
pub const SHAPE_GOAL_BODY: f32 = 75.0;
/// Cap on bodies the GROW body term pays for.
pub const BODY_CAP_MAX: usize = 3;

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

    for &idx in &city._territory {
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
        for &idx in &city._territory {
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
        for &idx in &city._territory {
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

/// v7: a stranded city is FLAGGED, not billed by depth. v6 summed every
/// stranded pop point, which made a level-up that leaves 2 overflow progress
/// book −150 against its own +150 of SPT — the level-ups we want paid for
/// themselves. Capping at one point per city keeps the "don't start a level
/// you cannot finish" signal (0 → 1 progress still costs a full unit) while
/// pricing the DECISION rather than the sunk history.
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
pub fn completion_progress(state: &GameState, player: i32) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    tribe
        .cities
        .iter()
        .filter(|c| c.progress > 0 && city_completable(state, player, c))
        .map(|c| c.progress as f32 / (c.level + 1).max(1) as f32)
        .sum()
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


/// v9 permanent floors (Verdi, Aug 2): all three drives always carry weight —
/// "a floor that acts as a gentle nudge for all 3 vertexes whereas the net has
/// the liberty to choose where to emphasize". The stance no longer switches a
/// drive OFF, it only decides which one runs at full weight; an off-stance
/// drive still pays this fraction. Deliberately a fraction rather than a
/// wholesale restructure: EXP_ELO_005 found search reacts violently to reward
/// changes, so the dominant term keeps its measured magnitude.
pub const SHAPE_FLOOR_FRAC: f32 = 0.3;

/// v9 economy depth: score-equivalents per point of HEADROOM a city still
/// has — the distance between its level and ITS OWN structural ceiling (max
/// level reachable from its territory routes at unlimited stars). Verdi
/// rejected a fixed level-5/giant target: "some villages have the ability to
/// scale beyond that, sometimes the ceiling is 4, for others it's 7".
/// Deliberately BELOW `SHAPE_GOAL_SPT` so converting headroom into a level is
/// always net-positive.
///
/// v9.1: capped at `HEADROOM_PER_CITY_CAP`. Uncapped, a fresh 4-level-headroom
/// city banked 240 Φ while LEVELLING one only netted +90 (150 SPT − 60 released)
/// — so grabbing land beat developing it, and the Aug 2 run shows exactly that:
/// cities +8% by t25 but city_levels −31% at t3, harvests −25%, giants −22%.
/// Same lesson as `STRANDED_PER_CITY_CAP`: FLAG that a city has room, do not
/// bill by depth. Capped, acquiring banks 60 and levelling still banks the full
/// 150 (the cap does not release until the last level), so development wins.
pub const SHAPE_GOAL_CEILING: f32 = 60.0;

/// Levels of headroom counted per city. See `SHAPE_GOAL_CEILING`.
pub const HEADROOM_PER_CITY_CAP: i32 = 1;

/// v9 territory: score-equivalents per owned tile. Territory is currently paid
/// only through order-driven EXPAND gradients, so it vanishes entirely
/// whenever no EXPAND order is live.
pub const SHAPE_GOAL_TERRITORY: f32 = 8.0;

/// v9 risk-adjusted star optionality (Verdi: "the reason we want to wait is
/// because of risk… you'd rather have the stars to have the optionability to
/// pivot"). Held stars are worth ~nothing today — spend/income median is
/// exactly 1.00 in every arm ever measured — so "wait" is dominated whatever
/// the net knows. Because Φ telescopes, the in-tree reward for spending
/// becomes (purchase gain − option value released): risk RAISES the bar for
/// spending rather than forbidding it, and goes to ~0 on a quiet board.
/// Held below what a completed level pays, or it hoards.
pub const SHAPE_GOAL_STAR_OPTION: f32 = 12.0;

/// v9: HP-weighted unit worth. A 1hp Rider scored the same as a fresh one, so
/// retaliation was free and standing in reach was unpriced.
pub fn unit_value(unit: &crate::states::UnitState) -> f32 {
    let cost = crate::settings::units::get_unit_setting(unit.unit_type).cost as f32;
    let max_hp = crate::functions::get_unit_max_health(unit).max(1.0);
    cost * (unit.health / max_hp).clamp(0.0, 1.0)
}

fn live_army_value(tribe: &crate::states::TribeState) -> f32 {
    tribe.units.iter().map(unit_value).sum()
}

/// The highest level this city could ever reach from its own territory routes,
/// ignoring stars. Walks `max_affordable_pop` against the `level+1` thresholds.
pub fn city_ceiling(state: &GameState, player: i32, city: &crate::states::CityState) -> i32 {
    let mut level = city.level;
    let mut pop = city.progress + max_affordable_pop(state, player, city, i32::MAX);
    let mut guard = 0;
    while pop >= level + 1 && guard < 16 {
        pop -= level + 1;
        level += 1;
        guard += 1;
    }
    level
}


/// v9 perf: one pass over the tile map. The Φ path previously walked
/// `state.tiles` FOUR times per node (own forests, owned tiles, explored
/// count, fog sum) — measured at ~47% of self-play throughput, since Φ is
/// evaluated per node. Everything the map can answer is collected here once.
struct TileScan {
    owned: f32,
    explored: f32,
    total: f32,
    own_forests: f32,
    fog_sum: f32,
    fog_n: f32,
}

fn scan_tiles(state: &GameState, player: i32, fog: Option<&[f32]>) -> TileScan {
    let mut sc = TileScan {
        owned: 0.0,
        explored: 0.0,
        total: 0.0,
        own_forests: 0.0,
        fog_sum: 0.0,
        fog_n: 0.0,
    };
    for (idx, tile) in state.tiles.iter() {
        sc.total += 1.0;
        let mine = tile.owner == player;
        if mine {
            sc.owned += 1.0;
            if tile.terrain_type == crate::types::TerrainType::Forest {
                sc.own_forests += 1.0;
            }
        }
        if tile.explorers.contains(&player) {
            sc.explored += 1.0;
        } else if let Some(probs) = fog {
            // Unexplored only: this is P(enemy | we cannot see it).
            if let Some(p) = probs.get(*idx as usize) {
                sc.fog_sum += *p;
                sc.fog_n += 1.0;
            }
        }
    }
    sc
}

/// v9 perf: one pass over the cities. `completion_stranded`, `completion_progress`
/// and `city_ceiling` each called `max_affordable_pop(.., i32::MAX)` separately
/// — the same expensive call three times per city per node. Computed once here.
struct CityScan {
    stranded: f32,
    progress_bonus: f32,
    headroom: f32,
}

fn scan_cities(state: &GameState, player: i32, tribe: &crate::states::TribeState) -> CityScan {
    let mut sc = CityScan { stranded: 0.0, progress_bonus: 0.0, headroom: 0.0 };
    for c in &tribe.cities {
        let reachable = max_affordable_pop(state, player, c, i32::MAX);
        let completable = c.progress + reachable >= c.level + 1;
        if c.progress > 0 {
            if completable {
                sc.progress_bonus += c.progress as f32 / (c.level + 1).max(1) as f32;
            } else if !city_threatened(state, player, c.idx) {
                sc.stranded += c.progress.min(STRANDED_PER_CITY_CAP) as f32;
            }
        }
        // Ceiling walk, reusing the same `reachable`.
        let (mut level, mut pop) = (c.level, c.progress + reachable);
        let mut guard = 0;
        while pop >= level + 1 && guard < 16 {
            pop -= level + 1;
            level += 1;
            guard += 1;
        }
        sc.headroom += (level - c.level).clamp(0, HEADROOM_PER_CITY_CAP) as f32;
    }
    sc
}

/// v9 risk in [0, 1]: how likely this position is to need stars for defence
/// rather than growth. Blends state-side signals with the mirrored `aux_fog`
/// head when the backend produced it (`None` on pre-aux checkpoints — the
/// caller must never substitute a zero, see the `progress` stub).
pub fn position_risk(state: &GameState, player: i32, fog: Option<&[f32]>) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    risk_from_scan(state, player, tribe, &scan_tiles(state, player, fog), fog.is_some())
}

fn risk_from_scan(
    state: &GameState,
    player: i32,
    tribe: &crate::states::TribeState,
    sc: &TileScan,
    has_fog: bool,
) -> f32 {
    // 1. Visible enemy pressure, relative to our own army.
    let own = live_army_value(tribe).max(1.0);
    let seen: f32 = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .flat_map(|(_, t)| t.units.iter())
        .filter(|u| {
            state
                .tiles
                .get(&u.coords.idx)
                .map_or(false, |t| t.explorers.contains(&player))
        })
        .map(unit_value)
        .sum();
    let pressure = (seen / own).clamp(0.0, 1.0);

    // v9.1: the raw unexplored FRACTION is gone. It made scouting a
    // self-inflicted loss: exploring lowers `dark`, which lowered risk, which
    // destroyed Φ the tribe was already banking on its held stars. Measured on
    // the Aug 2 run as revealed tiles −25% at t3 and down at every later turn.
    // The learned head does not have that defect — P(enemy | unexplored) falls
    // when you look and find nothing, which is information updating, not a
    // penalty for looking.

    // 3. The learned head, where available: mean P(enemy) over tiles we cannot
    //    see. This is the part that knows WHERE, and it is the reason the head
    //    was mirrored into Rust at all.
    let learned = if has_fog && sc.fog_n > 0.0 {
        Some((sc.fog_sum / sc.fog_n).clamp(0.0, 1.0))
    } else {
        None
    };

    // Contact pressure dominates once it exists; before contact the learned
    // fog head carries it. With no head available this degrades to
    // threat-only risk rather than to a scouting penalty.
    (0.6 * pressure + 0.4 * learned.unwrap_or(0.0)).clamp(0.0, 1.0)
}

/// Goal potential Φ_goal for `player` under `goal` (score-equivalent units).
pub fn goal_potential(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
) -> f32 {
    goal_potential_with_fog(state, player, goal, aux, None)
}

/// v9: `goal_potential` with the mirrored fog head's output for this state,
/// which the risk term needs. `None` means the backend could not produce it.
pub fn goal_potential_with_fog(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
    fog: Option<&[f32]>,
) -> f32 {
    use crate::ai::oracle_macro::{OrderKind, Stance};
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    // v9 permanent floors. The stance no longer switches a drive off — it
    // decides which runs at full weight, and the others still pay
    // SHAPE_FLOOR_FRAC. Stance-switched Φ was also not a function of STATE
    // (StanceCommit carries history), so a flip moved Φ by ~150×SPT with no
    // state change, breaking the invariance potential-based shaping needs.
    let economy_on = matches!(goal.stance, Stance::Grow | Stance::Save);
    let eco_w = if economy_on { 1.0 } else { SHAPE_FLOOR_FRAC };
    let mil_w = if goal.stance == Stance::Arm { 1.0 } else { SHAPE_FLOOR_FRAC };

    // v9 perf: ONE tile pass and ONE city pass feed every term below.
    let scan = scan_tiles(state, player, fog);
    let cities = scan_cities(state, player, tribe);

    let mut phi = 0.0;
    if goal.stance != Stance::Unlock {
        // Economy: income now, plus depth toward each city's OWN ceiling.
        phi += eco_w * SHAPE_GOAL_SPT * crate::functions::get_tribe_spt(state, tribe) as f32;
        // HEADROOM, paid as a BONUS — not the gap as a penalty. The first cut
        // subtracted it, which inverted the incentive: acquiring a resource
        // raises a city's ceiling, so improving the city LOWERED Φ. Paid this
        // way, a level-up converts 60 of headroom into 150 of SPT (net +90, so
        // growing still wins), gaining territory is rewarded, and clearing a
        // forest costs exactly the headroom it destroys — which is what
        // "price a forest by what it is FOR" means.
        phi += eco_w * SHAPE_GOAL_CEILING * cities.headroom;

        // Territory: always paid, not only while an EXPAND order is live.
        phi += eco_w * SHAPE_GOAL_TERRITORY * scan.owned;

        // Military: HP-weighted, so damage destroys value.
        phi += mil_w * SHAPE_GOAL_ARM_PER_COST * live_army_value(tribe);

        // Risk-adjusted star optionality: holding is worth something exactly
        // when the position might need a pivot.
        let risk = risk_from_scan(state, player, tribe, &scan, fog.is_some());
        phi += SHAPE_GOAL_STAR_OPTION * tribe.stars as f32 * risk;
    }
    // v6: stranded-progress discipline — GROW only (combat spending under
    // ARM shouldn't be taxed for unfinished levels).
    if matches!(goal.stance, Stance::Grow | Stance::Save) {
        phi -= SHAPE_GOAL_STRANDED * cities.stranded;
        phi += SHAPE_GOAL_COMPLETION * cities.progress_bonus;
    }
    // v7 savings ramp: progress toward the banked batch is itself scored, so
    // holding stars climbs a gradient instead of sitting in a flat valley that
    // any purchase strictly beats. This is what makes a multi-turn plan legible
    // to a search whose horizon is one game turn — the ramp is visible at
    // depth 1, so the tree never has to reach the purchase to value it.
    if goal.stance == Stance::Save {
        if let Some(cost) = goal.save_target.filter(|&c| c > 0) {
            phi += SHAPE_GOAL_SAVE * (tribe.stars.clamp(0, cost) as f32) / cost as f32;
        }
    }
    let width = state.settings.size as i32;
    let mut has_expand = false;
    if width > 0 {
        // v4: achieved/blocked targets price directly; approach-needing ones
        // go through the per-unit assignment so two scouts never bank the
        // same target. Unexplored (guessed) targets always pay approach —
        // reading their owner would leak FOW; the completion bonus needs a
        // real city capture, not a border-grown empty tile.
        let mut approach_targets: Vec<i32> = Vec::new();
        let mut target_weight: std::collections::HashMap<i32, f32> =
            std::collections::HashMap::new();
        for (kind, idx) in &goal.orders {
            if *kind != OrderKind::Expand {
                continue;
            }
            has_expand = true;
            let Some(tile) = state.tiles.get(idx) else {
                continue;
            };
            let mut weight = 1.0;
            if tile.explorers.contains(&player) {
                if tile.owner == player {
                    if crate::functions::get_city_at(state, *idx).is_some() {
                        phi += SHAPE_GOAL_EXPAND_PER_TILE
                            * (SHAPE_PROX_CAP as f32 + SHAPE_GOAL_EXPAND_DONE);
                    }
                    continue;
                } else if tile.owner != 0 {
                    // v6: enemy-taken village painted for retake — approach
                    // pays slightly under a free village (defended odds);
                    // recapture banks the full DONE bonus via the branch
                    // above once the tile flips.
                    weight = SHAPE_GOAL_RETAKE_W;
                }
            }
            approach_targets.push(*idx);
            target_weight.insert(*idx, weight);
        }
        if !approach_targets.is_empty() {
            let w_of = |t: i32| target_weight.get(&t).copied().unwrap_or(1.0);
            let pairs = crate::ai::oracle_macro::assign_expand_targets(
                state,
                player,
                &approach_targets,
            );
            for (unit_idx, target) in &pairs {
                let d = cheb(*unit_idx, *target, width);
                phi += SHAPE_GOAL_EXPAND_PER_TILE
                    * w_of(*target)
                    * (SHAPE_PROX_CAP - d).max(0) as f32;
            }
            // Targets beyond the unit count keep their closest-unit gradient,
            // so an under-scouted map still pulls.
            let assigned: std::collections::HashSet<i32> =
                pairs.iter().map(|(_, t)| *t).collect();
            for target in approach_targets.iter().filter(|t| !assigned.contains(t)) {
                let d = tribe
                    .units
                    .iter()
                    .map(|u| cheb(u.coords.idx, *target, width))
                    .min()
                    .unwrap_or(i32::MAX);
                phi += SHAPE_GOAL_EXPAND_PER_TILE
                    * w_of(*target)
                    * (SHAPE_PROX_CAP - d).max(0) as f32;
            }
            // v6: a CONTESTED target (visible enemy unit standing on it) pays
            // one extra converger — the nearest unit not already assigned to
            // it — at half gradient. Exactly one; no dogpile.
            for (unit_idx, target) in &pairs {
                let occupied = crate::functions::get_unit_at(state, *target)
                    .map_or(false, |u| u.owner != player)
                    && state
                        .tiles
                        .get(target)
                        .map_or(false, |t| t.explorers.contains(&player));
                if !occupied {
                    continue;
                }
                let second = tribe
                    .units
                    .iter()
                    .map(|u| u.coords.idx)
                    .filter(|idx| idx != unit_idx)
                    .map(|idx| cheb(idx, *target, width))
                    .min();
                if let Some(d) = second {
                    phi += SHAPE_GOAL_CONTEST_SECOND
                        * SHAPE_GOAL_EXPAND_PER_TILE
                        * w_of(*target)
                        * (SHAPE_PROX_CAP - d).max(0) as f32;
                }
            }
        }
    }
    // v6: early body count — GROW pays per living unit up to min(cities+1,
    // BODY_CAP_MAX) while there is still expansion or unexplored map. The
    // 2nd/3rd warrior finally prices in against a 2-star harvest.
    if goal.stance == Stance::Grow && width > 0 {
        let revealed = state
            .tiles
            .values()
            .filter(|t| t.explorers.contains(&player))
            .count();
        let map_unexplored = revealed < (width * width) as usize;
        if has_expand || map_unexplored {
            let cap = (tribe.cities.len() + 1).min(BODY_CAP_MAX);
            phi += SHAPE_GOAL_BODY * tribe.units.len().min(cap) as f32;
        }
    }
    // Scout term, v4: per-quadrant concave reveal payment — fresh quadrants
    // keep paying after covered ones flatten. Full weight while no target is
    // known; half weight alongside an active approach gradient.
    if goal.stance == Stance::Grow
        && width > 0
        && tribe.cities.len() < crate::ai::oracle_macro::COMMIT_CITY_TARGET
    {
        let half = width / 2;
        let mut quad = [0i32; 4];
        for (idx, t) in state.tiles.iter() {
            if t.explorers.contains(&player) {
                let q = ((idx % width > half) as usize) * 2 + ((idx / width > half) as usize);
                quad[q] += 1;
            }
        }
        let capped: i32 = quad.iter().map(|&c| c.min(SCOUT_QUADRANT_CAP)).sum();
        let w = if has_expand { 0.5 } else { 1.0 };
        phi += SHAPE_GOAL_SCOUT * w * capped as f32;
    }
    // Lighthouse nudge (v4): each explored map corner pays once.
    if width > 0 {
        for c in [0, width - 1, width * (width - 1), width * width - 1] {
            if state
                .tiles
                .get(&c)
                .map_or(false, |t| t.explorers.contains(&player))
            {
                phi += SHAPE_GOAL_LIGHTHOUSE;
            }
        }
    }
    // Explorer preference (v4): each Explorer reward taken pays scaled by the
    // hidden-map fraction — big on a dark map, ~nothing once revealed. The
    // reveal itself additionally banks the scout/lighthouse terms above, and
    // a city near a still-dark corner gets the lighthouse-chance lift.
    if width > 0 {
        let explorer_cities: Vec<i32> = tribe
            .cities
            .iter()
            .filter(|c| c.rewards.contains(&crate::types::CityRewardType::Explorer))
            .map(|c| c.idx)
            .collect();
        if !explorer_cities.is_empty() {
            let revealed = state
                .tiles
                .values()
                .filter(|t| t.explorers.contains(&player))
                .count() as f32;
            let hidden_frac = (1.0 - revealed / (width * width) as f32).max(0.0);
            if hidden_frac > 0.0 {
                let corners = [0, width - 1, width * (width - 1), width * width - 1];
                for city in explorer_cities {
                    let dark_in_reach = corners
                        .iter()
                        .filter(|&&k| {
                            cheb(city, k, width) <= EXPLORER_WALK_RANGE
                                && !state
                                    .tiles
                                    .get(&k)
                                    .map_or(false, |t| t.explorers.contains(&player))
                        })
                        .count()
                        .min(EXPLORER_CORNER_CAP);
                    let mut bonus = SHAPE_GOAL_EXPLORER
                        + SHAPE_GOAL_EXPLORER_LIGHTHOUSE * dark_in_reach as f32;
                    // v8: the capital's first reward is a constant, not a
                    // choice — discount it so Workshop's whole-game compounding
                    // can win the one slot where it is worth the most.
                    if tribe.cities.len() <= 1 {
                        bonus *= SHAPE_GOAL_EXPLORER_FIRST_CITY_SCALE;
                    }
                    // hidden² (Jul 31): the reveal itself drains this Φ term
                    // (the potential telescopes to the horizon's h), and a
                    // linear ramp priced Explorer too high on mostly-lit
                    // maps. Quadratic keeps the dark-map edge dominant and
                    // the lit-map edge below Workshop's measured Q lead.
                    phi += bonus * hidden_frac * hidden_frac;
                }
            }
        }
    }
    // Yield-structure placement (v5/v6): owned adjacency-yield structures
    // pay per partner beyond the first, derived from structures.rs —
    // reward_pop-scaled for pop hubs (Windmill/Sawmill/Forge) and
    // reward_stars-scaled at half weight for star hubs (Market).
    for (&s_idx, s) in state.structures.iter() {
        let Some(s) = s.as_ref() else { continue };
        let setting = crate::settings::structures::get_structure_setting(s.structure_type);
        if (setting.reward_pop <= 0 && setting.reward_stars <= 0)
            || setting.adjacent_types.is_empty()
        {
            continue;
        }
        let owned = crate::functions::get_city_owning_tile(state, s_idx)
            .map_or(false, |c| c.owner == player);
        if !owned {
            continue;
        }
        let partners = crate::functions::get_adjacent_indices(state, s_idx, 1)
            .iter()
            .filter(|&&adj| {
                state.tiles.get(&adj).map_or(false, |t| t.owner == player)
                    && crate::functions::get_structure_at(state, adj)
                        .map_or(false, |a| setting.adjacent_types.contains(&a.structure_type))
            })
            .count() as i32;
        let extra = (partners - 1).max(0) as f32;
        phi += SHAPE_GOAL_YIELD_ADJ * setting.reward_pop.max(0) as f32 * extra;
        phi += SHAPE_GOAL_YIELD_ADJ_STARS * setting.reward_stars.max(0) as f32 * extra;
    }
    // Standing-forest option value (v5): clearing pays only when the
    // follow-up (build / level-up funding) outweighs the lost option.
    phi += SHAPE_GOAL_FOREST_STANDING * scan.own_forests;
    if let Some(aux) = aux {
        let owned = aux
            .recommended_techs
            .iter()
            .filter(|t| {
                crate::settings::technology::is_tech_unlocked(&tribe.tech_vanilla, **t)
            })
            .count();
        phi += SHAPE_GOAL_TECH_FIT * owned as f32;
        if aux.rider_push {
            let riders = tribe
                .units
                .iter()
                .filter(|u| u.unit_type == crate::types::UnitType::Rider)
                .count();
            phi += SHAPE_GOAL_RIDER * riders as f32;
        }
        if !aux.preferred_units.is_empty() {
            let preferred: f32 = tribe
                .units
                .iter()
                .filter(|u| aux.preferred_units.contains(&u.unit_type))
                .map(unit_value)
                .sum();
            phi += SHAPE_GOAL_ARCHETYPE_PER_COST * preferred;
        }
    }
    phi
}

/// The development potential Φ for `player`, in score-equivalent units.
pub fn dev_potential(state: &GameState, player: i32) -> f32 {
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let tech_score: f32 = tribe
        .tech_vanilla
        .iter()
        .map(|t| {
            100.0
                * crate::settings::technology::get_technology_setting(t.tech_type)
                    .tier
                    .unwrap_or(1) as f32
        })
        .sum();
    let army_cost: f32 = tribe
        .units
        .iter()
        .map(|u| crate::settings::units::get_unit_setting(u.unit_type).cost as f32)
        .sum();
    let spt = crate::functions::get_tribe_spt(state, tribe) as f32;

    SHAPE_SPT * spt + SHAPE_ARMY_PER_COST * army_cost + village_proximity(state, player)
        - SHAPE_TECH_DEWEIGHT * tech_score
}

/// `score_snapshot` augmented with `dev_w`·Φ_dev + `pursuit_w`·Φ_pursuit per
/// side (EXP_ELO_016 development shaping + EXP_ELO_018 isolated pursuit-
/// progress shaping, independently weighted). Both weights zero short-circuits
/// to the raw snapshot (bit-exact legacy behavior, no Φ cost on the hot path).
pub fn shaped_snapshot(state: &GameState, player: i32, dev_w: f32, pursuit_w: f32) -> (f32, f32) {
    if dev_w == 0.0 && pursuit_w == 0.0 {
        let (my, opp) = score_snapshot(state, player);
        return (my as f32, opp as f32);
    }
    let phi = |id: i32| dev_w * dev_potential(state, id) + pursuit_w * pursuit_potential(state, id);
    let my = state
        .tribes
        .get(&player)
        .map(|t| t.score as f32 + phi(player))
        .unwrap_or(0.0);
    let opp = state
        .tribes
        .iter()
        .filter(|(id, _)| **id != player)
        .map(|(id, t)| t.score as f32 + phi(*id))
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0);
    (my, opp)
}

#[cfg(test)]
mod shaping_tests {
    use super::*;
    use crate::coords::Coords;
    use crate::settings::technology::get_technology_setting;
    use crate::settings::units::get_unit_setting;
    use crate::states::{StructureState, TechnologyState, TileState, TribeState, UnitState};
    use crate::types::{StructureType, TechnologyType, UnitType};

    fn unit_at(idx: i32, unit_type: UnitType) -> UnitState {
        let mut u = UnitState {
            unit_type,
            coords: Coords::from_index(idx, 11),
            ..Default::default()
        };
        // v9: Φ scales unit worth by health, so an undamaged test unit must
        // start at ITS max (Defender's is 15, not the struct default of 10).
        u.health = crate::functions::get_unit_max_health(&u);
        u
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

    #[test]
    fn wf_matches_legacy_reward_on_integer_inputs() {
        for (a, b, c, d) in [(1000, 800, 1300, 900), (0, 0, 50, 10), (4000, 4200, 4100, 4900)] {
            let legacy = normalized_reward(a, b, c, d);
            let f = normalized_reward_wf(a as f32, b as f32, c as f32, d as f32, REL_W);
            assert!((legacy - f).abs() < 1e-6, "({a},{b},{c},{d}): {legacy} vs {f}");
        }
    }

    #[test]
    fn shaped_snapshot_at_zero_w_is_the_raw_snapshot() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.score = 123;
        let mut t2 = TribeState::default();
        t2.score = 456;
        state.tribes.insert(1, t1);
        state.tribes.insert(2, t2);
        assert_eq!(shaped_snapshot(&state, 1, 0.0, 0.0), (123.0, 456.0));
        let (my, opp) = shaped_snapshot(&state, 1, 1.0, 1.0);
        assert_eq!((my, opp), (123.0, 456.0)); // empty tribes: phi = 0
    }

    #[test]
    fn tech_deweight_subtracts_the_scores_towering_subsidy() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Riding,
            discovered: true,
            discovered_turn: 0,
        });
        state.tribes.insert(1, t1);
        let tier = get_technology_setting(TechnologyType::Riding).tier.unwrap_or(1) as f32;
        let expected = -SHAPE_TECH_DEWEIGHT * 100.0 * tier;
        assert!((dev_potential(&state, 1) - expected).abs() < 1e-4);
    }

    #[test]
    fn army_term_pays_star_cost_of_living_units() {
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        assert!((dev_potential(&state, 1) - SHAPE_ARMY_PER_COST * cost).abs() < 1e-4);
    }

    #[test]
    fn proximity_pays_stepping_toward_a_visible_village_and_nothing_for_hovering() {
        let mk = |unit_idx: i32| {
            let mut state = GameState::default();
            add_visible_village(&mut state, 0);
            let mut t1 = TribeState::default();
            t1.units.push(unit_at(unit_idx, UnitType::Warrior));
            state.tribes.insert(1, t1);
            dev_potential(&state, 1)
        };
        // Row 0: idx = column = Chebyshev distance to the village at idx 0.
        let (d4, d2) = (mk(4), mk(2));
        assert!((d2 - d4 - 2.0 * SHAPE_PROX_PER_TILE).abs() < 1e-4);
        // Lateral move at equal distance banks nothing: (0,3) vs (3,3).
        assert!((mk(3) - mk(36)).abs() < 1e-4);
        // Beyond the cap there is no gradient.
        assert!((mk(9) - mk(10)).abs() < 1e-4);
    }

    #[test]
    fn pursuit_potential_is_the_data_sized_progress_gradient() {
        let mk = |unit_idx: i32| {
            let mut state = GameState::default();
            add_visible_village(&mut state, 0);
            let mut t1 = TribeState::default();
            t1.units.push(unit_at(unit_idx, UnitType::Warrior));
            state.tribes.insert(1, t1);
            pursuit_potential(&state, 1)
        };
        // Row 0: idx = column = Chebyshev distance to the village at idx 0.
        // A one-tile close banks exactly SHAPE_PURSUIT_PER_TILE.
        assert!((mk(2) - mk(3) - SHAPE_PURSUIT_PER_TILE).abs() < 1e-3);
        // Weighted ~15x above the EXP_ELO_016 proximity garnish.
        assert!(SHAPE_PURSUIT_PER_TILE > 10.0 * SHAPE_PROX_PER_TILE);
    }

    #[test]
    fn shaped_snapshot_pursuit_weight_is_independent_of_dev_weight() {
        let mut state = GameState::default();
        add_visible_village(&mut state, 0);
        let mut t1 = TribeState::default();
        t1.score = 100;
        t1.units.push(unit_at(2, UnitType::Warrior)); // 2 tiles from village
        state.tribes.insert(1, t1);
        // pursuit_w only: augments my score by pursuit_potential, dev off.
        let (my_dev_off, _) = shaped_snapshot(&state, 1, 0.0, 1.0);
        let expected = 100.0 + pursuit_potential(&state, 1);
        assert!((my_dev_off - expected).abs() < 1e-3);
        // dev_w does not leak into the pursuit-only run.
        let (my_both, _) = shaped_snapshot(&state, 1, 1.0, 1.0);
        assert!((my_both - my_dev_off - dev_potential(&state, 1)).abs() < 1e-3);
    }

    #[test]
    fn goal_potential_prices_each_stance_and_expand_progress() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        add_visible_village(&mut state, 0);
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(2, UnitType::Warrior)); // 2 tiles from village 0
        state.tribes.insert(1, t1);

        // v9 CONTRACT: the active stance's drive pays at full weight and the
        // OTHER drives still pay SHAPE_FLOOR_FRAC — a stance no longer switches
        // a drive off. (Stance-switched Φ was also not a function of state:
        // StanceCommit carries history, so a flip moved Φ by ~150×SPT with no
        // state change, breaking the invariance shaping depends on.)
        let corner = SHAPE_GOAL_LIGHTHOUSE;
        let tribe = state.tribes.get(&1).unwrap();
        let spt = crate::functions::get_tribe_spt(&state, tribe) as f32;
        let army = SHAPE_GOAL_ARM_PER_COST * unit_value(&tribe.units[0]);
        // Economy block: income, territory, and the gap to each city's ceiling.
        // This tribe holds no cities and no tiles, so only SPT is non-zero.
        let economy = SHAPE_GOAL_SPT * spt;
        // No stars banked -> the risk-scaled star option contributes nothing.
        assert_eq!(tribe.stars, 0);

        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
        let expect_arm = army + SHAPE_FLOOR_FRAC * economy + corner;
        assert!(
            (goal_potential(&state, 1, &arm, None) - expect_arm).abs() < 1e-4,
            "ARM pays the army in full and the economy at the floor"
        );

        // GROW pays the economy in full, the army at the floor, plus the scout
        // term (no EXPAND target known, <3 cities) and the v6 body term.
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let expect_grow = economy
            + SHAPE_FLOOR_FRAC * army
            + SHAPE_GOAL_SCOUT
            + corner
            + SHAPE_GOAL_BODY;
        assert!(
            (goal_potential(&state, 1, &grow, None) - expect_grow).abs() < 1e-4,
            "GROW pays the economy in full and the army at the floor"
        );

        // EXPAND order: a one-tile close banks one step of the gradient.
        let ex = |orders| MacroGoal { orders, stance: Stance::Arm, save_target: None };
        let base = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        state.tribes.get_mut(&1).unwrap().units[0] = unit_at(1, UnitType::Warrior);
        let closer = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        assert!((closer - base - SHAPE_GOAL_EXPAND_PER_TILE).abs() < 1e-3);

        // Achieved target holds cap + completion bonus (no cliff on capture);
        // enemy-owned pays 0. Capture makes the tile an owned CITY.
        state.tiles.get_mut(&0).unwrap().owner = 1;
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            ..Default::default()
        });
        let achieved = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        let arm_only = goal_potential(&state, 1, &arm, None);
        let done = SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP as f32 + SHAPE_GOAL_EXPAND_DONE);
        assert!((achieved - arm_only - done).abs() < 1e-3);
        assert!(achieved >= closer);
        state.tiles.get_mut(&0).unwrap().owner = 2;
        let lost = goal_potential(&state, 1, &ex(vec![(OrderKind::Expand, 0)]), None);
        // v9: losing the tile also moves the always-on territory term, so the
        // baseline has to be re-taken in the SAME state — otherwise this
        // measures the ownership change as well as the retake gradient.
        let arm_lost = goal_potential(&state, 1, &arm, None);
        // v6: an enemy-taken village pays the retake-weighted approach
        // (unit at 1 is one tile out) instead of dropping to zero.
        let retake = SHAPE_GOAL_RETAKE_W * SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP - 1) as f32;
        assert!((lost - arm_lost - retake).abs() < 1e-3);
        // …and the territory term did move: holding the tile is worth
        // SHAPE_GOAL_TERRITORY at the active stance's weight.
        assert!(
            (arm_only - arm_lost - SHAPE_FLOOR_FRAC * SHAPE_GOAL_TERRITORY).abs() < 1e-3,
            "an owned tile pays the territory term (ARM -> floor weight)"
        );
    }

    #[test]
    fn scout_term_pays_full_then_half_with_target_until_third_city() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };

        // Each newly explored tile banks SHAPE_GOAL_SCOUT.
        let base = goal_potential(&state, 1, &grow, None);
        let mut tile = TileState::default();
        tile.explorers.insert(1);
        state.tiles.insert(50, tile);
        let one = goal_potential(&state, 1, &grow, None);
        assert!((one - base - SHAPE_GOAL_SCOUT).abs() < 1e-4);

        // A known EXPAND target halves the scout term (v4 — info retains
        // value alongside the approach gradient; unit at 60 is cheb 1 from 50).
        let with_target = MacroGoal {
            orders: vec![(OrderKind::Expand, 50)],
            stance: Stance::Grow,
            save_target: None,
        };
        let anchored = goal_potential(&state, 1, &with_target, None);
        let tribe = state.tribes.get(&1).unwrap();
        let spt0 = crate::functions::get_tribe_spt(&state, tribe) as f32;
        let approach = SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP - 1) as f32;
        let half_scout = SHAPE_GOAL_SCOUT * 0.5;
        // v9: the army also pays its floor under GROW. No tiles are OWNED and
        // no cities exist here, so the territory and ceiling terms are 0.
        let army_floor = SHAPE_FLOOR_FRAC * SHAPE_GOAL_ARM_PER_COST * unit_value(&tribe.units[0]);
        // + v6 body term: 1 unit, 0 cities → cap 1.
        assert!(
            (anchored - SHAPE_GOAL_SPT * spt0 - approach - half_scout - SHAPE_GOAL_BODY
                - army_floor)
                .abs()
                < 1e-3
        );
        // ARM never scouts; neither does a 3-city tribe.
        let arm = MacroGoal { orders: vec![], stance: Stance::Arm, save_target: None };
        let arm_phi = goal_potential(&state, 1, &arm, None);
        let army_full = SHAPE_GOAL_ARM_PER_COST * unit_value(&tribe.units[0]);
        let eco_floor = SHAPE_FLOOR_FRAC * SHAPE_GOAL_SPT * spt0;
        assert!((arm_phi - army_full - eco_floor).abs() < 1e-4);
        let t1 = state.tribes.get_mut(&1).unwrap();
        for _ in 0..3 {
            t1.cities.push(Default::default());
        }
        let done = goal_potential(&state, 1, &grow, None);
        let spt = crate::functions::get_tribe_spt(&state, state.tribes.get(&1).unwrap()) as f32;
        // Scout retires at 3 cities; the body term still pays its 1 unit
        // while the map stays unexplored, and v9's army floor pays alongside.
        // The three default cities own no territory, so their ceiling gap is 0.
        assert!((done - SHAPE_GOAL_SPT * spt - SHAPE_GOAL_BODY - army_floor).abs() < 1e-4);
    }

    #[test]
    fn goal_aux_pays_tech_fit_and_riders() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        use crate::states::TechnologyState;
        use crate::types::TechnologyType;
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Rider));
        state.tribes.insert(1, t1);
        let goal = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let aux = GoalAux {
            recommended_techs: vec![TechnologyType::Mining],
            rider_push: true,
            ..Default::default()
        };
        let base = goal_potential(&state, 1, &goal, None);
        // Rider push pays per living Rider; the unowned recommendation pays 0.
        let with_aux = goal_potential(&state, 1, &goal, Some(&aux));
        assert!((with_aux - base - SHAPE_GOAL_RIDER).abs() < 1e-3);
        // Owning the recommended tech banks the fit bonus.
        state.tribes.get_mut(&1).unwrap().tech_vanilla.push(TechnologyState {
            tech_type: TechnologyType::Mining,
            discovered: true,
            discovered_turn: 0,
        });
        let owned = goal_potential(&state, 1, &goal, Some(&aux));
        assert!((owned - with_aux - SHAPE_GOAL_TECH_FIT).abs() < 1e-3);
    }

    #[test]
    fn explorer_reward_pays_by_hidden_fraction() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::types::CityRewardType;
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 24, ..Default::default() });
        // v8: the first-city discount applies at 1 city, so a second city keeps
        // this test on the full-rate branch it was written to measure.
        t1.cities.push(crate::states::CityState { idx: 108, ..Default::default() });
        state.tribes.insert(1, t1);
        // Unlock stance isolates the explorer term (no SPT/scout/ARM terms).
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let before = goal_potential(&state, 1, &goal, None);
        state.tribes.get_mut(&1).unwrap().cities[0].rewards.push(CityRewardType::Explorer);
        // Fully hidden map, city 24 within EXPLORER_WALK_RANGE of corner 0:
        // full bonus + the lighthouse-chance lift.
        let dark = goal_potential(&state, 1, &goal, None);
        assert!(
            (dark - before - SHAPE_GOAL_EXPLORER - SHAPE_GOAL_EXPLORER_LIGHTHOUSE).abs() < 1e-3
        );
        // A center city reaches all four dark corners (cheb 5) but the lift
        // caps at two — "one, sometimes two lighthouses per explorer".
        let mut mid = GameState::default();
        let mut t2 = TribeState::default();
        let mut c = crate::states::CityState { idx: 60, ..Default::default() };
        c.rewards.push(CityRewardType::Explorer);
        t2.cities.push(c);
        t2.cities.push(crate::states::CityState { idx: 12, ..Default::default() });
        mid.tribes.insert(1, t2);
        let mid_phi = goal_potential(&mid, 1, &goal, None);
        let capped = SHAPE_GOAL_EXPLORER + 2.0 * SHAPE_GOAL_EXPLORER_LIGHTHOUSE;
        assert!((mid_phi - capped).abs() < 1e-3);
        // Fully revealed map: the bonus decays to ~0 (corners add lighthouse).
        for idx in 0..121 {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.explorers.insert(1);
        }
        let lit = goal_potential(&state, 1, &goal, None);
        assert!((lit - before - 4.0 * SHAPE_GOAL_LIGHTHOUSE).abs() < 1e-3);
    }

    #[test]
    fn goal_potential_pays_archetype_preferred_units() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Archer));
        t1.units.push(unit_at(61, UnitType::Warrior));
        state.tribes.insert(1, t1);
        // Unlock stance zeroes the stance term; only the unit bonus differs.
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let aux = GoalAux {
            preferred_units: vec![UnitType::Archer],
            ..Default::default()
        };
        let base = goal_potential(&state, 1, &goal, None);
        let with = goal_potential(&state, 1, &goal, Some(&aux));
        // Cost-scaled (v6): Archer costs 3 → 99, within 1% of the old flat 100.
        let archer_cost = get_unit_setting(UnitType::Archer).cost as f32;
        assert!((with - base - SHAPE_GOAL_ARCHETYPE_PER_COST * archer_cost).abs() < 1e-3);
    }

    #[test]
    fn archetype_per_cost_prices_knight_above_defender() {
        use crate::ai::oracle_macro::{GoalAux, MacroGoal, Stance};
        let mut state = GameState::default();
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Knight));
        state.tribes.insert(1, t1);
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let aux = GoalAux {
            preferred_units: vec![UnitType::Knight, UnitType::Defender],
            ..Default::default()
        };
        let knight = goal_potential(&state, 1, &goal, Some(&aux));
        state.tribes.get_mut(&1).unwrap().units[0] = unit_at(60, UnitType::Defender);
        let defender = goal_potential(&state, 1, &goal, Some(&aux));
        let k_cost = get_unit_setting(UnitType::Knight).cost as f32;
        let d_cost = get_unit_setting(UnitType::Defender).cost as f32;
        assert!((knight - SHAPE_GOAL_ARCHETYPE_PER_COST * k_cost).abs() < 1e-3);
        assert!((defender - SHAPE_GOAL_ARCHETYPE_PER_COST * d_cost).abs() < 1e-3);
        assert!(knight > defender, "a knight must out-price a defender head-for-head");
    }

    #[test]
    fn yield_structures_pay_per_partner_beyond_first() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::coords::Coords;
        use crate::states::StructureState;
        use crate::types::StructureType;
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 60, owner: 1, ..Default::default() });
        state.tribes.insert(1, t1);
        let rule = Coords { x: 5, y: 5, idx: 60 };
        for idx in [59, 70] {
            let tile = state.tiles.entry(idx).or_insert_with(TileState::default);
            tile.ruling_city_coords = Some(rule.clone());
        }
        // Partner tiles must be FRIENDLY territory to count (real-game rule).
        for idx in [58, 48, 69, 71] {
            state.tiles.entry(idx).or_insert_with(TileState::default).owner = 1;
        }
        let farm = |st: &mut GameState, idx: i32| {
            st.structures.insert(idx, Some(StructureState {
                structure_type: StructureType::Farm,
                ..Default::default()
            }));
        };
        // Unlock stance isolates the term.
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        state.structures.insert(59, Some(StructureState {
            structure_type: StructureType::Windmill,
            ..Default::default()
        }));
        farm(&mut state, 58);
        // One partner: the windmill pays for itself, no bonus.
        let one = goal_potential(&state, 1, &goal, None);
        assert!(one.abs() < 1e-4);
        // Second adjacent farm: +YIELD_ADJ × reward_pop(1) × 1.
        farm(&mut state, 48);
        let two = goal_potential(&state, 1, &goal, None);
        assert!((two - one - SHAPE_GOAL_YIELD_ADJ).abs() < 1e-4);
        // Forge scales by its reward_pop (2 per extra mine).
        state.structures.insert(70, Some(StructureState {
            structure_type: StructureType::Forge,
            ..Default::default()
        }));
        let mine = |st: &mut GameState, idx: i32| {
            st.structures.insert(idx, Some(StructureState {
                structure_type: StructureType::Mine,
                ..Default::default()
            }));
        };
        mine(&mut state, 69);
        mine(&mut state, 71);
        let with_forge = goal_potential(&state, 1, &goal, None);
        assert!((with_forge - two - 2.0 * SHAPE_GOAL_YIELD_ADJ).abs() < 1e-4);
        // Enemy-ruled structures pay nothing.
        state.tribes.get_mut(&1).unwrap().cities[0].owner = 2;
        let enemy = goal_potential(&state, 1, &goal, None);
        assert!(enemy.abs() < 1e-4);
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
        // ARM is exempt from the STRANDED discipline (a GROW-only term), but
        // v9's floors mean ARM is no longer identically zero here.
        let tribe = state.tribes.get(&1).unwrap();
        let arm_phi = goal_potential(&state, 1, &arm, None);
        let eco_floor = SHAPE_FLOOR_FRAC
            * (SHAPE_GOAL_SPT * crate::functions::get_tribe_spt(&state, tribe) as f32
                + SHAPE_GOAL_TERRITORY
                    * state.tiles.values().filter(|t| t.owner == 1).count() as f32
                + SHAPE_GOAL_CEILING
                    * tribe
                        .cities
                        .iter()
                        .map(|c| (city_ceiling(&state, 1, c) - c.level).max(0) as f32)
                        .sum::<f32>());
        assert!(
            (arm_phi - eco_floor).abs() < 1e-3,
            "ARM skips the stranded discipline but still pays the economy floor"
        );

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
        // v9: the two fruit also lift this city's CEILING from 2 to 3 (1+2 pop
        // clears the 3-pop threshold), so a point of headroom appears too.
        // That is the interaction to watch: the stranded penalty and the
        // headroom bonus are both built on `max_affordable_pop`, and they now
        // move together — pulling the SAME way, which is what we want. If the
        // headroom term had been signed as a gap penalty they would have
        // cancelled, and territory gains would have read as losses.
        let headroom_gained = SHAPE_GOAL_CEILING;
        assert!(
            (completable
                - stranded
                - SHAPE_GOAL_STRANDED
                - SHAPE_GOAL_COMPLETION / 3.0
                - headroom_gained)
                .abs()
                < 1e-3,
            "completable progress drops the penalty, earns the v7 bonus, and opens headroom"
        );

        // Threatened city (enemy adjacent) is exempt even when stranded.
        state.resources.insert(61, None);
        state.resources.insert(62, None);
        let restranded = goal_potential(&state, 1, &grow, None);
        assert!((restranded - stranded).abs() < 1e-3);
        let mut t2 = TribeState::default();
        t2.id = 2;
        t2.units.push(unit_at(61, UnitType::Warrior));
        state.tribes.insert(2, t2);
        let threatened = goal_potential(&state, 1, &grow, None);
        assert!(
            (threatened - restranded - SHAPE_GOAL_STRANDED).abs() < 1e-3,
            "threat exemption must lift the penalty"
        );
    }

    /// v7: holding stars must climb a gradient. Before this, banked stars
    /// appeared nowhere in Phi, so spending them on anything scored strictly
    /// beat holding and the measured policy was hand-to-mouth.
    #[test]
    fn savings_ramp_pays_for_banked_stars_and_keeps_the_economy_potential() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.id = 1;
        t1.stars = 0;
        state.tribes.insert(1, t1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };
        let saving =
            MacroGoal { orders: vec![], stance: Stance::Save, save_target: Some(20) };

        // Empty bank: SAVE must equal GROW — the stance itself costs nothing.
        let base = goal_potential(&state, 1, &grow, None);
        assert!((goal_potential(&state, 1, &saving, None) - base).abs() < 1e-3);

        // v9: TWO terms now scale with the bank — the SAVE ramp and the
        // risk-adjusted star option. Risk depends only on units/tiles, not on
        // stars, so it is constant across the three measurements below.
        let risk = position_risk(&state, 1, None);
        let option = |stars: i32| SHAPE_GOAL_STAR_OPTION * stars as f32 * risk;

        // Half banked pays half the ramp; full banked pays all of it.
        state.tribes.get_mut(&1).unwrap().stars = 10;
        let half = goal_potential(&state, 1, &saving, None);
        assert!((half - base - SHAPE_GOAL_SAVE / 2.0 - option(10)).abs() < 1e-3);
        state.tribes.get_mut(&1).unwrap().stars = 20;
        let full = goal_potential(&state, 1, &saving, None);
        assert!((full - base - SHAPE_GOAL_SAVE - option(20)).abs() < 1e-3);

        // Overshooting the TARGET pays no more ramp — but the option term does
        // keep paying, because unspent stars really are worth more optionality.
        state.tribes.get_mut(&1).unwrap().stars = 60;
        let over = goal_potential(&state, 1, &saving, None);
        assert!((over - full - (option(60) - option(20))).abs() < 1e-3);

        // Under GROW the ramp is worth nothing (stance-gated) but the option
        // term still pays: risk does not care which stance is active.
        state.tribes.get_mut(&1).unwrap().stars = 20;
        assert!((goal_potential(&state, 1, &grow, None) - base - option(20)).abs() < 1e-3);

        // A full bank must never outweigh spending it — otherwise the agent
        // banks forever rather than buying the batch it saved for.
        assert!(SHAPE_GOAL_SAVE < SHAPE_GOAL_SPT);

        // v9 HOARDING INVARIANT. Holding one more star must be worth less than
        // spending it, at MAXIMUM risk (=1.0) which is the option term's
        // ceiling. The cheapest pop route is ~2★/pop and a level needs
        // level+1 pop, so ~6★ buys the +1 SPT of a level-up at level 2 —
        // i.e. a star spent productively is worth about SPT/6. If the option
        // term ever exceeds that, waiting dominates buying at every price and
        // spend/income collapses to zero instead of the 1.00 we measured.
        const STARS_PER_SPT_POINT: f32 = 6.0;
        assert!(
            SHAPE_GOAL_STAR_OPTION < SHAPE_GOAL_SPT / STARS_PER_SPT_POINT,
            "star option ({SHAPE_GOAL_STAR_OPTION}) must stay under the ~{} Φ a \
             productively-spent star buys, or the model hoards",
            SHAPE_GOAL_SPT / STARS_PER_SPT_POINT
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

    #[test]
    fn contested_target_pays_one_extra_converger() {
        use crate::ai::oracle_macro::{MacroGoal, OrderKind, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        add_visible_village(&mut state, 5);
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(3, UnitType::Warrior)); // assigned, d=2
        t1.units.push(unit_at(8, UnitType::Warrior)); // second, d=3
        state.tribes.insert(1, t1);
        let ex = MacroGoal {
            orders: vec![(OrderKind::Expand, 5)],
            stance: Stance::Arm,
            save_target: None,
        };
        let uncontested = goal_potential(&state, 1, &ex, None);

        // Enemy squatter on the village: the second unit's gradient pays at
        // half weight on top.
        let mut t2 = TribeState::default();
        t2.id = 2;
        t2.units.push(unit_at(5, UnitType::Warrior));
        state.tribes.insert(2, t2);
        let contested = goal_potential(&state, 1, &ex, None);
        let second = SHAPE_GOAL_CONTEST_SECOND
            * SHAPE_GOAL_EXPAND_PER_TILE
            * (SHAPE_PROX_CAP - 3) as f32;
        assert!(
            (contested - uncontested - second).abs() < 1e-3,
            "contested village must pay the second unit ({contested} vs {uncontested})"
        );
    }

    #[test]
    fn grow_body_term_pays_to_cap_only() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(60, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let grow = MacroGoal { orders: vec![], stance: Stance::Grow, save_target: None };

        // v9: a unit is no longer paid ONLY by the body term — the military
        // floor pays SHAPE_FLOOR_FRAC of its HP-weighted worth under every
        // stance. So "beyond the cap" now means "the body term stops", not
        // "the unit is free".
        let floor_per_warrior =
            SHAPE_FLOOR_FRAC * SHAPE_GOAL_ARM_PER_COST * unit_value(&unit_at(0, UnitType::Warrior));

        // 0 cities → cap 1: the second unit adds only the floor.
        let one = goal_potential(&state, 1, &grow, None);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(61, UnitType::Warrior));
        let two = goal_potential(&state, 1, &grow, None);
        assert!(
            (two - one - floor_per_warrior).abs() < 1e-4,
            "beyond-cap unit must pay the military floor and nothing more"
        );

        // A city raises the cap to 2: the second unit now pays.
        state.tribes.get_mut(&1).unwrap().cities.push(crate::states::CityState {
            idx: 0,
            owner: 1,
            ..Default::default()
        });
        let capped_two = goal_potential(&state, 1, &grow, None);
        let one_city_one_unit = {
            let mut s2 = state.clone();
            s2.tribes.get_mut(&1).unwrap().units.pop();
            goal_potential(&s2, 1, &grow, None)
        };
        assert!(
            (capped_two - one_city_one_unit - SHAPE_GOAL_BODY - floor_per_warrior).abs() < 1e-3,
            "unit within raised cap must pay the body term ON TOP of the floor"
        );
    }

    #[test]
    fn market_pays_star_adjacency_beyond_first_partner() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::coords::Coords;
        use crate::states::StructureState;
        use crate::types::StructureType;
        let mut state = GameState::default();
        state.settings.size = 11;
        let mut t1 = TribeState::default();
        t1.cities.push(crate::states::CityState { idx: 60, owner: 1, ..Default::default() });
        state.tribes.insert(1, t1);
        let rule = Coords { x: 5, y: 5, idx: 60 };
        state.tiles.entry(59).or_insert_with(TileState::default).ruling_city_coords =
            Some(rule);
        for idx in [58, 48, 59] {
            state.tiles.entry(idx).or_insert_with(TileState::default).owner = 1;
        }
        let put = |st: &mut GameState, idx: i32, s: StructureType| {
            st.structures.insert(idx, Some(StructureState {
                structure_type: s,
                ..Default::default()
            }));
        };
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        put(&mut state, 59, StructureType::Market);
        put(&mut state, 58, StructureType::Windmill);
        // One hub partner: no bonus (the market pays for itself).
        let one = goal_potential(&state, 1, &goal, None);
        assert!(one.abs() < 1e-4);
        // Second hub: +YIELD_ADJ_STARS × reward_stars(1) × 1.
        put(&mut state, 48, StructureType::Sawmill);
        let two = goal_potential(&state, 1, &goal, None);
        assert!((two - one - SHAPE_GOAL_YIELD_ADJ_STARS).abs() < 1e-4);
    }

    #[test]
    fn standing_forest_in_territory_holds_option_value() {
        use crate::ai::oracle_macro::{MacroGoal, Stance};
        use crate::types::TerrainType;
        let mut state = GameState::default();
        state.settings.size = 11;
        state.tribes.insert(1, TribeState::default());
        let goal = MacroGoal { orders: vec![], stance: Stance::Unlock, save_target: None };
        let tile = state.tiles.entry(60).or_insert_with(TileState::default);
        tile.owner = 1;
        tile.terrain_type = TerrainType::Forest;
        let with = goal_potential(&state, 1, &goal, None);
        assert!((with - SHAPE_GOAL_FOREST_STANDING).abs() < 1e-4);
        // Cleared (Field) or enemy-owned forest pays nothing.
        state.tiles.get_mut(&60).unwrap().terrain_type = TerrainType::Field;
        assert!(goal_potential(&state, 1, &goal, None).abs() < 1e-4);
        state.tiles.get_mut(&60).unwrap().terrain_type = TerrainType::Forest;
        state.tiles.get_mut(&60).unwrap().owner = 2;
        assert!(goal_potential(&state, 1, &goal, None).abs() < 1e-4);
    }

    #[test]
    fn unexplored_or_owned_villages_pay_no_proximity() {
        let mut state = GameState::default();
        add_visible_village(&mut state, 0);
        state.tiles.get_mut(&0).unwrap().explorers.clear(); // fogged
        let mut t1 = TribeState::default();
        t1.units.push(unit_at(2, UnitType::Warrior));
        state.tribes.insert(1, t1);
        let fogged = dev_potential(&state, 1);
        let cost = get_unit_setting(UnitType::Warrior).cost as f32;
        assert!((fogged - SHAPE_ARMY_PER_COST * cost).abs() < 1e-4);

        state.tiles.get_mut(&0).unwrap().explorers.insert(1);
        state.tiles.get_mut(&0).unwrap().owner = 2; // captured by someone
        assert!((dev_potential(&state, 1) - fogged).abs() < 1e-4);
    }
}
