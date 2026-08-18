//! `goal_potential`, the T2/T3 goal-priced Φ (Aug 2026 taxonomy split out of
//! reward.rs to keep every file under ~1000 lines). One function, moved
//! verbatim — decomposing its internals is a reward-shaping change, not a
//! file-organization one, so it stays as the single atomic unit it always
//! was. Re-exported through `reward` so existing `crate::ai::reward::X` call
//! sites keep resolving.

use super::cheb;
use super::dev_potential::SHAPE_PROX_CAP;
use super::economy_completion::{completion_progress, completion_stranded};
use super::goal_shape_consts::*;
use crate::states::GameState;

/// Goal potential Φ_goal for `player` under `goal` (score-equivalent units).
pub fn goal_potential(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
) -> f32 {
    goal_potential_with_threats(state, player, goal, aux, None)
}

/// Same as [`goal_potential`], but reuses an already-computed threat list
/// (see `combat::threat_units`) for the Defend-order term's `city_risks`
/// call instead of re-scanning for threats. Enemy threats never depend on
/// the acting player's own move, so a caller ranking many of its own
/// candidates against the same board (`macro_exec::rank_plies`) can compute
/// threats once per ply instead of once per candidate — `city_risks`'s
/// per-candidate re-scan was 64-86% of actor CPU time under macro-mcts
/// before this split (EXP_ELO_061 throughput investigation, Aug 2026).
pub fn goal_potential_with_threats(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
    threats: Option<&[(crate::states::UnitState, f32)]>,
) -> f32 {
    use crate::ai::oracle_macro::{OrderKind, Stance};
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let spt = crate::functions::get_tribe_spt(state, tribe) as f32;
    let mut phi = match goal.stance {
        // SAVE is an economy stance: it keeps GROW's whole potential and adds
        // the ramp below, so banking never costs the economy gradient.
        Stance::Grow | Stance::Save => SHAPE_GOAL_SPT * spt,
        // v9: ARM is no longer economy-blind. It holds 85% of plies after turn
        // 10, and with only the army term the whole mid-game carried zero
        // economy gradient — the window where a human is pushing cities to
        // level 5. A giant IS an army purchase; it is bought with population.
        // v11 (Verdi, Aug 2026): ARM is a magnitude, not a bit. The discrete
        // stance can commit on a marginal signal (one visible unit, a slim
        // momentum edge — see `stance_pressure`), and the old flat switch
        // paid full army pricing and zero GROW rate the instant it did.
        // Blend the SPT RATE (the economy-vs-military spending priority) by
        // `arm_strength`, the same continuous read `passes_stance_tech_mask`
        // gates the hard tech mask on: at 1.0 this is identical to the old
        // formula, and as intensity falls the rate interpolates toward
        // GROW's instead of cutting the economy gradient off a cliff.
        // The army-VALUE term is deliberately NOT blended: it prices units
        // already held, which does not become less true as intensity falls,
        // and blending it made a besieged state score higher than a quiet
        // one purely by "revealing" an unrelated unit's value, overpowering
        // the city_risk penalty below (caught by
        // `city_risk_is_priced_without_any_defend_order`).
        Stance::Arm => {
            let army_value: f32 =
                tribe.units.iter().map(|u| crate::rules::combat::unit_worth(u) as f32).sum();
            let intensity = aux.map_or(1.0, |a| a.arm_strength);
            let spt_rate = intensity * SHAPE_GOAL_ARM_SPT + (1.0 - intensity) * SHAPE_GOAL_SPT;
            SHAPE_GOAL_ARM_PER_COST * army_value + spt_rate * spt
        }
        Stance::Unlock => 0.0,
    };
    // v9: the completion bonus now pays under ARM too — level 5 is where the
    // super unit comes from, so progress toward it is armament, not a
    // distraction. The stranded TAX stays off ARM for the v6 reason: combat
    // spending shouldn't be penalised for levels it never planned to finish.
    if matches!(goal.stance, Stance::Grow | Stance::Save | Stance::Arm) {
        phi += SHAPE_GOAL_COMPLETION * completion_progress(state, player);
    }
    // EXP_ELO_050: the cost of losing a city, priced into the potential
    // itself rather than attached to a Defend order. Two consequences that
    // the order-keyed version could not have: it is live on EVERY stance and
    // every turn (the 049 fixture lost its capital on a Grow turn whose
    // directive named no Defend at all), and because it is a potential, the
    // ply that steps the last unit OFF a threatened city pays the same
    // amount the ply that garrisons it earns. PREVENTION is what it buys:
    // 049 measured a parked Giant cleared 6% of the time, so the cheap move
    // is to never let the tile go empty in the first place.
    // T2 assessed the threat facts and handed them down in the aux; T3 prices
    // its RESPONSE by re-resolving `residual_risk` against live occupancy —
    // the frozen assessment alone is constant across a turn's plies, so it
    // would have no gradient. Losing the city reads RISK_LOST, so no line
    // can win potential by letting one fall. Without an aux there is no
    // assessment to price, the convention every aux-carried term follows.
    if let Some(a) = aux {
        phi -= SHAPE_GOAL_CONNECT
            * a.connect_remaining.iter().map(|(_, n)| *n as f32).sum::<f32>();
    }
    if let Some(a) = aux {
        phi -= SHAPE_GOAL_CITY_RISK
            * crate::ai::combat::residual_city_loss(state, player, &a.city_risk);
    }
    if matches!(goal.stance, Stance::Grow | Stance::Save) {
        phi -= SHAPE_GOAL_STRANDED * completion_stranded(state, player) as f32;
    }
    // v10: pay for super units in EVERY stance. Only ARM ever outbid the Park's
    // +250 score, so the level-5 pick inverted the moment the macro was not
    // arming — and EXP_ELO_030, by producing more level-5 cities under SAVE,
    // made that worse (supers 89 -> 85 while parks went 23 -> 34).
    let supers = tribe
        .units
        .iter()
        .filter(|u| {
            !u.converted && crate::settings::units::get_unit_setting(u.unit_type).is_super
        })
        .count() as f32;
    if supers > 0.0 {
        let urgency = match goal.stance {
            Stance::Save => goal
                .save_target
                .as_ref()
                .map_or(0.0, |l| save_progress(state, player, l)),
            _ => 0.0,
        };
        phi += SHAPE_GOAL_SUPER * (1.0 - SHAPE_GOAL_SUPER_ECON_DAMP * urgency) * supers;
    }
    // v7 savings ramp: progress toward the banked batch is itself scored, so
    // holding stars climbs a gradient instead of sitting in a flat valley that
    // any purchase strictly beats. This is what makes a multi-turn plan legible
    // to a search whose horizon is one game turn — the ramp is visible at
    // depth 1, so the tree never has to reach the purchase to value it.
    // v11: `goal.save_target` is carried regardless of which stance won the
    // discrete pick (`compute_macro_goal_cached` never clears it when Arm
    // pre-empts Save) — previously the ramp went fully dark the instant a
    // marginal Arm signal won, so a Forge/Smithery plan lost its whole
    // gradient over a covered skirmish. Pay it in full under SAVE, pay the
    // (1 - arm_strength) remainder under ARM so the plan stays visible in
    // proportion to how little the threat actually earns.
    if let Some(lane) = goal.save_target.as_ref().filter(|l| l.cost > 0) {
        let save_weight = match goal.stance {
            Stance::Save => 1.0,
            Stance::Arm => (1.0 - aux.map_or(1.0, |a| a.arm_strength)).clamp(0.0, 1.0),
            _ => 0.0,
        };
        if save_weight > 0.0 {
            phi += SHAPE_GOAL_SAVE * save_weight * save_progress(state, player, lane);
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
    // EXP_ELO_040: Defend orders — coverage leash, not garrison pinning.
    // Pay per assigned covering unit (full in 1-turn strike reach, half in
    // the 2-turn ring) scaled by live urgency; pay tile-holding only while
    // the garrison is load-bearing; on shortfall, recall the single nearest
    // unassigned unit with an approach gradient. Threats recomputed from
    // state each eval, so prep outcomes (a trained unit, a road, a tech
    // that extends reach) raise Φ the ply they land — no discrete planner.
    let attack_targets: Vec<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Attack)
        .map(|(_, i)| *i)
        .collect();
    if width > 0 && goal.orders.iter().any(|(k, _)| *k == OrderKind::Defend) {
        let city_threats = match threats {
            Some(t) => crate::ai::combat::city_risks_with_threats(state, player, t),
            None => crate::ai::combat::city_risks(state, player),
        };
        for (kind, idx) in &goal.orders {
            if *kind != OrderKind::Defend {
                continue;
            }
            let Some(th) = city_threats.iter().find(|t| t.city == *idx) else {
                continue; // stale order: threat cleared, nothing to pay
            };
            let urgency = if th.at_risk { 1.0 } else { 0.5 };
            let plan = crate::ai::combat::defend_plan(state, player, th, &attack_targets);
            for (_, sat) in &plan.assigned {
                phi += SHAPE_GOAL_DEFEND_COVER * urgency * sat;
            }
            if plan.hold_needed {
                phi += SHAPE_GOAL_DEFEND_HOLD * urgency;
            }
            // EXP_ELO_042: recall never conscripts attack-committed units;
            // with none free, shortfall drives prep, not un-commitment.
            if plan.shortfall > 0.0 {
                let assigned: std::collections::HashSet<i32> =
                    plan.assigned.iter().map(|(t, _)| *t).collect();
                if let Some(d) = tribe
                    .units
                    .iter()
                    .filter(|u| {
                        !assigned.contains(&u.coords.idx)
                            && !crate::ai::combat::attack_committed(
                                state,
                                player,
                                u,
                                *idx,
                                &attack_targets,
                            )
                    })
                    .map(|u| cheb(u.coords.idx, *idx, width))
                    .min()
                {
                    phi += SHAPE_GOAL_DEFEND_COVER * urgency * 0.5
                        * ((SHAPE_PROX_CAP - d).max(0) as f32 / SHAPE_PROX_CAP as f32);
                }
            }
        }
    }
    // EXP_ELO_042: symmetric offense. Siege-hold pays by STATE-FACT (a unit
    // standing on an enemy city keeps its pay through Attack-order flicker);
    // ring units press toward ordered targets, capped and deterministic.
    if width > 0 {
        let mut sieging: Vec<i32> = Vec::new();
        for u in &tribe.units {
            if let Some(c) = crate::functions::get_city_at(state, u.coords.idx) {
                if c.owner != player && c.owner != 0 {
                    phi += SHAPE_GOAL_ATTACK_PRESS * SHAPE_GOAL_SIEGE_HOLD_MULT;
                    sieging.push(u.coords.idx);
                }
            }
        }
        for &h in &attack_targets {
            let mut cands: Vec<(i32, f32, i32)> = Vec::new(); // (tile, sat, dist)
            for u in &tribe.units {
                if sieging.contains(&u.coords.idx) {
                    continue; // already paid by the latch
                }
                let d = cheb(u.coords.idx, h, width);
                let m = crate::functions::get_unit_movement(state, u);
                let sat = if crate::ai::combat::unit_covers_threat(state, u, h) {
                    1.0
                } else if d <= 2 * m {
                    0.5
                } else {
                    continue;
                };
                cands.push((u.coords.idx, sat, d));
            }
            cands.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
            for &(_, sat, _) in cands.iter().take(4) {
                phi += SHAPE_GOAL_ATTACK_PRESS * sat;
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
        // Owned tiles only (`partner_ceiling_with` filters on ownership), and
        // owning a tile implies having explored it — so capacity carries no FOW
        // leak. It does read the raw resource map, i.e. it can see metal under
        // a mountain you own before Mining: a tech-visibility read the engine
        // already makes here deliberately, so the potential does not depend on
        // whose turn it is.
        let ceiling = crate::rules::economy::partner_ceiling_with(
            state,
            s_idx,
            &setting.adjacent_types,
            player,
        );
        let extra_real = (partners - 1).max(0) as f32;
        let extra_cap = (((ceiling - 1).max(0)) as f32 - extra_real).max(0.0);
        let weight = extra_real + SHAPE_GOAL_YIELD_CAPACITY_W * extra_cap;
        phi += SHAPE_GOAL_YIELD_ADJ * setting.reward_pop.max(0) as f32 * weight;
        phi += SHAPE_GOAL_YIELD_ADJ_STARS * setting.reward_stars.max(0) as f32 * weight;
    }
    // Standing-forest option value (v5): clearing pays only when the
    // follow-up (build / level-up funding) outweighs the lost option.
    let own_forests = state
        .tiles
        .values()
        .filter(|t| t.owner == player && t.terrain_type == crate::types::TerrainType::Forest)
        .count();
    phi += SHAPE_GOAL_FOREST_STANDING * own_forests as f32;
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
            let preferred = tribe
                .units
                .iter()
                .filter(|u| aux.preferred_units.contains(&u.unit_type))
                .map(crate::rules::combat::unit_worth)
                .sum::<i32>();
            phi += SHAPE_GOAL_LANE_PER_COST * preferred as f32;
        }
    }
    phi
}

#[cfg(test)]
#[path = "goal_potential_tests.rs"]
mod tests;
