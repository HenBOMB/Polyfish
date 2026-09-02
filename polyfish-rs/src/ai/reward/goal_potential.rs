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

/// EXP_ELO_114 diagnostic (temporary): how often `city_open_exposed`
/// actually fires (an owned city, open, reachable by a visible enemy next
/// turn, no active Defend order) across every `goal_potential` evaluation —
/// mirrors EXP_ELO_111's `STEP_LETHAL_ENTRY_CANDIDATES`/`_FIRES` pair.
pub static CITY_OPEN_EXPOSED_EVALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static CITY_OPEN_EXPOSED_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// EXP_ELO_116 diagnostic (temporary): how often the prepare-pull term is
/// eligible to fire (`goal.prepare` set, `unit_goals` threaded), how many of
/// those evaluations actually pull a unit, and how many were suppressed
/// entirely by a live Defend order (the starve case the design review
/// flagged: if this is high while ARM dominates mid-game, the pull is dark
/// exactly when the army is largest).
pub static PREPARE_PULL_EVALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PREPARE_PULL_FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PREPARE_SUPPRESSED_BY_DEFEND: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

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
    goal_potential_with_unit_goals(state, player, goal, aux, threats, None)
}

/// Same as [`goal_potential_with_threats`], but additionally prices EXPAND
/// against a persistent per-unit `UnitGoalStore` instead of re-deriving the
/// unit<->target matching fresh on every call. `None` (the only value every
/// caller except `MacroMctsAgent`'s real trajectory passes) is byte-for-byte
/// the legacy ephemeral-matching behavior.
pub fn goal_potential_with_unit_goals(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
    threats: Option<&[(crate::states::UnitState, f32)]>,
    unit_goals: Option<&crate::ai::search::unit_goals::UnitGoalStore>,
) -> f32 {
    goal_potential_with_belief(state, player, goal, aux, threats, unit_goals, None, None)
}

/// Same as [`goal_potential_with_unit_goals`], but additionally weights the
/// explorer and per-city completion terms by `MapBelief`'s frontier signal
/// (enemy-facing ground > possible-village ground > plain fog) instead of
/// treating all dark ground the same. `None` is byte-for-byte the legacy
/// corner-count/uniform-progress behavior. `belief` is a pure function of
/// the explored set (`MapBelief::observe`), so it is safe to compute once
/// per ply and reuse across every candidate's phi_post the same way
/// `threats` already is — see `macro_exec::rank_plies`. `pre_health`
/// (EXP_ELO_110, unit id -> health at ply start) floors the Defend
/// waterfall's own-roster contributions against a self-wound shrinking them
/// mid-comparison — see `combat::defend_plan_impl`'s doc comment. Also
/// safe to compute once per ply and reuse for both phi_pre and phi_post.
pub fn goal_potential_with_belief(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
    threats: Option<&[(crate::states::UnitState, f32)]>,
    unit_goals: Option<&crate::ai::search::unit_goals::UnitGoalStore>,
    belief: Option<&crate::ai::belief::map::MapBelief>,
    pre_health: Option<&rustc_hash::FxHashMap<u32, f32>>,
) -> f32 {
    goal_potential_inner(state, player, goal, aux, threats, unit_goals, belief, pre_health, &mut None)
}

/// Aug 2026 (reward_lab): same computation as [`goal_potential_with_unit_goals`],
/// but returns every named term's individual contribution alongside the sum
/// -- the observability the tuning loop needed after the turn-1
/// capital-block hunt required a temporary, one-off `POLYFISH_DPHI_PROBE`
/// rebuild to get an aggregate-only Δφ. A term can appear more than once
/// (loops over cities/targets/structures emit one entry per iteration) --
/// sum by label to get that term's total, same convention `POLYFISH_DPHI_PROBE`
/// already uses for per-candidate rows. `None`-sink production callers are
/// completely unaffected: `goal_potential_inner` takes the sink by
/// `&mut Option<&mut Vec<_>>` so the `None` path never allocates.
pub fn goal_potential_breakdown(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
    threats: Option<&[(crate::states::UnitState, f32)]>,
    unit_goals: Option<&crate::ai::search::unit_goals::UnitGoalStore>,
    belief: Option<&crate::ai::belief::map::MapBelief>,
    pre_health: Option<&rustc_hash::FxHashMap<u32, f32>>,
) -> (f32, Vec<(&'static str, f32)>) {
    let mut bd = Vec::new();
    let mut sink = Some(&mut bd);
    let phi = goal_potential_inner(
        state, player, goal, aux, threats, unit_goals, belief, pre_health, &mut sink,
    );
    (phi, bd)
}

/// Term-labeled accumulator: every `phi +=`/`phi -=` site below reports
/// through this instead, so the breakdown sink and the scalar sum can never
/// drift apart -- there is exactly one computation, not two maintained in
/// parallel.
struct PhiAcc<'a, 'b> {
    phi: f32,
    bd: &'a mut Option<&'b mut Vec<(&'static str, f32)>>,
}
impl PhiAcc<'_, '_> {
    fn add(&mut self, label: &'static str, v: f32) {
        self.phi += v;
        if let Some(bd) = self.bd.as_deref_mut() {
            bd.push((label, v));
        }
    }
    fn sub(&mut self, label: &'static str, v: f32) {
        self.add(label, -v);
    }
}

/// Multiplier on a target's approach pull: `SHAPE_GOAL_RUIN_W` while it's a
/// Ruin and the tribe hasn't found its first village yet (`cities.len() < 2`
/// -- still just the capital), 1.0 otherwise. Applied on top of whatever
/// weight the caller already computed (retake, fog-guess, etc.), never in
/// place of it.
fn ruin_pull_discount(state: &GameState, tribe: &crate::states::TribeState, idx: i32) -> f32 {
    if tribe.cities.len() >= 2 {
        return 1.0;
    }
    let is_ruin = crate::functions::get_structure_at(state, idx)
        .map_or(false, |s| s.structure_type == crate::types::StructureType::Ruin);
    if is_ruin {
        SHAPE_GOAL_RUIN_W
    } else {
        1.0
    }
}

/// EXP_ELO_118: 1.0 for a cheap unit (a Warrior, `unit_worth`<=2), floored
/// at 0.4 for an expensive one (a Giant, `unit_worth`>=5) -- a static,
/// per-unit function of the unit alone, never of live board state, so it
/// can't reproduce EXP_ELO_117's state-vs-transition inversion (a per-unit
/// weight is identical whether read pre- or post-move for that same unit).
fn defend_cheapness(u: &crate::states::UnitState) -> f32 {
    (2.0 / crate::rules::combat::unit_worth(u).max(1) as f32).clamp(0.4, 1.0)
}

fn goal_potential_inner(
    state: &GameState,
    player: i32,
    goal: &crate::ai::oracle_macro::MacroGoal,
    aux: Option<&crate::ai::oracle_macro::GoalAux>,
    threats: Option<&[(crate::states::UnitState, f32)]>,
    unit_goals: Option<&crate::ai::search::unit_goals::UnitGoalStore>,
    belief: Option<&crate::ai::belief::map::MapBelief>,
    pre_health: Option<&rustc_hash::FxHashMap<u32, f32>>,
    breakdown: &mut Option<&mut Vec<(&'static str, f32)>>,
) -> f32 {
    use crate::ai::oracle_macro::{OrderKind, Stance};
    let Some(tribe) = state.tribes.get(&player) else {
        return 0.0;
    };
    let spt = crate::functions::get_tribe_spt(state, tribe) as f32;
    let mut acc = PhiAcc { phi: 0.0, bd: breakdown };
    match goal.stance {
        // SAVE is an economy stance: it keeps GROW's whole potential and adds
        // the ramp below, so banking never costs the economy gradient.
        Stance::Grow | Stance::Save => acc.add("spt", SHAPE_GOAL_SPT * spt),
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
            acc.add("arm_value", SHAPE_GOAL_ARM_PER_COST * army_value);
            acc.add("arm_spt", spt_rate * spt);
        }
        Stance::Unlock => {}
    };
    // v9: the completion bonus now pays under ARM too — level 5 is where the
    // super unit comes from, so progress toward it is armament, not a
    // distraction. The stranded TAX stays off ARM for the v6 reason: combat
    // spending shouldn't be penalised for levels it never planned to finish.
    if matches!(goal.stance, Stance::Grow | Stance::Save | Stance::Arm) {
        acc.add(
            "completion_progress",
            SHAPE_GOAL_COMPLETION * completion_progress(state, player, belief),
        );
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
        acc.sub(
            "connect",
            SHAPE_GOAL_CONNECT * a.connect_remaining.iter().map(|(_, n)| *n as f32).sum::<f32>(),
        );
    }
    if let Some(a) = aux {
        acc.sub(
            "city_risk",
            SHAPE_GOAL_CITY_RISK * crate::ai::combat::residual_city_loss(state, player, &a.city_risk),
        );
    }
    // Aug 2026: a unit sitting on one of our own cities blocks Summon there
    // for as long as it stays (see SHAPE_CITY_TRAIN_BLOCKED's doc comment).
    // Gated on `unit_goals.is_some()` like SHAPE_UNIT_GOAL_PER_TILE/COMPLETE
    // above it -- real-trajectory only, same reasoning: this prices ONE
    // unit's occupancy choice, not a tribe-wide fact, so it belongs with the
    // other per-unit terms rollouts don't see. That scope also sidesteps a
    // real collision found while adding this: city_risk/DEFEND already use
    // occupancy as a POSITIVE garrison-value signal (`residual_city_loss`,
    // unconditional on any Defend order per EXP_ELO_050), and an earlier
    // draft that instead tried to exempt threatened cities from THIS penalty
    // broke `city_risk_is_priced_without_any_defend_order` (exempting made a
    // besieged city score higher than a quiet one -- the -200 loss swamped
    // city_risk's own ~4-magnitude delta). Every one of those garrison
    // tests calls `goal_potential()`, which always passes `None` here, so
    // this term never competes with them at all.
    if unit_goals.is_some() {
        for city in &tribe.cities {
            if crate::functions::get_city_unit_count(state, city) > city.level {
                continue; // no train capacity to protect
            }
            let occupied_by_us = crate::functions::get_unit_at(state, city.idx)
                .map_or(false, |u| u.owner == player);
            if occupied_by_us {
                acc.sub("city_train_blocked", SHAPE_CITY_TRAIN_BLOCKED);
            }
        }
    }
    if matches!(goal.stance, Stance::Grow | Stance::Save) {
        acc.sub("stranded", SHAPE_GOAL_STRANDED * completion_stranded(state, player) as f32);
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
        acc.add("super", SHAPE_GOAL_SUPER * (1.0 - SHAPE_GOAL_SUPER_ECON_DAMP * urgency) * supers);
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
            acc.add("save_ramp", SHAPE_GOAL_SAVE * save_weight * save_progress(state, player, lane));
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
        let mut achieved_targets: std::collections::HashSet<i32> =
            std::collections::HashSet::new();
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
                        acc.add(
                            "expand_achieved",
                            SHAPE_GOAL_EXPAND_PER_TILE * (SHAPE_PROX_CAP as f32 + SHAPE_GOAL_EXPAND_DONE),
                        );
                        achieved_targets.insert(*idx);
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
            weight *= ruin_pull_discount(state, tribe, *idx);
            approach_targets.push(*idx);
            target_weight.insert(*idx, weight);
        }
        match unit_goals {
            None => {
                if !approach_targets.is_empty() {
                    let w_of = |t: i32| target_weight.get(&t).copied().unwrap_or(1.0);
                    let pairs = crate::ai::oracle_macro::assign_expand_targets(
                        state,
                        player,
                        &approach_targets,
                    );
                    for (unit_idx, target) in &pairs {
                        let d = cheb(*unit_idx, *target, width);
                        acc.add(
                            "expand_approach",
                            SHAPE_GOAL_EXPAND_PER_TILE * w_of(*target) * (SHAPE_PROX_CAP - d).max(0) as f32,
                        );
                    }
                    // Targets beyond the unit count keep their closest-unit
                    // gradient, so an under-scouted map still pulls.
                    let assigned: std::collections::HashSet<i32> =
                        pairs.iter().map(|(_, t)| *t).collect();
                    for target in approach_targets.iter().filter(|t| !assigned.contains(t)) {
                        let d = tribe
                            .units
                            .iter()
                            .map(|u| cheb(u.coords.idx, *target, width))
                            .min()
                            .unwrap_or(i32::MAX);
                        acc.add(
                            "expand_approach_unassigned",
                            SHAPE_GOAL_EXPAND_PER_TILE * w_of(*target) * (SHAPE_PROX_CAP - d).max(0) as f32,
                        );
                    }
                    // v6: a CONTESTED target pays one extra converger — the
                    // nearest unit not already assigned to it (and not
                    // already standing on the target itself, which is paid
                    // separately via siege-hold/completion) — at half
                    // gradient. Exactly one; no dogpile. "Contested" is
                    // tile ownership (still enemy, i.e. not yet captured),
                    // not live-defender presence: capturing a city is a
                    // separate move from killing its garrison, so a target
                    // stays genuinely contested — worth a second unit
                    // converging to help hold/capture it — for exactly as
                    // long as it isn't ours, not just while its original
                    // defender is still alive. Gating on live-unit presence
                    // alone made this collapse to zero the instant an
                    // attack killed the garrison, wrongly erasing the
                    // reward for the very progress meant to earn it.
                    for (unit_idx, target) in &pairs {
                        let visible = state
                            .tiles
                            .get(target)
                            .map_or(false, |t| t.explorers.contains(&player));
                        let contested = visible
                            && (crate::functions::get_unit_at(state, *target)
                                .map_or(false, |u| u.owner != player)
                                || state
                                    .tiles
                                    .get(target)
                                    .map_or(false, |t| t.owner != 0 && t.owner != player));
                        if !contested {
                            continue;
                        }
                        let second = tribe
                            .units
                            .iter()
                            .map(|u| u.coords.idx)
                            .filter(|idx| idx != unit_idx && idx != target)
                            .map(|idx| cheb(idx, *target, width))
                            .min();
                        if let Some(d) = second {
                            acc.add(
                                "expand_contest_second",
                                SHAPE_GOAL_CONTEST_SECOND
                                    * SHAPE_GOAL_EXPAND_PER_TILE
                                    * w_of(*target)
                                    * (SHAPE_PROX_CAP - d).max(0) as f32,
                            );
                        }
                    }
                }
            }
            Some(store) => {
                // Sticky per-unit pricing: `pairs` reads the store's frozen
                // assignment directly — no re-matching — so one unit's
                // candidate move can never shift which target a different,
                // non-moving unit is priced against (the cross-talk the
                // legacy fresh-match-every-call design had). Independent of
                // whether `goal.orders` still names the target this ply: a
                // stance flip that drops Expand from the ballot must not
                // cut a mid-pursuit unit's pull, or it wanders back — the
                // bug that motivated this design.
                let pairs: Vec<(i32, i32)> = tribe
                    .units
                    .iter()
                    .filter_map(|u| {
                        let g = store.active(u.id)?;
                        (g.kind == OrderKind::Expand).then_some((u.coords.idx, g.target))
                    })
                    .collect();
                let w_of = |t: i32| {
                    target_weight.get(&t).copied().unwrap_or_else(|| {
                        // Unexplored (fog-guess) fallback targets always pay
                        // approach at weight 1.0 -- reading their owner
                        // would leak FOW, same reasoning as the goal.orders
                        // pre-loop above.
                        state.tiles.get(&t).map_or(1.0, |tile| {
                            (if tile.explorers.contains(&player) && tile.owner != 0 && tile.owner != player {
                                SHAPE_GOAL_RETAKE_W
                            } else {
                                1.0
                            }) * ruin_pull_discount(state, tribe, t)
                        })
                    })
                };
                for (unit_idx, target) in &pairs {
                    if achieved_targets.contains(target) {
                        continue; // already paid by the goal.orders-scoped branch above
                    }
                    // Single source of truth, shared with `reconcile_unit_goals`'s
                    // advance/invalidate decision, so the two can't drift on
                    // what counts as "this target is done" again -- see the
                    // Ruin-displacement regression `goal_outcome`'s doc notes.
                    match crate::ai::search::unit_goals::goal_outcome(state, *target, player) {
                        Some(true) => {
                            // Reached this ply, but the target had already
                            // dropped out of `goal.orders` (a churned ballot)
                            // — read completion straight from state instead
                            // of relying on the branch above.
                            acc.add(
                                "unit_goal_complete",
                                SHAPE_UNIT_GOAL_PER_TILE * (SHAPE_PROX_CAP as f32 + SHAPE_UNIT_GOAL_COMPLETE),
                            );
                        }
                        Some(false) => {} // invalidated -- reconcile pops it next ply
                        None => {
                            let d = cheb(*unit_idx, *target, width);
                            acc.add(
                                "unit_goal_approach",
                                SHAPE_UNIT_GOAL_PER_TILE * w_of(*target) * (SHAPE_PROX_CAP - d).max(0) as f32,
                            );
                        }
                    }
                }
                // Orders-listed targets no unit is currently assigned to
                // (more painted targets than idle units at reconcile time)
                // keep their closest-unit gradient, same as the legacy path
                // -- but ONLY over units that are themselves idle. A unit
                // with its own active goal used to count here too, so
                // stepping toward *its own* target could pay a second,
                // unrelated credit just for incidentally being the closest
                // thing to a DIFFERENT target it was never going to pursue
                // (the seed 1787500020 double-dip: it out-scored actually
                // reaching an adjacent village). Only a genuinely idle unit
                // could ever end up claiming this target, so only idle
                // units should move its gradient.
                //
                // EXP_ELO_108: reads the STORE's own record of claimed
                // targets, not `pairs` (derived from LIVE `tribe.units`).
                // A pursuer that dies mid-comparison simply drops out of
                // `tribe.units`, so `pairs`-derived `assigned` un-claims its
                // target for the SAME candidate that killed it -- letting a
                // different idle unit's sudden "closest to an unassigned
                // target" credit subsidize the pursuer's own suicide (ground
                // truth: a known-lethal Attack scored +101 over a safe
                // Step's +48, entirely from this reassignment, not the
                // attack itself). The store is frozen for the whole ply
                // (real reconciliation, including pruning dead units, only
                // runs between real plies -- see `unit_goals.rs`'s own doc
                // comment), so `active_targets()` stays identical across a
                // single candidate's pre/post comparison regardless of
                // whether that candidate kills its own pursuer.
                let assigned: rustc_hash::FxHashSet<i32> = store.active_targets();
                for target in approach_targets.iter().filter(|t| !assigned.contains(t)) {
                    let d = tribe
                        .units
                        .iter()
                        .filter(|u| store.active(u.id).is_none())
                        .map(|u| cheb(u.coords.idx, *target, width))
                        .min();
                    let Some(d) = d else { continue };
                    acc.add(
                        "unit_goal_approach_unassigned",
                        SHAPE_UNIT_GOAL_PER_TILE * w_of(*target) * (SHAPE_PROX_CAP - d).max(0) as f32,
                    );
                }
                // Same "still enemy-owned, not just still defended" fix as
                // the ephemeral-match branch above — see its comment.
                for (unit_idx, target) in &pairs {
                    let visible = state
                        .tiles
                        .get(target)
                        .map_or(false, |t| t.explorers.contains(&player));
                    let contested = visible
                        && (crate::functions::get_unit_at(state, *target)
                            .map_or(false, |u| u.owner != player)
                            || state
                                .tiles
                                .get(target)
                                .map_or(false, |t| t.owner != 0 && t.owner != player));
                    if !contested {
                        continue;
                    }
                    let second = tribe
                        .units
                        .iter()
                        .map(|u| u.coords.idx)
                        .filter(|idx| idx != unit_idx && idx != target)
                        .map(|idx| cheb(idx, *target, width))
                        .min();
                    if let Some(d) = second {
                        acc.add(
                            "unit_goal_contest_second",
                            SHAPE_GOAL_CONTEST_SECOND
                                * SHAPE_UNIT_GOAL_PER_TILE
                                * w_of(*target)
                                * (SHAPE_PROX_CAP - d).max(0) as f32,
                        );
                    }
                }
            }
        }
    }
    // Verdi (Aug 2026): opportunity cost of an extra unit. Real-trajectory
    // only (needs an accurate assigned-vs-live-target count from the store,
    // same reasoning as SHAPE_CITY_TRAIN_BLOCKED above). Fires only when
    // nothing needs defending (no Defend order — the frontline-safety
    // carve-out: a real threat zeroes this out entirely, not just discounts
    // it), a lane/save tech goal is known, and every live Expand target
    // already has a unit converging on it. Scales with total unit count, so
    // it prices ONLY the Train/Summon delta (every other move type leaves
    // unit count unchanged) and compounds if several are trained the same
    // turn.
    if let (Some(store), Some(a)) = (unit_goals, aux) {
        let has_tech_goal = a.lane_next_tech.is_some() || a.save_next_tech.is_some();
        let no_threat = !goal.orders.iter().any(|(k, _)| *k == OrderKind::Defend);
        if has_tech_goal && no_threat {
            let live_targets = goal
                .orders
                .iter()
                .filter(|(k, _)| *k == OrderKind::Expand)
                .filter(|(_, idx)| {
                    !state.tiles.get(idx).map_or(false, |t| t.owner == player)
                        || crate::functions::get_city_at(state, *idx).is_none()
                })
                .count();
            let assigned = tribe
                .units
                .iter()
                .filter(|u| store.active(u.id).map_or(false, |g| g.kind == OrderKind::Expand))
                .count();
            if assigned >= live_targets {
                acc.sub(
                    "unit_train_opportunity_cost",
                    SHAPE_GOAL_UNIT_OPPORTUNITY_COST * tribe.units.len() as f32,
                );
            }
        }
    }
    // EXP_ELO_040: Defend orders — coverage leash, not garrison pinning.
    // Pay per assigned covering unit (full in 1-turn strike reach, half in
    // the 2-turn ring) scaled by attacker pressure (EXP_ELO_103: garrison-
    // independent, so this doesn't collapse the moment prep succeeds); pay
    // tile-holding only while the garrison is load-bearing; on shortfall,
    // recall the single nearest unassigned unit with an approach gradient.
    // Threats recomputed from state each eval, so prep outcomes (a trained
    // unit, a road, a tech that extends reach) raise Φ the ply they land —
    // no discrete planner.
    let attack_targets: Vec<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Attack)
        .map(|(_, i)| *i)
        .collect();
    let city_threats_cache = if width > 0 {
        Some(match threats {
            Some(t) => crate::ai::combat::city_risks_with_threats(state, player, t),
            None => crate::ai::combat::city_risks(state, player),
        })
    } else {
        None
    };
    // EXP_ELO_114: a city can go from garrisoned to open WITHIN a turn (a
    // kill-and-advance Attack, or a Step to another target) with no Defend
    // order ever assigned -- orders commit once at turn start
    // (`oracle_macro::commit_macro_goal`), so a mid-turn vacate is
    // otherwise invisible to the Defend-order block below until the NEXT
    // commit (confirmed on both real flagged plies: neither city41/idx163
    // nor city49/idx88 had a Defend order active). This standing charge
    // prices "open + a visible enemy could walk in next turn" directly off
    // CURRENT state, independent of any order: the vacating candidate eats
    // -P*risk (Δφ from uncharged pre-state to charged post-state), a
    // same-turn refill or step-back-in earns it back, and every other
    // candidate on an already-open ply carries the same charge unchanged
    // (no distortion) -- the same regret-asymmetric shape as EXP_ELO_111's
    // step-lethal-entry gate. Skipped for any city already carrying a
    // Defend order: that city's exposure is priced by the richer
    // cover/hold/recall block below, and double-charging would starve
    // `defend_cover`'s own budget.
    if let Some(city_threats) = &city_threats_cache {
        let defended: std::collections::HashSet<i32> = goal
            .orders
            .iter()
            .filter(|(k, _)| *k == OrderKind::Defend)
            .map(|(_, idx)| *idx)
            .collect();
        for r in city_threats {
            if defended.contains(&r.city) {
                continue;
            }
            CITY_OPEN_EXPOSED_EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if r.open && r.arrives_next_turn {
                CITY_OPEN_EXPOSED_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                acc.sub("city_open_exposed", SHAPE_GOAL_CITY_OPEN_EXPOSED * r.risk);
            }
        }
    }
    if let (true, Some(city_threats)) =
        (goal.orders.iter().any(|(k, _)| *k == OrderKind::Defend), &city_threats_cache)
    {
        for (kind, idx) in &goal.orders {
            if *kind != OrderKind::Defend {
                continue;
            }
            let Some(th) = city_threats.iter().find(|t| t.city == *idx) else {
                continue; // stale order: threat cleared, nothing to pay
            };
            // EXP_ELO_103: every term in this block scales by
            // `attacker_pressure` (garrison-independent reachability,
            // EXP_ELO_095-weighted), not by the old risk-derived `urgency`.
            // `th.risk` is a P(lose) proxy that correctly (by design) drops
            // once a garrison is present -- scaling THIS block on it meant
            // the whole anticipatory reward (cover/hold/recall for every
            // unit converging on the city, plus the garrison's own hold)
            // collapsed together the instant a garrison actually landed,
            // since `need_damage` and `risk` both correctly head to zero on
            // success. `attacker_pressure` stays a pure function of visible
            // enemy reachability, unaffected by whether the threat has
            // already been resolved -- confirmed unchanged (`[(37, 1.0)]`
            // both sides) across the exact ply this was found on, unlike
            // `risk` (1.000->0.093). The live risk level still gates the
            // separate, direct `city_risk` term above (correctly, since
            // that term IS meant to reward the outcome of being safer) --
            // only the anticipatory prep/hold/recall terms move off it.
            let attacker_pressure =
                th.attackers.iter().map(|(_, w)| w).sum::<f32>().min(1.0);
            if attacker_pressure > 0.0 {
                if let Some(g) = crate::functions::get_unit_at(state, *idx) {
                    if g.owner == player {
                        acc.add(
                            "defend_garrison_hold",
                            SHAPE_GOAL_DEFEND_COVER * attacker_pressure,
                        );
                        // EXP_ELO_118: additive bonus so a cheap unit (a
                        // Warrior) filling this exact role earns MORE, not
                        // just the same, as a valuable one (a Giant) --
                        // ground-truthed: the search walked a Giant home to
                        // garrison while two idle Warriors sat closer. Never
                        // discounts the flat term above, so a lone Giant
                        // defender's hold floor doesn't weaken.
                        acc.add(
                            "defend_cheap_garrison",
                            SHAPE_GOAL_DEFEND_CHEAP * attacker_pressure * defend_cheapness(g),
                        );
                    }
                }
            }
            // EXP_ELO_106: a melee kill-and-advance vacates the garrison
            // tile the same ply it earns `defend_garrison_hold`'s credit
            // hardest, forfeiting it. A friendly unit can only be standing
            // on a frozen-listed attacker's own tile because that attacker
            // died to THIS candidate's move -- nothing else acts between
            // the frozen assessment and this eval -- so this is a safe,
            // state-pure signal to pay the same latch back on the kill tile.
            for (u, w) in &th.attackers {
                if let Some(occ) = crate::functions::get_unit_at(state, u.coords.idx) {
                    if occ.owner == player {
                        acc.add("defend_kill_advance", SHAPE_GOAL_DEFEND_COVER * w);
                        // EXP_ELO_118: mirror the cheap-unit bonus here too
                        // -- without it, a cheap unit that takes the kill
                        // forfeits the garrison_hold+cheap_garrison combo
                        // above but is refunded only the flat kill_advance
                        // rate, re-taxing exactly what EXP_ELO_106 un-taxed,
                        // specifically on the units this fix steers here.
                        acc.add(
                            "defend_cheap_kill_advance",
                            SHAPE_GOAL_DEFEND_CHEAP * w * defend_cheapness(occ),
                        );
                    }
                }
            }
            // EXP_ELO_103: cover/hold/recall all read the OPEN-framing plan
            // (need_damage computed as if no garrison existed), not
            // `defend_plan`'s real, garrison-collapsing one. The real plan's
            // need_damage is deliberately garrison-dependent for good
            // reason elsewhere (it's what makes THIS garrison's own hold
            // worth something), but reusing it here meant every OTHER
            // covering unit's credit collapsed to zero the instant any one
            // of them arrived, since the "residual need" it measures
            // legitimately hits zero on success. Cover/hold/recall want a
            // stable, garrison-independent read of "how much defense does
            // this city objectively need" so a covering unit's own value
            // doesn't evaporate the moment a teammate closes the gap.
            let plan = crate::ai::combat::defend_plan_open_framing(state, player, th, &attack_targets, pre_health);
            // EXP_ELO_096: credit scales with the unit's own contribution
            // (credit_frac), not a flat per-unit share — a strong unit and
            // a barely-relevant one no longer pay identically. The garrison
            // tile itself is excluded (EXP_ELO_103) since it's already paid
            // above, independent of whatever residual need the waterfall
            // still finds open.
            for (tile, sat, credit_frac) in &plan.assigned {
                if *tile == *idx {
                    continue;
                }
                acc.add(
                    "defend_cover",
                    SHAPE_GOAL_DEFEND_COVER * attacker_pressure * sat * credit_frac,
                );
            }
            if plan.hold_margin > 0.0 {
                acc.add(
                    "defend_hold",
                    SHAPE_GOAL_DEFEND_HOLD * attacker_pressure * plan.hold_margin,
                );
            }
            // EXP_ELO_042: recall never conscripts attack-committed units;
            // with none free, shortfall drives prep, not un-commitment.
            if plan.shortfall > 0.0 {
                let assigned: std::collections::HashSet<i32> =
                    plan.assigned.iter().map(|(t, _, _)| *t).collect();
                // EXP_ELO_118: weighted MAX over cheapness*proximity, not
                // nearest-unit-only -- inversion-safe by construction
                // (removing/retreating any single unit can only weakly
                // decrease a max, so this can't reproduce EXP_ELO_117's
                // collapse-on-success shape). Deliberately NOT cheapest-
                // first-then-nearest: that would pay MORE Φ the instant the
                // cheap unit dies and selection falls back to a pricier
                // one -- a state-dependent trap of the same shape just
                // reverted.
                let best = tribe
                    .units
                    .iter()
                    .filter(|u| {
                        // EXP_ELO_104: `plan.assigned` no longer contains the
                        // garrison's own tile (`defend_plan` excludes it from
                        // the waterfall entirely, paid instead by
                        // `defend_garrison_hold` above) -- without also
                        // excluding `*idx` here, a garrison sat at distance 0
                        // from its own city always wins this search, masking
                        // whatever real recall signal the rest of the roster
                        // has (confirmed: made `recall_skips_attack_committed_units`
                        // return the same value whether the real test unit was
                        // free or attack-committed, since the garrison always
                        // won first).
                        u.coords.idx != *idx
                            && !assigned.contains(&u.coords.idx)
                            && !crate::ai::combat::attack_committed(
                                state,
                                player,
                                u,
                                *idx,
                                &attack_targets,
                            )
                    })
                    .map(|u| {
                        let d = cheb(u.coords.idx, *idx, width);
                        let proximity = (SHAPE_PROX_CAP - d).max(0) as f32 / SHAPE_PROX_CAP as f32;
                        defend_cheapness(u) * proximity
                    })
                    .fold(0.0f32, f32::max);
                if best > 0.0 {
                    acc.add("defend_recall", SHAPE_GOAL_DEFEND_COVER * attacker_pressure * 0.5 * best);
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
                    acc.add("attack_siege_hold", SHAPE_GOAL_ATTACK_PRESS * SHAPE_GOAL_SIEGE_HOLD_MULT);
                    sieging.push(u.coords.idx);
                }
            }
        }
        // EXP_ELO_107: `attack_siege_hold` requires the target to still be
        // enemy-owned, so the exact ply that finishes the order (Capture)
        // forfeits its own credit -- the same collapse-on-success shape as
        // EXP_ELO_101/103/104/106, on the offense side's terminal action.
        // Pay the same rate once the ordered target is OURS -- ownership,
        // not occupancy, so stepping off the captured tile later can't
        // re-forfeit it (the order generator also stops re-issuing Attack
        // for cities that are now ours, since it only scans OTHER tribes'
        // `cities`, so this can't accumulate turn over turn).
        for &h in &attack_targets {
            if let Some(c) = crate::functions::get_city_at(state, h) {
                if c.owner == player {
                    acc.add("attack_capture_complete", SHAPE_GOAL_ATTACK_PRESS * SHAPE_GOAL_SIEGE_HOLD_MULT);
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
                acc.add("attack_press", SHAPE_GOAL_ATTACK_PRESS * sat);
            }
        }
    }
    // EXP_ELO_116: idle force pull toward `goal.prepare`'s target -- the
    // Attack gate almost fired but missed on assembled local value (ground
    // truth: EXP_ELO_112 claim #7, 84-vs-17 total army value, only 3 units
    // clustered, missing the gate by 2.5 of 16.5 needed). Reuses the SAME
    // deficit the gate itself computed (`oracle_macro::compute_macro_goal_
    // cached`), never re-derived, so this can never disagree with the gate
    // it's assembling toward. Zeroed by ANY live Defend order -- the same
    // frontline-safety carve-out `unit_train_opportunity_cost` uses above
    // ("a real threat zeroes this out entirely, not just discounts it").
    if width > 0 {
        if let (Some(pt), Some(store)) = (goal.prepare, unit_goals) {
            PREPARE_PULL_EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let has_defend = goal.orders.iter().any(|(k, _)| *k == OrderKind::Defend);
            if has_defend {
                PREPARE_SUPPRESSED_BY_DEFEND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                let mut eligible: Vec<(i32, i32, i32)> = tribe // (dist, tile, worth)
                    .units
                    .iter()
                    .filter(|u| {
                        !store.active(u.id).is_some_and(|g| g.kind == OrderKind::Expand)
                            && !crate::functions::get_city_at(state, u.coords.idx)
                                .is_some_and(|c| c.owner == player)
                            && !crate::ai::combat::attack_committed(
                                state,
                                player,
                                u,
                                pt.city,
                                &attack_targets,
                            )
                    })
                    .map(|u| {
                        (
                            cheb(u.coords.idx, pt.city, width),
                            u.coords.idx,
                            crate::rules::combat::unit_worth(u),
                        )
                    })
                    .collect();
                eligible.sort();
                // Count leg: the gate also requires `local.len() >= 2`, so a
                // single (however heavy) local unit's value-deficit can read
                // <= 0 while the gate still needs a second unit recruited in
                // -- the pulled count is floored at 1 regardless of the
                // worth math, capped at 3 so this can never outbid a Defend
                // or Expand commitment on sheer unit count.
                let mut covered = 0;
                let mut fired = false;
                for &(d, tile, worth) in eligible.iter().take(3) {
                    if covered >= pt.deficit && covered > 0 {
                        break;
                    }
                    covered += worth;
                    fired = true;
                    let Some(u) = tribe.units.iter().find(|u| u.coords.idx == tile) else {
                        continue;
                    };
                    let discount = threats.map_or(1.0, |th| {
                        1.0 - crate::ai::combat::lethal_threat_weight(state, u, th)
                    });
                    acc.add(
                        "prepare_pull",
                        SHAPE_GOAL_PREPARE_PER_TILE * (SHAPE_PROX_CAP - d).max(0) as f32 * discount,
                    );
                }
                if fired {
                    PREPARE_PULL_FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
            acc.add("body", SHAPE_GOAL_BODY * tribe.units.len().min(cap) as f32);
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
        acc.add("scout", SHAPE_GOAL_SCOUT * w * capped as f32);
    }
    // Lighthouse nudge (v4): each explored map corner pays once.
    if width > 0 {
        for c in crate::coords::map_corners(width) {
            if state
                .tiles
                .get(&c)
                .map_or(false, |t| t.explorers.contains(&player))
            {
                acc.add("lighthouse", SHAPE_GOAL_LIGHTHOUSE);
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
                let corners = crate::coords::map_corners(width);
                for city in explorer_cities {
                    let dark_corners: Vec<i32> = corners
                        .iter()
                        .copied()
                        .filter(|&k| {
                            cheb(city, k, width) <= EXPLORER_WALK_RANGE
                                && !state
                                    .tiles
                                    .get(&k)
                                    .map_or(false, |t| t.explorers.contains(&player))
                        })
                        .take(EXPLORER_CORNER_CAP)
                        .collect();
                    // EXP_ELO_097 (Verdi, Aug 2026): a corner an ordinary
                    // unit could walk over to on its own soon is worth far
                    // less than one genuinely stranded across water —
                    // `belief` gates this (see below) so a `None` caller
                    // keeps the legacy flat-per-corner count.
                    let lighthouse: f32 = if belief.is_some() {
                        dark_corners
                            .iter()
                            .map(|&k| {
                                SHAPE_GOAL_EXPLORER_LIGHTHOUSE
                                    * walkable_weight(state, tribe, city, k)
                            })
                            .sum()
                    } else {
                        SHAPE_GOAL_EXPLORER_LIGHTHOUSE * dark_corners.len() as f32
                    };
                    let bonus = SHAPE_GOAL_EXPLORER + lighthouse;
                    // hidden² (Jul 31): the reveal itself drains this part
                    // of the term (the potential telescopes to the
                    // horizon's h), and a linear ramp priced Explorer too
                    // high on mostly-lit maps. Quadratic keeps the dark-map
                    // edge dominant and the lit-map edge below Workshop's
                    // measured Q lead. Scoped to base+lighthouse only as of
                    // EXP_ELO_097 round 2 — see the frontier term below for
                    // why it doesn't share this decay.
                    let mut scaled = bonus * hidden_frac * hidden_frac;
                    // Frontier weighting (Verdi, Aug 2026): favors a city
                    // whose dark neighborhood leans enemy-facing over one
                    // that mostly reveals ground a walking unit could get
                    // for free. `belief` is hoisted per-ply (see
                    // `goal_potential_with_belief`'s doc); without one this
                    // is a no-op and behavior is byte-identical to legacy.
                    //
                    // EXP_ELO_097 round 2 (Verdi, Aug 2026): deliberately
                    // NOT scaled by the GLOBAL hidden_frac² above.
                    // `avg_frontier_in_reach` already decays on its own as
                    // THIS city's specific neighborhood gets revealed (it
                    // averages only over still-dark tiles there) — folding
                    // in the whole-map fraction too double-counts "is there
                    // still something to find" and crushed a strong, real
                    // signal: city 49's avg_frontier_in_reach measured 4.81
                    // (genuinely enemy-facing) at the seed0 turn-4 ply, but
                    // hidden_frac was already down to ~0.31 map-wide, so
                    // the old shared scaling cut a 300+ point signal to 47
                    // — Workshop won a pick that should have gone the other
                    // way (reward_choice_probe2.rs, verified against the
                    // real accumulator, not hand math).
                    if let Some(belief) = belief {
                        let avg = avg_frontier_in_reach(state, belief, city, EXPLORER_WALK_RANGE);
                        scaled += SHAPE_GOAL_EXPLORER_FRONTIER * (avg - FRONTIER_W_FOG).max(0.0);
                    }
                    // EXP_ELO_097: the capital is discounted every reward,
                    // not just its first — "Capital almost always
                    // workshop" (Verdi). Checked on the tile's own
                    // `capital_of`, not city count (see the constant's doc
                    // for why the old `cities.len() <= 1` proxy broke).
                    // Applied to the frontier component too — "almost
                    // always", not a hard exemption for the capital.
                    let is_capital = state
                        .tiles
                        .get(&city)
                        .map_or(false, |t| t.capital_of == player);
                    if is_capital {
                        scaled *= SHAPE_GOAL_EXPLORER_CAPITAL_SCALE;
                    }
                    acc.add("explorer", scaled);
                    // EXP_ELO_097 round 2: the "lighthouse" nudge above
                    // (v4, a few dozen lines up) pays SHAPE_GOAL_LIGHTHOUSE
                    // per revealed corner regardless of cause — including a
                    // corner this city's OWN Explorer pick just revealed.
                    // That's `moves/reward.rs`'s real engine effect
                    // (`predict_explorer` + `discover_tiles`, executed
                    // immediately, not a search-side approximation), so
                    // it's real, certain value the capital discount above
                    // never touches, since it lives in a different Φ term
                    // entirely. Left uncorrected, a capital whose Explorer
                    // happens to path through a corner silently un-does
                    // "almost always workshop" the instant it's picked —
                    // confirmed against the real accumulator at the seed0
                    // turn-3 capital ply (reward_choice_probe2.rs): a
                    // single corner hit (+120, undiscounted) was enough by
                    // itself to flip the choice back to Explorer. Correct
                    // it here rather than in the generic term, which must
                    // stay agnostic to cause for the (correct, undiscounted)
                    // non-capital and ordinary-exploration cases.
                    //
                    // This is a STANDING correction, not a one-shot: it
                    // reads as "corner is within this city's explorer reach
                    // AND already revealed", true both before and after any
                    // LATER candidate move once the corner has been
                    // revealed — so it nets to zero delta on every ply
                    // except the exact pick ply itself, where the `explorer`
                    // term's own gate (`c.rewards.contains(Explorer)`)
                    // flips this whole per-city block from absent (pre) to
                    // present (post), matching the generic lighthouse
                    // term's own pre/post asymmetry exactly.
                    //
                    // Not distance-gated: `predict_explorer`'s real 12-step
                    // walk can reach corners well beyond `EXPLORER_WALK_RANGE`
                    // (that constant is calibrated for the CHANCE-based
                    // lighthouse term above, a different question) — a first
                    // attempt gated on it missed the seed0 capital's actual
                    // revealed corner (distance 7, "in reach" only to 5) and
                    // measured zero correction (reward_choice_probe2.rs).
                    // Re-deriving via a fresh `predict_explorer` call doesn't
                    // work either: that call reads CURRENT fog, and by the
                    // post-pick state the corner is already lit, so a fresh
                    // simulated walk reroutes toward remaining darkness and
                    // never claims the very corner it just revealed (also
                    // confirmed empirically before this cut). Any corner
                    // revealed while this capital holds Explorer is
                    // credited to it. Two known, accepted imprecisions: a
                    // later, wholly unrelated corner reveal (an ordinary
                    // unit wandering there turns 20+) gets slightly
                    // under-priced too, since a single state snapshot can't
                    // distinguish cause; and a second Explorer-holding city
                    // at the same corner is a rare enough overlap to accept
                    // for now. Both are minor next to the bug this fixes
                    // (an undiscounted capital pick) and cheaper than
                    // diffing state transitions inside a pure potential
                    // function.
                    if is_capital {
                        let lit_corners = crate::coords::map_corners(width)
                            .into_iter()
                            .filter(|&c| {
                                state
                                    .tiles
                                    .get(&c)
                                    .map_or(false, |t| t.explorers.contains(&player))
                            })
                            .count();
                        if lit_corners > 0 {
                            let full = SHAPE_GOAL_LIGHTHOUSE * lit_corners as f32;
                            acc.add(
                                "explorer_capital_lighthouse_correction",
                                full * (SHAPE_GOAL_EXPLORER_CAPITAL_SCALE - 1.0),
                            );
                        }
                    }
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
        acc.add("yield_pop", SHAPE_GOAL_YIELD_ADJ * setting.reward_pop.max(0) as f32 * weight);
        acc.add("yield_stars", SHAPE_GOAL_YIELD_ADJ_STARS * setting.reward_stars.max(0) as f32 * weight);
    }
    // Standing-forest option value (v5): clearing pays only when the
    // follow-up (build / level-up funding) outweighs the lost option.
    let own_forests = state
        .tiles
        .values()
        .filter(|t| t.owner == player && t.terrain_type == crate::types::TerrainType::Forest)
        .count();
    acc.add("forest_standing", SHAPE_GOAL_FOREST_STANDING * own_forests as f32);
    if let Some(aux) = aux {
        let owned = aux
            .recommended_techs
            .iter()
            .filter(|t| {
                crate::settings::technology::is_tech_unlocked(&tribe.tech_vanilla, **t)
            })
            .count();
        acc.add("tech_fit", SHAPE_GOAL_TECH_FIT * owned as f32);
        if aux.rider_push {
            let riders = tribe
                .units
                .iter()
                .filter(|u| u.unit_type == crate::types::UnitType::Rider)
                .count();
            acc.add("rider", SHAPE_GOAL_RIDER * riders as f32);
        }
        if !aux.preferred_units.is_empty() {
            let preferred = tribe
                .units
                .iter()
                .filter(|u| aux.preferred_units.contains(&u.unit_type))
                .map(crate::rules::combat::unit_worth)
                .sum::<i32>();
            acc.add("lane_preferred", SHAPE_GOAL_LANE_PER_COST * preferred as f32);
        }
    }
    acc.phi
}

#[cfg(test)]
#[path = "goal_potential_tests.rs"]
mod tests;
