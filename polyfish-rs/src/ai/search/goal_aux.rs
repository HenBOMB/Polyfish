//! Per-ply auxiliary goal context (Aug 2026 taxonomy reorg: split out of
//! oracle_macro.rs to keep every file in `ai::` under ~1000 lines): `GoalAux`
//! plus the root-only move gates that consume it (tech caps, ability gate,
//! capture-first, star gate) and `scripted_goal_aux`/`market_ready`, which
//! build it. Re-exported through `oracle_macro` so existing
//! `crate::ai::oracle_macro::X` call sites keep resolving.

use crate::ai::economy::{recommended_techs, SaveLane};
use crate::ai::movement::{connect_remaining, rider_turns_saved, RIDER_PUSH_MIN_TURNS_SAVED};
use crate::ai::oracle_macro::{
    stance_strength, MacroGoal, OrderKind, Stance, COMMIT_CITY_TARGET, TECH_CAP_PER_GAME,
    TIER3_CAP_PER_GAME,
};
use crate::ai::search::archetype::{lane_techs, Archetype, ArchetypeState, Overlays};
use crate::moves::Move;
use crate::states::GameState;
use crate::states::PlayerId;
use crate::types::{AbilityType, MoveType, StructureType, TechnologyType, TerrainType};


/// Per-ply auxiliary goal context (v2.3), set on the agent alongside the
/// `MacroGoal` but NOT painted into features: environment-fit tech bias and
/// the whole-game purchase counters. Cached tree edges may carry rewards
/// from a slightly older aux — acceptable staleness, like tree reuse itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GoalAux {
    /// EXP_ELO_050: T2's risk assessment, handed DOWN to T3 — the threat
    /// facts (who can strike, who can walk in, is a siege breakable, what the
    /// city is worth) for every city under threat. The expensive reachability
    /// search happens here, once; T3 re-resolves only `residual_risk` against
    /// live occupancy, which is what gives its defensive plies a gradient.
    pub city_risk: Vec<crate::ai::combat::CityRisk>,
    /// EXP_ELO_051: the batch this seat is banking for, carried down so the
    /// star gate can ask "does this purchase advance the plan" instead of
    /// only "is this tech the right CLASS for the stance". Class-only gating
    /// is what let Organization through on a SAVE turn in seed 1786807403.
    pub save_lane: Option<SaveLane>,
    /// The one tech that batch is actually waiting on — the deepest unowned
    /// step of its prerequisite chain. While banking, this is the only
    /// research that is not a delay.
    pub save_next_tech: Option<TechnologyType>,
    /// EXP_ELO_052: the committed Tier-1 lane's next unowned tech. This
    /// outranks the hub batch: a lane is a plan for the whole game, the batch
    /// is a plan for the next few turns.
    pub lane_next_tech: Option<TechnologyType>,
    /// EXP_ELO_052: (city, road tiles still needed) for every city not yet
    /// linked to the capital. T2 runs the path search once; T3 prices each
    /// road tile by the progress it makes.
    pub connect_remaining: Vec<(i32, i32)>,
    /// Stance-intensity (Verdi, Aug 14): measured ARM pressure 0..1 from
    /// `stance_strength` — threat-vs-coverage truth, NOT the binary stance.
    /// The eco-tech mask fires only when this is near-certain (>= 0.98);
    /// below that ARM steers pricing, never masks.
    pub arm_strength: f32,
    /// Environment-recommended techs (owned ones pay the in-tree fit bonus).
    pub recommended_techs: Vec<TechnologyType>,
    /// Path-aware: a Rider reaches some EXPAND target at least
    /// `RIDER_PUSH_MIN_TURNS_SAVED` turns sooner than a walker would.
    pub rider_push: bool,
    /// Research moves this seat has executed this game.
    pub techs_bought: u32,
    /// …of which tier-3.
    pub tier3_bought: u32,
    /// v3 archetype: unit types the active doctrine + overlays prefer —
    /// each living one banks `SHAPE_GOAL_ARCHETYPE_UNIT` in the potential.
    pub preferred_units: Vec<crate::types::UnitType>,
    /// v3 reactive overlays; `knight_commit` also opens the
    /// FreeSpirit→Chivalry purchase lane (see `passes_tech_caps`).
    pub overlays: Overlays,
    /// v6 income lane: third city up + a hub structure standing → the
    /// Riding→Roads→Trade lane is recommended and Trade is exempt from the
    /// tier-3 cap (Market is the best ★/SPT purchase in the game).
    pub market_push: bool,
    /// v7: this seat already owns an economic tier-3 (by purchase OR ruin
    /// grant). Until it does, combat tier-3s are blocked — see
    /// `passes_tech_caps`. Ownership rather than purchases on purpose: a free
    /// economy tier-3 out of a ruin has already paid the ordering cost.
    pub eco_tier3_owned: bool,
    /// The map holds no water at all (Drylands), so the whole water tech lane
    /// buys nothing — see `passes_tech_caps`. Read from the true tile set, not
    /// the player's view: map type is public information at game start, unlike
    /// what happens to sit under the fog.
    pub water_dead: bool,
}

/// Build the per-ply `GoalAux` for the scripted driver: environment fit,
/// the path-aware rider push (a Rider beats a walker to some EXPAND target
/// → Riding joins the recommendations while unowned), and the caller-tracked
/// purchase counters.
pub fn scripted_goal_aux(
    state: &GameState,
    player: PlayerId,
    goal: &MacroGoal,
    techs_bought: u32,
    tier3_bought: u32,
    arch: Option<&ArchetypeState>,
) -> GoalAux {
    let mut recommended = recommended_techs(state, player);
    let expand_targets: Vec<i32> = goal
        .orders
        .iter()
        .filter(|(k, _)| *k == OrderKind::Expand)
        .map(|(_, idx)| *idx)
        .collect();
    let rider_push = !expand_targets.is_empty()
        && rider_turns_saved(state, player, &expand_targets) >= RIDER_PUSH_MIN_TURNS_SAVED;
    if rider_push {
        if let Some(tribe) = state.tribes.get(&player) {
            let riding = crate::settings::technology::resolve_tech_for_tribe(
                crate::types::TechnologyType::Riding,
                tribe.tribe_type,
            );
            if riding == crate::types::TechnologyType::Riding
                && !crate::settings::technology::is_tech_unlocked(&tribe.tech_vanilla, riding)
                && !recommended.contains(&riding)
            {
                recommended.insert(0, riding);
            }
        }
    }
    // v3 archetype expression: doctrine + overlay tech lanes join the
    // recommendations (next unowned tech per lane), preferred unit classes
    // feed the in-tree unit bonus. FreeSpirit/Chivalry appear ONLY under a
    // knight commitment — the stepping-stone rule (Verdi, Jul 30).
    let mut preferred_units: Vec<crate::types::UnitType> = Vec::new();
    let mut overlays = Overlays::default();
    if let Some(arch) = arch {
        use crate::types::{TechnologyType as Tech, UnitType as U};
        overlays = arch.overlays;
        let owned = |t: Tech| {
            state
                .tribes
                .get(&player)
                .map_or(false, |tr| crate::settings::technology::is_tech_unlocked(&tr.tech_vanilla, t))
        };
        let push_lane = |lane: &[Tech], recs: &mut Vec<Tech>| {
            if let Some(t) = lane.iter().find(|t| !owned(**t)) {
                if !recs.contains(t) {
                    recs.push(*t);
                }
            }
        };
        if let Some(a) = arch.archetype {
            push_lane(lane_techs(a), &mut recommended);
            match a {
                Archetype::RiderRoads => preferred_units.push(U::Rider),
                Archetype::ArcherLine => preferred_units.push(U::Archer),
                Archetype::ForgeGiants => {
                    preferred_units.push(U::Swordsman);
                    preferred_units.push(U::Giant);
                }
            }
        }
        if overlays.defender_screen {
            push_lane(&[Tech::Strategy], &mut recommended);
            preferred_units.push(U::Defender);
        }
        if overlays.catapult_counter {
            push_lane(&[Tech::Forestry, Tech::Mathematics], &mut recommended);
            preferred_units.push(U::Catapult);
        }
        if overlays.knight_commit {
            push_lane(&[Tech::Riding, Tech::FreeSpirit, Tech::Chivalry], &mut recommended);
            preferred_units.push(U::Knight);
        }
    }
    // v6 income lane — archetype-independent: with the third city up and a
    // hub structure standing, Riding→Roads→Trade opens the Market.
    let market_push = market_ready(state, player);
    if market_push {
        use crate::types::TechnologyType as Tech;
        let owned = |t: Tech| {
            state.tribes.get(&player).map_or(false, |tr| {
                crate::settings::technology::is_tech_unlocked(&tr.tech_vanilla, t)
            })
        };
        if let Some(t) = [Tech::Riding, Tech::Roads, Tech::Trade]
            .iter()
            .find(|t| !owned(**t))
        {
            if !recommended.contains(t) {
                recommended.push(*t);
            }
        }
    }
    // EXP_ELO_051: the batch we are banking for is by definition on-plan, so
    // its own next step joins the whitelist — otherwise `passes_tech_caps`
    // would gate the very purchase the savings exist to make.
    // The committed lane's own next step — the chain walked to its deepest
    // unowned tech, so a two-step lane buys Riding before Roads.
    // Only the lane's OPENING tech is privileged. Iteration 3 privileged the
    // whole chain and Imperius spent its early stars walking Riding→Roads, a
    // chain whose payoff (a connection: 9 stars of road for +2 pop) is
    // dominated by a 5-star Windmill for the same +2 — so it lost its hub and
    // its giants (hubs@t15 0.94 → 0.44, giants 1.12 → 0.78). The opening tech
    // is the commitment; the rest of the chain competes on price like
    // everything else.
    let lane_next_tech = arch.and_then(|a| a.archetype).and_then(|a| {
        let tribe = state.tribes.get(&player)?;
        let first = *lane_techs(a).first()?;
        (!crate::settings::technology::has_technology(&tribe.tech_vanilla, first))
            .then_some(first)
    });
    let mut save_next_tech = None;
    if let Some(lane) = goal.save_target.as_ref() {
        if let Some(tribe) = state.tribes.get(&player) {
            let mut cur = Some(lane.tech);
            let mut guard = 0;
            while let Some(t) = cur {
                guard += 1;
                if guard > 16
                    || crate::settings::technology::has_technology(&tribe.tech_vanilla, t)
                {
                    break;
                }
                save_next_tech = Some(t);
                cur = crate::settings::technology::get_technology_setting(t).requires;
            }
            if let Some(t) = save_next_tech {
                if !recommended.contains(&t) {
                    recommended.push(t);
                }
            }
        }
    }
    let water_dead = !state
        .tiles
        .values()
        .any(|t| matches!(t.terrain_type, TerrainType::Water | TerrainType::Ocean));
    // Aquatism's WaterTemple yields population, so it counts as an economic
    // tier-3 by the table — but on a dry map it can never be built, and letting
    // it satisfy the economy-first rule would unblock the combat lane for free.
    let eco_tier3_owned = state.tribes.get(&player).map_or(false, |t| {
        t.tech_vanilla.iter().any(|tech| {
            tech.discovered
                && crate::settings::technology::is_eco_tier3(tech.tech_type)
                && !(water_dead && crate::settings::technology::is_water_tech(tech.tech_type))
        })
    });
    GoalAux {
        // T2 assesses; T3 prices its response against it.
        city_risk: crate::ai::combat::city_risks(state, player),
        save_lane: goal.save_target.clone(),
        save_next_tech,
        lane_next_tech,
        connect_remaining: connect_remaining(state, player),
        arm_strength: stance_strength(state, player).arm,
        recommended_techs: recommended,
        rider_push,
        techs_bought,
        tier3_bought,
        preferred_units,
        overlays,
        market_push,
        eco_tier3_owned,
        water_dead,
    }
}

/// v6: the market limb is worth opening — third city up and at least one
/// hub structure (anything in Market's adjacent_types: Sawmill / Windmill /
/// Forge) standing in own territory. Derived from settings tables.
pub fn market_ready(state: &GameState, player: PlayerId) -> bool {
    let Some(tribe) = state.tribes.get(&player) else {
        return false;
    };
    if tribe.cities.len() < COMMIT_CITY_TARGET {
        return false;
    }
    let hubs = &crate::settings::structures::get_structure_setting(StructureType::Market)
        .adjacent_types;
    state.structures.iter().any(|(idx, s)| {
        s.as_ref().map_or(false, |s| hubs.contains(&s.structure_type))
            && state.tiles.get(idx).map_or(false, |t| t.owner == player)
    })
}

/// Root-only whole-game purchase caps — applied whenever a `GoalAux` is set,
/// independent of the stance gate's window. Non-Research moves always pass.
pub fn passes_tech_caps(m: &dyn Move, aux: &GoalAux) -> bool {
    if m.move_type() != MoveType::Research {
        return true;
    }
    if aux.techs_bought >= TECH_CAP_PER_GAME {
        return false;
    }
    // EXP_ELO_051 lane discipline. This lives here, not behind `star_gate`,
    // because that gate switches OFF under GROW once the third city is up —
    // which is exactly when the fixtures bought Organization. Tech discipline
    // is not conditional on the spending stance.
    //
    // Verdi: "the terrain is telling us we clearly just need to get metal and
    // forge thats it." While a batch is live, only purchases that advance it
    // pass; before one is affordable, `recommended_techs` — the committed
    // lane's next step PLUS whatever the overlays demand (defender screen,
    // catapult counter, knight commit, the market lane) — is the whitelist.
    // Overlays are what keep this a judgment call rather than a freeze: a
    // real threat still buys its answer.
    if let Ok(tech) = m.tech_type() {
        // A committed knight lane is a whole chain, not just its next step —
        // the exemption `passes_star_gate` already grants it.
        let knight_lane = aux.overlays.knight_commit
            && matches!(
                tech,
                TechnologyType::Riding | TechnologyType::FreeSpirit | TechnologyType::Chivalry
            );
        if !knight_lane {
            // EXP_ELO_052: the COMMITTED LANE'S OWN TECHS COME FIRST. Before
            // this, a RiderRoads seat could have Riding gated out by a
            // Windmill batch — the lane-discipline machinery forbidding the
            // lane's opening move. Measured: `Research Riding` reached the
            // ballot in 8 of 764 plies over t0-t5 on Imperius, and the first
            // Rider landed t11.6 against a target of t3.
            // The lane's own next tech is always permitted — never gated by
            // a hub batch, which is the bug that kept Riding off a
            // RiderRoads ballot. It is NOT exclusive: iteration 2 made it so
            // and starved the economy (Imperius hubs@t15 0.94 -> 0.09,
            // giants 1.12 -> 0.38), because no eco tech could be bought while
            // the lane was unfinished.
            if aux.lane_next_tech == Some(tech) {
                // allowed
            } else if let Some(next) = aux.save_next_tech {
                // Lane complete: bank for the hub, and only its next step is
                // not a delay. `save_batch_plan` self-terminates once the
                // batch is affordable, so this never freezes research.
                if tech != next {
                    return false;
                }
            } else if !aux.recommended_techs.is_empty()
                && !aux.recommended_techs.contains(&tech)
            {
                return false;
            }
        }
    }
    // v7.1 (Verdi, Aug 2026): on a map with no water the whole naval lane —
    // Fishing/Sailing/Ramming/Navigation/Aquatism — unlocks nothing buildable
    // and nothing reachable behind it. A mask, not a price: this is the
    // never-do case masks exist for.
    if aux.water_dead {
        if let Ok(tech) = m.tech_type() {
            if crate::settings::technology::is_water_dead_end(tech) {
                return false;
            }
        }
    }
    // v3 stepping-stone rule: FreeSpirit has no standalone value — it (and
    // Chivalry behind it) is buyable only under an active knight commitment.
    if !aux.overlays.knight_commit {
        if let Ok(t) = m.tech_type() {
            if t == TechnologyType::FreeSpirit || t == TechnologyType::Chivalry {
                return false;
            }
        }
    }
    // v7 economy-first ordering: a combat tier-3 (Chivalry/Navigation/
    // Diplomacy — the ones unlocking no yielding structure) waits until an
    // economic tier-3 is owned. Real games almost never take knights before
    // the level-3 pop buildings, because those are what lead to giants; the
    // exception (a lucky free-spirit ruin) is covered by reading OWNERSHIP,
    // so a free economy tier-3 unblocks the lane immediately.
    if let Ok(tech) = m.tech_type() {
        let setting = crate::settings::technology::get_technology_setting(tech);
        if setting.tier == Some(3)
            && !crate::settings::technology::is_eco_tier3(tech)
            && !aux.eco_tier3_owned
        {
            return false;
        }
    }
    if aux.tier3_bought >= TIER3_CAP_PER_GAME {
        if let Ok(tech) = m.tech_type() {
            if crate::settings::technology::get_technology_setting(tech).tier == Some(3) {
                // v6: per-overlay tier-3 exemptions, not a global cap raise —
                // an ACTIVE knight commitment exempts Chivalry, an active
                // market push exempts Trade (both still count toward the
                // whole-game cap). Otherwise the terrain lane's tier-3,
                // bought by ~t12, permanently locked these lanes out.
                let exempt = (aux.overlays.knight_commit && tech == TechnologyType::Chivalry)
                    || (aux.market_push && tech == TechnologyType::Trade);
                if !exempt {
                    return false;
                }
            }
        }
    }
    true
}

/// Root-only ability gate — applied whenever a `GoalAux` is set, like the
/// tech caps. Destroy (demolish own structure) is masked out entirely: the
/// Jul 30 2026 gauge audit measured ~9 destroys/game of pure churn, and the
/// rare strategic rebuild is deferred until the net owns the basics.
///
/// v8: Clear/Burn Forest on a tile carrying a resource joins it. The ability is
/// free and pays a star, but `consume_resource` DELETES the Game sitting there
/// — trading a harvestable pop source for one star is dominated at every star
/// price, so it is a mask rather than a price. Clearing bare forest stays
/// legal and is priced by `SHAPE_GOAL_FOREST_STANDING`.
pub fn passes_ability_gate(state: &GameState, m: &dyn Move) -> bool {
    if m.move_type() != MoveType::Ability {
        return true;
    }
    let Ok(ability) = m.ability_type() else {
        return true;
    };
    if ability == AbilityType::Destroy {
        return false;
    }
    if matches!(ability, AbilityType::ClearForest | AbilityType::BurnForest) {
        if let Ok(idx) = m.target_idx() {
            if matches!(state.resources.get(&(idx as i32)), Some(Some(_))) {
                return false;
            }
        }
    }
    true
}

/// Root-only capture-first gate (v6): a unit standing on a capturable
/// village/ruin never attacks — capture is strictly better (city defense
/// bonus, unit production; attacking forfeits the capture turn and eats
/// free retaliation). Applies even when Capture isn't legal this ply (the
/// unit stepped on this turn): it idles and captures next turn.
pub fn passes_capture_first(state: &GameState, m: &dyn Move) -> bool {
    if m.move_type() != MoveType::Attack {
        return true;
    }
    let Ok(src) = m.source_idx() else {
        return true;
    };
    let src = src as i32;
    let structure = state
        .structures
        .get(&src)
        .and_then(|s| s.as_ref())
        .map(|s| s.structure_type);
    match structure {
        Some(StructureType::Ruin) => false,
        Some(StructureType::Village) => {
            let owner = state.tiles.get(&src).map_or(0, |t| t.owner);
            owner == state.settings.current_player_turn_id
        }
        _ => true,
    }
}


/// Root-only research gate (v9, dual-exempt). Every non-Research move passes.
/// A gated Research move is dropped outright — the old "unless you keep
/// `STAR_GATE_RESERVE` stars" escape is gone. It read as a soft price but the
/// measured policy is hand-to-mouth (median 1 star at EndTurn), so it opened on
/// 0.5% of gated GROW plies: a constant pretending to be a decision.
///
/// A stance gates only the tech that is PURELY the other class. Dual-class tech
/// serves both and is never dropped — Smithery fields a Swordsman AND opens the
/// Forge, so gating it under GROW while ARM lets it through made the Forge lane
/// hostage to a stance that is ARM 85% of the time after turn 10.
/// - `Some(Grow)` / `Some(Save)`: pure-combat tech (no eco effect). Both are
///   economy stances in `reward::goal_potential`; SAVE additionally cannot gate
///   its own batch, whose cost includes an unowned tech chain.
/// - `Some(Arm)`: pure-eco tech (fields no combat unit).
/// - `Some(Unlock)`: nothing gated (no unlock policy yet).
/// - `None`: every tech (the EXP_ELO_026 legacy gate, kept reproducible for
///   arena `--macro-star-gate`; now a hard drop rather than a reserve test).
pub fn passes_star_gate(
    _state: &GameState,
    m: &dyn Move,
    stance: Option<Stance>,
    aux: Option<&GoalAux>,
) -> bool {
    if m.move_type() != MoveType::Research {
        return true;
    }
    let Ok(tech) = m.tech_type() else {
        return true;
    };
    // v6: an active knight commitment makes its lane stance-coherent — the
    // stance-class gating no longer applies (GROW gated Chivalry as combat
    // tech while ARM gated FreeSpirit as eco tech, blocking the lane from
    // both sides). passes_tech_caps already restricts the lane to commits.
    if (tech == TechnologyType::FreeSpirit || tech == TechnologyType::Chivalry)
        && aux.map_or(false, |a| a.overlays.knight_commit)
    {
        return true;
    }
    // EXP_ELO_051: a live savings plan outranks the stance classes. While we
    // are banking for a named lane, a tech that does not advance it is the
    // purchase that delays it — which is exactly the "random tech" Verdi
    // flagged (Organization on t4-t5 of every fixture seed). Bounded by the
    // plan's own lifetime: `save_batch_plan` self-terminates once the batch
    // is affordable, so this never becomes an open-ended research freeze.
    let effects = crate::settings::technology::get_tech_effects(tech);
    let arms = !effects.combat_units.is_empty();
    let grows = crate::settings::technology::is_eco_tech(tech);
    let gated = match stance {
        None => true,
        Some(Stance::Grow) | Some(Stance::Save) => arms && !grows,
        // Verdi (Aug 14): masking is legitimate only at near-certain need —
        // below that, a covered skirmish must not lock the eco lanes.
        Some(Stance::Arm) => grows && !arms && aux.map_or(true, |a| a.arm_strength >= 0.98),
        Some(Stance::Unlock) => false,
    };
    !gated
}

#[cfg(test)]
#[path = "goal_aux_tests.rs"]
mod tests;
