//! EXP_ELO_040: city threat model + defense coverage plans for the macro
//! executor. Pure functions of state, FOW-honest (visible enemies only),
//! built on the real engine math: `compute_reachable_tiles` for reach
//! (roads/terrain count) and `calculate_combat`/`get_defense_bonus` on
//! coord-swapped unit clones for hypothetical placements.

use crate::functions::{
    get_chebyshev_distance, get_defense_bonus, get_true_unit_at, get_unit_attack,
    get_unit_defense, get_unit_max_health, get_unit_movement, has_skill,
};
use crate::settings::units::get_unit_setting;
use crate::states::{GameState, UnitState};
use crate::types::SkillType;
use crate::PlayerId;

/// Strike fraction of garrison HP that already counts as at-risk: one hit
/// leaving the garrison below a second hit is a threat, not a nuisance.
const RISK_MARGIN: f32 = 0.8;
/// Extra Chebyshev pad past 2×movement for coverage candidates.
const RING2_PAD: i32 = 2;
/// No dogpile: a plan never assigns more units than this.
const MAX_ASSIGN: usize = 4;

#[derive(Debug, Clone)]
pub struct CityThreat {
    pub city: i32,
    /// Worst-case total damage visible enemies can deliver to the current
    /// garrison next enemy turn (0.0 when the city is unguarded).
    pub strike: f32,
    /// Damage required to kill the strongest deliverable attacker.
    pub need_damage: f32,
    /// Tile indices of contributing enemy units.
    pub attackers: Vec<i32>,
    /// An enemy unit is standing on the city right now.
    pub sieged: bool,
    /// City is unguarded and an enemy can end its move on it.
    pub reachable_unguarded: bool,
    pub at_risk: bool,
}

#[derive(Debug, Clone)]
pub struct DefendPlan {
    pub city: i32,
    /// (unit tile, satisfaction): 1.0 = can strike an attacker on the city
    /// next turn, 0.5 = inside the two-turn response ring.
    pub assigned: Vec<(i32, f32)>,
    /// Unmet kill damage after assignment — drives recall/prep gradients.
    pub shortfall: f32,
    /// The garrison is load-bearing: without it the assigned cover cannot
    /// meet `need_damage`. Only then does holding the tile get paid.
    pub hold_needed: bool,
}

/// Clone with fresh action flags: threat and coverage reason about NEXT
/// turn, when moved/attacked reset.
fn probe(unit: &UnitState) -> UnitState {
    let mut u = unit.clone();
    u.moved = false;
    u.attacked = false;
    u.attacks_performed = 0;
    u
}

/// Real engine damage for a hypothetical placement: `defender` cloned onto
/// `defender_tile` so `get_defense_bonus` reads the true tile/city rules.
fn hypo_damage(state: &GameState, attacker: &UnitState, defender: &UnitState, defender_tile: i32) -> f32 {
    let mut d = defender.clone();
    d.coords = crate::coords::Coords::from_index(defender_tile, state.settings.size);
    let bonus = get_defense_bonus(state, &d);
    let r = crate::actions::units::calculate_combat(
        get_unit_attack(state, attacker),
        attacker.health,
        get_unit_max_health(attacker),
        get_unit_defense(&d),
        d.health,
        get_unit_max_health(&d),
        bonus,
    );
    r.attack_damage
}

/// Can `unit` (fresh flags) attack a unit standing on `target_tile` within
/// one turn? Static in-range attack needs no Dash; move-and-attack does.
/// Hot path: inside `movement + range` plain distance decides (small
/// overestimate through blockers, acceptable); the exact road-aware search
/// only runs in the band beyond it, where roads are what make it true.
fn can_attack_tile(state: &GameState, unit: &UnitState, target_tile: i32) -> bool {
    let size = state.settings.size;
    let range = get_unit_setting(unit.unit_type).range;
    let d = get_chebyshev_distance(unit.coords.idx, target_tile, size);
    if d <= range {
        return true;
    }
    if !has_skill(unit, SkillType::Dash) {
        return false;
    }
    let m = get_unit_movement(state, unit);
    if d <= m + range {
        return true;
    }
    if d > 2 * m + range {
        return false;
    }
    crate::moves::reach_search(
        state,
        unit,
        Some(&|t: i32| {
            t != target_tile
                && get_chebyshev_distance(t, target_tile, size) <= range
                && (t == unit.coords.idx || get_true_unit_at(state, t).is_none())
        }),
    )
    .1
}

/// EXP_ELO_050 risk dials. `risk` is a P(lose this city) proxy, multiplied
/// by the city's worth to give a score-equivalent expected loss. The ordering
/// is the doctrine: an unbreakable siege is the disaster, a garrison that
/// holds is nearly free, and PREVENTION is what separates them — measured,
/// not assumed (EXP_ELO_049: a parked Giant is cleared 6% of the time, and
/// having a unit able to strike the besieger does not predict the outcome).
pub(crate) const RISK_LOST: f32 = 1.0;
pub(crate) const RISK_BREAKABLE: f32 = 0.45;
pub(crate) const RISK_GARRISON_FALLS: f32 = 0.35;
pub(crate) const RISK_GARRISON_HOLDS: f32 = 0.05;
/// Worth of a city in the score-equivalent units `goal_potential` uses:
/// base plus per-level, so a developed capital outprices a frontier village.
const CITY_WORTH_BASE: f32 = 12.0;
const CITY_WORTH_PER_LEVEL: f32 = 6.0;

/// What it would take the enemy to OWN this city, and whether I could undo
/// it. T2 assesses this once; T3 re-resolves only the cheap, live part of it
/// (`residual_risk`) so its own plies actually move the number.
#[derive(Debug, Clone, PartialEq)]
pub struct CityRisk {
    pub city: i32,
    /// An enemy is standing on it right now.
    pub sieged: bool,
    /// Nothing of mine is standing on it.
    pub open: bool,
    /// A visible enemy can end its move on the tile next turn.
    pub arrives_next_turn: bool,
    /// If they park there, my units could remove them within one turn.
    pub breakable: bool,
    /// Tile indices of visible enemies that can strike the city tile — the
    /// reachability search is T2's, resolving damage against whoever ends up
    /// standing there is T3's.
    pub attackers: Vec<i32>,
    /// Tile indices of visible enemies that can END their move on the tile.
    pub enterers: Vec<i32>,
    /// P(lose the city) proxy in [0,1], as assessed at turn start.
    pub risk: f32,
    /// Score-equivalent worth of the city.
    pub worth: f32,
}

impl CityRisk {
    pub fn expected_loss(&self) -> f32 {
        self.risk * self.worth
    }
    /// Worth a Defend order from T2. A garrison that merely *holds* is already
    /// doing its job — naming it would pin the stance to ARM for the rest of
    /// the game; its vacating is still priced through `residual_risk`.
    pub fn needs_order(&self) -> bool {
        self.risk >= RISK_GARRISON_FALLS
    }
}

/// Can `unit` (fresh flags) END its move on `target_tile`? Same distance
/// banding as `can_attack_tile`: plain distance inside one move, the exact
/// road-aware search only in the band beyond it.
fn can_reach_tile(state: &GameState, unit: &UnitState, target_tile: i32) -> bool {
    let size = state.settings.size;
    let m = get_unit_movement(state, unit);
    let d = get_chebyshev_distance(unit.coords.idx, target_tile, size);
    if d == 0 {
        return true;
    }
    if d <= m {
        return true;
    }
    if d > 2 * m {
        return false;
    }
    crate::moves::reach_search(state, unit, Some(&|t: i32| t == target_tile)).1
}

/// Per-city expected loss: who can reach the tile, whether a siege there
/// would be breakable, and what that costs me. FOW-honest — only visible
/// enemy units, read with their real movement (roads and tech included).
pub fn city_risks(state: &GameState, player: PlayerId) -> Vec<CityRisk> {
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let enemies: Vec<&UnitState> = state
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
        .collect();
    if enemies.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for city in &tribe.cities {
        let idx = city.idx;
        let occupant = get_true_unit_at(state, idx);
        let sieged = occupant.as_ref().map_or(false, |u| u.owner != player);
        let garrison = occupant.as_ref().filter(|u| u.owner == player);
        let open = occupant.is_none();

        let enterers: Vec<i32> = if sieged {
            Vec::new()
        } else {
            enemies
                .iter()
                .filter(|e| can_reach_tile(state, &probe(e), idx))
                .map(|e| e.coords.idx)
                .collect()
        };
        let attackers: Vec<i32> = enemies
            .iter()
            .filter(|e| can_attack_tile(state, &probe(e), idx))
            .map(|e| e.coords.idx)
            .collect();

        // Whoever is (or would be) standing there: can I remove them?
        let threat_unit: Option<&UnitState> = if sieged {
            occupant
        } else {
            enterers
                .iter()
                .filter_map(|i| get_true_unit_at(state, *i))
                .max_by(|a, b| {
                    get_unit_max_health(a)
                        .partial_cmp(&get_unit_max_health(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        };
        let arrives_next_turn = !sieged && !enterers.is_empty();
        let breakable = match threat_unit {
            Some(t) => {
                let mut dmg = 0.0;
                for u in tribe.units.iter().filter(|u| u.coords.idx != idx) {
                    if can_attack_tile(state, &probe(u), idx) {
                        dmg += hypo_damage(state, &probe(u), t, idx);
                    }
                }
                dmg >= t.health
            }
            None => true,
        };
        // Damage the visible enemies could put on the garrison next turn.
        let on_garrison: f32 = garrison.map_or(0.0, |g| {
            attackers
                .iter()
                .filter_map(|i| get_true_unit_at(state, *i))
                .map(|e| hypo_damage(state, &probe(e), g, idx))
                .sum()
        });

        let risk = if sieged {
            if breakable { RISK_BREAKABLE } else { RISK_LOST }
        } else if let Some(g) = garrison {
            if on_garrison <= 0.0 {
                0.0
            } else if on_garrison >= g.health {
                RISK_GARRISON_FALLS
            } else {
                RISK_GARRISON_HOLDS
            }
        } else if arrives_next_turn {
            if breakable { RISK_BREAKABLE } else { RISK_LOST }
        } else {
            0.0
        };
        // Keep any city a visible enemy could walk into even at risk 0 — a
        // safely-held tile is exactly the one whose VACATING must be priced.
        if risk <= 0.0 && enterers.is_empty() {
            continue;
        }
        out.push(CityRisk {
            city: idx,
            sieged,
            open,
            arrives_next_turn,
            breakable,
            attackers,
            enterers,
            risk,
            worth: CITY_WORTH_BASE + CITY_WORTH_PER_LEVEL * city.level as f32,
        });
    }
    out
}

/// Total score-equivalent expected loss across the player's cities — the
/// quantity `goal_potential` subtracts, so any ply that lowers it is paid
/// exactly what it saves.
pub fn expected_city_loss(state: &GameState, player: PlayerId) -> f32 {
    city_risks(state, player).iter().map(|r| r.expected_loss()).sum()
}

/// T3's live read of T2's assessment: the same risk ladder, re-resolved
/// against who is standing on the tile NOW and which named attackers are
/// still alive. The assessment is frozen for the turn, so this is the only
/// part with a gradient — without it the term is a constant and `rank_plies`
/// cannot see a defensive ply at all.
pub fn residual_risk(state: &GameState, player: PlayerId, d: &CityRisk) -> f32 {
    let still_mine = state
        .tribes
        .get(&player)
        .map_or(false, |t| t.cities.iter().any(|c| c.idx == d.city));
    if !still_mine {
        return RISK_LOST;
    }
    let live = |i: &i32| get_true_unit_at(state, *i).filter(|u| u.owner != player);
    match get_true_unit_at(state, d.city) {
        Some(u) if u.owner != player => {
            if d.breakable { RISK_BREAKABLE } else { RISK_LOST }
        }
        Some(g) => {
            let dmg: f32 = d
                .attackers
                .iter()
                .filter_map(live)
                .map(|e| hypo_damage(state, &probe(e), g, d.city))
                .sum();
            if dmg <= 0.0 {
                0.0
            } else if dmg >= g.health {
                RISK_GARRISON_FALLS
            } else {
                RISK_GARRISON_HOLDS
            }
        }
        None => {
            if d.enterers.iter().any(|i| live(i).is_some()) {
                if d.breakable { RISK_BREAKABLE } else { RISK_LOST }
            } else {
                0.0
            }
        }
    }
}

/// Score-equivalent expected loss under the CURRENT state, given T2's
/// assessment. Losing the city outright reads `RISK_LOST`, so no line can
/// win φ by letting one fall.
pub fn residual_city_loss(state: &GameState, player: PlayerId, risks: &[CityRisk]) -> f32 {
    risks
        .iter()
        .map(|d| d.worth * residual_risk(state, player, d))
        .sum()
}

/// Per-city worst-case threat from FOW-visible enemy units. Cities with no
/// deliverable threat produce no entry.
pub fn city_threats(state: &GameState, player: PlayerId) -> Vec<CityThreat> {
    let size = state.settings.size;
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for c in &tribe.cities {
        let garrison = get_true_unit_at(state, c.idx).filter(|u| u.owner == player);
        let mut strike = 0.0f32;
        let mut need = 0.0f32;
        let mut attackers = Vec::new();
        let mut sieged = false;
        let mut reachable_unguarded = false;
        for (id, t) in &state.tribes {
            if *id == player {
                continue;
            }
            for e in &t.units {
                let visible = state
                    .tiles
                    .get(&e.coords.idx)
                    .map_or(false, |tl| tl.explorers.contains(&player));
                if !visible {
                    continue;
                }
                if e.coords.idx == c.idx {
                    sieged = true;
                    attackers.push(e.coords.idx);
                    need = need.max(e.health);
                    continue;
                }
                let m = get_unit_movement(state, e);
                let range = get_unit_setting(e.unit_type).range;
                if get_chebyshev_distance(e.coords.idx, c.idx, size) > 2 * m + range {
                    continue;
                }
                let pe = probe(e);
                if let Some(g) = garrison {
                    if can_attack_tile(state, &pe, c.idx) {
                        strike += hypo_damage(state, &pe, g, c.idx);
                        attackers.push(e.coords.idx);
                        need = need.max(e.health);
                    }
                } else {
                    let target = c.idx;
                    let d = get_chebyshev_distance(pe.coords.idx, target, size);
                    // Same banding as can_attack_tile: plain distance decides
                    // inside one movement; the exact search covers the road band.
                    let reaches = d <= m
                        || (d <= 2 * m
                            && crate::moves::reach_search(state, &pe, Some(&|t: i32| t == target)).1);
                    if reaches {
                        reachable_unguarded = true;
                        attackers.push(e.coords.idx);
                        need = need.max(e.health);
                    } else if can_attack_tile(state, &pe, c.idx) {
                        attackers.push(e.coords.idx);
                        need = need.max(e.health);
                    }
                }
            }
        }
        if attackers.is_empty() {
            continue;
        }
        let at_risk = sieged
            || reachable_unguarded
            || garrison.map_or(false, |g| strike >= RISK_MARGIN * g.health);
        out.push(CityThreat {
            city: c.idx,
            strike,
            need_damage: need,
            attackers,
            sieged,
            reachable_unguarded,
            at_risk,
        });
    }
    out
}

/// EXP_ELO_042 duty partition: is `unit` attack-committed relative to
/// defend city `b`? True if it stands ON an enemy city (state-fact latch —
/// survives Attack-order flicker) or some Attack target is STRICTLY closer
/// than `b` (tie → defense). Comparative, not radius: a radius ring around
/// H contains B itself on Tiny maps (capitals sit at cheb 5).
pub fn attack_committed(
    state: &GameState,
    player: PlayerId,
    unit: &UnitState,
    b: i32,
    attack_targets: &[i32],
) -> bool {
    if let Some(c) = crate::functions::get_city_at(state, unit.coords.idx) {
        if c.owner != player && c.owner != 0 {
            return true;
        }
    }
    let size = state.settings.size;
    let db = get_chebyshev_distance(unit.coords.idx, b, size);
    attack_targets
        .iter()
        .any(|&h| get_chebyshev_distance(unit.coords.idx, h, size) < db)
}

/// Can `unit` strike a unit standing on `target` next turn (fresh flags)?
/// Public wrapper for the press pricing in `reward.rs`.
pub fn covers(state: &GameState, unit: &UnitState, target: i32) -> bool {
    can_attack_tile(state, &probe(unit), target)
}

/// Min-diversion cover assignment for one threatened city: closest units
/// first, full-cover before ring, until the kill damage is met. Ring units
/// (arrive next turn) contribute at half weight. Attack-committed units
/// (see `attack_committed`) are never conscripted. Deterministic.
pub fn defend_plan(
    state: &GameState,
    player: PlayerId,
    threat: &CityThreat,
    attack_targets: &[i32],
) -> DefendPlan {
    let size = state.settings.size;
    let sieger: Option<UnitState> = threat
        .attackers
        .iter()
        .filter_map(|&i| get_true_unit_at(state, i))
        .max_by(|a, b| a.health.total_cmp(&b.health))
        .cloned();
    let mut cands: Vec<(i32, f32, f32, i32)> = Vec::new(); // (tile, sat, dmg, dist)
    if let Some(tribe) = state.tribes.get(&player) {
        for u in &tribe.units {
            let d = get_chebyshev_distance(u.coords.idx, threat.city, size);
            let m = get_unit_movement(state, u);
            if d > 2 * m + RING2_PAD {
                continue;
            }
            if attack_committed(state, player, u, threat.city, attack_targets) {
                continue;
            }
            let is_garrison = u.coords.idx == threat.city;
            let pu = probe(u);
            let sat = if is_garrison || can_attack_tile(state, &pu, threat.city) {
                1.0
            } else if d <= 2 * m {
                0.5
            } else {
                continue;
            };
            let dmg = sieger
                .as_ref()
                .map_or(0.0, |s| hypo_damage(state, &pu, s, threat.city));
            cands.push((u.coords.idx, sat, dmg, d));
        }
    }
    cands.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then(a.3.cmp(&b.3))
            .then(a.0.cmp(&b.0))
    });
    // Greedy fill: full-cover before ring, closest first, until the kill
    // damage is met or the cap is hit.
    let fill = |skip_garrison: bool| -> (Vec<(i32, f32)>, f32) {
        let mut picked = Vec::new();
        let mut got = 0.0f32;
        for &(tile, sat, dmg, _) in &cands {
            if got >= threat.need_damage || picked.len() >= MAX_ASSIGN {
                break;
            }
            if skip_garrison && tile == threat.city {
                continue;
            }
            picked.push((tile, sat));
            got += dmg * sat;
        }
        (picked, got)
    };
    let (assigned, got) = fill(false);
    let has_garrison = assigned.iter().any(|&(t, _)| t == threat.city);
    // Load-bearing test: rebuild the plan without the garrison — if the
    // rest of the roster can meet the kill damage alone, the tile is free.
    let hold_needed = has_garrison && fill(true).1 < threat.need_damage;
    DefendPlan {
        city: threat.city,
        shortfall: (threat.need_damage - got).max(0.0),
        hold_needed,
        assigned,
    }
}

#[cfg(test)]
mod risk_tests {
    use super::tests::{board, unit_at};
    use super::*;
    use crate::ai::oracle_macro::MacroGoal;
    use crate::ai::reward::goal_potential;
    use crate::types::UnitType;

    /// The whole doctrine in one assertion: with an enemy able to walk onto
    /// an empty city, the potential must PREFER the tile occupied.
    #[test]
    fn garrisoning_a_reachable_city_raises_the_potential() {
        let mut state = board(60);
        state
            .tribes
            .get_mut(&2)
            .unwrap()
            .units
            .push(unit_at(61, UnitType::Warrior, 2));
        let goal = MacroGoal::default();
        let aux = |s: &GameState| {
            crate::ai::oracle_macro::scripted_goal_aux(s, 1, &goal, 0, 0, None)
        };
        let open = goal_potential(&state, 1, &goal, Some(&aux(&state)));
        assert!(expected_city_loss(&state, 1) > 0.0, "an empty reachable city must carry risk");

        state
            .tribes
            .get_mut(&1)
            .unwrap()
            .units
            .push(unit_at(60, UnitType::Warrior, 1));
        let held = goal_potential(&state, 1, &goal, Some(&aux(&state)));
        assert!(
            held > open,
            "garrisoned city must outprice the open one: open {open}, held {held}"
        );
    }

    /// The converse, so the term cannot tax quiet turns: no visible enemy,
    /// no risk.
    #[test]
    fn no_visible_enemy_means_no_risk_term() {
        let state = board(60);
        assert_eq!(expected_city_loss(&state, 1), 0.0);
    }

    /// The EXECUTOR contract, and the one the first cut of this term failed:
    /// `rank_plies` holds `aux` FIXED across a turn's plies, so a potential
    /// that only reads the frozen assessment has Δφ = 0 and cannot see a
    /// defensive ply at all. Assess once, then vary only the state.
    #[test]
    fn a_frozen_assessment_still_prices_the_garrison() {
        let mut state = board(60);
        state
            .tribes
            .get_mut(&2)
            .unwrap()
            .units
            .push(unit_at(61, UnitType::Warrior, 2));
        let goal = MacroGoal::default();
        let aux = crate::ai::oracle_macro::scripted_goal_aux(&state, 1, &goal, 0, 0, None);
        let open = goal_potential(&state, 1, &goal, Some(&aux));

        state
            .tribes
            .get_mut(&1)
            .unwrap()
            .units
            .push(unit_at(60, UnitType::Warrior, 1));
        let held = goal_potential(&state, 1, &goal, Some(&aux));
        assert!(
            held > open,
            "same aux, garrison added: held {held} must beat open {open}"
        );
    }

    /// Killing the unit that would walk in is a defense too — the attackers
    /// T2 named are looked up live, so their death shows up in the price.
    #[test]
    fn removing_the_named_threat_clears_the_residual() {
        let mut state = board(60);
        state
            .tribes
            .get_mut(&2)
            .unwrap()
            .units
            .push(unit_at(61, UnitType::Warrior, 2));
        let risks = city_risks(&state, 1);
        assert_eq!(risks.len(), 1);
        assert!(residual_risk(&state, 1, &risks[0]) > 0.0);

        state.tribes.get_mut(&2).unwrap().units.clear();
        assert_eq!(residual_risk(&state, 1, &risks[0]), 0.0);
    }

    /// No line may buy potential by letting the city fall: dropping it from
    /// my cities reads the maximum, never relief.
    #[test]
    fn losing_the_city_is_the_worst_residual() {
        let mut state = board(60);
        state
            .tribes
            .get_mut(&2)
            .unwrap()
            .units
            .push(unit_at(61, UnitType::Warrior, 2));
        let risks = city_risks(&state, 1);
        let standing = residual_risk(&state, 1, &risks[0]);
        state.tribes.get_mut(&1).unwrap().cities.clear();
        assert_eq!(residual_risk(&state, 1, &risks[0]), RISK_LOST);
        assert!(RISK_LOST >= standing);
    }

    /// A garrison that merely holds is doing its job — naming it would pin
    /// the stance to ARM every turn after first contact.
    #[test]
    fn a_holding_garrison_asks_for_no_order() {
        let mut state = board(60);
        state
            .tribes
            .get_mut(&1)
            .unwrap()
            .units
            .push(unit_at(60, UnitType::Warrior, 1));
        state
            .tribes
            .get_mut(&2)
            .unwrap()
            .units
            .push(unit_at(61, UnitType::Warrior, 2));
        let risks = city_risks(&state, 1);
        assert_eq!(risks.len(), 1, "the city stays on the books for pricing");
        assert!(!risks[0].needs_order(), "risk {}", risks[0].risk);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::coords::Coords;
    use crate::states::{CityState, TileState, TribeState};
    use crate::types::{TerrainType, UnitType};

    pub(crate) fn unit_at(idx: i32, unit_type: UnitType, owner: PlayerId) -> UnitState {
        UnitState {
            owner,
            unit_type,
            health: get_unit_max_health(&UnitState {
                unit_type,
                ..Default::default()
            }),
            coords: Coords::from_index(idx, 11),
            ..Default::default()
        }
    }

    /// 11×11 all-field board, every tile explored by both players; a P1
    /// city at `city_idx`.
    pub(crate) fn board(city_idx: i32) -> GameState {
        let mut state = GameState::default();
        state.settings.size = 11;
        for i in 0..121 {
            let mut tile = TileState::default();
            tile.terrain_type = TerrainType::Field;
            tile.explorers.insert(1);
            tile.explorers.insert(2);
            state.tiles.insert(i, tile);
        }
        let mut t1 = TribeState::default();
        t1.cities.push(CityState {
            owner: 1,
            idx: city_idx,
            ..Default::default()
        });
        state.tribes.insert(1, t1);
        state.tribes.insert(2, TribeState::default());
        state
    }

    #[test]
    fn single_adjacent_swordsman_is_at_risk() {
        // Old `near >= 2` proxy missed exactly this (fixture 1786670356).
        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let threats = city_threats(&state, 1);
        assert_eq!(threats.len(), 1);
        let th = &threats[0];
        assert_eq!(th.city, 60);
        assert!(th.strike > 0.0);
        assert!(th.at_risk, "strike {} vs rider hp", th.strike);
    }

    #[test]
    fn unguarded_city_with_reaching_enemy_is_at_risk() {
        let mut state = board(60);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let threats = city_threats(&state, 1);
        assert_eq!(threats.len(), 1);
        assert!(threats[0].reachable_unguarded);
        assert!(threats[0].at_risk);
    }

    #[test]
    fn distant_or_hidden_enemies_are_no_threat() {
        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
        // Far away: outside any strike ring.
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(0, UnitType::Swordsman, 2));
        assert!(city_threats(&state, 1).is_empty());
        // Adjacent but under fog: FOW-honest, not counted.
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        state.tiles.get_mut(&59).unwrap().explorers.remove(&1);
        assert!(city_threats(&state, 1).is_empty());
    }

    #[test]
    fn plan_covers_with_nearby_riders_and_reports_shortfall() {
        let mut state = board(60);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(38, UnitType::Rider, 1)); // cheb 2: rider m=2+Dash covers
        t1.units.push(unit_at(82, UnitType::Rider, 1)); // cheb 2: covers
        let threats = city_threats(&state, 1);
        let plan = defend_plan(&state, 1, &threats[0], &[]);
        assert_eq!(plan.assigned.iter().filter(|&&(_, s)| s == 1.0).count(), 2);
        // Two rider hits do not kill a full swordsman: shortfall is honest.
        let sword_hp = threats[0].need_damage;
        assert!(plan.shortfall > 0.0 && plan.shortfall < sword_hp);
    }

    #[test]
    fn hold_needed_only_when_garrison_is_load_bearing() {
        // Garrison alone vs a swordsman: it is the whole plan.
        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let th = city_threats(&state, 1);
        assert!(defend_plan(&state, 1, &th[0], &[]).hold_needed);
        // Add enough outside cover to meet the kill damage without it.
        for idx in [38, 82, 48, 72] {
            state.tribes.get_mut(&1).unwrap().units.push(unit_at(idx, UnitType::Swordsman, 1));
        }
        let th = city_threats(&state, 1);
        let plan = defend_plan(&state, 1, &th[0], &[]);
        assert!(plan.shortfall == 0.0);
        assert!(!plan.hold_needed);
    }

    /// EXP_ELO_042: latch (on enemy city), strict comparative rule, and
    /// tie-goes-to-defense — on real Tiny geometry (capitals cheb 5).
    #[test]
    fn duty_partition_latch_comparative_and_tie() {
        let mut state = board(29); // own city B = 29 (7,2)
        state.tribes.get_mut(&2).unwrap().cities.push(CityState {
            owner: 2,
            idx: 79, // enemy city H = 79 (2,7)
            ..Default::default()
        });
        let latch = unit_at(79, UnitType::Rider, 1);
        assert!(attack_committed(&state, 1, &latch, 29, &[]));
        let closer_h = unit_at(35, UnitType::Warrior, 1); // cheb B=5, H=4
        assert!(attack_committed(&state, 1, &closer_h, 29, &[79]));
        assert!(!attack_committed(&state, 1, &closer_h, 29, &[]));
        let tie = unit_at(60, UnitType::Warrior, 1); // cheb 3 to both
        assert!(!attack_committed(&state, 1, &tie, 29, &[79]));
    }

    /// EXP_ELO_042: defend B while attacking H — the attacker standing on H
    /// (inside B's candidate ring at Tiny distances!) is never conscripted;
    /// the home defender is.
    #[test]
    fn attacker_on_enemy_city_is_never_conscripted() {
        let mut state = board(60);
        state.tribes.get_mut(&2).unwrap().cities.push(CityState {
            owner: 2,
            idx: 79, // cheb(79, 60) = 3: inside a rider's cover ring
            ..Default::default()
        });
        {
            let t1 = state.tribes.get_mut(&1).unwrap();
            t1.units.push(unit_at(60, UnitType::Rider, 1)); // garrison
            t1.units.push(unit_at(48, UnitType::Rider, 1)); // home defender
            t1.units.push(unit_at(79, UnitType::Rider, 1)); // sieging H
        }
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let th = city_threats(&state, 1);
        let plan = defend_plan(&state, 1, &th[0], &[79]);
        assert!(!plan.assigned.iter().any(|&(t, _)| t == 79));
        assert!(plan.assigned.iter().any(|&(t, _)| t == 48));
    }
}
