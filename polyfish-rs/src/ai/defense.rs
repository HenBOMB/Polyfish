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

/// Min-diversion cover assignment for one threatened city: closest units
/// first, full-cover before ring, until the kill damage is met. Ring units
/// (arrive next turn) contribute at half weight. Deterministic.
pub fn defend_plan(state: &GameState, player: PlayerId, threat: &CityThreat) -> DefendPlan {
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
mod tests {
    use super::*;
    use crate::coords::Coords;
    use crate::states::{CityState, TileState, TribeState};
    use crate::types::{TerrainType, UnitType};

    fn unit_at(idx: i32, unit_type: UnitType, owner: PlayerId) -> UnitState {
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
    fn board(city_idx: i32) -> GameState {
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
        let plan = defend_plan(&state, 1, &threats[0]);
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
        assert!(defend_plan(&state, 1, &th[0]).hold_needed);
        // Add enough outside cover to meet the kill damage without it.
        for idx in [38, 82, 48, 72] {
            state.tribes.get_mut(&1).unwrap().units.push(unit_at(idx, UnitType::Swordsman, 1));
        }
        let th = city_threats(&state, 1);
        let plan = defend_plan(&state, 1, &th[0]);
        assert!(plan.shortfall == 0.0);
        assert!(!plan.hold_needed);
    }
}
