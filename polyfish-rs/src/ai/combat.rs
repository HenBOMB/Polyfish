//! The combat bucket (Aug 2026 taxonomy reorg; renamed from `defense.rs` to
//! sit next to `rules::combat` — AI-level judgment vs. engine-level
//! primitive): EXP_ELO_040's city threat model + defense coverage plans for
//! the macro executor. Pure functions of state, FOW-honest (visible enemies
//! only), built on the real engine math: `compute_reachable_tiles` for reach
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
pub struct DefendPlan {
    pub city: i32,
    /// (unit tile, satisfaction, credit_frac): satisfaction is 1.0 = can
    /// strike an attacker on the city next turn, 0.5 = inside the two-turn
    /// response ring. credit_frac (EXP_ELO_096) is the fraction of this
    /// unit's own damage contribution that was actually needed to close
    /// the city's coverage gap, in priority order — 1.0 while the gap is
    /// still open, tapering down for whichever unit closes it, 0 (and
    /// excluded from this list) once the gap is already shut. Replaces a
    /// flat per-unit share so pricing scales with the unit's own combat
    /// power (via `dmg`) and never cliffs between the last-included and
    /// first-excluded candidate.
    pub assigned: Vec<(i32, f32, f32)>,
    /// Unmet kill damage after assignment — drives recall/prep gradients.
    pub shortfall: f32,
    /// The garrison is load-bearing: without it the assigned cover cannot
    /// meet `need_damage`. Kept as a boolean fact (`hold_margin > 0.0`);
    /// use `hold_margin` for a continuous reward.
    pub hold_needed: bool,
    /// EXP_ELO_096: how load-bearing the garrison is, continuous in
    /// [0, 1] — 0 when the rest of the roster already covers `need_damage`
    /// without it, sliding up to 1 as removing it reopens the full gap.
    /// Replaces a flat yes/no so a barely-load-bearing garrison doesn't
    /// get paid the same as one that's the entire defense.
    pub hold_margin: f32,
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

/// EXP_ELO_095: how strongly `unit` should count as a threat to
/// `target_tile`, given it already passed `can_attack_tile`'s hard gate.
/// Continuous and tanh-bounded rather than a flat 1.0 -- a unit sitting
/// right next to a city is a near-certain threat there; the same unit at
/// the far edge of its own reach (e.g. a Rider spending its whole
/// road-assisted movement to just barely get in range) is a real but much
/// weaker one, and it's exactly this discount that lets it also carry a
/// smaller, non-zero share of threat against a SECOND city it could reach
/// instead this turn -- neither city gets hard-zeroed, both get an
/// intensity proportional to how comfortably this unit can actually reach
/// them.
fn attack_weight(state: &GameState, unit: &UnitState, target_tile: i32) -> f32 {
    let range = get_unit_setting(unit.unit_type).range as f32;
    let d = get_chebyshev_distance(unit.coords.idx, target_tile, state.settings.size) as f32;
    // Within pure attack range, the unit spends NO movement at all to hit
    // this tile -- full certainty, not a discount. The decay applies only
    // to the movement this unit would have to commit to close whatever gap
    // is left beyond its raw range; two units both comfortably within
    // range of their own single city are equally certain, single-city or
    // not -- discounting uncontested in-range attackers was the bug in
    // this function's first cut.
    if d <= range {
        return 1.0;
    }
    // `2*movement` (not just `movement`) is the budget to decay against --
    // `can_attack_tile`'s own outer bound is `2*movement + range` (the
    // road-aware search it falls back to before giving up), specifically
    // to cover a Rider-with-roads hitting a target several tiles out. A
    // narrower `movement`-only budget hard-zeroed that entire outer band
    // even though the unit genuinely CAN still attack there -- exactly the
    // "it could skip the near city for the far one" case this weight
    // exists to keep visible, not erase a second time.
    let m = get_unit_movement(state, unit) as f32;
    let movement_needed = d - range;
    SHARED_ATTACKER_PARTIAL_WEIGHTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    (2.0 * m - movement_needed).max(0.0).tanh()
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
/// EXP_ELO_051: an enemy that needs N turns to arrive is a real threat, just
/// a discounted one. The one-move cliff is what let two cities go: both were
/// vacated with the taker 2-3 tiles out, so the assessment saw nothing.
pub(crate) const RISK_BY_TURNS: [f32; 3] = [1.0, 0.55, 0.25];
/// Enemy turns of arrival the threat model looks ahead.
pub(crate) const THREAT_HORIZON: i32 = 3;
/// Sightings older than this are dropped rather than trusted.
const GHOST_MAX_AGE: i32 = 4;
/// Per-turn confidence decay on a remembered sighting.
const GHOST_TRUST: f32 = 0.6;
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
    /// (tile, weight) of visible enemies that can strike the city tile this
    /// turn — the reachability search is T2's, resolving damage against
    /// whoever ends up standing there is T3's. EXP_ELO_095: `weight` is a
    /// tanh-bounded commitment intensity in (0, 1], NOT a hard 1.0 — a unit
    /// within strike range of several of our cities can only actually
    /// deliver one attack this turn, but it genuinely could pick any of
    /// them (a Rider with roads can skip the nearer city for a farther
    /// one), so every reachable city keeps a reduced, distance-decayed
    /// share of the threat rather than the nearest city getting all of it
    /// and the rest getting none.
    pub attackers: Vec<(i32, f32)>,
    /// Threats that could END their move on the tile inside the horizon.
    pub enterers: Vec<Enterer>,
    /// P(lose the city) proxy in [0,1], as assessed at turn start.
    pub risk: f32,
    /// Score-equivalent worth of the city.
    pub worth: f32,
    /// Damage visible enemies could put on the current garrison next turn
    /// (0.0 when the city is unguarded).
    pub strike: f32,
    /// Damage required to kill the strongest deliverable attacker.
    pub need_damage: f32,
    /// Sharp, this-turn danger flag: sieged, open-and-reachable-next-turn, or
    /// the garrison is about to take near-lethal damage. Independent of
    /// `risk` — `risk` saturates at `RISK_GARRISON_FALLS` for any incoming
    /// damage at or above the garrison's health, so this is what still
    /// distinguishes "80% of health incoming" from "already lethal".
    pub at_risk: bool,
}

/// One way the city could change hands: who, how soon, and how sure we are.
/// `visible` separates a unit on the board (which T3 can kill, and then stop
/// counting) from a remembered sighting under fog (which it cannot verify).
#[derive(Debug, Clone, PartialEq)]
pub struct Enterer {
    pub tile: i32,
    pub turns: i32,
    pub trust: f32,
    pub visible: bool,
}

impl CityRisk {
    pub fn expected_loss(&self) -> f32 {
        self.risk * self.worth
    }
    /// Worth a Defend order from T2. A garrison that merely *holds* is already
    /// doing its job — naming it would pin the stance to ARM for the rest of
    /// the game; its vacating is still priced through `residual_risk`. ORs in
    /// `at_risk` so a garrison taking near-lethal (but not literally lethal)
    /// damage still gets named — `risk` alone saturates before that point.
    pub fn needs_order(&self) -> bool {
        self.risk >= RISK_GARRISON_FALLS || self.at_risk
    }
}

/// Can `unit` (fresh flags) END its move on `target_tile`? Same distance
/// banding as `can_attack_tile`: plain distance inside one move, the exact
/// road-aware search only in the band beyond it.
fn can_reach_tile(state: &GameState, unit: &UnitState, target_tile: i32) -> bool {
    turns_to_reach(state, unit, target_tile, 1).is_some()
}

/// EXP_ELO_051: how many enemy turns until `unit` could STAND on
/// `target_tile`, or None beyond `max_turns`. The engine's own cost search
/// does the work, so roads count — a Rider that a road chain puts three
/// tiles further out reads as a one-turn threat, which is exactly the city
/// snipe the one-move horizon was blind to.
///
/// Escape ("bounce") is honoured: a unit that may move again after a kill
/// covers up to twice its movement in a turn, but only when there is
/// something of mine in range to strike on the way through.
fn turns_to_reach(
    state: &GameState,
    unit: &UnitState,
    target_tile: i32,
    max_turns: i32,
) -> Option<i32> {
    let size = state.settings.size;
    let m = get_unit_movement(state, unit).max(1);
    let d = get_chebyshev_distance(unit.coords.idx, target_tile, size);
    if d == 0 {
        return Some(0);
    }
    let per_turn = if bounce_reach(state, unit) { 2 * m } else { m };
    // Chebyshev is a lower bound on path length: beyond this it cannot arrive.
    if d > per_turn * max_turns {
        return None;
    }
    if d <= per_turn {
        return Some(1);
    }
    let budget = max_turns * if per_turn > m { 2 } else { 1 };
    let (costs, _) = crate::moves::reach_search_turns(state, unit, budget, None);
    // The city tile is usually blocked by its own garrison, and "could you
    // stand here if it were empty" is the whole question — so price the
    // step-in from a neighbour, not a landing the current occupant forbids.
    let mut best = costs.get(&target_tile).copied();
    for n in crate::functions::get_adjacent_indices(state, target_tile, 1) {
        if let Some(&c) = costs.get(&n) {
            let v = c + 1.0;
            if best.map_or(true, |b| v < b) {
                best = Some(v);
            }
        }
    }
    let turns = (best? / per_turn as f32).ceil() as i32;
    (turns <= max_turns).then_some(turns.max(1))
}

/// Does this unit get a second move after a kill, with a victim in reach to
/// trigger it? Rider carries Escape, which is how a "safe" city two moves
/// away is taken in one turn.
fn bounce_reach(state: &GameState, unit: &UnitState) -> bool {
    if !has_skill(unit, SkillType::Escape) {
        return false;
    }
    let size = state.settings.size;
    let span = get_unit_movement(state, unit).max(1)
        + get_unit_setting(unit.unit_type).range.max(1);
    state.tribes.iter().any(|(id, t)| {
        *id != unit.owner
            && t.units
                .iter()
                .any(|v| get_chebyshev_distance(unit.coords.idx, v.coords.idx, size) <= span)
    })
}

/// Enemies this player can reason about: everything visible, plus units last
/// seen leaving into the fog (`TribeState::enemy_ghosts`). A sighting is not
/// forgotten the instant it steps out of vision — the rider that bounced
/// away is still out there, and pretending otherwise is what leaves a city
/// open. Ghosts are placed at their last-seen tile and their contribution is
/// discounted by age.
pub fn threat_units(state: &GameState, player: PlayerId) -> Vec<(UnitState, f32)> {
    let mut out: Vec<(UnitState, f32)> = state
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
        .map(|u| (probe(u), 1.0))
        .collect();
    let Some(me) = state.tribes.get(&player) else {
        return out;
    };
    let now = state.settings.turn;
    for (&idx, g) in &me.enemy_ghosts {
        let age = (now - g.turn).max(0);
        if age > GHOST_MAX_AGE {
            continue;
        }
        // Already accounted for if something visible stands there.
        if out.iter().any(|(u, _)| u.coords.idx == idx) {
            continue;
        }
        let mut u = UnitState {
            owner: g.owner,
            unit_type: g.unit_type,
            coords: crate::coords::Coords::from_index(idx, state.settings.size),
            ..Default::default()
        };
        u.health = get_unit_max_health(&u);
        out.push((u, GHOST_TRUST.powi(age)));
    }
    out
}

/// My opportunity to take ENEMY cities: `city_risks_with_threats` computed
/// from the ENEMY's own perspective, with MY units standing in as their
/// threats. Their risk of losing a city to me IS my opportunity of gaining
/// it (Verdi, Aug 2026: "a city with .75 risk of loss is .75 opp of gain
/// for your enemy") -- so this reuses the exact same risk math and dials
/// rather than a second, independently-tuned heuristic. `state` is the
/// caller's own fogged view, so whatever it knows (or doesn't) about the
/// enemy's cities/garrisons flows through unchanged -- no new omniscience.
/// My own units need no fog filter (always fully known to me), unlike
/// `threat_units`'s enemy-facing FOW check.
pub fn city_opportunities(state: &GameState, player: PlayerId, enemy: PlayerId) -> Vec<CityRisk> {
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let my_units_as_threats: Vec<(UnitState, f32)> =
        tribe.units.iter().map(|u| (probe(u), 1.0)).collect();
    city_risks_with_threats(state, enemy, &my_units_as_threats)
}

/// The single best (highest-opportunity) enemy city to attack, across every
/// visible enemy tribe -- `known_enemy_capital`'s counterpart for "any weak
/// city", not just the capital. `None` if no enemy city clears `min_risk`
/// (avoids proposing a candidate for a city with negligible opportunity).
pub fn best_attack_opportunity(
    state: &GameState,
    player: PlayerId,
    min_risk: f32,
) -> Option<(i32, f32)> {
    state
        .tribes
        .keys()
        .filter(|&&id| id != player)
        .flat_map(|&enemy| city_opportunities(state, player, enemy))
        .filter(|r| r.risk >= min_risk)
        .map(|r| (r.city, r.risk))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// Per-city expected loss: who can reach the tile, whether a siege there
/// would be breakable, and what that costs me. FOW-honest — only visible
/// enemy units, read with their real movement (roads and tech included).
pub fn city_risks(state: &GameState, player: PlayerId) -> Vec<CityRisk> {
    let threats = threat_units(state, player);
    city_risks_with_threats(state, player, &threats)
}

/// EXP_ELO_095 diagnostic (temporary): how many (unit, city) attacker
/// entries actually get a PARTIAL (< 1.0) weight, i.e. how often the
/// distance-decay in `attack_weight` fires at all versus every attacker
/// being comfortably within pure range everywhere it appears.
pub static SHARED_ATTACKER_PARTIAL_WEIGHTS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// EXP_ELO_096 diagnostic (temporary): how often the new waterfall credit
/// actually produces a fractional (0, 1) `credit_frac`/`hold_margin` —
/// i.e. how often the smoothing changes anything versus every assignment
/// landing at the old system's implicit 0 or 1.
pub static DEFEND_CREDIT_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DEFEND_CREDIT_PARTIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DEFEND_HOLD_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static DEFEND_HOLD_PARTIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Same as [`city_risks`], but takes an already-computed threat list instead
/// of scanning for one. `threat_units` depends only on the OPPONENT's units
/// and ghosts — never on the acting player's own move — so a caller ranking
/// many of ITS OWN candidate moves against the same board (e.g.
/// `macro_exec::rank_plies`) can compute it once per ply instead of once per
/// candidate. Profiling (EXP_ELO_061 throughput investigation, Aug 2026)
/// found `city_risks`'s per-candidate re-scan was 64-86% of actor CPU time
/// under macro-mcts — this split is the fix.
pub fn city_risks_with_threats(
    state: &GameState,
    player: PlayerId,
    threats: &[(UnitState, f32)],
) -> Vec<CityRisk> {
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    if threats.is_empty() {
        return Vec::new();
    }

    // Pass 1: everything that does NOT depend on cross-city information --
    // occupancy, the broader multi-turn `enterers` set, and the RAW
    // this-turn `attackers` set (unfiltered).
    struct Pre<'c> {
        city: &'c crate::states::CityState,
        occupant: Option<&'c UnitState>,
        sieged: bool,
        open: bool,
        enterers: Vec<Enterer>,
        attackers: Vec<(i32, f32)>,
        arrives_next_turn: bool,
        breakable: bool,
    }
    let mut pre: Vec<Pre> = Vec::with_capacity(tribe.cities.len());
    for city in &tribe.cities {
        let idx = city.idx;
        let occupant = get_true_unit_at(state, idx);
        let sieged = occupant.map_or(false, |u| u.owner != player);
        let open = occupant.is_none();

        let enterers: Vec<Enterer> = if sieged {
            Vec::new()
        } else {
            threats
                .iter()
                .filter_map(|(e, trust)| {
                    turns_to_reach(state, e, idx, THREAT_HORIZON).map(|turns| Enterer {
                        tile: e.coords.idx,
                        turns: turns.max(1),
                        trust: *trust,
                        visible: *trust >= 1.0,
                    })
                })
                .collect()
        };
        // EXP_ELO_095: weighted, not a bare tile list -- `attack_weight`
        // below is the tanh-bounded, distance-decayed commitment intensity.
        let attackers: Vec<(i32, f32)> = threats
            .iter()
            .filter(|(e, trust)| *trust >= 1.0 && can_attack_tile(state, e, idx))
            .map(|(e, _)| (e.coords.idx, attack_weight(state, e, idx)))
            .collect();

        let threat_unit: Option<&UnitState> = if sieged {
            occupant
        } else {
            enterers
                .iter()
                .filter_map(|e| get_true_unit_at(state, e.tile))
                .max_by(|a, b| {
                    get_unit_max_health(a)
                        .partial_cmp(&get_unit_max_health(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        };
        let arrives_next_turn = !sieged && enterers.iter().any(|e| e.turns <= 1);
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
        pre.push(Pre {
            city,
            occupant,
            sieged,
            open,
            enterers,
            attackers,
            arrives_next_turn,
            breakable,
        });
    }

    // EXP_ELO_094 found (then EXP_ELO_095 corrected): a single enemy unit
    // cannot deliver its whole attack to TWO of our cities in the same
    // turn, so counting it at FULL, undiscounted weight in every city
    // within strike range (a unit near two garrisons priced as if it could
    // kill both this turn) is wrong -- confirmed as the exact mechanism
    // behind a garrison-preserving Attack scoring -558 in a real game.
    // EXP_ELO_094's first fix over-corrected the other way: attributing the
    // unit ENTIRELY to its nearest city and zeroing every other one denies
    // a real possibility -- a mobile unit (a Rider with roads reaching 4+
    // tiles) can genuinely skip the near target for a farther one. Verdi:
    // "it's either 0.2 in both cities... or .6 in one, .15 in the other" --
    // the fix is a continuous, tanh-bounded commitment intensity per
    // (unit, city) pair (`attack_weight`, computed inline above), not a
    // winner-take-all attribution. A unit's total commitment is bounded by
    // construction (tanh saturates, and a unit far enough from EITHER city
    // gets a small share of BOTH) without ever hard-zeroing a city that is
    // still genuinely reachable this turn -- `can_attack_tile` (unchanged)
    // remains the only hard cutoff, for cities truly out of reach.

    // Pass 2: everything downstream of the (now weighted) `attackers`.
    let mut out = Vec::new();
    for p in pre {
        let Pre {
            city,
            occupant,
            sieged,
            open,
            enterers,
            attackers,
            arrives_next_turn,
            breakable,
        } = p;
        let idx = city.idx;
        let garrison = occupant.filter(|u| u.owner == player);

        // Damage the visible enemies could put on the garrison next turn,
        // weighted by each attacker's commitment intensity (EXP_ELO_095).
        let on_garrison: f32 = garrison.map_or(0.0, |g| {
            attackers
                .iter()
                .filter_map(|&(i, w)| get_true_unit_at(state, i).map(|e| (e, w)))
                .map(|(e, w)| hypo_damage(state, &probe(e), g, idx) * w)
                .sum()
        });

        let severity = if breakable { RISK_BREAKABLE } else { RISK_LOST };
        let risk = if sieged {
            severity
        } else if let Some(g) = garrison {
            garrison_risk(on_garrison, g.health, enterers.is_empty())
        } else {
            arrival_factor(&enterers) * severity
        };
        // Keep any city a threat could walk into even at risk 0 — a safely
        // held tile is exactly the one whose VACATING must be priced.
        if risk <= 0.0 && enterers.is_empty() {
            continue;
        }
        let at_risk = sieged
            || (open && arrives_next_turn)
            || garrison.map_or(false, |g| on_garrison >= RISK_MARGIN * g.health);
        // EXP_ELO_054 fix: sourced from `attackers` (this-turn-only), the same
        // pool `defend_plan`'s `sieger` reads — NOT `threat_unit` (sourced from
        // the broader multi-turn `enterers`). The two used to be the same set
        // by construction in the old city_threats model; decoupling them left
        // `need_damage` > 0 with no `attackers` entry to back it, so
        // `defend_plan`'s `sieger` came back `None`, every candidate's damage
        // contribution silently computed as 0, and `fill()` never met the
        // (unreachable) damage target — so it grabbed MAX_ASSIGN units on
        // every multi-turn-only threat regardless of whether they helped.
        // Measured: this alone was the EXP_ELO_054 regression (cities lost
        // 31 vs the 19 gate).
        let need_damage = if sieged {
            occupant.map_or(0.0, |u| u.health)
        } else {
            attackers
                .iter()
                .filter_map(|&(i, w)| get_true_unit_at(state, i).map(|u| u.health * w))
                .fold(0.0, f32::max)
        };
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
            strike: on_garrison,
            need_damage,
            at_risk,
        });
    }
    out
}

/// Soonest credible arrival, discounted by how far out it is and how much a
/// remembered sighting is trusted.
fn arrival_factor(enterers: &[Enterer]) -> f32 {
    enterers
        .iter()
        .map(|e| {
            let band = RISK_BY_TURNS
                .get((e.turns - 1).max(0) as usize)
                .copied()
                .unwrap_or(0.0);
            band * e.trust
        })
        .fold(0.0, f32::max)
}

/// Continuous in the garrison's health: a full-strength defender behind the
/// city's defence bonus prices far better than a wounded one, so holding the
/// tile at max health beats attacking out of it. The old three-step ladder
/// read both as `RISK_GARRISON_HOLDS` and gave the executor no reason to
/// keep its defender whole.
fn garrison_risk(incoming: f32, health: f32, no_enterers: bool) -> f32 {
    if incoming <= 0.0 && no_enterers {
        return 0.0;
    }
    let ratio = if health > 0.0 {
        (incoming / health).clamp(0.0, 1.0)
    } else {
        1.0
    };
    RISK_GARRISON_HOLDS + (RISK_GARRISON_FALLS - RISK_GARRISON_HOLDS) * ratio
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
    let live = |i: i32| get_true_unit_at(state, i).filter(|u| u.owner != player);
    // A named threat still counts unless it is visible AND now dead: a
    // remembered sighting cannot be disproved by looking at the board.
    let standing: Vec<&Enterer> = d
        .enterers
        .iter()
        .filter(|e| !e.visible || live(e.tile).is_some())
        .collect();
    let severity = if d.breakable { RISK_BREAKABLE } else { RISK_LOST };
    match get_true_unit_at(state, d.city) {
        Some(u) if u.owner != player => severity,
        Some(g) => {
            let dmg: f32 = d
                .attackers
                .iter()
                .filter_map(|&(i, w)| live(i).map(|e| (e, w)))
                .map(|(e, w)| hypo_damage(state, &probe(e), g, d.city) * w)
                .sum();
            garrison_risk(dmg, g.health, standing.is_empty())
        }
        None => {
            let owned: Vec<Enterer> = standing.into_iter().cloned().collect();
            arrival_factor(&owned) * severity
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
pub fn unit_covers_threat(state: &GameState, unit: &UnitState, target: i32) -> bool {
    can_attack_tile(state, &probe(unit), target)
}

/// Min-diversion cover assignment for one threatened city: closest units
/// first, full-cover before ring, until the kill damage is met. Ring units
/// (arrive next turn) contribute at half weight. Attack-committed units
/// (see `attack_committed`) are never conscripted. Deterministic.
pub fn defend_plan(
    state: &GameState,
    player: PlayerId,
    threat: &CityRisk,
    attack_targets: &[i32],
) -> DefendPlan {
    defend_plan_impl(state, player, threat, attack_targets, false)
}

/// EXP_ELO_103 diagnostic: `defend_plan` but ALWAYS uses the sieged/open
/// (garrison-independent) need_damage framing, even when a garrison is
/// present. Used to hand-verify whether decoupling `defend_cover`'s
/// waterfall cap from the garrisoned branch's collapsed need_damage would
/// actually restore credit to nearby non-garrison units, before committing
/// to that restructuring. Not wired into any pricing path yet.
pub fn defend_plan_open_framing(
    state: &GameState,
    player: PlayerId,
    threat: &CityRisk,
    attack_targets: &[i32],
) -> DefendPlan {
    defend_plan_impl(state, player, threat, attack_targets, true)
}

fn defend_plan_impl(
    state: &GameState,
    player: PlayerId,
    threat: &CityRisk,
    attack_targets: &[i32],
    force_open_framing: bool,
) -> DefendPlan {
    let size = state.settings.size;
    let garrison = get_true_unit_at(state, threat.city).filter(|u| u.owner == player);
    let framing_garrison = if force_open_framing { None } else { garrison.as_ref() };

    // Verdi (Aug 2026): an unsieged, garrisoned city can be hit by EVERY
    // listed attacker in the same enemy turn (sequential melee strikes,
    // not just its single strongest attacker) -- `threat.strike` already
    // sums that correctly. The old model asked "can I kill the strongest
    // attacker" and reported safe whenever that alone was true, even when
    // several smaller attackers combined already exceed the garrison's
    // health (confirmed: a 6hp garrison facing 4 attackers summing to 16
    // damage read shortfall=0.0 because one attacker's own health, 10, was
    // coverable). Greedily eliminate the biggest damage contributors first
    // -- most incoming damage removed per kill -- until what's left drops
    // under the garrison's HP; that's the set that must die this turn.
    // Sieged/open cities keep the single-strongest-unit framing: there is
    // no "garrison HP" to protect, only a siege to break or an entry to
    // deter.
    // (units that must die this turn, the kill-damage threshold they set).
    // Garrisoned: empty `must_kill` is a real, deliberate zero -- the
    // garrison already outlasts the total incoming strike on its own, and
    // nothing needs preparing. Sieged/open: no garrison HP to protect,
    // keep the original single-strongest-unit framing (siege-break /
    // entry-deterrence), so it never floors at zero on a real threat.
    let (must_kill, need_damage): (Vec<UnitState>, f32) = if let Some(g) = framing_garrison {
        // EXP_ELO_095: every quantity derived from an attacker -- its
        // contribution to the incoming strike, its priority, and the kill
        // damage it sets once selected -- is scaled by its commitment
        // weight to THIS city. A unit only weakly postured here (because
        // it's more credibly threatening a different one of our cities)
        // is proportionally cheaper to prepare for, not a full hit.
        let mut contribs: Vec<(UnitState, f32, f32)> = threat // (unit, weighted dmg, weight)
            .attackers
            .iter()
            .filter_map(|&(i, w)| get_true_unit_at(state, i).map(|u| (u, w)))
            .map(|(u, w)| {
                let dmg = hypo_damage(state, &probe(u), g, threat.city) * w;
                (u.clone(), dmg, w)
            })
            .collect();
        contribs.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut remaining = threat.strike;
        let mut out: Vec<(UnitState, f32)> = Vec::new();
        for (u, dmg, w) in contribs {
            // Same bar `at_risk` already uses (`RISK_MARGIN`): one hit
            // leaving the garrison below a second hit is a threat worth
            // preparing for, not a nuisance. Keeps this branch identical
            // to the legacy single-attacker behavior at or above the
            // margin (see this function's doc comment) and only changes
            // outcomes for genuinely weak pokes or multi-attacker overload.
            if remaining < RISK_MARGIN * g.health {
                break;
            }
            remaining -= dmg;
            out.push((u, w));
        }
        let need = out.iter().map(|(u, w)| u.health * w).sum();
        (out.into_iter().map(|(u, _)| u).collect(), need)
    } else {
        let sieger: Vec<(UnitState, f32)> = threat
            .attackers
            .iter()
            .filter_map(|&(i, w)| get_true_unit_at(state, i).map(|u| (u, w)))
            .max_by(|(a, wa), (b, wb)| (a.health * wa).total_cmp(&(b.health * wb)))
            .map(|(u, w)| (u.clone(), w))
            .into_iter()
            .collect();
        let need = sieger.first().map_or(threat.need_damage, |(u, w)| u.health * w);
        (sieger.into_iter().map(|(u, _)| u).collect(), need)
    };

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
            // Credited against whichever must-kill attacker this unit hits
            // hardest -- a simplification (kill damage isn't truly fungible
            // across different targets) but a strict improvement over
            // pricing against one fixed attacker only.
            let dmg = must_kill
                .iter()
                .map(|s| hypo_damage(state, &pu, s, threat.city))
                .fold(0.0f32, f32::max);
            cands.push((u.coords.idx, sat, dmg, d));
        }
    }
    cands.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then(a.3.cmp(&b.3))
            .then(a.0.cmp(&b.0))
    });
    // EXP_ELO_096: waterfall fill, full-cover before ring, closest first.
    // Each candidate is credited only for whatever slice of `need_damage`
    // is still open when it's reached — the unit that closes the gap gets
    // a partial `credit_frac` for just its needed sliver instead of a full
    // flat share, so there is no cliff between the last-included and
    // first-excluded candidate. A unit that contributes nothing to the
    // still-open gap (because the gap is already shut, or it deals no
    // damage at all) is skipped rather than recruited at zero value.
    let fill = |skip_garrison: bool| -> (Vec<(i32, f32, f32)>, f32) {
        let mut picked = Vec::new();
        let mut got = 0.0f32;
        for &(tile, sat, dmg, _) in &cands {
            if picked.len() >= MAX_ASSIGN {
                break;
            }
            if skip_garrison && tile == threat.city {
                continue;
            }
            let contribution = dmg * sat;
            let remaining = (need_damage - got).max(0.0);
            let credited = contribution.min(remaining);
            if credited <= 0.0 {
                continue;
            }
            let credit_frac = credited / contribution;
            DEFEND_CREDIT_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if credit_frac < 1.0 {
                DEFEND_CREDIT_PARTIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            picked.push((tile, sat, credit_frac));
            got += credited;
        }
        (picked, got)
    };
    let (assigned, got) = fill(false);
    let has_garrison = assigned.iter().any(|&(t, _, _)| t == threat.city);
    // Load-bearing test: rebuild the plan without the garrison — if the
    // rest of the roster can meet the kill damage alone, the tile is free.
    let without_garrison = fill(true).1;
    let hold_margin = if has_garrison && need_damage > 0.0 {
        DEFEND_HOLD_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let m = ((need_damage - without_garrison) / need_damage).clamp(0.0, 1.0);
        if m > 0.0 && m < 1.0 {
            DEFEND_HOLD_PARTIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        m
    } else {
        0.0
    };
    DefendPlan {
        city: threat.city,
        shortfall: (need_damage - got).max(0.0),
        hold_needed: hold_margin > 0.0,
        hold_margin,
        assigned,
    }
}

#[cfg(test)]
mod risk_tests {
    use super::tests::{board, unit_at};
    use super::*;
    use crate::ai::oracle_macro::MacroGoal;
    use crate::ai::reward::goal_potential;
    use crate::states::{CityState, TileState, TribeState};
    use crate::types::{TerrainType, UnitType};

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
            crate::ai::oracle_macro::compute_goal_aux(s, 1, &goal, 0, 0, None)
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
        let aux = crate::ai::oracle_macro::compute_goal_aux(&state, 1, &goal, 0, 0, None);
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

    /// EXP_ELO_094: one enemy unit within striking range of two of our
    /// cities cannot deliver its whole attack to both this turn — it has to
    /// pick one. Before this fix, `city_risks` counted it at full,
    /// undiscounted weight in EVERY city it could reach, pricing as if it
    /// could kill both simultaneously. Confirmed as the exact mechanism
    /// behind a garrison-preserving Attack scoring -558 in a real game:
    /// damaging the attacker shrank a SEPARATE city's `need_damage`, which
    /// silently deleted that city's defend_cover credit even though the
    /// actually-attacked city's own plan never changed.
    #[test]
    fn a_shared_attacker_keeps_a_reduced_but_nonzero_weight_at_the_farther_city() {
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
        // Rider (range 1, movement 2): city A at (5,6)=61 is distance 1 --
        // within pure range, so it must be FULL weight (no movement spent).
        // City B at (5,7)=62 is distance 2 -- beyond pure range, needs 1
        // tile of movement to reach, so it must carry a reduced but still
        // real weight (EXP_ELO_095: a shared attacker is never hard-zeroed
        // at a city it can still genuinely reach -- only discounted by how
        // much of its own movement budget that city actually costs it).
        t1.cities.push(CityState { owner: 1, idx: 61, ..Default::default() });
        t1.cities.push(CityState { owner: 1, idx: 62, ..Default::default() });
        state.tribes.insert(1, t1);
        let mut t2 = TribeState::default();
        t2.units.push(unit_at(60, UnitType::Rider, 2));
        state.tribes.insert(2, t2);

        let risks = city_risks(&state, 1);
        let a = risks.iter().find(|r| r.city == 61).expect("city A must be in city_risks");
        let b = risks.iter().find(|r| r.city == 62).expect("city B must be in city_risks");
        let wa = a.attackers.iter().find(|&&(t, _)| t == 60).map(|&(_, w)| w);
        let wb = b.attackers.iter().find(|&&(t, _)| t == 60).map(|&(_, w)| w);
        assert!(wa.is_some(), "the nearer city must list the attacker: {:?}", a.attackers);
        assert!(
            wb.is_some(),
            "the farther-but-still-reachable city must ALSO list the attacker, not be hard-zeroed: {:?}",
            b.attackers
        );
        let (wa, wb) = (wa.unwrap(), wb.unwrap());
        assert_eq!(wa, 1.0, "within pure range costs no movement -- must be full certainty, got {wa}");
        assert!(wb > 0.0 && wb < 1.0, "weight must be a bounded, genuinely partial intensity, got {wb}");
        assert!(
            wa > wb,
            "the nearer city must carry the LARGER share of this unit's threat: near={wa} far={wb}"
        );
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

    /// EXP_ELO_051, the seed-1786807405 loss in one test. City 85 was vacated
    /// on t9 with the taker THREE tiles out; the one-move horizon saw nothing,
    /// so stepping off was priced at exactly 0.0000. An enemy beyond one move
    /// must still make vacating cost something.
    #[test]
    fn an_enemy_two_moves_out_still_makes_vacating_cost() {
        let mut state = board(60);
        state
            .tribes
            .get_mut(&1)
            .unwrap()
            .units
            .push(unit_at(60, UnitType::Warrior, 1));
        // Chebyshev 2 from the city: outside a Warrior's single move.
        state
            .tribes
            .get_mut(&2)
            .unwrap()
            .units
            .push(unit_at(62, UnitType::Warrior, 2));
        let risks = city_risks(&state, 1);
        assert_eq!(risks.len(), 1, "a two-move threat must be on the books");
        assert!(risks[0].enterers.iter().all(|e| e.turns >= 2));
        let held = residual_risk(&state, 1, &risks[0]);

        state.tribes.get_mut(&1).unwrap().units.clear();
        let open = residual_risk(&state, 1, &risks[0]);
        assert!(
            open > held,
            "vacating must cost: held {held}, open {open}"
        );
    }

    /// EXP_ELO_054 regression: a multi-turn-only threat (no `attackers` this
    /// turn) must not make `defend_plan` grab units it cannot justify.
    /// `need_damage` used to be sourced from the broader `enterers` set while
    /// `defend_plan`'s `sieger` reads only `attackers` — decoupled, `sieger`
    /// came back `None`, every candidate's damage silently computed as 0,
    /// and `fill()` never met the (unreachable) target, so it grabbed
    /// `MAX_ASSIGN` nearby units on every such city regardless of whether
    /// they helped. Measured as the actual cause of the 054 gate failure
    /// (cities lost 31 vs the 19 gate, 42/48 wins vs the 46/48 gate).
    #[test]
    fn a_multi_turn_only_threat_does_not_over_recruit_defenders() {
        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Warrior, 1));
        // Chebyshev 3: beyond even Dash's move+range=2 reach (so nothing can
        // strike this turn), but within the 3-turn horizon (on the books).
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(63, UnitType::Warrior, 2));
        // Bystanders that a broken defend_plan would wrongly recruit.
        for idx in [38, 82, 48] {
            state.tribes.get_mut(&1).unwrap().units.push(unit_at(idx, UnitType::Warrior, 1));
        }
        let risks = city_risks(&state, 1);
        assert_eq!(risks.len(), 1);
        assert!(risks[0].attackers.is_empty(), "nothing can strike this turn");
        assert_eq!(risks[0].need_damage, 0.0, "no this-turn attacker to size a kill against");

        let plan = defend_plan(&state, 1, &risks[0], &[]);
        assert!(
            plan.assigned.is_empty(),
            "an unreachable target must not recruit bystanders: {:?}",
            plan.assigned
        );
        assert_eq!(plan.shortfall, 0.0);
        assert!(!plan.hold_needed);
    }

    /// The horizon is graded, not a second cliff: nearer is worse.
    #[test]
    fn risk_falls_off_with_distance() {
        let mut at = |d: i32| {
            let mut s = board(60);
            s.tribes.get_mut(&2).unwrap().units.push(unit_at(60 + d, UnitType::Warrior, 2));
            city_risks(&s, 1).first().map(|r| r.risk).unwrap_or(0.0)
        };
        let (near, mid, far) = (at(1), at(2), at(3));
        assert!(near > mid && mid > far, "near {near}, mid {mid}, far {far}");
        assert!(far > 0.0);
    }

    /// B2: holding at full health must price better than holding wounded, or
    /// the executor has no reason to keep its defender whole (Verdi: "staying
    /// put, and not even initiating an attack, is best").
    #[test]
    fn a_wounded_garrison_prices_worse_than_a_whole_one() {
        let mut state = board(60);
        state
            .tribes
            .get_mut(&2)
            .unwrap()
            .units
            .push(unit_at(61, UnitType::Warrior, 2));
        state
            .tribes
            .get_mut(&1)
            .unwrap()
            .units
            .push(unit_at(60, UnitType::Warrior, 1));
        let risks = city_risks(&state, 1);
        let whole = residual_risk(&state, 1, &risks[0]);

        state.tribes.get_mut(&1).unwrap().units[0].health = 2.0;
        let hurt = residual_risk(&state, 1, &risks[0]);
        assert!(hurt > whole, "whole {whole} must beat wounded {hurt}");
    }

    /// Verdi: "we had already seen the rider bounce away into the fog … theres
    /// a chance its within reach". A sighting is not forgotten the moment it
    /// leaves vision.
    #[test]
    fn a_remembered_sighting_still_counts() {
        use crate::states::GhostRecord;
        let mut state = board(60);
        state.settings.turn = 5;
        state.tribes.get_mut(&1).unwrap().enemy_ghosts.insert(
            61,
            GhostRecord { unit_type: UnitType::Rider, owner: 2, turn: 4 },
        );
        let risks = city_risks(&state, 1);
        assert_eq!(risks.len(), 1, "a fog sighting must still raise a risk");
        assert!(risks[0].risk > 0.0);
        assert!(risks[0].enterers.iter().all(|e| !e.visible));
        // …but discounted against a unit we can actually see.
        let mut seen = board(60);
        seen.tribes.get_mut(&2).unwrap().units.push(unit_at(61, UnitType::Rider, 2));
        assert!(city_risks(&seen, 1)[0].risk > risks[0].risk);
    }

    /// A Rider carries Escape: with something to kill on the way it moves
    /// again after the strike, so "two moves away" is a one-turn threat.
    #[test]
    fn escape_lets_a_rider_arrive_a_turn_sooner() {
        let mut far = board(60);
        far.tribes.get_mut(&2).unwrap().units.push(unit_at(64, UnitType::Rider, 2));
        let alone = city_risks(&far, 1).first().map(|r| r.risk).unwrap_or(0.0);

        let mut bait = far.clone();
        bait.tribes.get_mut(&1).unwrap().units.push(unit_at(63, UnitType::Warrior, 1));
        let with_victim = city_risks(&bait, 1)
            .iter()
            .find(|r| r.city == 60)
            .map(|r| r.enterers.iter().map(|e| e.turns).min().unwrap_or(9))
            .unwrap_or(9);
        assert!(alone > 0.0);
        assert_eq!(with_victim, 1, "a bounce puts the city one turn away");
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
        let risks = city_risks(&state, 1);
        assert_eq!(risks.len(), 1);
        let r = &risks[0];
        assert_eq!(r.city, 60);
        assert!(r.strike > 0.0);
        assert!(r.at_risk, "strike {} vs rider hp", r.strike);
    }

    #[test]
    fn unguarded_city_with_reaching_enemy_is_at_risk() {
        let mut state = board(60);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let risks = city_risks(&state, 1);
        assert_eq!(risks.len(), 1);
        assert!(risks[0].open && risks[0].arrives_next_turn);
        assert!(risks[0].at_risk);
    }

    #[test]
    fn distant_or_hidden_enemies_are_no_threat() {
        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
        // Far away: outside any strike ring.
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(0, UnitType::Swordsman, 2));
        assert!(city_risks(&state, 1).is_empty());
        // Adjacent but under fog: FOW-honest, not counted.
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        state.tiles.get_mut(&59).unwrap().explorers.remove(&1);
        assert!(city_risks(&state, 1).is_empty());
    }

    #[test]
    fn plan_covers_with_nearby_riders_and_reports_shortfall() {
        let mut state = board(60);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(38, UnitType::Rider, 1)); // cheb 2: rider m=2+Dash covers
        t1.units.push(unit_at(82, UnitType::Rider, 1)); // cheb 2: covers
        let risks = city_risks(&state, 1);
        let plan = defend_plan(&state, 1, &risks[0], &[]);
        assert_eq!(plan.assigned.iter().filter(|&&(_, s, _)| s == 1.0).count(), 2);
        // Two rider hits do not kill a full swordsman: shortfall is honest.
        let sword_hp = risks[0].need_damage;
        assert!(plan.shortfall > 0.0 && plan.shortfall < sword_hp);
    }

    /// EXP_ELO_096: the unit that closes the coverage gap gets a partial
    /// `credit_frac` for just its needed sliver, not the same full share as
    /// a unit that was needed outright — no cliff between "picked" and
    /// "not picked", and the share is priced off the unit's own damage
    /// output (health/attack/defense via `hypo_damage`), not a flat count.
    #[test]
    fn defend_credit_tapers_smoothly_instead_of_a_flat_per_unit_share() {
        let mut state = board(60);
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(61, UnitType::Warrior, 2));
        let t1 = state.tribes.get_mut(&1).unwrap();
        t1.units.push(unit_at(49, UnitType::Warrior, 1)); // weaker, priority order first (tile 49 < 59)
        t1.units.push(unit_at(59, UnitType::Giant, 1)); // stronger, priority order second
        let risks = city_risks(&state, 1);
        let plan = defend_plan(&state, 1, &risks[0], &[]);

        let weak = plan.assigned.iter().find(|&&(t, _, _)| t == 49).expect("weak defender assigned");
        let strong = plan.assigned.iter().find(|&&(t, _, _)| t == 59).expect("strong defender assigned");
        assert_eq!(weak.2, 1.0, "first-in-priority unit didn't close the gap alone: fully needed");
        assert!(
            strong.2 > 0.0 && strong.2 < 1.0,
            "second unit should only be partially needed, not a flat 0-or-1 share: {:?}",
            strong
        );
        // The pair together essentially close the gap: no artificial cliff
        // stranded either a real shortfall or a wasted overshoot.
        assert!(plan.shortfall < 1.0, "shortfall {}", plan.shortfall);
    }

    #[test]
    fn hold_needed_only_when_garrison_is_load_bearing() {
        // Garrison alone vs a swordsman: it is the whole plan.
        let mut state = board(60);
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
        state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
        let r = city_risks(&state, 1);
        assert!(defend_plan(&state, 1, &r[0], &[]).hold_needed);
        // Add enough outside cover to meet the kill damage without it.
        for idx in [38, 82, 48, 72] {
            state.tribes.get_mut(&1).unwrap().units.push(unit_at(idx, UnitType::Swordsman, 1));
        }
        let r = city_risks(&state, 1);
        let plan = defend_plan(&state, 1, &r[0], &[]);
        assert!(plan.shortfall == 0.0);
        assert!(!plan.hold_needed);
    }

    /// Verdi (Aug 2026): `must_kill`'s break condition is `RISK_MARGIN`
    /// (0.8), the same bar `at_risk` already uses ("one hit leaving the
    /// garrison below a second hit is a threat, not a nuisance") -- not
    /// bare survival. A Warrior's hit on a full-health Rider garrison (5
    /// dmg vs 10hp) sits below that margin and needs no kill; a
    /// Swordsman's (9 dmg) sits at/above it and does. Both sides of the
    /// boundary, on the same garrison, confirmed via `city_risks`' own
    /// `strike` field rather than assumed (see this fix's commit history:
    /// a naive bare-survival version broke three existing defense-pricing
    /// tests by inverting hold-vs-vacate incentives).
    #[test]
    fn must_kill_break_uses_the_risk_margin_not_bare_survival() {
        let below = {
            let mut state = board(60);
            state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
            state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Warrior, 2));
            let r = city_risks(&state, 1);
            assert!(r[0].strike < 0.8 * 10.0, "fixture sanity: must sit below the margin");
            defend_plan(&state, 1, &r[0], &[])
        };
        assert_eq!(below.shortfall, 0.0, "a strike below the margin needs no kill: {below:?}");
        assert!(!below.hold_needed, "nothing is being asked of the garrison here: {below:?}");

        let above = {
            let mut state = board(60);
            state.tribes.get_mut(&1).unwrap().units.push(unit_at(60, UnitType::Rider, 1));
            state.tribes.get_mut(&2).unwrap().units.push(unit_at(59, UnitType::Swordsman, 2));
            let r = city_risks(&state, 1);
            assert!(r[0].strike >= 0.8 * 10.0, "fixture sanity: must sit at/above the margin");
            defend_plan(&state, 1, &r[0], &[])
        };
        assert!(above.hold_needed, "a strike at/above the margin must be treated as load-bearing: {above:?}");
    }

    /// The confirmed EXP_ELO seed-020 regression, reproduced directly: a
    /// 6hp garrison facing four attackers whose combined strike (16) far
    /// exceeds it must NOT report shortfall=0 just because covering the
    /// single strongest attacker's own health (10) is affordable -- the
    /// other three still land unopposed. `need_damage` must reflect enough
    /// eliminated attackers to bring total incoming back under the
    /// garrison's health, not one attacker's raw health.
    #[test]
    fn shortfall_reflects_cumulative_multi_attacker_damage_not_one_attackers_health() {
        let mut state = board(49);
        let mut garrison = unit_at(49, UnitType::Warrior, 1);
        garrison.health = 6.0;
        let garrison_hp = garrison.health;
        state.tribes.get_mut(&1).unwrap().units.push(garrison);
        // A lone unit that could cover the garrison's own retaliation, but
        // nowhere near enough against four attackers at once.
        state.tribes.get_mut(&1).unwrap().units.push(unit_at(39, UnitType::Warrior, 1));
        for idx in [37, 38, 48, 60] {
            state.tribes.get_mut(&2).unwrap().units.push(unit_at(idx, UnitType::Warrior, 2));
        }
        let r = city_risks(&state, 1);
        let risk = r.iter().find(|c| c.city == 49).expect("city 49 must be at risk");
        assert!(risk.attackers.len() >= 3, "fixture must present multiple attackers: {risk:?}");
        assert!(
            risk.strike > garrison_hp,
            "fixture sanity: combined strike must exceed garrison hp"
        );
        let plan = defend_plan(&state, 1, risk, &[]);
        assert!(
            plan.shortfall > 0.0,
            "four attackers summing well past the garrison's hp must not read as fully covered: {plan:?}"
        );
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
        let r = city_risks(&state, 1);
        let plan = defend_plan(&state, 1, &r[0], &[79]);
        assert!(!plan.assigned.iter().any(|&(t, _, _)| t == 79));
        assert!(plan.assigned.iter().any(|&(t, _, _)| t == 48));
    }
}
