//! EXP_ELO_034: belief state over FOW-hidden information, built from legal
//! observables only — public score deltas (the in-game leaderboard), the
//! observer's own explored tiles, and the observer's ghost records.
//! `BeliefState` never touches true state; `CalibHarness` is the single
//! component allowed to read it, to feed observables and to log ground truth
//! for offline calibration.

use crate::moves::Move;
use crate::states::{GameState, PlayerId};
use crate::types::MoveType;

/// Exact support of capital placement per quadrant, replicating the mapgen
/// quadrant path (Drylands/Lakes/Archipelago/WaterWorld). Indexed by quad.
/// Verified against the generator by `capital_support_matches_generator_*`.
pub fn capital_support_by_quad(size: i32, player_count: usize) -> Vec<Vec<i32>> {
    let quad_count: i32 = if player_count <= 4 {
        4
    } else if player_count <= 9 {
        9
    } else {
        16
    };
    let quads_per_side = (quad_count as f32).sqrt() as i32;
    let quad_size = size / quads_per_side;
    let margin = 2;
    let mut quads = Vec::with_capacity(quad_count as usize);
    for quad in 0..quad_count {
        let qx = quad % quads_per_side;
        let qy = quad / quads_per_side;
        let start_x = (qx * quad_size + margin).min(size - 3);
        let end_x = ((qx + 1) * quad_size - margin)
            .max(start_x + 1)
            .min(size - 2);
        let start_y = (qy * quad_size + margin).min(size - 3);
        let end_y = ((qy + 1) * quad_size - margin)
            .max(start_y + 1)
            .min(size - 2);
        let mut cells = Vec::new();
        for y in start_y..end_y {
            for x in start_x..end_x {
                cells.push(y * size + x);
            }
        }
        quads.push(cells);
    }
    quads
}

/// Which quadrant's support box contains `idx`, if any.
pub fn quad_of(idx: i32, support: &[Vec<i32>]) -> Option<usize> {
    support.iter().position(|cells| cells.contains(&idx))
}

/// Opponent-capital prior for an observer whose own capital sits at
/// `own_capital`: uniform over every support cell outside the observer's
/// quadrant (mapgen removes the taken quadrant before placing the next).
pub fn opponent_capital_prior(
    size: i32,
    player_count: usize,
    own_capital: i32,
) -> Vec<(i32, f32)> {
    let support = capital_support_by_quad(size, player_count);
    let own_quad = quad_of(own_capital, &support);
    let cells: Vec<i32> = support
        .iter()
        .enumerate()
        .filter(|(q, _)| Some(*q) != own_quad)
        .flat_map(|(_, c)| c.iter().copied())
        .collect();
    let p = 1.0 / cells.len().max(1) as f32;
    cells.into_iter().map(|c| (c, p)).collect()
}

/// Unit costs that exist in the game; a +5k residual whose k is not one of
/// these cannot be a hidden unit build.
const PLAUSIBLE_UNIT_COSTS: [i32; 6] = [2, 3, 4, 5, 8, 10];

/// One observed opponent score change. `witnessed` deltas are fully explained
/// by visible events and carry no hidden information.
#[derive(Debug, Clone)]
pub struct ScoreEvent {
    pub turn: i32,
    pub delta: i32,
    pub witnessed: bool,
    pub ghost_departure: bool,
}

#[derive(Debug, Clone)]
pub struct BeliefState {
    pub observer: PlayerId,
    pub opponent: PlayerId,
    /// (cell, probability) over the generator support; empty only if every
    /// hypothesis was eliminated without a sighting (guarded, shouldn't occur).
    pub capital_posterior: Vec<(i32, f32)>,
    pub capital_confirmed: Option<i32>,
    /// Stars of hidden units inferred from unwitnessed build-like deltas
    /// (never includes ghost-tracked units — those are known, not inferred).
    pub residual_army_stars: f32,
    /// Hidden city captures inferred from capture-signature deltas.
    pub hidden_cities: f32,
    /// Hidden tech purchases inferred from +100·tier deltas.
    pub hidden_techs: f32,
    /// Evidence count behind `residual_army_stars`, drives tanh confidence.
    build_signals: f32,
    last_signal_turn: i32,
    pub events: Vec<ScoreEvent>,
}

impl BeliefState {
    pub fn new(
        size: i32,
        player_count: usize,
        own_capital: i32,
        observer: PlayerId,
        opponent: PlayerId,
    ) -> Self {
        Self {
            observer,
            opponent,
            capital_posterior: opponent_capital_prior(size, player_count, own_capital),
            capital_confirmed: None,
            residual_army_stars: 0.0,
            hidden_cities: 0.0,
            hidden_techs: 0.0,
            build_signals: 0.0,
            last_signal_turn: 0,
            events: Vec::new(),
        }
    }

    /// Tiles newly explored by the observer. `capital_seen_at` is the
    /// opponent capital if one of those tiles holds it (observer-visible).
    pub fn on_explored(&mut self, newly_explored: &[i32], capital_seen_at: Option<i32>) {
        if let Some(idx) = capital_seen_at {
            self.capital_posterior = vec![(idx, 1.0)];
            self.capital_confirmed = Some(idx);
            return;
        }
        if self.capital_confirmed.is_some() {
            return;
        }
        self.capital_posterior
            .retain(|(c, _)| !newly_explored.contains(c));
        let total: f32 = self.capital_posterior.iter().map(|(_, p)| p).sum();
        if total > 0.0 {
            for (_, p) in self.capital_posterior.iter_mut() {
                *p /= total;
            }
        }
    }

    /// An opponent score change. Witnessed deltas are recorded but carry no
    /// hidden information. Unwitnessed deltas are attributed by signature:
    /// exploration (+5k, esp. with a co-occurring ghost departure — a scout
    /// the observer saw walk into fog), unit build (+5·cost), tech
    /// (+100·tier), city capture (+100 + 20·territory + 5·pop).
    pub fn on_opponent_delta(
        &mut self,
        turn: i32,
        delta: i32,
        witnessed: bool,
        ghost_departure: bool,
    ) {
        self.events.push(ScoreEvent { turn, delta, witnessed, ghost_departure });
        if witnessed || delta == 0 {
            return;
        }
        if delta < 0 {
            // Hidden loss (disband / drowned): shrink the believed pool.
            self.residual_army_stars =
                (self.residual_army_stars + delta as f32 / 5.0).max(0.0);
            return;
        }
        // A unit the observer just watched leave into fog explains a small
        // +5k delta as its exploration — not hidden production.
        if ghost_departure && delta % 5 == 0 && delta <= 40 {
            return;
        }
        if delta % 100 == 0 && delta <= 300 {
            // 100 → tech tier 1 (temples are rare that early); 200/300 are
            // ambiguous with capture signatures — split the mass.
            let tiers = (delta / 100) as f32;
            if delta == 100 {
                self.hidden_techs += 1.0;
            } else {
                self.hidden_techs += 0.5;
                self.hidden_cities += 0.5;
                let _ = tiers;
            }
            self.last_signal_turn = turn;
            return;
        }
        if delta >= 150 {
            // Capture signature: 100 + 20·territory + 5·pop.
            self.hidden_cities += 1.0;
            self.last_signal_turn = turn;
            return;
        }
        if delta % 5 == 0 && delta >= 10 {
            let k = delta / 5;
            let is_cost = PLAUSIBLE_UNIT_COSTS.contains(&k);
            // Build vs hidden-unit exploration split. k>6 can't be one move's
            // exploration, so a valid cost there is almost surely a build.
            let w_build = if is_cost && k > 6 {
                0.8
            } else if is_cost {
                0.5
            } else {
                0.0
            };
            if w_build > 0.0 {
                self.residual_army_stars += w_build * k as f32;
                self.build_signals += w_build;
                self.last_signal_turn = turn;
            }
        }
        // delta == 5 or non-signature values: exploration / noise, ignore.
    }

    /// Previously-hidden opponent stars entered the observer's vision; the
    /// inferred pool shrinks by what materialized.
    pub fn on_emerged(&mut self, stars: f32) {
        self.residual_army_stars = (self.residual_army_stars - stars).max(0.0);
    }

    /// Total believed hidden army: inferred pool + known units currently out
    /// of sight (`ghost_stars` from the observer's own ghost records).
    pub fn believed_hidden_army(&self, ghost_stars: f32) -> f32 {
        self.residual_army_stars + ghost_stars
    }

    /// tanh-bounded confidence in the hidden-army estimate; decays while no
    /// new signal arrives.
    pub fn army_confidence(&self, turn: i32) -> f32 {
        let staleness = 0.92f32.powi((turn - self.last_signal_turn).max(0));
        (self.build_signals / 3.0).tanh() * staleness
    }

    /// Confidence in the capital estimate = mass on the MAP cell.
    pub fn capital_confidence(&self) -> f32 {
        self.capital_posterior
            .iter()
            .map(|(_, p)| *p)
            .fold(0.0, f32::max)
    }

    /// Posterior cells, most probable first.
    pub fn capital_top(&self, n: usize) -> Vec<(i32, f32)> {
        let mut v = self.capital_posterior.clone();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(n);
        v
    }
}

/// Tile indices a move touches, from its serialized form (`src`/`target`).
/// Tile-less moves (Research, EndTurn) return empty — and are therefore
/// always unwitnessed when made by the opponent.
pub fn involved_tiles(m: &dyn Move) -> Vec<i32> {
    let ser = m.serialize();
    let mut tiles = Vec::new();
    for key in ["src", "target"] {
        if let Some(v) = ser.get(key).and_then(|v| v.as_i64()) {
            tiles.push(v as i32);
        }
    }
    tiles
}

// ---------------------------------------------------------------------------
// Calibration harness (arena-side). Reads TRUE state — only to feed each
// observer its legal observables and to log ground truth next to the belief.
// ---------------------------------------------------------------------------

struct ObserverTrack {
    belief: BeliefState,
    explored: std::collections::HashSet<i32>,
    ghost_keys: std::collections::HashSet<i32>,
    visible_stars: f32,
    opp_score: i32,
}

pub struct CalibHarness {
    tracks: Vec<ObserverTrack>,
    pub rows: Vec<serde_json::Value>,
}

fn explored_set(state: &GameState, pov: PlayerId) -> std::collections::HashSet<i32> {
    state
        .tiles
        .iter()
        .filter(|(_, t)| t.explorers.contains(&pov))
        .map(|(&i, _)| i)
        .collect()
}

fn unit_stars(u: &crate::states::UnitState) -> f32 {
    if u.converted {
        return 0.0;
    }
    let cost = crate::settings::units::get_unit_setting(u.unit_type).cost
        + u.passenger_type
            .map(|p| crate::settings::units::get_unit_setting(p).cost)
            .unwrap_or(0);
    cost as f32
}

/// Opponent stars on observer-explored tiles (what the observer can see).
fn visible_opp_stars(
    state: &GameState,
    explored: &std::collections::HashSet<i32>,
    opp: PlayerId,
) -> f32 {
    state
        .tribes
        .get(&opp)
        .map(|t| {
            t.units
                .iter()
                .filter(|u| explored.contains(&u.coords.idx))
                .filter(|u| !u.effects.contains(&crate::types::UnitEffect::Invisible))
                .map(unit_stars)
                .sum()
        })
        .unwrap_or(0.0)
}

fn ghost_stars(state: &GameState, observer: PlayerId, opp: PlayerId) -> f32 {
    state
        .tribes
        .get(&observer)
        .map(|t| {
            t.enemy_ghosts
                .values()
                .filter(|g| g.owner == opp)
                .map(|g| crate::settings::units::get_unit_setting(g.unit_type).cost as f32)
                .sum()
        })
        .unwrap_or(0.0)
}

impl CalibHarness {
    /// Build one belief per seat. Call after `post_load` (capitals stamped).
    pub fn new(state: &GameState) -> Self {
        let size = state.settings.size;
        let players: Vec<PlayerId> = {
            let mut p: Vec<PlayerId> = state.tribes.keys().copied().collect();
            p.sort_unstable();
            p
        };
        let capital_of = |pid: PlayerId| -> i32 {
            state
                .tiles
                .iter()
                .find(|(_, t)| t.capital_of == pid)
                .map(|(&i, _)| i)
                .unwrap_or(-1)
        };
        let tracks = players
            .iter()
            .map(|&obs| {
                let opp = players.iter().copied().find(|&p| p != obs).unwrap_or(obs);
                let explored = explored_set(state, obs);
                ObserverTrack {
                    belief: BeliefState::new(size, players.len(), capital_of(obs), obs, opp),
                    ghost_keys: state
                        .tribes
                        .get(&obs)
                        .map(|t| t.enemy_ghosts.keys().copied().collect())
                        .unwrap_or_default(),
                    visible_stars: visible_opp_stars(state, &explored, opp),
                    opp_score: crate::functions::calculate_detailed_tribe_score(state, opp),
                    explored,
                }
            })
            .collect();
        Self { tracks, rows: Vec::new() }
    }

    /// Feed both observers everything one true move changed. `pre_tiles`
    /// visibility must be judged against pre-move exploration, so this runs
    /// on the post-move state with the pre-move snapshots held in `self`.
    pub fn after_move(&mut self, state: &GameState, mover: PlayerId, m: &dyn Move) {
        let turn = state.settings.turn;
        let tiles = involved_tiles(m);
        for track in &mut self.tracks {
            let obs = track.belief.observer;
            let opp = track.belief.opponent;

            let explored_now = explored_set(state, obs);
            let newly: Vec<i32> = explored_now
                .difference(&track.explored)
                .copied()
                .collect();
            if !newly.is_empty() {
                let capital_seen = newly
                    .iter()
                    .copied()
                    .find(|&i| state.tiles.get(&i).map(|t| t.capital_of) == Some(opp));
                track.belief.on_explored(&newly, capital_seen);
            }

            let ghost_keys_now: std::collections::HashSet<i32> = state
                .tribes
                .get(&obs)
                .map(|t| t.enemy_ghosts.keys().copied().collect())
                .unwrap_or_default();
            let new_ghosts: Vec<i32> = ghost_keys_now
                .difference(&track.ghost_keys)
                .copied()
                .collect();

            let score_now = crate::functions::calculate_detailed_tribe_score(state, opp);
            let delta = score_now - track.opp_score;
            if delta != 0 {
                let witnessed = mover == obs
                    || (!tiles.is_empty()
                        && tiles.iter().all(|t| track.explored.contains(t)));
                track
                    .belief
                    .on_opponent_delta(turn, delta, witnessed, !new_ghosts.is_empty());
            }

            // Emergence: hidden stars that just became visible. Witnessed
            // builds and fog departures are backed out; deaths self-clamp.
            let visible_now = visible_opp_stars(state, &explored_now, opp);
            let build_stars = if mover == opp && m.move_type() == MoveType::Summon {
                let target_visible = m
                    .target_idx()
                    .map(|t| track.explored.contains(&(t as i32)))
                    .unwrap_or(false);
                if target_visible {
                    m.unit_type()
                        .map(|u| crate::settings::units::get_unit_setting(u).cost as f32)
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let departed: f32 = new_ghosts
                .iter()
                .filter_map(|i| state.tribes.get(&obs).and_then(|t| t.enemy_ghosts.get(i)))
                .filter(|g| g.owner == opp)
                .map(|g| crate::settings::units::get_unit_setting(g.unit_type).cost as f32)
                .sum();
            let emerged = visible_now - track.visible_stars - build_stars + departed;
            if emerged > 0.0 {
                track.belief.on_emerged(emerged);
            }

            track.explored = explored_now;
            track.ghost_keys = ghost_keys_now;
            track.visible_stars = visible_now;
            track.opp_score = score_now;
        }
    }

    /// Log one belief-vs-truth row for `observer` (call at the start of the
    /// observer's own turn — the moment a planner would consume the belief).
    pub fn turn_row(&mut self, state: &GameState, observer: PlayerId) {
        let Some(track) = self
            .tracks
            .iter()
            .find(|t| t.belief.observer == observer)
        else {
            return;
        };
        let b = &track.belief;
        let opp = b.opponent;
        let turn = state.settings.turn;

        let truth_capital = state
            .tiles
            .iter()
            .find(|(_, t)| t.capital_of == opp)
            .map(|(&i, _)| i)
            .unwrap_or(-1);
        let top = b.capital_top(3);
        let truth_p = b
            .capital_posterior
            .iter()
            .find(|(c, _)| *c == truth_capital)
            .map(|(_, p)| *p)
            .unwrap_or(0.0);
        let unexplored = state.settings.size * state.settings.size
            - track.explored.len() as i32;

        // Baseline B: the repo's existing corner heuristic (opposite corner,
        // radius 3, unexplored only) — replicated from predict_enemy_capitals.
        let corner_hit = {
            let size = state.settings.size;
            let own_cap = state
                .tiles
                .iter()
                .find(|(_, t)| t.capital_of == observer)
                .map(|(&i, _)| i)
                .unwrap_or(0);
            let (px, py) = (own_cap % size, own_cap / size);
            let tx = if px < size / 2 { size - 1 } else { 0 };
            let ty = if py < size / 2 { size - 1 } else { 0 };
            let (cx, cy) = (truth_capital % size, truth_capital / size);
            (cx - tx).abs() <= 3
                && (cy - ty).abs() <= 3
                && !track.explored.contains(&truth_capital)
        };

        let truth_hidden_army: f32 = state
            .tribes
            .get(&opp)
            .map(|t| {
                t.units
                    .iter()
                    .filter(|u| !track.explored.contains(&u.coords.idx))
                    .map(unit_stars)
                    .sum()
            })
            .unwrap_or(0.0);
        let truth_hidden_cities = state
            .tribes
            .get(&opp)
            .map(|t| {
                t.cities
                    .iter()
                    .filter(|c| !track.explored.contains(&c.idx))
                    .count()
            })
            .unwrap_or(0);
        let truth_techs = state
            .tribes
            .get(&opp)
            .map(|t| t.tech_vanilla.iter().filter(|x| x.discovered_turn > 0).count())
            .unwrap_or(0);

        let g_stars = ghost_stars(state, observer, opp);
        let (w, u) = b
            .events
            .iter()
            .fold((0u32, 0u32), |(w, u), e| if e.witnessed { (w + 1, u) } else { (w, u + 1) });

        self.rows.push(serde_json::json!({
            "turn": turn,
            "observer": observer,
            "cap_top": top.iter().map(|(c, p)| serde_json::json!([c, p])).collect::<Vec<_>>(),
            "cap_top1_hit": top.first().map(|(c, _)| *c == truth_capital).unwrap_or(false),
            "cap_truth_p": truth_p,
            "cap_live": b.capital_posterior.iter().filter(|(_, p)| *p > 1e-6).count(),
            "cap_confirmed": b.capital_confirmed,
            "cap_conf": b.capital_confidence(),
            "cap_truth": truth_capital,
            "baseline_uniform_p": 1.0 / (unexplored.max(1) as f32),
            "baseline_corner_hit": corner_hit,
            "believed_hidden_army": b.believed_hidden_army(g_stars),
            "residual_army": b.residual_army_stars,
            "ghost_stars": g_stars,
            "army_conf": b.army_confidence(turn),
            "truth_hidden_army": truth_hidden_army,
            "hidden_cities_believed": b.hidden_cities,
            "truth_hidden_cities": truth_hidden_cities,
            "hidden_techs_believed": b.hidden_techs,
            "truth_techs_bought": truth_techs,
            "events_witnessed": w,
            "events_unwitnessed": u,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXP_ELO_034 P2 guardrail, checked before anything was built on it: on
    /// the arena's exact map settings (Tiny Drylands 1v1) the generator can
    /// only place capitals inside the quadrant support boxes.
    #[test]
    fn capital_support_matches_generator_tiny_drylands() {
        let support = capital_support_by_quad(11, 2);
        let flat: std::collections::HashSet<i32> =
            support.iter().flatten().copied().collect();
        let mut seen: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
        let mut quad_pairs_differ = true;
        for i in 0..100i64 {
            let state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
                size: crate::types::MapSize::Tiny,
                map_type: crate::types::MapType::Drylands,
                tribes: vec![
                    crate::types::TribeType::Imperius,
                    crate::types::TribeType::Imperius,
                ],
                seed: 9_100_000 + i,
                ..Default::default()
            });
            let caps: Vec<i32> = state
                .tiles
                .iter()
                .filter(|(_, t)| t.capital_of > 0)
                .map(|(&idx, _)| idx)
                .collect();
            assert_eq!(caps.len(), 2, "seed {i}: expected 2 capitals, got {caps:?}");
            for &c in &caps {
                assert!(
                    flat.contains(&c),
                    "seed {i}: capital {c} outside support {flat:?}"
                );
                *seen.entry(c).or_insert(0) += 1;
            }
            if quad_of(caps[0], &support) == quad_of(caps[1], &support) {
                quad_pairs_differ = false;
            }
        }
        assert!(quad_pairs_differ, "two capitals shared a quadrant");
        assert!(
            seen.len() >= 4,
            "generator never used all support cells: {seen:?}"
        );
        eprintln!("capital support hit distribution: {seen:?}");
    }

    #[test]
    fn prior_excludes_own_quadrant_and_sums_to_one() {
        let prior = opponent_capital_prior(11, 2, 24);
        assert!(!prior.iter().any(|(c, _)| *c == 24));
        let total: f32 = prior.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-5, "prior mass {total}");
    }

    #[test]
    fn exploration_eliminates_and_sighting_collapses() {
        let mut b = BeliefState::new(11, 2, 24, 1, 2);
        assert_eq!(b.capital_posterior.len(), 3);
        b.on_explored(&[29], None);
        assert_eq!(b.capital_posterior.len(), 2);
        for (_, p) in &b.capital_posterior {
            assert!((*p - 0.5).abs() < 1e-5);
        }
        b.on_explored(&[79], Some(79));
        assert_eq!(b.capital_confirmed, Some(79));
        assert!((b.capital_confidence() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn tileless_tech_delta_is_attributed_not_dropped() {
        let mut b = BeliefState::new(11, 2, 24, 1, 2);
        // Research is tile-less: the harness classifies it unwitnessed.
        b.on_opponent_delta(3, 100, false, false);
        assert!((b.hidden_techs - 1.0).abs() < 1e-5);
        assert_eq!(b.residual_army_stars, 0.0);
    }

    #[test]
    fn ghost_departure_explains_small_exploration_delta() {
        let mut b = BeliefState::new(11, 2, 24, 1, 2);
        b.on_opponent_delta(4, 10, false, true);
        assert_eq!(b.residual_army_stars, 0.0, "scout exploration misread as build");
        b.on_opponent_delta(5, 10, false, false);
        assert!((b.residual_army_stars - 1.0).abs() < 1e-5, "0.5 × cost 2");
    }

    #[test]
    fn emergence_shrinks_pool_and_floors_at_zero() {
        let mut b = BeliefState::new(11, 2, 24, 1, 2);
        b.on_opponent_delta(5, 50, false, false); // 0.8 × 10 = 8 stars
        assert!((b.residual_army_stars - 8.0).abs() < 1e-5);
        b.on_emerged(10.0);
        assert_eq!(b.residual_army_stars, 0.0);
    }

    #[test]
    fn capture_signature_increments_hidden_cities() {
        let mut b = BeliefState::new(11, 2, 24, 1, 2);
        b.on_opponent_delta(9, 280, false, false);
        assert!((b.hidden_cities - 1.0).abs() < 1e-5);
        b.on_opponent_delta(10, 200, false, false); // ambiguous with tier-2 tech
        assert!((b.hidden_cities - 1.5).abs() < 1e-5);
        assert!((b.hidden_techs - 0.5).abs() < 1e-5);
    }

    #[test]
    fn witnessed_deltas_carry_no_inference() {
        let mut b = BeliefState::new(11, 2, 24, 1, 2);
        b.on_opponent_delta(5, 100, true, false);
        b.on_opponent_delta(5, 25, true, false);
        assert_eq!(b.hidden_techs, 0.0);
        assert_eq!(b.residual_army_stars, 0.0);
    }

    #[test]
    fn confidence_is_bounded_and_decays() {
        let mut b = BeliefState::new(11, 2, 24, 1, 2);
        for t in 0..20 {
            b.on_opponent_delta(t, 25, false, false);
        }
        let fresh = b.army_confidence(19);
        assert!(fresh > 0.5 && fresh < 1.0);
        assert!(b.army_confidence(30) < fresh * 0.5, "staleness must decay");
    }
}
