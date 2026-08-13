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

/// Result of writing a belief into a planning view (EXP_ELO_035 telemetry).
#[derive(Debug, Clone, Copy, Default)]
pub struct MaterializeStats {
    pub capital: bool,
    pub ghost_units: u32,
    pub residual_units: u32,
}

fn explored_by(state: &GameState, idx: i32, pov: PlayerId) -> bool {
    state
        .tiles
        .get(&idx)
        .map(|t| t.explorers.contains(&pov))
        .unwrap_or(true)
}

fn land_spawnable(state: &GameState, idx: i32, ty: crate::types::UnitType) -> bool {
    use crate::types::{SkillType, TerrainType};
    let terrain_ok = matches!(
        state.tiles.get(&idx).map(|t| t.terrain_type),
        Some(TerrainType::Field) | Some(TerrainType::Forest)
    );
    let unit_ok = {
        let skills = &crate::settings::units::get_unit_setting(ty).skills;
        !skills.contains(&SkillType::Float) && !skills.contains(&SkillType::Carry)
    };
    let free = state
        .tiles
        .get(&idx)
        .map(|t| t._unit_owner_id.is_none())
        .unwrap_or(false);
    terrain_ok && unit_ok && free
}

/// EXP_ELO_035: write the belief's MAP world into a fogged planning view — a
/// throwaway clone, never the true game. Adds the believed capital city at
/// the posterior peak (unless already sighted), ghost units at their
/// recorded tiles, and the inferred residual army as warriors near the
/// capital hypothesis. Also REPLACES the opponent's star bank with a crude
/// public-info ramp: `obscure_fog` leaks the true bank, and a materialized
/// spend site would otherwise let rollouts consume hidden information.
pub fn materialize_into(view: &mut crate::game::Game, belief: &BeliefState) -> MaterializeStats {
    let pov = belief.observer;
    let opp = belief.opponent;
    let mut stats = MaterializeStats::default();
    let turn = view.state.settings.turn;

    if let Some(t) = view.state.tribes.get_mut(&opp) {
        t.stars = (2 + turn / 2).min(20);
    }

    // Believed capital: engine-native creation via capture_city Case 2 with
    // the current player temporarily swapped to the opponent.
    let mut anchor: Option<i32> = belief.capital_confirmed;
    if anchor.is_none() {
        if let Some(&(cell, _)) = belief.capital_top(1).first() {
            let has_city = view
                .state
                .tribes
                .get(&opp)
                .map(|t| t.cities.iter().any(|c| c.idx == cell))
                .unwrap_or(false);
            if !explored_by(&view.state, cell, pov) && !has_city {
                if let Some(tile) = view.state.tiles.get_mut(&cell) {
                    // Generator invariant: support cells are Field capitals.
                    tile.terrain_type = crate::types::TerrainType::Field;
                    tile.owner = 0;
                    tile._unit_owner_id = None;
                }
                let saved = view.state.settings.current_player_turn_id;
                view.state.settings.current_player_turn_id = opp;
                let ok = matches!(
                    crate::actions::city::capture_city(&mut view.state, cell),
                    Ok(_undo)
                );
                view.state.settings.current_player_turn_id = saved;
                if ok {
                    if let Some(tile) = view.state.tiles.get_mut(&cell) {
                        tile.capital_of = opp;
                    }
                    // Strip pov-visible tiles from the imagined border: the
                    // player KNOWS there is no enemy territory there.
                    let vis: Vec<i32> = view
                        .state
                        .tribes
                        .get(&opp)
                        .and_then(|t| t.cities.iter().find(|c| c.idx == cell))
                        .map(|c| {
                            c._territory
                                .iter()
                                .copied()
                                .filter(|&i| i != cell && explored_by(&view.state, i, pov))
                                .collect()
                        })
                        .unwrap_or_default();
                    for &i in &vis {
                        if let Some(t) = view.state.tiles.get_mut(&i) {
                            t.owner = 0;
                            t.ruling_city_coords = None;
                        }
                    }
                    if let Some(t) = view.state.tribes.get_mut(&opp) {
                        t.score -= crate::score::CITY_TERRITORY_SCORE * vis.len() as i32;
                        if let Some(c) = t.cities.iter_mut().find(|c| c.idx == cell) {
                            c._territory.retain(|i| !vis.contains(i));
                        }
                    }
                    stats.capital = true;
                    anchor = Some(cell);
                }
            }
        }
    }

    // Ghost units: known enemies last seen entering fog, at their tiles.
    // Naval ghosts are skipped (land guard) — v1 scope cut, in the ledger.
    let ghosts: Vec<(i32, crate::types::UnitType)> = view
        .state
        .tribes
        .get(&pov)
        .map(|t| {
            t.enemy_ghosts
                .iter()
                .filter(|(_, g)| g.owner == opp)
                .map(|(&i, g)| (i, g.unit_type))
                .collect()
        })
        .unwrap_or_default();
    for (idx, ty) in ghosts {
        if !explored_by(&view.state, idx, pov) && land_spawnable(&view.state, idx, ty) {
            let _ = crate::actions::units::spawn_unit(&mut view.state, opp, ty, idx, false);
            stats.ghost_units += 1;
        }
    }

    // Residual army: inferred never-seen production, as warriors ringed
    // around the capital hypothesis (they know where home is; we don't know
    // where they stand).
    if let Some(a) = anchor {
        let n = ((belief.residual_army_stars / 2.0).floor() as i32).clamp(0, 4);
        if n > 0 {
            let mut placed = 0u32;
            let mut candidates = vec![a];
            candidates.extend(crate::functions::get_square_indices(
                a,
                1,
                view.state.settings.size,
            ));
            candidates.extend(crate::functions::get_square_indices(
                a,
                2,
                view.state.settings.size,
            ));
            for idx in candidates {
                if placed >= n as u32 {
                    break;
                }
                if !explored_by(&view.state, idx, pov)
                    && land_spawnable(&view.state, idx, crate::types::UnitType::Warrior)
                {
                    let _ = crate::actions::units::spawn_unit(
                        &mut view.state,
                        opp,
                        crate::types::UnitType::Warrior,
                        idx,
                        false,
                    );
                    placed += 1;
                }
            }
            stats.residual_units = placed;
        }
    }
    stats
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
    /// Both ORIGINAL capital tiles, snapshotted at game start:
    /// `tile.capital_of` is reassigned to the capturer on capture
    /// (actions/city.rs), so live lookups break as truth after mid-game
    /// captures. Spawn location — the belief's target — never moves.
    opp_capital: i32,
    own_capital: i32,
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
                    opp_capital: capital_of(opp),
                    own_capital: capital_of(obs),
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
                // Sighting = exploring the spawn-capital tile (the city on it
                // is observer-visible at that moment, whoever holds it now).
                let capital_seen = newly
                    .iter()
                    .copied()
                    .find(|&i| i == track.opp_capital);
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

    /// The maintained belief for `observer` (EXP_ELO_035 consumers).
    pub fn belief_for(&self, observer: PlayerId) -> Option<&BeliefState> {
        self.tracks
            .iter()
            .map(|t| &t.belief)
            .find(|b| b.observer == observer)
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

        let truth_capital = track.opp_capital;
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
            let own_cap = track.own_capital.max(0);
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

    fn generated_game(seed: i64) -> crate::game::Game {
        let mut game = crate::game::Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: crate::types::MapSize::Tiny,
            map_type: crate::types::MapType::Drylands,
            tribes: vec![
                crate::types::TribeType::Imperius,
                crate::types::TribeType::Imperius,
            ],
            seed,
            ..Default::default()
        });
        game.post_load();
        game
    }

    fn own_capital(state: &GameState, pid: PlayerId) -> i32 {
        state
            .tiles
            .iter()
            .find(|(_, t)| t.capital_of == pid)
            .map(|(&i, _)| i)
            .unwrap()
    }

    #[test]
    fn materialize_adds_capital_ghost_and_residuals() {
        let mut game = generated_game(9_200_000);
        let own = own_capital(&game.state, 1);
        let mut b = BeliefState::new(11, 2, own, 1, 2);
        b.residual_army_stars = 6.0;
        let map_cell = b.capital_top(1)[0].0;
        // Plant a ghost the observer once saw departing next to the MAP cell.
        game.state.tribes.get_mut(&1).unwrap().enemy_ghosts.insert(
            map_cell + 1,
            crate::states::GhostRecord {
                unit_type: crate::types::UnitType::Warrior,
                owner: 2,
                turn: 0,
            },
        );
        let mut view = game.clone_for_mcts(1);
        // Deterministic spawnability for the ghost tile (fog terrain is
        // prediction-dependent).
        view.state.tiles.get_mut(&(map_cell + 1)).unwrap().terrain_type =
            crate::types::TerrainType::Field;
        let stats = materialize_into(&mut view, &b);
        assert!(stats.capital, "believed capital not materialized");
        assert_eq!(stats.ghost_units, 1, "ghost unit not spawned");
        assert!(stats.residual_units >= 1, "no residual warriors placed");
        let opp = view.state.tribes.get(&2).unwrap();
        let city = opp.cities.iter().find(|c| c.idx == map_cell).expect("city");
        assert_eq!(view.state.tiles.get(&map_cell).unwrap().capital_of, 2);
        // No imagined border on tiles the observer can see.
        for &t in &city._territory {
            assert!(
                !view.state.tiles.get(&t).unwrap().explorers.contains(&1),
                "imagined territory painted on a pov-visible tile {t}"
            );
        }
        assert!(opp.units.iter().any(|u| u.coords.idx == map_cell + 1));
    }

    /// EXP_ELO_036 rung 1: with a belief, the generator offers claim-safe /
    /// contest fog-expansion candidates; without one, the set is unchanged.
    #[test]
    fn belief_conditioned_candidates_offered_and_partitioned() {
        use crate::ai::macro_agent::{
            enumerate_candidates, enumerate_candidates_with_belief, CandidateClass,
        };
        use crate::ai::oracle_macro::{update_goal, StanceCommit};
        let mut found_belief_candidate = false;
        for seed in 0..8i64 {
            let mut game = generated_game(9_400_000 + seed);
            // A few scripted turns so exploration reveals evidence for
            // predict_villages (it needs explored orphan resources/climate).
            {
                use crate::ai::macro_exec;
                use crate::ai::oracle_macro::scripted_goal;
                let mut arch = crate::ai::oracle_macro::ArchetypeState::default();
                let mut counters = macro_exec::TurnCounters::default();
                for _ in 0..6 {
                    if game.state.settings._game_over {
                        break;
                    }
                    let player = game.state.settings.current_player_turn_id;
                    let goal = scripted_goal(&game.state, player, 0);
                    if !macro_exec::execute_turn(
                        &mut game, player, &goal, &mut arch, &mut counters, 1.0,
                    ) {
                        break;
                    }
                }
            }
            let pov = game.state.settings.current_player_turn_id;
            let opp = if pov == 1 { 2 } else { 1 };
            let own = own_capital(&game.state, pov);
            let b = BeliefState::new(11, 2, own, pov, opp);
            let view = game.clone_for_mcts(pov);
            let mut commit = StanceCommit::default();
            let base = update_goal(&view.state, pov, &mut commit, 0);
            let tagged = enumerate_candidates_with_belief(
                &view.state,
                pov,
                base.clone(),
                crate::ai::macro_exec::TurnCounters::default(),
                8,
                Some(&b),
            );
            let untagged = enumerate_candidates(
                &view.state,
                pov,
                base.clone(),
                crate::ai::macro_exec::TurnCounters::default(),
                8,
            );
            // Belief-blind path must be byte-identical to the old behavior.
            assert!(untagged.iter().all(|g| tagged.iter().any(|(t, _)| t == g)));
            let enemy_cap_guess = b.capital_top(1)[0].0;
            for (g, class) in &tagged {
                let mut sorted = g.orders.clone();
                sorted.sort();
                assert_eq!(sorted, g.orders, "orders must stay sorted");
                if matches!(class, CandidateClass::ClaimSafe | CandidateClass::Contest) {
                    found_belief_candidate = true;
                    for (_, t) in g.orders.iter().filter(|o| !base.orders.contains(o)) {
                        assert!(
                            !view
                                .state
                                .tiles
                                .get(t)
                                .map(|x| x.explorers.contains(&pov))
                                .unwrap_or(true),
                            "belief candidate target {t} is not a fog tile"
                        );
                        let w = view.state.settings.size;
                        let cheb = |a: i32, c: i32| {
                            ((a % w - c % w).abs()).max((a / w - c / w).abs())
                        };
                        assert!(
                            cheb(*t, enemy_cap_guess) > 1,
                            "target {t} sits on the believed capital"
                        );
                    }
                }
            }
        }
        assert!(
            found_belief_candidate,
            "no claim/contest candidate offered across 8 seeds — generator dead"
        );
    }

    #[test]
    fn confirmed_capital_skips_materialization() {
        let game = generated_game(9_200_001);
        let own = own_capital(&game.state, 1);
        let mut b = BeliefState::new(11, 2, own, 1, 2);
        b.on_explored(&[b.capital_top(1)[0].0], Some(b.capital_top(1)[0].0));
        let mut view = game.clone_for_mcts(1);
        let cities_before = view.state.tribes.get(&2).unwrap().cities.len();
        let stats = materialize_into(&mut view, &b);
        assert!(!stats.capital);
        assert_eq!(view.state.tribes.get(&2).unwrap().cities.len(), cities_before);
    }

    /// The novel engine surface is the OPPONENT acting at an imagined city
    /// (summons, border growth, income) — run several full simulated rounds
    /// on a materialized view and require clean execution.
    #[test]
    fn materialized_view_survives_simulated_rounds() {
        use crate::ai::macro_exec;
        use crate::ai::oracle_macro::{scripted_goal, ArchetypeState};
        for seed in 0..4i64 {
            let mut game = generated_game(9_210_000 + seed);
            let own = own_capital(&game.state, 1);
            let mut b = BeliefState::new(11, 2, own, 1, 2);
            b.residual_army_stars = 8.0;
            let mut sim = game.clone_for_mcts(1);
            let stats = materialize_into(&mut sim, &b);
            assert!(stats.capital, "seed {seed}: no capital materialized");
            let mut arch = ArchetypeState::default();
            let mut counters = macro_exec::TurnCounters::default();
            for _ in 0..8 {
                if sim.state.settings._game_over {
                    break;
                }
                let player = sim.state.settings.current_player_turn_id;
                let goal = scripted_goal(&sim.state, player, 0);
                if !macro_exec::execute_turn(&mut sim, player, &goal, &mut arch, &mut counters, 1.0)
                {
                    break;
                }
            }
            // True game must be untouched by everything above.
            assert!(game
                .state
                .tribes
                .get(&2)
                .unwrap()
                .cities
                .iter()
                .all(|c| c.idx != b.capital_top(1)[0].0
                    || own_capital(&game.state, 2) == c.idx));
            let _ = &mut game;
        }
    }

    /// EXP_ELO_036 rung-1 post-mortem: offer quality/timing. Per turn of 20
    /// scripted games: were claim/contest candidates OFFERED, how many
    /// safe/contested targets existed, and do predicted targets sit within
    /// 1 tile of a real village/city (truth query)? Distinguishes "tree
    /// declined good offers" from "offers were sparse, late, or wrong".
    #[test]
    #[ignore]
    fn fog_offer_quality_probe() {
        use crate::ai::macro_agent::{enumerate_candidates_with_belief, CandidateClass};
        use crate::ai::macro_exec;
        use crate::ai::oracle_macro::{scripted_goal, update_goal, ArchetypeState, StanceCommit};

        let mut per_turn: std::collections::BTreeMap<i32, (u32, u32, u32, u32, u32)> =
            std::collections::BTreeMap::new(); // (rows, claim_off, contest_off, targets, good_targets)
        for seed in 0..20i64 {
            let mut game = generated_game(9_500_000 + seed);
            game.state.settings.mode =
                crate::types::ModeType::from_repr(2).unwrap_or(crate::types::ModeType::Perfection);
            game.state.settings.max_turns = 30;
            let mut arch = ArchetypeState::default();
            let mut counters = macro_exec::TurnCounters::default();
            let mut commits = [StanceCommit::default(), StanceCommit::default()];
            let mut beliefs = [
                BeliefState::new(11, 2, own_capital(&game.state, 1), 1, 2),
                BeliefState::new(11, 2, own_capital(&game.state, 2), 2, 1),
            ];
            let mut prev_explored = [
                explored_set(&game.state, 1),
                explored_set(&game.state, 2),
            ];
            for _ in 0..60 {
                if game.state.settings._game_over {
                    break;
                }
                let pov = game.state.settings.current_player_turn_id;
                let seat = ((pov - 1) as usize).min(1);
                let opp: PlayerId = if pov == 1 { 2 } else { 1 };
                // Keep the capital posterior honest via exploration updates.
                let now = explored_set(&game.state, pov);
                let newly: Vec<i32> =
                    now.difference(&prev_explored[seat]).copied().collect();
                if !newly.is_empty() {
                    let opp_cap = game
                        .state
                        .tiles
                        .iter()
                        .find(|(_, t)| t.capital_of == opp)
                        .map(|(&i, _)| i);
                    let seen = opp_cap.filter(|c| newly.contains(c));
                    beliefs[seat].on_explored(&newly, seen);
                    prev_explored[seat] = now;
                }
                let view = game.clone_for_mcts(pov);
                let base = update_goal(&view.state, pov, &mut commits[seat], 0);
                let tagged = enumerate_candidates_with_belief(
                    &view.state,
                    pov,
                    base.clone(),
                    counters,
                    8,
                    Some(&beliefs[seat]),
                );
                let turn = game.state.settings.turn;
                let e = per_turn.entry(turn).or_default();
                e.0 += 1;
                for (g, class) in &tagged {
                    let is_claim = *class == CandidateClass::ClaimSafe;
                    let is_contest = *class == CandidateClass::Contest;
                    if is_claim {
                        e.1 += 1;
                    }
                    if is_contest {
                        e.2 += 1;
                    }
                    if is_claim || is_contest {
                        for (_, t) in g.orders.iter().filter(|o| !base.orders.contains(o)) {
                            e.3 += 1;
                            // Truth: a real village/city within Chebyshev 1.
                            let w = game.state.settings.size;
                            let good = (-1..=1).any(|dy| {
                                (-1..=1).any(|dx| {
                                    let x = t % w + dx;
                                    let y = t / w + dy;
                                    if x < 0 || x >= w || y < 0 || y >= w {
                                        return false;
                                    }
                                    let i = y * w + x;
                                    crate::functions::get_structure_type_at(&game.state, i)
                                        == Some(crate::types::StructureType::Village)
                                        || game
                                            .state
                                            .tribes
                                            .values()
                                            .any(|tr| tr.cities.iter().any(|c| c.idx == i))
                                })
                            });
                            if good {
                                e.4 += 1;
                            }
                        }
                    }
                }
                let player = pov;
                let goal = scripted_goal(&game.state, player, 0);
                if !macro_exec::execute_turn(&mut game, player, &goal, &mut arch, &mut counters, 1.0)
                {
                    break;
                }
            }
        }
        eprintln!("=== fog offer quality (20 games, per player-turn) ===");
        eprintln!("turn rows claim% contest% targets good%");
        for (t, (rows, cl, co, tg, gd)) in &per_turn {
            eprintln!(
                "  t{t:2} {rows:4} {:5.1} {:6.1} {tg:6} {:5.1}",
                100.0 * *cl as f32 / (*rows).max(1) as f32,
                100.0 * *co as f32 / (*rows).max(1) as f32,
                100.0 * *gd as f32 / (*tg).max(1) as f32,
            );
        }
    }

    /// EXP_ELO_034b: aux_fog offline calibration. Heavy probe, run manually:
    /// FOG_CALIB_MODEL=<pinned copy> cargo test --release --features apple \
    ///   --lib ai::belief -- --ignored aux_fog_calibration_probe --nocapture
    /// Games are play_move-driven (observation memory accumulates; the
    /// simulate path would starve the ghost channels the head trained on)
    /// and features copy self_play.rs:1694 exactly: TRUE state + pov +
    /// painted scripted goal — not a fogged clone.
    #[test]
    #[ignore]
    fn aux_fog_calibration_probe() {
        use crate::ai::features;
        use crate::ai::macro_agent::MacroScriptAgent;
        use crate::ai::network::PolyZeroNet;
        use crate::ai::oracle_macro::{update_goal, StanceCommit};
        use candle_core::{DType, Device};

        let model_path = std::env::var("FOG_CALIB_MODEL")
            .unwrap_or_else(|_| "model.safetensors".to_string());
        let device = Device::Cpu;
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)
        }
        .expect("model load");
        let net = PolyZeroNet::new(vb).expect("net build");
        assert!(net.has_fog_head(), "pinned model must carry aux_fog");

        // (pred, label) pools, partitioned; band pools for P2.
        let mut vis: Vec<(f32, f32)> = Vec::new();
        let mut fog: Vec<(f32, f32)> = Vec::new();
        let mut fog_by_turn: std::collections::BTreeMap<i32, Vec<(f32, f32)>> =
            std::collections::BTreeMap::new();
        let mut mass_bands: [Vec<(f32, f32)>; 3] = Default::default();
        let mut games_with_ghosts = 0usize;
        const GAMES: i64 = 40;

        for seed in 0..GAMES {
            let mut game = generated_game(9_300_000 + seed);
            game.state.settings.mode =
                crate::types::ModeType::from_repr(2).unwrap_or(crate::types::ModeType::Perfection);
            game.state.settings.max_turns = 30;
            let mut agents = [MacroScriptAgent::new(1.0), MacroScriptAgent::new(1.0)];
            let mut commits = [StanceCommit::default(), StanceCommit::default()];
            let mut tier3 = [0u32, 0u32];
            let mut last_key: Option<(i32, PlayerId)> = None;
            let mut ghost_seen = false;
            let mut moves = 0;
            while !crate::functions::is_game_over(&game.state) && moves < 500 {
                let pov = game.state.settings.current_player_turn_id;
                let seat = ((pov - 1) as usize).min(1);
                let key = (game.state.settings.turn, pov);
                if last_key != Some(key) {
                    last_key = Some(key);
                    let goal = update_goal(&game.state, pov, &mut commits[seat], tier3[seat]);
                    let feats = features::state_to_cpu_features_goal(
                        &game.state,
                        pov,
                        None,
                        Some(&goal),
                    )
                    .expect("features")
                    .into_game_features(&device)
                    .expect("tensors");
                    let (_, value) = net
                        .forward(&feats.spatial_map, &feats.player_state)
                        .expect("forward");
                    let probs: Vec<f32> = value
                        .fog_probs
                        .expect("fog head")
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap();
                    let opp: PlayerId = if pov == 1 { 2 } else { 1 };
                    let mut truth = vec![0.0f32; probs.len()];
                    if let Some(t) = game.state.tribes.get(&opp) {
                        for u in &t.units {
                            if !u.effects.contains(&crate::types::UnitEffect::Invisible) {
                                let i = u.coords.idx as usize;
                                if i < truth.len() {
                                    truth[i] = 1.0;
                                }
                            }
                        }
                    }
                    let turn = game.state.settings.turn;
                    let mut fog_mass = 0.0f32;
                    let mut hidden_count = 0.0f32;
                    for i in 0..probs.len() {
                        let explored = explored_by(&game.state, i as i32, pov);
                        if explored {
                            vis.push((probs[i], truth[i]));
                        } else {
                            fog.push((probs[i], truth[i]));
                            fog_by_turn.entry(turn).or_default().push((probs[i], truth[i]));
                            fog_mass += probs[i];
                            hidden_count += truth[i];
                        }
                    }
                    let band = match turn {
                        5..=10 => Some(0),
                        11..=17 => Some(1),
                        18..=25 => Some(2),
                        _ => None,
                    };
                    if let Some(bi) = band {
                        mass_bands[bi].push((fog_mass, hidden_count));
                    }
                    if game
                        .state
                        .tribes
                        .get(&pov)
                        .map(|t| !t.enemy_ghosts.is_empty())
                        .unwrap_or(false)
                    {
                        ghost_seen = true;
                    }
                }
                let Some(m) = agents[seat].select_move(&mut game) else { break };
                if m.move_type() == crate::types::MoveType::Research {
                    if let Ok(tech) = m.tech_type() {
                        if crate::settings::technology::get_technology_setting(tech).tier
                            == Some(3)
                        {
                            tier3[seat] += 1;
                        }
                    }
                }
                game.play_move(m.as_ref());
                moves += 1;
            }
            if ghost_seen {
                games_with_ghosts += 1;
            }
        }

        fn auc(pairs: &mut [(f32, f32)]) -> f32 {
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let n = pairs.len();
            let mut rank_sum_pos = 0.0f64;
            let mut pos = 0.0f64;
            let mut i = 0usize;
            while i < n {
                let mut j = i;
                while j + 1 < n && pairs[j + 1].0 == pairs[i].0 {
                    j += 1;
                }
                let avg_rank = ((i + 1 + j + 1) as f64) / 2.0;
                for p in pairs.iter().take(j + 1).skip(i) {
                    if p.1 > 0.5 {
                        rank_sum_pos += avg_rank;
                        pos += 1.0;
                    }
                }
                i = j + 1;
            }
            let neg = n as f64 - pos;
            if pos == 0.0 || neg == 0.0 {
                return f32::NAN;
            }
            ((rank_sum_pos - pos * (pos + 1.0) / 2.0) / (pos * neg)) as f32
        }

        fn pearson(pairs: &[(f32, f32)]) -> f32 {
            let n = pairs.len() as f64;
            if n < 3.0 {
                return f32::NAN;
            }
            let (mx, my) = pairs.iter().fold((0.0f64, 0.0f64), |(a, b), p| {
                (a + p.0 as f64, b + p.1 as f64)
            });
            let (mx, my) = (mx / n, my / n);
            let (mut sxy, mut sxx, mut syy) = (0.0f64, 0.0f64, 0.0f64);
            for p in pairs {
                let dx = p.0 as f64 - mx;
                let dy = p.1 as f64 - my;
                sxy += dx * dy;
                sxx += dx * dx;
                syy += dy * dy;
            }
            (sxy / (sxx.sqrt() * syy.sqrt()).max(1e-12)) as f32
        }

        let vis_auc = auc(&mut vis);
        let fog_auc = auc(&mut fog);
        eprintln!("=== EXP_ELO_034b aux_fog calibration ({GAMES} games) ===");
        eprintln!("P0 explored-tile AUC: {vis_auc:.3} (guardrail >= 0.90)");
        eprintln!("P1 fog-tile AUC:      {fog_auc:.3} (pass >= 0.70, falsified < 0.60)");
        for (bi, name) in ["t5-10", "t11-17", "t18-25"].iter().enumerate() {
            let r = pearson(&mass_bands[bi]);
            let n = mass_bands[bi].len();
            let mean_mass: f32 =
                mass_bands[bi].iter().map(|p| p.0).sum::<f32>() / n.max(1) as f32;
            let mean_true: f32 =
                mass_bands[bi].iter().map(|p| p.1).sum::<f32>() / n.max(1) as f32;
            eprintln!(
                "P2 {name}: r={r:.3} (n={n}, mean predicted mass {mean_mass:.2} vs true hidden {mean_true:.2})"
            );
        }
        eprintln!("per-turn fog AUC:");
        for (t, pairs) in fog_by_turn.iter_mut() {
            let a = auc(pairs);
            eprintln!("  t{t:2}: auc={a:.3} n={}", pairs.len());
        }
        eprintln!(
            "ghost tripwire: {games_with_ghosts}/{GAMES} games accumulated ghosts"
        );

        assert!(
            games_with_ghosts * 10 >= GAMES as usize * 3,
            "ghost tripwire: observation memory did not accumulate — instrument invalid"
        );
        assert!(
            vis_auc >= 0.90,
            "P0 instrument guardrail failed (explored AUC {vis_auc:.3}): feature parity bug, P1 void"
        );
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
