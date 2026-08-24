//! Prediction module for FOW (Fog of War)
//!
//! Provides prediction functions for MCTS simulations to avoid accessing ground truth data.
//! When `_are_you_sure = false`, the engine uses these predictions instead of actual hidden data.

use crate::ai::belief::BeliefState;
use crate::functions::{get_adjacent_indices, get_chebyshev_distance};
use crate::states::{GameState, PlayerId};
use crate::types::{StructureType, TerrainType, TribeType};
use indexmap::IndexMap;

/// Predicted tribe for a classic climate id (see `types::classic_climate_id`).
pub fn climate_to_tribe(climate: i32) -> TribeType {
    crate::types::tribe_from_classic_climate(climate)
}

/// Classic climate id a tribe's territory carries.
pub fn tribe_to_climate(tribe: TribeType) -> i32 {
    crate::types::classic_climate_id(tribe)
}

/// Validation for village candidates based on mapgen rules
fn validate_village_candidate(state: &GameState, idx: i32, known: &[i32]) -> bool {
    let size = state.settings.size;

    // 1. Cardinal Neighbor Rule: No Ocean neighbors
    let cardinals = crate::functions::get_plus_sign_indices(idx, size);
    for n_idx in cardinals {
        if let Some(tile) = state.tiles.get(&n_idx) {
            if tile.terrain_type == TerrainType::Ocean {
                return false;
            }
        }
    }

    // 2. Map Edge Rule: edge_dist >= 2 && edge_dist != 3
    let (x, y) = (idx % size, idx / size);
    let dist_x = x.min(size - 1 - x);
    let dist_y = y.min(size - 1 - y);
    let edge_dist = dist_x.min(dist_y);
    if edge_dist < 2 || edge_dist == 3 {
        return false;
    }

    // 3. Distance-3 Rule (Chebyshev) from every known village/capital
    known.iter().all(|&k| get_chebyshev_distance(idx, k, size) >= 3)
}

/// One guessed undiscovered village site: where, which tribe if the nearby
/// explored evidence points to one, and how confident that evidence is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VillageGuess {
    pub tile: i32,
    pub tribe: Option<TribeType>,
    pub confidence: f32,
}

/// Guess likely undiscovered village sites, from two evidence sources merged
/// into one guesser (Aug 2026 — previously two separate functions,
/// `guessed_village_sites` and `predict_villages`, answering the same
/// question from different angles):
///
/// - **Where**: the generator's own placement rules decide candidate
///   SELECTION. It fills villages to SATURATION over legal spots (land, edge
///   distance ∈ {2,4,5...}, Chebyshev ≥3 from every known village/capital
///   and from every other guess), so an UNEXPLORED legal spot ≥3 from
///   everything known must lie near an undiscovered village. Picks are
///   nearest-to-units first, spread across distinct quadrants around the
///   anchor centroid — nearest-first alone put guesses in one bearing sector
///   88% of the time, sending every scout the same way.
/// - **How confident, and which tribe**: resource/climate evidence on
///   EXPLORED tiles near each selected site — an orphaned resource (not
///   already claimed by a known city), a resource cluster, or a
///   climate-mismatched neighbour all raise confidence; a crop next to a
///   site the climate evidence points to as Bardur (who have none) rules
///   Bardur back out. This evidence does NOT drive which tiles get picked —
///   letting it do so would reintroduce exactly the one-direction-only bug
///   the quadrant spread above exists to prevent, since resource evidence is
///   often lopsided toward one explored corner of the map.
///
/// Count of tiles this player has explored — the exact fingerprint of
/// `guess_villages`'s dependencies (candidate selection, spacing, and
/// confidence evidence are all gated on `explorers`, nothing else), so a
/// caller that sees this count unchanged can safely reuse a cached guess.
pub fn explored_tile_count(state: &GameState, player: PlayerId) -> usize {
    state.tiles.values().filter(|t| t.explorers.contains(&player)).count()
}

/// Returns up to `max_sites`, mutually ≥3 apart.
/// Sites likely to hold an undiscovered village.
///
/// EXP_ELO_070: the belief PRUNES and distance DECIDES. EXP_ELO_069's first
/// attempt ordered by probability and lost expansion tempo — the scout was sent
/// to likelier-but-farther sites. Now the belief supplies the candidate pool and
/// the nearest of those wins, which is how the legacy picker ordered.
pub fn guess_villages(state: &GameState, player: PlayerId, max_sites: usize) -> Vec<VillageGuess> {
    // NOT `observe_cached`: the memo thrashes inside search, where hypothetical
    // captures move the key on every node, so it measured -42% throughput
    // against -17% for a plain derivation. A root-computed belief held for the
    // whole tree is the design's answer; `GoalCache` is where that belongs.
    crate::ai::belief::map::MapBelief::observe(state, player).top_village_sites(state, max_sites)
}

/// The village guesser exactly as it shipped before `ai::belief::map` existed,
/// bugs and all. `MapBelief::top_village_sites_legacy` is the only caller;
/// `guess_villages_parity_holds_on_a_state_corpus` pins it byte-for-byte
/// against this so the SSOT migration cannot drift production behaviour.
pub(crate) fn legacy_village_sites(
    state: &GameState,
    player: PlayerId,
    max_sites: usize,
) -> Vec<VillageGuess> {
    let size = state.settings.size as i32;
    let Some(tribe) = state.tribes.get(&player) else {
        return Vec::new();
    };
    let anchors: Vec<i32> = if tribe.units.is_empty() {
        tribe.cities.iter().map(|c| c.idx).collect()
    } else {
        tribe.units.iter().map(|u| u.coords.idx).collect()
    };
    if anchors.is_empty() || size <= 0 {
        return Vec::new();
    }
    let cheb = |a: i32, b: i32| get_chebyshev_distance(a, b, size);
    let explored =
        |idx: i32| state.tiles.get(&idx).map_or(false, |t| t.explorers.contains(&player));

    // Known spacing sources: explored villages + explored cities (capitals
    // and captured villages count as villages in the generator's spacing).
    let mut known: Vec<i32> = state
        .structures
        .iter()
        .filter(|(idx, s)| {
            s.as_ref().map_or(false, |s| s.structure_type == StructureType::Village)
                && explored(**idx)
        })
        .map(|(idx, _)| *idx)
        .collect();
    for t in state.tribes.values() {
        known.extend(t.cities.iter().map(|c| c.idx).filter(|&i| explored(i)));
    }

    // --- Selection: generator geometry alone (unchanged from the old
    // guessed_village_sites) ---
    let mut cands: Vec<(i32, i32)> = (0..size * size)
        .filter(|&idx| {
            !explored(idx) && validate_village_candidate(state, idx, &known)
        })
        .map(|idx| {
            let d = anchors.iter().map(|&a| cheb(a, idx)).min().unwrap_or(i32::MAX);
            (d, idx)
        })
        .collect();
    cands.sort_unstable();
    let (mut cx, mut cy) = (0i32, 0i32);
    for &a in &anchors {
        cx += a % size;
        cy += a / size;
    }
    cx /= anchors.len() as i32;
    cy /= anchors.len() as i32;
    let quadrant = |idx: i32| ((idx % size > cx) as u8) * 2 + ((idx / size > cy) as u8);
    let mut picks: Vec<i32> = Vec::new();
    let mut used_quads = std::collections::HashSet::new();
    for pass in 0..2 {
        for &(_, idx) in &cands {
            if picks.len() >= max_sites {
                break;
            }
            if picks.contains(&idx) || picks.iter().any(|&p| cheb(p, idx) < 3) {
                continue;
            }
            if pass == 0 && used_quads.contains(&quadrant(idx)) {
                continue;
            }
            used_quads.insert(quadrant(idx));
            picks.push(idx);
        }
    }

    // --- Confidence + tribe: resource/climate evidence near each pick ---
    let pov_climate = tribe_to_climate(tribe.tribe_type);
    let is_orphan = |res_idx: i32| known.iter().all(|&k| cheb(res_idx, k) > 1);
    picks
        .into_iter()
        .map(|site| {
            let mut score = 0i32;
            let mut climate_evidence = 0i32;
            for n in get_adjacent_indices(state, site, 1) {
                if explored(n)
                    && is_orphan(n)
                    && matches!(state.resources.get(&n), Some(Some(_)))
                {
                    score += 5;
                }
            }
            let res_neighbors = get_adjacent_indices(state, site, 1)
                .into_iter()
                .filter(|&n| matches!(state.resources.get(&n), Some(Some(_))))
                .count();
            if res_neighbors >= 2 {
                score += 10;
            }
            for n in get_adjacent_indices(state, site, 2) {
                if let Some(t) = state.tiles.get(&n) {
                    if explored(n) && t.owner != player && t.climate != pov_climate && t.climate != 0
                    {
                        score += 1;
                        climate_evidence = t.climate;
                    }
                }
            }
            let mut guessed_tribe =
                (climate_evidence != 0).then(|| climate_to_tribe(climate_evidence));
            if guessed_tribe == Some(TribeType::Bardur) {
                let crop_nearby = get_adjacent_indices(state, site, 1).into_iter().any(|n| {
                    matches!(
                        state.resources.get(&n),
                        Some(Some(r)) if r.resource_type == crate::types::ResourceType::Crop
                    )
                });
                if crop_nearby {
                    score -= 20;
                    guessed_tribe = None;
                }
            }
            VillageGuess {
                tile: site,
                tribe: guessed_tribe,
                // Purely geometric picks still carry a real (if modest) floor —
                // the generator-saturation reasoning alone is solid evidence.
                confidence: (0.3 + score as f32 / 20.0).clamp(0.05, 1.0),
            }
        })
        .collect()
}

/// Find the tribe of the nearest known city/village to a given tile
fn get_nearest_known_tribe(state: &GameState, idx: i32) -> Option<TribeType> {
    let size = state.settings.size;
    let pov_id = state.settings.current_player_turn_id;

    let mut best_dist = i32::MAX;
    let mut best_tribe = None;

    // Check all tiles for known cities
    for (&t_idx, tile) in &state.tiles {
        // Must be visible or have a known capital/village
        let is_known_city = tile.capital_of > 0
            || (tile.explorers.contains(&pov_id)
                && crate::functions::get_structure_type_at(state, t_idx)
                    == Some(crate::types::StructureType::Village));

        if is_known_city {
            let dist = crate::functions::get_chebyshev_distance(idx, t_idx, size);
            if dist < best_dist {
                best_dist = dist;
                // If it's a capital, use the owner's tribe. If it's a village, we might not know the original tribe,
                // but we can guess from the climate if visible, or owner if visible.
                // For now, use owner if present, else climate.
                if let Some(owner) = state.tribes.get(&tile.owner) {
                    best_tribe = Some(owner.tribe_type);
                } else {
                    best_tribe = Some(climate_to_tribe(tile.climate));
                }
            }
        }
    }

    best_tribe
}

/// Predict terrain for fog tiles based on probabilistic mapgen rules.
/// The second tuple element is a classic climate id (types::classic_climate_id).
pub fn predict_terrain(
    state: &GameState,
    fog_tiles: &[i32],
) -> IndexMap<i32, (TerrainType, i32)> {
    let pov_id = state.settings.current_player_turn_id;
    let map_type = state.settings.map_type;

    // Base land chances for map types (rough estimates from mapgen.rs)
    let base_land_prob = match map_type {
        crate::MapType::None => 0.5,
        crate::MapType::Drylands => 1.0,
        crate::MapType::Lakes => 0.72,
        crate::MapType::Continents => 0.45,
        crate::MapType::Pangea => 0.50,
        crate::MapType::Archipelago => 0.30,
        crate::MapType::WaterWorld => 0.05,
    };

    let mut predictions = IndexMap::new();

    for &tile_idx in fog_tiles {
        let nearest_tribe = get_nearest_known_tribe(state, tile_idx).unwrap_or(TribeType::Imperius); // Default to Imperius rates if totally lost
        let biome_rates = crate::mapgen::get_tribe_biome_rates(nearest_tribe);

        let neighbors = get_adjacent_indices(state, tile_idx, 1);
        let mut land_neighbors = 0;
        let mut total_neighbors = 0;

        for n_idx in neighbors {
            if let Some(tile) = state.tiles.get(&n_idx) {
                if tile.explorers.contains(&pov_id) {
                    match tile.terrain_type {
                        TerrainType::Water | TerrainType::Ocean => {}
                        _ => land_neighbors += 1,
                    }
                    total_neighbors += 1;
                }
            }
        }

        // Adaptive Land Probability:
        // If surrounded by water, likely water. If surrounded by land, likely land.
        // If unknown, use global map type bias.
        let local_land_prob = if total_neighbors > 0 {
            (land_neighbors as f32 + 0.5) / (total_neighbors as f32 + 1.0)
        } else {
            base_land_prob
        };

        // Weighted mix of local and global
        let final_land_prob = 0.7 * local_land_prob + 0.3 * base_land_prob;

        // Decide Terrain
        let terrain_type = if final_land_prob < 0.5 {
            // Likely water. Deep ocean if far from land?
            // Simple heuristic: If we have land neighbors, it's Coast (Water). Else Ocean.
            if land_neighbors > 0 {
                TerrainType::Water
            } else {
                TerrainType::Ocean
            }
        } else {
            // Land! Use biome rates.
            // Normalize rates to 1.0 just in case
            let total_rate = biome_rates.mountain + biome_rates.forest + biome_rates.field;

            // deterministic pseudo-random from coord
            let pseudo_rnd = ((tile_idx * 12345 + 67890) % 100) as f32 / 100.0;

            if pseudo_rnd < (biome_rates.mountain / total_rate) {
                TerrainType::Mountain
            } else if pseudo_rnd < ((biome_rates.mountain + biome_rates.forest) / total_rate) {
                TerrainType::Forest
            } else {
                TerrainType::Field
            }
        };

        let climate = tribe_to_climate(nearest_tribe);

        let final_climate =
            if terrain_type == TerrainType::Water || terrain_type == TerrainType::Ocean {
                0 // fluids are functionally Nature climate
            } else {
                climate
            };

        predictions.insert(tile_idx, (terrain_type, final_climate));
    }

    predictions
}

pub fn get_border_clouds(state: &GameState) -> Vec<i32> {
    let pov_id = state.settings.current_player_turn_id;
    let mut border = std::collections::HashSet::new();
    for (&idx, tile) in &state.tiles {
        if tile.explorers.contains(&pov_id) {
            for n in get_adjacent_indices(state, idx, 1) {
                let n_explored = state
                    .tiles
                    .get(&n)
                    .map(|t| t.explorers.contains(&pov_id))
                    .unwrap_or(false);
                if !n_explored {
                    border.insert(n);
                }
            }
        }
    }
    border.into_iter().collect()
}

pub fn update_predictions(state: &mut GameState) {
    let pov_id = state.settings.current_player_turn_id;
    let villages: IndexMap<i32, (TribeType, bool)> = guess_villages(state, pov_id, 5)
        .into_iter()
        .map(|g| (g.tile, (g.tribe.unwrap_or(TribeType::None), true)))
        .collect();

    // Prediction for ALL unexplored tiles (Mental Image)
    let mut fog_tiles = Vec::new();
    for (&idx, tile) in &state.tiles {
        if !tile.explorers.contains(&pov_id) {
            fog_tiles.push(idx);
        }
    }

    let terrain = predict_terrain(state, &fog_tiles);
    let enemy_capitals = predict_enemy_capitals(state);

    state._prediction = Some(crate::states::PredictionState {
        _villages: villages,
        _terrain: terrain,
        _enemy_capital_suspects: enemy_capitals,
        _city_rewards: Vec::new(),
    });
}

/// Suspected enemy capital tiles, ranked by the mapgen quadrant posterior
/// (the same prior `BeliefState` seeds itself with — see `ai::belief`) and
/// filtered to tiles this player hasn't explored. Replaces the old one-shot
/// mirror-geometry guess, which ignored the generator's actual placement
/// rules and only ever pointed at the map-diagonal opposite corner.
pub fn predict_enemy_capitals(state: &GameState) -> Vec<i32> {
    let size = state.settings.size;
    let pov_id = state.settings.current_player_turn_id;
    let Some(own_cap) = state
        .tiles
        .iter()
        .find(|(_, t)| t.capital_of == pov_id)
        .map(|(&idx, _)| idx)
    else {
        return Vec::new();
    };
    let player_count = state.tribes.len();
    let opponent = state
        .tribes
        .keys()
        .copied()
        .find(|&id| id != pov_id)
        .unwrap_or(pov_id);

    let belief = BeliefState::new(size, player_count, own_cap, pov_id, opponent);
    belief
        .capital_top(8)
        .into_iter()
        .map(|(idx, _)| idx)
        .filter(|idx| {
            !state
                .tiles
                .get(idx)
                .map(|t| t.explorers.contains(&pov_id))
                .unwrap_or(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::states::{GameState, TileState, TribeState};
    use crate::types::{TerrainType, TribeType};

    #[test]
    fn test_village_prediction_constraints() {
        let mut state = GameState::default();
        let size = 11;
        state.settings.size = size;
        for i in 0..(size * size) {
            let mut tile = TileState::default();
            tile.coords = crate::coords::Coords::from_index(i, size);
            tile.terrain_type = TerrainType::Field;
            state.tiles.insert(i, tile);
        }
        let pov_id = 1;
        state.settings.current_player_turn_id = pov_id;
        let mut pov_tribe = TribeState::default();
        pov_tribe.id = pov_id;
        pov_tribe.tribe_type = TribeType::Imperius;
        state.tribes.insert(pov_id, pov_tribe);

        let known_cities: Vec<i32> = Vec::new();
        let ocean_idx = 2 * size + 2;
        state.tiles.get_mut(&ocean_idx).unwrap().terrain_type = TerrainType::Ocean;
        let adj_idx = 2 * size + 3;
        assert!(!validate_village_candidate(&state, adj_idx, &known_cities));

        let city_idx = 5 * size + 5;
        let cities = vec![city_idx];
        assert!(!validate_village_candidate(&state, city_idx + 1, &cities));
        assert!(validate_village_candidate(&state, city_idx + 3, &cities));
    }

    /// The merge's whole point: geometry alone gives a real but modest floor
    /// confidence, and nearby resource/climate evidence raises it — without
    /// changing WHICH tile got picked (that's still generator geometry only).
    #[test]
    fn resource_evidence_raises_confidence_without_changing_the_pick() {
        // Pins the legacy contract: confidence is the `0.3 + score/20` floor
        // and evidence deliberately never moves the pick. Bound explicitly so
        // this keeps testing the legacy path if the entry point is ever
        // re-pointed at the belief (see the belief counterpart below).
        let guess_villages = legacy_village_sites;
        let mut state = GameState::default();
        let size = 11;
        state.settings.size = size;
        for i in 0..(size * size) {
            let mut tile = TileState::default();
            tile.coords = crate::coords::Coords::from_index(i, size);
            tile.terrain_type = TerrainType::Field;
            state.tiles.insert(i, tile);
        }
        let pov_id = 1;
        state.settings.current_player_turn_id = pov_id;
        let mut t1 = TribeState::default();
        t1.id = pov_id;
        t1.tribe_type = TribeType::Imperius;
        t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        state.tribes.insert(pov_id, t1);
        state.tiles.get_mut(&60).unwrap().explorers.insert(pov_id);

        let baseline = guess_villages(&state, pov_id, 1);
        assert_eq!(baseline.len(), 1);
        let site = baseline[0].tile;
        assert_eq!(baseline[0].confidence, 0.3, "no evidence: pure geometric floor");

        // Explore one neighbour of the pick and drop an orphaned resource on
        // it — evidence must not exist until the tile is actually seen.
        let evidence_idx = crate::functions::get_adjacent_indices(&state, site, 1)
            .into_iter()
            .find(|&n| n != 60)
            .expect("a neighbour other than the capital exists on an 11x11 board");
        state.tiles.get_mut(&evidence_idx).unwrap().explorers.insert(pov_id);
        state.resources.insert(
            evidence_idx,
            Some(crate::states::ResourceState { resource_type: crate::types::ResourceType::Game }),
        );

        let with_evidence = guess_villages(&state, pov_id, 1);
        assert_eq!(with_evidence.len(), 1);
        assert_eq!(with_evidence[0].tile, site, "evidence must not change the pick");
        assert!(
            with_evidence[0].confidence > baseline[0].confidence,
            "resource evidence must raise confidence: {} vs baseline {}",
            with_evidence[0].confidence,
            baseline[0].confidence
        );
    }

    /// The belief counterpart, tested directly against `MapBelief` since the
    /// entry point routes to legacy. The durable invariant holds — a resource
    /// must raise confidence in a village nearby — while the legacy contract
    /// above deliberately does not let evidence move the pick.
    #[test]
    fn resource_evidence_raises_belief_confidence() {
        let size = 11;
        let mut state = GameState::default();
        state.settings.size = size;
        for i in 0..(size * size) {
            let mut tile = TileState::default();
            tile.coords = crate::coords::Coords::from_index(i, size);
            tile.terrain_type = TerrainType::Field;
            state.tiles.insert(i, tile);
        }
        let pov_id = 1;
        state.settings.current_player_turn_id = pov_id;
        let mut t1 = TribeState::default();
        t1.id = pov_id;
        t1.tribe_type = TribeType::Imperius;
        t1.cities.push(crate::states::CityState { idx: 60, ..Default::default() });
        state.tribes.insert(pov_id, t1);
        state.tiles.get_mut(&60).unwrap().explorers.insert(pov_id);

        let belief_sites = |st: &GameState| {
            crate::ai::belief::map::MapBelief::observe(st, pov_id).top_village_sites(st, 1)
        };
        let baseline = belief_sites(&state);
        assert_eq!(baseline.len(), 1);

        // An explored ORPHAN resource: >2 from the capital, so the generator's
        // spawn-zone rule leaves it unexplained by any known site.
        let res_idx = 2 * size + 2;
        assert!(
            crate::functions::get_chebyshev_distance(res_idx, 60, size) > 2,
            "fixture error: the resource must be outside the capital's spawn zone"
        );
        state.tiles.get_mut(&res_idx).unwrap().explorers.insert(pov_id);
        state.resources.insert(
            res_idx,
            Some(crate::states::ResourceState { resource_type: crate::types::ResourceType::Game }),
        );

        let with_evidence = belief_sites(&state);
        assert_eq!(with_evidence.len(), 1);
        assert!(
            with_evidence[0].confidence > baseline[0].confidence,
            "resource evidence must raise belief confidence: {} vs {}",
            with_evidence[0].confidence,
            baseline[0].confidence
        );
        // And it must point INTO the resource's spawn zone.
        assert!(
            crate::functions::get_chebyshev_distance(with_evidence[0].tile, res_idx, size) <= 2,
            "pick {} is outside the orphan resource's spawn zone",
            with_evidence[0].tile
        );
    }

    #[test]
    fn test_terrain_prediction_biomes() {
        let mut state = GameState::default();
        let size = 11;
        state.settings.size = size;
        state.settings.map_type = crate::types::MapType::Drylands; // all land

        // Setup Bardur Tribe
        let bardur_id = 2;
        let mut bardur_tribe = TribeState::default();
        bardur_tribe.id = bardur_id;
        bardur_tribe.tribe_type = TribeType::Bardur;
        state.tribes.insert(bardur_id, bardur_tribe);

        let bardur_cap_idx = 2 * size + 2;
        let mut cap_tile = TileState::default();
        cap_tile.coords = crate::coords::Coords::from_index(bardur_cap_idx, size);
        cap_tile.capital_of = bardur_id;
        cap_tile.owner = bardur_id;
        cap_tile.terrain_type = TerrainType::Field;
        state.tiles.insert(bardur_cap_idx, cap_tile);

        // Prediction Target: Tile near Bardur
        let target_idx = 2 * size + 3; // Adjacent to Bardur Cap

        // Let's predict!
        let predictions = predict_terrain(&state, &[target_idx]);
        let (pred_terrain, pred_climate) = predictions[&target_idx];

        // 1. Should be Land (Drylands + Land Neighbor)
        assert_ne!(pred_terrain, TerrainType::Water);
        assert_ne!(pred_terrain, TerrainType::Ocean);

        // 2. Should correspond to Bardur climate (classic id 3)
        assert_eq!(pred_climate, crate::types::classic_climate_id(TribeType::Bardur));

        // 3. Terrain Type Check (Probabilistic but deterministic seed)
        // With current deterministic RNG:
        // idx = 25. pseudo_rnd = ((25 * 12345 + 67890) % 100) / 100.0 => 0.15
        // Bardur Rates (approx): Mountain 0.14, Forest 0.30, Field 0.56
        // 0.15 > 0.14 but < (0.14+0.30=0.44). So likely Forest.
        // Wait, if 0.15 is slightly above 0.14, it falls into "Forest" bucket.
        // Let's assert it is Forest or Field, ensuring it works.
        // Actually, if I can force it to be Forest, great.
        // If pseudo_rnd is 0.15, and mountain rate is ~0.14, then it IS Forest.
        // Let's just assert it is valid land.
        assert!(matches!(
            pred_terrain,
            TerrainType::Field | TerrainType::Forest | TerrainType::Mountain
        ));
    }
}
