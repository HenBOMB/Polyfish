use std::collections::{BTreeMap, BTreeSet};

use crate::ai::belief::{materialize_confident_into, BeliefState, MaterializeStats};
use crate::game::Game;
use crate::states::{GameState, PlayerId, ResourceState};
use crate::types::{ResourceType, StructureType, TerrainType, TribeType, UnitType};

const PARTICLE_VILLAGE_SALT: i32 = 0x61_72_65;
const PARTICLE_RESOURCE_SALT: i32 = 0x72_65_73;
const PARTICLE_UNIT_SALT: i32 = 0x75_6e_69;

fn explored_by(state: &GameState, idx: i32, pov: PlayerId) -> bool {
    state
        .tiles
        .get(&idx)
        .map(|tile| tile.explorers.contains(&pov))
        .unwrap_or(true)
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct MapParticle {
    capital: Option<i32>,
    villages: BTreeSet<i32>,
    resources: BTreeMap<i32, ResourceType>,
    units: BTreeMap<i32, UnitType>,
}

impl MapParticle {
    fn random(seed: i64, idx: i32, salt: i32) -> f32 {
        crate::hash::get_hash(seed as u32, &[idx, salt]) as f32 / u32::MAX as f32
    }

    fn observation_seed(state: &GameState, pov: PlayerId, salt: i32) -> i64 {
        let mut facts = vec![pov, state.settings.turn, salt];
        for (idx, tile) in &state.tiles {
            if !tile.explorers.contains(&pov) {
                continue;
            }
            facts.extend([
                idx,
                tile.terrain_type as i32,
                tile.climate,
                tile.capital_of,
                state.resources.get(&idx).and_then(|r| r.as_ref()).is_some() as i32,
                state
                    .structures
                    .get(&idx)
                    .and_then(|s| s.as_ref())
                    .is_some() as i32,
            ]);
        }
        crate::hash::get_hash(0x4d_43_54_53, &facts) as i64
    }

    fn weighted_capital(
        state: &GameState,
        map_belief: &crate::ai::belief::map::MapBelief,
        pov: PlayerId,
        seed: i64,
        salt: i32,
    ) -> Option<i32> {
        let mut draw = Self::random(seed, salt, PARTICLE_VILLAGE_SALT);
        let candidates: Vec<(i32, f32)> = (0..state.settings.size * state.settings.size)
            .filter(|&idx| !explored_by(state, idx, pov) || map_belief.p_capital(idx) >= 1.0)
            .map(|idx| (idx, map_belief.p_capital(idx)))
            .filter(|(_, probability)| *probability > 0.0)
            .collect();
        for &(idx, probability) in &candidates {
            if draw <= probability {
                return Some(idx);
            }
            draw -= probability;
        }
        candidates.last().map(|(idx, _)| *idx)
    }

    fn nearest_site(idx: i32, sites: &BTreeSet<i32>, size: i32) -> Option<(i32, bool)> {
        sites
            .iter()
            .filter_map(|&site| {
                let distance = crate::functions::get_chebyshev_distance(idx, site, size);
                (distance <= 2).then_some((site, distance <= 1))
            })
            .min_by_key(|(site, _)| *site)
    }

    fn resource_for(
        state: &GameState,
        idx: i32,
        inner: bool,
        tribe: TribeType,
        seed: i64,
    ) -> Option<ResourceType> {
        let terrain = state.tiles.get(&idx)?.terrain_type;
        let draw = Self::random(seed, idx, PARTICLE_RESOURCE_SALT);
        match terrain {
            TerrainType::Field => {
                let fruit = crate::mapgen::get_resource_prob("fruit", tribe, inner);
                let crop_name = if tribe == TribeType::Cymanti {
                    "spores"
                } else {
                    "crop"
                };
                let crop = crate::mapgen::get_resource_prob(crop_name, tribe, inner);
                if draw < fruit {
                    Some(ResourceType::Fruit)
                } else if draw < fruit + crop {
                    Some(if tribe == TribeType::Cymanti {
                        ResourceType::Spores
                    } else {
                        ResourceType::Crop
                    })
                } else {
                    None
                }
            }
            TerrainType::Forest => (draw < crate::mapgen::get_resource_prob("game", tribe, inner))
                .then_some(ResourceType::Game),
            TerrainType::Mountain => (draw
                < crate::mapgen::get_resource_prob("metal", tribe, inner))
            .then_some(ResourceType::Metal),
            TerrainType::Water => (draw < crate::mapgen::get_resource_prob("fish", tribe, inner))
                .then_some(ResourceType::Fish),
            _ => None,
        }
    }

    fn sample(state: &GameState, belief: &BeliefState, salt: i32) -> Self {
        let pov = belief.observer;
        let size = state.settings.size;
        let seed = Self::observation_seed(state, pov, salt);
        let map_belief = crate::ai::belief::map::MapBelief::observe(state, pov);
        let mut particle = Self {
            capital: Self::weighted_capital(state, &map_belief, pov, seed, salt),
            ..Self::default()
        };

        let mut sites: BTreeSet<i32> = crate::ai::belief::map::known_sites(state, pov)
            .into_iter()
            .collect();
        if let Some(capital) = particle.capital {
            sites.insert(capital);
        }

        let mut candidates: Vec<(i32, f32)> = (0..size * size)
            .filter(|&idx| !explored_by(state, idx, pov))
            .map(|idx| (idx, map_belief.p_village(idx)))
            .filter(|(_, p)| *p > 0.0)
            .collect();
        candidates.sort_by_key(|(idx, _)| *idx);
        for (idx, probability) in candidates {
            if sites
                .iter()
                .any(|&site| crate::functions::get_chebyshev_distance(idx, site, size) < 3)
            {
                continue;
            }
            if Self::random(seed, idx, PARTICLE_VILLAGE_SALT) < probability {
                particle.villages.insert(idx);
                sites.insert(idx);
            }
        }

        let own_tribe = state
            .tribes
            .get(&pov)
            .map(|tribe| tribe.tribe_type)
            .unwrap_or(TribeType::Imperius);
        let opponent_tribe = state
            .tribes
            .get(&belief.opponent)
            .map(|tribe| tribe.tribe_type)
            .unwrap_or(TribeType::Bardur);
        for idx in 0..size * size {
            if explored_by(state, idx, pov)
                || particle.villages.contains(&idx)
                || particle.capital == Some(idx)
            {
                continue;
            }
            let Some((site, inner)) = Self::nearest_site(idx, &sites, size) else {
                continue;
            };
            let tribe = if particle.capital == Some(site) {
                opponent_tribe
            } else {
                own_tribe
            };
            if let Some(resource) = Self::resource_for(state, idx, inner, tribe, seed) {
                particle.resources.insert(idx, resource);
            }
        }

        if let Some(capital) = particle.capital {
            let residual = ((belief.residual_army_stars / 2.0).floor() as usize).min(4);
            let mut candidates = crate::functions::get_square_indices(capital, 2, size);
            candidates.retain(|&idx| {
                idx != capital
                    && !explored_by(state, idx, pov)
                    && matches!(
                        state.tiles.get(&idx).map(|tile| tile.terrain_type),
                        Some(TerrainType::Field) | Some(TerrainType::Forest)
                    )
                    && !particle.villages.contains(&idx)
            });
            candidates.sort_by(|a, b| {
                Self::random(seed, *a, PARTICLE_UNIT_SALT)
                    .total_cmp(&Self::random(seed, *b, PARTICLE_UNIT_SALT))
                    .then(a.cmp(b))
            });
            for idx in candidates.into_iter().take(residual) {
                particle.units.insert(idx, UnitType::Warrior);
            }
        }
        particle
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BranchBeliefStats {
    pub revealed: u32,
    pub resource_revealed: u32,
    pub capital_refuted: u32,
    pub capital_confirmed: u32,
    pub village_confirmed: u32,
    pub capital_materialized: u32,
    pub village_materialized: u32,
    pub unit_materialized: u32,
}

/// A sampled, branch-local observation history. Its hidden map content comes
/// from the generator-constrained posterior, never the real fogged board.
#[derive(Clone)]
pub struct BranchBelief {
    belief: BeliefState,
    processed_sim_explored: BTreeSet<i32>,
    particle: MapParticle,
    particle_salt: i32,
    materialized_capital: Option<i32>,
    materialized_village: Option<i32>,
    materialized_units: BTreeSet<i32>,
}

impl BranchBelief {
    pub fn new(root: BeliefState, state: &GameState) -> Self {
        Self::new_with_salt(root, state, 0)
    }

    pub fn new_with_salt(root: BeliefState, state: &GameState, salt: i32) -> Self {
        Self {
            particle: MapParticle::sample(state, &root, salt),
            belief: root,
            processed_sim_explored: BTreeSet::new(),
            particle_salt: salt,
            materialized_capital: None,
            materialized_village: None,
            materialized_units: BTreeSet::new(),
        }
    }

    pub fn fork(&self, state: &GameState, salt: i32) -> Self {
        let mut branch = self.clone();
        let scoped_salt = self
            .particle_salt
            .wrapping_mul(1_000_003)
            .wrapping_add(salt);
        branch.particle = MapParticle::sample(state, &branch.belief, scoped_salt);
        branch
    }

    pub fn sanitize_fog_view(state: &mut GameState, pov: PlayerId) {
        for tile in state.tiles.values_mut() {
            if !tile.explorers.contains(&pov) {
                tile.capital_of = 0;
                tile.climate = 0;
                tile.skin_type = 0;
                tile.ruling_city_coords = None;
                tile.had_route = false;
            }
        }
    }

    pub fn candidate_belief_for(&self, player: PlayerId) -> Option<&BeliefState> {
        (player == self.belief.observer).then_some(&self.belief)
    }

    fn normalize_capital(&mut self) {
        let total: f32 = self.belief.capital_posterior.iter().map(|(_, p)| *p).sum();
        if total > 0.0 {
            for (_, p) in &mut self.belief.capital_posterior {
                *p /= total;
            }
        }
    }

    fn city_at(state: &GameState, player: PlayerId, idx: i32) -> bool {
        state
            .tribes
            .get(&player)
            .map(|tribe| tribe.cities.iter().any(|city| city.idx == idx))
            .unwrap_or(false)
    }

    fn village_at(state: &GameState, idx: i32) -> bool {
        matches!(
            state.structures.get(&idx),
            Some(Some(s)) if s.structure_type == StructureType::Village
        )
    }

    fn newly_explored(&self, state: &GameState) -> Vec<i32> {
        state
            .settings
            ._sim_explored
            .get(&self.belief.observer)
            .map(|tiles| {
                tiles
                    .iter()
                    .copied()
                    .filter(|idx| !self.processed_sim_explored.contains(idx))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn reveal_particle(&mut self, game: &mut Game, idx: i32) -> MaterializeStats {
        let mut stats = MaterializeStats::default();
        if let Some(resource) = self.particle.resources.get(&idx).copied() {
            game.state.resources.insert(
                idx,
                Some(ResourceState {
                    resource_type: resource,
                }),
            );
        }
        if self.particle.villages.contains(&idx) && !Self::village_at(&game.state, idx) {
            if let Some(tile) = game.state.tiles.get_mut(&idx) {
                tile.terrain_type = TerrainType::Field;
                tile.owner = 0;
                tile._unit_owner_id = None;
            }
            let _ = crate::actions::structure::create_structure(
                &mut game.state,
                idx,
                StructureType::Village,
                1,
            );
            self.materialized_village = Some(idx);
            stats.village = true;
            stats.village_idx = Some(idx);
        }
        if self.particle.capital == Some(idx)
            && !Self::city_at(&game.state, self.belief.opponent, idx)
        {
            let mut capital = self.belief.clone();
            capital.capital_posterior = vec![(idx, 1.0)];
            capital.capital_confirmed = None;
            let placed = materialize_confident_into(game, &capital, 0.0);
            if placed.capital {
                self.materialized_capital = Some(idx);
                stats.capital = true;
                stats.capital_idx = Some(idx);
            }
        }
        if let Some(unit) = self.particle.units.get(&idx).copied() {
            let vacant = game
                .state
                .tiles
                .get(&idx)
                .map(|tile| tile._unit_owner_id.is_none())
                .unwrap_or(false);
            if vacant
            {
                let _ = crate::actions::units::spawn_unit(
                    &mut game.state,
                    self.belief.opponent,
                    unit,
                    idx,
                    false,
                );
                self.materialized_units.insert(idx);
                stats.residual_units += 1;
            }
        }
        stats
    }

    fn consume_observations(
        &mut self,
        game: &mut Game,
        stats: &mut BranchBeliefStats,
    ) -> MaterializeStats {
        let mut materialized = MaterializeStats::default();
        for idx in self.newly_explored(&game.state) {
            self.processed_sim_explored.insert(idx);
            stats.revealed += 1;

            let revealed = self.reveal_particle(game, idx);
            if self.particle.resources.contains_key(&idx) {
                stats.resource_revealed += 1;
            }
            materialized.capital |= revealed.capital;
            materialized.capital_idx = materialized.capital_idx.or(revealed.capital_idx);
            materialized.village |= revealed.village;
            materialized.village_idx = materialized.village_idx.or(revealed.village_idx);

            let capital_seen = self.particle.capital == Some(idx)
                && Self::city_at(&game.state, self.belief.opponent, idx);
            let village_seen =
                self.particle.villages.contains(&idx) && Self::village_at(&game.state, idx);

            if let Some(tile) = game.state.tiles.get_mut(&idx) {
                tile.explorers.insert(self.belief.observer);
            }

            if self.belief.capital_confirmed.is_none() {
                if capital_seen {
                    self.belief.capital_posterior = vec![(idx, 1.0)];
                    self.belief.capital_confirmed = Some(idx);
                    stats.capital_confirmed += 1;
                } else {
                    let before = self.belief.capital_posterior.len();
                    self.belief
                        .capital_posterior
                        .retain(|(cell, _)| *cell != idx);
                    if before != self.belief.capital_posterior.len() {
                        self.normalize_capital();
                        stats.capital_refuted += 1;
                    }
                }
            }
            if village_seen {
                stats.village_confirmed += 1;
            }
            if self.materialized_units.contains(&idx) {
                stats.unit_materialized += 1;
            }
        }
        materialized
    }

    pub fn materialize_child(&mut self, game: &mut Game) -> (MaterializeStats, BranchBeliefStats) {
        let mut observations = BranchBeliefStats::default();
        let placed = self.consume_observations(game, &mut observations);
        observations.capital_materialized += placed.capital as u32;
        observations.village_materialized += placed.village as u32;
        (placed, observations)
    }

    #[cfg(test)]
    pub fn capital_posterior(&self) -> &[(i32, f32)] {
        &self.belief.capital_posterior
    }

    #[cfg(test)]
    pub fn force_particle_capital(&mut self, idx: i32) {
        self.particle.capital = Some(idx);
    }

    #[cfg(test)]
    pub fn force_particle_contents(&mut self, resource_idx: i32, village_idx: i32) {
        self.particle.resources.clear();
        self.particle
            .resources
            .insert(resource_idx, ResourceType::Fruit);
        self.particle.villages.clear();
        self.particle.villages.insert(village_idx);
    }

    #[cfg(test)]
    pub fn force_particle_unit(&mut self, idx: i32) {
        self.particle.units.insert(idx, UnitType::Warrior);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MapSize, MapType};

    fn generated_game(seed: i64) -> Game {
        let mut game = Game::new();
        game.state = crate::mapgen::generate(crate::mapgen::MapGenSettings {
            size: MapSize::Tiny,
            map_type: MapType::Drylands,
            tribes: vec![TribeType::Imperius, TribeType::Bardur],
            seed,
            version: 115,
        });
        game.post_load();
        game
    }

    fn root_belief(game: &Game, pov: PlayerId) -> BeliefState {
        let own = game.state.tribes.get(&pov).unwrap().cities[0].idx;
        BeliefState::new(
            game.state.settings.size,
            2,
            own,
            pov,
            if pov == 1 { 2 } else { 1 },
        )
    }

    #[test]
    fn particle_does_not_depend_on_the_secret_map_seed() {
        let game = generated_game(19_001);
        let pov = game.state.settings.current_player_turn_id;
        let mut a = game.clone_for_mcts(pov);
        BranchBelief::sanitize_fog_view(&mut a.state, pov);
        let mut b = a.clone();
        b.state.settings.seed = 999_999_999;

        let belief = root_belief(&a, pov);
        assert_eq!(
            MapParticle::sample(&a.state, &belief, 17),
            MapParticle::sample(&b.state, &belief, 17),
        );
    }

    #[test]
    fn sampled_contents_stay_latent_until_simulated_exploration() {
        let game = generated_game(19_002);
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        BranchBelief::sanitize_fog_view(&mut view.state, pov);
        let fog: Vec<i32> = view
            .state
            .tiles
            .iter()
            .filter(|(_, tile)| !tile.explorers.contains(&pov))
            .map(|(idx, _)| idx)
            .collect();
        let before = crate::ai::belief::map::MapBelief::observe(&view.state, pov);
        let known = crate::ai::belief::map::known_sites(&view.state, pov);
        let fruit = fog
            .iter()
            .copied()
            .find(|&idx| {
                known.iter().all(|&site| {
                    crate::functions::get_chebyshev_distance(idx, site, view.state.settings.size)
                        > 2
                }) && crate::functions::get_square_indices(idx, 2, view.state.settings.size)
                    .into_iter()
                    .any(|candidate| {
                        candidate != idx
                            && !view.state.tiles[&candidate].explorers.contains(&pov)
                            && before.p_village(candidate) > 0.0
                    })
            })
            .expect("an unexplained fog tile with village support nearby");
        let village = crate::functions::get_square_indices(fruit, 2, view.state.settings.size)
            .into_iter()
            .find(|&idx| {
                idx != fruit
                    && !view.state.tiles[&idx].explorers.contains(&pov)
                    && before.p_village(idx) > 0.0
            })
            .unwrap();
        let mut branch = BranchBelief::new(root_belief(&view, pov), &view.state);
        branch.force_particle_contents(fruit, village);

        assert!(view
            .state
            .resources
            .get(&fruit)
            .and_then(|r| r.as_ref())
            .is_none());
        assert!(!BranchBelief::village_at(&view.state, village));

        view.state
            .settings
            ._sim_explored
            .entry(pov)
            .or_default()
            .insert(fruit);
        let (_, first) = branch.materialize_child(&mut view);
        assert_eq!(first.resource_revealed, 1);
        assert_eq!(
            view.state.resources[&fruit].as_ref().unwrap().resource_type,
            ResourceType::Fruit,
        );
        assert!(!BranchBelief::village_at(&view.state, village));
        let after = crate::ai::belief::map::MapBelief::observe(&view.state, pov);
        let mass: f32 = crate::functions::get_square_indices(fruit, 2, view.state.settings.size)
            .into_iter()
            .map(|idx| after.p_village(idx))
            .sum();
        assert!(
            mass >= 0.99,
            "synthetic fruit did not constrain a village: {mass}"
        );

        view.state
            .settings
            ._sim_explored
            .entry(pov)
            .or_default()
            .insert(village);
        let (placed, second) = branch.materialize_child(&mut view);
        assert!(placed.village);
        assert_eq!(second.village_confirmed, 1);
        assert!(BranchBelief::village_at(&view.state, village));
    }

    #[test]
    fn sampled_enemy_unit_stays_latent_until_simulated_exploration() {
        let game = generated_game(19_003);
        let pov = game.state.settings.current_player_turn_id;
        let mut view = game.clone_for_mcts(pov);
        BranchBelief::sanitize_fog_view(&mut view.state, pov);
        let target = view
            .state
            .tiles
            .iter()
            .find_map(|(idx, tile)| {
                (!tile.explorers.contains(&pov)
                    && tile._unit_owner_id.is_none()
                    && matches!(tile.terrain_type, TerrainType::Field | TerrainType::Forest))
                .then_some(idx)
            })
            .expect("a fogged land tile without a unit");
        let opponent = if pov == 1 { 2 } else { 1 };
        let mut branch = BranchBelief::new(root_belief(&view, pov), &view.state);
        branch.force_particle_unit(target);

        assert!(view
            .state
            .tribes
            .get(&opponent)
            .unwrap()
            .units
            .iter()
            .all(|unit| unit.coords.idx != target));

        view.state
            .settings
            ._sim_explored
            .entry(pov)
            .or_default()
            .insert(target);
        let (_, observed) = branch.materialize_child(&mut view);
        assert_eq!(observed.unit_materialized, 1);
        assert!(view
            .state
            .tribes
            .get(&opponent)
            .unwrap()
            .units
            .iter()
            .any(|unit| unit.coords.idx == target && unit.unit_type == UnitType::Warrior));
    }
}
