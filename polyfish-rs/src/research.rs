use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use crate::functions::{
    get_chebyshev_distance, get_structure_type_at, get_tech_tier, get_total_production,
    get_unit_at, get_unit_attack, get_unit_defense,
};
use crate::game::Game;
use crate::mapgen::{MapGenSettings, generate};
use crate::moves::Move;
use crate::states::{GameState, PlayerId, TileState, TribeState};
use crate::types::{
    AbilityType, CityRewardType, MapSize, MapType, ModeType, MoveType, StructureType, TerrainType,
    TribeType,
};

pub const PRESET_VERSION: u16 = 1;
pub const RULES_VERSION: i32 = 115;
pub const DEFAULT_MAX_TURNS: i32 = 50;
pub const EXPERT_BUDGET: usize = 256;
pub const RHEA_HORIZON: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OpponentPreset {
    Easy,
    Normal,
    Expert,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Opponent {
    #[default]
    Human,
    Heuristic,
    Learned,
    Easy,
    Normal,
    Expert,
}

impl Opponent {
    pub fn preset(self) -> Option<OpponentPreset> {
        match self {
            Self::Easy => Some(OpponentPreset::Easy),
            Self::Normal => Some(OpponentPreset::Normal),
            Self::Expert => Some(OpponentPreset::Expert),
            _ => None,
        }
    }
}

impl From<OpponentPreset> for Opponent {
    fn from(preset: OpponentPreset) -> Self {
        match preset {
            OpponentPreset::Easy => Self::Easy,
            OpponentPreset::Normal => Self::Normal,
            OpponentPreset::Expert => Self::Expert,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchConfig {
    pub preset_version: u16,
    pub opponent: Opponent,
    pub bot_seat: PlayerId,
}

impl MatchConfig {
    pub fn new(opponent: Opponent, bot_seat: PlayerId) -> Self {
        Self {
            preset_version: PRESET_VERSION,
            opponent,
            bot_seat,
        }
    }
}

pub fn board_size(tiles: u16) -> Option<MapSize> {
    match tiles {
        121 => Some(MapSize::Tiny),
        196 => Some(MapSize::Small),
        256 => Some(MapSize::Normal),
        _ => None,
    }
}

pub fn research_tribe(name: &str) -> Option<TribeType> {
    match name.to_ascii_lowercase().as_str() {
        "xin-xi" | "xin_xi" | "xinxi" => Some(TribeType::XinXi),
        "imperius" => Some(TribeType::Imperius),
        "bardur" => Some(TribeType::Bardur),
        "oumaji" => Some(TribeType::Oumaji),
        _ => None,
    }
}

pub fn new_match(size: MapSize, tribes: [TribeType; 2], seed: i64, config: MatchConfig) -> Game {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size,
        map_type: MapType::Drylands,
        tribes: tribes.to_vec(),
        seed,
        version: RULES_VERSION,
        ..Default::default()
    });
    use_might_rules(&mut game, config);
    game.state.settings.max_turns = DEFAULT_MAX_TURNS;
    game.post_load();
    game
}

pub fn use_might_rules(game: &mut Game, config: MatchConfig) {
    game.state.initial_seed = game.state.settings.seed;
    game.state.settings.mode = ModeType::Might;
    game.state.settings.win_by_capital = true;
    game.state.settings.win_by_extermination = false;
    game.state.settings._fow = true;
    game.state.settings.research = Some(config);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(tag = "status", rename_all = "camelCase")]
pub enum MatchResult {
    InProgress,
    Won { winner: PlayerId },
    Truncated,
}

pub fn result(game: &Game) -> MatchResult {
    if let Some(winner) = game.state.settings.might_winner {
        return MatchResult::Won { winner };
    }
    if !game.is_game_over() {
        return MatchResult::InProgress;
    }
    let mut survivors = game
        .state
        .tribes
        .values()
        .filter(|t| t.killed_turn <= 0 && t.resigned_turn <= 0);
    match (survivors.next(), survivors.next()) {
        (Some(tribe), None) => MatchResult::Won { winner: tribe.id },
        _ => MatchResult::Truncated,
    }
}

/// A belief-state copy built from the acting player's board and public information.
pub fn observation(game: &Game, pov: PlayerId) -> Game {
    let mut view = game.clone();
    let state = &mut view.state;
    if !state.settings._fow {
        return view;
    }
    state._prediction = None;
    state._history.clear();
    state._messages.clear();
    state.settings._sim_explored.clear();
    state._end_of_turn_queue.clear();
    state.obscure_fog(pov);
    let hidden_resources: Vec<_> = state
        .resources
        .iter()
        .filter_map(|(&idx, resource)| {
            resource
                .as_ref()
                .filter(|r| {
                    !crate::functions::is_resource_visible_to_tribe(
                        state,
                        r.resource_type,
                        pov,
                        Some(idx),
                    )
                })
                .map(|_| idx)
        })
        .collect();
    for idx in hidden_resources {
        state.resources.insert(idx, None);
    }
    for tile in state.tiles.values_mut() {
        if tile.explorers.contains(&pov) {
            tile.explorers.retain(|id| *id == pov);
        } else {
            *tile = TileState {
                coords: tile.coords,
                terrain_type: TerrainType::Field,
                ..TileState::default()
            };
        }
    }
    let visible = state
        .tiles
        .values()
        .filter(|tile| tile.explorers.contains(&pov))
        .map(|tile| tile.coords.idx)
        .collect::<std::collections::HashSet<_>>();
    for tile in state.tiles.values_mut() {
        tile.ruling_city_coords = tile
            .ruling_city_coords
            .filter(|coords| visible.contains(&coords.idx));
    }
    for (&id, tribe) in &mut state.tribes {
        if id != pov {
            let mut cities = std::mem::take(&mut tribe.cities);
            for city in &mut cities {
                city._territory.retain(|idx| visible.contains(idx));
            }
            let mut units = std::mem::take(&mut tribe.units);
            units.retain(|unit| !unit.effects.contains(&crate::types::UnitEffect::Invisible));
            for unit in &mut units {
                unit.home_coords = None;
                unit.prev_coords = unit.coords;
                unit.last_attack_coords = None;
                unit.parent_unit_idx = None;
                unit.child_unit_idx = None;
            }
            *tribe = TribeState {
                id,
                tribe_type: tribe.tribe_type,
                killed_turn: tribe.killed_turn,
                resigned_turn: tribe.resigned_turn,
                starting_tile_coords: state
                    .tiles
                    .values()
                    .find(|tile| tile.capital_of == id && tile.owner == id)
                    .map_or_else(Default::default, |tile| tile.coords),
                units,
                cities,
                ..TribeState::default()
            };
        }
    }
    for tile in state.tiles.values_mut() {
        if tile._unit_owner_id.is_some_and(|id| id != pov)
            && !state
                .tribes
                .values()
                .flat_map(|tribe| &tribe.units)
                .any(|unit| unit.coords.idx == tile.coords.idx)
        {
            tile._unit_owner_id = None;
        }
    }
    view
}

pub fn choose_move(game: &Game, preset: OpponentPreset) -> Option<Box<dyn Move>> {
    choose_with_budget(game, preset, EXPERT_BUDGET).0
}

pub fn choose_with_budget(
    game: &Game,
    preset: OpponentPreset,
    budget: usize,
) -> (Option<Box<dyn Move>>, usize) {
    if result(game) != MatchResult::InProgress {
        return (None, 0);
    }
    let pov = game.current_player_id();
    let seed = (game.state.initial_seed as u64)
        ^ ((pov as u64) << 56)
        ^ (game.state._history.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut observed = observation(game, pov);
    let mut actions = order_candidates(legal_actions(game, &observed));
    if actions.len() <= 1 {
        return (actions.pop(), 0);
    }
    if preset == OpponentPreset::Expert && budget > 0 {
        observed.state.settings._verbose = false;
        observed.state.settings.seed = seed as i64;
        // Both simulated players see the known board. Unknown tiles remain blank.
        for tile in observed
            .state
            .tiles
            .values_mut()
            .filter(|tile| tile.explorers.contains(&pov))
        {
            tile.explorers.extend(observed.state.tribes.keys().copied());
        }
        let (index, advances) = rolling_horizon(&observed, &actions, budget, &mut rng);
        return (Some(actions.swap_remove(index)), advances);
    }

    actions.sort_by_cached_key(|action| {
        std::cmp::Reverse(rule_score(&observed.state, action.as_ref()))
    });
    let choices = if preset == OpponentPreset::Easy {
        actions.len().min(3)
    } else {
        let best = rule_score(&observed.state, actions[0].as_ref());
        actions
            .iter()
            .take_while(|action| rule_score(&observed.state, action.as_ref()) == best)
            .count()
    };
    let index = rng.random_range(0..choices);
    (Some(actions.swap_remove(index)), 0)
}

/// Keep commands available in both the player view and the authoritative state.
pub fn legal_actions(game: &Game, view: &Game) -> Vec<Box<dyn Move>> {
    let observed_keys: std::collections::HashSet<_> = view
        .legal_moves()
        .into_iter()
        .map(|action| action.serialize().to_string())
        .collect();
    game.legal_moves()
        .into_iter()
        .filter(|action| observed_keys.contains(&action.serialize().to_string()))
        .collect()
}

fn order_candidates(actions: Vec<Box<dyn Move>>) -> Vec<Box<dyn Move>> {
    let mut actions: Vec<_> = actions
        .into_iter()
        .filter(|action| {
            action.move_type() != MoveType::Resign
                && !matches!(
                    action.ability_type(),
                    Ok(AbilityType::Destroy | AbilityType::Disband)
                )
        })
        .collect();
    actions.sort_by_cached_key(|action| action.serialize().to_string());
    actions
}

fn rule_score(state: &GameState, action: &dyn Move) -> u8 {
    let pov = state.settings.current_player_turn_id;
    let source = action.source_idx().ok().map(|idx| idx as i32);
    let target = action.target_idx().ok().map(|idx| idx as i32);
    match action.move_type() {
        MoveType::Capture => 5,
        MoveType::Reward => match action.reward_type() {
            Ok(
                CityRewardType::Workshop
                | CityRewardType::Resources
                | CityRewardType::BorderGrowth
                | CityRewardType::SuperUnit,
            ) => 5,
            _ => 3,
        },
        MoveType::Attack => {
            let attacker = source.and_then(|idx| get_unit_at(state, idx));
            let defender = target.and_then(|idx| get_unit_at(state, idx));
            match (attacker, defender) {
                (Some(attacker), Some(defender))
                    if get_unit_attack(state, attacker) >= get_unit_defense(defender)
                        && attacker.health >= defender.health =>
                {
                    5
                }
                (Some(_), Some(defender)) if defender.health <= 5.0 => 4,
                _ => 1,
            }
        }
        MoveType::Step => {
            let Some((from, to)) = source.zip(target) else {
                return 1;
            };
            let distance = |index| {
                state
                    .tiles
                    .values()
                    .filter(|tile| {
                        tile.explorers.contains(&pov)
                            && tile.owner != pov
                            && (crate::functions::get_city_at(state, tile.coords.idx).is_some()
                                || matches!(
                                    get_structure_type_at(state, tile.coords.idx),
                                    Some(StructureType::Village | StructureType::Ruin)
                                ))
                    })
                    .map(|tile| get_chebyshev_distance(index, tile.coords.idx, state.settings.size))
                    .min()
            };
            if distance(to) < distance(from) {
                4
            } else if crate::functions::get_adjacent_indices(state, to, 1)
                .iter()
                .any(|idx| {
                    state
                        .tiles
                        .get(idx)
                        .is_some_and(|tile| !tile.explorers.contains(&pov))
                })
            {
                3
            } else {
                1
            }
        }
        MoveType::Harvest => 4,
        MoveType::Build => 3,
        MoveType::Summon => {
            if state
                .current_tribe()
                .is_some_and(|tribe| tribe.units.len() <= tribe.cities.len())
            {
                4
            } else {
                2
            }
        }
        MoveType::Research => action
            .tech_type()
            .map(|tech| (4 - get_tech_tier(tech)).clamp(1, 3) as u8)
            .unwrap_or(1),
        MoveType::Ability => match action.ability_type() {
            Ok(AbilityType::Promote) => 5,
            Ok(AbilityType::Convert | AbilityType::HealOthers) => 4,
            Ok(AbilityType::Recover) => 2,
            _ => 0,
        },
        MoveType::EndTurn => 1,
        _ => 0,
    }
}

fn fitness(game: &Game, pov: PlayerId) -> f32 {
    if let MatchResult::Won { winner } = result(game) {
        return if winner == pov {
            1_000_000.0
        } else {
            -1_000_000.0
        };
    }
    game.state
        .tribes
        .values()
        .map(|tribe| {
            let value = 2.0 * get_total_production(&game.state, &tribe.cities) as f32
                + 2.0 * tribe.tech_vanilla.len() as f32
                + 0.01 * tribe.score as f32
                + 12.0 * tribe.cities.len() as f32
                + 2.0 * tribe.units.len() as f32
                + tribe.kills as f32
                + tribe
                    .cities
                    .iter()
                    .map(|city| city.level as f32)
                    .sum::<f32>();
            if tribe.id == pov { value } else { -value }
        })
        .sum()
}

fn rolling_horizon(
    root: &Game,
    root_actions: &[Box<dyn Move>],
    budget: usize,
    rng: &mut StdRng,
) -> (usize, usize) {
    let mut best = std::array::from_fn::<usize, RHEA_HORIZON, _>(|_| rng.random::<u64>() as usize);
    let mut best_score = f32::NEG_INFINITY;
    let mut advances = 0;
    while advances < budget {
        let mut candidate = best;
        if advances > 0 {
            candidate[rng.random_range(0..RHEA_HORIZON)] = rng.random::<u64>() as usize;
        }
        let mut trial = root.clone();
        for (ply, &gene) in candidate.iter().enumerate() {
            if advances == budget || result(&trial) != MatchResult::InProgress {
                break;
            }
            let later;
            let actions = if ply == 0 {
                root_actions
            } else {
                later = order_candidates(trial.legal_moves());
                &later
            };
            if actions.is_empty() {
                break;
            }
            advances += 1;
            if trial
                .play_move(actions[gene % actions.len()].as_ref())
                .is_none()
            {
                break;
            }
        }
        let score = fitness(&trial, root.current_player_id());
        if score >= best_score {
            best_score = score;
            best = candidate;
        }
    }
    (best[0] % root_actions.len(), advances)
}
