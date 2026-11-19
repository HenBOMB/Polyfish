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

mod naval;

pub const PRESET_VERSION: u16 = 1;
pub const RULES_VERSION: i32 = 115;
pub const DEFAULT_MAX_TURNS: i32 = 50;
pub const EXPERT_BUDGET: usize = 256;
pub const RHEA_HORIZON: usize = 20;
pub const BOARD_SIZES: [MapSize; 6] = [
    MapSize::Tiny,
    MapSize::Small,
    MapSize::Normal,
    MapSize::Large,
    MapSize::Huge,
    MapSize::Massive,
];
pub const MAP_TYPES: [(MapType, &str, &str); 6] = [
    (MapType::Drylands, "drylands", "Drylands"),
    (MapType::Lakes, "lakes", "Lakes"),
    (MapType::Continents, "continents", "Continents"),
    (MapType::Pangea, "pangea", "Pangea"),
    (MapType::Archipelago, "archipelago", "Archipelago"),
    (MapType::WaterWorld, "water-world", "Water World"),
];
pub const REGULAR_TRIBES: [(TribeType, &str, &str); 12] = [
    (TribeType::AiMo, "ai-mo", "Ai-Mo"),
    (TribeType::Bardur, "bardur", "Bardur"),
    (TribeType::Hoodrick, "hoodrick", "Hoodrick"),
    (TribeType::Imperius, "imperius", "Imperius"),
    (TribeType::Kickoo, "kickoo", "Kickoo"),
    (TribeType::Luxidoor, "luxidoor", "Luxidoor"),
    (TribeType::Oumaji, "oumaji", "Oumaji"),
    (TribeType::Quetzali, "quetzali", "Quetzali"),
    (TribeType::Vengir, "vengir", "Vengir"),
    (TribeType::XinXi, "xin-xi", "Xin-Xi"),
    (TribeType::Yadakk, "yadakk", "Yădakk"),
    (TribeType::Zebasi, "zebasi", "Zebasi"),
];

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
    BOARD_SIZES
        .into_iter()
        .find(|size| size.get_size().pow(2) == i32::from(tiles))
}

pub fn research_tribe(name: &str) -> Option<TribeType> {
    let alias = match name.to_lowercase().as_str() {
        "xin_xi" | "xinxi" => "xin-xi",
        "ai_mo" | "aimo" => "ai-mo",
        "yădakk" => "yadakk",
        _ => name,
    };
    REGULAR_TRIBES
        .iter()
        .find_map(|&(tribe, key, _)| alias.eq_ignore_ascii_case(key).then_some(tribe))
}

pub fn map_type(name: &str) -> Option<MapType> {
    let alias = match name.to_ascii_lowercase().as_str() {
        "waterworld" | "water world" => "water-world",
        _ => name,
    };
    MAP_TYPES
        .iter()
        .find_map(|&(kind, key, _)| alias.eq_ignore_ascii_case(key).then_some(kind))
}

pub fn map_type_name(kind: MapType) -> Option<&'static str> {
    MAP_TYPES
        .iter()
        .find_map(|&(candidate, key, _)| (candidate == kind).then_some(key))
}

pub fn map_type_label(kind: MapType) -> Option<&'static str> {
    MAP_TYPES
        .iter()
        .find_map(|&(candidate, _, label)| (candidate == kind).then_some(label))
}

pub fn new_match(size: MapSize, tribes: [TribeType; 2], seed: i64, config: MatchConfig) -> Game {
    new_match_on(size, MapType::Drylands, tribes, seed, config)
}

pub fn new_match_on(
    size: MapSize,
    map_type: MapType,
    tribes: [TribeType; 2],
    seed: i64,
    config: MatchConfig,
) -> Game {
    let mut game = Game::new();
    game.state = generate(MapGenSettings {
        size,
        map_type,
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

/// The acting player's board and public information, with unexplored terrain set to Fog.
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
                terrain_type: TerrainType::Fog,
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
    choose_with_root(game, preset, budget, None)
}

pub fn choose_from_decision(
    decision: &crate::Decision<'_>,
    preset: OpponentPreset,
) -> Option<Box<dyn Move>> {
    choose_with_root(
        decision.game(),
        preset,
        EXPERT_BUDGET,
        Some(decision.search_moves()),
    )
    .0
}

fn choose_with_root(
    game: &Game,
    preset: OpponentPreset,
    budget: usize,
    root: Option<&[Box<dyn Move>]>,
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
    let generated;
    let root = match root {
        Some(moves) => moves,
        None => {
            generated = game.legal_moves();
            &generated
        }
    };
    let observed_keys = observed_action_keys(&observed);
    let mut actions = order_candidates(
        root.iter()
            .filter(|action| observed_keys.contains(&action.serialize().to_string()))
            .collect(),
    );
    if actions.len() <= 1 {
        return (actions.pop().map(|m| dyn_clone::clone_box(m.as_ref())), 0);
    }
    let naval = (observed.state.settings.map_type != MapType::Drylands)
        .then(|| naval::NavalPolicy::new(&observed.state));
    if let Some(policy) = &naval {
        actions.retain(|action| !policy.save_for_port(action.as_ref()));
    }
    if preset == OpponentPreset::Expert && budget > 0 {
        if let Some(policy) = &naval {
            // Search starts with the available naval opportunities when one is immediate.
            let best = actions
                .iter()
                .filter_map(|action| policy.priority(action.as_ref()))
                .max();
            if let Some(score) = best.filter(|&score| score >= naval::NAVAL_PRIORITY) {
                actions.retain(|action| policy.priority(action.as_ref()) == Some(score));
            }
        }
        observed.state.settings._verbose = false;
        observed.state.settings.seed = seed as i64;
        for tile in observed.state.tiles.values_mut() {
            if tile.terrain_type == TerrainType::Fog {
                // Expert rollouts assume unseen ground is open land when revealed.
                tile.terrain_type = TerrainType::Field;
            } else if tile.explorers.contains(&pov) {
                tile.explorers.extend(observed.state.tribes.keys().copied());
            }
        }
        let (index, advances) = rolling_horizon(&observed, &actions, budget, &mut rng);
        return (
            Some(dyn_clone::clone_box(actions.swap_remove(index).as_ref())),
            advances,
        );
    }

    let goals = visible_goals(&observed.state);
    let score = |action: &dyn Move| {
        naval
            .as_ref()
            .and_then(|policy| policy.priority(action))
            .unwrap_or_else(|| rule_score(&observed.state, &goals, action))
    };
    actions.sort_by_cached_key(|action| std::cmp::Reverse(score(action.as_ref())));
    let choices = if preset == OpponentPreset::Easy {
        actions.len().min(3)
    } else {
        let best = score(actions[0].as_ref());
        actions
            .iter()
            .take_while(|action| score(action.as_ref()) == best)
            .count()
    };
    let index = rng.random_range(0..choices);
    (
        Some(dyn_clone::clone_box(actions.swap_remove(index).as_ref())),
        0,
    )
}

/// Keep commands available in both the player view and the authoritative state.
pub fn legal_actions(game: &Game, view: &Game) -> Vec<Box<dyn Move>> {
    let observed_keys = observed_action_keys(view);
    game.legal_moves()
        .into_iter()
        .filter(|action| observed_keys.contains(&action.serialize().to_string()))
        .collect()
}

fn observed_action_keys(view: &Game) -> std::collections::HashSet<String> {
    view.legal_moves()
        .into_iter()
        .map(|action| action.serialize().to_string())
        .collect()
}

fn order_candidates<M: AsRef<dyn Move>>(actions: Vec<M>) -> Vec<M> {
    let mut actions: Vec<_> = actions
        .into_iter()
        .filter(|action| {
            action.as_ref().move_type() != MoveType::Resign
                && !matches!(
                    action.as_ref().ability_type(),
                    Ok(AbilityType::Destroy | AbilityType::Disband)
                )
        })
        .collect();
    actions.sort_by_cached_key(|action| action.as_ref().serialize().to_string());
    actions
}

fn visible_goals(state: &GameState) -> Vec<i32> {
    let pov = state.settings.current_player_turn_id;
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
        .map(|tile| tile.coords.idx)
        .collect()
}

fn rule_score(state: &GameState, goals: &[i32], action: &dyn Move) -> u8 {
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
                goals
                    .iter()
                    .map(|&goal| get_chebyshev_distance(index, goal, state.settings.size))
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
    root_actions: &[&Box<dyn Move>],
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
            let action = if ply == 0 {
                root_actions[gene % root_actions.len()].as_ref()
            } else {
                later = order_candidates(trial.legal_moves());
                if later.is_empty() {
                    break;
                }
                later[gene % later.len()].as_ref()
            };
            advances += 1;
            if trial.play_generated(action).is_none() {
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
