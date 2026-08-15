use crate::ai::features::{self, RawFeatures};
use crate::ai::mapper::{DecomposedMapper, DecomposedTargets};
use crate::ai::network::NUM_ACTION_TYPES;
use crate::functions::get_tribe_spt;
use crate::game::Game;
use crate::moves::Move;
use crate::states::PlayerId;
use crate::types::{TechnologyType, UnitEffect};
use candle_core::{Device, Tensor};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use strum::IntoEnumIterator;

use super::{
    ACTION_SCHEMA_VERSION, DATASET_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION, REPLAY_SCHEMA_VERSION,
    Replay, ReplayCommand, ReplayError, ReplayMoveContext, ReplayObserver, ReplayResult,
    validate_training_eligibility,
};

struct PendingSample {
    features: RawFeatures,
    targets: DecomposedTargets,
    player_id: PlayerId,
    opponent_id: PlayerId,
    turn: i32,
    enemy_units: Vec<f32>,
    spt: HashMap<PlayerId, i32>,
}

pub struct TrainingCollector {
    samples: Vec<PendingSample>,
}

impl TrainingCollector {
    pub fn new(replay: &Replay) -> Result<Self, ReplayError> {
        validate_training_eligibility(replay)?;
        Ok(Self {
            samples: Vec::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn finish(
        self,
        game: &Game,
        result: Option<&ReplayResult>,
        source_file: &Path,
    ) -> Result<Vec<TrainingSample>, ReplayError> {
        let result = result.ok_or_else(|| {
            ReplayError::Training(
                "behavior-cloning export requires replay.result for value labels".into(),
            )
        })?;
        let total_cities = game
            .state
            .tribes
            .values()
            .map(|t| t.cities.len())
            .sum::<usize>();
        let final_owner: Vec<i32> = (0..features::MAP_SIZE * features::MAP_SIZE)
            .map(|idx| game.state.tiles.get(&(idx as i32)).map_or(0, |t| t.owner))
            .collect();
        let tech_order: Vec<TechnologyType> = TechnologyType::iter().collect();
        let final_tech: HashMap<PlayerId, Vec<f32>> = game
            .state
            .tribes
            .iter()
            .map(|(&id, tribe)| {
                let mut target = vec![0.0; tech_order.len()];
                for tech in tribe.tech_vanilla.iter().filter(|t| t.discovered) {
                    if let Some(i) = tech_order
                        .iter()
                        .position(|candidate| *candidate == tech.tech_type)
                    {
                        target[i] = 1.0;
                    }
                }
                (id, target)
            })
            .collect();
        let final_spt: HashMap<PlayerId, i32> = game
            .state
            .tribes
            .iter()
            .map(|(&id, tribe)| (id, get_tribe_spt(&game.state, tribe)))
            .collect();

        let future_spt = |sample_index: usize, player: PlayerId, opponent: PlayerId, turn: i32| {
            self.samples
                .iter()
                .skip(sample_index + 1)
                .find(|later| later.player_id == player && later.turn >= turn + 5)
                .map(|later| {
                    (
                        later.spt.get(&player).copied().unwrap_or(0),
                        later.spt.get(&opponent).copied().unwrap_or(0),
                    )
                })
                .unwrap_or((
                    final_spt.get(&player).copied().unwrap_or(0),
                    final_spt.get(&opponent).copied().unwrap_or(0),
                ))
        };

        let future_spts: Vec<_> = self
            .samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                future_spt(index, sample.player_id, sample.opponent_id, sample.turn)
            })
            .collect();
        let samples = self.samples;
        let mut out = Vec::with_capacity(samples.len());
        for (sample, (my_spt, opponent_spt)) in samples.into_iter().zip(future_spts) {
            let value = value_for_player(result, sample.player_id)?;
            let city_count = game
                .state
                .tribes
                .get(&sample.player_id)
                .map_or(0, |t| t.cities.len());
            let progress = if total_cities == 0 {
                0.0
            } else {
                (city_count as f32 / total_cities as f32) * 2.0 - 1.0
            };
            let ownership = final_owner
                .iter()
                .map(|&owner| {
                    if owner == sample.player_id {
                        1.0
                    } else if owner == 0 {
                        0.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            out.push(TrainingSample {
                features: sample.features,
                targets: sample.targets,
                value,
                progress,
                progress_mask: 1.0,
                aux_ownership: ownership,
                aux_fog_units: sample.enemy_units,
                aux_spt: [my_spt as f32 / 20.0, opponent_spt as f32 / 20.0],
                aux_opp_tech: final_tech
                    .get(&sample.opponent_id)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; tech_order.len()]),
                aux_mask: 1.0,
                source_file: source_file.display().to_string(),
            });
        }
        Ok(out)
    }
}

impl ReplayObserver for TrainingCollector {
    fn before_move(
        &mut self,
        game: &Game,
        _context: &ReplayMoveContext,
        _legal_moves: &[Box<dyn Move>],
        selected_move: &dyn Move,
        _command: &ReplayCommand,
    ) -> Result<(), ReplayError> {
        let player_id = game.state.settings.current_player_turn_id;
        let opponent_id = game
            .state
            .tribes
            .iter()
            .filter(|(id, _)| **id != player_id)
            .max_by_key(|(_, tribe)| tribe.score)
            .map(|(&id, _)| id)
            .unwrap_or(player_id);
        let raw = features::state_to_cpu_features(&game.state, player_id)
            .map_err(|e| ReplayError::Training(format!("feature extraction failed: {e}")))?;
        let targets = DecomposedMapper::move_to_targets(selected_move, features::MAP_SIZE);
        let mut enemy_units = vec![0.0; features::MAP_SIZE * features::MAP_SIZE];
        for (&id, tribe) in &game.state.tribes {
            if id == player_id {
                continue;
            }
            for unit in &tribe.units {
                if !unit.effects.contains(&UnitEffect::Invisible)
                    && unit.coords.idx >= 0
                    && (unit.coords.idx as usize) < enemy_units.len()
                {
                    enemy_units[unit.coords.idx as usize] = 1.0;
                }
            }
        }
        let spt = game
            .state
            .tribes
            .iter()
            .map(|(&id, tribe)| (id, get_tribe_spt(&game.state, tribe)))
            .collect();
        self.samples.push(PendingSample {
            features: raw,
            targets,
            player_id,
            opponent_id,
            turn: game.state.settings.turn,
            enemy_units,
            spt,
        });
        Ok(())
    }
}

pub struct TrainingSample {
    features: RawFeatures,
    targets: DecomposedTargets,
    value: f32,
    progress: f32,
    progress_mask: f32,
    aux_ownership: Vec<f32>,
    aux_fog_units: Vec<f32>,
    aux_spt: [f32; 2],
    aux_opp_tech: Vec<f32>,
    aux_mask: f32,
    source_file: String,
}

impl TrainingSample {
    pub fn targets(&self) -> &DecomposedTargets {
        &self.targets
    }
    pub fn value(&self) -> f32 {
        self.value
    }
}

fn value_for_player(result: &ReplayResult, player_id: PlayerId) -> Result<f32, ReplayError> {
    if result.draw {
        return Ok(0.0);
    }
    if let Some(winner) = result.winner_player_id {
        return Ok(if winner == player_id { 1.0 } else { -1.0 });
    }
    if result.scores.is_empty() {
        return Err(ReplayError::Training(
            "result has neither winner, draw, nor scores".into(),
        ));
    }
    let own = result.scores.get(&player_id).copied().ok_or_else(|| {
        ReplayError::Training(format!("result has no score for acting player {player_id}"))
    })?;
    let best = result.scores.values().copied().max().unwrap_or(own);
    let winners = result
        .scores
        .values()
        .filter(|&&score| score == best)
        .count();
    Ok(if own == best && winners == 1 {
        1.0
    } else if own == best {
        0.0
    } else {
        -1.0
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DatasetManifest {
    pub dataset_schema_version: u32,
    pub replay_schema_version: u32,
    pub feature_schema_version: u32,
    pub action_schema_version: u32,
    pub num_channels: usize,
    pub player_state_dim: usize,
    pub map_width: usize,
    pub map_height: usize,
    pub num_action_types: usize,
    pub move_option_dim: usize,
    pub num_samples: usize,
    pub source_files: Vec<String>,
}

pub fn write_training_files(
    samples: &[TrainingSample],
    output: &Path,
    samples_per_file: usize,
) -> Result<Vec<PathBuf>, ReplayError> {
    if samples.is_empty() {
        return Err(ReplayError::Training(
            "no valid training samples were collected".into(),
        ));
    }
    if samples_per_file == 0 {
        return Err(ReplayError::Training(
            "samples-per-file must be positive".into(),
        ));
    }
    fs::create_dir_all(output).map_err(|e| {
        ReplayError::Training(format!(
            "cannot create output directory {}: {e}",
            output.display()
        ))
    })?;
    for entry in fs::read_dir(output).map_err(|e| {
        ReplayError::Training(format!("cannot read output directory {}: {e}", output.display()))
    })? {
        let entry = entry.map_err(|e| {
            ReplayError::Training(format!("cannot list output directory: {e}"))
        })?;
        let name = entry.file_name();
        if let Some(name_str) = name.to_str() {
            if name_str.starts_with("games_pro_") && name_str.ends_with(".safetensors") {
                fs::remove_file(entry.path()).map_err(|e| {
                    ReplayError::Training(format!(
                        "cannot remove stale shard {}: {e}",
                        entry.path().display()
                    ))
                })?;
            }
        }
    }
    let mut paths = Vec::new();
    for (part, chunk) in samples.chunks(samples_per_file).enumerate() {
        let path = output.join(format!("games_pro_{:06}.safetensors", part + 1));
        write_chunk(chunk, &path)?;
        paths.push(path);
    }
    Ok(paths)
}

fn write_chunk(samples: &[TrainingSample], path: &Path) -> Result<(), ReplayError> {
    let n = samples.len();
    let area = features::MAP_SIZE * features::MAP_SIZE;
    let tech_dim = TechnologyType::iter().count();
    let mut spatial = Vec::with_capacity(n * RawFeatures::spatial_len());
    let mut player = Vec::with_capacity(n * RawFeatures::player_len());
    let mut action = vec![0.0f32; n * NUM_ACTION_TYPES];
    let mut source = vec![0.0f32; n * area];
    let mut target = vec![0.0f32; n * area];
    let mut option = vec![0.0f32; n * 192];
    let mut values = Vec::with_capacity(n);
    let mut progress = Vec::with_capacity(n);
    let mut progress_mask = Vec::with_capacity(n);
    let mut ownership = Vec::with_capacity(n * area);
    let mut fog = Vec::with_capacity(n * area);
    let mut spt = Vec::with_capacity(n * 2);
    let mut opp_tech = Vec::with_capacity(n * tech_dim);
    let mut aux_mask = Vec::with_capacity(n);
    let mut source_files = BTreeSet::new();

    for (row, sample) in samples.iter().enumerate() {
        spatial.extend_from_slice(&sample.features.spatial);
        player.extend_from_slice(&sample.features.player);
        if sample.targets.action_type >= NUM_ACTION_TYPES {
            return Err(ReplayError::Training(format!(
                "action_type {} exceeds NUM_ACTION_TYPES={} in sample from {}",
                sample.targets.action_type, NUM_ACTION_TYPES, sample.source_file
            )));
        }
        action[row * NUM_ACTION_TYPES + sample.targets.action_type] = 1.0;
        if let Some(i) = sample.targets.source_spatial {
            if i >= area {
                return Err(ReplayError::Training(format!(
                    "source_spatial {} exceeds area={} in sample from {}",
                    i, area, sample.source_file
                )));
            }
            source[row * area + i] = 1.0;
        }
        if let Some(i) = sample.targets.target_spatial {
            if i >= area {
                return Err(ReplayError::Training(format!(
                    "target_spatial {} exceeds area={} in sample from {}",
                    i, area, sample.source_file
                )));
            }
            target[row * area + i] = 1.0;
        }
        if let Some(i) = sample.targets.target_type {
            if i >= 192 {
                return Err(ReplayError::Training(format!(
                    "target_type {} exceeds 192 in sample from {}",
                    i, sample.source_file
                )));
            }
            option[row * 192 + i] = 1.0;
        }
        values.push(sample.value);
        progress.push(sample.progress);
        progress_mask.push(sample.progress_mask);
        ownership.extend_from_slice(&sample.aux_ownership);
        fog.extend_from_slice(&sample.aux_fog_units);
        spt.extend_from_slice(&sample.aux_spt);
        opp_tech.extend_from_slice(&sample.aux_opp_tech);
        aux_mask.push(sample.aux_mask);
        source_files.insert(sample.source_file.clone());
    }

    let device = Device::Cpu;
    let tensor = |data: Vec<f32>, shape: &[usize]| -> Result<Tensor, ReplayError> {
        Tensor::from_vec(data, shape, &device)
            .map_err(|e| ReplayError::Training(format!("tensor construction failed: {e}")))
    };
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert(
        "spatial_maps".into(),
        tensor(spatial, &[n, RawFeatures::spatial_len()])?,
    );
    tensors.insert(
        "player_states".into(),
        tensor(player, &[n, RawFeatures::player_len()])?,
    );
    tensors.insert("values".into(), tensor(values, &[n, 1])?);
    tensors.insert(
        "action_type".into(),
        tensor(action, &[n, NUM_ACTION_TYPES])?,
    );
    tensors.insert("source_spatial".into(), tensor(source, &[n, area])?);
    tensors.insert("target_spatial".into(), tensor(target, &[n, area])?);
    tensors.insert("move_option".into(), tensor(option, &[n, 192])?);
    tensors.insert("progress".into(), tensor(progress, &[n, 1])?);
    tensors.insert("progress_mask".into(), tensor(progress_mask, &[n])?);
    tensors.insert("aux_ownership".into(), tensor(ownership, &[n, area])?);
    tensors.insert("aux_fog_units".into(), tensor(fog, &[n, area])?);
    tensors.insert("aux_spt".into(), tensor(spt, &[n, 2])?);
    tensors.insert("aux_opp_tech".into(), tensor(opp_tech, &[n, tech_dim])?);
    tensors.insert("aux_mask".into(), tensor(aux_mask, &[n])?);
    candle_core::safetensors::save(&tensors, path)
        .map_err(|e| ReplayError::Training(format!("cannot write {}: {e}", path.display())))?;

    let manifest = DatasetManifest {
        dataset_schema_version: DATASET_SCHEMA_VERSION,
        replay_schema_version: REPLAY_SCHEMA_VERSION,
        feature_schema_version: FEATURE_SCHEMA_VERSION,
        action_schema_version: ACTION_SCHEMA_VERSION,
        num_channels: features::NUM_CHANNELS,
        player_state_dim: RawFeatures::PLAYER_STATE_DIM,
        map_width: features::MAP_SIZE,
        map_height: features::MAP_SIZE,
        num_action_types: NUM_ACTION_TYPES,
        move_option_dim: 192,
        num_samples: n,
        source_files: source_files.into_iter().collect(),
    };
    let manifest_path = PathBuf::from(format!("{}.manifest.json", path.display()));
    let json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| ReplayError::Training(format!("manifest serialization failed: {e}")))?;
    fs::write(&manifest_path, json).map_err(|e| {
        ReplayError::Training(format!(
            "cannot write manifest {}: {e}",
            manifest_path.display()
        ))
    })?;
    Ok(())
}
