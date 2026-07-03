use crate::ai::features::{GameFeatures, state_to_tensor};
use crate::ai::mapper::{DecomposedMapper, DecomposedTargets};
use crate::moves::Move;
use crate::states::GameState;
use candle_core::{Device, Tensor};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Records game states and human moves for training
pub struct GameRecorder {
    // Buffer stores: (Features, PlayerState, DecomposedTargets, ValueTargets)
    buffer: Mutex<Vec<(GameFeatures, Tensor, DecomposedTargets, Vec<f32>)>>,
    device: Device,
}

impl GameRecorder {
    pub fn new() -> Self {
        // Select best available device: Metal (macOS) > CUDA (NVIDIA) > CPU
        let device = Device::metal_if_available(0)
            .or_else(|_| Device::cuda_if_available(0))
            .unwrap_or(Device::Cpu);
        Self {
            buffer: Mutex::new(Vec::new()),
            device,
        }
    }

    /// Record a step (State + Human Move).
    /// Assumes the move is the "ground truth" (prob 1.0).
    pub fn record_step(
        &self,
        state: &GameState,
        move_obj: &dyn Move,
        eco_val: f32, // Normalized economy score (0-1)
        mil_val: f32, // Normalized military score (0-1)
    ) {
        // 1. Convert state to features
        let pov = state.settings.current_player_turn_id;
        let features = match state_to_tensor(state, pov, &self.device) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Failed to convert state to tensor: {}", e);
                return;
            }
        };

        // 2. Map move to Targets
        let map_size = state.map_size() as usize;
        let targets = DecomposedMapper::move_to_targets(move_obj, map_size);

        // 3. Value target
        // We'll set Win=0.0 (unknown) for now
        let value_target = vec![0.0, eco_val, mil_val];

        let mut buf = self.buffer.lock().unwrap();
        // features.spatial_map is [1, C, H, W], features.player_state is [1, P_DIM]
        let p_state = features.player_state.clone();
        buf.push((features, p_state, targets, value_target));
    }

    /// Save buffered games to .safetensors
    pub fn save(&self) -> anyhow::Result<String> {
        let mut buf = self.buffer.lock().unwrap();
        if buf.is_empty() {
            return Ok("Buffer empty".to_string());
        }

        let n = buf.len();
        println!("Saving {} human steps...", n);

        // We need to collate data into tensors matching train.rs expectations:
        // "spatial_maps": [N, C, H, W]
        // "player_states": [N, P_DIM]
        // "action_type": [N, 11]
        // "source_spatial": [N, H*W]
        // "target_spatial": [N, H*W]
        // "move_option": [N, 192]
        // "values": [N, 1] (win)
        // "eco_targets": [N, 1]
        // "mil_targets": [N, 1] (optional in train.rs, but we can save them)

        let (first_feats, _, _, _) = &buf[0];
        let (_b, _c, h, w) = first_feats.spatial_map.shape().dims4()?;
        let map_area = h * w;

        // -- Accumulate Tensors --
        let mut t_spatial = Vec::with_capacity(n);
        let mut t_player = Vec::with_capacity(n);

        // Policy
        let mut t_action = Vec::with_capacity(n);
        let mut t_source = Vec::with_capacity(n);
        let mut t_target = Vec::with_capacity(n);
        let mut t_option = Vec::with_capacity(n);

        // Value
        let mut t_win = Vec::with_capacity(n);
        let mut t_eco = Vec::with_capacity(n);
        let mut t_mil = Vec::with_capacity(n);

        // Helper for one-hot
        fn one_hot_tensor(
            idx: Option<usize>,
            size: usize,
            device: &Device,
        ) -> anyhow::Result<Tensor> {
            let mut v = vec![0.0f32; size];
            if let Some(i) = idx {
                if i < size {
                    v[i] = 1.0;
                }
            }
            Ok(Tensor::from_vec(v, (1, size), device)?)
        }

        for (feat, p_state, tgt, val) in buf.iter() {
            t_spatial.push(feat.spatial_map.clone());
            t_player.push(p_state.clone());

            t_action.push(one_hot_tensor(Some(tgt.action_type), 11, &self.device)?);
            t_source.push(one_hot_tensor(tgt.source_spatial, map_area, &self.device)?);
            t_target.push(one_hot_tensor(tgt.target_spatial, map_area, &self.device)?);
            t_option.push(one_hot_tensor(tgt.target_type, 192, &self.device)?);

            t_win.push(Tensor::from_vec(vec![val[0]], (1, 1), &self.device)?);
            t_eco.push(Tensor::from_vec(vec![val[1]], (1, 1), &self.device)?);
            t_mil.push(Tensor::from_vec(vec![val[2]], (1, 1), &self.device)?);
        }

        let batch_spatial = Tensor::cat(&t_spatial, 0)?;
        let batch_player = Tensor::cat(&t_player, 0)?;

        let batch_action = Tensor::cat(&t_action, 0)?;
        let batch_source = Tensor::cat(&t_source, 0)?;
        let batch_target = Tensor::cat(&t_target, 0)?;
        let batch_option = Tensor::cat(&t_option, 0)?;

        let batch_win = Tensor::cat(&t_win, 0)?;
        let batch_eco = Tensor::cat(&t_eco, 0)?;
        let batch_mil = Tensor::cat(&t_mil, 0)?;

        // Save
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!("games_human_{}.safetensors", timestamp);

        use candle_core::safetensors::save;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("spatial_maps".to_string(), batch_spatial);
        tensors.insert("player_states".to_string(), batch_player);

        tensors.insert("action_type".to_string(), batch_action);
        tensors.insert("source_spatial".to_string(), batch_source);
        tensors.insert("target_spatial".to_string(), batch_target);
        tensors.insert("move_option".to_string(), batch_option);

        tensors.insert("values".to_string(), batch_win);
        tensors.insert("eco_targets".to_string(), batch_eco);
        tensors.insert("mil_targets".to_string(), batch_mil);

        save(&tensors, &filename)?;

        // Clear buffer
        buf.clear();

        Ok(format!("Saved {} samples to {}", n, filename))
    }
}
