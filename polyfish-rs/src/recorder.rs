use crate::ai::features::{GameFeatures, state_to_tensor};
use crate::ai::mapper::{DecomposedMapper, DecomposedTargets};
use crate::moves::Move;
use crate::states::GameState;
use candle_core::{Device, Tensor};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Records game states and human moves for training
pub struct GameRecorder {
    // Buffer stores: (Features, ActionType, SourceSpatial, TargetSpatial, TargetType, ValueTarget)
    // We store raw indices/targets rather than one-hot vectors to save space/complexity here.
    // The training loop (or a converter) will handle one-hot encoding if needed,
    // but `self_play.rs` usually saves aggregated probability distributions.
    // Since human play is "hard" labels (prob 1.0), we can just store the indices
    // and expand them to one-hot tensors during save.
    buffer: Mutex<Vec<(GameFeatures, DecomposedTargets, Vec<f32>)>>,
    device: Device,
}

impl GameRecorder {
    pub fn new() -> Self {
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        Self {
            buffer: Mutex::new(Vec::new()),
            device,
        }
    }

    /// Record a step (State + Human Move)
    /// We assume the human move is "correct" (probability 1.0)
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
        // We'll assume Win=0.0 (unknown) for now, or maybe 1.0 if we assume human is god?
        // Let's stick to 0.0 for now, as we don't know the outcome.
        // Or we can update it later if we track the game result.
        // For supervised learning of *policy*, value is less critical, but good to have Eco/Mil.
        let value_target = vec![0.0, eco_val, mil_val];

        let mut buf = self.buffer.lock().unwrap();
        buf.push((features, targets, value_target));
    }

    /// Save buffered games to .safetensors
    pub fn save(&self) -> anyhow::Result<String> {
        let mut buf = self.buffer.lock().unwrap();
        if buf.is_empty() {
            return Ok("Buffer empty".to_string());
        }

        let n = buf.len();
        println!("Saving {} human steps...", n);

        // We need to collate data into tensors.
        // features: [N, C, H, W]
        // policy_action: [N, 11] (one-hot)
        // policy_src: [N, H*W] (one-hot)
        // policy_tgt: [N, H*W] (one-hot)
        // policy_opt: [N, 192] (one-hot)
        // value: [N, 3]

        // Actually, `self_play.rs` saves them as:
        // "input": [N, C, H, W]
        // "policy_targets": [N, TOTAL_POLICY_SIZE] (concatenated flattened one-hots)
        // "value_targets": [N, 3]

        // We need `TOTAL_POLICY_SIZE`.
        // 11 + 900 + 900 + 192 = 2003 (for 30x30 map? No, map size varies).
        // Let's assume standard map size or dynamic?
        // PolyZeroNet uses dynamic map size, but policy heads are usually fixed size output?
        // `network.rs` -> Output is usually MapSize*MapSize.
        // If we train on multiple map sizes, we usually mask or pad.
        // `self_play.rs` handles this.

        // For simplicity, let's assume the map size of the FIRST sample in the buffer
        if n == 0 {
            return Ok("Empty".into());
        }

        // Helper to create one-hot vector
        fn one_hot(idx: Option<usize>, size: usize) -> Vec<f32> {
            let mut v = vec![0.0; size];
            if let Some(i) = idx {
                if i < size {
                    v[i] = 1.0;
                }
            }
            v
        }

        // Assuming all samples have same map size for now (typical if user plays one game).
        // But human might restart on different map sizes.
        // That's tricky. Tensors must be uniform.
        // We should Group by Map Size or just enforce Single Batch per Save.
        // Given the use case, saving often is fine.

        let (first_feats, _, _) = &buf[0];
        let (_b, _c, h, w) = first_feats.spatial_map.shape().dims4()?;
        let map_area = h * w; // e.g. 11*11=121

        let mut input_list = Vec::with_capacity(n);
        let mut policy_list = Vec::with_capacity(n);
        let mut value_list = Vec::with_capacity(n);

        for (feat, tgt, val_tgt) in buf.iter() {
            // 1. Input
            // feat.spatial_map is [1, C, H, W]
            // We need to flatten or keep it... `Tensor::cat` expects [N, ...].
            input_list.push(feat.spatial_map.clone());

            // 2. Policy One-Hots
            // Sizes: Action=11, Src=MapArea, Tgt=MapArea, Opt=192
            let mut p_vec = Vec::new();
            p_vec.extend(one_hot(Some(tgt.action_type), 11));
            p_vec.extend(one_hot(tgt.source_spatial, map_area));
            p_vec.extend(one_hot(tgt.target_spatial, map_area));
            p_vec.extend(one_hot(tgt.target_type, 192));

            // Convert to Tensor
            let p_tensor = Tensor::from_vec(p_vec, (1, 11 + map_area * 2 + 192), &self.device)?;
            policy_list.push(p_tensor);

            // 3. Value
            let v_tensor = Tensor::from_vec(val_tgt.clone(), (1, 3), &self.device)?;
            value_list.push(v_tensor);
        }

        let batch_input = Tensor::cat(&input_list, 0)?;
        let batch_policy = Tensor::cat(&policy_list, 0)?;
        let batch_value = Tensor::cat(&value_list, 0)?;

        // Save to file
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!("human_games_{}.safetensors", timestamp);

        use candle_core::safetensors::save;
        let mut tensors = std::collections::HashMap::new();
        tensors.insert("input".to_string(), batch_input);
        tensors.insert("policy_targets".to_string(), batch_policy);
        tensors.insert("value_targets".to_string(), batch_value);

        save(&tensors, &filename)?;

        // Clear buffer after save
        buf.clear();

        Ok(format!("Saved {} samples to {}", n, filename))
    }
}
