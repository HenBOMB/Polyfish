import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
import glob
import json
import os
import random
import gc
import time

# Disable advanced SDP backends that may lack kernel images on older Kaggle GPUs (e.g. P100/T4)
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(False)

# --- Configuration ---
BATCH_SIZE = 256
EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "2"))
# sqrt-scaled with the 64->256 batch bump (Adam responds closer to sqrt than
# linear scaling; 0.004 linear would risk instability on a small net).
# TRAIN_LR override: use a lower value (e.g. 0.0005) when re-running on the
# same data — the cosine scheduler restarts at this LR every invocation.
LEARNING_RATE = float(os.environ.get("TRAIN_LR", "0.002"))
# Weight on the value loss's contribution to the shared trunk's gradient.
# Default 3.0: with TD labels (Jul 2026) the value target carries real per-move
# signal, and at 1.0 its gradient (~0.02) is invisible next to policy (~2.0) —
# the trunk barely learns value-relevant features. Set to 0 to isolate whether
# value-gradient trunk interference corrodes the policy (bisect Arm C) —
# total_loss/policy_loss are unaffected either way.
VALUE_LOSS_WEIGHT = float(os.environ.get("VALUE_LOSS_WEIGHT", "3.0"))
# Detach the value head's input from the shared trunk (bisect Arm D). Unlike
# VALUE_LOSS_WEIGHT=0, the value head's own layers (v_pool_conv/v_fc_shared/
# v_win) still get full-strength gradient — only the trunk is shielded.
# Forward-pass values are identical either way, so this is training-only and
# needs no change on the Rust/candle inference side.
DETACH_VALUE_TRUNK = os.environ.get("DETACH_VALUE_TRUNK", "0") == "1"
# Weight on the auxiliary per-tile final-ownership loss (KataGo-style). Kept
# small so the dense spatial gradient shapes the trunk without competing with
# policy/value. Set to 0 to disable. Training-only: search never reads the
# ownership head, and the head reads the trunk directly (NOT gated by
# DETACH_VALUE_TRUNK — trunk gradient is its entire purpose).
OWNERSHIP_LOSS_WEIGHT = float(os.environ.get("OWNERSHIP_LOSS_WEIGHT", "0.15"))

# Device selection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
# Each backend gets its own try/except so a probe failure in one
# doesn't skip the other (e.g. MPS raising on Linux kills the CUDA path).
DEVICE = "cpu"
try:
    if torch.cuda.is_available():
        t = torch.tensor([1.0], device="cuda")
        DEVICE = "cuda"
except Exception as e:
    print(f"Warning: CUDA available but failed to initialize ({e}).")

if DEVICE == "cpu":
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            t = torch.tensor([1.0], device="mps")
            DEVICE = "mps"
    except Exception as e:
        print(f"Warning: MPS available but failed to initialize ({e}).")

print(f"Device: {DEVICE}  |  torch {torch.__version__}  |  CUDA build: {torch.version.cuda}  |  cuda.is_available(): {torch.cuda.is_available()}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif DEVICE == "cpu":
    print("WARNING: Running on CPU! Training will be extremely slow.")
    
# Architecture matching Rust `network.rs` (decomposed policy + auxiliary values)
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.c1(x)))
        out = self.bn2(self.c2(out))
        out += residual
        out = self.relu(out)
        return out

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        
    def forward(self, q, kv):
        # q: (B, Nq, D) - spatial tokens
        # kv: (B, Nkv, D) - player state tokens
        attn_out, _ = self.attn(q, kv, kv)
        return self.norm(q + attn_out)

class PolyZeroNet(nn.Module):
    """
    Enhanced architecture with:
    - Player state embedding (global context)
    - 20 ResBlocks (increased capacity for A40)
    - 7 decomposed policy heads
    - 1 value head (win)
    """
    def __init__(self, spatial_channels, player_state_dim, map_height, map_width):
        super().__init__()
        self.map_height = map_height
        self.map_width = map_width
        self.filters = 64
        
        # Player state tokens: Project each of the 10 features into 64-dim embeddings
        # We learn a base embedding for each feature index and scale it by the value
        self.player_feature_embeddings = nn.Parameter(torch.randn(player_state_dim, self.filters))
        self.player_pos_embeddings = nn.Parameter(torch.randn(player_state_dim, self.filters))
        self.player_fc = nn.Linear(self.filters, self.filters)
        self.player_relu = nn.ReLU()
        
        # Initial conv on spatial features
        self.conv1 = nn.Conv2d(spatial_channels, self.filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(self.filters)
        self.relu = nn.ReLU()
        
        # ResBlocks (Match Rust config)
        self.res_blocks = nn.ModuleList([ResBlock(self.filters) for _ in range(6)])
        
        # --- Cross Attention Layer ---
        # Allow each spatial tile (Q) to attend to global player features (K,V)
        self.cross_attention = CrossAttention(self.filters, nhead=4)
        
        # --- Decomposed Policy Heads ---
        # 1. Action Type (11 categories: Attack, Step, Build, etc.)
        self.p_pool_conv = nn.Conv2d(self.filters, 1, 1)
        self.p_pool_bn = nn.BatchNorm2d(1)
        self.p_fc_shared = nn.Linear(map_height * map_width, self.filters)
        self.pi_action = nn.Linear(self.filters, 11)
        
        # 2. Unified Options (192 categories: Structures, Units, Techs, Abilities, Rewards)
        self.pi_option = nn.Linear(self.filters, 192)
        
        # 3. Spatial Heads (Source and Target tile selection)
        self.pi_source = nn.Conv2d(self.filters, 1, 1)
        self.pi_target = nn.Conv2d(self.filters, 1, 1)
        
        # --- Value Heads ---
        self.v_pool_conv = nn.Conv2d(self.filters, 1, 1)
        self.v_pool_bn = nn.BatchNorm2d(1)
        self.v_fc_shared = nn.Linear(1 * map_height * map_width, self.filters)
        self.v_win = nn.Linear(self.filters, 1)
        self.v_progress = nn.Linear(self.filters, 1)
        # Aux spatial head: predicted end-of-game per-tile ownership
        # (+1 mine / -1 enemy / 0 neutral), trained with a small-weight MSE
        # purely to densify trunk gradient. Never used by search.
        self.v_ownership = nn.Conv2d(self.filters, 1, 1)

    def forward(self, spatial_map, player_state):
        batch_size = spatial_map.size(0)
        
        # 1. Spatial Backbone
        x = self.relu(self.bn1(self.conv1(spatial_map)))
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 2. Prepare Cross-Attention Inputs
        spatial_tokens = x.flatten(2).transpose(1, 2)
        player_tokens = player_state.unsqueeze(-1) * self.player_feature_embeddings.unsqueeze(0)
        player_tokens = player_tokens + self.player_pos_embeddings.unsqueeze(0)
        player_tokens = self.player_relu(self.player_fc(player_tokens))
        
        # 3. Apply Cross-Attention
        x_attended = self.cross_attention(spatial_tokens, player_tokens)
        x = x_attended.transpose(1, 2).view(batch_size, self.filters, self.map_height, self.map_width)
        
        # --- Policy Heads ---
        p_pooled = self.relu(self.p_pool_bn(self.p_pool_conv(x)))
        p_pooled = p_pooled.flatten(1)
        p_latent = self.relu(self.p_fc_shared(p_pooled))
        
        policy = {}
        policy['action_type'] = self.pi_action(p_latent)
        policy['move_option'] = self.pi_option(p_latent)
        policy['source_spatial'] = self.pi_source(x).flatten(1)
        policy['target_spatial'] = self.pi_target(x).flatten(1)
        
        # --- Value Heads ---
        v_input = x.detach() if DETACH_VALUE_TRUNK else x
        v_pooled = self.relu(self.v_pool_bn(self.v_pool_conv(v_input)))
        v_pooled = v_pooled.flatten(1)
        v_latent = self.relu(self.v_fc_shared(v_pooled))
        
        values = {}
        values['win'] = self.v_win(v_latent)
        values['progress'] = self.v_progress(v_latent)
        values['ownership'] = self.v_ownership(x).flatten(1)

        return policy, values

def compute_loss(policy_pred, values_pred, policy_targets, value_target):
    """
    Compute multi-head loss using decomposed targets.
    policy_targets is a dict containing the 7 target tensors.
    """
    total_policy_loss = 0.0
    
    # Loss weights for each head (tune as needed)
    weights = {
        'action_type': 1.0,
        'source_spatial': 1.0,
        'target_spatial': 1.0,
        # 'structure_option': 0.2,
        # 'unit_option': 0.2,
        # 'tech_option': 0.2,
        # 'ability_option': 0.2,
        # 'reward_choice': 0.1
        'move_option': 1.0,
    }
    
    # Helper for cross entropy with soft targets (probabilities)
    def soft_cross_entropy(logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()

    for head_name, target in policy_targets.items():
        if head_name in policy_pred:
            pred = policy_pred[head_name]
            head_loss = soft_cross_entropy(pred, target)
            total_policy_loss += head_loss * weights.get(head_name, 1.0)
            
    loss_win = nn.MSELoss()(values_pred['win'], value_target['win'])

    loss_progress = 0.0
    if 'progress' in value_target and 'progress' in values_pred:
        loss_progress = nn.MSELoss()(values_pred['progress'], value_target['progress'])

    # Prioritize winning/losing.
    value_loss = VALUE_LOSS_WEIGHT * loss_win + loss_progress

    # Aux ownership loss, masked per sample so pre-ownership game files
    # (mask=0) contribute nothing. Reported separately from value_loss to
    # keep the value_loss series and value_r2 comparable across runs.
    ownership_loss = torch.tensor(0.0, device=values_pred['win'].device)
    if OWNERSHIP_LOSS_WEIGHT > 0 and 'ownership' in value_target and 'ownership' in values_pred:
        mask = value_target['ownership_mask']  # [B, 1]
        n_valid = mask.sum()
        if n_valid > 0:
            per_sample = ((values_pred['ownership'] - value_target['ownership']) ** 2).mean(dim=1, keepdim=True)
            ownership_loss = OWNERSHIP_LOSS_WEIGHT * (per_sample * mask).sum() / n_valid

    # Total loss
    total_loss = total_policy_loss + value_loss + ownership_loss

    return total_loss, total_policy_loss, value_loss, ownership_loss

def batch_report_indices(total_batches, max_reports=10):
    """Pick up to `max_reports` evenly spaced batch numbers to log."""
    if total_batches <= 0:
        return set()
    if total_batches <= max_reports:
        return set(range(1, total_batches + 1))
    indices = set()
    for i in range(max_reports):
        batch_num = 1 + i * (total_batches - 1) // (max_reports - 1)
        indices.add(batch_num)
    return indices

def train():
    # 1. Load Data
    fresh_files = glob.glob("games_*.safetensors")
    archive_files = sorted(glob.glob("archive/games_*.safetensors"), key=os.path.getmtime, reverse=True)
    # Replay window in FILES; run_training_loop.sh exports REPLAY_BUFFER_FILES
    # scaled by its -g so the buffer stays ~constant in GAMES (default 10
    # files ≈ 700 games at 64 games/file). Each sample is trained ~20 times
    # before pruning; reduces overfitting risk. Archive pruning keeps window+1.
    replay_buffer_size = int(os.environ.get("REPLAY_BUFFER_FILES", "10"))
    # Teacher anchor: always mix these into every iteration so gradients keep
    # pulling toward known-good play regardless of self-play drift (RLHF-style
    # reference anchor). Never archived or pruned.
    teacher_files = sorted(glob.glob("teachers/games_*.safetensors"))
    game_files = fresh_files + archive_files[:replay_buffer_size] + teacher_files

    if not game_files:
        print("No training data found (checked ./, ./archive/, and ./teachers/)!")
        return

    print(f"Training on {len(game_files)} files ({len(fresh_files)} fresh, "
          f"{len(archive_files[:replay_buffer_size])} archived, {len(teacher_files)} teacher).")
    
    # 2. Init Model
    MAP_SIZE = 11
    SPATIAL_CHANNELS = 142  # 136 + 6 fog-memory channels (see notes-memory.md)
    PLAYER_STATE_DIM = 16

    model = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE).to(DEVICE)
    if os.path.exists("model.safetensors"):
        try:
            # strict=False lets checkpoints that predate the aux ownership
            # head load in place (the head keeps its fresh init). Any OTHER
            # mismatch is still fatal so a wrong-architecture checkpoint
            # can't silently half-load.
            missing, unexpected = model.load_state_dict(load_file("model.safetensors"), strict=False)
            hard_missing = [k for k in missing if not k.startswith("v_ownership.")]
            if hard_missing or unexpected:
                raise RuntimeError(f"state_dict mismatch: missing={hard_missing} unexpected={list(unexpected)}")
            if missing:
                print(f"Checkpoint predates ownership head; initializing {missing} fresh.")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting from scratch.")
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use CosineAnnealing with Warm Restarts for better convergence on short cycles
    # T_0=5 means it resets every 5 epochs (which is exactly our run length)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS, T_mult=1, eta_min=1e-5)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        total_o_loss = 0
        total_batches = 0
        # Streaming mean/variance of the value targets seen this epoch, so
        # value_r2 (below) compares MSE against the actual training-mix
        # variance instead of a guess — small MSE alone doesn't mean the head
        # fits anything if the targets barely vary.
        target_sum = 0.0
        target_sumsq = 0.0
        target_n = 0

        random.shuffle(game_files)

        # Process in chunks. All files in a chunk are held in RAM at once, so
        # lower TRAIN_CHUNK_FILES when individual game files are large.
        CHUNK_SIZE = int(os.environ.get("TRAIN_CHUNK_FILES", "10"))
        num_chunks = (len(game_files) + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        report_batch_indices = None
        epoch_batch_estimate = None

        for i in range(0, len(game_files), CHUNK_SIZE):
            chunk_files = game_files[i : i + CHUNK_SIZE]
            chunk_idx = i // CHUNK_SIZE + 1
            print(f"Epoch {epoch+1}: Loading chunk {chunk_idx}/{num_chunks} ({len(chunk_files)} files)...")
            
            # Temporary storage for chunk data
            c_spatial = []
            c_player = []
            c_win = []
            c_progress = []
            c_ownership = []
            c_ownership_mask = []
            c_n_samples = []

            c_heads = {
                'action_type': [], 'source_spatial': [], 'target_spatial': [], 'move_option': []
            }

            for f in chunk_files:
                try:
                    data = load_file(f)
                    c_spatial.append(data["spatial_maps"])
                    c_player.append(data["player_states"])
                    c_win.append(data["values"])
                    if "progress" in data:
                        c_progress.append(data["progress"])
                    else:
                        c_progress.append(torch.zeros_like(data["values"]))
                    n_samples = data["values"].shape[0]
                    if "ownership" in data:
                        c_ownership.append(data["ownership"])
                        c_ownership_mask.append(torch.ones(n_samples, 1))
                    else:
                        # Pre-ownership file: zero-fill and mask out.
                        c_ownership.append(torch.zeros(n_samples, MAP_SIZE * MAP_SIZE))
                        c_ownership_mask.append(torch.zeros(n_samples, 1))
                    
                    c_n_samples.append(n_samples)
                    
                    # Load all policy heads
                    for head in c_heads.keys():
                        if head in data:
                            c_heads[head].append(data[head])
                        else:
                            c_heads[head].append(None)
                            
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    continue
            
            if not c_spatial:
                continue
                
            # Stack into tensors
            try:
                spatial_maps = torch.cat(c_spatial)
                player_states = torch.cat(c_player)
                
                targets_win = torch.cat(c_win)
                targets_progress = torch.cat(c_progress) if c_progress else None
                targets_ownership = torch.cat(c_ownership) if c_ownership else None
                targets_ownership_mask = torch.cat(c_ownership_mask) if c_ownership_mask else None

                target_heads = {}
                for head, tensors in c_heads.items():
                    valid_tensors = [t for t in tensors if t is not None]
                    if valid_tensors:
                        head_shape = valid_tensors[0].shape[1:]
                        head_dtype = valid_tensors[0].dtype
                        filled = []
                        for i, t in enumerate(tensors):
                            if t is not None:
                                filled.append(t)
                            else:
                                filled.append(torch.zeros((c_n_samples[i], *head_shape), dtype=head_dtype))
                        target_heads[head] = torch.cat(filled)
                    
            except RuntimeError as e:
                print(f"OOM loading chunk: {e}")
                continue
            
            # Cleanup lists
            del c_spatial, c_player, c_win, c_progress, c_ownership, c_ownership_mask
            gc.collect()
            
            dataset_size = len(spatial_maps)
            print(f"  Loaded {dataset_size} samples.")

            if epoch_batch_estimate is None and len(chunk_files) > 0:
                est_samples = int(dataset_size / len(chunk_files) * len(game_files))
                epoch_batch_estimate = (est_samples + BATCH_SIZE - 1) // BATCH_SIZE
                report_batch_indices = batch_report_indices(epoch_batch_estimate)
                if epoch_batch_estimate <= 10:
                    print(f"  Reporting all {epoch_batch_estimate} batches.")
                else:
                    print(f"  Reporting ~10/{epoch_batch_estimate} sampled batches.")

            indices = torch.randperm(dataset_size)
            num_batches_in_chunk = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE
            chunk_start_time = time.time()

            for batch_num, j in enumerate(range(0, dataset_size, BATCH_SIZE), start=1):
                batch_idx = indices[j : j + BATCH_SIZE]
                
                batch_spatial = spatial_maps[batch_idx].to(DEVICE)
                batch_player = player_states[batch_idx].to(DEVICE)
                
                batch_values = {
                    'win': targets_win[batch_idx].to(DEVICE),
                }
                target_sum += batch_values['win'].sum().item()
                target_sumsq += (batch_values['win'] * batch_values['win']).sum().item()
                target_n += batch_values['win'].numel()

                if targets_progress is not None:
                    batch_values['progress'] = targets_progress[batch_idx].to(DEVICE)
                if targets_ownership is not None:
                    batch_values['ownership'] = targets_ownership[batch_idx].to(DEVICE)
                    batch_values['ownership_mask'] = targets_ownership_mask[batch_idx].to(DEVICE)
                batch_targets = {}
                for head, tensor in target_heads.items():
                    batch_targets[head] = tensor[batch_idx].to(DEVICE)
                
                # Reshape spatial to (B, C, H, W)
                batch_spatial = batch_spatial.view(-1, SPATIAL_CHANNELS, MAP_SIZE, MAP_SIZE)
                
                # # --- DATA AUGMENTATION (Dihedral Group D4) ---
                
                optimizer.zero_grad()
                
                policy_pred, values_pred = model(batch_spatial, batch_player)
                
                loss, p_loss, v_loss, o_loss = compute_loss(policy_pred, values_pred, batch_targets, batch_values)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_p_loss += p_loss.item()
                total_v_loss += v_loss.item()
                total_o_loss += o_loss.item()
                total_batches += 1

                elapsed = time.time() - chunk_start_time
                global_batch_num = total_batches
                if report_batch_indices and global_batch_num in report_batch_indices:
                    batches_per_sec = batch_num / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  Epoch {epoch+1} batch {global_batch_num}"
                        f"{f'/{epoch_batch_estimate}' if epoch_batch_estimate else ''} "
                        f"(chunk {chunk_idx}/{num_chunks} {batch_num}/{num_batches_in_chunk}) "
                        f"- loss: {total_loss/total_batches:.4f} "
                        f"(policy: {total_p_loss/total_batches:.4f}, value: {total_v_loss/total_batches:.4f}, "
                        f"ownership: {total_o_loss/total_batches:.4f}) "
                        f"- {batches_per_sec:.1f} batch/s"
                    )

            del spatial_maps, player_states, targets_win, targets_ownership, targets_ownership_mask, target_heads
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()
            gc.collect()

        if total_batches > 0:
            avg_loss = total_loss / total_batches
            avg_p_loss = total_p_loss / total_batches
            avg_v_loss = total_v_loss / total_batches
            avg_o_loss = total_o_loss / total_batches
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f}, Ownership: {avg_o_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - No data processed")

        scheduler.step()

    final_loss = total_loss / total_batches if total_batches > 0 else 0.0
    final_p_loss = total_p_loss / total_batches if total_batches > 0 else 0.0
    final_v_loss = total_v_loss / total_batches if total_batches > 0 else 0.0
    final_o_loss = total_o_loss / total_batches if total_batches > 0 else 0.0

    # R^2 of the value head against the LAST epoch's own target distribution:
    # 1 - MSE/Var. Small MSE alone is meaningless if the targets barely vary
    # (a constant-mean predictor would also score low MSE) — this is the
    # number that actually says whether the head explains anything. final_v_loss
    # has VALUE_LOSS_WEIGHT baked in (see compute_loss); unweight it first so
    # it's comparable to the raw target variance.
    if target_n > 0 and VALUE_LOSS_WEIGHT > 0:
        target_mean = target_sum / target_n
        target_var = target_sumsq / target_n - target_mean * target_mean
        raw_v_mse = final_v_loss / VALUE_LOSS_WEIGHT
        value_r2 = 1.0 - raw_v_mse / target_var if target_var > 1e-8 else 0.0
    else:
        value_r2 = 0.0

    # 4. Save Model in f16 for blazing fast CPU inference
    half_state = {k: v.half() for k, v in model.state_dict().items()}
    save_file(half_state, "model.safetensors")

    with open(".last_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": round(final_loss, 4),
                "policy_loss": round(final_p_loss, 4),
                "value_loss": round(final_v_loss, 4),
                "ownership_loss": round(final_o_loss, 4),
                "value_r2": round(value_r2, 4),
            },
            f,
        )

if __name__ == "__main__":
    train()

