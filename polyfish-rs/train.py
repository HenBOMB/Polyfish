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

# --- Configuration ---
# Larger batches amortize MPS/CUDA per-op dispatch overhead (fixed cost per
# kernel launch regardless of tensor size) over more samples — measured Jul
# 2026: this net is small enough (64 filters) that dispatch overhead, not
# compute, dominates batch time on Apple Silicon. Default kept at 256 for
# reproducibility; try TRAIN_BATCH_SIZE=1024+ and retune TRAIN_LR (Adam
# responds closer to sqrt-scaling than linear) as a deliberate experiment.
BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "256"))
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
# Random rot90/flip per batch (D4 dihedral): 8x effective spatial data.
# Geometrically valid (no feature plane, player scalar, or rule is
# orientation-dependent) but OFF by default: enabling it MID-RUN on the
# 586K-param net collapsed play for ~8 iterations (run 1783556259 — policy
# lost its orientation-specific fit, degraded games then fed back through
# self-play). Opt in only for from-scratch runs, where the net never learns
# orientation shortcuts to begin with.
AUGMENT_D4 = os.environ.get("AUGMENT_D4", "0") == "1"
# Auxiliary training-only heads (end-game ownership / fog occupancy / SPT+5 /
# opponent tech). Rust inference loads model.safetensors by name and never
# reads these. Targets ship in games files from Jul 2026; files without them
# (old archives, teachers) are masked out per sample, never zero-filled.
AUX_DIMS = {'aux_ownership': 121, 'aux_fog_units': 121, 'aux_spt': 2, 'aux_opp_tech': 42}
AUX_WEIGHTS = {
    'aux_ownership': float(os.environ.get("AUX_OWN_W", "0.3")),
    'aux_fog_units': float(os.environ.get("AUX_FOG_W", "0.2")),
    'aux_spt': float(os.environ.get("AUX_SPT_W", "0.1")),
    'aux_opp_tech': float(os.environ.get("AUX_TECH_W", "0.1")),
}

# Device selection: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
try:
    if torch.backends.mps.is_available():
        # Apple Silicon Metal Performance Shaders
        t = torch.tensor([1.0], device="mps")
        DEVICE = "mps"
    elif torch.cuda.is_available():
        # NVIDIA CUDA
        t = torch.tensor([1.0], device="cuda")
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except Exception as e:
    print(f"Warning: GPU available but failed to initialize ({e}). Fallback to CPU.")
    DEVICE = "cpu"

# Architecture matching Rust `network.rs` (decomposed policy + auxiliary values)
# GroupNorm everywhere (no BatchNorm): per-sample statistics mean the exact
# same function runs in train and eval mode — no running stats, no train/serve
# gap (the BN calibration gap measured Jul 2026 was eval R² -0.75..-2.0 vs
# train +0.50). Must stay mirrored with the Rust backends (GN_GROUPS there).
GN_GROUPS = 8

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(GN_GROUPS, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(GN_GROUPS, channels)
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
        self.player_fc = nn.Linear(self.filters, self.filters)
        self.player_relu = nn.ReLU()
        
        # Initial conv on spatial features
        self.conv1 = nn.Conv2d(spatial_channels, self.filters, 3, padding=1)
        self.bn1 = nn.GroupNorm(GN_GROUPS, self.filters)
        self.relu = nn.ReLU()
        
        # ResBlocks (Match Rust config)
        self.res_blocks = nn.ModuleList([ResBlock(self.filters) for _ in range(6)])
        
        # --- Cross Attention Layer ---
        # Allow each spatial tile (Q) to attend to global player features (K,V)
        self.cross_attention = CrossAttention(self.filters, nhead=4)
        
        # --- Decomposed Policy Heads ---
        # 1. Action Type (11 categories: Attack, Step, Build, etc.)
        # No norm and no activation on the 1-channel pools: a per-sample norm
        # would erase the map's overall level, and an unnormed ReLU here goes
        # irreversibly dead (killed the value/action heads, Jul 2026).
        self.p_pool_conv = nn.Conv2d(self.filters, 1, 1)
        self.p_fc_shared = nn.Linear(map_height * map_width, self.filters)
        self.pi_action = nn.Linear(self.filters, 11)
        
        # 2. Unified Options (192 categories: Structures, Units, Techs, Abilities, Rewards)
        self.pi_option = nn.Linear(self.filters, 192)
        
        # 3. Spatial Heads (Source and Target tile selection)
        self.pi_source = nn.Conv2d(self.filters, 1, 1)
        self.pi_target = nn.Conv2d(self.filters, 1, 1)
        
        # --- Value Heads ---
        self.v_pool_conv = nn.Conv2d(self.filters, 1, 1)
        self.v_fc_shared = nn.Linear(1 * map_height * map_width, self.filters)
        self.v_win = nn.Linear(self.filters, 1)
        self.v_progress = nn.Linear(self.filters, 1)

        # --- Aux Heads (training-only; Rust inference ignores these keys) ---
        # Spatial pair reads the trunk directly; scalar pair reads a global
        # average pool — deliberately not v_latent, whose 1-channel pool is a
        # known bottleneck.
        self.aux_own = nn.Conv2d(self.filters, 1, 1)
        self.aux_fog = nn.Conv2d(self.filters, 1, 1)
        self.aux_spt = nn.Linear(self.filters, 2)
        self.aux_opp_tech = nn.Linear(self.filters, 42)

    def forward(self, spatial_map, player_state):
        batch_size = spatial_map.size(0)
        
        # 1. Spatial Backbone
        x = self.relu(self.bn1(self.conv1(spatial_map)))
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 2. Prepare Cross-Attention Inputs
        spatial_tokens = x.flatten(2).transpose(1, 2)
        player_tokens = player_state.unsqueeze(-1) * self.player_feature_embeddings.unsqueeze(0)
        player_tokens = self.player_relu(self.player_fc(player_tokens))
        
        # 3. Apply Cross-Attention
        x_attended = self.cross_attention(spatial_tokens, player_tokens)
        x = x_attended.transpose(1, 2).view(batch_size, self.filters, self.map_height, self.map_width)
        
        # --- Policy Heads ---
        p_pooled = self.p_pool_conv(x).flatten(1)
        p_latent = self.relu(self.p_fc_shared(p_pooled))
        
        policy = {}
        policy['action_type'] = self.pi_action(p_latent)
        policy['move_option'] = self.pi_option(p_latent)
        policy['source_spatial'] = self.pi_source(x).flatten(1)
        policy['target_spatial'] = self.pi_target(x).flatten(1)
        
        # --- Aux Heads (off x, before any value-trunk detach) ---
        aux = {}
        aux['aux_ownership'] = torch.tanh(self.aux_own(x)).flatten(1)
        aux['aux_fog_units'] = self.aux_fog(x).flatten(1)  # logits
        gap = x.mean(dim=[2, 3])
        aux['aux_spt'] = self.aux_spt(gap)
        aux['aux_opp_tech'] = self.aux_opp_tech(gap)

        # --- Value Heads ---
        v_input = x.detach() if DETACH_VALUE_TRUNK else x
        v_pooled = self.v_pool_conv(v_input).flatten(1)
        v_latent = self.relu(self.v_fc_shared(v_pooled))

        values = {}
        values['win'] = torch.tanh(self.v_win(v_latent))
        values['progress'] = self.v_progress(v_latent)

        return policy, values, aux

def compute_loss(policy_pred, values_pred, policy_targets, value_target,
                 aux_pred=None, aux_targets=None, aux_mask=None):
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

    # Total loss
    total_loss = total_policy_loss + value_loss

    # Aux heads: per-sample loss, masked to samples whose file carried aux
    # targets (old archives/teachers never do). Raw values are returned for
    # metrics; only the weighted sum joins total_loss — value_loss/loss_win
    # semantics stay untouched.
    aux_losses = {}
    if aux_pred is not None and aux_targets and aux_mask is not None:
        bce = nn.functional.binary_cross_entropy_with_logits
        per_sample = {
            'aux_ownership': lambda p, t: ((p - t) ** 2).mean(dim=1),
            'aux_fog_units': lambda p, t: bce(p, t, reduction='none').mean(dim=1),
            'aux_spt': lambda p, t: ((p - t) ** 2).mean(dim=1),
            'aux_opp_tech': lambda p, t: bce(p, t, reduction='none').mean(dim=1),
        }
        denom = aux_mask.sum().clamp(min=1.0)
        for k, fn in per_sample.items():
            if AUX_WEIGHTS[k] == 0.0 or k not in aux_targets:
                continue
            l = (fn(aux_pred[k], aux_targets[k]) * aux_mask).sum() / denom
            aux_losses[k] = l
            total_loss = total_loss + AUX_WEIGHTS[k] * l

    # loss_win returned raw (unweighted, no aux terms) — value_r2 needs it;
    # value_loss alone can't be unweighted once loss_progress is mixed in.
    return total_loss, total_policy_loss, value_loss, loss_win, aux_losses

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
    SPATIAL_CHANNELS = 161  # mirror of features.rs NUM_CHANNELS (incl. observation memory)
    PLAYER_STATE_DIM = 10

    model = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE).to(DEVICE)
    if os.path.exists("model.safetensors"):
        try:
            # strict=False: newly added heads (e.g. v_progress) are absent
            # from older checkpoints and must start fresh WITHOUT discarding
            # the trained trunk — a strict load would throw and silently
            # reinitialize everything via the except branch below.
            missing, unexpected = model.load_state_dict(
                load_file("model.safetensors"), strict=False
            )
            if missing or unexpected:
                print(f"Partial checkpoint load — missing: {missing}, unexpected: {unexpected}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting from scratch.")
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use CosineAnnealing with Warm Restarts for better convergence on short cycles
    # T_0=5 means it resets every 5 epochs (which is exactly our run length)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS, T_mult=1, eta_min=1e-5)

    # 3. Load + concatenate every chunk ONCE, up front, and keep all of it
    # resident across every epoch. The old code re-ran load_file + torch.cat
    # over the whole replay buffer inside the epoch loop — torch.cat on the
    # full buffer measured ~20-25s of pure CPU time (zero GPU use) per pass,
    # paid EPOCHS times for byte-identical data. Trade-off: peak RAM is now
    # the whole cat'd replay buffer for the entire run (every chunk's tensors
    # stay in `cached_chunks` at once) — TRAIN_CHUNK_FILES no longer bounds
    # resident RAM the way it used to (it only limits the in-flight list of
    # un-concatenated per-file tensors during loading); the only real RAM
    # knob now is REPLAY_BUFFER_FILES. Lower that if the buffer doesn't fit.
    random.shuffle(game_files)
    CHUNK_SIZE = int(os.environ.get("TRAIN_CHUNK_FILES", "10"))
    num_chunks = (len(game_files) + CHUNK_SIZE - 1) // CHUNK_SIZE
    expected_flat = SPATIAL_CHANNELS * MAP_SIZE * MAP_SIZE
    legacy_flat = 154 * MAP_SIZE * MAP_SIZE  # pre-observation-memory layout

    cached_chunks = []

    for i in range(0, len(game_files), CHUNK_SIZE):
        chunk_files = game_files[i : i + CHUNK_SIZE]
        chunk_idx = i // CHUNK_SIZE + 1
        print(f"Loading chunk {chunk_idx}/{num_chunks} ({len(chunk_files)} files)...")

        # Temporary storage for chunk data
        c_spatial = []
        c_player = []
        c_win = []
        c_progress = []

        c_heads = {
            'action_type': [], 'source_spatial': [], 'target_spatial': [], 'move_option': []
        }
        c_aux = {k: [] for k in AUX_DIMS}
        c_aux_mask = []

        for f in chunk_files:
            try:
                data = load_file(f)
                sp = data["spatial_maps"]
                if sp.shape[1] != expected_flat:
                    # Channels were appended at the end of the layout, so old
                    # files are index-stable: zero-pad the missing planes.
                    if sp.shape[1] == legacy_flat:
                        sp = torch.nn.functional.pad(sp, (0, expected_flat - legacy_flat))
                    else:
                        raise ValueError(
                            f"{f}: spatial width {sp.shape[1]} matches neither "
                            f"current ({expected_flat}) nor legacy ({legacy_flat}) layout"
                        )
                # Halved to fp16 in RAM — this is the dominant tensor (e.g.
                # ~29GB at fp32 for a 13-file/380K-sample buffer) and now
                # stays resident across all epochs (see cached_chunks below),
                # which pushed a real run into memory pressure (34/36GB used,
                # batch/s dropped ~4x). Model input, not a weight or label,
                # so upcasting to fp32 per-batch (below) costs nothing and
                # loses nothing training-relevant.
                c_spatial.append(sp.half())
                c_player.append(data["player_states"])
                c_win.append(data["values"])
                if "progress" in data:
                    c_progress.append(data["progress"])
                else:
                    c_progress.append(torch.zeros_like(data["values"]))

                # Load all policy heads
                for head in c_heads.keys():
                    if head in data:
                        c_heads[head].append(data[head])
                    else:
                        pass

                # Aux targets: per-file all-or-nothing presence mask.
                # Masked, never zero-filled — a zero-filled target would
                # silently train the head toward 0 on legacy samples.
                n = data["values"].shape[0]
                if all(k in data for k in AUX_DIMS):
                    for k in AUX_DIMS:
                        c_aux[k].append(data[k])
                    c_aux_mask.append(torch.ones(n))
                else:
                    for k, d in AUX_DIMS.items():
                        c_aux[k].append(torch.zeros(n, d))
                    c_aux_mask.append(torch.zeros(n))

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

            target_heads = {}
            for head, tensors in c_heads.items():
                if tensors:
                    target_heads[head] = torch.cat(tensors)

            # Guard against unnormalized policy rows: games generated before
            # Jul 2026 carry raw visit counts (sums up to 64) in move_option —
            # each acts as a ~30-64x weighted sample and destabilizes training.
            # Rows legitimately sum to <=1 (partial mass by design); only
            # renormalize rows whose mass exceeds 1.
            for head, t in target_heads.items():
                rs = t.sum(dim=1, keepdim=True)
                if (rs > 1.001).any():
                    target_heads[head] = torch.where(rs > 1.001, t / rs, t)

            # Kept out of target_heads: the renorm guard above would
            # corrupt fog/ownership rows (their sums are counts, not mass).
            target_aux = {k: torch.cat(v) for k, v in c_aux.items()}
            aux_mask = torch.cat(c_aux_mask)

        except RuntimeError as e:
            print(f"OOM loading chunk: {e}")
            continue

        # Cleanup lists
        del c_spatial, c_player, c_win, c_progress, c_aux, c_aux_mask
        gc.collect()

        dataset_size = len(spatial_maps)
        print(f"  Loaded {dataset_size} samples.")

        cached_chunks.append({
            "spatial_maps": spatial_maps,
            "player_states": player_states,
            "targets_win": targets_win,
            "targets_progress": targets_progress,
            "target_heads": target_heads,
            "target_aux": target_aux,
            "aux_mask": aux_mask,
            "dataset_size": dataset_size,
            "chunk_idx": chunk_idx,
        })

    if not cached_chunks:
        print("No usable training data after loading!")
        return

    epoch_batch_estimate = sum(
        (c["dataset_size"] + BATCH_SIZE - 1) // BATCH_SIZE for c in cached_chunks
    )
    report_batch_indices = batch_report_indices(epoch_batch_estimate)
    if epoch_batch_estimate <= 10:
        print(f"Reporting all {epoch_batch_estimate} batches/epoch.")
    else:
        print(f"Reporting ~10/{epoch_batch_estimate} sampled batches/epoch.")

    # 4. Training Loop
    for epoch in range(EPOCHS):
        # Loss accumulators live on-device and are only pulled to Python at
        # report points / epoch end (below). A per-batch .item() forces an
        # MPS/CUDA sync that serializes the queue and stalls the pipeline —
        # this was previously happening on every single batch.
        total_loss_t = torch.zeros((), device=DEVICE)
        total_p_loss_t = torch.zeros((), device=DEVICE)
        total_v_loss_t = torch.zeros((), device=DEVICE)
        total_v_win_t = torch.zeros((), device=DEVICE)
        total_batches = 0
        # Streaming mean/variance of the value targets seen this epoch, so
        # value_r2 (below) compares MSE against the actual training-mix
        # variance instead of a guess — small MSE alone doesn't mean the head
        # fits anything if the targets barely vary.
        target_sum_t = torch.zeros((), device=DEVICE)
        target_sumsq_t = torch.zeros((), device=DEVICE)
        target_n = 0
        total_aux_t = {k: torch.zeros((), device=DEVICE) for k in AUX_DIMS}
        total_aux_n_t = torch.zeros((), device=DEVICE)

        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        for chunk in cached_chunks:
            spatial_maps = chunk["spatial_maps"]
            player_states = chunk["player_states"]
            targets_win = chunk["targets_win"]
            targets_progress = chunk["targets_progress"]
            target_heads = chunk["target_heads"]
            target_aux = chunk["target_aux"]
            aux_mask = chunk["aux_mask"]
            dataset_size = chunk["dataset_size"]
            chunk_idx = chunk["chunk_idx"]

            indices = torch.randperm(dataset_size)
            num_batches_in_chunk = (dataset_size + BATCH_SIZE - 1) // BATCH_SIZE
            chunk_start_time = time.time()

            for batch_num, j in enumerate(range(0, dataset_size, BATCH_SIZE), start=1):
                batch_idx = indices[j : j + BATCH_SIZE]

                batch_spatial = spatial_maps[batch_idx].to(DEVICE, dtype=torch.float32)
                batch_player = player_states[batch_idx].to(DEVICE)

                batch_values = {
                    'win': targets_win[batch_idx].to(DEVICE),
                }
                target_sum_t += batch_values['win'].sum().detach()
                target_sumsq_t += (batch_values['win'] * batch_values['win']).sum().detach()
                target_n += batch_values['win'].numel()

                if targets_progress is not None:
                    batch_values['progress'] = targets_progress[batch_idx].to(DEVICE)
                batch_targets = {}
                for head, tensor in target_heads.items():
                    batch_targets[head] = tensor[batch_idx].to(DEVICE)
                batch_aux = {k: t[batch_idx].to(DEVICE) for k, t in target_aux.items()}
                batch_aux_mask = aux_mask[batch_idx].to(DEVICE)

                # Reshape spatial to (B, C, H, W)
                batch_spatial = batch_spatial.view(-1, SPATIAL_CHANNELS, MAP_SIZE, MAP_SIZE)

                # D4 dihedral augmentation: one random board symmetry per batch.
                # Only the two spatial policy heads live in tile space and must
                # co-transform; action/option/value/progress are orientation-free.
                if AUGMENT_D4:
                    k = random.randint(0, 3)
                    do_flip = random.random() < 0.5
                    if k > 0 or do_flip:
                        if k > 0:
                            batch_spatial = torch.rot90(batch_spatial, k, [2, 3])
                        if do_flip:
                            batch_spatial = torch.flip(batch_spatial, [3])
                        for head in ('source_spatial', 'target_spatial'):
                            if head in batch_targets:
                                t = batch_targets[head].view(-1, 1, MAP_SIZE, MAP_SIZE)
                                if k > 0:
                                    t = torch.rot90(t, k, [2, 3])
                                if do_flip:
                                    t = torch.flip(t, [3])
                                batch_targets[head] = t.flatten(1)
                        # The two tile-space aux targets must co-transform too.
                        for head in ('aux_ownership', 'aux_fog_units'):
                            t = batch_aux[head].view(-1, 1, MAP_SIZE, MAP_SIZE)
                            if k > 0:
                                t = torch.rot90(t, k, [2, 3])
                            if do_flip:
                                t = torch.flip(t, [3])
                            batch_aux[head] = t.flatten(1)

                optimizer.zero_grad()

                policy_pred, values_pred, aux_pred = model(batch_spatial, batch_player)

                loss, p_loss, v_loss, v_win_loss, aux_losses = compute_loss(
                    policy_pred, values_pred, batch_targets, batch_values,
                    aux_pred, batch_aux, batch_aux_mask)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss_t += loss.detach()
                total_p_loss_t += p_loss.detach()
                total_v_loss_t += v_loss.detach()
                total_v_win_t += v_win_loss.detach()
                total_batches += 1
                # Mask-count-weighted: batches of teacher/legacy samples add
                # zero weight, so the epoch average can't drift toward 0.
                # Unconditional (no "if aux_n > 0" guard): a zero mask
                # contributes exactly zero either way, and skipping it would
                # require a .item() sync to evaluate the branch.
                aux_n_t = batch_aux_mask.sum().detach()
                total_aux_n_t += aux_n_t
                for k, l in aux_losses.items():
                    total_aux_t[k] += l.detach() * aux_n_t

                global_batch_num = total_batches
                if global_batch_num in report_batch_indices:
                    elapsed = time.time() - chunk_start_time
                    batches_per_sec = batch_num / elapsed if elapsed > 0 else 0.0
                    cur_loss = (total_loss_t / total_batches).item()
                    cur_p_loss = (total_p_loss_t / total_batches).item()
                    cur_v_loss = (total_v_loss_t / total_batches).item()
                    print(
                        f"  Epoch {epoch+1} batch {global_batch_num}/{epoch_batch_estimate} "
                        f"(chunk {chunk_idx}/{num_chunks} {batch_num}/{num_batches_in_chunk}) "
                        f"- loss: {cur_loss:.4f} "
                        f"(policy: {cur_p_loss:.4f}, value: {cur_v_loss:.4f}) "
                        f"- {batches_per_sec:.1f} batch/s"
                    )

            # Drain the device caching allocator's pool of per-batch
            # activation/gradient buffers between chunks. The old code did
            # this every chunk too; caching all chunks in RAM across epochs
            # (above) doesn't change that need — without it, RSS climbed
            # from ~11GB to ~29GB over the course of one epoch on a live run
            # (Jul 2026), because nothing ever returned MPS-wired buffers to
            # the OS until the very end. cached_chunks itself is untouched.
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            elif DEVICE == "mps":
                torch.mps.empty_cache()

        if total_batches > 0:
            avg_loss = (total_loss_t / total_batches).item()
            avg_p_loss = (total_p_loss_t / total_batches).item()
            avg_v_loss = (total_v_loss_t / total_batches).item()
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - No data processed")

        scheduler.step()

    del cached_chunks
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()
    gc.collect()

    final_loss = (total_loss_t / total_batches).item() if total_batches > 0 else 0.0
    final_p_loss = (total_p_loss_t / total_batches).item() if total_batches > 0 else 0.0
    final_v_loss = (total_v_loss_t / total_batches).item() if total_batches > 0 else 0.0
    final_v_win = (total_v_win_t / total_batches).item() if total_batches > 0 else 0.0
    total_aux_n = total_aux_n_t.item()
    final_aux = {
        k: (total_aux_t[k].item() / total_aux_n if total_aux_n > 0 else 0.0)
        for k in AUX_DIMS
    }

    # R^2 of the win head against the LAST epoch's own target distribution:
    # 1 - MSE/Var. Small MSE alone is meaningless if the targets barely vary
    # (a constant-mean predictor would also score low MSE) — this is the
    # number that actually says whether the head explains anything. Uses the
    # raw win MSE tracked separately in compute_loss, so neither
    # VALUE_LOSS_WEIGHT nor the aux progress loss bundled into value_loss
    # can skew it.
    if target_n > 0:
        target_sum = target_sum_t.item()
        target_sumsq = target_sumsq_t.item()
        target_mean = target_sum / target_n
        target_var = target_sumsq / target_n - target_mean * target_mean
        value_r2 = 1.0 - final_v_win / target_var if target_var > 1e-8 else 0.0
    else:
        value_r2 = 0.0

    # (BatchNorm recalibration used to live here; GroupNorm has no running
    # stats and no train/eval duality, so there is nothing to calibrate.)

    # 4. Save Model in f16 for blazing fast CPU inference
    half_state = {k: v.half() for k, v in model.state_dict().items()}
    save_file(half_state, "model.safetensors")

    with open(".last_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss": round(final_loss, 4),
                "policy_loss": round(final_p_loss, 4),
                "value_loss": round(final_v_loss, 4),
                "value_r2": round(value_r2, 4),
                "aux_own_loss": round(final_aux['aux_ownership'], 4),
                "aux_fog_loss": round(final_aux['aux_fog_units'], 4),
                "aux_spt_loss": round(final_aux['aux_spt'], 4),
                "aux_tech_loss": round(final_aux['aux_opp_tech'], 4),
            },
            f,
        )

if __name__ == "__main__":
    train()

