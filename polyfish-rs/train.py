import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
import glob
import hashlib
import json
import math
import os
import random
import gc
import time
import argparse

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
# This is the peak of a cosine decay that spans the whole RUN (see
# TRAIN_TOTAL_ITERS), not a per-invocation restart.
LEARNING_RATE = float(os.environ.get("TRAIN_LR", "0.002"))
# Run identity and length, used to span the LR schedule and the Adam moments
# across the loop's per-iteration invocations. Exported by run_training_loop.sh;
# absent (a bare `python train.py`) the schedule falls back to these defaults
# and the sidecar is still honoured.
RUN_ID = os.environ.get("TRAIN_RUN_ID", "")
TOTAL_ITERS = max(1, int(os.environ.get("TRAIN_TOTAL_ITERS", "500")))
# Fraction of training FILES permanently held out from fitting so value_r2 can
# be read out-of-sample. 0 disables the holdout entirely.
HOLDOUT_FRAC = float(os.environ.get("TRAIN_HOLDOUT_FRAC", "0.15"))
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
# D4 symmetry augmentation (AlphaZero-style): a random rotation/reflection per
# batch on the spatial input and every spatial target. Geometrically valid — no
# feature plane, player scalar, or rule is orientation-dependent — but OFF by
# default, and this is a measured result, not caution: enabling it MID-RUN
# collapsed play for ~8 iterations (run 1783556259; the policy lost its
# orientation-specific fit and the degraded games fed back through self-play).
# Legitimate ONLY for from-scratch runs, where the net never learns orientation
# shortcuts to begin with. Training-only. Enable: AUGMENT_D4=1.
AUGMENT_D4 = os.environ.get("AUGMENT_D4", "0") == "1"

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
# Width of the action_type head. Mirrors NUM_ACTION_TYPES in src/ai/network.rs:
# MoveType has 12 variants (Resign = 11) and both sides read the same
# model.safetensors, so a drift here is a silent load failure or garbage.
NUM_ACTION_TYPES = 12

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.c2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.c1(x)))
        out = self.gn2(self.c2(out))
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
        self.gn1 = nn.GroupNorm(8, self.filters)
        self.relu = nn.ReLU()
        
        # ResBlocks (Match Rust config)
        self.res_blocks = nn.ModuleList([ResBlock(self.filters) for _ in range(6)])
        
        # --- Cross Attention Layer ---
        # Allow each spatial tile (Q) to attend to global player features (K,V)
        self.cross_attention = CrossAttention(self.filters, nhead=4)
        
        # --- Decomposed Policy Heads ---
        # 1. Action Type (Attack, Step, Build, etc.)
        self.p_pool_conv = nn.Conv2d(self.filters, 1, 1)
        self.p_fc_shared = nn.Linear(map_height * map_width, self.filters)
        self.pi_action = nn.Linear(self.filters, NUM_ACTION_TYPES)
        
        # 2. Unified Options (192 categories: Structures, Units, Techs, Abilities, Rewards)
        self.pi_option = nn.Linear(self.filters, 192)
        
        # 3. Spatial Heads (Source and Target tile selection)
        self.pi_source = nn.Conv2d(self.filters, 1, 1)
        self.pi_target = nn.Conv2d(self.filters, 1, 1)
        
        # --- Value Heads ---
        # Heavy value head (EXP_ARCH_001): global mean+max pool over the FULL
        # 64-channel trunk -> 2-layer MLP. The old head collapsed the trunk to
        # ONE channel and ran a single linear layer — a near-linear probe that
        # cannot represent "am I winning" for a strategic game. This can.
        self.v_fc1 = nn.Linear(2 * self.filters, self.filters)
        self.v_fc2 = nn.Linear(self.filters, self.filters)
        self.v_win = nn.Linear(self.filters, 1)
        self.v_progress = nn.Linear(self.filters, 1)
        # Aux spatial head: predicted end-of-game per-tile ownership
        # (+1 mine / -1 enemy / 0 neutral), trained with a small-weight MSE
        # purely to densify trunk gradient. Never used by search.
        self.v_ownership = nn.Conv2d(self.filters, 1, 1)

    def forward(self, spatial_map, player_state):
        batch_size = spatial_map.size(0)
        
        # 1. Spatial Backbone
        x = self.relu(self.gn1(self.conv1(spatial_map)))
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
        p_pooled = self.p_pool_conv(x)
        p_pooled = p_pooled.flatten(1)
        p_latent = self.relu(self.p_fc_shared(p_pooled))
        
        policy = {}
        policy['action_type'] = self.pi_action(p_latent)
        policy['move_option'] = self.pi_option(p_latent)
        policy['source_spatial'] = self.pi_source(x).flatten(1)
        policy['target_spatial'] = self.pi_target(x).flatten(1)
        
        # --- Value Heads ---
        v_input = x.detach() if DETACH_VALUE_TRUNK else x
        v_mean = v_input.mean(dim=(2, 3))          # [B, filters]
        v_max = v_input.amax(dim=(2, 3))           # [B, filters]
        v_feat = torch.cat([v_mean, v_max], dim=1)  # [B, 2*filters]
        v_latent = self.relu(self.v_fc2(self.relu(self.v_fc1(v_feat))))
        
        values = {}
        values['win'] = self.v_win(v_latent)
        values['progress'] = self.v_progress(v_latent)
        values['ownership'] = self.v_ownership(x).flatten(1)

        return policy, values

def apply_d4(x, k, flip):
    """Rotate 90°·k then optionally mirror, over the trailing (H, W) dims."""
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if flip:
        x = torch.flip(x, dims=(-1,))
    return x

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
            # self_play normalizes action/source/target by total visits but
            # leaves move_option as raw visit COUNTS, which scales that head's
            # loss by N. Rows already summing to <=1 (and all-zero rows, which
            # correctly contribute nothing) pass through untouched.
            target = target / target.sum(dim=1, keepdim=True).clamp(min=1.0)
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

    return total_loss, total_policy_loss, value_loss, ownership_loss, loss_win

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

def _migrate_checkpoint(state_dict, model, player_state_dim):
    """Transform a legacy state_dict to match the current model architecture.

    Returns (state_dict, migrations) where *migrations* is a list of
    human-readable strings describing what was changed.  An empty list means
    the checkpoint was already up-to-date.
    """
    migrations = []
    filters = state_dict['v_win.weight'].shape[1]  # always 64

    # ------------------------------------------------------------------
    # 1. Value-head swap: v_pool_conv / v_fc_shared  →  v_fc1 / v_fc2
    #    The old head collapsed the trunk to 1 channel then ran a single
    #    linear; the new head uses global mean+max pool → 2-layer MLP.
    #    Weights are not transferable, so we initialise from the model's
    #    freshly-created parameters and delete the stale keys.
    # ------------------------------------------------------------------
    stale_value_prefixes = ("v_pool_conv.", "v_fc_shared.")
    stale_keys = [k for k in state_dict if k.startswith(stale_value_prefixes)]
    new_value_prefixes = ("v_fc1.", "v_fc2.")
    new_keys_missing = any(
        k.startswith(new_value_prefixes)
        for k in model.state_dict()
    ) and not any(
        k.startswith(new_value_prefixes)
        for k in state_dict
    )
    if stale_keys or new_keys_missing:
        for k in stale_keys:
            del state_dict[k]
        # Seed v_fc1/v_fc2 from model's random init so load_state_dict
        # won't report them as missing.
        for k, v in model.state_dict().items():
            if k.startswith(new_value_prefixes):
                state_dict[k] = v.clone()
        dropped = ", ".join(sorted(stale_keys)) if stale_keys else "(none)"
        migrations.append(
            f"value-head: dropped [{dropped}], initialised v_fc1/v_fc2 fresh"
        )

    # ------------------------------------------------------------------
    # 2. Player embeddings: resize (old_dim, filters) → (player_state_dim, filters)
    #    Preserve existing rows and pad new rows with the model's fresh init.
    # ------------------------------------------------------------------
    for name in ("player_pos_embeddings", "player_feature_embeddings"):
        if name not in state_dict:
            # Very old checkpoint; take the model's full init.
            state_dict[name] = model.state_dict()[name].clone()
            migrations.append(f"{name}: created ({player_state_dim}, {filters})")
            continue
        old = state_dict[name]
        if old.shape[0] != player_state_dim:
            new = model.state_dict()[name].clone()  # (player_state_dim, filters)
            keep = min(old.shape[0], player_state_dim)
            new[:keep] = old[:keep]
            state_dict[name] = new
            migrations.append(
                f"{name}: resized ({old.shape[0]}, {filters}) → ({player_state_dim}, {filters})"
            )

    # ------------------------------------------------------------------
    # 3. v_progress head (added after initial arch)
    # ------------------------------------------------------------------
    if "v_progress.weight" not in state_dict:
        state_dict["v_progress.weight"] = torch.randn(1, filters) * 0.01
        state_dict["v_progress.bias"] = torch.zeros(1)
        migrations.append("v_progress: initialised fresh")

    # ------------------------------------------------------------------
    # 4. Fog-memory: zero-pad conv1 input channels 136 → 142
    # ------------------------------------------------------------------
    conv1 = state_dict.get("conv1.weight")
    if conv1 is not None and conv1.shape[1] < 142:
        old_ch = conv1.shape[1]
        pad = torch.zeros(conv1.shape[0], 142 - old_ch, conv1.shape[2], conv1.shape[3])
        state_dict["conv1.weight"] = torch.cat([conv1, pad], dim=1)
        migrations.append(f"conv1.weight: padded input channels {old_ch} → 142")

    # ------------------------------------------------------------------
    # 5. action_type head widened to NUM_ACTION_TYPES (Resign = 11 was one
    #    past the old 11-wide head). Pad with a zero row/bias so the existing
    #    11 categories keep their learned weights.
    # ------------------------------------------------------------------
    pi_w = state_dict.get("pi_action.weight")
    if pi_w is not None and pi_w.shape[0] < NUM_ACTION_TYPES:
        old_n = pi_w.shape[0]
        state_dict["pi_action.weight"] = torch.cat(
            [pi_w, torch.zeros(NUM_ACTION_TYPES - old_n, pi_w.shape[1], dtype=pi_w.dtype)], dim=0
        )
        pi_b = state_dict.get("pi_action.bias")
        if pi_b is not None:
            state_dict["pi_action.bias"] = torch.cat(
                [pi_b, torch.zeros(NUM_ACTION_TYPES - old_n, dtype=pi_b.dtype)], dim=0
            )
        migrations.append(f"pi_action: padded {old_n} → {NUM_ACTION_TYPES} action types")

    return state_dict, migrations


MODEL_PATH = "model.safetensors"
# Overridable so diagnostic arms (bisect_arm.sh) neither resume nor overwrite
# the production sidecar.
OPTIMIZER_STATE_PATH = os.environ.get("TRAIN_OPTIMIZER_STATE", "optimizer_state.pt")
METRICS_PATH = ".last_train_metrics.json"


def atomic_write(path, write_fn):
    """Write through `path`.tmp + os.replace so a crash can't leave a torn file."""
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def load_optimizer_state(optimizer, model, run_id):
    """Restore Adam moments and the schedule position from the sidecar.

    Returns the schedule step to resume at (0 when nothing usable was found).
    Every failure path degrades to a fresh optimizer with a loud warning.
    """
    if not os.path.exists(OPTIMIZER_STATE_PATH):
        return 0
    try:
        blob = torch.load(OPTIMIZER_STATE_PATH, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"WARNING: {OPTIMIZER_STATE_PATH} unreadable ({e}); Adam moments reset.")
        return 0

    saved_run = blob.get("run_id", "")
    if run_id and saved_run and saved_run != run_id:
        print(f"WARNING: optimizer state is from run {saved_run}, not {run_id}; Adam moments reset.")
        return 0

    shapes = [list(p.shape) for p in model.parameters()]
    if blob.get("param_shapes") != shapes:
        print("WARNING: model architecture changed since the optimizer state was written; "
              "Adam moments reset (this is expected on the iteration a head is resized).")
        return 0

    try:
        optimizer.load_state_dict(blob["optimizer"])
    except Exception as e:
        print(f"WARNING: optimizer state incompatible ({e}); Adam moments reset.")
        return 0

    step = int(blob.get("sched_step", 0))
    print(f"Resumed optimizer state at schedule step {step}.")
    return step


def save_optimizer_state(optimizer, model, run_id, sched_step):
    blob = {
        "optimizer": optimizer.state_dict(),
        "param_shapes": [list(p.shape) for p in model.parameters()],
        "run_id": run_id,
        "sched_step": sched_step,
    }
    atomic_write(OPTIMIZER_STATE_PATH, lambda p: torch.save(blob, p))


def cosine_lr(base_lr, step, total, eta_min=1e-5):
    """Closed-form cosine decay spanning the whole run, not one invocation."""
    t = min(max(step, 0), total) / float(total)
    return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * t))


def is_holdout_file(path, frac):
    """Membership is a function of the basename alone, so a file stays on the
    same side for its whole life in the buffer (a per-iteration reshuffle would
    train on last iteration's holdout and inflate the reading)."""
    if frac <= 0:
        return False
    h = int(hashlib.sha1(os.path.basename(path).encode()).hexdigest()[:8], 16)
    return (h % 10000) < frac * 10000


def split_holdout(files, frac):
    """Split the buffer into (train, holdout) by FILE, returning both lists.

    Not by position: games_*.safetensors records no game boundaries, so a
    position-level split would put positions from the same game on both sides
    and report a falsely good holdout. Files never split a game, so this leaks
    nothing — it is just coarser than a per-game split, and with a stable
    per-file rule the holdout can come out empty on a small buffer.
    """
    held = [f for f in files if is_holdout_file(f, frac)]
    kept = [f for f in files if not is_holdout_file(f, frac)]
    if not kept:
        return list(files), []
    return kept, held


def pad_spatial(smaps, channels, map_size):
    """Zero-pad legacy spatial maps up to `channels` (channels were appended)."""
    if smaps.dim() == 4:
        if smaps.shape[1] >= channels:
            return smaps
        b, c, h, w = smaps.shape
        return torch.cat([smaps, torch.zeros(b, channels - c, h, w, dtype=smaps.dtype)], dim=1)
    area = map_size * map_size
    old_c = smaps.shape[1] // area
    if old_c >= channels:
        return smaps
    pad = torch.zeros(smaps.shape[0], (channels - old_c) * area, dtype=smaps.dtype)
    return torch.cat([smaps, pad], dim=1)


def evaluate_value_holdout(model, files, batch_size, spatial_channels, map_size):
    """Win-head MSE / target variance on files the trainer never fit.

    Returns (r2, n_samples); r2 is None when the holdout has no usable data.
    """
    sq_err = 0.0
    tgt_sum = 0.0
    tgt_sumsq = 0.0
    n = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for f in files:
            try:
                data = load_file(f)
                smaps = pad_spatial(data["spatial_maps"], spatial_channels, map_size)
                players = data["player_states"]
                wins = data["values"]
            except Exception as e:
                print(f"Holdout: skipping {f} ({e})")
                continue
            for j in range(0, wins.shape[0], batch_size):
                bs = smaps[j : j + batch_size].to(DEVICE).view(-1, spatial_channels, map_size, map_size)
                bp = players[j : j + batch_size].to(DEVICE)
                bw = wins[j : j + batch_size].to(DEVICE)
                _, values_pred = model(bs, bp)
                sq_err += ((values_pred["win"] - bw) ** 2).sum().item()
                tgt_sum += bw.sum().item()
                tgt_sumsq += (bw * bw).sum().item()
                n += bw.numel()
    if was_training:
        model.train()
    if n == 0:
        return None, 0
    mean = tgt_sum / n
    var = tgt_sumsq / n - mean * mean
    if var <= 1e-8:
        return None, n
    return 1.0 - (sq_err / n) / var, n


def train(batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LEARNING_RATE, chunk_size=None, benchmark_mode=False):
    if chunk_size is None:
        chunk_size = int(os.environ.get("TRAIN_CHUNK_FILES", "10"))

    # 1. Load Data
    # `games_*` also matches the two non-self-play writers, which must NOT be
    # trained on as self-play: games_human_* (recorder.rs hardcodes win = 0.0,
    # so it drags the value head toward 0 on strong human positions) and
    # games_pro_* (behaviour-cloning exports — real labels, but they belong in
    # teachers/, not in the archive/prune rotation).
    def self_play_only(paths):
        kept, skipped = [], []
        for p in paths:
            (skipped if os.path.basename(p).startswith(("games_human_", "games_pro_")) else kept).append(p)
        for p in skipped:
            print(f"Skipping non-self-play file {p} (bogus or imitation value labels).")
        return kept

    fresh_files = self_play_only(glob.glob("games_*.safetensors"))
    archive_files = self_play_only(
        sorted(glob.glob("archive/games_*.safetensors"), key=os.path.getmtime, reverse=True)
    )
    # Replay window in FILES; run_training_loop.sh exports REPLAY_BUFFER_FILES
    # scaled by its -g so the buffer stays ~constant in GAMES (default 10
    # files ≈ 700 games at 64 games/file). Each sample is trained ~20 times
    # before pruning; reduces overfitting risk. Archive pruning keeps window+1.
    replay_buffer_size = int(os.environ.get("REPLAY_BUFFER_FILES", "10"))
    # Teacher anchor: always mix these into every iteration so gradients keep
    # pulling toward known-good play regardless of self-play drift (RLHF-style
    # reference anchor). Never archived or pruned.
    # games_pro_* IS welcome here (real ±1 outcome labels); games_human_* never is.
    teacher_files = [
        f for f in sorted(glob.glob("teachers/games_*.safetensors"))
        if not os.path.basename(f).startswith("games_human_")
    ]
    game_files = fresh_files + archive_files[:replay_buffer_size] + teacher_files

    if not game_files:
        print("No training data found (checked ./, ./archive/, and ./teachers/)!")
        if benchmark_mode:
            return 0, 0
        return

    print(f"Training on {len(game_files)} files ({len(fresh_files)} fresh, "
          f"{len(archive_files[:replay_buffer_size])} archived, {len(teacher_files)} teacher).")

    holdout_files = []
    if not benchmark_mode:
        game_files, holdout_files = split_holdout(game_files, HOLDOUT_FRAC)
        if holdout_files:
            print(f"Holdout: {len(holdout_files)} file(s) withheld from fitting "
                  f"({', '.join(os.path.basename(f) for f in holdout_files)}).")
        elif HOLDOUT_FRAC > 0:
            print("Holdout: no file in this buffer hashes into the holdout bucket; "
                  "value_r2 is in-sample only this iteration.")

    # 2. Init Model
    MAP_SIZE = 11
    SPATIAL_CHANNELS = 142  # 136 + 6 fog-memory channels (see notes-memory.md)
    PLAYER_STATE_DIM = 16

    model = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        # A checkpoint that exists but will not load is FATAL. Silently falling
        # back to random weights mid-run throws the run away without a trace.
        try:
            ckpt = load_file(MODEL_PATH)
            ckpt, migrations = _migrate_checkpoint(ckpt, model, PLAYER_STATE_DIM)
            if migrations:
                print(f"Checkpoint migrated: {'; '.join(migrations)}")
            # strict=False lets checkpoints that predate the aux ownership
            # head load in place (the head keeps its fresh init). Any OTHER
            # mismatch is still fatal so a wrong-architecture checkpoint
            # can't silently half-load.
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            hard_missing = [k for k in missing if not k.startswith("v_ownership.")]
            if hard_missing or unexpected:
                raise RuntimeError(f"state_dict mismatch: missing={hard_missing} unexpected={list(unexpected)}")
            if missing:
                print(f"Checkpoint predates ownership head; initializing {missing} fresh.")
        except Exception as e:
            raise RuntimeError(
                f"{MODEL_PATH} exists but could not be loaded: {e}. Refusing to restart "
                f"training from random weights — restore a checkpoint from checkpoints/ "
                f"(or delete {MODEL_PATH} deliberately to start over)."
            ) from e
    else:
        print(f"No {MODEL_PATH} found; initializing fresh weights (expected only on iteration 1).")
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Adam moments and the cosine position live in a sidecar so they survive the
    # loop's per-iteration re-invocation; rebuilding them every call threw the
    # moment estimates away and restarted the LR at its peak (a sawtooth).
    sched_step = load_optimizer_state(optimizer, model, RUN_ID)
    total_sched_steps = max(1, TOTAL_ITERS * max(1, epochs))

    # In benchmark mode, limit to 1 epoch and max 2 chunks to save time
    if benchmark_mode:
        epochs = 1
        num_benchmark_samples = 0
        benchmark_start_time = 0

    # 3. Training Loop
    for epoch in range(epochs):
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        total_o_loss = 0
        total_w_loss = 0
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
        num_chunks = (len(game_files) + chunk_size - 1) // chunk_size

        # In benchmark mode, only process the first chunk to get throughput
        if benchmark_mode:
            num_chunks = 1

        epoch_lr = cosine_lr(lr, sched_step, total_sched_steps)
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr

        print(f"\n=== Epoch {epoch+1}/{epochs} === (schedule step {sched_step}/{total_sched_steps}, lr {epoch_lr:.2e})")

        report_batch_indices = None
        epoch_batch_estimate = None

        for i in range(0, len(game_files), chunk_size):
            if benchmark_mode and (i // chunk_size) >= 1:
                break
            
            chunk_files = game_files[i : i + chunk_size]
            chunk_idx = i // chunk_size + 1
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
                    
                    smaps = pad_spatial(data["spatial_maps"], SPATIAL_CHANNELS, MAP_SIZE)
                    c_spatial.append(smaps)
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
                            t = data[head]
                            # Files written before the action head widened.
                            if head == "action_type" and t.shape[1] < NUM_ACTION_TYPES:
                                pad = torch.zeros(t.shape[0], NUM_ACTION_TYPES - t.shape[1], dtype=t.dtype)
                                t = torch.cat([t, pad], dim=1)
                            c_heads[head].append(t)
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
                    
            except Exception as e:
                if "out of memory" not in str(e).lower() and "OutOfMemoryError" not in e.__class__.__name__:
                    raise e
                print(f"OOM loading chunk: {e}")
                continue
            
            # Cleanup lists
            del c_spatial, c_player, c_win, c_progress, c_ownership, c_ownership_mask
            gc.collect()
            
            dataset_size = len(spatial_maps)
            print(f"  Loaded {dataset_size} samples.")

            if epoch_batch_estimate is None and len(chunk_files) > 0:
                est_samples = int(dataset_size / len(chunk_files) * len(game_files))
                epoch_batch_estimate = (est_samples + batch_size - 1) // batch_size
                report_batch_indices = batch_report_indices(epoch_batch_estimate)
                if epoch_batch_estimate <= 10:
                    print(f"  Reporting all {epoch_batch_estimate} batches.")
                else:
                    print(f"  Reporting ~10/{epoch_batch_estimate} sampled batches.")

            indices = torch.randperm(dataset_size)
            num_batches_in_chunk = (dataset_size + batch_size - 1) // batch_size
            chunk_start_time = time.time()
            if benchmark_mode and benchmark_start_time == 0:
                benchmark_start_time = chunk_start_time
                num_benchmark_samples = dataset_size

            for batch_num, j in enumerate(range(0, dataset_size, batch_size), start=1):
                batch_idx = indices[j : j + batch_size]
                
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

                # --- DATA AUGMENTATION (Dihedral Group D4) ---
                # One random transform per batch; global heads (action_type,
                # move_option), player state, and value targets are invariant.
                if AUGMENT_D4:
                    k = random.randrange(4)
                    flip = random.random() < 0.5
                    if k or flip:
                        batch_spatial = apply_d4(batch_spatial, k, flip)
                        for head in ('source_spatial', 'target_spatial'):
                            if head in batch_targets:
                                t = batch_targets[head].view(-1, MAP_SIZE, MAP_SIZE)
                                batch_targets[head] = apply_d4(t, k, flip).reshape(-1, MAP_SIZE * MAP_SIZE)
                        if 'ownership' in batch_values:
                            t = batch_values['ownership'].view(-1, MAP_SIZE, MAP_SIZE)
                            batch_values['ownership'] = apply_d4(t, k, flip).reshape(-1, MAP_SIZE * MAP_SIZE)

                optimizer.zero_grad()
                
                policy_pred, values_pred = model(batch_spatial, batch_player)
                
                loss, p_loss, v_loss, o_loss, w_loss = compute_loss(policy_pred, values_pred, batch_targets, batch_values)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_p_loss += p_loss.item()
                total_v_loss += v_loss.item()
                total_o_loss += o_loss.item()
                total_w_loss += w_loss.item()
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
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f}, Ownership: {avg_o_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} - No data processed")

        sched_step += 1

    final_loss = total_loss / total_batches if total_batches > 0 else 0.0
    final_p_loss = total_p_loss / total_batches if total_batches > 0 else 0.0
    final_v_loss = total_v_loss / total_batches if total_batches > 0 else 0.0
    final_o_loss = total_o_loss / total_batches if total_batches > 0 else 0.0
    final_w_loss = total_w_loss / total_batches if total_batches > 0 else 0.0

    # R^2 of the value head against the LAST epoch's own target distribution:
    # 1 - MSE/Var. Small MSE alone is meaningless if the targets barely vary
    # (a constant-mean predictor would also score low MSE) — this is the
    # number that actually says whether the head explains anything.
    if target_n > 0:
        target_mean = target_sum / target_n
        target_var = target_sumsq / target_n - target_mean * target_mean
        value_r2 = 1.0 - final_w_loss / target_var if target_var > 1e-8 else 0.0
    else:
        value_r2 = 0.0

    if benchmark_mode:
        if num_benchmark_samples > 0 and benchmark_start_time > 0:
            total_time = time.time() - benchmark_start_time
            samples_per_sec = num_benchmark_samples / total_time
            max_memory_mb = 0
            if DEVICE == "cuda":
                max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            return samples_per_sec, max_memory_mb
        return 0, 0

    # `value_r2` above is IN-SAMPLE — the buffer the net just fit. Compare it
    # against the holdout number: in-sample high / holdout low is overfitting,
    # both low is underfitting. That contrast is the diagnostic, so report both.
    holdout_r2, holdout_n = evaluate_value_holdout(
        model, holdout_files, batch_size, SPATIAL_CHANNELS, MAP_SIZE
    )
    if holdout_r2 is None:
        print(f"value_r2: {value_r2:.4f} in-sample | holdout unavailable ({holdout_n} samples)")
    else:
        print(f"value_r2: {value_r2:.4f} in-sample | {holdout_r2:.4f} holdout "
              f"({holdout_n} samples, {len(holdout_files)} file(s))")

    # 4. Save Model in f16 for blazing fast CPU inference
    half_state = {k: v.half() for k, v in model.state_dict().items()}
    atomic_write(MODEL_PATH, lambda p: save_file(half_state, p))
    save_optimizer_state(optimizer, model, RUN_ID, sched_step)

    metrics = {
        "loss": round(final_loss, 4),
        "policy_loss": round(final_p_loss, 4),
        "value_loss": round(final_v_loss, 4),
        "ownership_loss": round(final_o_loss, 4),
        "value_r2": round(value_r2, 4),
        "value_r2_insample": round(value_r2, 4),
        "value_r2_holdout": round(holdout_r2, 4) if holdout_r2 is not None else "",
        "holdout_samples": holdout_n,
        "holdout_files": len(holdout_files),
    }

    def _write_metrics(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f)

    atomic_write(METRICS_PATH, _write_metrics)

def run_benchmark():
    print("=========================================================")
    print("🚀 RUNNING TRAINING BENCHMARK")
    print("Sweeping hyperparameters to maximize GPU throughput")
    print("=========================================================")
    
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    results = []
    
    for bs in batch_sizes:
        print(f"\n--- Testing Batch Size: {bs} ---")
        try:
            samples_per_sec, max_mem = train(batch_size=bs, epochs=1, benchmark_mode=True)
            results.append((bs, samples_per_sec, max_mem))
        except Exception as e:
            if "out of memory" in str(e).lower() or "OutOfMemoryError" in e.__class__.__name__:
                print(f"Batch size {bs} caused OUT OF MEMORY. Stopping sweep.")
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise e
    
    print("\n=========================================================")
    print("📊 BENCHMARK RESULTS")
    print("=========================================================")
    print(f"{'Batch Size':>12} | {'Samples/Sec':>15} | {'Peak VRAM (MB)':>15}")
    print("-" * 50)
    
    best_bs = None
    best_throughput = 0
    for bs, sps, mem in results:
        print(f"{bs:>12} | {sps:>15.2f} | {mem:>15.2f}")
        if sps > best_throughput:
            best_throughput = sps
            best_bs = bs
            
    print("-" * 50)
    if best_bs:
        print(f"🏆 OPTIMAL BATCH SIZE: {best_bs} (Throughput: {best_throughput:.2f} samples/s)")
        print(f"Update your train.py or environment variables to use BATCH_SIZE={best_bs}.")
    print("=========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PolyZero Net")
    parser.add_argument("-b", "--benchmark", action="store_true", help="Run a benchmark to find optimal training parameters")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    else:
        train()

