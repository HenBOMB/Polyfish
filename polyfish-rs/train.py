import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
import glob
import hashlib
import json
import os
import random
import gc
import time

OPTIMIZER_STATE_PATH = "optimizer_state.pt"

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
# Mix teachers/ into every iteration (see the teacher-anchor note in train()).
# Set 0 to train on self-play data only — matters most right after an archive
# clear, where the teachers would otherwise dominate the first few iterations.
USE_TEACHERS_DS = os.environ.get("USE_TEACHERS_DS", "1") == "1"
# Auxiliary training-only heads (end-game ownership / fog occupancy / SPT+5 /
# opponent tech). Rust inference loads model.safetensors by name and never
# reads these. Targets ship in games files from Jul 2026; files without them
# (old archives, teachers) are masked out per sample, never zero-filled.
AUX_DIMS = {'aux_ownership': 121, 'aux_fog_units': 121, 'aux_spt': 2, 'aux_opp_tech': 42, 'aux_pursuit': 1,
            'aux_city_spt': 121}
AUX_WEIGHTS = {
    'aux_ownership': float(os.environ.get("AUX_OWN_W", "0.3")),
    'aux_fog_units': float(os.environ.get("AUX_FOG_W", "0.2")),
    'aux_spt': float(os.environ.get("AUX_SPT_W", "0.1")),
    'aux_opp_tech': float(os.environ.get("AUX_TECH_W", "0.1")),
    'aux_pursuit': float(os.environ.get("AUX_PURSUIT_W", "0.1")),
    # Per-city SPT five turns out, on that city's tile. The question behind it
    # is what a city's growth is worth -- whether unlocking its outer ring pays
    # -- which nothing in the input says. Level and progress tell the net which
    # city is near a threshold; this asks it to learn what crossing one buys.
    'aux_city_spt': float(os.environ.get("AUX_CITY_SPT_W", "0.1")),
}
# EXP_ELO_013: persistent KL-anchor to a frozen reference policy (AlphaStar-
# style — pulls the live policy toward a known-good checkpoint throughout RL,
# rather than only at init). KL_REF_MODEL is a path to the frozen checkpoint;
# unset/empty disables the feature entirely (zero cost — no ref model is
# loaded). KL(ref || policy) w.r.t. the trainable policy reduces to plain
# cross-entropy against the ref's (fixed) distribution, so this reuses the
# same soft_cross_entropy already used for the main policy loss.
KL_REF_MODEL = os.environ.get("KL_REF_MODEL", "")
KL_REF_WEIGHT = float(os.environ.get("KL_REF_WEIGHT", "0.0"))

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
        # v_pool_conv widened 1->8 channels (Jul 2026): removes the 1-channel
        # value bottleneck. v_fc_shared in_features track it (8 * H * W).
        self.v_pool_conv = nn.Conv2d(self.filters, 8, 1)
        self.v_fc_shared = nn.Linear(8 * map_height * map_width, self.filters)
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
        self.aux_pursuit = nn.Linear(self.filters, 1)
        self.aux_city_spt = nn.Conv2d(self.filters, 1, 1)

        # --- Macro policy head (EXP_ELO_061, Stage 3b) ---
        # Unlike the aux_* heads above, this one IS mirrored into Rust
        # (macro-mcts root prior at inference, like aux_fog) -- see
        # network.rs's pi_macro_stance/pi_macro_order for the Rust side,
        # which this must match exactly: same shapes, same trunk source,
        # same activation APPLIED HERE (not left as loss-friendly logits
        # the way the aux dict is) so `values`-style consumers on both
        # sides read the same probabilities. Stance: mutually exclusive
        # over Stance (Grow/Arm/Unlock/Save), off v_latent, softmax. Order:
        # one per-tile intensity plane per OrderKind (3), off the trunk
        # directly like aux_fog, sigmoid -- non-exclusive across kinds and
        # across same-kind targets (a goal routinely carries more than one
        # order), so per-tile independent probabilities, not a spatial
        # softmax. When the training loss for this head is wired, it reads
        # these post-activation values with BCE/NLL rather than the raw
        # logits (a deliberate divergence from the aux_fog convention,
        # noted so it isn't "fixed" to match aux_fog by mistake later).
        self.pi_macro_stance = nn.Linear(self.filters, 4)
        self.pi_macro_order = nn.Conv2d(self.filters, 3, 1)

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
        aux['aux_pursuit'] = self.aux_pursuit(gap)
        aux['aux_city_spt'] = self.aux_city_spt(x).flatten(1)

        # --- Value Heads ---
        v_input = x.detach() if DETACH_VALUE_TRUNK else x
        v_pooled = self.v_pool_conv(v_input).flatten(1)
        v_latent = self.relu(self.v_fc_shared(v_pooled))

        values = {}
        values['win'] = torch.tanh(self.v_win(v_latent))
        values['progress'] = self.v_progress(v_latent)

        # EXP_ELO_061: reads v_latent (stance) / x (order) -- the SAME
        # trunk tensors network.rs's mirror reads (v_latent there too,
        # `shared` there == x here), post cross-attention either way.
        values['macro_stance'] = torch.softmax(self.pi_macro_stance(v_latent), dim=1)
        values['macro_order'] = torch.sigmoid(self.pi_macro_order(x).flatten(1))

        return policy, values, aux

def city_masked_mse(pred, target):
    """MSE over the city tiles only, not all 121.

    98% of every aux_city_spt row is empty board, so an unmasked mean makes
    "output zero everywhere" overwhelmingly the cheapest answer: after ten
    iterations the head had learned WHERE cities are (it separates city from
    empty by +0.054) but not what they are worth — predicting mean 0.057
    against a target mean of 0.187, for an R^2 of -1.73 on the cells that
    actually carry the signal. Masking removes the 98% that was drowning it.

    The mask is `target != 0`, so a city with zero production — under siege,
    per get_city_production — is skipped rather than taught. Rare, and the
    alternative is shipping a separate presence plane in the games file.

    Scale note: this reads ~10x higher than the unmasked version did. It is a
    different quantity, not a regression; readings before Aug 9 2026 are NOT
    comparable.
    """
    m = (target != 0).float()
    n = m.sum(dim=1).clamp(min=1.0)
    return (((pred - target) ** 2) * m).sum(dim=1) / n


def compute_loss(policy_pred, values_pred, policy_targets, value_target,
                 aux_pred=None, aux_targets=None, aux_mask=None,
                 ref_policy_pred=None, kl_ref_weight=0.0):
    """
    Compute multi-head loss using decomposed targets.
    policy_targets is a dict containing the 7 target tensors.
    ref_policy_pred/kl_ref_weight (EXP_ELO_013): if given, adds
    kl_ref_weight * KL(ref || policy) per head, anchoring the live policy to
    a frozen reference so RL fine-tuning can't drift arbitrarily far from it.
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

    # KL-anchor (EXP_ELO_013): KL(ref || policy) w.r.t. the trainable policy
    # is, up to an additive constant independent of policy params, the same
    # as cross-entropy against the ref's (fixed, no-grad) distribution — so
    # this reuses soft_cross_entropy rather than a separate KL formula.
    kl_losses = {}
    if ref_policy_pred is not None and kl_ref_weight > 0.0:
        for head_name in policy_targets:
            if head_name in policy_pred and head_name in ref_policy_pred:
                ref_probs = torch.nn.functional.softmax(ref_policy_pred[head_name], dim=1)
                kl_losses[head_name] = soft_cross_entropy(policy_pred[head_name], ref_probs)

    loss_win = nn.MSELoss()(values_pred['win'], value_target['win'])

    loss_progress = 0.0
    if 'progress' in value_target and 'progress' in values_pred:
        loss_progress = nn.MSELoss()(values_pred['progress'], value_target['progress'])

    # Prioritize winning/losing.
    value_loss = VALUE_LOSS_WEIGHT * loss_win + loss_progress

    # Total loss
    total_loss = total_policy_loss + value_loss

    for head_name, l in kl_losses.items():
        total_loss = total_loss + kl_ref_weight * l * weights.get(head_name, 1.0)

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
            'aux_pursuit': lambda p, t: ((p - t) ** 2).mean(dim=1),
            'aux_city_spt': city_masked_mse,
        }
        for k, fn in per_sample.items():
            if AUX_WEIGHTS[k] == 0.0 or k not in aux_targets:
                continue
            mask_k = aux_mask[k]
            denom = mask_k.sum().clamp(min=1.0)
            l = (fn(aux_pred[k], aux_targets[k]) * mask_k).sum() / denom
            aux_losses[k] = l
            total_loss = total_loss + AUX_WEIGHTS[k] * l

    # loss_win returned raw (unweighted, no aux terms) — value_r2 needs it;
    # value_loss alone can't be unweighted once loss_progress is mixed in.
    return total_loss, total_policy_loss, value_loss, loss_win, aux_losses, kl_losses

def hash_state_dict(state_dict):
    """Fingerprint of a model's weights, used to verify optimizer_state.pt was
    saved against the exact model.safetensors currently on disk — this
    campaign restores model.safetensors from checkpoints constantly, and
    replaying Adam's momentum against a different set of weights than the
    ones it was tracking would corrupt the step, not just waste it.
    """
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()

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
    # Teacher anchor: mix these into every iteration so gradients keep pulling
    # toward known-good play regardless of self-play drift (RLHF-style reference
    # anchor). Never archived or pruned. USE_TEACHERS_DS=0 opts out entirely.
    teacher_files = sorted(glob.glob("teachers/games_*.safetensors")) if USE_TEACHERS_DS else []
    game_files = fresh_files + archive_files[:replay_buffer_size] + teacher_files

    if not game_files:
        print("No training data found (checked ./, ./archive/, and ./teachers/)!")
        return

    print(f"Training on {len(game_files)} files ({len(fresh_files)} fresh, "
          f"{len(archive_files[:replay_buffer_size])} archived, {len(teacher_files)} teacher"
          f"{'' if USE_TEACHERS_DS else ', USE_TEACHERS_DS=0'}).")
    
    # 2. Init Model
    MAP_SIZE = 11
    SPATIAL_CHANNELS = 169  # mirror of features.rs NUM_CHANNELS (incl. obs memory, pursuit, goal channels + v7 SAVE stance plane)
    PLAYER_STATE_DIM = 10

    model = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE).to(DEVICE)
    loaded_state = None
    if os.path.exists("model.safetensors"):
        try:
            loaded_state = load_file("model.safetensors")
            # strict=False: newly added heads (e.g. v_progress) are absent
            # from older checkpoints and must start fresh WITHOUT discarding
            # the trained trunk — a strict load would throw and silently
            # reinitialize everything via the except branch below.
            missing, unexpected = model.load_state_dict(loaded_state, strict=False)
            if missing or unexpected:
                print(f"Partial checkpoint load — missing: {missing}, unexpected: {unexpected}")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting from scratch.")
            loaded_state = None
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use CosineAnnealing with Warm Restarts for better convergence on short cycles
    # T_0=5 means it resets every 5 epochs (which is exactly our run length)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS, T_mult=1, eta_min=1e-5)

    # Resume Adam/scheduler state from the last call, but only if it was saved
    # against the exact weights we just loaded — a fresh Adam is safer than a
    # stale one if model.safetensors was swapped (checkpoint restore, --reset,
    # manual revert) since last training. Fresh runs and mismatches both fall
    # through to the plain optimizer/scheduler created above.
    if loaded_state is not None and os.path.exists(OPTIMIZER_STATE_PATH):
        try:
            current_hash = hash_state_dict(loaded_state)
            saved = torch.load(OPTIMIZER_STATE_PATH, map_location=DEVICE)
            if saved.get("model_hash") == current_hash:
                optimizer.load_state_dict(saved["optimizer"])
                scheduler.load_state_dict(saved["scheduler"])
                print("Resumed Adam/scheduler state (model hash matched).")
            else:
                print(
                    f"{OPTIMIZER_STATE_PATH} model hash mismatch (weights changed "
                    "since last save) — starting fresh Adam."
                )
        except Exception as e:
            print(f"Could not load {OPTIMIZER_STATE_PATH} ({e}) — starting fresh Adam.")
    elif loaded_state is not None:
        print(f"No {OPTIMIZER_STATE_PATH} found — starting fresh Adam.")

    # EXP_ELO_013: frozen reference model for the KL-anchor. Loaded once,
    # never trained (no_grad forward only) — a separate instance from `model`
    # so the optimizer only ever sees `model.parameters()`.
    ref_model = None
    if KL_REF_MODEL and KL_REF_WEIGHT > 0.0:
        ref_model = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE).to(DEVICE)
        ref_missing, ref_unexpected = ref_model.load_state_dict(
            load_file(KL_REF_MODEL), strict=False
        )
        if ref_missing or ref_unexpected:
            print(f"KL-anchor ref partial load — missing: {ref_missing}, unexpected: {ref_unexpected}")
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"⚓ KL-anchor active: ref={KL_REF_MODEL} weight={KL_REF_WEIGHT}")
    elif KL_REF_MODEL and KL_REF_WEIGHT <= 0.0:
        print(f"KL_REF_MODEL set but KL_REF_WEIGHT<=0 — KL-anchor disabled.")

    # 3. Two load modes, decided by TRAIN_RAM_GB (below): buffers projected to
    # fit are loaded + concatenated ONCE and cached across every epoch (the
    # reload was ~20-25s of pure CPU per pass); larger buffers STREAM — each
    # chunk reloaded per epoch and freed after use — so resident RAM stays
    # bounded by one chunk regardless of buffer size.
    random.shuffle(game_files)
    CHUNK_SIZE = int(os.environ.get("TRAIN_CHUNK_FILES", "10"))
    num_chunks = (len(game_files) + CHUNK_SIZE - 1) // CHUNK_SIZE
    expected_flat = SPATIAL_CHANNELS * MAP_SIZE * MAP_SIZE

    def load_chunk(chunk_files, chunk_idx):
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
        c_aux_mask = {k: [] for k in AUX_DIMS}

        for f in chunk_files:
            try:
                data = load_file(f)
                sp = data["spatial_maps"]
                if sp.shape[1] != expected_flat:
                    # Channels are only ever appended at the end of the layout,
                    # so any narrower file is index-stable: zero-pad the missing
                    # trailing planes (covers 154 pre-obs-memory, 161 pre-pursuit,
                    # and any future append). Wider-than-expected is a real bug.
                    if sp.shape[1] < expected_flat and sp.shape[1] % (MAP_SIZE * MAP_SIZE) == 0:
                        sp = torch.nn.functional.pad(sp, (0, expected_flat - sp.shape[1]))
                    else:
                        raise ValueError(
                            f"{f}: spatial width {sp.shape[1]} is not a zero-pad of "
                            f"the current layout ({expected_flat}) — channels only append"
                        )
                # Halved to fp16 in RAM — this is the dominant tensor (e.g.
                # ~29GB at fp32 for a 13-file/380K-sample buffer) and now
                # stays resident across all epochs (see cached_chunks below),
                # which pushed a real run into memory pressure (34/36GB used,
                # batch/s dropped ~4x). Model input, not a weight or label,
                # so upcasting to fp32 per-batch (below) costs nothing and
                # loses nothing training-relevant.
                c_spatial.append(sp.half())
                # Since Jul 28 self_play writes f16 files; legacy files are
                # f32. .float() normalizes both so torch.cat never sees a
                # dtype mix (spatial stays half by design, upcast per-batch).
                c_player.append(data["player_states"].float())
                c_win.append(data["values"].float())
                if "progress" in data:
                    c_progress.append(data["progress"].float())
                else:
                    c_progress.append(torch.zeros_like(data["values"], dtype=torch.float32))

                # Load all policy heads
                for head in c_heads.keys():
                    if head in data:
                        c_heads[head].append(data[head].float())
                    else:
                        pass

                # Aux targets: presence is PER KEY, not per file. All-or-
                # nothing meant adding one head silently unsupervised every
                # other head on every file written before it existed.
                # Masked, never zero-filled — a zero-filled target would
                # silently train the head toward 0 on legacy samples.
                n = data["values"].shape[0]
                for k, d in AUX_DIMS.items():
                    if k in data:
                        c_aux[k].append(data[k].float())
                        c_aux_mask[k].append(torch.ones(n))
                    else:
                        c_aux[k].append(torch.zeros(n, d))
                        c_aux_mask[k].append(torch.zeros(n))

            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

        if not c_spatial:
            return None

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
            aux_mask = {k: torch.cat(v) for k, v in c_aux_mask.items()}

        except RuntimeError as e:
            print(f"OOM loading chunk: {e}")
            return None

        # Cleanup lists
        del c_spatial, c_player, c_win, c_progress, c_aux, c_aux_mask
        gc.collect()

        dataset_size = len(spatial_maps)
        print(f"  Loaded {dataset_size} samples.")

        return {
            "spatial_maps": spatial_maps,
            "player_states": player_states,
            "targets_win": targets_win,
            "targets_progress": targets_progress,
            "target_heads": target_heads,
            "target_aux": target_aux,
            "aux_mask": aux_mask,
            "dataset_size": dataset_size,
            "chunk_idx": chunk_idx,
        }

    chunk_groups = [
        (game_files[i : i + CHUNK_SIZE], i // CHUNK_SIZE + 1)
        for i in range(0, len(game_files), CHUNK_SIZE)
    ]
    # RAM budget: resident ≈ file bytes (f16 spatial dominates and stays f16;
    # the small non-spatial tensors upcast to f32) — ~1.1x disk. Over budget,
    # chunks STREAM per epoch (reloaded each pass, freed after use) instead of
    # caching for the whole run — re-buying the ~20-25s/pass torch.cat cost
    # the cache removed, in exchange for bounded RSS at any buffer size.
    TRAIN_RAM_GB = float(os.environ.get("TRAIN_RAM_GB", "16"))
    on_disk = sum(os.path.getsize(f) for f in game_files if os.path.exists(f))
    projected_gb = on_disk * 1.1 / 1e9
    cached_chunks = None
    if projected_gb > TRAIN_RAM_GB:
        print(
            f"📦 Streaming chunks: projected ~{projected_gb:.1f}GB resident "
            f"> TRAIN_RAM_GB={TRAIN_RAM_GB:.0f} — reloading per epoch."
        )
    else:
        cached_chunks = [
            c for files, idx in chunk_groups if (c := load_chunk(files, idx))
        ]
        if not cached_chunks:
            print("No usable training data after loading!")
            return

    def iter_epoch_chunks():
        if cached_chunks is not None:
            yield from cached_chunks
            return
        groups = list(chunk_groups)
        random.shuffle(groups)
        for files, idx in groups:
            c = load_chunk(files, idx)
            if c is not None:
                yield c

    if cached_chunks is not None:
        epoch_batch_estimate = sum(
            (c["dataset_size"] + BATCH_SIZE - 1) // BATCH_SIZE for c in cached_chunks
        )
    else:
        # Sample counts straight from safetensors headers — no data read.
        from safetensors import safe_open
        n_samples = 0
        for f in game_files:
            try:
                with safe_open(f, framework="pt") as sf:
                    n_samples += sf.get_slice("values").get_shape()[0]
            except Exception:
                pass
        epoch_batch_estimate = (n_samples + BATCH_SIZE - 1) // BATCH_SIZE
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
        total_aux_n_t = {k: torch.zeros((), device=DEVICE) for k in AUX_DIMS}
        total_kl_t = torch.zeros((), device=DEVICE)

        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        for chunk in iter_epoch_chunks():
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
                batch_aux_mask = {k: m[batch_idx].to(DEVICE) for k, m in aux_mask.items()}

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
                        # Every tile-space aux target must co-transform too.
                        for head in ('aux_ownership', 'aux_fog_units', 'aux_city_spt'):
                            t = batch_aux[head].view(-1, 1, MAP_SIZE, MAP_SIZE)
                            if k > 0:
                                t = torch.rot90(t, k, [2, 3])
                            if do_flip:
                                t = torch.flip(t, [3])
                            batch_aux[head] = t.flatten(1)

                optimizer.zero_grad()

                policy_pred, values_pred, aux_pred = model(batch_spatial, batch_player)

                # EXP_ELO_013: ref forward pass on the identical (possibly
                # D4-augmented) batch, so its spatial heads align tile-for-
                # tile with the live model's without any extra transform.
                ref_policy_pred = None
                if ref_model is not None:
                    with torch.no_grad():
                        ref_policy_pred, _, _ = ref_model(batch_spatial, batch_player)

                loss, p_loss, v_loss, v_win_loss, aux_losses, kl_losses = compute_loss(
                    policy_pred, values_pred, batch_targets, batch_values,
                    aux_pred, batch_aux, batch_aux_mask,
                    ref_policy_pred, KL_REF_WEIGHT)

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
                for k, l in aux_losses.items():
                    n_k = batch_aux_mask[k].sum().detach()
                    total_aux_n_t[k] += n_k
                    total_aux_t[k] += l.detach() * n_k
                if kl_losses:
                    total_kl_t += sum(kl_losses.values()).detach()

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
    total_aux_n = {k: v.item() for k, v in total_aux_n_t.items()}
    final_aux = {
        k: (total_aux_t[k].item() / total_aux_n[k] if total_aux_n.get(k, 0) > 0 else 0.0)
        for k in AUX_DIMS
    }
    final_kl = (total_kl_t / total_batches).item() if total_batches > 0 else 0.0

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

    # Persist Adam/scheduler state alongside the weights it was computed
    # against (fingerprinted via hash_state_dict) so the next train() call can
    # resume momentum instead of starting from zero every iteration.
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_hash": hash_state_dict(half_state),
        },
        OPTIMIZER_STATE_PATH,
    )

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
                "aux_pursuit_loss": round(final_aux['aux_pursuit'], 4),
                "aux_city_spt_loss": round(final_aux['aux_city_spt'], 4),
                # Per-head supervised-sample counts. A head added later reads
                # lower than the rest until legacy archives age out; a head
                # reading 0 that used to read high means the mask broke.
                "aux_supervised": {k: int(v) for k, v in total_aux_n.items()},
                "kl_ref_loss": round(final_kl, 4),
            },
            f,
        )

if __name__ == "__main__":
    train()

