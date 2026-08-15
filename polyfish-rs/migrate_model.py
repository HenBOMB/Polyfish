import torch
from safetensors.torch import load_file, save_file
import os

# Must match train.py and features.rs
PLAYER_STATE_DIM = 16
FILTERS = 64


def migrate_model(file_path):
    print(f"Loading {file_path}...")
    try:
        state_dict = load_file(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    print("Checking and migrating state_dict heads...")
    if 'v_win.weight' not in state_dict:
        print("Not a PolyZero checkpoint (no v_win.weight); nothing to migrate.")
        return

    # Reject BatchNorm-era checkpoints (GroupNorm-era only)
    bn_keys = [k for k in state_dict if 'bn1.' in k or 'bn2.' in k or 'running_mean' in k]
    if bn_keys:
        print("ERROR: BatchNorm-era checkpoint detected (found keys like bn1.*, bn2.*, running_mean).")
        print("These checkpoints are incompatible with current GroupNorm-only code.")
        print("Please retrain with a GroupNorm-compatible model (checkpoints/bn_era/ holds old ones for reference).")
        return

    filters = state_dict['v_win.weight'].shape[1]

    migrated = False

    # ------------------------------------------------------------------
    # 1. Value-head swap: v_pool_conv / v_fc_shared  →  v_fc1 / v_fc2
    # ------------------------------------------------------------------
    stale_value_prefixes = ("v_pool_conv.", "v_fc_shared.")
    stale_keys = [k for k in state_dict if k.startswith(stale_value_prefixes)]
    if stale_keys or "v_fc1.weight" not in state_dict:
        for k in stale_keys:
            print(f"  Removing stale key: {k}")
            del state_dict[k]
        if "v_fc1.weight" not in state_dict:
            # v_fc1: Linear(2*filters, filters)
            print("  Initialising v_fc1 (Linear(128, 64)) fresh")
            state_dict["v_fc1.weight"] = torch.randn(filters, 2 * filters) * 0.01
            state_dict["v_fc1.bias"] = torch.zeros(filters)
        if "v_fc2.weight" not in state_dict:
            # v_fc2: Linear(filters, filters)
            print("  Initialising v_fc2 (Linear(64, 64)) fresh")
            state_dict["v_fc2.weight"] = torch.randn(filters, filters) * 0.01
            state_dict["v_fc2.bias"] = torch.zeros(filters)
        migrated = True

    # ------------------------------------------------------------------
    # 2. v_progress head
    # ------------------------------------------------------------------
    if "v_progress.weight" not in state_dict:
        print("  Initialising v_progress fresh")
        state_dict["v_progress.weight"] = torch.randn(1, filters) * 0.01
        state_dict["v_progress.bias"] = torch.zeros(1)
        migrated = True

    # ------------------------------------------------------------------
    # 3. Player embeddings: resize to PLAYER_STATE_DIM
    # ------------------------------------------------------------------
    for name in ("player_pos_embeddings", "player_feature_embeddings"):
        if name not in state_dict:
            print(f"  Creating {name} ({PLAYER_STATE_DIM}, {filters})")
            state_dict[name] = torch.randn(PLAYER_STATE_DIM, filters) * 0.01
            migrated = True
        elif state_dict[name].shape[0] != PLAYER_STATE_DIM:
            old_dim = state_dict[name].shape[0]
            print(f"  Resizing {name} ({old_dim}, {filters}) → ({PLAYER_STATE_DIM}, {filters})")
            new_param = torch.randn(PLAYER_STATE_DIM, filters) * 0.01
            keep = min(old_dim, PLAYER_STATE_DIM)
            new_param[:keep] = state_dict[name][:keep]
            state_dict[name] = new_param
            migrated = True

    # ------------------------------------------------------------------
    # 4. Fog-memory: zero-pad conv1 input channels → 142
    # ------------------------------------------------------------------
    conv1 = state_dict.get("conv1.weight")
    if conv1 is not None and conv1.shape[1] < 142:
        old_ch = conv1.shape[1]
        print(f"  Padding conv1.weight input channels {old_ch} → 142")
        pad = torch.zeros(conv1.shape[0], 142 - old_ch, conv1.shape[2], conv1.shape[3])
        state_dict["conv1.weight"] = torch.cat([conv1, pad], dim=1)
        migrated = True

    if not migrated:
        print("Checkpoint is already up-to-date. No changes needed.")
        return

    backup_path = file_path + ".bak"
    os.rename(file_path, backup_path)
    print(f"Backed up original model to {backup_path}")

    save_file(state_dict, file_path)
    print(f"Successfully saved migrated model to {file_path}")


if __name__ == "__main__":
    if os.path.exists("model.safetensors"):
        migrate_model("model.safetensors")
    else:
        print("model.safetensors not found in current directory.")
