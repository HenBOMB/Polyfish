import torch
from safetensors.torch import load_file, save_file
import os

def migrate_model(file_path):
    print(f"Loading {file_path}...")
    try:
        state_dict = load_file(file_path)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    print("Checking and migrating state_dict heads...")
    # We need to match the filter size, which is 64.
    # v_win.weight is [1, 64], v_win.bias is [1]
    filters = state_dict['v_win.weight'].shape[1]
    
    # Initialize v_progress weights (using Xavier/Kaiming init or similar, 
    # here just a small random normal scaled to avoid huge initial gradients)
    if "v_progress.weight" not in state_dict:
        state_dict["v_progress.weight"] = torch.randn(1, filters) * 0.01
        state_dict["v_progress.bias"] = torch.zeros(1)
        
    if "player_pos_embeddings" not in state_dict:
        print("Adding player_pos_embeddings to state_dict...")
        state_dict["player_pos_embeddings"] = torch.randn(10, filters) * 0.01

    # Fog memory (Jul 2026): zero-pad conv1 input 136 -> 142 so the six new
    # memory channels start invisible to a pre-memory checkpoint (notes-memory.md).
    conv1 = state_dict.get("conv1.weight")
    if conv1 is not None and conv1.shape[1] == 136:
        print(f"Padding conv1.weight input channels {conv1.shape[1]} -> 142...")
        pad = torch.zeros(conv1.shape[0], 142 - conv1.shape[1], conv1.shape[2], conv1.shape[3])
        state_dict["conv1.weight"] = torch.cat([conv1, pad], dim=1)

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
