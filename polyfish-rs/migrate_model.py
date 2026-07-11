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

    if "v_progress.weight" in state_dict:
        print("Model already contains v_progress. Migration not needed.")
        return

    print("Adding v_progress head to state_dict...")
    
    # We need to match the filter size, which is 64.
    # v_win.weight is [1, 64], v_win.bias is [1]
    filters = state_dict['v_win.weight'].shape[1]
    
    # Initialize v_progress weights (using Xavier/Kaiming init or similar, 
    # here just a small random normal scaled to avoid huge initial gradients)
    state_dict["v_progress.weight"] = torch.randn(1, filters) * 0.01
    state_dict["v_progress.bias"] = torch.zeros(1)

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
