
import torch
from safetensors.torch import load_file
import sys
import os

try:
    if not os.path.exists("model.safetensors"):
        print("model.safetensors not found")
        sys.exit(0)
        
    data = load_file("model.safetensors")
    
    # Check conv1.weight explicitly
    if "conv1.weight" in data:
        w = data["conv1.weight"]
        print(f"conv1.weight: Mean={w.float().mean().item():.6f}, Std={w.float().std().item():.6f}, Max={w.abs().max().item():.6f}")
        if w.abs().max().item() == 0:
             print("CRITICAL: conv1.weight IS ALL ZEROS")
    else:
        print("conv1.weight not found in keys:", data.keys())

    # Check last layer
    if "v_fc2.weight" in data:
        w = data["v_fc2.weight"]
        print(f"v_fc2.weight: Mean={w.float().mean().item():.6f}, Std={w.float().std().item():.6f}")
        
except Exception as e:
    print(f"Error: {e}")
