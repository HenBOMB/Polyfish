
import torch
from safetensors.torch import load_file
import sys
import os

try:
    if not os.path.exists("model.safetensors"):
        print("model.safetensors not found")
        sys.exit(0)
        
    data = load_file("model.safetensors")
    print("Keys:", data.keys())
    
    # Check for zeros
    all_zeros = True
    for k, v in data.items():
        if torch.any(v != 0):
            all_zeros = False
            print(f"{k} has non-zero values. Mean: {v.float().mean().item():.4f}, Std: {v.float().std().item():.4f}")
            break
            
    if all_zeros:
        print("CRITICAL: ALL WEIGHTS ARE ZERO!")
    else:
        print("Model initialized with non-zero weights.")
        
except Exception as e:
    print(f"Error: {e}")
