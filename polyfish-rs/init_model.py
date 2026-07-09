"""
Initialize model weights for the enhanced PolyZero network.

Creates a model.safetensors file with random weights matching the new architecture:
- Player state embedding (2 FC layers)
- 6 ResBlocks
- 7 decomposed policy heads
- 1 value head (win)
"""

import torch
import torch.nn as nn
from safetensors.torch import save_file

# Import architecture from train.py
import sys
sys.path.insert(0, '.')
from train import PolyZeroNet

def init_model():
    MAP_SIZE = 11
    SPATIAL_CHANNELS = 161  # Mirror of features.rs NUM_CHANNELS
    PLAYER_STATE_DIM = 10
    
    # Check if model already exists to avoid overwriting trained weights!
    import os
    if os.path.exists("model.safetensors"):
        print("✅ model.safetensors already exists. Skipping initialization.")
        return

    print("Initializing enhanced PolyZero network...")
    print(f"  Spatial input: {SPATIAL_CHANNELS} channels, {MAP_SIZE}x{MAP_SIZE}")
    print(f"  Player state: {PLAYER_STATE_DIM} features")
    print(f"  Architecture: 6 ResBlocks + 7 policy heads + 1 value head")
    
    model = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE)
    
    # Initialize weights with Xavier/He initialization
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Save
    save_file(model.state_dict(), "model.safetensors")
    print("✅ Saved to model.safetensors")
    
    # Verify save/load works
    print("\nVerifying save/load...")
    from safetensors.torch import load_file
    loaded_state = load_file("model.safetensors")
    model2 = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE)
    model2.load_state_dict(loaded_state)
    print("✅ Model loads successfully")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    spatial_input = torch.randn(batch_size, SPATIAL_CHANNELS, MAP_SIZE, MAP_SIZE)
    player_input = torch.randn(batch_size, PLAYER_STATE_DIM)
    
    model2.eval()
    with torch.no_grad():
        policy, values = model2(spatial_input, player_input)
    
    print(f"  Policy outputs:")
    for name, tensor in policy.items():
        print(f"    {name}: {tensor.shape}")
    
    print(f"  Value outputs:")
    for name, tensor in values.items():
        print(f"    {name}: {tensor.shape}")
    
    print("\n✅ Model initialization complete!")

if __name__ == "__main__":
    init_model()
