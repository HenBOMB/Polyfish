
import torch
import torch.nn as nn
from safetensors.torch import save_file
import os

# Configuration matching train.py and Rust
MAP_SIZE = 30
INPUT_CHANNELS = 155
NUM_ACTIONS = 30 * 30 * 64

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

class PolyZeroNet(nn.Module):
    def __init__(self, input_channels, num_actions, map_height, map_width):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # 8 Residual Blocks (matching train.py and network.rs)
        self.res0 = ResBlock(64)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        self.res4 = ResBlock(64)
        self.res5 = ResBlock(64)
        self.res6 = ResBlock(64)
        self.res7 = ResBlock(64)
        
        # Policy Head
        self.p_conv1 = nn.Conv2d(64, 32, 1)
        self.p_bn1 = nn.BatchNorm2d(32)
        self.p_conv2 = nn.Conv2d(32, 64, 1)
        
        # Value Head
        self.v_conv = nn.Conv2d(64, 1, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * map_height * map_width, 64)
        self.v_fc2 = nn.Linear(64, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        pass # Not needed for init

if __name__ == "__main__":
    if os.path.exists("model.safetensors"):
        print("model.safetensors already exists. Skipping initialization.")
    else:
        print("Initializing new model with Kaiming weights...")
        model = PolyZeroNet(INPUT_CHANNELS, NUM_ACTIONS, MAP_SIZE, MAP_SIZE)
        save_file(model.state_dict(), "model.safetensors")
        print("Created model.safetensors")
