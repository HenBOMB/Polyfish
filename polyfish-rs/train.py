import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
import glob
import os

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture must match Rust `network.rs`
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
        
        # 4 Residual Blocks (matching Rust)
        # We name them res0, res1, res2, res3 to match Rust's pp("res0"), etc.
        self.res0 = ResBlock(64)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        
        # Policy Head (Fully Conv)
        self.p_conv1 = nn.Conv2d(64, 32, 1)
        self.p_bn1 = nn.BatchNorm2d(32)
        self.p_conv2 = nn.Conv2d(32, 64, 1)
        
        # Value Head
        self.v_conv = nn.Conv2d(64, 1, 1)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * map_height * map_width, 64)
        self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Policy
        p = self.relu(self.p_bn1(self.p_conv1(x)))
        p = self.p_conv2(p) # (B, 64, H, W)
        p_logits = p.flatten(1) # (B, 57600)
        
        # Value
        v = self.relu(self.v_bn(self.v_conv(x)))
        v = v.flatten(1)
        v = self.relu(self.v_fc1(v))
        v_out = torch.tanh(self.v_fc2(v))
        
        return p_logits, v_out

def train():
    print(f"Training on {DEVICE}")
    
    # 1. Load Data
    game_files = glob.glob("games_*.safetensors") # Matches format from Rust
    if not game_files:
        print("No training data found!")
        return

    all_states = []
    all_policies = []
    all_values = []
    
    for f in game_files:
        data = load_file(f)
        all_states.append(data["states"])
        all_policies.append(data["policies"])
        all_values.append(data["values"])
    
    states = torch.cat(all_states).to(DEVICE)
    policies = torch.cat(all_policies).to(DEVICE)
    values = torch.cat(all_values).to(DEVICE)
    
    print(f"Loaded {len(states)} samples.")
    
    # 2. Init Model
    # Dimensions must match Rust constants in `features.rs` / `mapper.rs`
    # You might want to pass these dynamically or parse from headers, 
    # but for now we hardcode based on known Rust constants.
    # Assuming Small Map (11x11? Or 256 tiles? MapSize::Small is 11x11 = 121 tiles usually, need to check `features.rs`)
    # Wait, `features::MAP_HEIGHT` was used in `self_play.rs`
    
    # Loaded from features.rs and mapper.rs
    MAP_SIZE = 30 
    INPUT_CHANNELS = 27
    NUM_ACTIONS = 30 * 30 * 64 # 57600
    
    model = PolyZeroNet(INPUT_CHANNELS, NUM_ACTIONS, MAP_SIZE, MAP_SIZE).to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    dataset_size = len(states)
    indices = torch.arange(dataset_size)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        indices = indices[torch.randperm(dataset_size)]
        
        for i in range(0, dataset_size, BATCH_SIZE):
            batch_idx = indices[i : i + BATCH_SIZE]
            batch_states = states[batch_idx]
            batch_policies = policies[batch_idx]
            batch_values = values[batch_idx]
            
            p_logits, v_pred = model(batch_states)
            
            # Loss
            # Policy: CrossEntropy (prob targets need Softmax? No, CrossEntropyLoss expects class indices usually, 
            # but if we have prob distribution targets, we use KLDiv or CrossEntropy with soft targets)
            # PyTorch `CrossEntropyLoss` supports probabilistic targets in newer versions, or we use `v_pred` vs `batch_policies`
            
            # Custom Cross Entropy for Probabilities: -sum(target * log_softmax(pred))
            log_probs = torch.log_softmax(p_logits, dim=1)
            p_loss = -(batch_policies * log_probs).sum(dim=1).mean()
            
            # Value: MSE
            v_loss = nn.MSELoss()(v_pred, batch_values)
            
            loss = p_loss + v_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss / (dataset_size / BATCH_SIZE):.4f}")
        
    # 4. Save Model for Rust
    save_file(model.state_dict(), "model.safetensors")
    print("Saved model.safetensors")

if __name__ == "__main__":
    train()
