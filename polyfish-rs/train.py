import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
import glob
import os

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 5
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
        
        # 8 Residual Blocks (matching Rust)
        # We name them res0, res1, res2, res3, res4, res5, res6, res7 to match Rust's pp("res0"), etc.
        self.res0 = ResBlock(64)
        self.res1 = ResBlock(64)
        self.res2 = ResBlock(64)
        self.res3 = ResBlock(64)
        self.res4 = ResBlock(64)
        self.res5 = ResBlock(64)
        self.res6 = ResBlock(64)
        self.res7 = ResBlock(64)
        
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
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        
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
    # 1. Load Data (Replay Buffer: Fresh + Archive)
    # We load the fresh games AND the most recent archived games to prevent catastrophic forgetting.
    fresh_files = glob.glob("games_*.safetensors")
    archive_files = sorted(glob.glob("archive/games_*.safetensors"), key=os.path.getmtime, reverse=True)
    
    # Keep window of last 20 batches instead of 50 (approx 300 games)
    # Reduced to manage 16GB RAM during CPU training
    replay_buffer_size = 5  # Reduced from 10
    game_files = fresh_files + archive_files[:replay_buffer_size]

    if not game_files:
        print("No training data found (checked ./ and ./archive/)!")
        return
        
    print(f"Training on {len(game_files)} files ({len(fresh_files)} fresh, {len(game_files)-len(fresh_files)} archived).")

    # Chunked loading implemented below

    
    # 2. Init Model
    # Dimensions must match Rust constants in `features.rs` / `mapper.rs`
    MAP_SIZE = 30
    INPUT_CHANNELS = 155
    NUM_ACTIONS = 30 * 30 * 64

    model = PolyZeroNet(INPUT_CHANNELS, NUM_ACTIONS, MAP_SIZE, MAP_SIZE).to(DEVICE)
    if os.path.exists("model.safetensors"):
        print("Loading existing model for fine-tuning...")
        model.load_state_dict(load_file("model.safetensors"))
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # 3. Training Loop with Chunked Loading
    # Shuffle files globally once per run or per epoch? 
    # Better to shuffle per epoch if we could, but rebuilding dataset is expensive.
    # We will iterate EPOCHS, and inside that, iterate chunks.
    
    import random
    import gc
    
    for epoch in range(EPOCHS):
        total_loss = 0
        total_batches = 0
        
        # Shuffle files for this epoch
        random.shuffle(game_files)
        
        # Process in chunks of 2 files (~2GB RAM)
        CHUNK_SIZE = 10
        
        for i in range(0, len(game_files), CHUNK_SIZE):
            chunk_files = game_files[i : i + CHUNK_SIZE]
            print(f"Epoch {epoch+1}: Loading chunk {i//CHUNK_SIZE + 1}/{(len(game_files)+CHUNK_SIZE-1)//CHUNK_SIZE} ({len(chunk_files)} files)...")
            
            chunk_states = []
            chunk_policies = []
            chunk_values = []
            
            for f in chunk_files:
                try:
                    data = load_file(f)
                    chunk_states.append(data["states"])
                    chunk_policies.append(data["policies"])
                    chunk_values.append(data["values"])
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    continue
            
            if not chunk_states:
                continue
                
            # Move chunk to device (or keep on CPU and move batches? Move to device for speed if VRAM allows)
            # 2GB of states might fit in VRAM (T4 has 16GB). 
            # 2 files * 1GB = 2GB. 2GB * ~2 (tensors) = 4GB. Should fit easily.
            try:
                states = torch.cat(chunk_states).to(DEVICE)
                policies = torch.cat(chunk_policies).to(DEVICE)
                values = torch.cat(chunk_values).to(DEVICE)
            except RuntimeError as e:
                print(f"OOM loading chunk to GPU: {e}. Falling back to CPU for storage.")
                states = torch.cat(chunk_states) # Keep on CPU
                policies = torch.cat(chunk_policies)
                values = torch.cat(chunk_values)
            
            # Clear temp lists
            del chunk_states, chunk_policies, chunk_values
            gc.collect()
            
            dataset_size = len(states)
            print(f"  Loaded {dataset_size} samples.")
            
            indices = torch.arange(dataset_size)
            indices = indices[torch.randperm(dataset_size)]
            
            # Mini-epochs on this chunk? Or just one pass?
            # Standard is one pass per global epoch.
            
            for j in range(0, dataset_size, BATCH_SIZE):
                batch_idx = indices[j : j + BATCH_SIZE]
                
                batch_states = states[batch_idx].to(DEVICE) # Ensure on device
                batch_policies = policies[batch_idx].to(DEVICE)
                batch_values = values[batch_idx].to(DEVICE)
                
                # Reshape (B, C, H, W)
                batch_states = batch_states.view(-1, INPUT_CHANNELS, MAP_SIZE, MAP_SIZE)
                
                p_logits, v_pred = model(batch_states)
                
                log_probs = torch.log_softmax(p_logits, dim=1)
                p_loss = -(batch_policies * log_probs).sum(dim=1).mean()
                v_loss = nn.MSELoss()(v_pred, batch_values)
                
                loss = p_loss + v_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_batches += 1
            
            # Cleanup chunk
            del states, policies, values
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if total_batches > 0:
            print(f"Epoch {epoch+1}/{EPOCHS} Avg Loss: {total_loss / total_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - No data processed")
        
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
            
    final_loss = total_loss / total_batches if total_batches > 0 else 0.0
    print(f"METRICS: {{\"loss\": {final_loss:.4f}}}")

    # 4. Save Model for Rust
    save_file(model.state_dict(), "model.safetensors")
    print("Saved model.safetensors")

if __name__ == "__main__":
    train()
