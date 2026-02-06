import torch
import torch.nn as nn
import torch.optim as optim
from safetensors.torch import load_file, save_file
import glob
import os
import random
import gc

# --- Configuration ---
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Architecture matching Rust `network.rs` (decomposed policy + auxiliary values)
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
    """
    Enhanced architecture with:
    - Player state embedding (global context)
    - 20 ResBlocks (increased capacity for A40)
    - 7 decomposed policy heads
    - 3 value heads (win, eco, mil)
    """
    def __init__(self, spatial_channels, player_state_dim, map_height, map_width):
        super().__init__()
        self.map_height = map_height
        self.map_width = map_width
        
        # Player state embedding: 10 -> 128 -> 128
        self.player_fc1 = nn.Linear(player_state_dim, 128)
        self.player_fc2 = nn.Linear(128, 128)
        self.player_relu = nn.ReLU()
        
        # Initial conv on spatial features (will concatenate with player embedding)
        # Input: spatial (149 channels) + player embedding (128 channels) = 277 total
        self.conv1 = nn.Conv2d(spatial_channels + 128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        
        # 20 Residual Blocks (matching Rust)
        self.res_blocks = nn.ModuleList([ResBlock(128) for _ in range(20)])
        
        # --- Decomposed Policy Heads ---
        
        # 1. Shared Policy Processing (Pooling -> FC)
        self.p_pool_conv = nn.Conv2d(128, 1, 1)
        self.p_pool_bn = nn.BatchNorm2d(1)
        self.p_fc_shared = nn.Linear(map_height * map_width, 128)
        
        # 2. Categorical Heads (Linear from shared latent)
        self.pi_action = nn.Linear(128, 10)      # vs.pp("pi_action")
        self.pi_struct = nn.Linear(128, 35)      # vs.pp("pi_struct")
        self.pi_unit = nn.Linear(128, 46)        # vs.pp("pi_unit")
        self.pi_tech = nn.Linear(128, 24)        # vs.pp("pi_tech")
        self.pi_ability = nn.Linear(128, 20)     # vs.pp("pi_ability")
        self.pi_reward = nn.Linear(128, 2)       # vs.pp("pi_reward")
        
        # 3. Spatial Heads (Conv from backbone)
        self.pi_source = nn.Conv2d(128, 1, 1)    # vs.pp("pi_source")
        self.pi_target = nn.Conv2d(128, 1, 1)    # vs.pp("pi_target")
        
        # --- Value Heads ---
        
        # 1. Shared Value Processing
        self.v_pool_conv = nn.Conv2d(128, 1, 1)
        self.v_pool_bn = nn.BatchNorm2d(1)
        self.v_fc_shared = nn.Linear(map_height * map_width, 128)
        
        # 2. Value Output Heads
        self.v_win = nn.Linear(128, 1)
        self.v_eco = nn.Linear(128, 1)
        self.v_mil = nn.Linear(128, 1)

    def forward(self, spatial_map, player_state):
        """
        Args:
            spatial_map: (B, 149, H, W) - tile features
            player_state: (B, 10) - global player stats
        
        Returns:
            Dictionary with policy and value outputs
        """
        batch_size = spatial_map.size(0)
        
        # Embed player state
        player_emb = self.player_relu(self.player_fc1(player_state))
        player_emb = self.player_relu(self.player_fc2(player_emb))
        
        # Broadcast player embedding to spatial dimensions
        player_emb = player_emb.view(batch_size, 128, 1, 1)
        player_emb = player_emb.expand(batch_size, 128, self.map_height, self.map_width)
        
        # Concatenate spatial features with player embedding
        x = torch.cat([spatial_map, player_emb], dim=1)
        
        # Initial conv + resblocks
        x = self.relu(self.bn1(self.conv1(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # --- Policy Heads ---
        # 1. Non-spatial policies (pooled -> shared latent)
        p_pooled = self.relu(self.p_pool_bn(self.p_pool_conv(x)))
        p_pooled = p_pooled.flatten(1)
        p_latent = self.relu(self.p_fc_shared(p_pooled))
        
        policy = {}
        policy['action_type'] = self.pi_action(p_latent)
        policy['structure_option'] = self.pi_struct(p_latent)
        policy['unit_option'] = self.pi_unit(p_latent)
        policy['tech_option'] = self.pi_tech(p_latent)
        policy['ability_option'] = self.pi_ability(p_latent)
        policy['reward_choice'] = self.pi_reward(p_latent)
        
        # 2. Spatial policies (direct from backbone)
        policy['source_spatial'] = self.pi_source(x).flatten(1)
        policy['target_spatial'] = self.pi_target(x).flatten(1)
        
        # --- Value Heads ---
        v_pooled = self.relu(self.v_pool_bn(self.v_pool_conv(x)))
        v_pooled = v_pooled.flatten(1)
        v_latent = self.relu(self.v_fc_shared(v_pooled))
        
        values = {}
        values['win'] = torch.tanh(self.v_win(v_latent))
        values['eco'] = self.v_eco(v_latent)
        values['mil'] = self.v_mil(v_latent)
        
        return policy, values

def compute_loss(policy_pred, values_pred, policy_targets, value_target):
    """
    Compute multi-head loss using decomposed targets.
    policy_targets is a dict containing the 7 target tensors.
    """
    total_policy_loss = 0.0
    
    # Loss weights for each head (tune as needed)
    weights = {
        'action_type': 1.0,
        'source_spatial': 0.5,  # harder, give it good weight
        'target_spatial': 0.5,
        'structure_option': 0.2,
        'unit_option': 0.2,
        'tech_option': 0.2,
        'ability_option': 0.2,
        'reward_choice': 0.1
    }
    
    # Helper for cross entropy with soft targets (probabilities)
    # Target shape: [B, C], Pred shape: [B, C] (logits)
    def soft_cross_entropy(logits, targets):
        log_probs = torch.nn.functional.log_softmax(logits, dim=1)
        return -(targets * log_probs).sum(dim=1).mean()

    for head_name, target in policy_targets.items():
        if head_name in policy_pred:
            pred = policy_pred[head_name]
            # Verify shapes match (except batch dim)
            if pred.shape != target.shape:
                # print(f"Shape mismatch for {head_name}: pred {pred.shape}, target {target.shape}")
                continue
                
            head_loss = soft_cross_entropy(pred, target)
            total_policy_loss += head_loss * weights.get(head_name, 1.0)
            
    # Value loss (Main Win + Aux Eco + Aux Mil)
    # Target shape for aux might be missing in dict if not loaded?
    # We passed 'value_target' as a dict of values now
    
    loss_win = nn.MSELoss()(values_pred['win'], value_target['win'])
    loss_eco = nn.MSELoss()(values_pred['eco'], value_target['eco'])
    loss_mil = nn.MSELoss()(values_pred['mil'], value_target['mil'])
    
    value_loss = loss_win + 0.2 * loss_eco + 0.2 * loss_mil
    
    # Total loss
    total_loss = total_policy_loss + value_loss
    
    return total_loss, total_policy_loss, value_loss

def train():
    print(f"Training on {DEVICE}")
    
    # 1. Load Data
    fresh_files = glob.glob("games_*.safetensors")
    archive_files = sorted(glob.glob("archive/games_*.safetensors"), key=os.path.getmtime, reverse=True)
    
    replay_buffer_size = 50
    game_files = fresh_files + archive_files[:replay_buffer_size]

    if not game_files:
        print("No training data found (checked ./ and ./archive/)!")
        return
        
    print(f"Training on {len(game_files)} files ({len(fresh_files)} fresh, {len(game_files)-len(fresh_files)} archived).")
    
    # 2. Init Model
    MAP_SIZE = 30
    SPATIAL_CHANNELS = 154
    PLAYER_STATE_DIM = 10

    model = PolyZeroNet(SPATIAL_CHANNELS, PLAYER_STATE_DIM, MAP_SIZE, MAP_SIZE).to(DEVICE)
    if os.path.exists("model.safetensors"):
        print("Loading existing model for fine-tuning...")
        try:
            model.load_state_dict(load_file("model.safetensors"))
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Starting from scratch.")
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Use CosineAnnealing with Warm Restarts for better convergence on short cycles
    # T_0=5 means it resets every 5 epochs (which is exactly our run length)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS, T_mult=1, eta_min=1e-5)

    # 3. Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        total_p_loss = 0
        total_v_loss = 0
        total_batches = 0
        
        random.shuffle(game_files)
        
        # Process in chunks
        CHUNK_SIZE = 10
        
        for i in range(0, len(game_files), CHUNK_SIZE):
            chunk_files = game_files[i : i + CHUNK_SIZE]
            print(f"Epoch {epoch+1}: Loading chunk {i//CHUNK_SIZE + 1}/{(len(game_files)+CHUNK_SIZE-1)//CHUNK_SIZE} ({len(chunk_files)} files)...")
            
            # Temporary storage for chunk data
            c_spatial = []
            c_player = []
            c_win = []
            c_eco = []
            c_mil = []
            
            c_heads = {
                'action_type': [], 'source_spatial': [], 'target_spatial': [],
                'structure_option': [], 'unit_option': [], 'tech_option': [], 'reward_choice': [], 'ability_option': []
            }
            
            for f in chunk_files:
                try:
                    data = load_file(f)
                    c_spatial.append(data["spatial_maps"])
                    c_player.append(data["player_states"])
                    c_win.append(data["values"])
                    
                    if "eco_targets" in data:
                        c_eco.append(data["eco_targets"])
                    if "mil_targets" in data:
                        c_mil.append(data["mil_targets"])
                    
                    # Load all policy heads
                    for head in c_heads.keys():
                        if head in data:
                            c_heads[head].append(data[head])
                        else:
                            pass
                            
                except Exception as e:
                    print(f"Error loading {f}: {e}")
                    continue
            
            if not c_spatial:
                continue
                
            # Stack into tensors
            try:
                spatial_maps = torch.cat(c_spatial)
                player_states = torch.cat(c_player)
                
                targets_win = torch.cat(c_win)
                # Handle cases where some files might miss eco/mil (though they should have them now)
                targets_eco = torch.cat(c_eco) if c_eco else torch.zeros_like(targets_win)
                targets_mil = torch.cat(c_mil) if c_mil else torch.zeros_like(targets_win)
                
                target_heads = {}
                for head, tensors in c_heads.items():
                    if tensors:
                        target_heads[head] = torch.cat(tensors)
                    
            except RuntimeError as e:
                print(f"OOM loading chunk: {e}")
                continue
            
            # Cleanup lists
            del c_spatial, c_player, c_win, c_eco, c_mil, c_heads
            gc.collect()
            
            dataset_size = len(spatial_maps)
            print(f"  Loaded {dataset_size} samples.")
            
            indices = torch.randperm(dataset_size)
            
            for j in range(0, dataset_size, BATCH_SIZE):
                batch_idx = indices[j : j + BATCH_SIZE]
                
                batch_spatial = spatial_maps[batch_idx].to(DEVICE)
                batch_player = player_states[batch_idx].to(DEVICE)
                
                batch_values = {
                    'win': targets_win[batch_idx].to(DEVICE),
                    'eco': targets_eco[batch_idx].to(DEVICE),
                    'mil': targets_mil[batch_idx].to(DEVICE)
                }
                
                batch_targets = {}
                for head, tensor in target_heads.items():
                    batch_targets[head] = tensor[batch_idx].to(DEVICE)
                
                # Reshape spatial to (B, C, H, W)
                batch_spatial = batch_spatial.view(-1, SPATIAL_CHANNELS, MAP_SIZE, MAP_SIZE)
                
                # --- DATA AUGMENTATION (Dihedral Group D4) ---
                # Randomly rotate and flip the batch to multiply effective data by 8x
                # This is standard for grid-based games like Go/Chess/Polytopia
                
                # 1. Random k for rot90 (0, 1, 2, 3)
                k = random.randint(0, 3)
                # 2. Random flip (True/False)
                do_flip = random.random() > 0.5
                
                if k > 0:
                    batch_spatial = torch.rot90(batch_spatial, k, [2, 3])
                    # Rotate spatial targets (source/target)
                    # Requires reshaping targets to (B, 1, H, W) then flattening back
                    if 'source_spatial' in batch_targets:
                        t = batch_targets['source_spatial'].view(-1, 1, MAP_SIZE, MAP_SIZE)
                        t = torch.rot90(t, k, [2, 3])
                        batch_targets['source_spatial'] = t.flatten(1)
                        
                    if 'target_spatial' in batch_targets:
                        t = batch_targets['target_spatial'].view(-1, 1, MAP_SIZE, MAP_SIZE)
                        t = torch.rot90(t, k, [2, 3])
                        batch_targets['target_spatial'] = t.flatten(1)
                        
                if do_flip:
                    batch_spatial = torch.flip(batch_spatial, [3]) # Flip horizontal
                    
                    if 'source_spatial' in batch_targets:
                        t = batch_targets['source_spatial'].view(-1, 1, MAP_SIZE, MAP_SIZE)
                        t = torch.flip(t, [3])
                        batch_targets['source_spatial'] = t.flatten(1)
                        
                    if 'target_spatial' in batch_targets:
                        t = batch_targets['target_spatial'].view(-1, 1, MAP_SIZE, MAP_SIZE)
                        t = torch.flip(t, [3])
                        batch_targets['target_spatial'] = t.flatten(1)
                
                # --- END AUGMENTATION ---
                
                optimizer.zero_grad()
                
                policy_pred, values_pred = model(batch_spatial, batch_player)
                
                loss, p_loss, v_loss = compute_loss(policy_pred, values_pred, batch_targets, batch_values)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                total_p_loss += p_loss.item()
                total_v_loss += v_loss.item()
                total_batches += 1
            
            del spatial_maps, player_states, values, target_heads
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if total_batches > 0:
            avg_loss = total_loss / total_batches
            avg_p_loss = total_p_loss / total_batches
            avg_v_loss = total_v_loss / total_batches
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} (Policy: {avg_p_loss:.4f}, Value: {avg_v_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} - No data processed")
        
        scheduler.step()
            
    final_loss = total_loss / total_batches if total_batches > 0 else 0.0
    print(f"METRICS: {{\"loss\": {final_loss:.4f}}}")

    # 4. Save Model
    save_file(model.state_dict(), "model.safetensors")
    print("Saved model.safetensors")

if __name__ == "__main__":
    train()

