import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class PolytopiaNet(nn.Module):
    def __init__(self,
        dim_map_channels,
        dim_map_size,
        dim_player,
        dim_struct,
        dim_skill,
        dim_unit,
        dim_tech,
        num_action_types,
        num_res_blocks=8,
        num_hidden_channels=128,
        num_player_hidden=32,
    ):
        super().__init__()

        self.dim_map_size = dim_map_size
        self.num_hidden_channels = num_hidden_channels

        # Calculate total number of options across all types
        self.num_option_total = dim_struct + dim_skill + dim_unit

        # --- Initial Map Convolution ---
        self.initial_conv = nn.Conv2d(dim_map_channels, num_hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_hidden_channels)
        self.initial_relu = nn.ReLU(inplace=True)

        # --- Residual Backbone ---
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_hidden_channels) for _ in range(num_res_blocks)]
        )

        # --- Player State Embedding ---
        self.player_fc1 = nn.Linear(dim_player, num_player_hidden)
        self.player_relu = nn.ReLU(inplace=True)
        self.player_fc2 = nn.Linear(num_player_hidden, num_hidden_channels) # Project to match spatial channels for fusion

        # --- Fusion Layer ---
        self.fusion_conv = nn.Conv2d(num_hidden_channels * 2, num_hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fusion_bn = nn.BatchNorm2d(num_hidden_channels)
        self.fusion_relu = nn.ReLU(inplace=True)

        # --- Shared Representation Processing ---
        self.post_fusion_resblock = ResidualBlock(num_hidden_channels)

        # --- Policy Heads ---
        # Common processing for non-spatial policies
        self.policy_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.policy_fc_shared = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.policy_fc_relu = nn.ReLU(inplace=True)

        # π_action: Softmax over action types
        self.pi_action_fc = nn.Linear(num_hidden_channels, num_action_types)

        # π_actor: Spatial head over map tiles (w,h)
        self.pi_actor_conv = nn.Conv2d(num_hidden_channels, 1, kernel_size=1, stride=1, padding=0)

        # π_target: Spatial head over target tiles (w,h)
        self.pi_target_conv = nn.Conv2d(num_hidden_channels, 1, kernel_size=1, stride=1, padding=0)

        # π_option_struct: Softmax over structure types
        self.pi_option_struct_fc = nn.Linear(num_hidden_channels, dim_struct)

        # π_option_skill: Softmax over skill types
        self.pi_option_skill_fc = nn.Linear(num_hidden_channels, dim_skill)

        # π_option_unit: Softmax over unit types
        self.pi_option_unit_fc = nn.Linear(num_hidden_channels, dim_unit)

        # π_tech: Softmax for technology types
        self.pi_tech_fc = nn.Linear(num_hidden_channels, dim_tech)

        # π_reward: Binary for city level-up rewards
        self.pi_reward_fc = nn.Linear(num_hidden_channels, 1)

        # --- Value Heads ---
        # Common processing for value heads
        self.value_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.value_fc_shared = nn.Linear(num_hidden_channels, num_hidden_channels)
        self.value_fc_relu = nn.ReLU(inplace=True)

        # v_win: Scalar in [-1,1] for win probability
        self.v_win_fc = nn.Linear(num_hidden_channels, 1)

        # v_econ: Predicted future economic score (normalized)
        self.v_econ_fc = nn.Linear(num_hidden_channels, 1)

        # v_mil: Predicted future military strength (normalized)
        self.v_mil_fc = nn.Linear(num_hidden_channels, 1)

    def forward(self, obs):
        map_input = obs['map'] # [B, C_map, H, W]
        player_input = obs['player'] # [B, C_player]
        batch_size = map_input.size(0)
        map_h, map_w = map_input.size(2), map_input.size(3)

        # 1. & 2. Process Map through CNN Backbone
        spatial_features = self.initial_conv(map_input)
        spatial_features = self.initial_bn(spatial_features)
        spatial_features = self.initial_relu(spatial_features)
        spatial_features = self.res_blocks(spatial_features)

        # 3. Process Player State
        player_embed = self.player_fc1(player_input)
        player_embed = self.player_relu(player_embed)
        player_embed = self.player_fc2(player_embed)

        # 4. Fuse Spatial and Player features
        player_broadcast = player_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, map_h, map_w)
        fused_features = torch.cat([spatial_features, player_broadcast], dim=1)
        fused_features = self.fusion_conv(fused_features)
        fused_features = self.fusion_bn(fused_features)
        fused_features = self.fusion_relu(fused_features)

        # [B, num_hidden, H, W] 
        shared_representation = self.post_fusion_resblock(fused_features) 

        # --- Policy Head Calculations ---
        # [B, num_hidden]
        policy_pooled = self.policy_pool(shared_representation).view(batch_size, -1) 
        policy_latent = self.policy_fc_shared(policy_pooled)
        # [B, num_hidden]
        policy_latent = self.policy_fc_relu(policy_latent) 

        # π_action logits (categorical)
        pi_action_logits = self.pi_action_fc(policy_latent)

        # π_actor logits (spatial)
        pi_actor_logits_map = self.pi_actor_conv(shared_representation)
        pi_actor_logits = pi_actor_logits_map.view(batch_size, -1) # [B, H * W]

        # π_target logits (spatial)
        pi_target_logits_map = self.pi_target_conv(shared_representation)
        pi_target_logits = pi_target_logits_map.view(batch_size, -1) # [B, H * W]

        # Separate π_option logits (categorical)
        pi_option_struct_logits = self.pi_option_struct_fc(policy_latent) # [B, dim_struct]
        pi_option_skill_logits = self.pi_option_skill_fc(policy_latent) # [B, dim_skill]
        pi_option_unit_logits = self.pi_option_unit_fc(policy_latent) # [B, dim_unit]

        # π_tech logits
        pi_tech_logits = self.pi_tech_fc(policy_latent) # [B, dim_tech]

        # π_reward_choice logits
        # Logit < 0 -> choose option 0 (first presented)
        # Logit > 0 -> choose option 1 (second presented)
        pi_reward_logits = self.pi_reward_fc(policy_latent) # [B, 1]

        # --- Value Head Calculations ---
        value_pooled = self.value_pool(shared_representation).view(batch_size, -1)
        value_latent = self.value_fc_shared(value_pooled)
        value_latent = self.value_fc_relu(value_latent)

        # v_win prediction
        v_win = torch.tanh(self.v_win_fc(value_latent))

        # v_econ prediction
        v_econ = self.v_econ_fc(value_latent)

        # v_mil prediction
        v_mil = self.v_mil_fc(value_latent)

        return {
            # Policy Logits
            'pi_action_logits': pi_action_logits,
            'pi_actor_logits': pi_actor_logits,
            'pi_target_logits': pi_target_logits,
            'pi_option_struct_logits': pi_option_struct_logits,
            'pi_option_skill_logits': pi_option_skill_logits,
            'pi_option_unit_logits': pi_option_unit_logits,
            'pi_tech_logits': pi_tech_logits,
            'pi_reward_logits': pi_reward_logits,

            # Value Predictions
            'v_win': v_win,
            'v_econ': v_econ,
            'v_mil': v_mil
        }

if __name__ == '__main__':
    # Dummy dimensions
    DIM_MAP_CHANNELS = 20
    DIM_MAP_SIZE = 15
    DIM_PLAYER = 50
    DIM_STRUCT = 15
    DIM_SKILL = 10
    DIM_UNIT = 30
    DIM_TECH = 35

    # Hyperparameters
    BATCH_SIZE = 4
    NUM_RES_BLOCKS = 12
    NUM_HIDDEN_CHANNELS = 128
    NUM_ACTION_TYPES = 9

    # Create dummy input
    dummy_obs = {
        'map': torch.randn(BATCH_SIZE, DIM_MAP_CHANNELS, DIM_MAP_SIZE, DIM_MAP_SIZE),
        'player': torch.randn(BATCH_SIZE, DIM_PLAYER)
    }

    # Instantiate the network
    model = PolytopiaNet(
        dim_map_channels=DIM_MAP_CHANNELS,
        dim_map_size=DIM_MAP_SIZE,
        dim_player=DIM_PLAYER,
        dim_struct=DIM_STRUCT,
        dim_skill=DIM_SKILL,
        dim_unit=DIM_UNIT,
        dim_tech=DIM_TECH,
        num_res_blocks=NUM_RES_BLOCKS,
        num_hidden_channels=NUM_HIDDEN_CHANNELS,
        num_action_types=NUM_ACTION_TYPES
    )

    # Perform a forward pass
    model.eval() # Set model to evaluation mode for consistent output if using dropout/batchnorm etc.
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(dummy_obs)

    # Print shapes of the outputs
    print("Output Shapes:")
    for key, value in output.items():
        print(f"{key}: {value.shape}")

    # Example: Get probabilities for action types (requires softmax)
    action_probs = F.softmax(output['pi_action_logits'], dim=-1)
    print("\nExample Action Type Probabilities (first batch element):\n", action_probs[0])

    # Example: Get probabilities for structure options (requires softmax)
    # Note: You would only apply softmax/use these if the chosen action type was 'build'
    struct_option_probs = F.softmax(output['pi_option_struct_logits'], dim=-1)
    print("\nExample Struct Option Probabilities (first batch element):\n", struct_option_probs[0])

    # Example: Interpret the reward choice logits
    reward_choice_logits = output['pi_reward_choice_logits']
    # Apply sigmoid to get probability of choosing the *second* option
    reward_choice_prob_option1 = torch.sigmoid(reward_choice_logits)
    # Probability of choosing the *first* option is 1 - prob(option1)
    reward_choice_prob_option0 = 1.0 - reward_choice_prob_option1

    print(f"\nExample Reward Choice Logits (batch):\n{reward_choice_logits.squeeze(-1)}")
    print(f"\nExample Reward Choice Prob[Choose Option 1] (batch):\n{reward_choice_prob_option1.squeeze(-1)}")
    print(f"\nExample Reward Choice Prob[Choose Option 0] (batch):\n{reward_choice_prob_option0.squeeze(-1)}")

    # Decision rule based on logits:
    # choose_option_1 = reward_choice_logits > 0
    # choose_option_0 = reward_choice_logits <= 0
    print(f"\nDecision (Choose Option 1?): {reward_choice_logits.squeeze(-1) > 0}")