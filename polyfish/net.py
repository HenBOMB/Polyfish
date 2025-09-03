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

        # # v_eco: Predicted future economic score (normalized)
        # self.v_eco_fc = nn.Linear(num_hidden_channels, 1)

        # # v_mil: Predicted future military strength (normalized)
        # self.v_mil_fc = nn.Linear(num_hidden_channels, 1)

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

        # --- Policy Head ---
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

        # --- Value Head ---
        value_pooled = self.value_pool(shared_representation).view(batch_size, -1)
        value_latent = self.value_fc_shared(value_pooled)
        value_latent = self.value_fc_relu(value_latent)

        v_win = torch.tanh(self.v_win_fc(value_latent))

        # v_eco = self.v_eco_fc(value_latent)

        # v_mil = self.v_mil_fc(value_latent)

        return {
            'pi_action': pi_action_logits,
            'pi_source': pi_actor_logits,
            'pi_target': pi_target_logits,
            'pi_struct': pi_option_struct_logits,
            'pi_skill': pi_option_skill_logits,
            'pi_unit': pi_option_unit_logits,
            'pi_tech': pi_tech_logits,
            'pi_reward': pi_reward_logits,
            'v_win': v_win,
            # 'v_eco': v_eco,
            # 'v_mil': v_mil
        }