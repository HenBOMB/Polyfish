import torch.nn as nn
import torch
import torch.nn.functional as F
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)
    
class PolytopiaZeroNet(nn.Module):
    def __init__(self, n_res_blocks=10, config: dict = {}):
        super().__init__()
        C = config['tile_channels']
        WH = config['tile_count']
        A = config['max_actions']
        tech_dim = config['max_tech'] + config['settings_channels']

        # shared Tower
        self.conv0 = nn.Conv2d(C, 128, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResidualBlock(128,128) for _ in range(n_res_blocks)])

        # policy head
        self.conv_pi = nn.Conv2d(128, 2, kernel_size=1)
        self.bn_pi = nn.BatchNorm2d(2)
        self.fc_pi = nn.Linear(2*WH + tech_dim, A)

        # value head
        self.conv_v = nn.Conv2d(128, 1, kernel_size=1)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(1*WH + tech_dim, 256)
        self.fc_v2 = nn.Linear(256, 1)

    def forward(self, obs):
        x = obs['map'].float()    # [B,C,H,W]
        t = obs['player'].float() # [B,tech_dim]

        # shared
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.res_blocks(x)

        # policy
        p = F.relu(self.bn_pi(self.conv_pi(x)))
        p = p.view(p.size(0), -1)
        p = torch.cat([p, t], dim=1)
        logits = self.fc_pi(p)

        # value
        v = F.relu(self.bn_v(self.conv_v(x)))
        v = v.view(v.size(0), -1)
        v = torch.cat([v, t], dim=1)
        v = F.relu(self.fc_v1(v))
        value = torch.tanh(self.fc_v2(v)).squeeze(-1)

        return logits, value