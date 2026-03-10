"""
HexNet — AlphaZero-style ResNet for Hex.

Architecture:
  Input: (B, C, H, W)  — C feature planes (default 8)
  Stem:  conv 3×3, 128 filters, BN, ReLU
  Tower: 8 × ResBlock(128 filters, 3×3)
  Policy head: conv 1×1 → (B, H*W) log-probabilities
  Value head:  global avg pool → concat(pool, size_scalar) → FC(256) → ReLU → FC(1) → tanh

Board-size conditioning: the value head receives the board size as a
normalised scalar appended to the pooled feature vector, so a single
checkpoint works across all board sizes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import HexZeroConfig


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class HexNet(nn.Module):
    def __init__(self, cfg: HexZeroConfig):
        super().__init__()
        F_ = cfg.num_filters
        C  = cfg.num_input_planes

        # Stem
        self.stem_conv = nn.Conv2d(C, F_, 3, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(F_)

        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(F_) for _ in range(cfg.num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(F_, 1, 1, bias=True)

        # Value head
        # After global avg pool: vector of size F_ + 1 (size_scalar appended)
        self.value_fc1 = nn.Linear(F_ + 1, cfg.value_fc_hidden)
        self.value_fc2 = nn.Linear(cfg.value_fc_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,          # (B, C, H, W)
        size_scalar: torch.Tensor, # (B, 1)  normalised board size
    ):
        # Stem
        x = F.relu(self.stem_bn(self.stem_conv(x)))

        # Tower
        x = self.tower(x)                           # (B, F, H, W)

        # Policy head
        policy_logits = self.policy_conv(x)         # (B, 1, H, W)
        B, _, H, W = policy_logits.shape
        policy_logits = policy_logits.view(B, H * W) # (B, H*W)
        log_policy = F.log_softmax(policy_logits, dim=1)

        # Value head
        pooled = x.mean(dim=[2, 3])                 # (B, F)  global avg pool
        v_input = torch.cat([pooled, size_scalar], dim=1)  # (B, F+1)
        v = F.relu(self.value_fc1(v_input))
        value = torch.tanh(self.value_fc2(v))       # (B, 1)

        return log_policy, value.squeeze(1)         # (B, H*W), (B,)


def build_net(cfg: HexZeroConfig, device: torch.device = None) -> HexNet:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = HexNet(cfg).to(device)
    return net
