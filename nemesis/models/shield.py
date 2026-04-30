"""
nemesis.models.shield
=====================
NEMESIS-Shield (NEMESISv2) — multichannel scalogram encoder with
Squeeze-and-Excitation (SE) channel attention.

Requires: ``pip install nemesis-gnss[shield]``

Architecture
------------
Extends the base JEPA encoder with:
- Multi-channel scalogram input (configurable number of channels)
- SE block after each convolutional stage for adaptive channel weighting
- Shared projection head compatible with the base MLPProbe

Input  : (B, C, scales, time)  C = number of scalogram channels
Output : (B, embed_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        scale = self.fc(s).view(b, c, 1, 1)
        return x * scale


class NEMESISShield(nn.Module):
    """
    NEMESIS-Shield encoder with SE channel attention.

    Parameters
    ----------
    input_channels : int
        Number of scalogram channels. Default 6.
    embed_dim : int
        Output embedding dimensionality. Default 256.
    base_filters : int
        Base number of convolutional filters. Default 32.
    se_reduction : int
        SE squeeze ratio. Default 16.
    """

    def __init__(
        self,
        input_channels: int = 6,
        embed_dim: int = 256,
        base_filters: int = 32,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        def stage(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                SEBlock(out_ch, reduction=se_reduction),
                nn.MaxPool2d(2),
            )

        self.backbone = nn.Sequential(
            stage(input_channels, base_filters),
            stage(base_filters, base_filters * 2),
            stage(base_filters * 2, base_filters * 4),
            stage(base_filters * 4, base_filters * 8),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        backbone_out = base_filters * 8 * 4 * 4
        self.projector = nn.Sequential(
            nn.Linear(backbone_out, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C, scales, time)

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
        """
        features = self.backbone(x)
        features = features.flatten(1)
        return self.projector(features)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
