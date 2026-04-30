"""
nemesis.models.encoder
======================
Wavelet-domain JEPA encoder for GNSS IQ scalograms.

Architecture
------------
A lightweight convolutional encoder that maps a 2-channel scalogram
(I and Q channels) to a fixed-size embedding vector. Pretrained via
Joint Embedding Predictive Architecture (JEPA) on clear-sky GPS data.

Input  : (B, 2, scales, time) float32 scalogram
Output : (B, embed_dim)       float32 embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletJEPAEncoder(nn.Module):
    """
    Convolutional JEPA encoder for GPS L1 C/A scalograms.

    Parameters
    ----------
    input_channels : int
        Number of scalogram channels (2 for I+Q). Default 2.
    embed_dim : int
        Output embedding dimensionality. Default 256.
    base_filters : int
        Base number of convolutional filters. Default 32.
    """

    def __init__(
        self,
        input_channels: int = 2,
        embed_dim: int = 256,
        base_filters: int = 32,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        # Convolutional backbone
        self.backbone = nn.Sequential(
            _ConvBNReLU(input_channels, base_filters, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            _ConvBNReLU(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            _ConvBNReLU(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.MaxPool2d(2),

            _ConvBNReLU(base_filters * 4, base_filters * 8, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Projection head
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
        x : torch.Tensor, shape (B, 2, scales, time)

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
        """
        features = self.backbone(x)
        features = features.flatten(1)
        return self.projector(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for ``forward``."""
        return self.forward(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, **kwargs):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
