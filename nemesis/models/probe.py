"""
nemesis.models.probe
====================
MLP probe for spoofing classification on top of JEPA embeddings.

This is a shallow MLP trained with focal loss on frozen (or fine-tuned)
encoder features. Classifies into 4 classes:
  0 = Clear Sky
  1 = Meaconing
  2 = Slow Drift
  3 = Adversarial
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPProbe(nn.Module):
    """
    Shallow MLP classification probe.

    Parameters
    ----------
    input_dim : int
        Dimensionality of encoder embeddings. Default 256.
    hidden_dim : int
        Hidden layer width. Default 128.
    num_classes : int
        Number of output classes. Default 4.
    dropout : float
        Dropout probability. Default 0.3.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, input_dim)

        Returns
        -------
        torch.Tensor, shape (B, num_classes) — raw logits
        """
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
