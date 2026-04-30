"""
nemesis.train.losses
====================
Focal loss for imbalanced GNSS spoofing classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Reduces loss contribution from easy examples and focuses on hard ones.
    Particularly effective when clear-sky samples dominate the dataset.

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 recovers standard cross-entropy. Default 2.0.
    alpha : torch.Tensor or None
        Per-class weight tensor of shape (num_classes,). Useful for
        further balancing rare attack classes. Default None (uniform).
    reduction : str
        ``"mean"`` or ``"sum"``. Default ``"mean"``.

    References
    ----------
    Lin et al. (2017) "Focal Loss for Dense Object Detection." ICCV.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor, shape (B, C)
        targets : torch.Tensor, shape (B,) — integer class labels

        Returns
        -------
        torch.Tensor — scalar loss
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather per-sample log-probs and probs for true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = -focal_weight * log_pt

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
