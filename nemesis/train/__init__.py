"""nemesis.train — Training loops, losses, and callbacks."""

from nemesis.train.trainer import Trainer
from nemesis.train.losses import FocalLoss

__all__ = ["Trainer", "FocalLoss"]
