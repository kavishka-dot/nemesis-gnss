"""nemesis.train.callbacks — Early stopping and checkpoint helpers."""

from __future__ import annotations


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


class CheckpointCallback:
    """Save best model state dict when validation loss improves."""

    def __init__(self):
        self._best = float("inf")
        self.best_state = None

    def __call__(self, val_loss: float, state_dict: dict) -> bool:
        if val_loss < self._best:
            self._best = val_loss
            self.best_state = {k: v.clone() for k, v in state_dict.items()}
            return True
        return False
