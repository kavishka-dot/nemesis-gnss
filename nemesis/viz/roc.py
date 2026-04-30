"""nemesis.viz.roc — ROC and PR curve plots."""

from __future__ import annotations

from typing import Optional, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["Clear Sky", "Meaconing", "Slow Drift", "Adversarial"]
COLORS = ["#00d4ff", "#ff6b6b", "#ffd93d", "#c77dff"]


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot per-class ROC curves (One-vs-Rest).

    Parameters
    ----------
    y_true : np.ndarray, shape (N,)
    y_proba : np.ndarray, shape (N, num_classes)
    save_path : str or Path, optional
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_classes = y_proba.shape[1]
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")

    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", color="#7eb8f7")
    ax.set_ylabel("True Positive Rate", color="#7eb8f7")
    ax.set_title("ROC Curves — NEMESIS", color="#00d4ff", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#111827", edgecolor="#1e3a5f", labelcolor="#c8d6ef")
    ax.tick_params(colors="#7eb8f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    if show:
        plt.show()
    return fig
