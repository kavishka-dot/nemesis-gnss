"""
nemesis.viz.pareto
==================
Pareto front visualization for NEMESIS-Nav beta vs tau_s tradeoff.
Requires nemesis-gnss[nav].
"""

from __future__ import annotations

from typing import Optional, Union, List
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_pareto_front(
    beta_values: List[float],
    tau_values: List[float],
    metric_a: List[float],
    metric_b: List[float],
    metric_a_label: str = "Detection Rate",
    metric_b_label: str = "HPL Exceedance",
    hpl_threshold: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a 2D Pareto front of beta vs tau_s with two competing metrics.

    Parameters
    ----------
    beta_values : list of float
        Noise inflation beta values tested.
    tau_values : list of float
        Detection threshold tau_s values tested.
    metric_a : list of float
        Primary metric (higher = better), e.g. detection rate.
    metric_b : list of float
        Secondary metric (lower = better), e.g. HPL exceedance rate.
    hpl_threshold : float, optional
        Draw a horizontal admissibility line at this HPL level.
    save_path : str or Path, optional
    show : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    metric_a = np.array(metric_a)
    metric_b = np.array(metric_b)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0a0e1a")
    ax.set_facecolor("#0a0e1a")

    # Scatter coloured by beta
    scatter = ax.scatter(
        metric_b, metric_a,
        c=beta_values, cmap="plasma",
        s=80, edgecolors="#1e3a5f", linewidths=0.5, zorder=3,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Beta", color="#7eb8f7")
    cbar.ax.yaxis.set_tick_params(color="#7eb8f7")

    # Annotate tau_s
    for i, (x, y, t) in enumerate(zip(metric_b, metric_a, tau_values)):
        ax.annotate(
            f"τ={t:.2f}", (x, y),
            textcoords="offset points", xytext=(5, 4),
            fontsize=7, color="#7eb8f7", alpha=0.75,
        )

    if hpl_threshold is not None:
        ax.axvline(x=hpl_threshold, color="#ff6b6b", lw=1.5, ls="--",
                   label=f"HPL threshold = {hpl_threshold:.2f}")
        ax.legend(facecolor="#111827", edgecolor="#1e3a5f", labelcolor="#c8d6ef")

    ax.set_xlabel(metric_b_label, color="#7eb8f7")
    ax.set_ylabel(metric_a_label, color="#7eb8f7")
    ax.set_title("NEMESIS-Nav: Pareto Front (β vs τs)", color="#00d4ff",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="#7eb8f7")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e3a5f")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    if show:
        plt.show()
    return fig
