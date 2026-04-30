"""
nemesis.viz.scalogram
=====================
Scalogram (wavelet time-frequency) visualization for IQ data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from nemesis.data.loader import load_iq_file
from nemesis.data.transforms import WaveletTransform

CLASS_COLORS = {
    "Clear Sky": "#00d4ff",
    "Meaconing": "#ff6b6b",
    "Slow Drift": "#ffd93d",
    "Adversarial": "#c77dff",
}


def plot_scalogram(
    iq_path: Union[str, Path],
    label: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    segment_length: int = 4096,
    scales: int = 64,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the wavelet scalogram (I and Q channels) of an IQ file.

    Parameters
    ----------
    iq_path : str or Path
        Path to an IQ file.
    label : str, optional
        Class label to display in the title (e.g. "Meaconing").
    save_path : str or Path, optional
        If provided, save the figure here.
    segment_length : int
        Number of samples to visualize. Default 4096.
    scales : int
        Number of wavelet scales. Default 64.
    show : bool
        Call ``plt.show()``. Default True.

    Returns
    -------
    matplotlib.figure.Figure
    """
    iq = load_iq_file(iq_path, segment_length=segment_length)
    transform = WaveletTransform(scales=scales)
    scalogram = transform(iq)   # (2, scales, time)

    accent = CLASS_COLORS.get(label, "#00d4ff") if label else "#00d4ff"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0a0e1a")
    fig.subplots_adjust(wspace=0.35)

    for ax, ch, name in zip(axes, [0, 1], ["I channel", "Q channel"]):
        ax.set_facecolor("#0a0e1a")
        im = ax.imshow(
            scalogram[ch],
            aspect="auto",
            origin="lower",
            cmap="plasma",
            interpolation="nearest",
        )
        ax.set_xlabel("Time (samples)", color="#7eb8f7", fontsize=10)
        ax.set_ylabel("Scale index", color="#7eb8f7", fontsize=10)
        ax.set_title(name, color=accent, fontsize=12, fontweight="bold")
        ax.tick_params(colors="#7eb8f7")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#7eb8f7")

    title = f"GNSS IQ Scalogram"
    if label:
        title += f" — {label}"
    fig.suptitle(title, color=accent, fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0a0e1a")
    if show:
        plt.show()
    return fig
