"""
nemesis.eval.metrics
====================
Evaluation metrics for NEMESIS spoofing detection.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from rich.console import Console
from rich.table import Table

console = Console()

CLASS_NAMES = ["Clear Sky", "Meaconing", "Slow Drift", "Adversarial"]


def evaluate_detector(
    detector,
    data_path: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a NEMESISDetector on a dataset directory.

    Parameters
    ----------
    detector : NEMESISDetector
        Trained detector instance.
    data_path : str or Path
        Root data directory (same structure as training).
    verbose : bool
        Print a rich results table. Default True.

    Returns
    -------
    dict with keys:
        - ``accuracy``
        - ``f1_macro``
        - ``auc_ovr``
        - ``confusion_matrix``
        - ``per_class`` (dict of per-class metrics)
        - ``y_true``, ``y_pred``, ``y_proba``
    """
    from pathlib import Path
    from nemesis.data.dataset import IQDataset
    from torch.utils.data import DataLoader
    import torch

    dataset = IQDataset(data_path, verbose=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    y_true, y_pred, y_proba = [], [], []

    detector.encoder.eval()
    detector.probe.eval()

    import torch
    device = detector.device

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            z = detector.encoder(batch_x).cpu().numpy()
            z_scaled = detector.scaler.transform(z)
            z_tensor = torch.tensor(z_scaled, dtype=torch.float32).to(device)
            logits = detector.probe(z_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=-1)
            y_true.extend(batch_y.numpy().tolist())
            y_pred.extend(preds.tolist())
            y_proba.extend(probs.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_proba = np.array(y_proba)

    num_classes = y_proba.shape[1]
    present = np.unique(y_true)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    try:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        )
    except Exception:
        auc = float("nan")

    results = {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "auc_ovr": float(auc),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

    if verbose:
        _print_results(results, cm, num_classes)

    return results


def classification_report_nemesis(y_true, y_pred) -> str:
    """Return a sklearn classification report with NEMESIS class names."""
    present = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    names = [CLASS_NAMES[i] for i in present]
    return classification_report(y_true, y_pred, labels=present, target_names=names, zero_division=0)


def _print_results(results: dict, cm: np.ndarray, num_classes: int) -> None:
    table = Table(title="NEMESIS Evaluation Results", header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Accuracy",  f"{results['accuracy'] * 100:.2f}%")
    table.add_row("F1 (macro)", f"{results['f1_macro']:.4f}")
    table.add_row("AUC (OvR)",  f"{results['auc_ovr']:.4f}")
    console.print(table)

    cm_table = Table(title="Confusion Matrix", header_style="bold magenta")
    cm_table.add_column("True \\ Pred")
    for name in CLASS_NAMES[:num_classes]:
        cm_table.add_column(name[:10], justify="right")
    for i, row in enumerate(cm):
        cm_table.add_row(CLASS_NAMES[i][:12], *[str(v) for v in row])
    console.print(cm_table)
