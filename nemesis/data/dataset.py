"""
nemesis.data.dataset
====================
IQDataset: auto-discovers IQ files from a user-provided data directory.

Expected directory layout
--------------------------
::

    my_dataset/
    ├── clear_sky/          # label 0 — nominal GPS signal
    │   ├── sample_001.bin
    │   ├── sample_002.npy
    │   └── ...
    └── spoofed/            # labelled by sub-folder
        ├── meaconing/      # label 1
        │   └── ...
        ├── slow_drift/     # label 2
        │   └── ...
        └── adversarial/    # label 3
            └── ...

Alternatively, a flat spoofed/ folder (all files treated as a generic
"spoofed" class) is also supported. NEMESIS will assign label 1 to all
files in that case.

Supported IQ file formats
--------------------------
- ``.bin``  / ``.dat``  / ``.iq``  / ``.cf32`` — raw interleaved float32
- ``.npy``  — numpy array, shape (N,) complex64 or (2, N) float32
- ``.npz``  — numpy archive; key ``"iq"`` is used
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
from rich.console import Console
from rich.table import Table

from nemesis.data.loader import load_iq_file
from nemesis.data.transforms import WaveletTransform

console = Console()

LABEL_MAP: Dict[str, int] = {
    "clear_sky": 0,
    "meaconing": 1,
    "slow_drift": 2,
    "adversarial": 3,
    "spoofed": 1,   # flat spoofed/ folder fallback
}

CLASS_NAMES = {0: "Clear Sky", 1: "Meaconing", 2: "Slow Drift", 3: "Adversarial"}


class IQDataset(Dataset):
    """
    PyTorch Dataset for GNSS IQ data.

    Automatically discovers files under ``data_path`` and assigns labels
    based on subdirectory names.

    Parameters
    ----------
    data_path : str or Path
        Root directory containing ``clear_sky/`` and ``spoofed/``
        (or nested attack-class sub-folders).
    transform : WaveletTransform, optional
        Preprocessing transform to apply. Defaults to a standard
        64-scale Morlet wavelet transform.
    segment_length : int
        Number of complex IQ samples per segment. Default 4096.
    verbose : bool
        Print a summary table on construction. Default True.

    Examples
    --------
    >>> dataset = IQDataset("./my_dataset")
    >>> sample, label = dataset[0]
    >>> print(sample.shape, label)
    """

    SUPPORTED_EXTENSIONS = {".bin", ".dat", ".iq", ".cf32", ".npy", ".npz"}

    def __init__(
        self,
        data_path: str | Path,
        transform: Optional[WaveletTransform] = None,
        segment_length: int = 4096,
        verbose: bool = True,
    ):
        self.data_path = Path(data_path)
        self.segment_length = segment_length
        self.transform = transform or WaveletTransform()

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_path}\n\n"
                "Please create the following structure:\n"
                "  my_dataset/\n"
                "  ├── clear_sky/     <- nominal GPS IQ files\n"
                "  └── spoofed/       <- spoofed IQ files\n"
                "       ├── meaconing/\n"
                "       ├── slow_drift/\n"
                "       └── adversarial/"
            )

        self.samples: List[Tuple[Path, int]] = self._discover(self.data_path)

        if len(self.samples) == 0:
            raise ValueError(
                f"No IQ files found in {self.data_path}.\n"
                f"Supported formats: {self.SUPPORTED_EXTENSIONS}"
            )

        if verbose:
            self._print_summary()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        iq = load_iq_file(path, segment_length=self.segment_length)
        scalogram = self.transform(iq)
        tensor = torch.tensor(scalogram, dtype=torch.float32)
        return tensor, label

    # ------------------------------------------------------------------
    # Class distribution helpers
    # ------------------------------------------------------------------

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name in CLASS_NAMES.values()}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self, root: Path) -> List[Tuple[Path, int]]:
        samples = []
        for subdir in sorted(root.iterdir()):
            if not subdir.is_dir():
                continue
            key = subdir.name.lower().replace(" ", "_").replace("-", "_")
            if key in LABEL_MAP:
                label = LABEL_MAP[key]
                files = self._find_files(subdir)
                samples.extend((f, label) for f in files)
            else:
                # Check one level deeper (e.g. spoofed/meaconing/)
                for nested in sorted(subdir.iterdir()):
                    if not nested.is_dir():
                        continue
                    nested_key = nested.name.lower().replace(" ", "_").replace("-", "_")
                    if nested_key in LABEL_MAP:
                        label = LABEL_MAP[nested_key]
                        files = self._find_files(nested)
                        samples.extend((f, label) for f in files)
        return samples

    def _find_files(self, directory: Path) -> List[Path]:
        return sorted(
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

    def _print_summary(self) -> None:
        table = Table(title="IQDataset Summary", show_header=True, header_style="bold cyan")
        table.add_column("Class", style="white")
        table.add_column("Count", justify="right")
        table.add_column("Label ID", justify="center")

        counts = self.class_counts
        for label_id, name in CLASS_NAMES.items():
            count = counts.get(name, 0)
            style = "green" if count > 0 else "dim"
            table.add_row(name, str(count), str(label_id), style=style)

        table.add_row("TOTAL", str(len(self.samples)), "", style="bold")
        console.print(table)
        console.print(f"[dim]Data root:[/dim] {self.data_path}")
