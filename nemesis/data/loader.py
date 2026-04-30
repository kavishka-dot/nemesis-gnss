"""
nemesis.data.loader
===================
Low-level IQ file reading utilities.

Supports raw interleaved float32 (.bin, .dat, .iq, .cf32),
numpy (.npy), and numpy archives (.npz).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


SUPPORTED_EXTENSIONS = {".bin", ".dat", ".iq", ".cf32", ".npy", ".npz"}


def load_iq_file(
    path: Path | str,
    segment_length: int = 4096,
    offset: int = 0,
) -> np.ndarray:
    """
    Load IQ samples from a file and return as a float32 array of shape (2, N).

    Row 0 = I (real), Row 1 = Q (imaginary).

    Parameters
    ----------
    path : Path or str
        Path to the IQ file.
    segment_length : int
        Number of complex samples to read. Pads with zeros if file is shorter.
    offset : int
        Sample offset into the file (for windowing long captures).

    Returns
    -------
    np.ndarray of shape (2, segment_length), dtype float32.

    Raises
    ------
    ValueError
        If the file format is not recognised.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"IQ file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in {".bin", ".dat", ".iq", ".cf32"}:
        samples = _load_raw_float32(path, segment_length, offset)
    elif suffix == ".npy":
        samples = _load_npy(path, segment_length, offset)
    elif suffix == ".npz":
        samples = _load_npz(path, segment_length, offset)
    else:
        raise ValueError(
            f"Unsupported IQ file format: {suffix!r}. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    return _to_iq_array(samples, segment_length)


# ------------------------------------------------------------------
# Format-specific readers
# ------------------------------------------------------------------

def _load_raw_float32(path: Path, n: int, offset: int) -> np.ndarray:
    """Raw interleaved float32: [I0, Q0, I1, Q1, ...]"""
    byte_offset = offset * 2 * 4  # 2 floats per sample, 4 bytes each
    count = n * 2
    data = np.fromfile(path, dtype=np.float32, count=count, offset=byte_offset)
    return data


def _load_npy(path: Path, n: int, offset: int) -> np.ndarray:
    arr = np.load(path)
    if np.iscomplexobj(arr):
        arr = arr[offset: offset + n]
        return np.stack([arr.real, arr.imag], axis=0).flatten(order="F")
    else:
        # Assume shape (2, N) or (N, 2) or (2N,) interleaved
        arr = arr.astype(np.float32).ravel()
        return arr[offset * 2: (offset + n) * 2]


def _load_npz(path: Path, n: int, offset: int) -> np.ndarray:
    archive = np.load(path)
    key = "iq" if "iq" in archive else list(archive.keys())[0]
    arr = archive[key]
    if np.iscomplexobj(arr):
        arr = arr[offset: offset + n]
        return np.stack([arr.real, arr.imag], axis=0).flatten(order="F")
    return arr.astype(np.float32).ravel()[offset * 2: (offset + n) * 2]


# ------------------------------------------------------------------
# Common normalisation
# ------------------------------------------------------------------

def _to_iq_array(flat: np.ndarray, n: int) -> np.ndarray:
    """Convert flat interleaved float32 to (2, N) array. Pad/truncate as needed."""
    flat = flat.astype(np.float32)
    need = n * 2
    if len(flat) < need:
        flat = np.pad(flat, (0, need - len(flat)))
    flat = flat[:need]
    iq = flat.reshape(2, n, order="F")
    # Normalise to unit power
    power = np.mean(iq[0] ** 2 + iq[1] ** 2)
    if power > 0:
        iq = iq / np.sqrt(power)
    return iq
