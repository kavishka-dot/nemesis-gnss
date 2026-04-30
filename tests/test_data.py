"""Tests for nemesis.data module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nemesis.data.loader import load_iq_file, _to_iq_array
from nemesis.data.transforms import WaveletTransform
from nemesis.data.dataset import IQDataset


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def iq_npy_file(tmp_path):
    """Create a synthetic .npy IQ file."""
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
    path = tmp_path / "sample.npy"
    np.save(path, iq)
    return path


@pytest.fixture
def iq_bin_file(tmp_path):
    """Create a synthetic interleaved float32 .bin IQ file."""
    iq_c = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
    interleaved = np.empty(4096 * 2, dtype=np.float32)
    interleaved[0::2] = iq_c.real
    interleaved[1::2] = iq_c.imag
    path = tmp_path / "sample.bin"
    interleaved.tofile(path)
    return path


@pytest.fixture
def dataset_dir(tmp_path):
    """Create a minimal dataset directory with synthetic IQ files."""
    for folder in ["clear_sky", "spoofed/meaconing", "spoofed/slow_drift", "spoofed/adversarial"]:
        d = tmp_path / folder
        d.mkdir(parents=True)
        for i in range(3):
            iq = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
            np.save(d / f"sample_{i:03d}.npy", iq)
    return tmp_path


# ------------------------------------------------------------------
# loader tests
# ------------------------------------------------------------------

def test_load_npy_returns_correct_shape(iq_npy_file):
    iq = load_iq_file(iq_npy_file, segment_length=4096)
    assert iq.shape == (2, 4096)
    assert iq.dtype == np.float32


def test_load_bin_returns_correct_shape(iq_bin_file):
    iq = load_iq_file(iq_bin_file, segment_length=4096)
    assert iq.shape == (2, 4096)
    assert iq.dtype == np.float32


def test_load_nonexistent_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_iq_file(tmp_path / "ghost.bin")


def test_load_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "file.xyz"
    bad.write_bytes(b"\x00" * 100)
    with pytest.raises(ValueError, match="Unsupported"):
        load_iq_file(bad)


def test_normalisation_unit_power(iq_npy_file):
    iq = load_iq_file(iq_npy_file, segment_length=4096)
    power = np.mean(iq[0] ** 2 + iq[1] ** 2)
    assert abs(power - 1.0) < 0.05


# ------------------------------------------------------------------
# transform tests
# ------------------------------------------------------------------

def test_wavelet_transform_output_shape(iq_npy_file):
    iq = load_iq_file(iq_npy_file, segment_length=4096)
    transform = WaveletTransform(scales=32)
    out = transform(iq)
    assert out.shape == (2, 32, 4096)
    assert out.dtype == np.float32


def test_wavelet_transform_values_in_range(iq_npy_file):
    iq = load_iq_file(iq_npy_file, segment_length=4096)
    transform = WaveletTransform(scales=16)
    out = transform(iq)
    assert out.min() >= 0.0
    assert out.max() <= 1.0 + 1e-6


# ------------------------------------------------------------------
# dataset tests
# ------------------------------------------------------------------

def test_dataset_discovers_all_classes(dataset_dir):
    ds = IQDataset(dataset_dir, verbose=False)
    labels = set(ds.labels)
    assert 0 in labels   # clear sky
    assert len(labels) > 1   # at least one spoofed class


def test_dataset_len(dataset_dir):
    ds = IQDataset(dataset_dir, verbose=False)
    assert len(ds) == 12   # 3 files x 4 classes


def test_dataset_getitem_shape(dataset_dir):
    ds = IQDataset(dataset_dir, verbose=False, segment_length=256)
    sample, label = ds[0]
    assert sample.ndim == 3   # (2, scales, time)
    assert isinstance(label, int)


def test_dataset_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        IQDataset("/nonexistent/path", verbose=False)


def test_dataset_class_counts(dataset_dir):
    ds = IQDataset(dataset_dir, verbose=False)
    counts = ds.class_counts
    assert counts["Clear Sky"] == 3
