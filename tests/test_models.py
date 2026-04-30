"""Tests for nemesis.models module."""

import torch
import pytest

from nemesis.models.encoder import WaveletJEPAEncoder
from nemesis.models.probe import MLPProbe


def test_encoder_forward_shape():
    encoder = WaveletJEPAEncoder(input_channels=2, embed_dim=256)
    x = torch.randn(4, 2, 64, 4096)
    out = encoder(x)
    assert out.shape == (4, 256)


def test_encoder_different_embed_dims():
    for dim in [64, 128, 512]:
        enc = WaveletJEPAEncoder(embed_dim=dim)
        x = torch.randn(2, 2, 64, 512)
        assert enc(x).shape == (2, dim)


def test_encoder_num_parameters():
    enc = WaveletJEPAEncoder()
    assert enc.num_parameters > 0


def test_probe_forward_shape():
    probe = MLPProbe(input_dim=256, num_classes=4)
    x = torch.randn(8, 256)
    out = probe(x)
    assert out.shape == (8, 4)


def test_probe_different_classes():
    for n in [2, 4]:
        probe = MLPProbe(input_dim=64, num_classes=n)
        x = torch.randn(3, 64)
        assert probe(x).shape == (3, n)


def test_encoder_eval_mode_no_grad():
    encoder = WaveletJEPAEncoder()
    encoder.eval()
    x = torch.randn(1, 2, 64, 256)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape[-1] == 256


def test_shield_import():
    try:
        from nemesis.models.shield import NEMESISShield, SEBlock
        model = NEMESISShield(input_channels=4, embed_dim=128)
        x = torch.randn(2, 4, 32, 256)
        out = model(x)
        assert out.shape == (2, 128)
    except ImportError:
        pytest.skip("shield extra not installed")
