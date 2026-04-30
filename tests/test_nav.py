"""Tests for nemesis.nav module."""

import numpy as np
import pytest


def test_ekf_predict_increases_uncertainty():
    from nemesis.nav.ekf import NEMESISNavEKF
    ekf = NEMESISNavEKF()
    p_before = np.trace(ekf.P)
    ekf.predict(dt=1.0)
    p_after = np.trace(ekf.P)
    assert p_after > p_before


def test_ekf_update_returns_position():
    from nemesis.nav.ekf import NEMESISNavEKF
    ekf = NEMESISNavEKF()
    ekf.predict()
    pos, inflation = ekf.update(np.array([10.0, 5.0]), spoof_confidence=0.0)
    assert pos.shape == (2,)
    assert inflation >= 1.0


def test_ekf_inflation_increases_with_spoofing():
    from nemesis.nav.ekf import NEMESISNavEKF
    ekf = NEMESISNavEKF(beta=12.0, tau_s=0.1)
    ekf.predict()
    _, low = ekf.update(np.array([0.0, 0.0]), spoof_confidence=0.0)
    ekf2 = NEMESISNavEKF(beta=12.0, tau_s=0.1)
    ekf2.predict()
    _, high = ekf2.update(np.array([0.0, 0.0]), spoof_confidence=1.0)
    assert high > low


def test_ekf_reset_clears_state():
    from nemesis.nav.ekf import NEMESISNavEKF
    ekf = NEMESISNavEKF()
    ekf.predict()
    ekf.update(np.array([100.0, 200.0]), spoof_confidence=0.9)
    ekf.reset()
    assert np.allclose(ekf.x, 0.0)
    assert len(ekf.history) == 0


def test_ekf_history_records_steps():
    from nemesis.nav.ekf import NEMESISNavEKF
    ekf = NEMESISNavEKF()
    for _ in range(5):
        ekf.predict()
        ekf.update(np.random.randn(2), spoof_confidence=0.3)
    assert len(ekf.history) == 5
