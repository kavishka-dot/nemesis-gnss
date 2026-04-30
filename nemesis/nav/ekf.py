"""
nemesis.nav.ekf
===============
2-state position-only Extended Kalman Filter for NEMESIS-Nav.
Requires: pip install nemesis-gnss[nav]

The EKF is coupled to the NEMESIS detector via soft measurement noise
inflation: when spoofing is detected with confidence p, the measurement
noise covariance R is scaled by (1 + beta * p), degrading the influence
of the spoofed measurement gracefully rather than hard-switching.

Parameters
----------
beta : float
    Noise inflation factor. Default 12. Controls aggressiveness of
    measurement rejection under spoofing. See NEMESIS-Nav paper.
tau_s : float
    Detection confidence threshold (seconds equivalent). Default 0.5.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


class NEMESISNavEKF:
    """
    2-state position-only EKF with soft NEMESIS-coupled noise inflation.

    State vector: [x, y]  (2D position in metres)
    Measurement:  [x_m, y_m] from GPS pseudorange

    Parameters
    ----------
    beta : float
        Noise inflation gain. Default 12.
    tau_s : float
        Smoothing time constant for spoofing confidence. Default 0.5.
    process_noise : float
        Process noise standard deviation (m). Default 1.0.
    meas_noise : float
        Baseline measurement noise standard deviation (m). Default 5.0.
    """

    def __init__(
        self,
        beta: float = 12.0,
        tau_s: float = 0.5,
        process_noise: float = 1.0,
        meas_noise: float = 5.0,
    ):
        self.beta = beta
        self.tau_s = tau_s
        self._q = process_noise ** 2
        self._r0 = meas_noise ** 2

        # State and covariance
        self.x = np.zeros(2)
        self.P = np.eye(2) * 100.0

        # Matrices
        self.F = np.eye(2)               # state transition
        self.H = np.eye(2)               # observation
        self.Q = np.eye(2) * self._q
        self.R0 = np.eye(2) * self._r0

        # Smoothed spoofing confidence
        self._p_smooth: float = 0.0
        self._history: list = []

    def predict(self, dt: float = 1.0) -> None:
        """EKF prediction step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q * dt

    def update(
        self,
        measurement: np.ndarray,
        spoof_confidence: float = 0.0,
    ) -> Tuple[np.ndarray, float]:
        """
        EKF update step with soft noise inflation.

        Parameters
        ----------
        measurement : np.ndarray, shape (2,)
            GPS-derived position measurement [x, y] in metres.
        spoof_confidence : float
            Spoofing probability from NEMESIS detector, range [0, 1].
            0 = clear sky, 1 = definitely spoofed.

        Returns
        -------
        (estimated_position, inflation_factor) : (np.ndarray, float)
        """
        # Smooth the spoofing confidence
        alpha = 1.0 - np.exp(-1.0 / max(self.tau_s, 1e-6))
        self._p_smooth = (1 - alpha) * self._p_smooth + alpha * spoof_confidence

        # Soft noise inflation
        inflation = 1.0 + self.beta * self._p_smooth
        R_inflated = self.R0 * inflation

        # Kalman gain
        S = self.H @ self.P @ self.H.T + R_inflated
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        innovation = measurement - self.H @ self.x
        self.x = self.x + K @ innovation
        self.P = (np.eye(2) - K @ self.H) @ self.P

        self._history.append({
            "position": self.x.copy(),
            "inflation": inflation,
            "spoof_confidence": spoof_confidence,
        })
        return self.x.copy(), float(inflation)

    def reset(self, initial_position: Optional[np.ndarray] = None) -> None:
        """Reset state to zero (or provided position)."""
        self.x = initial_position.copy() if initial_position is not None else np.zeros(2)
        self.P = np.eye(2) * 100.0
        self._p_smooth = 0.0
        self._history.clear()

    @property
    def history(self) -> list:
        return self._history
