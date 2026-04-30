"""
nemesis.data.transforms
========================
Signal preprocessing transforms for GPS L1 C/A IQ data.
"""

from __future__ import annotations

import numpy as np
import pywt


class WaveletTransform:
    """
    Continuous Wavelet Transform (CWT) applied to IQ data.

    Produces a 2D scalogram (scales x time) from raw IQ samples,
    suitable as input to the JEPA encoder.

    Parameters
    ----------
    wavelet : str
        PyWavelets wavelet name. Default ``"morl"`` (Morlet).
    scales : int
        Number of frequency scales. Default 64.
    sampling_rate : float
        IQ sample rate in Hz. Used to compute frequency axis. Default 2.046e6.

    Output shape: ``(2, scales, segment_length)``
    Row 0 = magnitude scalogram of I channel,
    Row 1 = magnitude scalogram of Q channel.
    """

    def __init__(
        self,
        wavelet: str = "morl",
        scales: int = 64,
        sampling_rate: float = 2.046e6,
    ):
        self.wavelet = wavelet
        self.scales = scales
        self.sampling_rate = sampling_rate
        self._scale_arr = np.arange(1, scales + 1, dtype=np.float64)

    def __call__(self, iq: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        iq : np.ndarray, shape (2, N)
            I/Q signal. Row 0 = I, Row 1 = Q.

        Returns
        -------
        np.ndarray, shape (2, scales, N), dtype float32
        """
        i_scalo = self._cwt(iq[0])
        q_scalo = self._cwt(iq[1])
        out = np.stack([i_scalo, q_scalo], axis=0).astype(np.float32)
        return out

    def _cwt(self, signal: np.ndarray) -> np.ndarray:
        coeffs, _ = pywt.cwt(signal, self._scale_arr, self.wavelet)
        magnitude = np.abs(coeffs)
        # Log-scale normalisation for stable training
        magnitude = np.log1p(magnitude)
        # Normalise to [0, 1]
        vmax = magnitude.max()
        if vmax > 0:
            magnitude /= vmax
        return magnitude  # shape (scales, N)

    def __repr__(self) -> str:
        return (
            f"WaveletTransform(wavelet={self.wavelet!r}, "
            f"scales={self.scales}, "
            f"sampling_rate={self.sampling_rate:.3e})"
        )
