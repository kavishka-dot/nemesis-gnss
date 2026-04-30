"""
nemesis.nav — EKF-coupled GNSS navigation with spoofing-aware noise inflation.

Requires: pip install nemesis-gnss[nav]
"""

from nemesis.nav.ekf import NEMESISNavEKF

__all__ = ["NEMESISNavEKF"]
