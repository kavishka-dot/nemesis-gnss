"""
NEMESIS-GNSS: Wavelet-domain JEPA-based GNSS Spoofing Detection Framework
==========================================================================

A pip-installable, research-grade framework for GNSS spoofing detection,
featuring:

  - NEMESIS     : Wavelet-JEPA encoder + focal-loss MLP probe (core)
  - NEMESIS-Shield : Multichannel scalogram + SE channel attention [shield extra]
  - NEMESIS-Nav : EKF-coupled soft noise inflation detector [nav extra]

Quick Start
-----------
>>> from nemesis import NEMESISDetector, Trainer
>>> # Train on your own data
>>> trainer = Trainer(data_path="path/to/dataset/", output_dir="./checkpoints")
>>> trainer.fit()
>>> # Load and predict
>>> model = NEMESISDetector.load("./checkpoints")
>>> result = model.predict("path/to/iq_sample.bin")

Paper References
----------------
- NEMESIS       : IEEE ACES (accepted)
- NEMESIS-Shield : IEEE VTC (under review)
- NEMESIS-Nav   : IEEE TVT (submitted)

GitHub: https://github.com/kavishka-dot/nemesis-gnss
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("nemesis-gnss")
except PackageNotFoundError:
    __version__ = "1.0.0-dev"

__author__ = "Kavishka"
__license__ = "Apache-2.0"

# Core public API — always available
from nemesis.detector import NEMESISDetector
from nemesis.train.trainer import Trainer

__all__ = [
    "NEMESISDetector",
    "Trainer",
    "__version__",
]
