# NEMESIS-GNSS

**Wavelet-domain JEPA GNSS Spoofing Detection Framework**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue)](https://pypi.org/project/nemesis-gnss)
[![PyPI](https://img.shields.io/pypi/v/nemesis-gnss)](https://pypi.org/project/nemesis-gnss)
[![IEEE ACES](https://img.shields.io/badge/paper-IEEE%20ACES-green)](https://ieeexplore.ieee.org)

NEMESIS-GNSS is a research-grade, pip-installable framework for GPS L1 C/A spoofing detection using self-supervised wavelet-domain learning. It is designed to be intuitive, reproducible, and easy to use with your own SDR-captured IQ data.

---

## Features

| Component | Description | Extra |
|-----------|-------------|-------|
| **NEMESIS** | Wavelet-JEPA encoder + focal-loss MLP probe | core |
| **NEMESIS-Shield** | Multichannel scalogram + SE channel attention | `[shield]` |
| **NEMESIS-Nav** | EKF-coupled soft noise inflation (beta=12, tau_s=0.5) | `[nav]` |

**Three attack classes supported:** Meaconing, Slow Drift, Adversarial

---

## Installation

```bash
# Core only
pip install nemesis-gnss

# With NEMESIS-Shield
pip install nemesis-gnss[shield]

# With NEMESIS-Nav
pip install nemesis-gnss[nav]

# Everything
pip install nemesis-gnss[all]
```

---

## 60-Second Quickstart

### 1. Prepare your dataset

```
my_dataset/
├── clear_sky/           <- nominal GPS IQ files (.bin, .npy, .npz)
└── spoofed/
    ├── meaconing/
    ├── slow_drift/
    └── adversarial/
```

Supported IQ formats: `.bin` `.npy` `.npz` `.dat` `.iq` `.cf32`
(interleaved float32 or complex64)

### 2. Train

```bash
nemesis train ./my_dataset --output ./checkpoints
```

```python
from nemesis import Trainer

trainer = Trainer(data_path="./my_dataset", output_dir="./checkpoints")
trainer.fit()
```

### 3. Evaluate

```bash
nemesis eval ./checkpoints ./my_dataset
```

### 4. Predict on a new file

```bash
nemesis predict ./checkpoints path/to/sample.bin
```

```python
from nemesis import NEMESISDetector

model = NEMESISDetector.load("./checkpoints")
result = model.predict("path/to/sample.bin")

print(result["label"])       # "Meaconing"
print(result["confidence"])  # 0.97
print(result["spoofed"])     # True
```

### 5. Visualize

```bash
nemesis visualize ./checkpoints ./my_dataset
```

---

## Python API

```python
from nemesis import NEMESISDetector, Trainer
from nemesis.data import IQDataset
from nemesis.models import WaveletJEPAEncoder, MLPProbe
from nemesis.viz import plot_scalogram, plot_roc_curve

# Train
trainer = Trainer("./data", "./checkpoints")
trainer.fit()

# Predict
model = NEMESISDetector.load("./checkpoints")
result = model.predict("sample.bin")

# Batch predict
results = model.predict_batch("./test_data/")

# Visualize
plot_scalogram("sample.bin", label="Meaconing")
```

---

## NEMESIS-Nav (EKF Integration)

```python
from nemesis import NEMESISDetector
from nemesis.nav import NEMESISNavEKF
import numpy as np

detector = NEMESISDetector.load("./checkpoints")
ekf = NEMESISNavEKF(beta=12.0, tau_s=0.5)

for measurement in gps_stream:
    result = detector.predict(measurement.iq_file)
    ekf.predict(dt=1.0)
    pos, inflation = ekf.update(
        measurement.position,
        spoof_confidence=result["probabilities"]["Meaconing"]
                        + result["probabilities"]["Slow Drift"]
                        + result["probabilities"]["Adversarial"],
    )
```

---

## CLI Reference

```
nemesis train      Train a detector on your dataset
nemesis eval       Evaluate a saved checkpoint
nemesis predict    Detect spoofing in a single IQ file
nemesis visualize  Generate ROC curves and scalogram plots
nemesis info       Show system and version info
nemesis demo       Interactive step-by-step guided walkthrough
```

Run `nemesis <command> --help` for full options.

---

## Checkpoint Files

After training, the following files are saved to your output directory:

```
checkpoints/
├── nemesis_encoder.pt       # Wavelet-JEPA encoder weights
├── nemesis_mlp_probe.pt     # MLP classification probe weights
├── nemesis_mlp_scaler.pkl   # StandardScaler (fitted on training features)
├── config.yaml              # Full hyperparameter config (for reproducibility)
├── training_history.yaml    # Loss and accuracy per epoch
└── nemesis_report.html      # Interactive evaluation report
```

---

## Papers

If you use NEMESIS-GNSS, please cite:

**NEMESIS (core)**
```
@article{nemesis2024,
  title   = {NEMESIS: Wavelet-domain JEPA Encoder for GPS Spoofing Detection},
  author  = {Kavishka et al.},
  journal = {IEEE ACES},
  year    = {2024}
}
```

**NEMESIS-Shield**
```
@article{nemesis_shield2024,
  title  = {NEMESIS-Shield: Multichannel Scalogram with SE Attention for GNSS Spoofing},
  author = {Kavishka et al.},
  note   = {Under review, IEEE VTC}
}
```

**NEMESIS-Nav**
```
@article{nemesis_nav2024,
  title  = {NEMESIS-Nav: EKF-Coupled Soft Noise Inflation for GNSS Navigation Under Spoofing},
  author = {Kavishka et al.},
  note   = {Submitted, IEEE TVT}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
