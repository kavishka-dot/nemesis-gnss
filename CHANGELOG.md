# Changelog

All notable changes to `nemesis-gnss` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2025-01-01

### Added
- Initial public release
- `NEMESISDetector` core class with `predict()`, `save()`, `load()`
- `Trainer` with JEPA pretraining and focal-loss MLP probe
- `IQDataset` with automatic folder-based label discovery
- Support for `.bin`, `.npy`, `.npz`, `.dat`, `.iq`, `.cf32` IQ formats
- NEMESIS-Shield (`nemesis-gnss[shield]`) with SE channel attention
- NEMESIS-Nav (`nemesis-gnss[nav]`) with 2-state EKF and soft noise inflation
- Full CLI (`nemesis train`, `eval`, `predict`, `visualize`, `info`, `demo`)
- ROC curve, scalogram, and Pareto front visualizations
- HTML evaluation report generator
- Apache 2.0 license
