"""
nemesis.detector
================
High-level NEMESISDetector class — the primary interface for inference.
Supports loading saved checkpoints and running predictions on IQ data files.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Union, Optional, Dict, Any

import numpy as np
import torch
import yaml
from rich.console import Console

from nemesis.data.loader import load_iq_file
from nemesis.data.transforms import WaveletTransform
from nemesis.models.encoder import WaveletJEPAEncoder
from nemesis.models.probe import MLPProbe

console = Console()

ATTACK_CLASSES = {0: "Clear Sky", 1: "Meaconing", 2: "Slow Drift", 3: "Adversarial"}


class NEMESISDetector:
    """
    GNSS spoofing detector using a pretrained wavelet-JEPA encoder and MLP probe.

    Supports three attack classes: Meaconing, Slow Drift, and Adversarial.

    Usage
    -----
    Load from a saved checkpoint directory::

        model = NEMESISDetector.load("./checkpoints")
        result = model.predict("path/to/iq_sample.bin")

    Or build programmatically::

        model = NEMESISDetector(encoder=my_encoder, probe=my_probe, scaler=my_scaler)
    """

    def __init__(
        self,
        encoder: WaveletJEPAEncoder,
        probe: MLPProbe,
        scaler,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.probe = probe.to(self.device)
        self.scaler = scaler
        self.config = config or {}
        self._wavelet_transform = WaveletTransform(
            wavelet=self.config.get("wavelet", "morl"),
            scales=self.config.get("scales", 64),
        )
        self.encoder.eval()
        self.probe.eval()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, iq_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Run spoofing detection on a single IQ data file.

        Parameters
        ----------
        iq_path : str or Path
            Path to a raw IQ file (.bin, .npy, .dat, .iq).
            Expected format: interleaved complex float32 samples.

        Returns
        -------
        dict with keys:
            - ``label``      : predicted class name (str)
            - ``class_id``   : class index (int)
            - ``confidence`` : confidence score 0-1 (float)
            - ``probabilities`` : per-class probabilities (dict)
            - ``spoofed``    : bool, True if not Clear Sky
        """
        iq_path = Path(iq_path)
        if not iq_path.exists():
            raise FileNotFoundError(f"IQ file not found: {iq_path}")

        iq_data = load_iq_file(iq_path)
        scalogram = self._wavelet_transform(iq_data)
        features = self._extract_features(scalogram)
        return self._classify(features)

    def predict_batch(self, iq_dir: Union[str, Path]) -> list:
        """
        Run detection on all IQ files in a directory.

        Parameters
        ----------
        iq_dir : str or Path
            Directory containing IQ files.

        Returns
        -------
        list of dicts, one per file, each matching the format of ``predict()``.
        """
        iq_dir = Path(iq_dir)
        extensions = {".bin", ".npy", ".dat", ".iq", ".cf32"}
        files = [f for f in iq_dir.iterdir() if f.suffix in extensions]
        if not files:
            raise ValueError(f"No IQ files found in {iq_dir}. "
                             f"Supported formats: {extensions}")

        results = []
        for f in files:
            result = self.predict(f)
            result["file"] = f.name
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, output_dir: Union[str, Path]) -> None:
        """
        Save the full detector (encoder, probe, scaler, config) to a directory.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save checkpoint files. Created if it does not exist.

        Saved files
        -----------
        - ``nemesis_encoder.pt``
        - ``nemesis_mlp_probe.pt``
        - ``nemesis_mlp_scaler.pkl``
        - ``config.yaml``
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.encoder.state_dict(), output_dir / "nemesis_encoder.pt")
        torch.save(self.probe.state_dict(), output_dir / "nemesis_mlp_probe.pt")
        with open(output_dir / "nemesis_mlp_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

        console.print(f"[green]Checkpoint saved to:[/green] {output_dir}")

    @classmethod
    def load(cls, checkpoint_dir: Union[str, Path], device: Optional[str] = None) -> "NEMESISDetector":
        """
        Load a saved NEMESISDetector from a checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory containing ``nemesis_encoder.pt``, ``nemesis_mlp_probe.pt``,
            ``nemesis_mlp_scaler.pkl``, and ``config.yaml``.
        device : str, optional
            ``"cuda"`` or ``"cpu"``. Auto-detected if not provided.

        Returns
        -------
        NEMESISDetector
        """
        checkpoint_dir = Path(checkpoint_dir)
        _require_file(checkpoint_dir / "nemesis_encoder.pt")
        _require_file(checkpoint_dir / "nemesis_mlp_probe.pt")
        _require_file(checkpoint_dir / "nemesis_mlp_scaler.pkl")

        config = {}
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        encoder = WaveletJEPAEncoder(
            input_channels=config.get("input_channels", 2),
            embed_dim=config.get("embed_dim", 256),
        )
        encoder.load_state_dict(torch.load(checkpoint_dir / "nemesis_encoder.pt", map_location=device))

        probe = MLPProbe(
            input_dim=config.get("embed_dim", 256),
            num_classes=config.get("num_classes", 4),
        )
        probe.load_state_dict(torch.load(checkpoint_dir / "nemesis_mlp_probe.pt", map_location=device))

        with open(checkpoint_dir / "nemesis_mlp_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        console.print(f"[green]Checkpoint loaded from:[/green] {checkpoint_dir}")
        return cls(encoder=encoder, probe=probe, scaler=scaler, config=config, device=device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_features(self, scalogram: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(scalogram, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.encoder(tensor)
        features_np = features.cpu().numpy()
        return self.scaler.transform(features_np)

    def _classify(self, features: np.ndarray) -> Dict[str, Any]:
        tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.probe(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        class_id = int(np.argmax(probs))
        label = ATTACK_CLASSES[class_id]
        confidence = float(probs[class_id])

        return {
            "label": label,
            "class_id": class_id,
            "confidence": confidence,
            "probabilities": {ATTACK_CLASSES[i]: float(p) for i, p in enumerate(probs)},
            "spoofed": class_id != 0,
        }

    def __repr__(self) -> str:
        return (
            f"NEMESISDetector("
            f"device={self.device!r}, "
            f"embed_dim={self.config.get('embed_dim', 256)}, "
            f"num_classes={self.config.get('num_classes', 4)})"
        )


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {path}\n"
            f"Make sure you have saved a trained model first using:\n"
            f"  trainer.save('./checkpoints')\n"
            f"  # or\n"
            f"  detector.save('./checkpoints')"
        )
