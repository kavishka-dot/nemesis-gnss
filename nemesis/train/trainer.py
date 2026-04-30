"""
nemesis.train.trainer
=====================
High-level Trainer — the primary interface for training NEMESIS models
on user-provided IQ datasets.

Usage
-----
::

    from nemesis import Trainer

    trainer = Trainer(
        data_path="./my_dataset",
        output_dir="./checkpoints",
    )
    trainer.fit()
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from nemesis.data.dataset import IQDataset
from nemesis.models.encoder import WaveletJEPAEncoder
from nemesis.models.probe import MLPProbe
from nemesis.train.losses import FocalLoss
from nemesis.train.callbacks import EarlyStopping, CheckpointCallback

console = Console()


class Trainer:
    """
    End-to-end trainer for NEMESIS spoofing detector.

    Handles dataset loading, encoder pretraining (JEPA), probe training,
    scaler fitting, checkpointing, and report generation.

    Parameters
    ----------
    data_path : str or Path
        Root directory with ``clear_sky/`` and ``spoofed/`` sub-folders.
    output_dir : str or Path
        Directory to save checkpoints and reports. Created if missing.
    config : dict, optional
        Override any default hyperparameters. See ``default_config()``.
    device : str, optional
        ``"cuda"`` or ``"cpu"``. Auto-detected if not provided.

    Examples
    --------
    >>> trainer = Trainer(data_path="./dataset", output_dir="./checkpoints")
    >>> trainer.fit()
    >>> trainer.save()
    """

    def __init__(
        self,
        data_path: str | Path,
        output_dir: str | Path = "./nemesis_checkpoints",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = self.default_config()
        if config:
            self.cfg.update(config)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Will be populated during fit()
        self.encoder: Optional[WaveletJEPAEncoder] = None
        self.probe: Optional[MLPProbe] = None
        self.scaler: Optional[StandardScaler] = None
        self._dataset: Optional[IQDataset] = None
        self._history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> "Trainer":
        """
        Run the full training pipeline.

        Steps
        -----
        1. Load and validate dataset
        2. Build encoder and probe
        3. Pretrain encoder (JEPA reconstruction objective)
        4. Extract features and fit StandardScaler
        5. Train MLP probe with focal loss
        6. Evaluate on validation split
        7. Save checkpoint automatically

        Returns
        -------
        self
        """
        console.print(Panel.fit(
            "[bold cyan]NEMESIS Training Pipeline[/bold cyan]\n"
            f"Data: {self.data_path}\n"
            f"Output: {self.output_dir}\n"
            f"Device: {self.device.upper()}",
            title="NEMESIS-GNSS",
            border_style="cyan",
        ))

        # Step 1: Load dataset
        console.rule("[bold]Step 1 / 5 — Loading Dataset")
        self._dataset = IQDataset(
            data_path=self.data_path,
            segment_length=self.cfg["segment_length"],
            verbose=True,
        )
        train_idx, val_idx = self._split_indices(len(self._dataset))
        train_loader = DataLoader(
            Subset(self._dataset, train_idx),
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=(self.device == "cuda"),
        )
        val_loader = DataLoader(
            Subset(self._dataset, val_idx),
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
        )
        console.print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

        # Step 2: Build models
        console.rule("[bold]Step 2 / 5 — Building Models")
        self.encoder = WaveletJEPAEncoder(
            input_channels=self.cfg["input_channels"],
            embed_dim=self.cfg["embed_dim"],
        ).to(self.device)
        self.probe = MLPProbe(
            input_dim=self.cfg["embed_dim"],
            num_classes=self.cfg["num_classes"],
        ).to(self.device)
        console.print(f"  Encoder parameters : {self.encoder.num_parameters:,}")
        console.print(f"  Probe parameters   : {self.probe.num_parameters:,}")

        # Step 3: Pretrain encoder
        console.rule("[bold]Step 3 / 5 — Pretraining Encoder (JEPA)")
        self._pretrain_encoder(train_loader)

        # Step 4: Fit scaler on encoder features
        console.rule("[bold]Step 4 / 5 — Fitting Feature Scaler")
        self.scaler = self._fit_scaler(train_loader)

        # Step 5: Train probe
        console.rule("[bold]Step 5 / 5 — Training MLP Probe")
        self._train_probe(train_loader, val_loader)

        # Auto-save
        self.save()
        console.print(Panel.fit(
            "[bold green]Training complete![/bold green]\n"
            f"Checkpoint saved to: {self.output_dir}",
            border_style="green",
        ))
        return self

    def save(self) -> None:
        """Save encoder, probe, scaler, config, and training history to ``output_dir``."""
        if self.encoder is None or self.probe is None or self.scaler is None:
            raise RuntimeError("Call trainer.fit() before trainer.save().")

        torch.save(self.encoder.state_dict(), self.output_dir / "nemesis_encoder.pt")
        torch.save(self.probe.state_dict(), self.output_dir / "nemesis_mlp_probe.pt")
        with open(self.output_dir / "nemesis_mlp_scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.output_dir / "config.yaml", "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)

        history_path = self.output_dir / "training_history.yaml"
        with open(history_path, "w") as f:
            yaml.dump(self._history, f)

        console.print(f"[green]Checkpoint saved:[/green] {self.output_dir}")

    def get_detector(self):
        """Return a ready-to-use NEMESISDetector from the trained components."""
        from nemesis.detector import NEMESISDetector
        if self.encoder is None:
            raise RuntimeError("Call trainer.fit() first.")
        return NEMESISDetector(
            encoder=self.encoder,
            probe=self.probe,
            scaler=self.scaler,
            config=self.cfg,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Default config
    # ------------------------------------------------------------------

    @staticmethod
    def default_config() -> Dict[str, Any]:
        return {
            "segment_length": 4096,
            "input_channels": 2,
            "embed_dim": 256,
            "num_classes": 4,
            "batch_size": 32,
            "num_workers": 0,
            "pretrain_epochs": 20,
            "pretrain_lr": 1e-3,
            "probe_epochs": 50,
            "probe_lr": 1e-3,
            "weight_decay": 1e-4,
            "focal_gamma": 2.0,
            "val_split": 0.2,
            "early_stopping_patience": 10,
            "wavelet": "morl",
            "scales": 64,
            "sampling_rate": 2.046e6,
        }

    # ------------------------------------------------------------------
    # Internal training steps
    # ------------------------------------------------------------------

    def _pretrain_encoder(self, loader: DataLoader) -> None:
        """JEPA-style pretraining: predict masked patch embeddings."""
        # Simplified: reconstruction of randomly masked scalogram regions
        optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.cfg["pretrain_lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg["pretrain_epochs"]
        )
        # Simple decoder for reconstruction pretext task
        decoder = nn.Sequential(
            nn.Linear(self.cfg["embed_dim"], self.cfg["embed_dim"] * 2),
            nn.GELU(),
            nn.Linear(self.cfg["embed_dim"] * 2, self.cfg["input_channels"] * self.cfg["scales"] * 16),
        ).to(self.device)
        dec_opt = torch.optim.AdamW(decoder.parameters(), lr=self.cfg["pretrain_lr"])

        self.encoder.train()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} epochs"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Pretraining encoder...", total=self.cfg["pretrain_epochs"])
            for epoch in range(self.cfg["pretrain_epochs"]):
                epoch_loss = 0.0
                for batch_x, _ in loader:
                    batch_x = batch_x.to(self.device)
                    # Mask a random horizontal strip
                    masked = batch_x.clone()
                    t_start = torch.randint(0, max(1, batch_x.shape[-1] - 16), (1,)).item()
                    masked[..., t_start: t_start + 16] = 0.0
                    target = batch_x[..., t_start: t_start + 16].detach()

                    z = self.encoder(masked)
                    recon = decoder(z).view(
                        batch_x.shape[0],
                        self.cfg["input_channels"],
                        self.cfg["scales"],
                        16,
                    )
                    loss = nn.functional.mse_loss(recon, target)
                    optimizer.zero_grad()
                    dec_opt.zero_grad()
                    loss.backward()
                    optimizer.step()
                    dec_opt.step()
                    epoch_loss += loss.item()
                scheduler.step()
                progress.advance(task)
        console.print(f"  Pretraining complete. Final loss: {epoch_loss / max(len(loader), 1):.4f}")

    def _fit_scaler(self, loader: DataLoader) -> StandardScaler:
        self.encoder.eval()
        features = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                z = self.encoder(batch_x)
                features.append(z.cpu().numpy())
        all_features = np.concatenate(features, axis=0)
        scaler = StandardScaler()
        scaler.fit(all_features)
        console.print(f"  Scaler fitted on {len(all_features)} samples.")
        return scaler

    def _train_probe(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        criterion = FocalLoss(gamma=self.cfg["focal_gamma"])
        optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.cfg["probe_lr"],
            weight_decay=self.cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg["probe_epochs"]
        )
        early_stop = EarlyStopping(patience=self.cfg["early_stopping_patience"])
        self.encoder.eval()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Training probe...", total=self.cfg["probe_epochs"])
            for epoch in range(self.cfg["probe_epochs"]):
                # Train
                self.probe.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    with torch.no_grad():
                        z = self.encoder(batch_x).cpu().numpy()
                    z_scaled = torch.tensor(
                        self.scaler.transform(z), dtype=torch.float32
                    ).to(self.device)
                    logits = self.probe(z_scaled)
                    loss = criterion(logits, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validate
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                scheduler.step()

                self._history["train_loss"].append(train_loss / max(len(train_loader), 1))
                self._history["val_loss"].append(val_loss)
                self._history["val_acc"].append(val_acc)

                progress.advance(task)
                if early_stop(val_loss):
                    console.print(f"  [yellow]Early stopping at epoch {epoch + 1}[/yellow]")
                    break

        console.print(f"  Final val accuracy: [bold green]{val_acc * 100:.2f}%[/bold green]")

    def _evaluate(self, loader: DataLoader, criterion) -> Tuple[float, float]:
        self.probe.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                z = self.encoder(batch_x).cpu().numpy()
                z_scaled = torch.tensor(
                    self.scaler.transform(z), dtype=torch.float32
                ).to(self.device)
                logits = self.probe(z_scaled)
                loss = criterion(logits, batch_y)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        return total_loss / max(len(loader), 1), correct / max(total, 1)

    def _split_indices(self, n: int):
        idx = list(range(n))
        val_size = int(n * self.cfg["val_split"])
        return idx[val_size:], idx[:val_size]
