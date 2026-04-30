"""
nemesis.cli.main
================
Command-line interface for NEMESIS-GNSS.

Commands
--------
  nemesis train      Train a detector on your IQ dataset
  nemesis eval       Evaluate a saved checkpoint
  nemesis predict    Run detection on a single IQ file
  nemesis visualize  Generate performance visualizations
  nemesis info       Show system and package info
  nemesis demo       Interactive guided walkthrough
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="nemesis",
    help="[bold cyan]NEMESIS-GNSS[/bold cyan] — Wavelet JEPA GNSS Spoofing Detector",
    rich_markup_mode="rich",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

# ------------------------------------------------------------------
# nemesis train
# ------------------------------------------------------------------

@app.command()
def train(
    data: Path = typer.Argument(
        ...,
        help="Path to dataset root directory (must contain clear_sky/ and spoofed/).",
        show_default=False,
    ),
    output: Path = typer.Option(
        Path("./nemesis_checkpoints"),
        "--output", "-o",
        help="Directory to save trained model checkpoints.",
    ),
    epochs_pretrain: int = typer.Option(20, "--pretrain-epochs", help="JEPA pretraining epochs."),
    epochs_probe: int = typer.Option(50, "--probe-epochs", help="MLP probe training epochs."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Training batch size."),
    embed_dim: int = typer.Option(256, "--embed-dim", help="Encoder embedding dimension."),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    device: Optional[str] = typer.Option(None, "--device", help="'cuda' or 'cpu'. Auto-detected if omitted."),
    report: bool = typer.Option(True, "--report/--no-report", help="Generate HTML training report."),
):
    """
    [bold]Train a NEMESIS detector on your own IQ dataset.[/bold]

    [dim]Dataset structure expected:[/dim]

    [cyan]  my_dataset/[/cyan]
    [cyan]  ├── clear_sky/          [/cyan][dim]← nominal GPS IQ files[/dim]
    [cyan]  └── spoofed/[/cyan]
    [cyan]      ├── meaconing/      [/cyan][dim]← label 1[/dim]
    [cyan]      ├── slow_drift/     [/cyan][dim]← label 2[/dim]
    [cyan]      └── adversarial/    [/cyan][dim]← label 3[/dim]

    [dim]Supported formats: .bin .npy .npz .dat .iq .cf32[/dim]
    """
    from nemesis.train.trainer import Trainer

    if not data.exists():
        console.print(f"[red]Error:[/red] Data directory not found: {data}")
        console.print("\n[yellow]Tip:[/yellow] Create your dataset folder first:")
        console.print("  mkdir -p my_dataset/clear_sky my_dataset/spoofed/meaconing")
        console.print("  # then add your .bin or .npy IQ files")
        raise typer.Exit(1)

    config = {
        "pretrain_epochs": epochs_pretrain,
        "probe_epochs": epochs_probe,
        "batch_size": batch_size,
        "embed_dim": embed_dim,
        "probe_lr": lr,
        "pretrain_lr": lr,
    }

    trainer = Trainer(
        data_path=data,
        output_dir=output,
        config=config,
        device=device,
    )
    trainer.fit()

    if report:
        from nemesis.eval.metrics import evaluate_detector
        from nemesis.eval.report import generate_report
        detector = trainer.get_detector()
        results = evaluate_detector(detector, data, verbose=True)
        report_path = output / "nemesis_report.html"
        generate_report(results, trainer._history, output_path=report_path)
        console.print(f"\n[cyan]Report saved:[/cyan] {report_path}")


# ------------------------------------------------------------------
# nemesis eval
# ------------------------------------------------------------------

@app.command()
def eval(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint directory (output of 'nemesis train').",
    ),
    data: Path = typer.Argument(
        ...,
        help="Path to evaluation dataset root directory.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Directory to save evaluation report. Defaults to checkpoint dir.",
    ),
    device: Optional[str] = typer.Option(None, "--device", help="'cuda' or 'cpu'."),
):
    """
    [bold]Evaluate a saved NEMESIS checkpoint on a dataset.[/bold]

    Prints accuracy, F1, AUC, and confusion matrix. Generates an HTML report.
    """
    from nemesis.detector import NEMESISDetector
    from nemesis.eval.metrics import evaluate_detector
    from nemesis.eval.report import generate_report

    _require_checkpoint(checkpoint)
    _require_data(data)

    console.print(f"[cyan]Loading checkpoint:[/cyan] {checkpoint}")
    detector = NEMESISDetector.load(checkpoint, device=device)

    console.print(f"[cyan]Evaluating on:[/cyan] {data}\n")
    results = evaluate_detector(detector, data, verbose=True)

    out_dir = output or checkpoint
    report_path = out_dir / "nemesis_eval_report.html"
    generate_report(results, output_path=report_path)
    console.print(f"\n[cyan]Report saved:[/cyan] {report_path}")


# ------------------------------------------------------------------
# nemesis predict
# ------------------------------------------------------------------

@app.command()
def predict(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint directory."),
    iq_file: Path = typer.Argument(..., help="Path to IQ file (.bin, .npy, etc.)."),
    device: Optional[str] = typer.Option(None, "--device", help="'cuda' or 'cpu'."),
):
    """
    [bold]Run spoofing detection on a single IQ file.[/bold]
    """
    from nemesis.detector import NEMESISDetector

    _require_checkpoint(checkpoint)
    if not iq_file.exists():
        console.print(f"[red]IQ file not found:[/red] {iq_file}")
        raise typer.Exit(1)

    detector = NEMESISDetector.load(checkpoint, device=device)
    result = detector.predict(iq_file)

    status = "[bold red]SPOOFED[/bold red]" if result["spoofed"] else "[bold green]CLEAR SKY[/bold green]"
    console.print(Panel.fit(
        f"File: {iq_file.name}\n"
        f"Status    : {status}\n"
        f"Class     : [bold]{result['label']}[/bold]\n"
        f"Confidence: [cyan]{result['confidence'] * 100:.1f}%[/cyan]\n\n"
        + "\n".join(
            f"  {name:<14}: {prob * 100:5.1f}%"
            for name, prob in result["probabilities"].items()
        ),
        title="NEMESIS Prediction",
        border_style="cyan" if not result["spoofed"] else "red",
    ))


# ------------------------------------------------------------------
# nemesis visualize
# ------------------------------------------------------------------

@app.command()
def visualize(
    checkpoint: Path = typer.Argument(..., help="Path to checkpoint directory."),
    data: Path = typer.Argument(..., help="Path to dataset directory."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Directory to save plots."),
    device: Optional[str] = typer.Option(None, "--device"),
):
    """
    [bold]Generate performance visualizations.[/bold]

    Produces ROC curves, confusion matrix heatmap, and sample scalograms.
    """
    from nemesis.detector import NEMESISDetector
    from nemesis.eval.metrics import evaluate_detector
    from nemesis.viz.roc import plot_roc_curve
    from nemesis.viz.scalogram import plot_scalogram

    _require_checkpoint(checkpoint)
    _require_data(data)

    out_dir = output or (checkpoint / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = NEMESISDetector.load(checkpoint, device=device)
    results = evaluate_detector(detector, data, verbose=False)

    roc_path = out_dir / "roc_curves.png"
    plot_roc_curve(results["y_true"], results["y_proba"], save_path=roc_path, show=False)
    console.print(f"[green]ROC curve saved:[/green] {roc_path}")

    # Plot one scalogram per class
    from nemesis.data.dataset import IQDataset, CLASS_NAMES
    dataset = IQDataset(data, verbose=False)
    seen = set()
    for path, label in dataset.samples:
        if label not in seen:
            scalo_path = out_dir / f"scalogram_{CLASS_NAMES[label].lower().replace(' ', '_')}.png"
            plot_scalogram(path, label=CLASS_NAMES[label], save_path=scalo_path, show=False)
            console.print(f"[green]Scalogram saved:[/green] {scalo_path}")
            seen.add(label)
        if len(seen) == 4:
            break

    console.print(f"\n[cyan]All plots saved to:[/cyan] {out_dir}")


# ------------------------------------------------------------------
# nemesis info
# ------------------------------------------------------------------

@app.command()
def info():
    """[bold]Show system information and installed NEMESIS version.[/bold]"""
    import torch
    import platform
    import nemesis

    table = Table(title="NEMESIS System Info", header_style="bold cyan")
    table.add_column("Component")
    table.add_column("Value", style="white")

    table.add_row("nemesis-gnss", nemesis.__version__)
    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("GPU", torch.cuda.get_device_name(0))
    table.add_row("Platform", platform.platform())

    console.print(table)
    console.print("\n[dim]GitHub: https://github.com/kavishka-dot/nemesis-gnss[/dim]")


# ------------------------------------------------------------------
# nemesis demo
# ------------------------------------------------------------------

@app.command()
def demo():
    """[bold]Interactive guided walkthrough of the NEMESIS pipeline.[/bold]"""
    console.print(Panel.fit(
        "[bold cyan]Welcome to NEMESIS-GNSS[/bold cyan]\n\n"
        "This demo will walk you through the full pipeline:\n"
        "  1. Prepare your dataset\n"
        "  2. Train a detector\n"
        "  3. Evaluate performance\n"
        "  4. Visualize results\n"
        "  5. Run inference on new IQ files",
        title="NEMESIS Demo",
        border_style="cyan",
    ))

    steps = [
        ("Prepare Dataset", (
            "Create the following folder structure:\n\n"
            "  my_dataset/\n"
            "  ├── clear_sky/       <- .bin or .npy IQ files\n"
            "  └── spoofed/\n"
            "      ├── meaconing/\n"
            "      ├── slow_drift/\n"
            "      └── adversarial/\n\n"
            "Supported: .bin .npy .npz .dat .iq .cf32\n"
            "Each file should be interleaved float32 IQ samples."
        )),
        ("Train", (
            "Run:\n\n"
            "  nemesis train ./my_dataset --output ./checkpoints\n\n"
            "Options:\n"
            "  --pretrain-epochs 20   JEPA pretraining\n"
            "  --probe-epochs 50      MLP probe training\n"
            "  --batch-size 32\n"
            "  --device cuda\n\n"
            "Checkpoint files saved automatically:\n"
            "  nemesis_encoder.pt\n"
            "  nemesis_mlp_probe.pt\n"
            "  nemesis_mlp_scaler.pkl\n"
            "  config.yaml"
        )),
        ("Evaluate", (
            "Run:\n\n"
            "  nemesis eval ./checkpoints ./my_dataset\n\n"
            "Outputs:\n"
            "  - Accuracy, F1, AUC printed to terminal\n"
            "  - Confusion matrix\n"
            "  - nemesis_eval_report.html"
        )),
        ("Visualize", (
            "Run:\n\n"
            "  nemesis visualize ./checkpoints ./my_dataset\n\n"
            "Generates:\n"
            "  - ROC curves (per class)\n"
            "  - Sample scalograms (one per class)"
        )),
        ("Predict", (
            "Run:\n\n"
            "  nemesis predict ./checkpoints path/to/sample.bin\n\n"
            "Output:\n"
            "  - Status: CLEAR SKY or SPOOFED\n"
            "  - Detected class (Meaconing / Slow Drift / Adversarial)\n"
            "  - Confidence score\n"
            "  - Per-class probabilities"
        )),
    ]

    for i, (title, content) in enumerate(steps, 1):
        console.print(f"\n[bold cyan]Step {i}: {title}[/bold cyan]")
        console.print(Panel(content, border_style="dim"))
        if i < len(steps):
            typer.confirm("Continue to next step?", default=True)

    console.print(Panel.fit(
        "[bold green]You are ready to use NEMESIS![/bold green]\n\n"
        "For help on any command:\n"
        "  nemesis train --help\n"
        "  nemesis eval --help\n\n"
        "GitHub: https://github.com/kavishka-dot/nemesis-gnss",
        border_style="green",
    ))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _require_checkpoint(path: Path) -> None:
    required = ["nemesis_encoder.pt", "nemesis_mlp_probe.pt", "nemesis_mlp_scaler.pkl"]
    missing = [f for f in required if not (path / f).exists()]
    if missing:
        console.print(f"[red]Error:[/red] Checkpoint directory incomplete: {path}")
        console.print(f"Missing files: {missing}")
        console.print("\n[yellow]Tip:[/yellow] Run 'nemesis train' first to create a checkpoint.")
        raise typer.Exit(1)


def _require_data(path: Path) -> None:
    if not path.exists():
        console.print(f"[red]Error:[/red] Data directory not found: {path}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
