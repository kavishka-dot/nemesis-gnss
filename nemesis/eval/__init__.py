"""nemesis.eval — Metrics and evaluation utilities."""

from nemesis.eval.metrics import evaluate_detector, classification_report_nemesis
from nemesis.eval.report import generate_report

__all__ = ["evaluate_detector", "classification_report_nemesis", "generate_report"]
