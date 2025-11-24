"""Backward-compatible entry point for velocity processing.

All functionality has moved into the :mod:`ivt` package. Importing from this
module keeps existing scripts working while delegating to the new structure.
"""
from ivt.classifier import IVTClassifier, apply_ivt_classifier
from ivt.metrics import evaluate_ivt_vs_ground_truth
from ivt.velocity import VelocityCalculator, compute_olsen_velocity_from_slim_tsv

__all__ = [
    "IVTClassifier",
    "apply_ivt_classifier",
    "evaluate_ivt_vs_ground_truth",
    "VelocityCalculator",
    "compute_olsen_velocity_from_slim_tsv",
]


if __name__ == "__main__":
    from ivt.cli import main

    main()
