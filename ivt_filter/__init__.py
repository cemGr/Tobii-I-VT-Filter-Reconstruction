"""Tobii I-VT filter reconstruction package."""
from __future__ import annotations

from typing import Any

from .config import OlsenVelocityConfig, PipelineConfig
from .processing.velocity import compute_olsen_velocity, compute_olsen_velocity_from_slim_tsv
from .processing.classification import apply_ivt_classifier
from .postprocessing import merge_short_saccade_blocks
from .evaluation.evaluation import evaluate_ivt_vs_ground_truth, compute_ivt_metrics

_PLOTTING_EXPORTS = {"plot_velocity_only", "plot_velocity_and_classification"}


def __getattr__(name: str) -> Any:
    """Load plotting helpers lazily so importing the package does not need matplotlib."""
    if name in _PLOTTING_EXPORTS:
        from .evaluation import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OlsenVelocityConfig",
    "PipelineConfig",
    "compute_olsen_velocity",
    "compute_olsen_velocity_from_slim_tsv",
    "apply_ivt_classifier",
    "merge_short_saccade_blocks",
    "evaluate_ivt_vs_ground_truth",
    "compute_ivt_metrics",
    "plot_velocity_only",
    "plot_velocity_and_classification",
]
