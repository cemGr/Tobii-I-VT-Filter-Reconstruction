"""Evaluation: analysis, validation, and optional visualization tools."""
from __future__ import annotations

from typing import Any

from .evaluation import compute_ivt_metrics, evaluate_ivt_vs_ground_truth
from .experiment import ExperimentConfig, ExperimentManager
from .event_iou import (
    Event,
    MatchResult,
    extract_events,
    compute_iou,
    match_events_max_iou,
    compute_event_iou_metrics,
    format_event_iou_report,
)

_PLOTTING_EXPORTS = {"plot_velocity_only", "plot_velocity_and_classification"}


def __getattr__(name: str) -> Any:
    """Load plotting helpers lazily so matplotlib remains optional."""
    if name in _PLOTTING_EXPORTS:
        from . import plotting

        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "compute_ivt_metrics",
    "evaluate_ivt_vs_ground_truth",
    "plot_velocity_only",
    "plot_velocity_and_classification",
    "ExperimentConfig",
    "ExperimentManager",
    # Maximum IoU event-level evaluation (Startsev & Zemblys, 2022)
    "Event",
    "MatchResult",
    "extract_events",
    "compute_iou",
    "match_events_max_iou",
    "compute_event_iou_metrics",
    "format_event_iou_report",
]
