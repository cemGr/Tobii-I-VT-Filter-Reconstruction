"""Evaluation: analysis, validation, and visualization tools."""

from .evaluation import compute_ivt_metrics, evaluate_ivt_vs_ground_truth
from .plotting import plot_velocity_only, plot_velocity_and_classification
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

__all__ = [
    'compute_ivt_metrics',
    'evaluate_ivt_vs_ground_truth',
    'plot_velocity_only',
    'plot_velocity_and_classification',
    'ExperimentConfig',
    'ExperimentManager',
    # Maximum IoU event-level evaluation (Startsev & Zemblys, 2022)
    'Event',
    'MatchResult',
    'extract_events',
    'compute_iou',
    'match_events_max_iou',
    'compute_event_iou_metrics',
    'format_event_iou_report',
]
