"""Evaluation: analysis, validation, and visualization tools."""

from .evaluation import compute_ivt_metrics, evaluate_ivt_vs_ground_truth
from .plotting import plot_velocity_only, plot_velocity_and_classification
from .experiment import ExperimentConfig, ExperimentManager

__all__ = [
    'compute_ivt_metrics',
    'evaluate_ivt_vs_ground_truth',
    'plot_velocity_only',
    'plot_velocity_and_classification',
    'ExperimentConfig',
    'ExperimentManager',
]
