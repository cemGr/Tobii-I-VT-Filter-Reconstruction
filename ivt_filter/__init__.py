# ivt_filter/__init__.py
"""
IVT Filter Package.

Enthält:
- Olsen Velocity Berechnung
- I VT Klassifikation
- Post Processing
- Evaluation und Plotting
"""

from .config import OlsenVelocityConfig, IVTClassifierConfig, SaccadeMergeConfig
from .velocity import compute_olsen_velocity, compute_olsen_velocity_from_slim_tsv
from .classification import apply_ivt_classifier
from .postprocess import merge_short_saccade_blocks
from .evaluation import evaluate_ivt_vs_ground_truth, compute_ivt_metrics
from .plotting import plot_velocity_only, plot_velocity_and_classification  # optional
