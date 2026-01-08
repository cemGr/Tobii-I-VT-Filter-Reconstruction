"""Core modules - legacy compatibility layer."""

# Import from new locations for backwards compatibility
from ..processing.velocity import compute_olsen_velocity, compute_olsen_velocity_from_slim_tsv
from ..processing.classification import apply_ivt_classifier
from .gaze import prepare_combined_columns, smooth_combined_gaze, gap_fill_gaze

__all__ = [
    "compute_olsen_velocity",
    "compute_olsen_velocity_from_slim_tsv",
    "apply_ivt_classifier",
    "prepare_combined_columns",
    "smooth_combined_gaze",
    "gap_fill_gaze",
]
