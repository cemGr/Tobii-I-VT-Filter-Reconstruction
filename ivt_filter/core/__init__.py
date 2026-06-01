"""Legacy compatibility imports for canonical processing and preprocessing APIs."""

from ..preprocessing import gap_fill_gaze, prepare_combined_columns, smooth_combined_gaze
from ..processing.classification import apply_ivt_classifier
from ..processing.velocity import compute_olsen_velocity, compute_olsen_velocity_from_slim_tsv

__all__ = [
    "compute_olsen_velocity",
    "compute_olsen_velocity_from_slim_tsv",
    "apply_ivt_classifier",
    "prepare_combined_columns",
    "smooth_combined_gaze",
    "gap_fill_gaze",
]
