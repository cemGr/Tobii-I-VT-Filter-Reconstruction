"""Backward-compatible imports for the canonical preprocessing implementation.

New code should import from :mod:`ivt_filter.preprocessing` directly.
"""

from ..preprocessing import (
    apply_tobii_eye_offset_interpolation,
    gap_fill_gaze,
    prepare_combined_columns,
    smooth_combined_gaze,
)

__all__ = [
    "apply_tobii_eye_offset_interpolation",
    "gap_fill_gaze",
    "prepare_combined_columns",
    "smooth_combined_gaze",
]
