"""Python implementation of the Tobii I-VT fixation filter.

This package reconstructs the seven-stage I-VT pipeline described by Olsen
(*Identification by Velocity Thresholding*, 2012) and exposes a small,
type-annotated API for running the filter on recordings loaded from Tobii
exports or synthetic data.
"""

from .engine import IVTFilterEngine, FilterResult, IIVTFilter
from .config import IVTFilterConfiguration, EyeSelectionMode

__all__ = [
    "IVTFilterEngine",
    "FilterResult",
    "IIVTFilter",
    "IVTFilterConfiguration",
    "EyeSelectionMode",
]
