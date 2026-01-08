"""Utility functions for window management and sampling rate detection."""

from .window_utils import (
    samples_to_milliseconds,
    milliseconds_to_samples,
    detect_sampling_rate,
)
from .sampling import (
    KNOWN_SAMPLING_RATES,
)

__all__ = [
    'samples_to_milliseconds',
    'milliseconds_to_samples',
    'detect_sampling_rate',
    'KNOWN_SAMPLING_RATES',
]
