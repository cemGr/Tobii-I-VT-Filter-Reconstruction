"""Filter configuration and enums."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .noise.base import INoiseFilterStrategy
from .noise.no_noise import NoNoiseFilterStrategy


class EyeSelectionMode(Enum):
    """Available strategies for combining binocular gaze data."""

    LEFT = "left"
    RIGHT = "right"
    AVERAGE = "average"
    STRICT_AVERAGE = "strict_average"


@dataclass
class IVTFilterConfiguration:
    """Configuration parameters for the I-VT pipeline."""

    max_gap_length_ms: int = 75
    eye_selection_mode: EyeSelectionMode = EyeSelectionMode.AVERAGE
    velocity_threshold_deg_per_sec: float = 30.0
    window_length_ms: int = 20
    max_time_between_fixations_ms: int = 75
    max_angle_between_fixations_deg: float = 0.5
    minimum_fixation_duration_ms: int = 60
    noise_filter_strategy: INoiseFilterStrategy = NoNoiseFilterStrategy()
