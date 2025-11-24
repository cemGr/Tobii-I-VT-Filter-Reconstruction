"""Configuration dataclasses for IVT processing."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OlsenVelocityConfig:
    """Configuration for Olsen-style velocity computation."""

    window_length_ms: float = 20.0
    eye_mode: Literal["left", "right", "average"] = "average"
    max_validity: int = 1
    min_dt_ms: float = 0.1
    default_eye_distance_mm: float = 600.0


@dataclass(frozen=True)
class IVTClassifierConfig:
    """Configuration for the I-VT velocity-threshold classifier."""

    velocity_threshold_deg_per_sec: float = 30.0
