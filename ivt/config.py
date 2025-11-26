"""Configuration dataclasses for IVT processing."""
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class OlsenVelocityConfig:
    """Configuration for Olsen-style velocity computation."""

    window_length_ms: float = 20.0
    eye_mode: Literal["left", "right", "average"] = "average"
    max_validity: int = 1
    min_dt_ms: float = 0.1
    gap_fill: bool = False
    max_gap_length_ms: float = 75.0

    # Use mm gaze as primary representation
    use_gaze_mm: bool = True

    # Optional pixel size in mm (only needed if we ever fall back to px)
    pixel_size_mm: Optional[float] = None


@dataclass(frozen=True)
class IVTClassifierConfig:
    """Configuration for the I-VT velocity-threshold classifier."""

    velocity_threshold_deg_per_sec: float = 30.0
