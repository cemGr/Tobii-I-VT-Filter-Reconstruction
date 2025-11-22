from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EyeData:
    gaze_x_px: float
    gaze_y_px: float
    eye_pos_x_mm: Optional[float]
    eye_pos_y_mm: Optional[float]
    eye_pos_z_mm: Optional[float]


@dataclass
class Sample:
    time_ms: float
    left: EyeData
    right: EyeData
    validity_left: int
    validity_right: int
    combined_gaze_x_px: Optional[float] = None
    combined_gaze_y_px: Optional[float] = None
    eye_x_mm: Optional[float] = None
    eye_y_mm: Optional[float] = None
    eye_z_mm: Optional[float] = None
    velocity_deg_per_sec: Optional[float] = None


@dataclass
class Recording:
    id: str
    samples: List[Sample]


@dataclass
class IVTVelocityConfig:
    window_length_ms: float
