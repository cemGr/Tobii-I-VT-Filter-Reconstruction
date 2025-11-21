"""Data structures representing Tobii recordings.

The classes in this module intentionally mirror the vocabulary used by Olsen's
I-VT filter description (Section 3.1) and the Tobii Pro Lab TSV exports. They
carry only data and minimal helpers; the pipeline stages implement the
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class EyeData:
    """Per-eye measurements for a single sample."""

    gaze_x: float
    gaze_y: float
    eye_pos_x_3d: Optional[float] = None
    eye_pos_y_3d: Optional[float] = None
    eye_pos_z_3d: Optional[float] = None


@dataclass
class Sample:
    """Single gaze observation captured at a timestamp."""

    timestamp: datetime
    left_validity: int
    right_validity: int
    left_eye: EyeData
    right_eye: EyeData

    combined_gaze_x: Optional[float] = None
    combined_gaze_y: Optional[float] = None
    combined_valid: bool = False
    angular_velocity_deg_per_sec: Optional[float] = None
    label: Optional[str] = None

    def is_left_valid(self) -> bool:
        return self.left_validity in (0, 1)

    def is_right_valid(self) -> bool:
        return self.right_validity in (0, 1)


@dataclass
class Recording:
    """A continuous eye tracking recording comprised of samples."""

    id: str
    start_time: datetime
    end_time: datetime
    samples: List[Sample]
    dataset: Optional["EyeTrackingDataSet"] = None


@dataclass
class EyeTrackingDataSet:
    """Container for one or more recordings and metadata."""

    name: str
    sampling_rate_hz: float
    description: Optional[str] = None
    source_path: Optional[str] = None
    recordings: List[Recording] = field(default_factory=list)
