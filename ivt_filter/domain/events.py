"""Output gaze events produced by the I-VT classifier."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import List

from .dataset import Sample


class GazeEventType(Enum):
    """Enumeration of high-level gaze events produced by the filter."""

    FIXATION = auto()
    SACCADE = auto()
    UNKNOWN = auto()


@dataclass
class GazeEvent:
    """Base class capturing timing and sample membership."""

    start_time: datetime
    end_time: datetime
    event_type: GazeEventType
    samples: List[Sample]

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time


@dataclass
class Fixation(GazeEvent):
    """Stable gaze period where velocity falls below the I-VT threshold."""

    position_x: float
    position_y: float
    sample_count: int


@dataclass
class Saccade(GazeEvent):
    """Rapid movement segment."""

    peak_velocity: float
    amplitude_deg: float


@dataclass
class UnknownSegment(GazeEvent):
    """Represents gaps or samples that could not be classified."""

    pass
