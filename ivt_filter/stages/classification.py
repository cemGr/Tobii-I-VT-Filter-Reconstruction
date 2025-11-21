"""IVT classification stage (Olsen Section 3.1.5)."""
from __future__ import annotations

from statistics import mean
from typing import List

from .base import IFilterStage
from ..config import IVTFilterConfiguration
from ..domain.dataset import Recording, Sample
from ..domain.events import Fixation, GazeEvent, GazeEventType, Saccade, UnknownSegment


class IVTClassificationStage(IFilterStage):
    """Label samples based on velocity threshold and build gaze events."""

    def __init__(self) -> None:
        self.events: List[GazeEvent] = []

    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        events: List[GazeEvent] = []
        current_samples: List[Sample] = []
        current_type: GazeEventType | None = None

        def flush() -> None:
            nonlocal current_samples, current_type
            if not current_samples or current_type is None:
                current_samples = []
                current_type = None
                return
            start = current_samples[0].timestamp
            end = current_samples[-1].timestamp
            if current_type == GazeEventType.FIXATION:
                xs = [s.combined_gaze_x for s in current_samples if s.combined_gaze_x is not None]
                ys = [s.combined_gaze_y for s in current_samples if s.combined_gaze_y is not None]
                fx = mean(xs) if xs else 0.0
                fy = mean(ys) if ys else 0.0
                events.append(Fixation(start, end, current_type, list(current_samples), fx, fy, len(current_samples)))
            elif current_type == GazeEventType.SACCADE:
                peak = max(s.angular_velocity_deg_per_sec or 0 for s in current_samples)
                amplitude = peak * (current_samples[-1].timestamp - current_samples[0].timestamp).total_seconds()
                events.append(Saccade(start, end, current_type, list(current_samples), peak, amplitude))
            else:
                events.append(UnknownSegment(start, end, current_type, list(current_samples)))
            current_samples = []
            current_type = None

        for sample in recording.samples:
            velocity = sample.angular_velocity_deg_per_sec
            if velocity is None or not sample.combined_valid:
                new_type = GazeEventType.UNKNOWN
            elif velocity < config.velocity_threshold_deg_per_sec:
                new_type = GazeEventType.FIXATION
            else:
                new_type = GazeEventType.SACCADE
            sample.label = new_type.name
            if current_type is None:
                current_type = new_type
                current_samples = [sample]
            elif new_type == current_type:
                current_samples.append(sample)
            else:
                flush()
                current_type = new_type
                current_samples = [sample]
        flush()
        self.events = events
        recording.events = events  # type: ignore[attr-defined]
