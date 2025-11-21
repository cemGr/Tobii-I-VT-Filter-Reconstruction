"""Discard short fixations stage (Olsen Section 3.1.7)."""
from __future__ import annotations

from datetime import timedelta
from typing import List

from .base import IFilterStage
from ..config import IVTFilterConfiguration
from ..domain.dataset import Recording
from ..domain.events import Fixation, GazeEvent, GazeEventType, UnknownSegment


class ShortFixationDiscardStage(IFilterStage):
    """Remove fixations shorter than the configured minimum duration."""

    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        if not hasattr(recording, "events"):
            return
        minimum = timedelta(milliseconds=config.minimum_fixation_duration_ms)
        filtered: List[GazeEvent] = []
        for event in getattr(recording, "events"):
            if event.event_type == GazeEventType.FIXATION and event.duration < minimum:
                # Reclassify contained samples as UNKNOWN
                for sample in event.samples:
                    sample.label = GazeEventType.UNKNOWN.name
                # Merge into neighbouring unknown segments when possible
                if filtered and isinstance(filtered[-1], UnknownSegment):
                    filtered[-1].samples.extend(event.samples)
                    filtered[-1].end_time = event.end_time
                else:
                    filtered.append(UnknownSegment(event.start_time, event.end_time, GazeEventType.UNKNOWN, list(event.samples)))
                continue
            filtered.append(event)
        recording.events = filtered  # type: ignore[attr-defined]
