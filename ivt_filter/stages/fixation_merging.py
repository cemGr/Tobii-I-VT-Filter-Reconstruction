"""Merge adjacent fixations stage (Olsen Section 3.1.6)."""
from __future__ import annotations

import math
from statistics import mean
from typing import List

from .base import IFilterStage
from ..config import IVTFilterConfiguration
from ..domain.dataset import Recording
from ..domain.events import Fixation, GazeEvent, GazeEventType


def _angle_between(f1: Fixation, f2: Fixation) -> float:
    dx = f2.position_x - f1.position_x
    dy = f2.position_y - f1.position_y
    return math.degrees(math.atan2(math.hypot(dx, dy), 1.0))


class FixationMergingStage(IFilterStage):
    """Merge neighbouring fixations separated by short gaps and angles."""

    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        if not hasattr(recording, "events"):
            return
        merged: List[GazeEvent] = []
        events: List[GazeEvent] = list(getattr(recording, "events"))
        idx = 0
        while idx < len(events):
            event = events[idx]
            if event.event_type != GazeEventType.FIXATION or idx == len(events) - 1:
                merged.append(event)
                idx += 1
                continue
            nxt = events[idx + 1]
            if nxt.event_type != GazeEventType.FIXATION:
                merged.append(event)
                idx += 1
                continue
            gap = (nxt.start_time - event.end_time).total_seconds() * 1000
            angle = _angle_between(event, nxt)
            if gap <= config.max_time_between_fixations_ms and angle <= config.max_angle_between_fixations_deg:
                samples = event.samples + nxt.samples
                fx = mean([s.combined_gaze_x for s in samples if s.combined_gaze_x is not None])
                fy = mean([s.combined_gaze_y for s in samples if s.combined_gaze_y is not None])
                merged_fixation = Fixation(
                    start_time=event.start_time,
                    end_time=nxt.end_time,
                    event_type=GazeEventType.FIXATION,
                    samples=samples,
                    position_x=fx,
                    position_y=fy,
                    sample_count=len(samples),
                )
                merged.append(merged_fixation)
                idx += 2
            else:
                merged.append(event)
                idx += 1
        recording.events = merged  # type: ignore[attr-defined]
