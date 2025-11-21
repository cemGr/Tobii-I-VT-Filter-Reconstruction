"""Velocity computation stage (Olsen Section 3.1.4)."""
from __future__ import annotations

import math
from statistics import mean
from typing import List, Optional

from .base import IFilterStage
from ..config import IVTFilterConfiguration
from ..domain.dataset import Recording, Sample


def _sampling_interval_seconds(samples: List[Sample]) -> float:
    if len(samples) < 2:
        return 0.0
    deltas = []
    for a, b in zip(samples, samples[1:100]):
        deltas.append((b.timestamp - a.timestamp).total_seconds())
    return mean(deltas) if deltas else 0.0


def _visual_angle_deg(p1: Sample, p2: Sample) -> Optional[float]:
    if p1.combined_gaze_x is None or p2.combined_gaze_x is None:
        return None
    dx = p2.combined_gaze_x - p1.combined_gaze_x
    dy = p2.combined_gaze_y - p1.combined_gaze_y
    # If eye position available, approximate using small angle formula.
    z = None
    if p1.left_eye.eye_pos_z_3d is not None:
        z = p1.left_eye.eye_pos_z_3d
    elif p1.right_eye.eye_pos_z_3d is not None:
        z = p1.right_eye.eye_pos_z_3d
    if z:
        dist = math.hypot(dx, dy)
        return math.degrees(math.atan2(dist, z))
    return math.degrees(math.atan2(math.hypot(dx, dy), 1.0))


class VelocityComputationStage(IFilterStage):
    """Compute angular velocity from combined gaze coordinates."""

    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        samples = recording.samples
        if not samples:
            return
        interval = _sampling_interval_seconds(samples)
        if interval == 0:
            interval = config.window_length_ms / 1000.0
        window_ms = max(config.window_length_ms, 1)
        half_window = window_ms / 2000.0
        for idx, sample in enumerate(samples):
            center_time = sample.timestamp
            window_samples = [s for s in samples if abs((s.timestamp - center_time).total_seconds()) <= half_window]
            if len(window_samples) < 2:
                continue
            first = window_samples[0]
            last = window_samples[-1]
            angle = _visual_angle_deg(first, last)
            if angle is None:
                continue
            dt = (last.timestamp - first.timestamp).total_seconds()
            if dt == 0:
                dt = interval
            sample.angular_velocity_deg_per_sec = angle / dt
