from __future__ import annotations

import math
from typing import Optional

from domain import IVTVelocityConfig, Recording, Sample


class IVTVelocityCalculator:
    def __init__(self, config: IVTVelocityConfig):
        self.config = config

    def estimate_dt_ms(self, recording: Recording) -> float:
        samples = recording.samples
        if len(samples) < 2:
            return self.config.window_length_ms

        deltas = []
        for i in range(1, min(len(samples), 100)):
            dt = samples[i].time_ms - samples[i - 1].time_ms
            if dt > 0:
                deltas.append(dt)
        if not deltas:
            return self.config.window_length_ms
        return sum(deltas) / len(deltas)

    def compute_velocities(self, recording: Recording) -> None:
        dt_ms = self.estimate_dt_ms(recording)
        if dt_ms <= 0:
            return
        window_samples = max(2, round(self.config.window_length_ms / dt_ms))
        half = window_samples // 2

        samples = recording.samples
        for i in range(len(samples)):
            start_idx = i - half
            end_idx = i + half
            if start_idx < 0 or end_idx >= len(samples):
                continue

            s_start = samples[start_idx]
            s_end = samples[end_idx]

            if not self._has_valid_geometry(s_start) or not self._has_valid_geometry(s_end):
                continue

            dt_s = (s_end.time_ms - s_start.time_ms) / 1000.0
            if dt_s <= 0:
                continue

            angle_deg = self._visual_angle_deg(s_start, s_end)
            if angle_deg is None:
                continue

            samples[i].velocity_deg_per_sec = angle_deg / dt_s

    def _has_valid_geometry(self, sample: Sample) -> bool:
        attrs = [
            sample.combined_gaze_x_px,
            sample.combined_gaze_y_px,
            sample.eye_x_mm,
            sample.eye_y_mm,
            sample.eye_z_mm,
        ]
        return all(value is not None for value in attrs)

    def _visual_angle_deg(self, a: Sample, b: Sample) -> Optional[float]:
        def vector_from_sample(sample: Sample) -> Optional[tuple[float, float, float]]:
            if not self._has_valid_geometry(sample):
                return None
            x = sample.combined_gaze_x_px - sample.eye_x_mm
            y = sample.combined_gaze_y_px - sample.eye_y_mm
            z = -sample.eye_z_mm
            return (x, y, z)

        v1 = vector_from_sample(a)
        v2 = vector_from_sample(b)
        if v1 is None or v2 is None:
            return None

        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm1 = math.sqrt(sum(x * x for x in v1))
        norm2 = math.sqrt(sum(x * x for x in v2))
        if norm1 == 0 or norm2 == 0:
            return None

        cos_angle = max(-1.0, min(1.0, dot_product / (norm1 * norm2)))
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)
