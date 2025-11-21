"""Gap fill-in stage (Olsen Section 3.1.1)."""
from __future__ import annotations

from datetime import timedelta
from typing import List, Tuple

from .base import IFilterStage
from ..config import IVTFilterConfiguration
from ..domain.dataset import Recording, Sample


def _valid(sample: Sample, eye: str) -> bool:
    return sample.is_left_valid() if eye == "left" else sample.is_right_valid()


def _interpolate(start: Sample, end: Sample, target: Sample, eye: str) -> Tuple[float, float]:
    total = (end.timestamp - start.timestamp).total_seconds()
    if total == 0:
        return start.left_eye.gaze_x, start.left_eye.gaze_y
    fraction = (target.timestamp - start.timestamp).total_seconds() / total
    eye_start = start.left_eye if eye == "left" else start.right_eye
    eye_end = end.left_eye if eye == "left" else end.right_eye
    x = eye_start.gaze_x + (eye_end.gaze_x - eye_start.gaze_x) * fraction
    y = eye_start.gaze_y + (eye_end.gaze_y - eye_start.gaze_y) * fraction
    return x, y


class GapFillingStage(IFilterStage):
    """Interpolates short gaps so downstream stages operate on continuous data."""

    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        threshold = timedelta(milliseconds=config.max_gap_length_ms)
        for eye in ("left", "right"):
            samples: List[Sample] = recording.samples
            last_valid_idx = None
            for idx, sample in enumerate(samples):
                if _valid(sample, eye):
                    if last_valid_idx is None:
                        last_valid_idx = idx
                        continue
                    gap_indices = list(range(last_valid_idx + 1, idx))
                    if not gap_indices:
                        last_valid_idx = idx
                        continue
                    start = samples[last_valid_idx]
                    end = sample
                    gap_duration = end.timestamp - start.timestamp
                    if gap_duration <= threshold:
                        for gi in gap_indices:
                            interp_x, interp_y = _interpolate(start, end, samples[gi], eye)
                            if eye == "left":
                                samples[gi].left_eye.gaze_x = interp_x
                                samples[gi].left_eye.gaze_y = interp_y
                                samples[gi].left_validity = 0
                            else:
                                samples[gi].right_eye.gaze_x = interp_x
                                samples[gi].right_eye.gaze_y = interp_y
                                samples[gi].right_validity = 0
                    last_valid_idx = idx
