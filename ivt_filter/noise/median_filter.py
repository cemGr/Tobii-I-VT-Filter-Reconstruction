"""Median filter noise strategy."""
from __future__ import annotations

from statistics import median
from typing import List

from .base import INoiseFilterStrategy
from ..domain.dataset import Sample


class MedianNoiseFilterStrategy(INoiseFilterStrategy):
    """Apply median filtering to combined gaze coordinates."""

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = max(1, window_size)

    def apply(self, samples: List[Sample]) -> List[Sample]:
        radius = self.window_size // 2
        original_x = [s.combined_gaze_x for s in samples]
        original_y = [s.combined_gaze_y for s in samples]
        smoothed: List[Sample] = []
        for idx, sample in enumerate(samples):
            start = max(0, idx - radius)
            end = min(len(samples), idx + radius + 1)
            xs = [original_x[i] for i in range(start, end) if samples[i].combined_valid and original_x[i] is not None]
            ys = [original_y[i] for i in range(start, end) if samples[i].combined_valid and original_y[i] is not None]
            if xs and ys:
                sample.combined_gaze_x = median(xs)
                sample.combined_gaze_y = median(ys)
            smoothed.append(sample)
        return smoothed
