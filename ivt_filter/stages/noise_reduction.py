"""Noise reduction stage (Olsen Section 3.1.3)."""
from __future__ import annotations

from .base import IFilterStage
from ..config import IVTFilterConfiguration
from ..domain.dataset import Recording


class NoiseReductionStage(IFilterStage):
    """Apply the configured smoothing strategy to combined gaze coordinates."""

    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        strategy = config.noise_filter_strategy
        strategy.apply(recording.samples)
