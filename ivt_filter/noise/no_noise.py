"""Noise strategy that leaves gaze untouched."""
from __future__ import annotations

from typing import List

from .base import INoiseFilterStrategy
from ..domain.dataset import Sample


class NoNoiseFilterStrategy(INoiseFilterStrategy):
    """Pass-through strategy useful for testing and strict Olsen reproduction."""

    def apply(self, samples: List[Sample]) -> List[Sample]:
        return samples
