"""Noise filtering strategies for the I-VT pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..domain.dataset import Sample


class INoiseFilterStrategy(ABC):
    """Strategy interface for gaze smoothing."""

    @abstractmethod
    def apply(self, samples: List[Sample]) -> List[Sample]:
        """Return samples with smoothed combined gaze positions."""
        raise NotImplementedError
