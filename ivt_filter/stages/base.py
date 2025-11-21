"""Base class for each step in the I-VT pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import IVTFilterConfiguration
from ..domain.dataset import Recording


class IFilterStage(ABC):
    """Abstract processing stage.

    Each concrete implementation is responsible for a single function from
    Olsen's I-VT pipeline.
    """

    @abstractmethod
    def process(self, recording: Recording, config: IVTFilterConfiguration) -> None:
        """Mutate the recording in-place according to the stage's behaviour."""
        raise NotImplementedError
