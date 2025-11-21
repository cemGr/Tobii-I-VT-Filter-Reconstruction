"""High level IVT filter orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Protocol

from .config import IVTFilterConfiguration
from .domain.dataset import Recording
from .domain.events import GazeEvent
from .stages import (
    GapFillingStage,
    EyeSelectionStage,
    NoiseReductionStage,
    VelocityComputationStage,
    IVTClassificationStage,
    FixationMergingStage,
    ShortFixationDiscardStage,
    IFilterStage,
)


class IIVTFilter(Protocol):
    """Protocol for running the I-VT pipeline."""

    def run(self, recording: Recording, config: IVTFilterConfiguration) -> "FilterResult":
        ...


@dataclass
class FilterResult:
    """Wrapper holding the processed recording and derived events."""

    recording: Recording
    events: List[GazeEvent]
    created_at: datetime


class IVTFilterEngine(IIVTFilter):
    """Pipeline composed of seven dedicated stages mirroring Olsen's algorithm."""

    def __init__(self, stages: List[IFilterStage] | None = None) -> None:
        self.stages: List[IFilterStage] = stages or [
            GapFillingStage(),
            EyeSelectionStage(),
            NoiseReductionStage(),
            VelocityComputationStage(),
            IVTClassificationStage(),
            FixationMergingStage(),
            ShortFixationDiscardStage(),
        ]

    def run(self, recording: Recording, config: IVTFilterConfiguration) -> FilterResult:
        for stage in self.stages:
            stage.process(recording, config)
        events: List[GazeEvent] = getattr(recording, "events", [])  # type: ignore[arg-type]
        return FilterResult(recording=recording, events=events, created_at=datetime.now(timezone.utc))
