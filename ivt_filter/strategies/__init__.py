"""Velocity calculation strategies and windowing strategies."""

from .velocity_calculation import (
    VelocityCalculationStrategy,
    VelocityContext,
    Olsen2DApproximation,
    Ray3DAngle,
    Ray3DGazeDir,
)
from .windowing import (
    WindowSelector,
    TimeSymmetricWindowSelector,
    SampleSymmetricWindowSelector,
    FixedSampleSymmetricWindowSelector,
    AsymmetricNeighborWindowSelector,
    ShiftedValidWindowSelector,
    TimeBasedShiftedValidWindowSelector,
)
from .coordinate_rounding import CoordinateRoundingStrategy
from .window_rounding import WindowRoundingStrategy
from .smoothing_strategy import (
    SmoothingStrategy,
    NoSmoothing,
    MedianSmoothing,
    MovingAverageSmoothing,
    MedianSmoothingStrict,
    MovingAverageSmoothingStrict,
    MedianSmoothingAdaptive,
    MovingAverageSmoothingAdaptive,
)

__all__ = [
    "VelocityCalculationStrategy",
    "VelocityContext",
    "Olsen2DApproximation",
    "Ray3DAngle",
    "Ray3DGazeDir",
    "WindowSelector",
    "TimeSymmetricWindowSelector",
    "SampleSymmetricWindowSelector",
    "FixedSampleSymmetricWindowSelector",
    "AsymmetricNeighborWindowSelector",
    "ShiftedValidWindowSelector",
    "TimeBasedShiftedValidWindowSelector",
    "CoordinateRoundingStrategy",
    "WindowRoundingStrategy",
    "SmoothingStrategy",
    "NoSmoothing",
    "MedianSmoothing",
    "MovingAverageSmoothing",
    "MedianSmoothingStrict",
    "MovingAverageSmoothingStrict",
    "MedianSmoothingAdaptive",
    "MovingAverageSmoothingAdaptive",
]
