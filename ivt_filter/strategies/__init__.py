"""Velocity calculation strategies and windowing strategies."""

from .velocity_calculation import (
    VelocityCalculationStrategy,
    VelocityContext,
    Olsen2DApproximation,
    Ray3DAngle,
    Ray3DGazeDir,
    TobiiGazeDirAngle,
)
from .windowing import (
    WindowSelector,
    TimeSymmetricWindowSelector,
    SampleSymmetricWindowSelector,
    FixedSampleSymmetricWindowSelector,
    AsymmetricNeighborWindowSelector,
    ShiftedValidWindowSelector,
    TimeBasedShiftedValidWindowSelector,
    TobiiGazeVelocityWindowSelector,
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
    "TobiiGazeDirAngle",
    "WindowSelector",
    "TimeSymmetricWindowSelector",
    "SampleSymmetricWindowSelector",
    "FixedSampleSymmetricWindowSelector",
    "AsymmetricNeighborWindowSelector",
    "ShiftedValidWindowSelector",
    "TimeBasedShiftedValidWindowSelector",
    "TobiiGazeVelocityWindowSelector",
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
