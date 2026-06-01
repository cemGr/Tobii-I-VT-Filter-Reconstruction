"""Factories for velocity calculation and window-selection strategies."""
from __future__ import annotations

from ..config import OlsenVelocityConfig
from ..strategies import (
    AsymmetricNeighborWindowSelector,
    CoordinateRoundingStrategy,
    FixedSampleSymmetricWindowSelector,
    Olsen2DApproximation,
    Ray3DAngle,
    Ray3DGazeDir,
    SampleSymmetricWindowSelector,
    ShiftedValidWindowSelector,
    TimeBasedShiftedValidWindowSelector,
    TimeSymmetricWindowSelector,
    TobiiGazeDirAngle,
    TobiiGazeVelocityWindowSelector,
    VelocityCalculationStrategy,
    WindowRoundingStrategy,
    WindowSelector,
)
from ..strategies.window_rounding import (
    StandardWindowRounding,
    SymmetricRoundWindowStrategy,
)
from ..strategies.coordinate_rounding import (
    CeilRounding,
    FloorRounding,
    NoRounding,
    RoundHalfUp,
    RoundToNearest,
)

def _get_velocity_calculation_strategy(method: str) -> VelocityCalculationStrategy:
    """Factory for velocity calculation strategies."""
    if method == "olsen2d":
        return Olsen2DApproximation()
    elif method == "ray3d":
        return Ray3DAngle()
    elif method == "ray3d_gaze_dir":
        return Ray3DGazeDir()
    elif method == "tobii_gaze_dir":
        return TobiiGazeDirAngle()
    else:
        raise ValueError(f"Unknown velocity calculation method: {method}")

def make_window_selector(cfg: OlsenVelocityConfig) -> WindowSelector:
    """
    Waehlt die passende Fenster-Strategie basierend auf der Config.
    Prioritaet:
      0) tobii_window_mode (Tobii-exakter GazeVelocityCalculator, höchste Priorität)
      1) asymmetric_neighbor_window (2-Sample asymmetrisch, Backward/Forward)
      2) fixed_window_samples (reines Sample-Fenster)
      3) sample_symmetric_window (Zeit + sample-symmetrisch)
      4) reines Zeitfenster

    Nutzt WindowRoundingStrategy zur Bestimmung der half_size.
    """
    # Tobii-exakter GazeVelocityCalculator (höchste Priorität)
    if getattr(cfg, "tobii_window_mode", False):
        sample_interval_ms = getattr(cfg, "tobii_sample_interval_ms", None)
        if sample_interval_ms is None or sample_interval_ms <= 0:
            raise ValueError(
                "tobii_window_mode requires tobii_sample_interval_ms > 0. "
                "Set e.g. tobii_sample_interval_ms=16.67 for 60 Hz."
            )
        return TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)

    # Asymmetrisches Nachbar-Fenster (2 Samples)
    if cfg.asymmetric_neighbor_window:
        return AsymmetricNeighborWindowSelector()

    # Select strategy
    rounding_strategy: WindowRoundingStrategy
    if cfg.symmetric_round_window:
        rounding_strategy = SymmetricRoundWindowStrategy()
    else:
        rounding_strategy = StandardWindowRounding()

    if cfg.shifted_valid_window:
        if cfg.fixed_window_samples is not None:
            # Sample-based shifted valid window
            n = int(cfg.fixed_window_samples)
            if n < 3:
                raise ValueError(
                    "fixed_window_samples must be >= 3 for shifted_valid_window."
                )
            half_size = rounding_strategy.calculate_half_size(n)
            return ShiftedValidWindowSelector(
                half_size=half_size, fallback_mode=cfg.shifted_valid_fallback
            )
        else:
            # Time-based shifted valid window
            return TimeBasedShiftedValidWindowSelector(
                fallback_mode=cfg.shifted_valid_fallback
            )

    if cfg.fixed_window_samples is not None:
        n = int(cfg.fixed_window_samples)
        if n < 3:
            raise ValueError("fixed_window_samples must be >= 3.")

        half_size = rounding_strategy.calculate_half_size(n)
        return FixedSampleSymmetricWindowSelector(half_size=half_size)

    if cfg.sample_symmetric_window:
        return SampleSymmetricWindowSelector()

    return TimeSymmetricWindowSelector()


def _get_coordinate_rounding_strategy(mode: str) -> CoordinateRoundingStrategy:
    """Factory für Koordinaten-Rounding-Strategien."""
    if mode == "none":
        return NoRounding()
    elif mode == "nearest":
        return RoundToNearest()
    elif mode == "halfup":
        return RoundHalfUp()
    elif mode == "floor":
        return FloorRounding()
    elif mode == "ceil":
        return CeilRounding()
    else:
        raise ValueError(f"Unknown coordinate rounding mode: {mode}")

class VelocityStrategyFactory:
    """Create visual-angle calculation strategies from configuration values."""

    @staticmethod
    def create(method: str) -> VelocityCalculationStrategy:
        return _get_velocity_calculation_strategy(method)


class WindowSelectorFactory:
    """Create velocity-window selectors from a velocity configuration."""

    @staticmethod
    def create(cfg: OlsenVelocityConfig) -> WindowSelector:
        return make_window_selector(cfg)
