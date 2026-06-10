"""Factories for velocity calculation and window-selection strategies."""
from __future__ import annotations

from ..config import (
    AsymmetricNeighborWindowPolicy,
    FixedSampleWindowPolicy,
    OlsenVelocityConfig,
    SampleSymmetricWindowPolicy,
    ShiftedValidWindowPolicy,
    TimeSymmetricWindowPolicy,
    TobiiWindowPolicy,
)
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
    """Create the selector described by the normalized tagged window policy."""
    policy = cfg.window_policy
    if isinstance(policy, TobiiWindowPolicy):
        sample_interval_ms = policy.sample_interval_ms
        if sample_interval_ms is None or sample_interval_ms <= 0:
            raise ValueError(
                "TobiiWindowPolicy requires a resolved sample_interval_ms > 0. "
                "Use compute_olsen_velocity/IVTPipeline to derive it from timestamps, "
                "or set sample_interval_ms explicitly."
            )
        return TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
    if isinstance(policy, AsymmetricNeighborWindowPolicy):
        return AsymmetricNeighborWindowSelector()

    rounding_strategy: WindowRoundingStrategy
    if isinstance(policy, (FixedSampleWindowPolicy, ShiftedValidWindowPolicy)) and policy.symmetric_round:
        rounding_strategy = SymmetricRoundWindowStrategy()
    else:
        rounding_strategy = StandardWindowRounding()

    if isinstance(policy, ShiftedValidWindowPolicy):
        if policy.samples is None:
            return TimeBasedShiftedValidWindowSelector(fallback_mode=policy.fallback)
        if policy.samples < 3:
            raise ValueError("samples must be >= 3 for ShiftedValidWindowPolicy.")
        return ShiftedValidWindowSelector(
            half_size=rounding_strategy.calculate_half_size(policy.samples),
            fallback_mode=policy.fallback,
        )
    if isinstance(policy, FixedSampleWindowPolicy):
        if policy.samples is None:
            raise ValueError("FixedSampleWindowPolicy requires samples >= 3 after derivation.")
        if policy.samples < 3:
            raise ValueError("samples must be >= 3 for FixedSampleWindowPolicy.")
        return FixedSampleSymmetricWindowSelector(
            half_size=rounding_strategy.calculate_half_size(policy.samples)
        )
    if isinstance(policy, SampleSymmetricWindowPolicy):
        return SampleSymmetricWindowSelector()
    if isinstance(policy, TimeSymmetricWindowPolicy):
        return TimeSymmetricWindowSelector()
    raise TypeError(f"Unsupported window policy: {policy!r}")


def _get_coordinate_rounding_strategy(mode: str) -> CoordinateRoundingStrategy:
    """Factory for coordinate rounding strategies."""
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
