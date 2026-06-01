"""Backward-compatible imports for the canonical velocity implementation.

New code should import from :mod:`ivt_filter.processing.velocity` directly.
"""

from ..processing.velocity import (
    AverageNeighborImputer,
    ComputedVelocitySample,
    FixedWindowEdgeFallbackContext,
    SamplingAnalyzer,
    VelocityComputationResult,
    VelocityInputArrays,
    VelocitySampleComputer,
    VelocityStrategyFactory,
    WindowSelectorFactory,
    apply_fixed_window_edge_fallback,
    compute_olsen_velocity,
    compute_olsen_velocity_from_slim_tsv,
    find_single_eye_endpoints,
    make_window_selector,
    normalize_timestamps,
    prepare_velocity_input,
    visual_angle_deg,
)

__all__ = [
    "AverageNeighborImputer",
    "ComputedVelocitySample",
    "FixedWindowEdgeFallbackContext",
    "SamplingAnalyzer",
    "VelocityComputationResult",
    "VelocityInputArrays",
    "VelocitySampleComputer",
    "VelocityStrategyFactory",
    "WindowSelectorFactory",
    "apply_fixed_window_edge_fallback",
    "compute_olsen_velocity",
    "compute_olsen_velocity_from_slim_tsv",
    "find_single_eye_endpoints",
    "make_window_selector",
    "normalize_timestamps",
    "prepare_velocity_input",
    "visual_angle_deg",
]
