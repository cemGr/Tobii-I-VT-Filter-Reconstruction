"""Public facade for velocity processing.

The implementation is split by responsibility while this module keeps the historical
import surface stable for callers.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from ..config import OlsenVelocityConfig
from .velocity_factory import (
    VelocityStrategyFactory,
    WindowSelectorFactory,
    _get_coordinate_rounding_strategy,
    _get_velocity_calculation_strategy,
    make_window_selector,
)
from .velocity_input import (
    SamplingAnalyzer,
    VelocityComputationResult,
    VelocityInputArrays,
    normalize_timestamps,
    prepare_velocity_input,
)
from .velocity_samples import (
    AverageNeighborImputer,
    ComputedVelocitySample,
    FixedWindowEdgeFallbackContext,
    VelocitySampleComputer,
    _apply_eye_consistent_override,
    _calculate_dt_ms,
    _get_direction_vectors,
    apply_fixed_window_edge_fallback,
    find_single_eye_endpoints,
    visual_angle_deg,
)


def compute_olsen_velocity(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
) -> pd.DataFrame:
    """Compute Olsen-style velocity through explicit, testable pipeline stages."""
    normalized = normalize_timestamps(df, cfg)
    prepared = prepare_velocity_input(normalized, cfg)
    return VelocitySampleComputer(cfg).compute(prepared)


def compute_olsen_velocity_from_slim_tsv(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Optional[OlsenVelocityConfig] = None,
) -> pd.DataFrame:
    from ..io import read_tsv, write_tsv

    if cfg is None:
        cfg = OlsenVelocityConfig()

    df = read_tsv(input_path)
    df = compute_olsen_velocity(df, cfg)

    if output_path is not None:
        write_tsv(df, output_path)

    return df
