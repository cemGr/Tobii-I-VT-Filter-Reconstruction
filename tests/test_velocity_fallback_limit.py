"""WS-B(b): the fixed-window edge-fallback search limit is configurable.

The limit was hard-coded to 50 samples. Making it a config field (default 50,
preserving behaviour) lets the EOF-gap edge case be governed explicitly and
keeps the reconstruction assumption auditable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.processing.velocity_samples import (
    FixedWindowEdgeFallbackContext,
    apply_fixed_window_edge_fallback,
)
from ivt_filter.strategies.windowing import FixedSampleSymmetricWindowSelector

HALF = 2
TARGET = 10
DISTANCE = 30  # the only valid velocity sits 30 samples away from the target


def _frame_with_far_valid_velocity(n: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    """Target sample has a NaN velocity and an invalid window edge; the nearest
    fillable velocity is DISTANCE samples away."""
    df = pd.DataFrame(
        {
            "velocity_deg_per_sec": np.full(n, np.nan),
            "velocity_fallback_applied": [False] * n,
        }
    )
    df.loc[TARGET + DISTANCE, "velocity_deg_per_sec"] = 42.0
    valid = np.ones(n, dtype=bool)
    valid[TARGET - HALF] = False  # makes the target's window edge invalid -> fallback triggers
    return df, valid


def _run(limit: int) -> pd.DataFrame:
    df, valid = _frame_with_far_valid_velocity()
    cfg = OlsenVelocityConfig(
        fixed_window_edge_fallback=True,
        fixed_window_edge_fallback_limit=limit,
    )
    context = FixedWindowEdgeFallbackContext(
        cfg=cfg, selector=FixedSampleSymmetricWindowSelector(half_size=HALF), valid=valid
    )
    apply_fixed_window_edge_fallback(df, context)
    return df


def test_default_limit_50_fills_far_sample() -> None:
    df = _run(limit=50)  # 30 < 50 -> found
    assert df.loc[TARGET, "velocity_deg_per_sec"] == 42.0
    assert bool(df.loc[TARGET, "velocity_fallback_applied"]) is True


def test_small_limit_leaves_target_nan() -> None:
    df = _run(limit=10)  # 30 > 10 -> not found
    assert pd.isna(df.loc[TARGET, "velocity_deg_per_sec"])
    assert bool(df.loc[TARGET, "velocity_fallback_applied"]) is False


def test_limit_just_below_distance_leaves_nan() -> None:
    df = _run(limit=DISTANCE)  # range(1, 30) excludes offset 30 -> not found
    assert pd.isna(df.loc[TARGET, "velocity_deg_per_sec"])


def test_limit_validation_rejects_zero() -> None:
    with pytest.raises(ValueError):
        OlsenVelocityConfig(fixed_window_edge_fallback_limit=0)
