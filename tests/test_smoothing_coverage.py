"""WS-D: exercise every smoothing mode end-to-end (coverage + robustness)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.processing.velocity import compute_olsen_velocity

SMOOTHING_MODES = [
    "none",
    "median",
    "moving_average",
    "median_strict",
    "moving_average_strict",
    "median_adaptive",
    "moving_average_adaptive",
]


def _frame(n: int = 120) -> pd.DataFrame:
    # a saccade in the middle so smoothing has a peak to act on
    gx = np.concatenate([np.full(50, 100.0), np.linspace(100, 150, 10), np.full(n - 60, 150.0)])
    return pd.DataFrame(
        {
            "time_ms": np.arange(n, dtype=float) * 3.333,
            "gaze_left_x_mm": gx,
            "gaze_left_y_mm": np.full(n, 200.0),
            "gaze_right_x_mm": gx,
            "gaze_right_y_mm": np.full(n, 200.0),
            "eye_left_x_mm": np.zeros(n),
            "eye_left_y_mm": np.zeros(n),
            "eye_left_z_mm": np.full(n, 600.0),
            "eye_right_x_mm": np.zeros(n),
            "eye_right_y_mm": np.zeros(n),
            "eye_right_z_mm": np.full(n, 600.0),
            "validity_left": np.zeros(n, dtype=int),
            "validity_right": np.zeros(n, dtype=int),
        }
    )


@pytest.mark.parametrize("mode", SMOOTHING_MODES)
def test_smoothing_mode_runs(mode: str) -> None:
    cfg = OlsenVelocityConfig(
        window_length_ms=20.0,
        velocity_method="olsen2d",
        smoothing_mode=mode,
        smoothing_window_samples=5,
    )
    out = compute_olsen_velocity(_frame(), cfg)
    assert "velocity_deg_per_sec" in out.columns
    velocity = pd.to_numeric(out["velocity_deg_per_sec"], errors="coerce")
    assert velocity.notna().any()


def test_moving_average_attenuates_saccade_peak() -> None:
    base = compute_olsen_velocity(
        _frame(), OlsenVelocityConfig(window_length_ms=20.0, velocity_method="olsen2d", smoothing_mode="none")
    )
    smoothed = compute_olsen_velocity(
        _frame(),
        OlsenVelocityConfig(
            window_length_ms=20.0, velocity_method="olsen2d",
            smoothing_mode="moving_average", smoothing_window_samples=9,
        ),
    )
    base_peak = pd.to_numeric(base["velocity_deg_per_sec"], errors="coerce").max()
    smooth_peak = pd.to_numeric(smoothed["velocity_deg_per_sec"], errors="coerce").max()
    assert smooth_peak < base_peak
