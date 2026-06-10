# ivt_filter/preprocessing/noise_reduction.py
"""Noise reduction: applies smoothing to combined gaze coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import OlsenVelocityConfig
from ..strategies.smoothing_strategy import (
    SmoothingStrategy,
    NoSmoothing,
    MedianSmoothing,
    MovingAverageSmoothing,
    MedianSmoothingStrict,
    MovingAverageSmoothingStrict,
    MedianSmoothingAdaptive,
    MovingAverageSmoothingAdaptive,
)


def _get_smoothing_strategy(
    mode: str, 
    window_samples: int,
    min_samples: int = 1,
    expansion_radius: int = 0
) -> SmoothingStrategy:
    """Factory for smoothing strategies."""
    if mode == "none":
        return NoSmoothing(window_samples)
    elif mode == "median":
        return MedianSmoothing(window_samples)
    elif mode == "moving_average":
        return MovingAverageSmoothing(window_samples)
    elif mode == "median_strict":
        return MedianSmoothingStrict(window_samples)
    elif mode == "moving_average_strict":
        return MovingAverageSmoothingStrict(window_samples)
    elif mode == "median_adaptive":
        return MedianSmoothingAdaptive(window_samples, min_samples, expansion_radius)
    elif mode == "moving_average_adaptive":
        return MovingAverageSmoothingAdaptive(window_samples, min_samples, expansion_radius)
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")


def smooth_combined_gaze(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """
    Optional smoothing on combined_x_mm / combined_y_mm.

    - We smooth only where combined_valid == True.
    - Rolling window in the sample domain (not time).
    - Creates columns:
        - smoothed_x_mm
        - smoothed_y_mm

    Uses the SmoothingStrategy pattern for the different methods.
    """
    df = df.copy()

    valid_mask = df["combined_valid"]
    x_series = df["combined_x_mm"]
    y_series = df["combined_y_mm"]

    # Select strategy based on the config
    strategy = _get_smoothing_strategy(
        cfg.smoothing_mode, 
        cfg.smoothing_window_samples,
        cfg.smoothing_min_samples,
        cfg.smoothing_expansion_radius
    )

    # Apply smoothing
    x_smooth = strategy.smooth(x_series, valid_mask)
    y_smooth = strategy.smooth(y_series, valid_mask)

    df["smoothed_x_mm"] = x_smooth
    df["smoothed_y_mm"] = y_smooth

    # Smooth direction vectors (tobii_gaze_dir velocity method reads these)
    for eye, valid_col in (("left", "left_eye_valid"), ("right", "right_eye_valid")):
        x_col = f"gaze_dir_{eye}_x"
        if x_col not in df.columns:
            continue
        eye_mask = df[valid_col]
        smoothed = {}
        for axis in ("x", "y", "z"):
            col = f"gaze_dir_{eye}_{axis}"
            smoothed[axis] = strategy.smooth(df[col], eye_mask).to_numpy(dtype=float)

        # Renormalize to unit length
        sx, sy, sz = smoothed["x"], smoothed["y"], smoothed["z"]
        norms = np.sqrt(sx**2 + sy**2 + sz**2)
        norms[norms == 0] = np.nan
        df[f"smoothed_gaze_dir_{eye}_x"] = sx / norms
        df[f"smoothed_gaze_dir_{eye}_y"] = sy / norms
        df[f"smoothed_gaze_dir_{eye}_z"] = sz / norms

    return df
