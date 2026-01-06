# ivt_filter/preprocessing/noise_reduction.py
"""Noise reduction: applies smoothing to combined gaze coordinates."""

from __future__ import annotations

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
    """Factory für Smoothing-Strategien."""
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
    Optionales Smoothing auf combined_x_mm / combined_y_mm.

    - Wir smoothing nur dort, wo combined_valid == True.
    - Rolling-Window in Sample-Domaene (nicht Zeit).
    - Erzeugt Spalten:
        - smoothed_x_mm
        - smoothed_y_mm
        
    Nutzt SmoothingStrategy Pattern für verschiedene Methoden.
    """
    df = df.copy()

    valid_mask = df["combined_valid"]
    x_series = df["combined_x_mm"]
    y_series = df["combined_y_mm"]

    # Select strategy basierend auf Config
    strategy = _get_smoothing_strategy(
        cfg.smoothing_mode, 
        cfg.smoothing_window_samples,
        cfg.smoothing_min_samples,
        cfg.smoothing_expansion_radius
    )

    # Anwende Smoothing
    x_smooth = strategy.smooth(x_series, valid_mask)
    y_smooth = strategy.smooth(y_series, valid_mask)

    df["smoothed_x_mm"] = x_smooth
    df["smoothed_y_mm"] = y_smooth
    return df
