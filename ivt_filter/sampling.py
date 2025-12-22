# ivt_filter/sampling.py
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


KNOWN_SAMPLING_RATES = [
    30.0, 50.0, 60.0, 90.0,
    120.0, 125.0, 150.0, 200.0,
    250.0, 300.0, 500.0, 600.0,
    1000.0, 1200.0, 2000.0,
]


def _nearest_nominal_rate(hz_measured: float) -> float:
    """
    Finde die naechste typische Sampling Rate.
    Wenn die Abweichung zu gross ist (> 25%), geben wir die gemessene zurueck.
    """
    best = min(KNOWN_SAMPLING_RATES, key=lambda f: abs(f - hz_measured))
    if hz_measured <= 0:
        return hz_measured
    rel_diff = abs(best - hz_measured) / hz_measured
    if rel_diff > 0.25:
        # zu weit weg -> nichts „schoenreden“
        return hz_measured
    return best


def estimate_sampling_rate(
    df: pd.DataFrame,
    time_col: str = "time_ms",
) -> Dict[str, float]:
    """
    Schaetze Sampling Rate aus den time_ms Abstaenden.

    Gibt ein Dict zurueck mit:
      - median_dt_ms
      - mean_dt_ms
      - min_dt_ms
      - max_dt_ms
      - hz_from_median
      - hz_from_mean
      - nominal_hz  (naechste typische Frequenz, z.B. 300 statt 333)
    """
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column.")

    times = df[time_col].to_numpy(dtype=float)
    if len(times) < 2:
        raise ValueError("Need at least two time samples to estimate sampling rate.")

    diffs = np.diff(times)
    # nur positive Abstaende verwenden (falls 0 oder Rueckspruenge vorkommen)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise ValueError("No positive time differences found to estimate sampling rate.")

    median_dt_ms = float(np.median(diffs))
    mean_dt_ms = float(np.mean(diffs))
    min_dt_ms = float(np.min(diffs))
    max_dt_ms = float(np.max(diffs))

    hz_from_median = 1000.0 / median_dt_ms
    hz_from_mean = 1000.0 / mean_dt_ms

    nominal_hz = _nearest_nominal_rate(hz_from_median)

    return {
        "median_dt_ms": median_dt_ms,
        "mean_dt_ms": mean_dt_ms,
        "min_dt_ms": min_dt_ms,
        "max_dt_ms": max_dt_ms,
        "hz_from_median": hz_from_median,
        "hz_from_mean": hz_from_mean,
        "nominal_hz": nominal_hz,
    }
