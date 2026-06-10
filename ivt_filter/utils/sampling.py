# ivt_filter/sampling.py
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


SOURCE_ROW_ID_COLUMN = "source_row_id"


KNOWN_SAMPLING_RATES = [
    30.0, 50.0, 60.0, 90.0,
    120.0, 125.0, 150.0, 200.0,
    250.0, 300.0, 500.0, 600.0,
    1000.0, 1200.0, 2000.0,
]


def _invalid_value_details(values: pd.Series, mask: np.ndarray) -> str:
    """Return a compact source-row report for invalid timestamp values."""
    positions = np.flatnonzero(mask)
    details = [f"{position}: {values.iloc[position]!r}" for position in positions[:10]]
    if len(positions) > 10:
        details.append(f"... and {len(positions) - 10} more")
    return ", ".join(details)


def coerce_finite_timestamps(values: pd.Series, *, time_col: str) -> pd.Series:
    """Convert timestamps to floats and explicitly reject unusable values."""
    numeric = pd.to_numeric(values, errors="coerce")
    non_numeric = numeric.isna().to_numpy()
    numeric_values = numeric.to_numpy(dtype=float)
    non_finite = ~non_numeric & ~np.isfinite(numeric_values)

    errors = []
    if np.any(non_numeric):
        errors.append(
            "non-numeric or missing values at source rows "
            f"[{_invalid_value_details(values, non_numeric)}]"
        )
    if np.any(non_finite):
        errors.append(
            "non-finite values at source rows "
            f"[{_invalid_value_details(values, non_finite)}]"
        )
    if errors:
        raise ValueError(
            f"Timestamp column '{time_col}' must contain only numeric, finite values: "
            + "; ".join(errors)
        )
    return numeric.astype(float)


def sort_by_time_with_source_row_id(
    df: pd.DataFrame, *, time_col: str = "time_ms"
) -> pd.DataFrame:
    """Validate timestamps, reject duplicates, and stably sort with provenance."""
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column.")

    result = df.copy()
    if SOURCE_ROW_ID_COLUMN not in result.columns:
        result[SOURCE_ROW_ID_COLUMN] = np.arange(len(result))
    result[time_col] = coerce_finite_timestamps(result[time_col], time_col=time_col)

    duplicate_mask = result[time_col].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = result.loc[duplicate_mask, time_col].drop_duplicates().tolist()
        raise ValueError(
            f"Timestamp column '{time_col}' contains duplicate timestamps; "
            f"duplicates are rejected for reproducible processing: {duplicates}"
        )

    return result.sort_values(time_col, kind="stable").reset_index(drop=True)


def positive_timestamp_deltas(times: np.ndarray) -> np.ndarray:
    """Return finite, strictly positive intervals or reject unusable samples."""
    deltas = np.diff(times)
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if deltas.size == 0:
        raise ValueError(
            "No finite, strictly positive timestamp intervals remain to estimate "
            "the sampling rate."
        )
    return deltas


def _nearest_nominal_rate(hz_measured: float) -> float:
    """
    Find the nearest typical sampling rate.
    If the deviation is too large (> 25%), return the measured rate.
    """
    best = min(KNOWN_SAMPLING_RATES, key=lambda f: abs(f - hz_measured))
    if hz_measured <= 0:
        return hz_measured
    rel_diff = abs(best - hz_measured) / hz_measured
    if rel_diff > 0.25:
        # Too far away -> don't adjust
        return hz_measured
    return best


def estimate_sampling_rate(
    df: pd.DataFrame,
    time_col: str = "time_ms",
) -> Dict[str, float]:
    """
    Estimate the sampling rate from the time_ms intervals.

    Returns a dict with:
      - median_dt_ms
      - mean_dt_ms
      - min_dt_ms
      - max_dt_ms
      - hz_from_median
      - hz_from_mean
      - nominal_hz  (nearest typical frequency, e.g. 300 instead of 333)
    """
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column.")

    times = df[time_col].to_numpy(dtype=float)
    if len(times) < 2:
        raise ValueError("Need at least two time samples to estimate sampling rate.")

    diffs = positive_timestamp_deltas(times)

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
