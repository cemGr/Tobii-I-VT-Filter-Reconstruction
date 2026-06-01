"""Input normalization and preparatory transformations for velocity computation."""
from __future__ import annotations

import dataclasses
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from ..config import OlsenVelocityConfig
from ..domain.schema import validate_preprocessed_frame, validate_raw_gaze_frame
from ..utils.sampling import (
    coerce_finite_timestamps,
    positive_timestamp_deltas,
    sort_by_time_with_source_row_id,
)
from ..preprocessing import (
    apply_tobii_eye_offset_interpolation,
    gap_fill_gaze,
    prepare_combined_columns,
    smooth_combined_gaze,
)

logger = logging.getLogger("ivt_filter.processing.velocity")

@dataclasses.dataclass(frozen=True)
class VelocityInputArrays:
    """NumPy views used while computing velocity samples."""

    times: np.ndarray
    combined_x: np.ndarray
    combined_y: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    valid: np.ndarray
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, valid: Optional[np.ndarray] = None
    ) -> "VelocityInputArrays":
        left_valid = df["left_eye_valid"].to_numpy()
        right_valid = df["right_eye_valid"].to_numpy()
        if valid is None:
            valid = df["combined_valid"].to_numpy()
        return cls(
            times=df["time_ms"].to_numpy(),
            combined_x=df["smoothed_x_mm"].to_numpy(),
            combined_y=df["smoothed_y_mm"].to_numpy(),
            left_valid=left_valid,
            right_valid=right_valid,
            valid=valid,
            left_x=df["gaze_left_x_mm"].to_numpy(),
            left_y=df["gaze_left_y_mm"].to_numpy(),
            right_x=df["gaze_right_x_mm"].to_numpy(),
            right_y=df["gaze_right_y_mm"].to_numpy(),
        )


@dataclasses.dataclass(frozen=True)
class VelocityComputationResult:
    """Structured result of sampling analysis, including the effective config."""

    dt_ms: Optional[float]
    hz_measured: Optional[float]
    config: OlsenVelocityConfig


class SamplingAnalyzer:
    """Analyze timestamp spacing and derive an optional fixed sample window."""

    NOMINAL_RATES = (30.0, 50.0, 60.0, 120.0, 150.0, 250.0, 300.0, 500.0, 600.0, 1000.0)

    def analyze(
        self, times: np.ndarray, cfg: OlsenVelocityConfig
    ) -> VelocityComputationResult:
        if cfg.sampling_rate_method == "first_100":
            sample_count = min(100, len(times) - 1)
            deltas = positive_timestamp_deltas(times[: sample_count + 1])
            method_desc = f"first {sample_count} samples"
        else:
            deltas = positive_timestamp_deltas(times)
            method_desc = "all samples"
        dt_ms = float(
            np.median(deltas)
            if cfg.dt_calculation_method == "median"
            else np.mean(deltas)
        )
        hz_measured = 1000.0 / dt_ms if dt_ms > 0 else float("nan")
        logger.info(
            "[Sampling] %s dt = %.3f ms -> measured ~%.1f Hz (using %s)",
            cfg.dt_calculation_method,
            dt_ms,
            hz_measured,
            method_desc,
        )
        if math.isfinite(hz_measured):
            nearest = min(self.NOMINAL_RATES, key=lambda rate: abs(rate - hz_measured))
            logger.info("[Sampling] nearest nominal rate: %.1f Hz", nearest)
        should_convert = cfg.auto_fixed_window_from_ms or cfg.symmetric_round_window
        if should_convert and cfg.fixed_window_samples is None and dt_ms > 0:
            intervals = max(1, int(round(cfg.window_length_ms / dt_ms)))
            samples = max(3, intervals + 1)
            if samples % 2 == 0:
                samples += 1
            effective_ms = (samples - 1) * dt_ms
            cfg = dataclasses.replace(cfg, fixed_window_samples=samples)
            logger.info(
                "[Window] auto sample window: %s samples total (~%.1f pro Seite um "
                "das Zentrum, effektive Spannweite ~%.2f ms)",
                samples,
                (samples - 1) / 2.0,
                effective_ms,
            )
        return VelocityComputationResult(dt_ms, hz_measured, cfg)


def normalize_timestamps(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """Copy, normalize and sort the configured timestamp column to milliseconds."""
    result = df.copy()
    time_col = getattr(cfg, "time_column", "time_ms")
    time_unit = getattr(cfg, "time_unit", "ms")
    if time_col not in result.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column")
    divisor = {"ms": 1.0, "us": 1000.0, "ns": 1_000_000.0}.get(time_unit, 1.0)
    timestamps = coerce_finite_timestamps(result[time_col], time_col=time_col)
    result["time_ms"] = timestamps / divisor
    return sort_by_time_with_source_row_id(result)


def prepare_velocity_input(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """Run gaze preprocessing required before velocity sample computation."""
    validate_raw_gaze_frame(df)
    result = gap_fill_gaze(df, cfg)
    if getattr(cfg, "tobii_eye_offset_interpolation", False):
        result = apply_tobii_eye_offset_interpolation(result, cfg)
    result = prepare_combined_columns(result, cfg)
    result = smooth_combined_gaze(result, cfg)
    validate_preprocessed_frame(result)
    return result
