"""Lightweight DataFrame stage contracts for the I-VT pipeline."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

TIME_MS = "time_ms"
GAZE_LEFT_X_MM = "gaze_left_x_mm"
GAZE_LEFT_Y_MM = "gaze_left_y_mm"
GAZE_RIGHT_X_MM = "gaze_right_x_mm"
GAZE_RIGHT_Y_MM = "gaze_right_y_mm"
VALIDITY_LEFT = "validity_left"
VALIDITY_RIGHT = "validity_right"
COMBINED_X_MM = "combined_x_mm"
COMBINED_Y_MM = "combined_y_mm"
COMBINED_VALID = "combined_valid"
SMOOTHED_X_MM = "smoothed_x_mm"
SMOOTHED_Y_MM = "smoothed_y_mm"
VELOCITY_DEG_PER_SEC = "velocity_deg_per_sec"
IVT_SAMPLE_TYPE = "ivt_sample_type"
IVT_EVENT_TYPE = "ivt_event_type"
IVT_EVENT_INDEX = "ivt_event_index"

RAW_GAZE_COLUMNS = (
    TIME_MS,
    GAZE_LEFT_X_MM,
    GAZE_LEFT_Y_MM,
    GAZE_RIGHT_X_MM,
    GAZE_RIGHT_Y_MM,
    VALIDITY_LEFT,
    VALIDITY_RIGHT,
)
PREPROCESSED_COLUMNS = RAW_GAZE_COLUMNS + (
    COMBINED_X_MM,
    COMBINED_Y_MM,
    COMBINED_VALID,
    SMOOTHED_X_MM,
    SMOOTHED_Y_MM,
)
VELOCITY_COLUMNS = (VELOCITY_DEG_PER_SEC,)
CLASSIFIED_COLUMNS = (IVT_SAMPLE_TYPE, IVT_EVENT_TYPE, IVT_EVENT_INDEX)


def require_columns(df: pd.DataFrame, required: Iterable[str], *, stage: str) -> None:
    """Raise a clear error when a DataFrame omits stage-required columns."""
    missing = sorted(set(required).difference(df.columns))
    if missing:
        columns = ", ".join(repr(column) for column in missing)
        raise ValueError(f"{stage} DataFrame is missing required columns: {columns}")


def validate_raw_gaze_frame(df: pd.DataFrame) -> None:
    """Validate normalized raw gaze input before preprocessing."""
    require_columns(df, RAW_GAZE_COLUMNS, stage="raw gaze")


def validate_preprocessed_frame(df: pd.DataFrame) -> None:
    """Validate gaze input after eye selection and smoothing."""
    require_columns(df, PREPROCESSED_COLUMNS, stage="preprocessed gaze")


def validate_velocity_frame(df: pd.DataFrame) -> None:
    """Validate classifier input after velocity computation."""
    require_columns(df, VELOCITY_COLUMNS, stage="velocity")


def validate_classified_frame(df: pd.DataFrame) -> None:
    """Validate classifier output after event reconstruction."""
    require_columns(df, CLASSIFIED_COLUMNS, stage="classified")
