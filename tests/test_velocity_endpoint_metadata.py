"""Regression tests for persisted Velocity endpoint metadata."""

import math

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import IVTClassifierConfig, OlsenVelocityConfig
from ivt_filter.processing.classification import IVTClassifier
from ivt_filter.processing.velocity import compute_olsen_velocity


def _gaze_df(n: int = 8) -> pd.DataFrame:
    gaze = np.arange(n, dtype=float) * 5.0
    return pd.DataFrame(
        {
            "time_ms": np.arange(n, dtype=float) * 10.0,
            "gaze_left_x_mm": gaze,
            "gaze_left_y_mm": np.zeros(n),
            "gaze_right_x_mm": gaze,
            "gaze_right_y_mm": np.zeros(n),
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


def _metadata(result: pd.DataFrame, idx: int) -> tuple[int, int, str, bool]:
    return (
        int(result.at[idx, "velocity_first_idx"]),
        int(result.at[idx, "velocity_last_idx"]),
        str(result.at[idx, "velocity_window_selector"]),
        bool(result.at[idx, "velocity_fallback_applied"]),
    )


def test_symmetric_window_persists_actual_endpoints_for_computed_samples() -> None:
    result = compute_olsen_velocity(
        _gaze_df(), OlsenVelocityConfig(fixed_window_samples=3)
    )

    assert _metadata(result, 0) == (0, 1, "FixedSampleSymmetricWindowSelector", False)
    assert _metadata(result, 3) == (2, 4, "FixedSampleSymmetricWindowSelector", False)
    assert _metadata(result, 7) == (6, 7, "FixedSampleSymmetricWindowSelector", False)
    computed = result["velocity_deg_per_sec"].notna()
    assert result.loc[computed, "velocity_first_idx"].notna().all()
    assert result.loc[computed, "velocity_last_idx"].notna().all()
    assert result.loc[computed, "velocity_eye_used"].eq("average").all()
    assert result.loc[computed, "velocity_window_selector"].notna().all()
    assert result.loc[computed, "velocity_fallback_applied"].notna().all()


def test_asymmetric_neighbor_window_persists_backward_and_forward_endpoints() -> None:
    df = _gaze_df(4)
    df.loc[0, ["validity_left", "validity_right"]] = 999
    result = compute_olsen_velocity(
        df, OlsenVelocityConfig(asymmetric_neighbor_window=True)
    )

    assert _metadata(result, 1) == (1, 2, "AsymmetricNeighborWindowSelector", False)
    assert _metadata(result, 2) == (1, 2, "AsymmetricNeighborWindowSelector", False)


def test_shifted_valid_window_persists_shifted_endpoints() -> None:
    df = _gaze_df()
    df.loc[2, ["validity_left", "validity_right"]] = 999
    result = compute_olsen_velocity(
        df,
        OlsenVelocityConfig(
            fixed_window_samples=3,
            shifted_valid_window=True,
            shifted_valid_fallback="unclassified",
        ),
    )

    assert _metadata(result, 3) == (3, 5, "ShiftedValidWindowSelector", False)


def test_shifted_valid_window_marks_internal_shrink_fallback() -> None:
    df = _gaze_df()
    df.loc[4, ["validity_left", "validity_right"]] = 999
    result = compute_olsen_velocity(
        df,
        OlsenVelocityConfig(
            fixed_window_samples=5,
            shifted_valid_window=True,
            shifted_valid_fallback="shrink",
        ),
    )

    assert _metadata(result, 3) == (1, 5, "ShiftedValidWindowSelector", True)


def test_sample_symmetric_selector_records_time_selector_fallback() -> None:
    result = compute_olsen_velocity(
        _gaze_df(),
        OlsenVelocityConfig(window_length_ms=20.0, sample_symmetric_window=True),
    )

    assert _metadata(result, 0) == (0, 1, "TimeSymmetricWindowSelector", True)


def test_time_window_records_valid_sample_endpoint_adjustment() -> None:
    df = _gaze_df(7)
    df.loc[1, ["validity_left", "validity_right"]] = 999
    result = compute_olsen_velocity(df, OlsenVelocityConfig(window_length_ms=40.0))

    assert _metadata(result, 3) == (2, 5, "TimeSymmetricWindowSelector", True)


def test_single_eye_fallback_persists_adjusted_eye_endpoints() -> None:
    df = _gaze_df(7)
    df.loc[[0, 1, 6], "validity_left"] = 999
    df.loc[[0, 4, 5, 6], "validity_right"] = 999
    result = compute_olsen_velocity(
        df,
        OlsenVelocityConfig(
            fixed_window_samples=5,
            average_fallback_single_eye=True,
        ),
    )

    assert _metadata(result, 3) == (2, 5, "FixedSampleSymmetricWindowSelector", True)
    assert result.at[3, "velocity_eye_used"] == "left"
    assert result.at[3, "window_width_samples"] == 4


def _refinement_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_ms": [0.0, 10.0, 20.0, 30.0, 40.0],
            "smoothed_x_mm": [0.0, 2.0, 4.0, 8.0, 20.0],
            "smoothed_y_mm": [0.0] * 5,
            "eye_x_mm": [0.0] * 5,
            "eye_y_mm": [0.0] * 5,
            "eye_z_mm": [600.0] * 5,
            "combined_valid": [True] * 5,
            "velocity_deg_per_sec": [30.0] * 5,
            "window_width_samples": [pd.NA, 4, pd.NA, pd.NA, pd.NA],
        }
    )


def _ray_velocity(x1: float, x2: float, dt_ms: float) -> float:
    angle = math.degrees(math.atan2(x2, 600.0) - math.atan2(x1, 600.0))
    return round(angle / (dt_ms / 1000.0), 1)


def _classifier() -> IVTClassifier:
    return IVTClassifier(
        IVTClassifierConfig(
            enable_near_threshold_hybrid=True,
            near_threshold_band=1000.0,
            near_threshold_strategy="replace",
        )
    )


def test_refinement_prefers_persisted_actual_endpoints() -> None:
    df = _refinement_df()
    df["velocity_first_idx"] = pd.NA
    df["velocity_last_idx"] = pd.NA
    df.at[1, "velocity_first_idx"] = 1
    df.at[1, "velocity_last_idx"] = 4

    result = _classifier().classify(df)

    assert result.at[1, "velocity_refined"] == pytest.approx(_ray_velocity(2.0, 20.0, 30.0))


def test_refinement_keeps_legacy_symmetric_fallback_for_external_dataframe_edges() -> None:
    df = _refinement_df().drop(columns="window_width_samples")

    result = _classifier().classify(df)

    assert result.at[0, "velocity_refined"] == pytest.approx(_ray_velocity(0.0, 2.0, 10.0))
