"""WS-D: robustness edge cases and simple_api coverage.

Degenerate inputs (empty, single/two samples, all-invalid, all-NaN gaze, large
gaps at the recording boundaries) must not raise and must yield sane labels. The
simple_api tests exercise the user-facing entry points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import IVTClassifierConfig, OlsenVelocityConfig, PipelineConfig
from ivt_filter.io.io import write_tsv
from ivt_filter.io.pipeline import IVTPipeline
from ivt_filter.simple_api import (
    get_statistics,
    print_statistics,
    process_eye_tracking,
)

_SCHEMA_COLS = [
    "time_ms",
    "gaze_left_x_mm", "gaze_left_y_mm", "gaze_right_x_mm", "gaze_right_y_mm",
    "eye_left_x_mm", "eye_left_y_mm", "eye_left_z_mm",
    "eye_right_x_mm", "eye_right_y_mm", "eye_right_z_mm",
    "validity_left", "validity_right",
]


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_ms": np.arange(n, dtype=float) * 3.333,
            "gaze_left_x_mm": np.linspace(100, 110, n) if n else [],
            "gaze_left_y_mm": np.full(n, 200.0),
            "gaze_right_x_mm": np.linspace(100, 110, n) if n else [],
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


def _pipeline(**velkw) -> IVTPipeline:
    vel = OlsenVelocityConfig(velocity_method="olsen2d", eye_mode="average", **velkw)
    return IVTPipeline(PipelineConfig(velocity=vel, classifier=IVTClassifierConfig()))


def _run(df: pd.DataFrame, **velkw) -> pd.DataFrame:
    return _pipeline(**velkw).process_dataframe(df)


# --- degenerate sizes ------------------------------------------------------

def test_empty_frame_returns_empty_without_raising() -> None:
    out = _run(_frame(0))
    assert len(out) == 0


def test_single_sample_no_window() -> None:
    out = _run(_frame(1))
    assert len(out) == 1
    # a single sample cannot form a velocity window
    assert out.loc[0, "ivt_sample_type"] in {"Unclassified", "EyesNotFound"}


def test_two_sample_input() -> None:
    out = _run(_frame(2))
    assert len(out) == 2


# --- degenerate validity / values -----------------------------------------

def test_all_invalid_input_is_eyes_not_found() -> None:
    df = _frame(40)
    df["validity_left"] = 2  # > max_validity (1)
    df["validity_right"] = 2
    out = _run(df)
    assert (out["ivt_sample_type"].isin(["EyesNotFound", "Unclassified"])).all()


def test_all_nan_gaze_no_crash() -> None:
    df = _frame(40)
    for col in ("gaze_left_x_mm", "gaze_left_y_mm", "gaze_right_x_mm", "gaze_right_y_mm"):
        df[col] = np.nan
    out = _run(df)
    assert len(out) == 40
    assert "ivt_sample_type" in out.columns


# --- gaps at the recording boundaries --------------------------------------

def test_gap_at_eof_exceeds_fallback_limit() -> None:
    df = _frame(80)
    df.loc[60:, "validity_left"] = 2  # 20-sample invalid tail
    df.loc[60:, "validity_right"] = 2
    out = _run(df, fixed_window_edge_fallback=True, fixed_window_edge_fallback_limit=5)
    assert len(out) == 80
    assert out.loc[79, "ivt_sample_type"] in {"Unclassified", "EyesNotFound"}


def test_gap_at_bof() -> None:
    df = _frame(80)
    df.loc[:20, "validity_left"] = 2
    df.loc[:20, "validity_right"] = 2
    out = _run(df)
    assert len(out) == 80
    assert out.loc[0, "ivt_sample_type"] in {"Unclassified", "EyesNotFound"}


# --- simple_api ------------------------------------------------------------

def test_process_eye_tracking_file_roundtrip(tmp_path) -> None:
    path = tmp_path / "rec.tsv"
    write_tsv(_frame(60), str(path))
    out = process_eye_tracking(str(path), velocity_threshold=30.0, show_plot=False)
    assert "velocity_deg_per_sec" in out.columns
    assert "ivt_sample_type" in out.columns


def test_get_and_print_statistics(capsys) -> None:
    df = _run(_frame(60))
    stats = get_statistics(df)
    assert stats["total_samples"] == 60
    assert stats["fixation_count"] + stats["saccade_count"] <= 60
    print_statistics(df)
    captured = capsys.readouterr()
    assert "Eye Tracking Statistics" in captured.out


def test_get_statistics_requires_classification() -> None:
    with pytest.raises(ValueError):
        get_statistics(_frame(10))
