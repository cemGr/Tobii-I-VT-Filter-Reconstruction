"""Focused tests for Tobii archive extraction and native timestamp handling."""

from __future__ import annotations

import pandas as pd
import pytest

from extractor import TimestampUnitDetector, TobiiDataExtractor


def test_timestamp_detector_accepts_ascii_microsecond_header() -> None:
    detector = TimestampUnitDetector()
    source = pd.DataFrame(columns=["Recording timestamp [ms]", "Eyetracker timestamp [us]"])

    assert detector.detect_columns(source) == (
        "Recording timestamp [ms]",
        "Eyetracker timestamp [us]",
    )


def test_extract_filters_rows_sorts_by_recording_time_and_preserves_native_timestamps(
    tmp_path,
) -> None:
    input_path = tmp_path / "raw.tsv"
    output_path = tmp_path / "slim.tsv"
    source = pd.DataFrame(
        {
            "Sensor": ["Eye Tracker", "Eye Tracker", "Mouse", "Eye Tracker", "Eye Tracker"],
            "Event": ["", "", "", "Eyetracker Calibration", ""],
            "Presented Stimulus name": ["Target", "Target", "Target", "Target", "  "],
            "Recording timestamp [ms]": [20, 10, 5, 30, 40],
            "Eyetracker timestamp [μs]": [20_123, 10_123, 5_123, 30_123, 40_123],
            "Gaze point left X [DACS mm]": [2.0, 1.0, 0.5, 3.0, 4.0],
        }
    )
    source.to_csv(input_path, sep="\t", index=False, decimal=",")

    TobiiDataExtractor().extract(input_path, output_path)

    result = pd.read_csv(output_path, sep="\t", decimal=",")
    assert result["time_ms"].tolist() == [10, 20]
    assert result["time_us"].tolist() == [10_123, 20_123]
    assert result["gaze_left_x_mm"].tolist() == [1.0, 2.0]


def test_extract_rejects_missing_recording_timestamp_column(tmp_path) -> None:
    input_path = tmp_path / "raw.tsv"
    output_path = tmp_path / "slim.tsv"
    pd.DataFrame({"Presented Stimulus name": ["Target"]}).to_csv(
        input_path, sep="\t", index=False
    )

    with pytest.raises(ValueError, match=r"Recording timestamp \[ms\] column not found"):
        TobiiDataExtractor().extract(input_path, output_path)

    assert not output_path.exists()


def test_extract_keeps_microsecond_timestamp_optional(tmp_path) -> None:
    input_path = tmp_path / "raw.tsv"
    output_path = tmp_path / "slim.tsv"
    pd.DataFrame(
        {
            "Presented Stimulus name": ["Target"],
            "Recording timestamp [ms]": [12],
        }
    ).to_csv(input_path, sep="\t", index=False)

    TobiiDataExtractor().extract(input_path, output_path)

    result = pd.read_csv(output_path, sep="\t", decimal=",")
    assert result["time_ms"].tolist() == [12]
    assert result["time_us"].isna().all()
