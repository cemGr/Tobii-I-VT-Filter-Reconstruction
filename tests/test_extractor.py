"""Focused tests for Tobii archive extraction and timestamp conversion."""

from __future__ import annotations

import pandas as pd
import pytest

from extractor import TimestampUnitDetector, TobiiDataExtractor


def _convert(source: pd.DataFrame, timestamp_unit: str = "auto") -> pd.DataFrame:
    return TobiiDataExtractor()._convert_timestamps(source, pd.DataFrame(), timestamp_unit)


def test_timestamp_detector_accepts_ascii_microsecond_header() -> None:
    detector = TimestampUnitDetector()
    source = pd.DataFrame(columns=["Recording timestamp [ms]", "Eyetracker timestamp [us]"])

    assert detector.detect_columns(source) == (
        "Recording timestamp [ms]",
        "Eyetracker timestamp [us]",
    )


def test_auto_derives_microseconds_from_native_milliseconds() -> None:
    result = _convert(pd.DataFrame({"Recording timestamp [ms]": [12, 13.5]}))

    assert result["time_ms"].tolist() == [12, 13.5]
    assert result["time_us"].tolist() == [12_000, 13_500]


def test_auto_derives_milliseconds_from_native_microseconds() -> None:
    result = _convert(pd.DataFrame({"Eyetracker timestamp [μs]": [12_000, 13_500]}))

    assert result["time_ms"].tolist() == [12, 13.5]
    assert result["time_us"].tolist() == [12_000, 13_500]


def test_auto_preserves_both_native_timestamp_columns() -> None:
    result = _convert(
        pd.DataFrame(
            {
                "Recording timestamp [ms]": [12, 13],
                "Eyetracker timestamp [μs]": [12_123, 13_123],
            }
        )
    )

    assert result["time_ms"].tolist() == [12, 13]
    assert result["time_us"].tolist() == [12_123, 13_123]


@pytest.mark.parametrize(
    ("timestamp_unit", "values", "expected_ms", "expected_us"),
    [
        ("ms", [1, 2], [1, 2], [1_000, 2_000]),
        ("us", [1_000, 2_000], [1, 2], [1_000, 2_000]),
        ("ns", [1_000_000, 2_000_000], [1, 2], [1_000, 2_000]),
    ],
)
def test_explicit_timestamp_unit_override_converts_preferred_source(
    timestamp_unit, values, expected_ms, expected_us
) -> None:
    result = _convert(pd.DataFrame({"Recording timestamp [ms]": values}), timestamp_unit)

    assert result["time_ms"].tolist() == expected_ms
    assert result["time_us"].tolist() == expected_us


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


def test_extract_rejects_missing_timestamp_columns(tmp_path) -> None:
    input_path = tmp_path / "raw.tsv"
    output_path = tmp_path / "slim.tsv"
    pd.DataFrame({"Presented Stimulus name": ["Target"]}).to_csv(
        input_path, sep="\t", index=False
    )

    with pytest.raises(ValueError, match="No usable timestamp column found"):
        TobiiDataExtractor().extract(input_path, output_path)

    assert not output_path.exists()


def test_explicit_override_rejects_missing_timestamp_columns() -> None:
    with pytest.raises(ValueError, match="No usable timestamp column found"):
        _convert(pd.DataFrame({"Presented Stimulus name": ["Target"]}), "us")
