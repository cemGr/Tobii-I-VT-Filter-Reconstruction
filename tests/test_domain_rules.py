"""Tests for centralized domain rules and lightweight stage contracts."""

import math

import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.domain.events import (
    assign_event_indices,
    iter_contiguous_events,
    rebuild_events_from_sample_labels,
)
from ivt_filter.domain.schema import (
    CLASSIFIED_COLUMNS,
    PREPROCESSED_COLUMNS,
    RAW_GAZE_COLUMNS,
    VELOCITY_DEG_PER_SEC,
    validate_classified_frame,
    validate_preprocessed_frame,
    validate_raw_gaze_frame,
    validate_velocity_frame,
)
from ivt_filter.domain.validity import INVALID_VALIDITY, parse_tobii_validity
from ivt_filter.processing.classification import apply_ivt_classifier
from ivt_filter.processing.velocity import compute_olsen_velocity


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Valid", 0),
        (" invalid ", INVALID_VALIDITY),
        ("2", 2),
        (3, 3),
        (2.9, 2),
        (None, INVALID_VALIDITY),
        ("unknown", INVALID_VALIDITY),
        (math.inf, INVALID_VALIDITY),
    ],
)
def test_parse_tobii_validity(value: object, expected: int) -> None:
    assert parse_tobii_validity(value) == expected


def test_event_reconstruction_supports_empty_frames() -> None:
    result = rebuild_events_from_sample_labels(
        pd.DataFrame({"sample": pd.Series(dtype=str)}),
        "sample",
        "event_type",
        "event_index",
    )

    assert result.empty
    assert result.columns.tolist() == ["sample", "event_type", "event_index"]


def test_single_sample_is_one_event() -> None:
    assert assign_event_indices(["Fixation"]) == (["Fixation"], [1])


def test_event_indices_reset_contiguity_after_non_event_labels() -> None:
    labels = [
        "Fixation",
        "Fixation",
        "Saccade",
        "Fixation",
        "Unclassified",
        "Fixation",
        "EyesNotFound",
        "Saccade",
    ]

    assert assign_event_indices(labels) == (
        labels,
        [1, 1, 2, 3, None, 4, None, 5],
    )


def test_contiguous_event_iteration_includes_non_event_labels() -> None:
    events = list(iter_contiguous_events(["Fixation", "Saccade", "Saccade", "EyesNotFound"]))

    assert [(event.label, event.start, event.end) for event in events] == [
        ("Fixation", 0, 0),
        ("Saccade", 1, 2),
        ("EyesNotFound", 3, 3),
    ]


@pytest.mark.parametrize(
    ("validator", "columns", "stage", "missing"),
    [
        (validate_raw_gaze_frame, RAW_GAZE_COLUMNS, "raw gaze", "time_ms"),
        (validate_preprocessed_frame, PREPROCESSED_COLUMNS, "preprocessed gaze", "smoothed_x_mm"),
        (validate_velocity_frame, (VELOCITY_DEG_PER_SEC,), "velocity", VELOCITY_DEG_PER_SEC),
        (validate_classified_frame, CLASSIFIED_COLUMNS, "classified", "ivt_event_index"),
    ],
)
def test_stage_contracts_report_missing_required_columns(
    validator, columns: tuple[str, ...], stage: str, missing: str
) -> None:
    frame = pd.DataFrame(columns=[column for column in columns if column != missing])

    with pytest.raises(ValueError, match=rf"{stage} DataFrame is missing required columns: '{missing}'"):
        validator(frame)


def test_velocity_pipeline_supports_empty_raw_frames() -> None:
    result = compute_olsen_velocity(
        pd.DataFrame(columns=RAW_GAZE_COLUMNS), OlsenVelocityConfig()
    )

    assert result.empty
    assert VELOCITY_DEG_PER_SEC in result.columns


def test_classification_empty_frame_preserves_explicit_contract() -> None:
    result = apply_ivt_classifier(pd.DataFrame({VELOCITY_DEG_PER_SEC: pd.Series(dtype=float)}))

    assert result.empty
    assert set(CLASSIFIED_COLUMNS).issubset(result.columns)
