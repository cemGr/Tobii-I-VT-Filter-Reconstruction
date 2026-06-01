"""Regression tests for the shared inclusive event-duration convention."""

from __future__ import annotations

import pandas as pd
import pytest

from ivt_filter.config import FixationPostConfig, SaccadeMergeConfig
from ivt_filter.domain.durations import event_duration_ms, estimate_sample_interval_ms
from ivt_filter.postprocessing.discard_short_fixations import discard_short_fixations
from ivt_filter.postprocessing.merge_saccades import merge_short_saccade_blocks


@pytest.mark.parametrize(
    ("timestamps_ms", "expected_duration_ms"),
    [
        ([100.0], 1.0),
        ([100.0, 110.0], 20.0),
        ([0.0, 7.0, 20.0], 30.0),
    ],
)
def test_event_duration_includes_final_sample_interval(
    timestamps_ms: list[float], expected_duration_ms: float
) -> None:
    assert event_duration_ms(timestamps_ms) == expected_duration_ms


def test_single_sample_event_can_use_recording_interval() -> None:
    recording_interval_ms = estimate_sample_interval_ms([0.0, 10.0, 20.0])

    assert event_duration_ms([10.0], sample_interval_ms=recording_interval_ms) == 10.0


def _discard_fixation(
    timestamps_ms: list[float], *, threshold_ms: float
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.DataFrame(
        {
            "time_ms": timestamps_ms,
            "ivt_event_type": ["Fixation"] * len(timestamps_ms),
            "ivt_event_index": [1] * len(timestamps_ms),
        }
    )
    return discard_short_fixations(
        df,
        FixationPostConfig(min_fixation_duration_ms=threshold_ms),
        event_type_col="ivt_event_type",
        event_index_col="ivt_event_index",
        time_col="time_ms",
    )


def test_discard_short_fixations_keeps_event_at_exact_threshold() -> None:
    result, stats = _discard_fixation([0.0, 10.0], threshold_ms=20.0)

    assert result["ivt_event_type"].tolist() == ["Fixation", "Fixation"]
    assert stats["fixation_events"][0]["duration_ms"] == 20.0


def test_discard_short_fixations_discards_event_just_below_threshold() -> None:
    result, stats = _discard_fixation([0.0, 9.0], threshold_ms=19.0)

    assert result["ivt_event_type"].tolist() == ["Unclassified", "Unclassified"]
    assert stats["fixation_events"][0]["duration_ms"] == 18.0


def test_discard_short_fixations_uses_robust_interval_for_irregular_samples() -> None:
    result, stats = _discard_fixation([0.0, 7.0, 20.0], threshold_ms=30.0)

    assert result["ivt_event_type"].tolist() == ["Fixation"] * 3
    assert stats["dt_ms"] == 10.0
    assert stats["fixation_events"][0]["duration_ms"] == 30.0


def _merge_single_sample_saccade(*, threshold_ms: float) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.DataFrame(
        {
            "time_ms": [0.0, 10.0, 20.0],
            "ivt_sample_type": ["Fixation", "Saccade", "Fixation"],
            "ivt_event_type": ["Fixation", "Saccade", "Fixation"],
        }
    )
    return merge_short_saccade_blocks(
        df, SaccadeMergeConfig(max_saccade_block_duration_ms=threshold_ms)
    )


def test_merge_short_saccade_blocks_keeps_event_at_exact_threshold() -> None:
    result, stats = _merge_single_sample_saccade(threshold_ms=10.0)

    assert result["ivt_sample_type_smoothed"].tolist() == ["Fixation", "Saccade", "Fixation"]
    assert stats["n_blocks_merged"] == 0


def test_merge_short_saccade_blocks_merges_event_just_below_threshold() -> None:
    result, stats = _merge_single_sample_saccade(threshold_ms=10.01)

    assert result["ivt_sample_type_smoothed"].tolist() == ["Fixation"] * 3
    assert stats["n_blocks_merged"] == 1
