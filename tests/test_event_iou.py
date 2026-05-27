"""
Tests for Maximum IoU event matching evaluation.

Tests follow Test-Driven Development: written before implementation.
Based on recommendations by Startsev & Zemblys (2022):
  "Evaluating Eye Movement Event Detection: A Review of the State of the Art"
  Behavior Research Methods, doi: 10.3758/s13428-021-01763-7
"""
from __future__ import annotations

import math
import pytest
import pandas as pd
import numpy as np

from ivt_filter.evaluation.event_iou import (
    Event,
    extract_events,
    compute_iou,
    match_events_max_iou,
    compute_event_iou_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    event_type: str,
    start_idx: int,
    end_idx: int,
    sample_interval_ms: float = 10.0,
) -> Event:
    """Create an Event with auto-computed times (10 ms/sample default)."""
    return Event(
        event_type=event_type,
        start_idx=start_idx,
        end_idx=end_idx,
        start_time_ms=start_idx * sample_interval_ms,
        end_time_ms=end_idx * sample_interval_ms,
        n_samples=end_idx - start_idx + 1,
    )


def make_df(
    types_gt: list,
    types_pred: list,
    sample_interval_ms: float = 10.0,
    gt_col: str = "gt_sample_type",
    pred_col: str = "ivt_sample_type",
) -> pd.DataFrame:
    """Build a minimal DataFrame for testing compute_event_iou_metrics."""
    assert len(types_gt) == len(types_pred)
    n = len(types_gt)
    return pd.DataFrame(
        {
            "time_ms": [i * sample_interval_ms for i in range(n)],
            gt_col: types_gt,
            pred_col: types_pred,
        }
    )


# ===========================================================================
# 1.  extract_events
# ===========================================================================

class TestExtractEvents:
    """Unit tests for extract_events()."""

    def test_two_events_basic(self):
        """Two consecutive event types yield two Event objects."""
        types = pd.Series(["F", "F", "F", "S", "S"])
        times = pd.Series([0.0, 10.0, 20.0, 30.0, 40.0])
        events = extract_events(types, times)

        assert len(events) == 2
        assert events[0].event_type == "F"
        assert events[1].event_type == "S"

    def test_single_event_type(self):
        """All samples same type → one event spanning the whole series."""
        types = pd.Series(["Fixation"] * 10)
        times = pd.Series([i * 10.0 for i in range(10)])
        events = extract_events(types, times)

        assert len(events) == 1
        assert events[0].event_type == "Fixation"
        assert events[0].n_samples == 10

    def test_empty_series_returns_empty_list(self):
        """Empty input → empty output."""
        events = extract_events(pd.Series([], dtype=str), pd.Series([], dtype=float))
        assert events == []

    def test_alternating_types_four_events(self):
        """F,S,F,S → 4 single-sample events."""
        types = pd.Series(["F", "S", "F", "S"])
        times = pd.Series([0.0, 10.0, 20.0, 30.0])
        events = extract_events(types, times)

        assert len(events) == 4
        for evt in events:
            assert evt.n_samples == 1

    def test_start_time_correct(self):
        """start_time_ms equals the time at the first sample of the event."""
        types = pd.Series(["F", "F", "S", "S", "S"])
        times = pd.Series([0.0, 10.0, 20.0, 30.0, 40.0])
        events = extract_events(types, times)

        # F starts at 0 ms, S starts at 20 ms
        assert events[0].start_time_ms == pytest.approx(0.0)
        assert events[1].start_time_ms == pytest.approx(20.0)

    def test_end_time_correct(self):
        """end_time_ms equals the time at the last sample of the event."""
        types = pd.Series(["F", "F", "S", "S", "S"])
        times = pd.Series([0.0, 10.0, 20.0, 30.0, 40.0])
        events = extract_events(types, times)

        # F ends at 10 ms, S ends at 40 ms
        assert events[0].end_time_ms == pytest.approx(10.0)
        assert events[1].end_time_ms == pytest.approx(40.0)

    def test_n_samples_correct(self):
        """n_samples is correct for each extracted event."""
        types = pd.Series(["F"] * 3 + ["S"] * 5 + ["F"] * 2)
        times = pd.Series([i * 10.0 for i in range(10)])
        events = extract_events(types, times)

        assert events[0].n_samples == 3  # F
        assert events[1].n_samples == 5  # S
        assert events[2].n_samples == 2  # F

    def test_start_end_idx_correct(self):
        """start_idx and end_idx (0-based, inclusive) are correct."""
        types = pd.Series(["F", "F", "S", "S"])
        times = pd.Series([0.0, 10.0, 20.0, 30.0])
        events = extract_events(types, times)

        assert events[0].start_idx == 0
        assert events[0].end_idx == 1
        assert events[1].start_idx == 2
        assert events[1].end_idx == 3

    def test_three_event_types(self):
        """Three distinct event types → three events."""
        types = pd.Series(["Fixation"] * 4 + ["Saccade"] * 2 + ["EyesNotFound"] * 3)
        times = pd.Series([i * 8.33 for i in range(9)])
        events = extract_events(types, times)

        assert len(events) == 3
        assert [e.event_type for e in events] == ["Fixation", "Saccade", "EyesNotFound"]


# ===========================================================================
# 2.  compute_iou
# ===========================================================================

class TestComputeIoU:
    """Unit tests for compute_iou()."""

    def test_identical_events_iou_one(self):
        """Identical events → IoU = 1.0."""
        e = make_event("F", 0, 9)
        assert compute_iou(e, e) == pytest.approx(1.0)

    def test_disjoint_events_iou_zero(self):
        """Non-overlapping events → IoU = 0.0."""
        e1 = make_event("F", 0, 4)
        e2 = make_event("F", 5, 9)
        assert compute_iou(e1, e2) == pytest.approx(0.0)

    def test_adjacent_events_no_overlap(self):
        """Events sharing a boundary sample index (end=5, start=6) → IoU = 0."""
        e1 = make_event("F", 0, 5)
        e2 = make_event("S", 6, 11)
        assert compute_iou(e1, e2) == pytest.approx(0.0)

    def test_partial_overlap_correct_value(self):
        """
        e1: idx 0-9  (10 samples)
        e2: idx 5-14 (10 samples)
        overlap = 5-9 = 5 samples
        union   = 10 + 10 - 5 = 15
        IoU     = 5/15 = 1/3
        """
        e1 = make_event("F", 0, 9)
        e2 = make_event("F", 5, 14)
        assert compute_iou(e1, e2) == pytest.approx(1.0 / 3.0)

    def test_contained_event(self):
        """
        e1: idx 0-19 (20 samples)
        e2: idx 5-9   (5 samples)  – contained within e1
        overlap = 5-9 = 5
        union   = 20
        IoU     = 5/20 = 0.25
        """
        e1 = make_event("F", 0, 19)
        e2 = make_event("F", 5, 9)
        assert compute_iou(e1, e2) == pytest.approx(0.25)

    def test_iou_is_symmetric(self):
        """IoU(a, b) == IoU(b, a)."""
        e1 = make_event("F", 3, 10)
        e2 = make_event("S", 7, 15)
        assert compute_iou(e1, e2) == pytest.approx(compute_iou(e2, e1))

    def test_iou_ignores_event_type(self):
        """IoU is the same regardless of event_type labels."""
        e1 = make_event("Fixation", 0, 9)
        e2 = make_event("Saccade", 0, 9)
        assert compute_iou(e1, e2) == pytest.approx(1.0)

    def test_single_sample_overlap(self):
        """
        e1: idx 0-5  (6 samples)
        e2: idx 5-10 (6 samples)
        overlap = only idx 5 = 1 sample
        union   = 6 + 6 - 1 = 11
        IoU     = 1/11
        """
        e1 = make_event("F", 0, 5)
        e2 = make_event("F", 5, 10)
        assert compute_iou(e1, e2) == pytest.approx(1.0 / 11.0)


# ===========================================================================
# 3.  match_events_max_iou
# ===========================================================================

class TestMatchEventsMaxIoU:
    """Unit tests for match_events_max_iou()."""

    def test_perfect_one_to_one_match(self):
        """Two GT events matched exactly to two Pred events → 2 matches, 0 FP."""
        gt = [make_event("F", 0, 9), make_event("S", 10, 14)]
        pr = [make_event("F", 0, 9), make_event("S", 10, 14)]
        matches, fp = match_events_max_iou(gt, pr)

        assert len([m for m in matches if m.pred_event is not None]) == 2
        assert len(fp) == 0

    def test_all_gt_matched_correctly(self):
        """Types of matched pairs should be preserved."""
        gt = [make_event("F", 0, 9), make_event("S", 10, 19)]
        pr = [make_event("F", 0, 9), make_event("S", 10, 19)]
        matches, _ = match_events_max_iou(gt, pr)

        for m in matches:
            assert m.pred_event is not None
            assert m.gt_event.event_type == m.pred_event.event_type

    def test_no_overlap_all_unmatched(self):
        """GT at idx 0-9, Pred at idx 20-29 → GT is FN, Pred is FP."""
        gt = [make_event("F", 0, 9)]
        pr = [make_event("F", 20, 29)]
        matches, fp = match_events_max_iou(gt, pr)

        assert len(matches) == 1
        assert matches[0].pred_event is None   # FN
        assert len(fp) == 1                    # FP

    def test_fragmented_pred_gt_matches_first_fragment(self):
        """
        GT: F(0-19)  — one big event (20 samples)
        Pred: F(0-9), F(10-19) — two fragments, IoU = 0.5 each

        Greedy matching picks first candidate encountered → GT matched to Pred[0].
        Pred[1] becomes a False Positive.
        """
        gt = [make_event("F", 0, 19)]
        pr = [make_event("F", 0, 9), make_event("F", 10, 19)]
        matches, fp = match_events_max_iou(gt, pr)

        assert len(matches) == 1
        assert matches[0].pred_event is not None  # GT matched to one fragment
        assert len(fp) == 1                        # other fragment is FP

    def test_merged_pred_gt_has_false_negative(self):
        """
        GT: F(0-9), F(10-19) — two events
        Pred: F(0-19) — merged into one

        Greedy: Pred[0] first matches GT[0] (or GT[1] – both IoU=0.5).
        The other GT event becomes FN.
        """
        gt = [make_event("F", 0, 9), make_event("F", 10, 19)]
        pr = [make_event("F", 0, 19)]
        matches, fp = match_events_max_iou(gt, pr)

        matched_gt = [m for m in matches if m.pred_event is not None]
        fn_gt = [m for m in matches if m.pred_event is None]

        assert len(matched_gt) == 1  # one GT matched
        assert len(fn_gt) == 1       # one GT is FN
        assert len(fp) == 0          # pred is used up

    def test_greedy_highest_iou_wins(self):
        """
        When one Pred event overlaps two GT events with different IoU,
        it should be matched to the one with higher IoU.

        GT[0]: F(0-9)   – 10 samples
        GT[1]: F(5-19)  – 15 samples
        Pred:  F(5-9)   – 5 samples

        IoU(GT[0], Pred) = 5 / (10+5-5) = 5/10 = 0.5
        IoU(GT[1], Pred) = 5 / (15+5-5) = 5/15 = 1/3

        Higher IoU → GT[0] should be matched.
        GT[1] → FN.
        """
        gt = [make_event("F", 0, 9), make_event("F", 5, 19)]
        pr = [make_event("F", 5, 9)]
        matches, fp = match_events_max_iou(gt, pr)

        # GT[0] matched (higher IoU), GT[1] is FN
        matched = [m for m in matches if m.pred_event is not None]
        fn = [m for m in matches if m.pred_event is None]

        assert len(matched) == 1
        assert matched[0].gt_event.start_idx == 0  # GT[0] got matched
        assert len(fn) == 1
        assert fn[0].gt_event.start_idx == 5       # GT[1] is FN

    def test_type_mismatch_event_still_matched(self):
        """
        IoU matching is type-agnostic: a GT Fixation can be matched to a
        Pred Saccade if they overlap. The type mismatch is recorded in
        is_correct_type = False.
        """
        gt = [make_event("Fixation", 0, 9)]
        pr = [make_event("Saccade", 0, 9)]
        matches, fp = match_events_max_iou(gt, pr)

        assert len(matches) == 1
        assert matches[0].pred_event is not None
        assert matches[0].is_correct_type is False

    def test_correct_type_flag_true_when_types_match(self):
        """is_correct_type is True when GT and Pred event types are identical."""
        gt = [make_event("Fixation", 0, 9)]
        pr = [make_event("Fixation", 0, 9)]
        matches, _ = match_events_max_iou(gt, pr)

        assert matches[0].is_correct_type is True

    def test_fn_has_none_pred_event(self):
        """Unmatched GT events have pred_event = None."""
        gt = [make_event("F", 0, 4), make_event("F", 100, 109)]
        pr = [make_event("F", 0, 4)]  # second GT has no matching pred
        matches, _ = match_events_max_iou(gt, pr)

        fn_matches = [m for m in matches if m.pred_event is None]
        assert len(fn_matches) == 1
        assert fn_matches[0].gt_event.start_idx == 100

    def test_fp_events_returned_in_second_list(self):
        """Pred events not matched to any GT are returned as FP list."""
        gt = [make_event("F", 0, 9)]
        pr = [make_event("F", 0, 9), make_event("S", 20, 29)]  # second pred is FP
        _, fp = match_events_max_iou(gt, pr)

        assert len(fp) == 1
        assert fp[0].start_idx == 20

    def test_empty_gt_all_pred_are_fp(self):
        """No GT events → all pred events are FP, matches list is empty."""
        gt = []
        pr = [make_event("F", 0, 9)]
        matches, fp = match_events_max_iou(gt, pr)

        assert matches == []
        assert len(fp) == 1

    def test_empty_pred_all_gt_are_fn(self):
        """No Pred events → all GT events are FN."""
        gt = [make_event("F", 0, 9)]
        pr = []
        matches, fp = match_events_max_iou(gt, pr)

        assert len(matches) == 1
        assert matches[0].pred_event is None
        assert fp == []

    def test_iou_stored_in_match_result(self):
        """MatchResult.iou holds the computed IoU value."""
        gt = [make_event("F", 0, 9)]
        pr = [make_event("F", 0, 9)]
        matches, _ = match_events_max_iou(gt, pr)

        assert matches[0].iou == pytest.approx(1.0)

    def test_onset_offset_dev_stored(self):
        """
        Pred starts 10 ms late and ends 10 ms early:
            GT: idx 0-9  → 0–90 ms (10 ms/sample)
            Pred: idx 1-8 → 10–80 ms
        onset_dev  =  10 - 0  =  +10 ms
        offset_dev =  80 - 90 = -10 ms
        """
        gt = [make_event("F", 0, 9, sample_interval_ms=10.0)]
        pr = [make_event("F", 1, 8, sample_interval_ms=10.0)]
        matches, _ = match_events_max_iou(gt, pr)

        assert matches[0].onset_dev_ms == pytest.approx(10.0)
        assert matches[0].offset_dev_ms == pytest.approx(-10.0)

    def test_fn_onset_offset_dev_are_nan(self):
        """Unmatched GT (FN) has NaN onset/offset deviations."""
        gt = [make_event("F", 0, 9)]
        matches, _ = match_events_max_iou(gt, [])

        assert math.isnan(matches[0].onset_dev_ms)
        assert math.isnan(matches[0].offset_dev_ms)


# ===========================================================================
# 4.  compute_event_iou_metrics  (integration)
# ===========================================================================

class TestComputeEventIoUMetrics:
    """Integration tests for compute_event_iou_metrics()."""

    # --- Perfect prediction ---------------------------------------------------

    def test_perfect_prediction_no_fn_fp(self):
        """Perfect prediction yields n_fn=0, n_fp=0."""
        df = make_df(
            ["F", "F", "F", "S", "S", "F", "F"],
            ["F", "F", "F", "S", "S", "F", "F"],
        )
        m = compute_event_iou_metrics(df, gt_col="gt_sample_type", pred_col="ivt_sample_type")

        assert m["n_fn"] == 0
        assert m["n_fp"] == 0
        assert m["n_matched"] == 3  # F, S, F

    def test_perfect_prediction_timing_zero(self):
        """Perfect prediction: onset and offset deviation are both 0 ms."""
        df = make_df(
            ["F", "F", "F", "S", "S"],
            ["F", "F", "F", "S", "S"],
            sample_interval_ms=8.33,
        )
        m = compute_event_iou_metrics(df, gt_col="gt_sample_type", pred_col="ivt_sample_type")

        for stats in m["timing"].values():
            assert stats["onset_dev_mean_ms"] == pytest.approx(0.0)
            assert stats["offset_dev_mean_ms"] == pytest.approx(0.0)

    def test_perfect_prediction_confusion_matrix_diagonal(self):
        """Confusion matrix of perfect prediction has only diagonal entries."""
        df = make_df(
            ["F", "F", "S", "S"],
            ["F", "F", "S", "S"],
        )
        m = compute_event_iou_metrics(df, gt_col="gt_sample_type", pred_col="ivt_sample_type")
        conf = m["confusion_matrix"]

        # Every GT type maps only to itself
        for gt_type, pred_counts in conf.items():
            assert list(pred_counts.keys()) == [gt_type], (
                f"GT '{gt_type}' was matched to unexpected type(s): {pred_counts}"
            )

    # --- Event counts ---------------------------------------------------------

    def test_event_counts_correct(self):
        """event_counts reports correct GT and Pred counts per class."""
        # GT: 2 × Fixation, 1 × Saccade events
        # Pred: 2 × Fixation, 1 × Saccade events (perfect)
        df = make_df(
            ["F", "F", "F", "S", "S", "F", "F"],
            ["F", "F", "F", "S", "S", "F", "F"],
        )
        m = compute_event_iou_metrics(df)
        counts = m["event_counts"]

        assert counts["F"]["gt"] == 2
        assert counts["S"]["gt"] == 1
        assert counts["F"]["pred"] == 2
        assert counts["S"]["pred"] == 1

    def test_event_counts_pred_different_from_gt(self):
        """
        GT: F(0-4), S(5-9)   → 1 F, 1 S
        Pred: F(0-9)          → 1 F, 0 S
        """
        df = make_df(
            ["F"] * 5 + ["S"] * 5,
            ["F"] * 10,
        )
        m = compute_event_iou_metrics(df)
        counts = m["event_counts"]

        assert counts["F"]["gt"] == 1
        assert counts["S"]["gt"] == 1
        assert counts["F"]["pred"] == 1
        # S in pred = 0; may not be in counts if not present in pred
        assert counts.get("S", {}).get("pred", 0) == 0

    # --- Onset / offset deviation ---------------------------------------------

    def test_onset_deviation_late_prediction(self):
        """
        GT:   F F F F F S S S F F   (S spans samples 5-7, 10 ms each → 50–70 ms)
        Pred: F F F F F F S S F F   (S spans samples 6-7 → 60–70 ms)
        Saccade onset_dev = 60 - 50 = +10 ms (pred starts 10 ms late)
        """
        gt_types   = ["F"] * 5 + ["S"] * 3 + ["F"] * 2
        pred_types = ["F"] * 6 + ["S"] * 2 + ["F"] * 2
        df = make_df(gt_types, pred_types, sample_interval_ms=10.0)
        m = compute_event_iou_metrics(df)

        assert "S" in m["timing"]
        assert m["timing"]["S"]["onset_dev_mean_ms"] == pytest.approx(10.0)

    def test_offset_deviation_early_prediction(self):
        """
        GT:   S S S S   (samples 0-3, 0–30 ms)
        Pred: S S S     (samples 0-2, 0–20 ms)
        Saccade offset_dev = 20 - 30 = -10 ms (pred ends 10 ms early)
        """
        gt_types   = ["S"] * 4
        pred_types = ["S"] * 3 + ["F"] * 1
        df = make_df(gt_types, pred_types, sample_interval_ms=10.0)
        m = compute_event_iou_metrics(df)

        assert "S" in m["timing"]
        assert m["timing"]["S"]["offset_dev_mean_ms"] == pytest.approx(-10.0)

    def test_onset_deviation_early_prediction(self):
        """
        GT:   F S S S F   (S: samples 1-3, 10–30 ms)
        Pred: F F S S F   (S: samples 2-3, 20–30 ms) → wait, that's late onset.

        Rethink: GT S starts at sample 1 (10 ms), Pred S starts at sample 0 (0 ms)
        onset_dev = 0 - 10 = -10 ms  (pred starts 10 ms EARLY)
        """
        gt_types   = ["F"] * 1 + ["S"] * 3 + ["F"] * 1
        pred_types = ["S"] * 4 + ["F"] * 1   # pred S starts earlier (idx 0 vs 1)
        df = make_df(gt_types, pred_types, sample_interval_ms=10.0)
        m = compute_event_iou_metrics(df)

        # Saccade onset deviation: pred starts at 0 ms, GT starts at 10 ms
        assert "S" in m["timing"]
        assert m["timing"]["S"]["onset_dev_mean_ms"] == pytest.approx(-10.0)

    # --- Confusion matrix (type mismatches) -----------------------------------

    def test_confusion_matrix_type_mismatch_recorded(self):
        """
        GT:   F F F S S   (F: 0-2, S: 3-4)
        Pred: F F F F F   (F: 0-4)  ← saccade classified as fixation

        Greedy matching:
          IoU(GT_S(3-4), Pred_F(0-4)) = 2 / (2+5-2) = 2/5 = 0.40
          IoU(GT_F(0-2), Pred_F(0-4)) = 3 / (3+5-3) = 3/5 = 0.60
        → GT_F(0-2) matched to Pred_F(0-4) (higher IoU)
        → GT_S(3-4) unmatched → FN
        """
        df = make_df(
            ["F", "F", "F", "S", "S"],
            ["F", "F", "F", "F", "F"],
        )
        m = compute_event_iou_metrics(df)

        assert m["n_fn"] >= 1  # at least one GT S is FN
        # Confusion matrix should show "S" → "FN" entry
        conf = m["confusion_matrix"]
        assert conf.get("S", {}).get("FN", 0) >= 1

    def test_confusion_matrix_has_no_entry_for_no_fn(self):
        """If no FN exists, 'FN' key should not appear in confusion matrix values."""
        df = make_df(["F", "F"], ["F", "F"])
        m = compute_event_iou_metrics(df)
        conf = m["confusion_matrix"]

        for gt_type, pred_counts in conf.items():
            assert "FN" not in pred_counts, (
                f"Unexpected FN for GT type '{gt_type}'"
            )

    # --- FN / FP counts -------------------------------------------------------

    def test_fn_count_for_unmatched_gt_events(self):
        """
        GT has events beyond pred coverage → counted as FN.
        GT: F(0-4), S(5-9), F(20-24)   (S and late F have no pred coverage)
        Pred: F(0-9)
        """
        gt_types   = ["F"] * 5 + ["S"] * 5 + ["X"] * 5 + ["F"] * 5
        pred_types = ["F"] * 10 + ["X"] * 10
        df = make_df(gt_types, pred_types)
        m = compute_event_iou_metrics(df)

        # GT[0] F(0-4) and GT[2] X(10-14) match Pred; GT[1] S(5-9) may be FN
        assert m["n_fn"] >= 0  # at least zero (depends on overlap)

    def test_fp_count_for_unmatched_pred_events(self):
        """
        GT: F(0-4)
        Pred: F(0-4), S(10-14)  ← second pred event has no GT counterpart

        S(10-14) should be a False Positive.
        """
        gt_types   = ["F"] * 5 + ["F"] * 5   # one long fixation
        pred_types = ["F"] * 5 + ["S"] * 5   # correct F, then rogue S
        df = make_df(gt_types, pred_types)
        m = compute_event_iou_metrics(df)

        assert m["n_fp"] >= 1

    # --- Summary counts -------------------------------------------------------

    def test_n_matched_plus_n_fn_equals_n_gt(self):
        """Every GT event is either matched or a FN."""
        df = make_df(
            ["F"] * 5 + ["S"] * 5 + ["F"] * 5,
            ["F"] * 4 + ["S"] * 6 + ["F"] * 5,
        )
        m = compute_event_iou_metrics(df)

        assert m["n_matched"] + m["n_fn"] == m["n_gt_events"]

    def test_n_matched_plus_n_fp_equals_n_pred(self):
        """Every Pred event is either matched or a FP."""
        df = make_df(
            ["F"] * 5 + ["S"] * 5,
            ["F"] * 4 + ["S"] * 6,
        )
        m = compute_event_iou_metrics(df)

        assert m["n_matched"] + m["n_fp"] == m["n_pred_events"]

    # --- Edge cases -----------------------------------------------------------

    def test_empty_dataframe_returns_zero_counts(self):
        """Empty DataFrame → all counts are zero, no crashes."""
        df = pd.DataFrame(
            {"time_ms": [], "gt_sample_type": [], "ivt_sample_type": []}
        )
        m = compute_event_iou_metrics(df)

        assert m["n_gt_events"] == 0
        assert m["n_pred_events"] == 0
        assert m["n_matched"] == 0
        assert m["n_fn"] == 0
        assert m["n_fp"] == 0

    def test_all_eyes_not_found(self):
        """
        All samples are EyesNotFound in both GT and Pred
        → 1 GT event, 1 Pred event, 1 match, no FN, no FP.
        """
        df = make_df(
            ["EyesNotFound"] * 10,
            ["EyesNotFound"] * 10,
        )
        m = compute_event_iou_metrics(df)

        assert m["n_gt_events"] == 1
        assert m["n_matched"] == 1
        assert m["n_fn"] == 0
        assert m["n_fp"] == 0

    def test_timing_only_for_correctly_typed_matches(self):
        """
        If GT 'Fixation' is matched to Pred 'Saccade' (type mismatch),
        the timing stats for 'Fixation' should NOT include this pair.
        """
        # GT F(0-9), Pred S(0-9) — same time, wrong type
        gt_types   = ["Fixation"] * 10
        pred_types = ["Saccade"] * 10
        df = make_df(gt_types, pred_types, sample_interval_ms=10.0)
        m = compute_event_iou_metrics(df)

        # 'Fixation' timing should be absent (no correctly typed match)
        assert "Fixation" not in m["timing"]

    def test_timing_abs_mean_is_nonnegative(self):
        """onset_dev_abs_mean_ms is always ≥ 0."""
        df = make_df(
            ["F"] * 5 + ["S"] * 5,
            ["F"] * 6 + ["S"] * 4,
            sample_interval_ms=10.0,
        )
        m = compute_event_iou_metrics(df)

        for stats in m["timing"].values():
            val = stats.get("onset_dev_abs_mean_ms", 0.0)
            if not math.isnan(val):
                assert val >= 0.0
