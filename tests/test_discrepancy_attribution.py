"""TDD tests for the per-sample discrepancy attribution (WS-A).

Each test builds a small classified frame with exactly one known disagreement
cause and asserts the assigned category, plus overlap cases (ambiguous),
exhaustiveness (counts sum to total disagreements) and determinism.

The category set is evidence-based, refined from forensics on the real
conditions: the clean-condition residual is dominated by event-boundary timing
(``event_boundary``) and confident velocity-magnitude differences
(``velocity_magnitude``), alongside the thesis's gap, eye-offset, near-threshold
and smoothing mechanisms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.evaluation.discrepancy import (
    CATEGORIES,
    DISCREPANCY_CATEGORY,
    attribute_discrepancies,
    format_attribution_report,
)

THRESHOLD = 30.0


def _base_frame(n: int, *, dt_ms: float = 2.0) -> pd.DataFrame:
    """A frame that agrees everywhere, far from all boundaries and the threshold.

    GT is one long fixation (no event transitions), velocity is far below the
    threshold, and a single eye is never used. Tests introduce one disagreement
    with one engineered cause.
    """
    return pd.DataFrame(
        {
            "time_ms": np.arange(n, dtype=float) * dt_ms,
            "gt_event_type": ["Fixation"] * n,
            "ivt_sample_type": ["Fixation"] * n,
            "velocity_deg_per_sec": np.full(n, 5.0),
            "combined_valid": [True] * n,
            "gap_rule_triggered": [False] * n,
            "velocity_eye_used": ["average"] * n,
            "window_width_samples": [7] * n,
        }
    )


def _avg_cfg(**kw) -> OlsenVelocityConfig:
    params = dict(eye_mode="average", window_length_ms=20.0, smoothing_mode="none")
    params.update(kw)
    return OlsenVelocityConfig(**params)


def _category_at(df: pd.DataFrame, idx: int) -> str:
    return df.loc[idx, DISCREPANCY_CATEGORY]


# --- gap_edge --------------------------------------------------------------

def test_gap_edge_first_sample_unclassified() -> None:
    df = _base_frame(40)
    df.loc[0, "ivt_sample_type"] = "Unclassified"  # disagreement at the very edge
    out, summary = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    assert _category_at(out, 0) == "gap_edge"
    assert summary["counts"]["gap_edge"] == 1


def test_gap_edge_adjacent_to_gap() -> None:
    df = _base_frame(40)
    df.loc[18:22, "combined_valid"] = False
    df.loc[18:22, "gap_rule_triggered"] = True
    df.loc[23, "ivt_sample_type"] = "Saccade"  # one sample past the gap edge
    out, _ = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    assert _category_at(out, 23) == "gap_edge"


# --- event_boundary --------------------------------------------------------

def test_event_boundary_off_by_one() -> None:
    n = 40
    df = _base_frame(n)
    df.loc[20:, "gt_event_type"] = "Saccade"   # GT event boundary at index 20
    df.loc[21, "velocity_deg_per_sec"] = 29.0   # not near, not confident
    df.loc[21, "ivt_sample_type"] = "Fixation"  # GT Saccade, we still say Fixation
    out, _ = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    assert _category_at(out, 21) == "event_boundary"


# --- eye_offset_avg --------------------------------------------------------

def test_eye_offset_avg_single_eye_run() -> None:
    df = _base_frame(40)
    df.loc[15:25, "velocity_eye_used"] = "right"  # 11-sample single-eye run
    df.loc[20, "ivt_sample_type"] = "EyesNotFound"
    out, _ = attribute_discrepancies(df, _avg_cfg(eye_mode="average"), THRESHOLD)
    assert _category_at(out, 20) == "eye_offset_avg"


def test_eye_offset_avg_not_triggered_in_left_mode() -> None:
    df = _base_frame(40)
    df.loc[15:25, "velocity_eye_used"] = "left"
    df.loc[20, "ivt_sample_type"] = "EyesNotFound"
    out, _ = attribute_discrepancies(df, _avg_cfg(eye_mode="left"), THRESHOLD)
    assert _category_at(out, 20) != "eye_offset_avg"


# --- near_threshold_fp -----------------------------------------------------

def test_near_threshold_fp_within_eps() -> None:
    df = _base_frame(40)
    df.loc[20, "velocity_deg_per_sec"] = 30.02
    df.loc[20, "ivt_sample_type"] = "Saccade"
    out, _ = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    assert _category_at(out, 20) == "near_threshold_fp"


# --- velocity_magnitude ----------------------------------------------------

def test_velocity_magnitude_confident_flip() -> None:
    df = _base_frame(40)
    df.loc[20, "velocity_deg_per_sec"] = 135.0   # confident, far above threshold
    df.loc[20, "ivt_sample_type"] = "Saccade"     # GT Fixation, we say Saccade
    out, _ = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    assert _category_at(out, 20) == "velocity_magnitude"


# --- smoothing_attenuation -------------------------------------------------

def test_smoothing_attenuation_sign_flip() -> None:
    df = _base_frame(40)
    df.loc[20, "velocity_deg_per_sec"] = 25.0
    df["velocity_unsmoothed_deg_per_sec"] = 5.0
    df.loc[20, "velocity_unsmoothed_deg_per_sec"] = 42.0
    df.loc[20, "ivt_sample_type"] = "Saccade"
    out, _ = attribute_discrepancies(df, _avg_cfg(smoothing_mode="moving_average"), THRESHOLD)
    assert _category_at(out, 20) == "smoothing_attenuation"


def test_smoothing_inactive_no_smoothing_category() -> None:
    df = _base_frame(40)
    df.loc[20, "velocity_deg_per_sec"] = 25.0
    df["velocity_unsmoothed_deg_per_sec"] = 42.0
    df.loc[20, "ivt_sample_type"] = "Saccade"
    out, _ = attribute_discrepancies(df, _avg_cfg(smoothing_mode="none"), THRESHOLD)
    assert _category_at(out, 20) != "smoothing_attenuation"


# --- ambiguous (>= 2 mechanisms) -------------------------------------------

def test_ambiguous_gap_edge_and_near_threshold() -> None:
    df = _base_frame(40)
    df.loc[0, "velocity_deg_per_sec"] = 30.01  # near threshold AND at the edge
    df.loc[0, "ivt_sample_type"] = "Saccade"
    out, _ = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    assert _category_at(out, 0) == "ambiguous"


def test_ambiguous_eye_offset_and_smoothing() -> None:
    df = _base_frame(40)
    df.loc[15:25, "velocity_eye_used"] = "right"
    df.loc[20, "velocity_deg_per_sec"] = 25.0
    df["velocity_unsmoothed_deg_per_sec"] = 5.0
    df.loc[20, "velocity_unsmoothed_deg_per_sec"] = 42.0
    df.loc[20, "ivt_sample_type"] = "Saccade"
    out, _ = attribute_discrepancies(df, _avg_cfg(smoothing_mode="moving_average"), THRESHOLD)
    assert _category_at(out, 20) == "ambiguous"


# --- exhaustiveness, determinism, robustness -------------------------------

def test_exhaustiveness_counts_sum() -> None:
    df = _base_frame(40)
    df.loc[0, "ivt_sample_type"] = "Unclassified"      # gap_edge
    df.loc[20, "velocity_deg_per_sec"] = 30.02         # near_threshold
    df.loc[20, "ivt_sample_type"] = "Saccade"
    df.loc[30, "velocity_deg_per_sec"] = 80.0          # velocity_magnitude
    df.loc[30, "ivt_sample_type"] = "Saccade"
    out, summary = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    assert summary["n_disagreements"] == 3
    assert sum(summary["counts"].values()) == summary["n_disagreements"]
    assert set(summary["counts"]) == set(CATEGORIES)
    assert (out.loc[~out.index.isin([0, 20, 30]), DISCREPANCY_CATEGORY] == "").all()


def test_priority_determinism_under_column_shuffle() -> None:
    df = _base_frame(40)
    df.loc[0, "ivt_sample_type"] = "Unclassified"
    df.loc[20, "velocity_deg_per_sec"] = 30.02
    df.loc[20, "ivt_sample_type"] = "Saccade"
    out_a, sum_a = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    shuffled = df[list(reversed(df.columns))].copy()
    out_b, sum_b = attribute_discrepancies(shuffled, _avg_cfg(), THRESHOLD)
    assert list(out_a[DISCREPANCY_CATEGORY]) == list(out_b[DISCREPANCY_CATEGORY])
    assert sum_a["counts"] == sum_b["counts"]


def test_mild_velocity_difference_falls_through_to_other() -> None:
    # Long frame, mild velocity difference (not near, not confident), no mechanism.
    n = 60
    df = pd.DataFrame(
        {
            "time_ms": np.arange(n, dtype=float) * 2.0,
            "gt_event_type": ["Fixation"] * n,
            "ivt_sample_type": ["Fixation"] * n,
            "velocity_deg_per_sec": np.full(n, 29.0),  # |29-30| = 1: not near, not confident
        }
    )
    df.loc[30, "ivt_sample_type"] = "Saccade"
    cfg = _avg_cfg(window_length_ms=4.0)
    out, summary = attribute_discrepancies(df, cfg, THRESHOLD)
    assert _category_at(out, 30) == "other"
    assert summary["n_disagreements"] == 1


def test_report_is_formattable() -> None:
    df = _base_frame(40)
    df.loc[0, "ivt_sample_type"] = "Unclassified"
    _, summary = attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
    text = format_attribution_report(summary)
    assert "gap_edge" in text
    assert isinstance(text, str)


def test_pred_column_missing_raises() -> None:
    df = _base_frame(10).drop(columns=["ivt_sample_type"])
    with pytest.raises((KeyError, ValueError)):
        attribute_discrepancies(df, _avg_cfg(), THRESHOLD)
