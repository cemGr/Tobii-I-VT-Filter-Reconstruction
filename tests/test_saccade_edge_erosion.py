"""Tests for the flag-gated saccade-edge erosion (event-boundary off-by-one).

The centred Olsen window smears velocity across a fixation/saccade transition,
lifting the boundary sample just over threshold so a reconstructed saccade runs
one sample too wide on each fixation-facing edge. The optional
``erode_saccade_edges`` rule relabels such borderline edge samples back to
Fixation while leaving genuine high-velocity saccade edges untouched. Default
off preserves the strict I-VT baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import IVTClassifierConfig
from ivt_filter.processing.classification import apply_ivt_classifier


def _labels(velocities: list[float], *, erode: bool, band: float = 2.0) -> list[str]:
    cfg = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=30.0,
        erode_saccade_edges=erode,
        saccade_edge_velocity_band_deg=band,
    )
    result = apply_ivt_classifier(pd.DataFrame({"velocity_deg_per_sec": velocities}), cfg)
    return result["ivt_sample_type"].tolist()


def test_default_off_leaves_borderline_boundary_unchanged() -> None:
    # Saccade run [2,4]; edges (v=31) border fixations on both sides.
    velocities = [10.0, 10.0, 31.0, 100.0, 31.0, 10.0, 10.0]

    assert _labels(velocities, erode=False) == [
        "Fixation", "Fixation", "Saccade", "Saccade", "Saccade", "Fixation", "Fixation",
    ]


def test_erodes_borderline_edges_adjacent_to_fixation() -> None:
    velocities = [10.0, 10.0, 31.0, 100.0, 31.0, 10.0, 10.0]

    # Both borderline edges fall back to Fixation; the genuine 100 deg/s core stays.
    assert _labels(velocities, erode=True) == [
        "Fixation", "Fixation", "Fixation", "Saccade", "Fixation", "Fixation", "Fixation",
    ]


def test_preserves_high_velocity_saccade_edges() -> None:
    # Edges are genuine fast saccade samples (80 >= 30 + 2) — must not erode.
    velocities = [10.0, 10.0, 80.0, 80.0, 10.0, 10.0]

    assert _labels(velocities, erode=True) == [
        "Fixation", "Fixation", "Saccade", "Saccade", "Fixation", "Fixation",
    ]


def test_only_one_sample_per_edge_interior_untouched() -> None:
    # Run [1,3]: leading edge (31) erodes, trailing edge (100) stays, the
    # borderline interior sample (idx 2, v=31) is NOT eroded.
    velocities = [10.0, 31.0, 31.0, 100.0, 10.0]

    assert _labels(velocities, erode=True) == [
        "Fixation", "Fixation", "Saccade", "Saccade", "Fixation",
    ]


def test_single_sample_borderline_saccade_collapses_to_fixation() -> None:
    velocities = [10.0, 31.0, 10.0]

    assert _labels(velocities, erode=True) == ["Fixation", "Fixation", "Fixation"]


def test_does_not_erode_edges_bordering_nonfixation() -> None:
    # Saccade run bordered by Unclassified (nan velocity) — no fixation contact.
    velocities = [np.nan, 31.0, 100.0, 31.0, np.nan]

    assert _labels(velocities, erode=True) == [
        "Unclassified", "Saccade", "Saccade", "Saccade", "Unclassified",
    ]


def test_band_is_half_open_at_upper_bound() -> None:
    # v == threshold + band (32.0) is outside [30, 32) and must stay Saccade.
    velocities = [10.0, 32.0, 100.0, 10.0]

    assert _labels(velocities, erode=True) == [
        "Fixation", "Saccade", "Saccade", "Fixation",
    ]


def test_threshold_value_itself_is_eroded() -> None:
    # v == threshold (30.0) is inside [30, 32) and erodes when bordering fixation.
    velocities = [10.0, 30.0, 100.0, 10.0]

    assert _labels(velocities, erode=True) == [
        "Fixation", "Fixation", "Saccade", "Fixation",
    ]


def test_negative_band_is_rejected() -> None:
    with pytest.raises(ValueError):
        IVTClassifierConfig(saccade_edge_velocity_band_deg=-1.0)
