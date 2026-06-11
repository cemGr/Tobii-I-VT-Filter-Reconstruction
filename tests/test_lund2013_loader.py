"""TDD tests for the Lund2013 (Andersson et al. 2017) loader (WS-C).

The loader maps one annotated ``.mat`` into the ivt_filter slim schema. The
tests build a synthetic ``ETdata`` struct with the real column layout via
``scipy.io.savemat`` so they run in CI without the dataset; a separate path test
runs against real data and skips when it is absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from ivt_filter.io.lund2013 import (  # noqa: E402
    LUND_PROTOCOLS,
    load_lund2013_mat,
    map_lund_labels,
)

# Lund event-label integers.
FIX, SACC, PSO, PURSUIT, BLINK = 1, 2, 3, 4, 5


def _write_mat(tmp_path: Path, labels, *, x_px=None, y_px=None) -> Path:
    n = len(labels)
    t_us = np.arange(n, dtype=float) * 2000.0  # 500 Hz in microseconds
    if x_px is None:
        x_px = np.full(n, 512.0)  # screen-centre x (res 1024)
    if y_px is None:
        y_px = np.full(n, 384.0)  # screen-centre y (res 768)
    pos = np.column_stack([t_us, np.full(n, 3.0), np.full(n, 3.0), x_px, y_px, np.asarray(labels, float)])
    etdata = {
        "pos": pos,
        "screenDim": np.array([[380.0, 300.0]]),
        "screenRes": np.array([[1024, 768]], dtype=np.uint16),
        "viewDist": np.array([[670.0]]),
        "sampFreq": np.array([[500]], dtype=np.uint16),
    }
    path = tmp_path / "synthetic_img_test_labelled_MN.mat"
    scipy_io.savemat(str(path), {"ETdata": etdata})
    return path


def test_loader_maps_columns_and_duplicates_eyes(tmp_path: Path) -> None:
    path = _write_mat(tmp_path, [FIX, FIX, SACC, FIX])
    df = load_lund2013_mat(path)
    assert len(df) == 4
    # time microseconds -> milliseconds
    assert np.allclose(df["time_ms"].to_numpy(), [0.0, 2.0, 4.0, 6.0])
    # single gaze stream duplicated to both eyes
    assert np.allclose(df["gaze_left_x_mm"], df["gaze_right_x_mm"])
    assert np.allclose(df["gaze_left_y_mm"], df["gaze_right_y_mm"])
    assert "gaze_left_x_px" in df.columns


def test_screen_centre_maps_near_zero_mm(tmp_path: Path) -> None:
    path = _write_mat(tmp_path, [FIX])
    df = load_lund2013_mat(path)
    assert abs(float(df.loc[0, "gaze_left_x_mm"])) < 1e-6
    assert abs(float(df.loc[0, "gaze_left_y_mm"])) < 1e-6


def test_eye_origin_constant_view_distance(tmp_path: Path) -> None:
    path = _write_mat(tmp_path, [FIX, SACC, FIX])
    df = load_lund2013_mat(path)
    assert (df["eye_left_z_mm"] == 670.0).all()
    assert (df["eye_right_z_mm"] == 670.0).all()


def test_blink_marks_invalid_and_eyes_not_found(tmp_path: Path) -> None:
    path = _write_mat(tmp_path, [FIX, BLINK, FIX])
    df = load_lund2013_mat(path)  # default protocol P0
    assert df.loc[1, "validity_left"] != 0
    assert df.loc[1, "validity_right"] != 0
    assert df.loc[1, "gt_event_type"] == "EyesNotFound"
    assert df.loc[0, "validity_left"] == 0


def test_protocol_pso_mapping(tmp_path: Path) -> None:
    path = _write_mat(tmp_path, [PSO])
    p0 = load_lund2013_mat(path, protocol="P0")
    s1 = load_lund2013_mat(path, protocol="S1")
    assert p0.loc[0, "gt_event_type"] == "Saccade"      # PSO -> saccade in P0
    assert s1.loc[0, "gt_event_type"] == "Fixation"     # PSO -> fixation in S1


def test_pursuit_excluded_in_p0_included_in_s2(tmp_path: Path) -> None:
    labels = map_lund_labels(np.array([PURSUIT]), "P0")
    assert labels[0] == ""  # excluded from scoring
    labels_s2 = map_lund_labels(np.array([PURSUIT]), "S2")
    assert labels_s2[0] == "Fixation"


def test_known_protocols_exist() -> None:
    assert {"P0", "S1", "S2"} <= set(LUND_PROTOCOLS)


def test_fixation_and_saccade_mapping(tmp_path: Path) -> None:
    path = _write_mat(tmp_path, [FIX, SACC])
    df = load_lund2013_mat(path)
    assert df.loc[0, "gt_event_type"] == "Fixation"
    assert df.loc[1, "gt_event_type"] == "Saccade"


_REAL_DATA_DIR = Path(__file__).parent.parent / "test_data" / "external" / "lund2013" / "img"


def test_real_lund_file_loads_and_runs() -> None:
    """End-to-end smoke on a real file, skipped when the dataset is absent."""
    mats = sorted(_REAL_DATA_DIR.glob("*_MN.mat")) if _REAL_DATA_DIR.exists() else []
    if not mats:
        pytest.skip("Lund2013 data not present (see docs/external_data.md)")
    df = load_lund2013_mat(mats[0], protocol="P0")
    assert len(df) > 100
    assert {"time_ms", "gaze_left_x_mm", "validity_left", "gt_event_type"} <= set(df.columns)
    # the rebuilt time axis is strictly increasing and unique
    t = df["time_ms"].to_numpy()
    assert (np.diff(t) > 0).all()
