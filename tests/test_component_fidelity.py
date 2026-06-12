"""TDD tests for the component-fidelity analysis (Phase 1).

For each minimal pair (baseline B, one-component variant V, same recording) we
measure how faithfully the reconstruction reproduces Tobii's *marginal* effect of
that component:

* ``T_delta`` -- samples where Tobii's label changes between V and B.
* ``M_delta`` -- samples where our own filter's label changes between config V and
  config B run on the *same* raw gaze.
* ``K_dir``   -- co-located changes that also land on the same destination label
  (direction-faithful); ``K_colo`` -- plain co-location.

The metric core (:func:`fidelity_metrics`) and the sample aligner
(:func:`align_pair`) are pure and unit-tested here against known inputs; the
end-to-end pipeline path is exercised by the integration run, not by these tests.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "component_fidelity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("component_fidelity", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # let dataclass resolve annotations during exec
    spec.loader.exec_module(module)
    return module


cf = _load_script()


# --------------------------------------------------------------------------- #
# fidelity_metrics                                                            #
# --------------------------------------------------------------------------- #
def test_fidelity_metrics_perfect_directional():
    # 5 samples: Tobii changes 0..2; we change the same 0..2, all same destination.
    t = [True, True, True, False, False]
    m = [True, True, True, False, False]
    dir_ok = [True, True, True, True, True]
    out = cf.fidelity_metrics(t, m, dir_ok)
    assert out["t_delta"] == 3
    assert out["m_delta"] == 3
    assert out["k_dir"] == 3
    assert out["k_colo"] == 3
    assert out["recall_dir"] == 1.0
    assert out["precision_dir"] == 1.0
    assert out["jaccard_dir"] == 1.0


def test_fidelity_metrics_counter_directional_split():
    # Sample 0: both change but to *different* destination -> counts for colo only.
    # Sample 1: both change, same destination -> counts for both.
    # Sample 2: Tobii changes, we don't. Sample 3: we change, Tobii doesn't.
    t = [True, True, True, False]
    m = [True, True, False, True]
    dir_ok = [False, True, False, False]
    out = cf.fidelity_metrics(t, m, dir_ok)
    assert out["t_delta"] == 3
    assert out["m_delta"] == 3
    assert out["k_colo"] == 2  # samples 0 and 1 co-located
    assert out["k_dir"] == 1  # only sample 1 is direction-faithful
    assert out["recall_colo"] == pytest.approx(2 / 3)
    assert out["precision_colo"] == pytest.approx(2 / 3)
    assert out["recall_dir"] == pytest.approx(1 / 3)
    assert out["precision_dir"] == pytest.approx(1 / 3)
    # union = {0,1,2,3} = 4
    assert out["jaccard_dir"] == pytest.approx(1 / 4)
    assert out["jaccard_colo"] == pytest.approx(2 / 4)


def test_fidelity_metrics_empty_t_delta_returns_none():
    out = cf.fidelity_metrics([False, False], [True, False], [False, False])
    assert out["t_delta"] == 0
    assert out["recall_dir"] is None  # undefined, not a silent 0.0
    assert out["m_delta"] == 1
    assert out["precision_dir"] == 0.0  # 0 hits / 1 change


def test_fidelity_metrics_dir_ok_defaults_to_colocation():
    out = cf.fidelity_metrics([True, True, False], [True, False, True])
    # no dir_ok provided -> k_dir == k_colo
    assert out["k_dir"] == out["k_colo"] == 1


def test_fidelity_metrics_deterministic():
    t = [True, False, True, True]
    m = [True, True, False, True]
    d = [True, False, False, True]
    assert cf.fidelity_metrics(t, m, d) == cf.fidelity_metrics(t, m, d)


def test_fidelity_metrics_intersection_bounded():
    rng = np.random.default_rng(0)
    t = rng.random(200) < 0.3
    m = rng.random(200) < 0.4
    out = cf.fidelity_metrics(t, m)
    assert out["k_colo"] <= min(out["t_delta"], out["m_delta"])


# --------------------------------------------------------------------------- #
# align_pair                                                                  #
# --------------------------------------------------------------------------- #
def _frame(times, labels=None):
    df = pd.DataFrame({"time_ms": np.asarray(times, dtype=float)})
    df["gt_event_type"] = labels if labels is not None else ["Fixation"] * len(df)
    return df


def test_align_pair_identical():
    b = _frame([0.0, 2.0, 4.0, 6.0])
    v = _frame([0.0, 2.0, 4.0, 6.0])
    ab, av, info = cf.align_pair(b, v)
    assert info["time_match"] is True
    assert info["n_aligned"] == 4
    assert info["method"] == "identity"
    assert len(ab) == len(av) == 4


def test_align_pair_shifted_intersection():
    # V is missing the first sample and has an extra trailing one.
    b = _frame([0.0, 2.0, 4.0, 6.0])
    v = _frame([2.0, 4.0, 6.0, 8.0])
    ab, av, info = cf.align_pair(b, v)
    assert info["time_match"] is False
    assert info["n_aligned"] == 3  # times {2,4,6}
    assert info["n_dropped_b"] == 1
    assert info["n_dropped_v"] == 1
    assert list(ab["time_ms"]) == [2.0, 4.0, 6.0]
    assert list(av["time_ms"]) == [2.0, 4.0, 6.0]


def test_align_pair_no_overlap():
    b = _frame([0.0, 2.0])
    v = _frame([100.0, 102.0])
    ab, av, info = cf.align_pair(b, v)
    assert info["n_aligned"] == 0


# --------------------------------------------------------------------------- #
# parsing of component families                                              #
# --------------------------------------------------------------------------- #
def test_baseline_and_families_present():
    assert cf.STUDY_A1_BASELINE == "A1averageWindow20T30.tsv"
    fams = cf.STUDY_A1_FAMILIES
    assert "smoothing" in fams
    assert "merge" in fams
    assert "discard" in fams
    assert "gap_fill" in fams
    # each variant entry is (filename, dose_label)
    for variants in fams.values():
        for entry in variants:
            assert len(entry) == 2
            assert entry[0].endswith(".tsv")
