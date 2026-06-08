# tests/test_anchor_window_strategy.py
"""
TDD tests for the AnchorWindowStrategy module.

Covers compute_window_samples, estimate_avg_dt_us, SymmetricHalf, MidIndex,
and agreement on a deterministic synthetic 120 Hz recording.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivt_filter.strategies.anchor_window import (
    AnchorRoles,
    AnchorWindowStrategy,
    MidIndex,
    SymmetricHalf,
    compute_window_samples,
    estimate_avg_dt_us,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_AVG_DT_120HZ_US = 1_000_000.0 / 120.0  # ≈ 8333.33 µs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference_fixture() -> pd.DataFrame:
    """Return deterministic valid-left-eye samples at 120 Hz."""
    n_samples = 240
    return pd.DataFrame(
        {
            "time_us": np.arange(n_samples) * _AVG_DT_120HZ_US,
            "validity_left": ["Valid"] * n_samples,
        }
    )


# ---------------------------------------------------------------------------
# Group A — compute_window_samples
# ---------------------------------------------------------------------------


class TestComputeWindowSamples:
    def test_window_samples_1ms_120hz_tol1_0(self):
        N = compute_window_samples(1_000.0, _AVG_DT_120HZ_US, tolerance=1.0)
        assert N == 2

    def test_window_samples_1ms_120hz_tol1_01(self):
        N = compute_window_samples(1_000.0, _AVG_DT_120HZ_US, tolerance=1.01)
        assert N == 2

    def test_window_samples_20ms_120hz(self):
        N = compute_window_samples(20_000.0, _AVG_DT_120HZ_US, tolerance=1.0)
        assert N == 3

    def test_window_samples_minimum_always_two(self):
        # Tiny window relative to sample interval → truncates to 0 → result = 2
        N = compute_window_samples(1.0, 1_000_000.0, tolerance=1.0)
        assert N >= 2

    def test_window_samples_invalid_tolerance_below(self):
        with pytest.raises(ValueError):
            compute_window_samples(1_000.0, _AVG_DT_120HZ_US, tolerance=0.99)

    def test_window_samples_invalid_tolerance_above(self):
        with pytest.raises(ValueError):
            compute_window_samples(1_000.0, _AVG_DT_120HZ_US, tolerance=1.06)

    def test_window_samples_invalid_avg_dt_zero(self):
        with pytest.raises(ValueError):
            compute_window_samples(1_000.0, 0.0)

    def test_window_samples_invalid_window_zero(self):
        with pytest.raises(ValueError):
            compute_window_samples(0.0, _AVG_DT_120HZ_US)


# ---------------------------------------------------------------------------
# Group B — N=3 baseline (both strategies must agree)
# ---------------------------------------------------------------------------


class TestN3Baseline:
    def test_symmetric_half_n3(self):
        roles = SymmetricHalf().get_roles(anchor=5, N=3)
        assert roles == AnchorRoles(first=4, mid=5, last=6)

    def test_mid_index_n3(self):
        roles = MidIndex().get_roles(anchor=5, N=3)
        assert roles == AnchorRoles(first=4, mid=5, last=6)

    def test_strategies_agree_n3(self):
        sh = SymmetricHalf().get_roles(anchor=10, N=3)
        mi = MidIndex().get_roles(anchor=10, N=3)
        assert sh == mi


# ---------------------------------------------------------------------------
# Group C — N=2 critical difference (core thesis finding)
# ---------------------------------------------------------------------------


class TestN2CriticalDifference:
    def test_symmetric_half_n2_degenerates(self):
        roles = SymmetricHalf().get_roles(anchor=5, N=2)
        assert roles.first == roles.last, (
            "SymmetricHalf with N=2 must produce a degenerate window "
            "(first == last)"
        )

    def test_mid_index_n2_valid(self):
        roles = MidIndex().get_roles(anchor=5, N=2)
        assert roles.first == 4
        assert roles.last == 5
        assert roles.first < roles.last

    def test_strategies_disagree_n2(self):
        sh = SymmetricHalf().get_roles(anchor=5, N=2)
        mi = MidIndex().get_roles(anchor=5, N=2)
        assert sh != mi


# ---------------------------------------------------------------------------
# Group D — even N=4
# ---------------------------------------------------------------------------


class TestEvenN4:
    def test_symmetric_half_n4_wastes_one_sample(self):
        roles = SymmetricHalf().get_roles(anchor=5, N=4)
        # half = (4-1)//2 = 1 → first=4, last=6 → span = 2, not 3
        assert roles.last - roles.first == 2

    def test_mid_index_n4_uses_all_samples(self):
        roles = MidIndex().get_roles(anchor=5, N=4)
        # mid_pos = 2 → first=3, last=6 → span = 3
        assert roles.last - roles.first == 3

    def test_mid_index_n4_is_asymmetric(self):
        roles = MidIndex().get_roles(anchor=5, N=4)
        # anchor=5 is not centred: first=3, last=6, distance to first=2, to last=1
        assert roles.mid - roles.first != roles.last - roles.mid


# ---------------------------------------------------------------------------
# Group E — estimate_avg_dt_us
# ---------------------------------------------------------------------------


class TestEstimateAvgDtUs:
    def test_avg_dt_uniform_8333us(self):
        ts = np.arange(100) * 8333.0
        result = estimate_avg_dt_us(ts)
        assert abs(result - 8333.0) < 1e-6

    def test_avg_dt_uses_only_first_100(self):
        # First 100 samples at 8333 µs; next 100 at 1000 µs
        ts_first = np.arange(100) * 8333.0
        ts_rest = ts_first[-1] + np.arange(1, 101) * 1000.0
        ts = np.concatenate([ts_first, ts_rest])
        result = estimate_avg_dt_us(ts, n_samples=100)
        assert abs(result - 8333.0) < 1e-6

    def test_avg_dt_raises_on_too_few(self):
        with pytest.raises(ValueError):
            estimate_avg_dt_us(np.array([12345.0]))

    def test_avg_dt_raises_on_empty(self):
        with pytest.raises(ValueError):
            estimate_avg_dt_us(np.array([]))


# ---------------------------------------------------------------------------
# Group F — Agreement test on synthetic 120 Hz data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy,tolerance",
    [
        (SymmetricHalf(), 1.0),
        (SymmetricHalf(), 1.01),
        (MidIndex(), 1.0),
        (MidIndex(), 1.01),
    ],
)
def test_agreement_synthetic_120hz_reference(
    strategy: AnchorWindowStrategy,
    tolerance: float,
    reference_fixture: pd.DataFrame,
) -> None:
    """
    At 120 Hz with a 1 ms window, N=2 regardless of tolerance.

    SymmetricHalf must produce 100% degenerate windows (first == last).
    MidIndex must produce 0% degenerate windows (first < last).
    """
    df = reference_fixture
    assert len(df) >= 2, "Need at least 2 valid left-eye samples"

    timestamps = df["time_us"].values
    avg_dt = estimate_avg_dt_us(timestamps)
    N = compute_window_samples(1_000.0, avg_dt, tolerance)

    # Iterate over all valid anchor positions; get_roles is purely arithmetic
    # so bounds checking is not needed for the degeneracy assertion.
    mid_pos = N // 2
    anchors = range(mid_pos, len(df))
    roles = [strategy.get_roles(anchor, N) for anchor in anchors]
    degenerate_count = sum(1 for r in roles if r.first == r.last)
    total = len(roles)

    if isinstance(strategy, SymmetricHalf):
        assert degenerate_count == total, (
            f"SymmetricHalf should produce 100% degenerate windows at N={N}, "
            f"got {degenerate_count}/{total} (tolerance={tolerance})"
        )
    else:
        assert degenerate_count == 0, (
            f"MidIndex should produce 0% degenerate windows at N={N}, "
            f"got {degenerate_count}/{total} (tolerance={tolerance})"
        )
