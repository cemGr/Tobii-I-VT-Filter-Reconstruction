# tests/test_tobii_compat.py
"""
Tests for the new Tobii-compatible strategy classes and algorithms.

Components tested:
  1. TobiiGazeDirAngle  – asin-based angle computation (numerically stable)
  2. TobiiGazeVelocityWindowSelector – Tobii window size with 1.01 factor
  3. apply_tobii_eye_offset_interpolation – offset-based eye estimation
  4. merge_adjacent_fixations(weighting="sample_count") – sample-count weighting

Source reference: decompiled Tobii C# source code (Point3DVectorExtensions,
GazeVelocityCalculatorHelper, RemoteTrackerGazeDataToRecordedTwoEyedGazeDataConverter,
MergeFixationsFilter).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from ivt_filter.strategies.velocity_calculation import (
    TobiiGazeDirAngle,
    Ray3DGazeDir,
    VelocityContext,
)
from ivt_filter.strategies.windowing import TobiiGazeVelocityWindowSelector
from ivt_filter.preprocessing.eye_selection import (
    apply_tobii_eye_offset_interpolation,
    _parse_validity,
)
from ivt_filter.postprocessing.merge_fixations import merge_adjacent_fixations
from ivt_filter.config import OlsenVelocityConfig, FixationPostConfig, TobiiWindowPolicy


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _make_unit_vec(x: float, y: float, z: float) -> np.ndarray:
    """Return a normalized 3D vector."""
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)


def _angle_ref(v1: np.ndarray, v2: np.ndarray) -> float:
    """Reference angle computation via acos (standard method) in degrees."""
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# 1. TobiiGazeDirAngle – angle computation
# ---------------------------------------------------------------------------

class TestTobiiGazeDirAngle:
    """Tests for the asin-based angle formula."""

    def setup_method(self):
        self.strat = TobiiGazeDirAngle()

    def _ctx(self, v1, v2):
        return VelocityContext(
            x1_mm=0.0, y1_mm=0.0, x2_mm=0.0, y2_mm=0.0,
            eye_x_mm=None, eye_y_mm=None, eye_z_mm=None,
            dir1=v1, dir2=v2,
        )

    def test_zero_angle(self):
        """Identical vectors → 0°."""
        v = _make_unit_vec(0, 0, 1)
        assert self.strat.calculate_visual_angle_ctx(self._ctx(v, v)) == pytest.approx(0.0, abs=1e-8)

    def test_90_degrees(self):
        """Orthogonal vectors → 90°."""
        v1 = _make_unit_vec(1, 0, 0)
        v2 = _make_unit_vec(0, 1, 0)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_45_degrees(self):
        """45° angle."""
        v1 = _make_unit_vec(1, 0, 0)
        v2 = _make_unit_vec(1, 1, 0)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(45.0, abs=1e-5)

    def test_180_degrees(self):
        """Opposite vectors → 180°."""
        v1 = _make_unit_vec(0, 0, 1)
        v2 = _make_unit_vec(0, 0, -1)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(180.0, abs=1e-5)

    def test_small_angle_numerical_stability(self):
        """Very small angle (~0.001°) – numerical stability versus acos."""
        v1 = _make_unit_vec(0, 0, 1)
        # Slightly deviating vector (very small angle)
        epsilon = 1e-5
        v2 = _make_unit_vec(epsilon, 0, 1)
        angle_tobii = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        angle_ref = _angle_ref(v1, v2)
        # Both should be very close, but the Tobii formula is more stable
        assert angle_tobii >= 0.0
        assert angle_tobii == pytest.approx(angle_ref, abs=1e-4)

    def test_obtuse_angle_uses_complement_formula(self):
        """Obtuse angle (>90°) – the complementary formula is used."""
        # 150° between the vectors
        v1 = _make_unit_vec(0, 0, 1)
        angle_target = 150.0
        # Rotate about the x-axis
        rad = math.radians(angle_target)
        v2 = np.array([0, math.sin(rad), math.cos(rad)])
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(angle_target, abs=1e-4)

    def test_unnormalized_vectors_are_handled(self):
        """Non-normalized vectors are normalized internally."""
        v1 = np.array([3.0, 0.0, 0.0])   # normalized to [1, 0, 0]
        v2 = np.array([0.0, 5.0, 0.0])   # normalized to [0, 1, 0]
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_zero_vector_returns_nan(self):
        """Zero vector → NaN, since the angle is not computable."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = _make_unit_vec(0, 0, 1)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert math.isnan(angle)

    def test_none_direction_returns_nan(self):
        """Missing direction (None) → NaN."""
        ctx = VelocityContext(
            x1_mm=0.0, y1_mm=0.0, x2_mm=0.0, y2_mm=0.0,
            eye_x_mm=None, eye_y_mm=None, eye_z_mm=None,
            dir1=None, dir2=_make_unit_vec(0, 0, 1),
        )
        assert math.isnan(self.strat.calculate_visual_angle_ctx(ctx))

    def test_consistent_with_ray3d_gaze_dir_for_moderate_angles(self):
        """For moderate angles (<90°), TobiiGazeDirAngle and Ray3DGazeDir
        should produce very similar results (both dot-product based)."""
        ref_strat = Ray3DGazeDir()
        for angle_deg in [5.0, 15.0, 30.0, 45.0, 60.0, 80.0]:
            rad = math.radians(angle_deg)
            v1 = _make_unit_vec(0, 0, 1)
            v2 = _make_unit_vec(math.sin(rad), 0, math.cos(rad))
            ctx = self._ctx(v1, v2)
            angle_tobii = self.strat.calculate_visual_angle_ctx(ctx)
            angle_ray = ref_strat.calculate_visual_angle_ctx(ctx)
            # Both methods should agree within 0.001°
            assert abs(angle_tobii - angle_ray) < 0.001, (
                f"Angle {angle_deg}°: Tobii={angle_tobii:.6f}, Ray3D={angle_ray:.6f}"
            )

    def test_description_contains_asin(self):
        """The description should contain 'asin'."""
        assert "asin" in self.strat.get_description().lower()

    def test_tobii_asin_formula_matches_reference_cases(self):
        """Check specific angles against analytical values."""
        strat = self.strat
        cases = [
            (10.0,),
            (30.0,),
            (60.0,),
            (90.0,),
            (120.0,),
            (170.0,),
        ]
        for (target_deg,) in cases:
            rad = math.radians(target_deg)
            v1 = np.array([0.0, 0.0, 1.0])
            v2 = np.array([math.sin(rad), 0.0, math.cos(rad)])
            ctx = self._ctx(v1, v2)
            result = strat.calculate_visual_angle_ctx(ctx)
            assert result == pytest.approx(target_deg, abs=1e-4), (
                f"Target {target_deg}°, got {result:.6f}°"
            )


# ---------------------------------------------------------------------------
# 2. TobiiGazeVelocityWindowSelector – window computation
# ---------------------------------------------------------------------------

class TestTobiiGazeVelocityWindowSelector:
    """Tests for the Tobii-matched window selector."""

    def _make_valid_data(self, n: int):
        """Create n valid samples (all valid=True, equidistant)."""
        times = np.arange(n, dtype=float)
        valid = np.ones(n, dtype=bool)
        return times, valid

    def test_window_size_60hz(self):
        """60 Hz, 20 ms window: N = max(1, floor(20/16.67*1.01)) + 1 = 2."""
        sample_interval_ms = 1000.0 / 60  # ≈ 16.67 ms
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        N = sel._compute_window_samples(half_window_ms=10.0)
        assert N == 2

    def test_window_size_120hz_20ms(self):
        """120 Hz, 20 ms window: N = max(1, floor(20/8.33*1.01)) + 1 = 3."""
        sample_interval_ms = 1000.0 / 120  # ≈ 8.33 ms
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        N = sel._compute_window_samples(half_window_ms=10.0)
        assert N == 3

    def test_window_size_240hz_20ms(self):
        """240 Hz, 20 ms window: N = max(1, floor(20/4.17*1.01)) + 1 = 5."""
        sample_interval_ms = 1000.0 / 240  # ≈ 4.17 ms
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        N = sel._compute_window_samples(half_window_ms=10.0)
        assert N == 5

    def test_window_size_120hz_1ms(self):
        """120 Hz, 1 ms window: N=2; MidIndex must yield a valid window."""
        sample_interval_ms = 1000.0 / 120
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        N = sel._compute_window_samples(half_window_ms=0.5)
        assert N == 2

    def test_select_returns_valid_endpoints(self):
        """Window selection returns valid endpoints."""
        sample_interval_ms = 8.33  # 120 Hz
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        n = 20
        times, valid = self._make_valid_data(n)
        idx = 10
        first, last = sel.select(idx, times, valid, half_window_ms=10.0)
        assert first is not None and last is not None
        assert first < last
        assert first <= idx <= last

    def test_select_1ms_120hz_valid(self):
        """At 120 Hz / 1 ms window (N=2), MidIndex must produce a valid pair."""
        sample_interval_ms = 1000.0 / 120
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        n = 20
        times, valid = self._make_valid_data(n)
        idx = 5
        first, last = sel.select(idx, times, valid, half_window_ms=0.5)
        assert first is not None and last is not None
        assert first < last

    def test_select_invalid_center_returns_none(self):
        """Invalid centre sample returns (None, None)."""
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=8.33)
        n = 20
        times = np.arange(n, dtype=float)
        valid = np.ones(n, dtype=bool)
        valid[10] = False
        first, last = sel.select(10, times, valid, half_window_ms=10.0)
        assert first is None and last is None

    def test_select_at_data_boundary(self):
        """Window at data boundary is clamped correctly."""
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=8.33)
        n = 20
        times, valid = self._make_valid_data(n)
        first, last = sel.select(0, times, valid, half_window_ms=10.0)
        if first is not None:
            assert first >= 0
            assert last <= n - 1

    def test_invalid_sample_interval_raises(self):
        """Invalid sample_interval_ms raises ValueError."""
        with pytest.raises(ValueError):
            TobiiGazeVelocityWindowSelector(sample_interval_ms=0.0)
        with pytest.raises(ValueError):
            TobiiGazeVelocityWindowSelector(sample_interval_ms=-1.0)

    def test_tolerance_factor_increases_or_maintains_window(self):
        """Tolerance 1.01 produces N >= N with tolerance 1.0."""
        from ivt_filter.strategies.anchor_window import compute_window_samples
        interval = 8.33
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=interval)
        N_with = sel._compute_window_samples(10.0)
        N_without = compute_window_samples(20_000.0, interval * 1000.0, tolerance=1.0)
        assert N_with >= N_without

    def test_custom_strategy_is_used(self):
        """A custom AnchorWindowStrategy is respected."""
        from ivt_filter.strategies.anchor_window import SymmetricHalf
        sel = TobiiGazeVelocityWindowSelector(
            sample_interval_ms=1000.0 / 120,
            strategy=SymmetricHalf(),
        )
        assert isinstance(sel.strategy, SymmetricHalf)


# ---------------------------------------------------------------------------
# 3. apply_tobii_eye_offset_interpolation
# ---------------------------------------------------------------------------

class TestTobiiEyeOffsetInterpolation:
    """Tests for the Tobii eye-offset interpolation."""

    def _make_df(self, n: int = 10) -> pd.DataFrame:
        """Simple DataFrame with both eyes valid."""
        return pd.DataFrame({
            "time_ms": np.arange(n, dtype=float),
            "gaze_left_x_mm": np.full(n, 100.0),
            "gaze_left_y_mm": np.full(n, 200.0),
            "gaze_right_x_mm": np.full(n, 110.0),  # right is 10 mm further to the right
            "gaze_right_y_mm": np.full(n, 205.0),  # right is 5 mm further down
            "eye_left_x_mm": np.zeros(n),
            "eye_left_y_mm": np.zeros(n),
            "eye_left_z_mm": np.full(n, 600.0),
            "eye_right_x_mm": np.full(n, 65.0),    # IPD ~ 65 mm
            "eye_right_y_mm": np.zeros(n),
            "eye_right_z_mm": np.full(n, 600.0),
            "validity_left": np.zeros(n, dtype=int),
            "validity_right": np.zeros(n, dtype=int),
        })

    def _cfg(self) -> OlsenVelocityConfig:
        return OlsenVelocityConfig(max_validity=1)

    def test_both_valid_no_change(self):
        """If both eyes are valid → no change to the gaze data."""
        df = self._make_df(5)
        cfg = self._cfg()
        result = apply_tobii_eye_offset_interpolation(df, cfg)
        pd.testing.assert_series_equal(result["gaze_left_x_mm"], df["gaze_left_x_mm"])
        pd.testing.assert_series_equal(result["gaze_right_x_mm"], df["gaze_right_x_mm"])

    def test_missing_right_eye_reconstructed_from_offset(self):
        """Right eye missing → reconstructed via the stored offset."""
        df = self._make_df(10)
        cfg = self._cfg()

        # Sample 0: both valid → offset = (10, 5) is stored
        # Samples 1–5: right eye invalid
        df.loc[1:5, "validity_right"] = 2   # > max_validity=1 → invalid
        df.loc[1:5, "gaze_right_x_mm"] = float("nan")
        df.loc[1:5, "gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # Reconstructed right gaze = left gaze + offset(10, 5)
        for i in range(1, 6):
            expected_rx = df.loc[i, "gaze_left_x_mm"] + 10.0
            expected_ry = df.loc[i, "gaze_left_y_mm"] + 5.0
            assert result.loc[i, "gaze_right_x_mm"] == pytest.approx(expected_rx, abs=1e-6)
            assert result.loc[i, "gaze_right_y_mm"] == pytest.approx(expected_ry, abs=1e-6)

    def test_missing_left_eye_reconstructed_from_offset(self):
        """Left eye missing → reconstructed via the stored offset."""
        df = self._make_df(10)
        cfg = self._cfg()

        # Sample 0: both valid → offset = (10, 5)
        # Samples 1–3: left eye invalid
        df.loc[1:3, "validity_left"] = 2
        df.loc[1:3, "gaze_left_x_mm"] = float("nan")
        df.loc[1:3, "gaze_left_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        for i in range(1, 4):
            # left = right - offset
            expected_lx = df.loc[i, "gaze_right_x_mm"] - 10.0
            expected_ly = df.loc[i, "gaze_right_y_mm"] - 5.0
            assert result.loc[i, "gaze_left_x_mm"] == pytest.approx(expected_lx, abs=1e-6)
            assert result.loc[i, "gaze_left_y_mm"] == pytest.approx(expected_ly, abs=1e-6)

    def test_no_offset_yet_missing_eye_stays_nan(self):
        """If no offset is known yet and an eye is missing → NaN stays NaN."""
        df = self._make_df(5)
        cfg = self._cfg()

        # All samples: right eye invalid (no preceding valid pair)
        df["validity_right"] = 2
        df["gaze_right_x_mm"] = float("nan")
        df["gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)
        # No offset known → NaN stays
        assert result["gaze_right_x_mm"].isna().all()

    def test_original_df_not_modified(self):
        """Original DataFrame is not modified (copy)."""
        df = self._make_df(5)
        cfg = self._cfg()
        df.loc[2, "validity_right"] = 2
        df.loc[2, "gaze_right_x_mm"] = float("nan")

        original_rx = df["gaze_right_x_mm"].copy()
        _ = apply_tobii_eye_offset_interpolation(df, cfg)
        pd.testing.assert_series_equal(df["gaze_right_x_mm"], original_rx)

    def test_eye_origin_offset_also_reconstructed(self):
        """Eye-origin positions are also reconstructed via the offset."""
        df = self._make_df(10)
        cfg = self._cfg()

        # Sample 0: both valid → eye_offset = (65, 0, 0) stored
        df.loc[1, "validity_right"] = 2
        df.loc[1, "eye_right_x_mm"] = float("nan")
        df.loc[1, "eye_right_y_mm"] = float("nan")
        df.loc[1, "eye_right_z_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # Reconstructed right eye = left eye + IPD offset
        expected_rex = df.loc[1, "eye_left_x_mm"] + 65.0
        assert result.loc[1, "eye_right_x_mm"] == pytest.approx(expected_rex, abs=1e-6)

    def test_validity_flag_updated_after_interpolation(self):
        """Critical: validity_right is set to the valid marker after successful
        interpolation (0 for int columns), so that prepare_combined_columns
        includes the interpolated eye in the average."""
        df = self._make_df(5)  # uses int validity (0 = valid, 2 = invalid)
        cfg = self._cfg()

        # Sample 2: right eye invalid → should be interpolated
        df.loc[2, "validity_right"] = 2
        df.loc[2, "gaze_right_x_mm"] = float("nan")
        df.loc[2, "gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # Coordinates must be reconstructed
        assert pd.notna(result.loc[2, "gaze_right_x_mm"])
        # Validity must be recognized as "valid" (int→0, str→"Valid")
        parsed = _parse_validity(result.loc[2, "validity_right"])
        assert parsed <= cfg.max_validity, (
            f"After interpolation validity must be valid (≤{cfg.max_validity}), "
            f"got: {result.loc[2, 'validity_right']!r} → parsed={parsed}"
        )
        # Untouched samples remain unchanged
        assert result.loc[0, "validity_right"] == df.loc[0, "validity_right"]

    def test_validity_flag_left_updated_after_interpolation(self):
        """validity_left is set to the valid marker when the left eye is interpolated."""
        df = self._make_df(5)
        cfg = self._cfg()

        df.loc[2, "validity_left"] = 2
        df.loc[2, "gaze_left_x_mm"] = float("nan")
        df.loc[2, "gaze_left_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        assert pd.notna(result.loc[2, "gaze_left_x_mm"])
        parsed = _parse_validity(result.loc[2, "validity_left"])
        assert parsed <= cfg.max_validity

    def test_no_validity_update_when_no_offset_known(self):
        """If no offset is known → validity stays invalid."""
        df = self._make_df(5)
        cfg = self._cfg()

        # All samples: right eye invalid (never a valid pair → no offset)
        df["validity_right"] = 2
        df["gaze_right_x_mm"] = float("nan")
        df["gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # No offset → stays invalid
        for v in result["validity_right"]:
            assert _parse_validity(v) > cfg.max_validity

    def test_velocity_artifact_reduced_with_interpolation(self):
        """Phantom velocity at gap edges is reduced by offset interpolation.

        Scenario: the left eye toggles between valid/invalid – without interpolation
        a high phantom velocity appears at the gap edge (position jump due to a
        one-sided average). With interpolation the velocity stays low.
        """
        from ivt_filter.processing.velocity import compute_olsen_velocity

        rng = np.random.default_rng(42)
        n = 30
        hz = 120.0
        dt = 1000.0 / hz  # ~8.33 ms

        # Stable fixation on a constant point
        lx = np.full(n, 260.0)  # left eye always ~260 mm
        rx = np.full(n, 265.0)  # right eye always ~265 mm  (IPD offset = 5 mm)
        ly = np.full(n, 133.0)
        ry = np.full(n, 133.0)

        df = pd.DataFrame({
            "time_ms": np.arange(n) * dt,
            "gaze_left_x_mm":  lx,  "gaze_left_y_mm":  ly,
            "gaze_right_x_mm": rx,  "gaze_right_y_mm": ry,
            "validity_left":  ["Valid"] * n,
            "validity_right": ["Valid"] * n,
            "eye_left_z_mm":  np.full(n, 600.0),
            "eye_right_z_mm": np.full(n, 600.0),
        })

        # Gap: samples 10-12 left eye invalid
        for i in [10, 11, 12]:
            df.loc[i, "validity_left"] = "Invalid"
            df.loc[i, "gaze_left_x_mm"] = float("nan")
            df.loc[i, "gaze_left_y_mm"] = float("nan")

        cfg_base = OlsenVelocityConfig(
            window_length_ms=20.0, eye_mode="average",
            velocity_method="olsen2d", smoothing_mode="none",
            tobii_eye_offset_interpolation=False,
        )
        cfg_interp = OlsenVelocityConfig(
            window_length_ms=20.0, eye_mode="average",
            velocity_method="olsen2d", smoothing_mode="none",
            tobii_eye_offset_interpolation=True,
        )

        df_base   = compute_olsen_velocity(df.copy(), cfg_base)
        df_interp = compute_olsen_velocity(df.copy(), cfg_interp)

        # Samples directly after the gap (idx 13-15) – phantom velocity appears here
        post_gap = slice(13, 16)
        max_vel_base   = df_base.loc[post_gap, "velocity_deg_per_sec"].max()
        max_vel_interp = df_interp.loc[post_gap, "velocity_deg_per_sec"].max()

        # With interpolation the phantom velocity must be clearly smaller
        assert max_vel_interp < max_vel_base, (
            f"Interpolation should reduce phantom velocity: "
            f"base={max_vel_base:.1f}, interp={max_vel_interp:.1f}"
        )
        # And stay below 30 deg/s (no false saccade alarm)
        assert max_vel_interp < 30.0, (
            f"No phantom saccade after interpolation: vel={max_vel_interp:.1f} deg/s"
        )


# ---------------------------------------------------------------------------
# 4. merge_adjacent_fixations(weighting="sample_count")
# ---------------------------------------------------------------------------

class TestMergeFixationsSampleCountWeighting:
    """Tests for the Tobii-exact sample-count weighting in the fixation merge."""

    def _make_fixation_df(
        self,
        fix1_len: int = 100,
        fix2_len: int = 50,
        fix1_x: float = 0.0,
        fix2_x: float = 10.0,
        gap: int = 3,
    ) -> pd.DataFrame:
        """
        Create a DataFrame with two fixations (x coordinates) and a small gap.
        """
        n = fix1_len + gap + fix2_len
        x = np.concatenate([
            np.full(fix1_len, fix1_x),
            np.full(gap, float("nan")),   # gap
            np.full(fix2_len, fix2_x),
        ])
        y = np.zeros(n)
        times = np.arange(n, dtype=float)  # 1 ms spacing
        sample_type = (
            ["Fixation"] * fix1_len +
            ["Unclassified"] * gap +
            ["Fixation"] * fix2_len
        )
        velocity = np.zeros(n)  # all 0°/s → gap is merged

        return pd.DataFrame({
            "time_ms": times,
            "combined_x_mm": x,
            "combined_y_mm": y,
            "eye_z_mm": np.full(n, 600.0),
            "ivt_sample_type": sample_type,
            "velocity_deg_per_sec": velocity,
        })

    def test_uniform_weighting_uses_simple_mean(self):
        """Uniform mode: both fixation centers averaged with equal weight."""
        df = self._make_fixation_df(fix1_len=100, fix2_len=100, fix1_x=0.0, fix2_x=10.0)
        cfg = FixationPostConfig(
            merge_adjacent_fixations=True,
            max_time_gap_ms=10.0,
            max_angle_deg=10.0,
            merge_weighting="uniform",
        )
        # Both fixations have equally many samples → uniform == sample_count
        result, stats = merge_adjacent_fixations(
            df, cfg,
            sample_col="ivt_sample_type",
            time_col="time_ms",
            x_col="combined_x_mm",
            y_col="combined_y_mm",
            eye_z_col="eye_z_mm",
        )
        assert stats["merged_pairs"] == 1

    def test_sample_count_weighted_mean_different_from_uniform(self):
        """Sample-count weighting gives a different value than uniform when n1≠n2."""
        # Fix1: 100 Samples bei x=0, Fix2: 50 Samples bei x=6
        # uniform: mean(0, 6) = 3.0
        # sample_count: (0*100 + 6*50) / 150 = 2.0
        fix1_x = 0.0
        fix2_x = 6.0
        n1 = 100
        n2 = 50
        expected_sample_count = (fix1_x * n1 + fix2_x * n2) / (n1 + n2)
        expected_uniform = (fix1_x + fix2_x) / 2.0
        assert expected_sample_count != expected_uniform  # 2.0 != 3.0

    def test_sample_count_with_equal_fixations_matches_uniform(self):
        """For equally sized fixations: sample_count == uniform."""
        fix1_x = 0.0
        fix2_x = 10.0
        n1 = n2 = 50
        expected_sc = (fix1_x * n1 + fix2_x * n2) / (n1 + n2)
        expected_uniform = (fix1_x + fix2_x) / 2.0
        assert expected_sc == pytest.approx(expected_uniform, abs=1e-9)

    def test_merge_happens_regardless_of_weighting(self):
        """Merge happens regardless of the weighting mode."""
        df = self._make_fixation_df(fix1_len=50, fix2_len=30, gap=2)
        for weighting in ("uniform", "sample_count"):
            cfg = FixationPostConfig(
                merge_adjacent_fixations=True,
                max_time_gap_ms=10.0,
                max_angle_deg=5.0,
                merge_weighting=weighting,
            )
            result, stats = merge_adjacent_fixations(
                df.copy(), cfg,
                sample_col="ivt_sample_type",
                time_col="time_ms",
                x_col="combined_x_mm",
                y_col="combined_y_mm",
                eye_z_col="eye_z_mm",
            )
            assert stats["merged_pairs"] == 1, f"No merge for weighting={weighting}"

    def test_default_weighting_is_uniform(self):
        """Default config has weighting='uniform'."""
        cfg = FixationPostConfig()
        assert cfg.merge_weighting == "uniform"

    def test_tobii_weighting_config(self):
        """sample_count is accepted as a Literal in FixationPostConfig."""
        cfg = FixationPostConfig(merge_weighting="sample_count")
        assert cfg.merge_weighting == "sample_count"


class TestMergeFixationsGapVelocityCap:
    """Regression tests for the configurable velocity cap during gap relabeling."""

    @staticmethod
    def _merge_gap_velocities(
        gap_velocities: list[float],
        *,
        max_gap_velocity_deg_per_sec: float = 35.0,
    ) -> tuple[pd.DataFrame, dict[str, object], slice]:
        gap = len(gap_velocities)
        fixation_len = 3
        n = fixation_len + gap + fixation_len
        gap_slice = slice(fixation_len, fixation_len + gap)
        df = pd.DataFrame({
            "time_ms": np.arange(n, dtype=float),
            "combined_x_mm": np.zeros(n),
            "combined_y_mm": np.zeros(n),
            "eye_z_mm": np.full(n, 600.0),
            "ivt_sample_type": (
                ["Fixation"] * fixation_len
                + ["Unclassified"] * gap
                + ["Fixation"] * fixation_len
            ),
            "velocity_deg_per_sec": (
                [0.0] * fixation_len
                + gap_velocities
                + [0.0] * fixation_len
            ),
        })
        cfg = FixationPostConfig(
            merge_adjacent_fixations=True,
            max_time_gap_ms=10.0,
            max_angle_deg=0.5,
            max_gap_velocity_deg_per_sec=max_gap_velocity_deg_per_sec,
        )
        result, stats = merge_adjacent_fixations(
            df, cfg,
            sample_col="ivt_sample_type",
            time_col="time_ms",
            x_col="combined_x_mm",
            y_col="combined_y_mm",
            eye_z_col="eye_z_mm",
        )
        return result, stats, gap_slice

    def test_sample_at_configured_cap_is_eligible_for_relabeling(self):
        """The cap is inclusive: velocity == cap may become a fixation."""
        result, stats, gap_slice = self._merge_gap_velocities(
            [12.0], max_gap_velocity_deg_per_sec=12.0
        )

        assert result.iloc[gap_slice]["ivt_sample_type"].tolist() == ["Fixation"]
        assert stats["gap_samples_to_fixation"] == 1

    def test_sample_above_configured_cap_is_preserved(self):
        """Velocity > cap keeps the original gap label."""
        result, stats, gap_slice = self._merge_gap_velocities(
            [12.01], max_gap_velocity_deg_per_sec=12.0
        )

        assert result.iloc[gap_slice]["ivt_sample_type"].tolist() == ["Unclassified"]
        assert stats["gap_samples_to_fixation"] == 0

    def test_default_cap_preserves_existing_35_deg_per_sec_behavior(self):
        """The new config default matches the previous fixed 35-degree limit."""
        assert FixationPostConfig().max_gap_velocity_deg_per_sec == 35.0

        result, stats, gap_slice = self._merge_gap_velocities([35.0, 35.01])

        assert result.iloc[gap_slice]["ivt_sample_type"].tolist() == [
            "Fixation",
            "Unclassified",
        ]
        assert stats["gap_samples_to_fixation"] == 1

    def test_custom_cap_changes_only_intended_gap_fill_decision(self):
        """A custom cap relabels only the additionally eligible gap sample."""
        default_result, _, gap_slice = self._merge_gap_velocities([34.0, 36.0, 37.0])
        custom_result, _, _ = self._merge_gap_velocities(
            [34.0, 36.0, 37.0], max_gap_velocity_deg_per_sec=36.0
        )

        assert default_result.iloc[gap_slice]["ivt_sample_type"].tolist() == [
            "Fixation",
            "Unclassified",
            "Unclassified",
        ]
        assert custom_result.iloc[gap_slice]["ivt_sample_type"].tolist() == [
            "Fixation",
            "Fixation",
            "Unclassified",
        ]
        assert custom_result.loc[[0, 1, 2, 6, 7, 8], "ivt_sample_type"].tolist() == [
            "Fixation",
        ] * 6


# ---------------------------------------------------------------------------
# 5. Integration: OlsenVelocityConfig with Tobii flags
# ---------------------------------------------------------------------------

class TestTobiiConfigFields:
    """Ensures the new config fields are present and correctly typed."""

    def test_tobii_gaze_dir_is_valid_method(self):
        """'tobii_gaze_dir' is a valid velocity_method value."""
        cfg = OlsenVelocityConfig(velocity_method="tobii_gaze_dir")
        assert cfg.velocity_method == "tobii_gaze_dir"

    def test_tobii_window_mode_default_false(self):
        """tobii_window_mode defaults to False."""
        cfg = OlsenVelocityConfig()
        assert cfg.tobii_window_mode is False

    def test_tobii_sample_interval_ms_default_none(self):
        """tobii_sample_interval_ms defaults to None."""
        cfg = OlsenVelocityConfig()
        assert cfg.tobii_sample_interval_ms is None
        assert cfg.window_policy == TobiiWindowPolicy(sample_interval_ms=None)

    def test_tobii_eye_offset_interpolation_default_false(self):
        """tobii_eye_offset_interpolation defaults to False."""
        cfg = OlsenVelocityConfig()
        assert cfg.tobii_eye_offset_interpolation is False

    def test_tobii_window_mode_without_interval_is_resolved_by_pipeline(self):
        """The config allows an interval to be derived later from timestamps."""
        cfg = OlsenVelocityConfig(tobii_window_mode=True, tobii_sample_interval_ms=None)
        assert cfg.window_policy == TobiiWindowPolicy(sample_interval_ms=None)

    def test_make_window_selector_returns_tobii_selector(self):
        """make_window_selector() returns TobiiGazeVelocityWindowSelector."""
        from ivt_filter.processing.velocity import make_window_selector
        cfg = OlsenVelocityConfig(tobii_window_mode=True, tobii_sample_interval_ms=8.33)
        sel = make_window_selector(cfg)
        assert isinstance(sel, TobiiGazeVelocityWindowSelector)

    def test_velocity_strategy_factory_tobii_gaze_dir(self):
        """_get_velocity_calculation_strategy('tobii_gaze_dir') returns TobiiGazeDirAngle."""
        from ivt_filter.processing.velocity import _get_velocity_calculation_strategy
        strat = _get_velocity_calculation_strategy("tobii_gaze_dir")
        assert isinstance(strat, TobiiGazeDirAngle)


# ---------------------------------------------------------------------------
# 6. Integration: full pipeline with Tobii flags (synthetic data)
# ---------------------------------------------------------------------------

class TestTobiiPipelineIntegration:
    """Smoke tests for the full pipeline with Tobii-compatible settings."""

    def _make_gaze_dir_data(self, n: int = 100, hz: float = 120.0) -> pd.DataFrame:
        """Creates a DataFrame with normalized gaze-direction columns."""
        dt_ms = 1000.0 / hz
        # Fixation: gaze direction varies slightly around (0, 0, 1)
        np.random.seed(42)
        noise = np.random.normal(0, 0.001, (n, 3))
        base = np.array([0, 0, 1], dtype=float)
        dirs = base + noise
        # Normalize
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / norms

        df = pd.DataFrame({
            "time_ms": np.arange(n) * dt_ms,
            "gaze_left_x_mm": np.full(n, 100.0),
            "gaze_left_y_mm": np.full(n, 200.0),
            "gaze_right_x_mm": np.full(n, 110.0),
            "gaze_right_y_mm": np.full(n, 205.0),
            "eye_left_x_mm": np.zeros(n),
            "eye_left_y_mm": np.zeros(n),
            "eye_left_z_mm": np.full(n, 600.0),
            "eye_right_x_mm": np.full(n, 65.0),
            "eye_right_y_mm": np.zeros(n),
            "eye_right_z_mm": np.full(n, 600.0),
            "validity_left": np.zeros(n, dtype=int),
            "validity_right": np.zeros(n, dtype=int),
            "gaze_dir_left_x": dirs[:, 0],
            "gaze_dir_left_y": dirs[:, 1],
            "gaze_dir_left_z": dirs[:, 2],
            "gaze_dir_right_x": dirs[:, 0],
            "gaze_dir_right_y": dirs[:, 1],
            "gaze_dir_right_z": dirs[:, 2],
        })
        return df

    def test_tobii_gaze_dir_pipeline_runs(self):
        """Pipeline with tobii_gaze_dir + tobii_window_mode runs without errors."""
        from ivt_filter.processing.velocity import compute_olsen_velocity

        df = self._make_gaze_dir_data(n=50, hz=120.0)
        cfg = OlsenVelocityConfig(
            velocity_method="tobii_gaze_dir",
            tobii_window_mode=True,
            tobii_sample_interval_ms=1000.0 / 120,
            eye_mode="average",
        )
        result = compute_olsen_velocity(df, cfg)
        assert "velocity_deg_per_sec" in result.columns
        # Velocities should be present and not all NaN
        velocities = result["velocity_deg_per_sec"].dropna()
        assert len(velocities) > 0

    def test_tobii_eye_offset_interpolation_flag(self):
        """tobii_eye_offset_interpolation=True runs without errors."""
        from ivt_filter.processing.velocity import compute_olsen_velocity

        df = self._make_gaze_dir_data(n=30, hz=120.0)
        # Make some samples have a missing right eye
        df.loc[10:15, "validity_right"] = 2
        df.loc[10:15, "gaze_right_x_mm"] = float("nan")
        df.loc[10:15, "gaze_right_y_mm"] = float("nan")

        cfg = OlsenVelocityConfig(
            velocity_method="olsen2d",
            tobii_eye_offset_interpolation=True,
            eye_mode="average",
        )
        result = compute_olsen_velocity(df, cfg)
        assert "velocity_deg_per_sec" in result.columns

    def test_tobii_gaze_dir_produces_lower_velocities_than_acos_for_small_angles(self):
        """For very small angles (fixation), TobiiGazeDirAngle should be nearly
        identical to Ray3DGazeDir (difference < 0.01 deg/s)."""
        strat_tobii = TobiiGazeDirAngle()
        strat_ray = Ray3DGazeDir()

        # Small angle: ~1 deg
        v1 = _make_unit_vec(0, 0, 1)
        v2 = _make_unit_vec(math.sin(math.radians(1)), 0, math.cos(math.radians(1)))
        ctx = VelocityContext(
            x1_mm=0, y1_mm=0, x2_mm=0, y2_mm=0,
            eye_x_mm=None, eye_y_mm=None, eye_z_mm=None,
            dir1=v1, dir2=v2,
        )
        angle_tobii = strat_tobii.calculate_visual_angle_ctx(ctx)
        angle_ray = strat_ray.calculate_visual_angle_ctx(ctx)
        assert abs(angle_tobii - angle_ray) < 0.001
