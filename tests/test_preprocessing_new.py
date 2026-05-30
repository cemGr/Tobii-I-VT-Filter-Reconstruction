"""
Tests for gap filling, preprocessing operations.
"""
import pytest
import pandas as pd
import numpy as np
from ivt_filter.preprocessing.gap_fill import gap_fill_gaze
from ivt_filter.config import OlsenVelocityConfig


class TestGapFill:
    """Tests for gap filling functionality."""
    
    def test_gap_fill_no_gaps(self, simple_eye_tracking_data):
        """Should return data when no gaps exist."""
        df = simple_eye_tracking_data.copy()
        original_len = len(df)
        
        cfg = OlsenVelocityConfig(gap_fill_enabled=True, gap_fill_max_gap_ms=75)
        result = gap_fill_gaze(df, cfg)
        
        # Length should not change
        assert len(result) == original_len
    
    def test_gap_fill_with_short_gaps(self, data_with_gaps):
        """Should fill short gaps."""
        df = data_with_gaps.copy()
        
        # Get number of valid samples before
        valid_before = df['gaze_left_x_mm'].notna().sum()
        
        cfg = OlsenVelocityConfig(gap_fill_enabled=True, gap_fill_max_gap_ms=75)
        result = gap_fill_gaze(df, cfg)
        
        # Should have filled some samples
        valid_after = result['gaze_left_x_mm'].notna().sum()
        assert valid_after >= valid_before
    
    def test_gap_fill_preserves_original_valid(self, data_with_gaps):
        """Should not modify original valid data."""
        df = data_with_gaps.copy()
        
        # Get original valid data
        mask_valid = df['validity_left'] == 0
        original_data = df.loc[mask_valid, 'gaze_left_x_mm'].copy()
        
        cfg = OlsenVelocityConfig(gap_fill_enabled=True, gap_fill_max_gap_ms=75)
        result = gap_fill_gaze(df, cfg)
        
        # Check valid samples are unchanged
        if len(original_data) > 0:
            result_data = result.loc[mask_valid, 'gaze_left_x_mm']
            pd.testing.assert_series_equal(result_data, original_data, check_dtype=False)
    
    def test_gap_fill_respects_max_gap(self):
        """Should respect max_gap_ms parameter."""
        # Create data with gaps of different sizes
        df = pd.DataFrame({
            'time_ms': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            'gaze_left_x_mm': [100, 105, np.nan, np.nan, np.nan, 110, np.nan, np.nan, np.nan, np.nan],
            'gaze_left_y_mm': [200, 205, np.nan, np.nan, np.nan, 210, np.nan, np.nan, np.nan, np.nan],
            'gaze_right_x_mm': [100, 105, np.nan, np.nan, np.nan, 110, np.nan, np.nan, np.nan, np.nan],
            'gaze_right_y_mm': [200, 205, np.nan, np.nan, np.nan, 210, np.nan, np.nan, np.nan, np.nan],
            'validity_left': [0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
            'validity_right': [0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        })
        
        # With strict max_gap_ms=15
        cfg_strict = OlsenVelocityConfig(gap_fill_enabled=True, gap_fill_max_gap_ms=15)
        result_strict = gap_fill_gaze(df.copy(), cfg_strict)
        
        # With permissive max_gap_ms=100
        cfg_perm = OlsenVelocityConfig(gap_fill_enabled=True, gap_fill_max_gap_ms=100)
        result_perm = gap_fill_gaze(df.copy(), cfg_perm)
        
        # Permissive should fill more
        filled_strict = result_strict['gaze_left_x_mm'].notna().sum()
        filled_perm = result_perm['gaze_left_x_mm'].notna().sum()
        
        assert filled_perm >= filled_strict
    
    def test_gap_fill_extended_single_eye_gap(self):
        """Regression: one eye drops before the joint gap (span-based measurement).

        Scenario (9ms sample rate, max_gap_ms=75):
          idx  t    L        R
          0–4  0–36 Valid    Valid    <- both valid
          5    45   Valid    INVALID  <- R drops first
          6–10 54–90 INVALID INVALID  <- joint gap (span=36ms)
          11–12 99–108 Valid INVALID  <- L recovers, R still invalid
          13+  117+ Valid    Valid    <- R recovers

        R-invalid run idx 5–12: span = 108-45 = 63ms < 75ms -> must be filled.
        Old boundary-based measurement: times[13]-times[4] = 81ms >= 75ms -> was NOT filled.
        """
        NaN = np.nan
        INV = 999  # above max_validity -> treated as invalid
        VAL = 0

        n_pre = 5   # idx 0-4: both valid
        # idx 5: L valid, R invalid
        # idx 6-10: both invalid (joint gap, 5 samples)
        # idx 11-12: L valid, R invalid
        # idx 13-17: both valid
        n_post = 5  # idx 13-17

        times = list(range(0, (n_pre + 1 + 5 + 2 + n_post) * 9, 9))  # 9ms per sample

        # Build index by index
        val_L = [VAL]*n_pre + [VAL] + [INV]*5 + [VAL]*2 + [VAL]*n_post
        val_R = [VAL]*n_pre + [INV] + [INV]*5 + [INV]*2 + [VAL]*n_post
        gaze_x_L = [100.0]*n_pre + [100.0] + [NaN]*5 + [100.0]*2 + [100.0]*n_post
        gaze_y_L = [200.0]*n_pre + [200.0] + [NaN]*5 + [200.0]*2 + [200.0]*n_post
        gaze_x_R = [100.0]*n_pre + [NaN] + [NaN]*5 + [NaN]*2 + [100.0]*n_post
        gaze_y_R = [200.0]*n_pre + [NaN] + [NaN]*5 + [NaN]*2 + [200.0]*n_post

        df = pd.DataFrame({
            'time_ms': times,
            'gaze_left_x_mm': gaze_x_L,
            'gaze_left_y_mm': gaze_y_L,
            'gaze_right_x_mm': gaze_x_R,
            'gaze_right_y_mm': gaze_y_R,
            'validity_left': val_L,
            'validity_right': val_R,
        })

        cfg = OlsenVelocityConfig(gap_fill_enabled=True, gap_fill_max_gap_ms=75)

        # R-invalid run: idx 5-12 (span = times[12]-times[5] = 108-45 = 63ms < 75ms)
        # Must be filled.
        r_invalid_indices = list(range(5, 13))
        assert all(pd.isna(df.loc[i, 'gaze_right_x_mm']) for i in r_invalid_indices), \
            "Pre-condition: R gaze should be NaN before fill"

        result = gap_fill_gaze(df, cfg)

        filled_right = [not pd.isna(result.loc[i, 'gaze_right_x_mm']) for i in r_invalid_indices]
        assert all(filled_right), (
            f"R eye should be filled for indices {r_invalid_indices}; "
            f"NaN remains at: {[i for i, ok in zip(r_invalid_indices, filled_right) if not ok]}"
        )

        # L-invalid run: idx 6-10 (joint gap) also must be filled
        l_invalid_indices = list(range(6, 11))
        filled_left = [not pd.isna(result.loc[i, 'gaze_left_x_mm']) for i in l_invalid_indices]
        assert all(filled_left), "L eye should be filled during joint gap"

    def test_gap_fill_span_at_max_not_filled(self):
        """Regression: a gap whose span equals max_gap_ms must NOT be filled.

        R-invalid run spans exactly max_gap_ms=75ms -> should remain unfilled.
        """
        NaN = np.nan
        INV = 999
        VAL = 0

        # 9ms sample rate; R-invalid from idx 1 to idx 9 (t=9 to t=81)
        # span = times[9]-times[1] = 81-9 = 72ms... need exactly 75ms
        # With 5ms samples: t=5 to t=80 -> span = 75ms (16 samples)
        dt = 5  # 5ms sample rate
        n = 20
        times = list(range(0, n * dt, dt))  # 0,5,10,...,95

        # R invalid from idx 2 to idx 16 (t=10 to t=80), span=70ms < 75ms would fill
        # For span = 75ms: idx 2 to idx 17 (t=10 to t=85), span=75ms -> NOT filled
        r_inv_start = 2
        r_inv_end = 17  # span = times[17]-times[2] = 85-10 = 75ms

        val_L = [VAL] * n
        val_R = ([VAL] * r_inv_start
                 + [INV] * (r_inv_end - r_inv_start + 1)
                 + [VAL] * (n - r_inv_end - 1))
        gaze_x_L = [100.0] * n
        gaze_y_L = [200.0] * n
        gaze_x_R = ([100.0] * r_inv_start
                    + [NaN] * (r_inv_end - r_inv_start + 1)
                    + [100.0] * (n - r_inv_end - 1))
        gaze_y_R = ([200.0] * r_inv_start
                    + [NaN] * (r_inv_end - r_inv_start + 1)
                    + [200.0] * (n - r_inv_end - 1))

        df = pd.DataFrame({
            'time_ms': times,
            'gaze_left_x_mm': gaze_x_L,
            'gaze_left_y_mm': gaze_y_L,
            'gaze_right_x_mm': gaze_x_R,
            'gaze_right_y_mm': gaze_y_R,
            'validity_left': val_L,
            'validity_right': val_R,
        })

        cfg = OlsenVelocityConfig(gap_fill_enabled=True, gap_fill_max_gap_ms=75)
        result = gap_fill_gaze(df, cfg)

        r_invalid_indices = list(range(r_inv_start, r_inv_end + 1))
        still_nan = [pd.isna(result.loc[i, 'gaze_right_x_mm']) for i in r_invalid_indices]
        assert all(still_nan), (
            f"R gap with span=75ms must NOT be filled (span >= max_gap_ms); "
            f"unexpectedly filled at: {[i for i, nan in zip(r_invalid_indices, still_nan) if not nan]}"
        )

    def test_gap_fill_disabled(self, data_with_gaps):
        """Should not fill gaps when disabled."""
        df = data_with_gaps.copy()
        
        # Get number of valid samples before
        valid_before = df['gaze_left_x_mm'].notna().sum()
        
        cfg = OlsenVelocityConfig(gap_fill_enabled=False, gap_fill_max_gap_ms=75)
        result = gap_fill_gaze(df, cfg)
        
        # Should not have filled anything
        valid_after = result['gaze_left_x_mm'].notna().sum()
        assert valid_after == valid_before
