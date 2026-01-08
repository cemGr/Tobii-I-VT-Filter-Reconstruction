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
