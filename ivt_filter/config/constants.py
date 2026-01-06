# ivt_filter/constants.py
"""Physical and computational constants for eye tracking analysis."""

from __future__ import annotations


class PhysicalConstants:
    """Physical constants for eye tracking calculations."""
    
    # Default eye-to-screen distance when z-coordinate is unavailable (mm)
    DEFAULT_EYE_SCREEN_DISTANCE_MM: float = 600.0
    
    # Reasonable viewing distance range for validation (mm)
    MIN_VALID_DISTANCE_MM: float = 300.0
    MAX_VALID_DISTANCE_MM: float = 1200.0
    
    # Minimum time difference for valid velocity calculation (ms)
    MIN_DELTA_TIME_MS: float = 0.1


class ComputationalConstants:
    """Computational constants and defaults."""
    
    # Default smoothing window size (samples)
    DEFAULT_SMOOTHING_WINDOW: int = 5
    
    # Default velocity window length (ms)
    DEFAULT_VELOCITY_WINDOW_MS: float = 20.0
    
    # Default I-VT threshold (degrees per second)
    DEFAULT_VELOCITY_THRESHOLD: float = 30.0
    
    # Maximum validity code for "valid" Tobii samples
    MAX_TOBII_VALIDITY_CODE: int = 1
    
    # Default gap filling parameters
    DEFAULT_GAP_FILL_MAX_MS: float = 75.0
    
    # Default fixation filtering parameters
    DEFAULT_MIN_FIXATION_DURATION_MS: float = 60.0
    DEFAULT_FIXATION_MERGE_MAX_GAP_MS: float = 75.0
    DEFAULT_FIXATION_MERGE_MAX_ANGLE_DEG: float = 0.5
    
    # Default saccade merge parameters
    DEFAULT_MAX_SACCADE_BLOCK_DURATION_MS: float = 20.0


class ValidationMessages:
    """Standard validation and error messages."""
    
    MISSING_VELOCITY_COLUMN = "DataFrame must contain 'velocity_deg_per_sec'"
    MISSING_TIME_COLUMN = "DataFrame must contain 'time_ms'"
    INVALID_WINDOW_SIZE = "Window size must be >= 3"
    INVALID_FIXED_WINDOW = "fixed_window_samples must be >= 3"
