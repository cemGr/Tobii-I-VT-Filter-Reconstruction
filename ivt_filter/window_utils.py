# ivt_filter/window_utils.py
"""
Utility functions for window size configuration.

Provides helpers for both time-based and sample-based window sizing.
"""
from __future__ import annotations

from typing import Tuple
import pandas as pd

from .config import OlsenVelocityConfig, IVTClassifierConfig
from .sampling import estimate_sampling_rate


def samples_to_milliseconds(n_samples: int, sampling_rate_hz: float) -> float:
    """
    Convert number of samples to milliseconds.
    
    Args:
        n_samples: Number of samples
        sampling_rate_hz: Sampling rate in Hz
    
    Returns:
        Time duration in milliseconds
    
    Example:
        >>> samples_to_milliseconds(3, 120.0)
        25.0  # 3 samples at 120 Hz = 25 ms
        
        >>> samples_to_milliseconds(3, 60.0)
        50.0  # 3 samples at 60 Hz = 50 ms
    """
    return (n_samples / sampling_rate_hz) * 1000.0


def milliseconds_to_samples(window_ms: float, sampling_rate_hz: float) -> int:
    """
    Convert milliseconds to number of samples (rounded).
    
    Args:
        window_ms: Window duration in milliseconds
        sampling_rate_hz: Sampling rate in Hz
    
    Returns:
        Number of samples
    
    Example:
        >>> milliseconds_to_samples(20.0, 120.0)
        2  # 20 ms at 120 Hz ≈ 2.4 samples → 2
        
        >>> milliseconds_to_samples(25.0, 120.0)
        3  # 25 ms at 120 Hz = 3.0 samples
    """
    return int(round((window_ms / 1000.0) * sampling_rate_hz))


def detect_sampling_rate(df: pd.DataFrame, time_col: str = "time_ms") -> float:
    """
    Detect sampling rate from DataFrame.
    
    Args:
        df: DataFrame with time column
        time_col: Name of time column (in milliseconds)
    
    Returns:
        Nominal sampling rate in Hz
    
    Example:
        >>> df = read_tsv("data.tsv")
        >>> rate = detect_sampling_rate(df)
        >>> print(f"Detected: {rate} Hz")
        Detected: 120.0 Hz
    """
    sampling_info = estimate_sampling_rate(df, time_col=time_col)
    return sampling_info["nominal_hz"]


def create_time_based_config(
    window_ms: float,
    velocity_method: str = "olsen2d",
    eye_mode: str = "average",
    threshold: float = 30.0,
    smoothing_mode: str = None,
) -> Tuple[OlsenVelocityConfig, IVTClassifierConfig]:
    """
    Create configurations with TIME-BASED window sizing (fixed milliseconds).
    
    Args:
        window_ms: Window size in milliseconds (e.g., 20.0)
        velocity_method: "olsen2d" or "ray3d"
        eye_mode: "left", "right", or "average"
        threshold: Velocity threshold in deg/s
        smoothing_mode: None, "median", or "moving_average"
    
    Returns:
        Tuple of (velocity_config, classifier_config)
    
    Example:
        >>> vel_cfg, clf_cfg = create_time_based_config(window_ms=20.0)
        >>> # Always uses 20 ms, regardless of sampling rate
    """
    velocity_config = OlsenVelocityConfig(
        window_length_ms=window_ms,
        velocity_method=velocity_method,
        eye_mode=eye_mode,
        smoothing_mode=smoothing_mode,
    )
    
    classifier_config = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=threshold,
    )
    
    return velocity_config, classifier_config


def create_sample_based_config(
    n_samples: int,
    sampling_rate_hz: float,
    velocity_method: str = "olsen2d",
    eye_mode: str = "average",
    threshold: float = 30.0,
    smoothing_mode: str = None,
) -> Tuple[OlsenVelocityConfig, IVTClassifierConfig]:
    """
    Create configurations with SAMPLE-BASED window sizing (fixed number of samples).
    
    This approach is more robust across different sampling rates:
    - 3 samples at 60 Hz  = 50.0 ms
    - 3 samples at 120 Hz = 25.0 ms
    - 3 samples at 300 Hz = 10.0 ms
    
    Args:
        n_samples: Number of samples for window (e.g., 3)
        sampling_rate_hz: Sampling rate in Hz (e.g., 120.0)
        velocity_method: "olsen2d" or "ray3d"
        eye_mode: "left", "right", or "average"
        threshold: Velocity threshold in deg/s
        smoothing_mode: None, "median", or "moving_average"
    
    Returns:
        Tuple of (velocity_config, classifier_config)
    
    Example:
        >>> vel_cfg, clf_cfg = create_sample_based_config(
        ...     n_samples=3,
        ...     sampling_rate_hz=120.0
        ... )
        >>> print(vel_cfg.window_length_ms)
        25.0  # 3 samples at 120 Hz
    """
    window_ms = samples_to_milliseconds(n_samples, sampling_rate_hz)
    
    velocity_config = OlsenVelocityConfig(
        window_length_ms=window_ms,
        velocity_method=velocity_method,
        eye_mode=eye_mode,
        smoothing_mode=smoothing_mode,
    )
    
    classifier_config = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=threshold,
    )
    
    return velocity_config, classifier_config


def create_adaptive_config(
    df: pd.DataFrame,
    n_samples: int,
    time_col: str = "time_ms",
    velocity_method: str = "olsen2d",
    eye_mode: str = "average",
    threshold: float = 30.0,
    smoothing_mode: str = None,
) -> Tuple[OlsenVelocityConfig, IVTClassifierConfig]:
    """
    Create configurations with ADAPTIVE window sizing (auto-detect sampling rate).
    
    Automatically detects sampling rate from DataFrame and creates
    sample-based configuration.
    
    Args:
        df: DataFrame with time column
        n_samples: Number of samples for window
        time_col: Name of time column
        velocity_method: "olsen2d" or "ray3d"
        eye_mode: "left", "right", or "average"
        threshold: Velocity threshold in deg/s
        smoothing_mode: None, "median", or "moving_average"
    
    Returns:
        Tuple of (velocity_config, classifier_config)
    
    Example:
        >>> df = read_tsv("data.tsv")
        >>> vel_cfg, clf_cfg = create_adaptive_config(df, n_samples=3)
        >>> # Automatically adapts window to detected sampling rate
    """
    sampling_rate_hz = detect_sampling_rate(df, time_col=time_col)
    
    return create_sample_based_config(
        n_samples=n_samples,
        sampling_rate_hz=sampling_rate_hz,
        velocity_method=velocity_method,
        eye_mode=eye_mode,
        threshold=threshold,
        smoothing_mode=smoothing_mode,
    )


def print_window_info(
    window_ms: float,
    sampling_rate_hz: float,
    verbose: bool = True,
) -> None:
    """
    Print information about window size at given sampling rate.
    
    Args:
        window_ms: Window size in milliseconds
        sampling_rate_hz: Sampling rate in Hz
        verbose: If True, prints detailed information
    
    Example:
        >>> print_window_info(20.0, 120.0)
        Window: 20.0 ms
        Sampling rate: 120.0 Hz
        Number of samples: 2 (2.4 actual)
        Sample duration: 8.33 ms
    """
    n_samples_actual = (window_ms / 1000.0) * sampling_rate_hz
    n_samples_rounded = milliseconds_to_samples(window_ms, sampling_rate_hz)
    sample_duration_ms = 1000.0 / sampling_rate_hz
    
    if verbose:
        print(f"Window Information:")
        print(f"  Window size: {window_ms:.2f} ms")
        print(f"  Sampling rate: {sampling_rate_hz:.1f} Hz")
        print(f"  Number of samples: {n_samples_rounded} ({n_samples_actual:.2f} actual)")
        print(f"  Sample duration: {sample_duration_ms:.2f} ms")
        
        if abs(n_samples_actual - n_samples_rounded) > 0.1:
            print(f"  ⚠️  Warning: Window not aligned with sample rate!")
            print(f"     Consider using {n_samples_rounded} samples exactly:")
            optimal_ms = samples_to_milliseconds(n_samples_rounded, sampling_rate_hz)
            print(f"     → {optimal_ms:.2f} ms for {n_samples_rounded} samples")
    else:
        print(f"{window_ms:.2f} ms @ {sampling_rate_hz:.1f} Hz = {n_samples_rounded} samples")


def recommend_window_size(
    sampling_rate_hz: float,
    min_samples: int = 2,
    max_samples: int = 7,
) -> list[tuple[int, float]]:
    """
    Recommend window sizes for given sampling rate.
    
    Args:
        sampling_rate_hz: Sampling rate in Hz
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples
    
    Returns:
        List of (n_samples, window_ms) tuples
    
    Example:
        >>> recommendations = recommend_window_size(120.0)
        >>> for n, ms in recommendations:
        ...     print(f"{n} samples = {ms:.2f} ms")
        2 samples = 16.67 ms
        3 samples = 25.00 ms
        4 samples = 33.33 ms
        5 samples = 41.67 ms
    """
    recommendations = []
    
    for n_samples in range(min_samples, max_samples + 1):
        window_ms = samples_to_milliseconds(n_samples, sampling_rate_hz)
        recommendations.append((n_samples, window_ms))
    
    return recommendations
