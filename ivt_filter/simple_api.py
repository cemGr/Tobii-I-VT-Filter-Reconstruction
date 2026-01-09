# ivt_filter/simple_api.py
"""Simple user-friendly API for IVT filter without ground truth requirements.

This module provides an easy-to-use interface for users who just want to
apply the I-VT filter to their eye tracking data without evaluation or
complex configuration.

Example:
    >>> from ivt_filter.simple_api import process_eye_tracking
    >>> 
    >>> # Simplest usage - just input file
    >>> df = process_eye_tracking("my_data.tsv")
    >>> 
    >>> # Save results
    >>> df = process_eye_tracking("my_data.tsv", output="results.tsv")
    >>> 
    >>> # With custom threshold
    >>> df = process_eye_tracking("my_data.tsv", velocity_threshold=30.0)
"""
from __future__ import annotations

from typing import Optional, Literal
import pandas as pd

from .io.pipeline import IVTPipeline
from .window_utils import create_time_based_config, create_adaptive_config
from .io.io import read_tsv


def process_eye_tracking(
    input_path: str,
    output_path: Optional[str] = None,
    velocity_threshold: float = 30.0,
    window_ms: float = 20.0,
    eye_selection: Literal["left", "right", "average"] = "average",
    velocity_method: Literal["olsen2d", "ray3d"] = "olsen2d",
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Process eye tracking data with I-VT filter (simple interface).
    
    This is the easiest way to apply the I-VT filter to your data.
    No ground truth required, no complex configuration needed.
    
    Args:
        input_path: Path to your Tobii TSV export file
        output_path: Optional path to save results (with velocity and classification)
        velocity_threshold: Velocity threshold in deg/s (default: 30.0)
                           Lower = more fixations, Higher = fewer fixations
        window_ms: Time window for velocity calculation in milliseconds (default: 20.0)
        eye_selection: Which eye to use ("left", "right", or "average")
        velocity_method: Calculation method ("olsen2d" or "ray3d")
        show_plot: Whether to show a visualization plot
    
    Returns:
        DataFrame with added columns:
            - velocity_deg_per_sec: Computed eye movement velocity
            - ivt_sample_type: Classification ("Fixation" or "Saccade")
    
    Example:
        >>> # Quick analysis with defaults
        >>> df = process_eye_tracking("my_recording.tsv")
        >>> 
        >>> # More sensitive (detects smaller saccades)
        >>> df = process_eye_tracking("data.tsv", velocity_threshold=20.0)
        >>> 
        >>> # Save results for later
        >>> df = process_eye_tracking("data.tsv", output="classified_data.tsv")
        >>> 
        >>> # Use left eye only with 3D method
        >>> df = process_eye_tracking(
        ...     "data.tsv",
        ...     eye_selection="left",
        ...     velocity_method="ray3d"
        ... )
    """
    # Build configuration
    vel_config, clf_config = create_time_based_config(
        window_ms=window_ms,
        eye_mode=eye_selection,
        velocity_method=velocity_method,
        threshold=velocity_threshold,
        smoothing_mode="none"  # Explicitly set to "none" instead of None
    )
    
    # Create pipeline
    pipeline = IVTPipeline(vel_config, clf_config)
    
    # Process data
    df = pipeline.run(
        input_path=input_path,
        output_path=output_path,
        classify=True,
        evaluate=False,  # No ground truth needed!
        plot=show_plot,
        with_events=False  # No GT events to show
    )
    
    return df


def process_eye_tracking_adaptive(
    input_path: str,
    output_path: Optional[str] = None,
    velocity_threshold: float = 30.0,
    n_samples: int = 3,
    eye_selection: Literal["left", "right", "average"] = "average",
    velocity_method: Literal["olsen2d", "ray3d"] = "olsen2d",
    show_plot: bool = True,
) -> pd.DataFrame:
    """
    Process eye tracking data with adaptive sample-based window sizing.
    
    This function automatically detects your sampling rate and uses a fixed
    number of samples for the velocity window, which is more robust across
    different recording devices.
    
    Args:
        input_path: Path to your Tobii TSV export file
        output_path: Optional path to save results
        velocity_threshold: Velocity threshold in deg/s (default: 30.0)
        n_samples: Number of samples for velocity window (default: 3)
                   Recommended: 2-5 samples
        eye_selection: Which eye to use ("left", "right", or "average")
        velocity_method: Calculation method ("olsen2d" or "ray3d")
        show_plot: Whether to show a visualization plot
    
    Returns:
        DataFrame with velocity and classification columns
    
    Example:
        >>> # Automatic sampling rate detection with 3 samples
        >>> df = process_eye_tracking_adaptive("data.tsv")
        >>> 
        >>> # Use 5 samples for more smoothing
        >>> df = process_eye_tracking_adaptive("data.tsv", n_samples=5)
    
    Note:
        Sample-based windows are recommended when:
        - Processing multiple recordings with different sampling rates
        - You want consistent behavior across different eye trackers
        - Recording quality varies (sample-based adapts better)
    """
    # Read data to detect sampling rate
    df_temp = read_tsv(input_path)
    
    # Build adaptive configuration
    vel_config, clf_config = create_adaptive_config(
        df=df_temp,
        n_samples=n_samples,
        eye_mode=eye_selection,
        velocity_method=velocity_method,
        threshold=velocity_threshold,
        smoothing_mode="none"  # Explicitly set to "none"
    )
    
    # Create pipeline
    pipeline = IVTPipeline(vel_config, clf_config)
    
    # Process data
    df = pipeline.run(
        input_path=input_path,
        output_path=output_path,
        classify=True,
        evaluate=False,  # No ground truth needed!
        plot=show_plot,
        with_events=False
    )
    
    return df


def get_statistics(df: pd.DataFrame) -> dict:
    """
    Get simple statistics from processed eye tracking data.
    
    Args:
        df: DataFrame returned from process_eye_tracking()
    
    Returns:
        Dictionary with statistics:
            - total_samples: Number of data points
            - fixation_count: Number of fixations detected
            - saccade_count: Number of saccades detected
            - fixation_percentage: Percentage of time spent in fixations
            - avg_velocity: Average eye movement velocity
            - max_velocity: Maximum velocity observed
    
    Example:
        >>> df = process_eye_tracking("data.tsv")
        >>> stats = get_statistics(df)
        >>> print(f"Fixations: {stats['fixation_count']}")
        >>> print(f"Fixation time: {stats['fixation_percentage']:.1f}%")
    """
    if "ivt_sample_type" not in df.columns:
        raise ValueError("DataFrame must be processed with classify=True first")
    
    stats = {}
    
    # Basic counts
    stats["total_samples"] = len(df)
    stats["fixation_count"] = (df["ivt_sample_type"] == "Fixation").sum()
    stats["saccade_count"] = (df["ivt_sample_type"] == "Saccade").sum()
    
    # Percentages
    stats["fixation_percentage"] = (stats["fixation_count"] / stats["total_samples"]) * 100
    stats["saccade_percentage"] = (stats["saccade_count"] / stats["total_samples"]) * 100
    
    # Velocity statistics  
    vel_col = "velocity_deg_per_sec"  # Standard column name from velocity.py
    if vel_col in df.columns:
        stats["avg_velocity"] = df[vel_col].mean()
        stats["max_velocity"] = df[vel_col].max()
        stats["median_velocity"] = df[vel_col].median()
    
    return stats


def print_statistics(df: pd.DataFrame) -> None:
    """
    Print human-readable statistics from processed eye tracking data.
    
    Args:
        df: DataFrame returned from process_eye_tracking()
    
    Example:
        >>> df = process_eye_tracking("data.tsv")
        >>> print_statistics(df)
        
        === Eye Tracking Statistics ===
        Total samples: 5000
        Fixations: 3500 (70.0%)
        Saccades: 1500 (30.0%)
        
        Velocity:
          Average: 15.3 deg/s
          Median: 12.1 deg/s
          Maximum: 245.7 deg/s
    """
    stats = get_statistics(df)
    
    print("\n=== Eye Tracking Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Fixations: {stats['fixation_count']} ({stats['fixation_percentage']:.1f}%)")
    print(f"Saccades: {stats['saccade_count']} ({stats['saccade_percentage']:.1f}%)")
    
    if "avg_velocity" in stats:
        print("\nVelocity:")
        print(f"  Average: {stats['avg_velocity']:.1f} deg/s")
        print(f"  Median: {stats['median_velocity']:.1f} deg/s")
        print(f"  Maximum: {stats['max_velocity']:.1f} deg/s")
    
    print()


# Quick usage examples as docstring
__doc__ += """

Quick Start Examples
====================

1. Simplest usage:
------------------
```python
from ivt_filter.simple_api import process_eye_tracking

df = process_eye_tracking("my_data.tsv")
```

2. Save results and adjust threshold:
-------------------------------------
```python
df = process_eye_tracking(
    "my_data.tsv",
    output="results.tsv",
    velocity_threshold=25.0  # More sensitive
)
```

3. Adaptive sample-based (recommended):
---------------------------------------
```python
from ivt_filter.simple_api import process_eye_tracking_adaptive

df = process_eye_tracking_adaptive(
    "my_data.tsv",
    n_samples=3  # Adapts to your sampling rate
)
```

4. Get statistics:
------------------
```python
from ivt_filter.simple_api import process_eye_tracking, print_statistics

df = process_eye_tracking("my_data.tsv")
print_statistics(df)
```

Input File Format
=================
Your TSV file must contain these columns from Tobii Pro Lab:
- time_ms: Timestamp in milliseconds
- gaze_left_x_mm, gaze_left_y_mm: Left eye gaze position
- gaze_right_x_mm, gaze_right_y_mm: Right eye gaze position
- eye_left_z_mm, eye_right_z_mm: Distance to screen
- validity_left, validity_right: Data validity (0 or 1)

Output Columns
==============
The returned DataFrame includes all original columns plus:
- velocity_deg_per_s: Eye movement velocity in degrees per second
- ivt_sample_type: Classification ("Fixation" or "Saccade")

Velocity Threshold Guide
=========================
- 20 deg/s: Very sensitive, more saccades detected
- 30 deg/s: Standard (default), good for most cases
- 40 deg/s: Less sensitive, only fast saccades
- 50+ deg/s: Very conservative, mainly large saccades

Recommended: Start with 30 deg/s and adjust based on your needs.
"""
