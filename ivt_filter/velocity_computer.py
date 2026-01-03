# ivt_filter/velocity_computer.py
"""High-level velocity computation orchestrator.

Breaks down the large compute_olsen_velocity function into manageable components (SRP).
"""
from __future__ import annotations

from typing import Optional
import dataclasses
import math

import numpy as np
import pandas as pd

from .config import OlsenVelocityConfig
from .constants import PhysicalConstants
from .gaze import prepare_combined_columns, smooth_combined_gaze, gap_fill_gaze
from .windowing import (
    WindowSelector,
    FixedSampleSymmetricWindowSelector,
    AsymmetricNeighborWindowSelector,
)
from .velocity import make_window_selector, _get_coordinate_rounding_strategy, _get_velocity_calculation_strategy


class SamplingAnalyzer:
    """Analyzes sampling rate and timing characteristics.
    
    Responsibilities:
        - Calculate median/mean time deltas
        - Determine effective sampling rate
        - Suggest nominal rates
        - Auto-convert window size from ms to samples
    """

    NOMINAL_RATES = [30.0, 50.0, 60.0, 120.0, 150.0, 250.0, 300.0, 500.0, 600.0, 1000.0]

    def __init__(self, cfg: OlsenVelocityConfig):
        self.cfg = cfg

    def analyze(self, times: np.ndarray) -> tuple[Optional[float], OlsenVelocityConfig]:
        """Analyze sampling rate and potentially auto-adjust window config.
        
        Returns:
            Tuple of (median_dt, updated_config)
        """
        n = len(times)
        if n < 2:
            return None, self.cfg

        # Extract time differences
        dt = self._extract_time_differences(times, n)
        if dt.size == 0:
            return None, self.cfg

        # Calculate delta time statistic
        dt_med = self._calculate_dt_statistic(dt)
        hz_measured = 1000.0 / dt_med if dt_med > 0 else float("nan")

        # Print sampling info
        self._print_sampling_info(dt_med, hz_measured, n)

        # Auto-convert window size if needed
        updated_cfg = self._auto_convert_window(dt_med)

        return dt_med, updated_cfg

    def _extract_time_differences(self, times: np.ndarray, n: int) -> np.ndarray:
        """Extract time differences based on sampling method."""
        if self.cfg.sampling_rate_method == "first_100":
            n_samples = min(100, n - 1)
            dt = np.diff(times[:n_samples + 1])
            self.method_desc = f"first {n_samples} samples"
        else:
            dt = np.diff(times)
            self.method_desc = "all samples"

        return dt[np.isfinite(dt)]

    def _calculate_dt_statistic(self, dt: np.ndarray) -> float:
        """Calculate median or mean of time differences."""
        if self.cfg.dt_calculation_method == "median":
            return float(np.median(dt))
        else:
            return float(np.mean(dt))

    def _print_sampling_info(self, dt_med: float, hz_measured: float, n: int) -> None:
        """Print sampling analysis information."""
        method_name = self.cfg.dt_calculation_method
        print(
            f"[Sampling] {method_name} dt = {dt_med:.3f} ms -> "
            f"measured ~{hz_measured:.1f} Hz (using {self.method_desc})"
        )

        if math.isfinite(hz_measured):
            nearest_nom = min(self.NOMINAL_RATES, key=lambda f: abs(f - hz_measured))
            print(f"[Sampling] nearest nominal rate: {nearest_nom:.1f} Hz")

    def _auto_convert_window(self, dt_med: float) -> OlsenVelocityConfig:
        """Auto-convert window from milliseconds to samples if configured."""
        should_auto_convert = (
            self.cfg.auto_fixed_window_from_ms or 
            self.cfg.symmetric_round_window
        )

        if not should_auto_convert or self.cfg.fixed_window_samples is not None or dt_med <= 0:
            return self.cfg

        # Calculate sample window size
        n_intervals = max(1, int(round(self.cfg.window_length_ms / dt_med)))
        n_samples = n_intervals + 1

        if n_samples < 3:
            n_samples = 3
        if n_samples % 2 == 0:
            n_samples += 1

        effective_ms = (n_samples - 1) * dt_med
        per_side = (n_samples - 1) / 2.0

        print(
            f"[Window] auto sample window: {n_samples} samples total "
            f"(~{per_side:.1f} per side around center, "
            f"effective span ~{effective_ms:.2f} ms)"
        )

        return dataclasses.replace(self.cfg, fixed_window_samples=n_samples)


class VelocityComputer:
    """Computes Olsen-style velocity for eye tracking data.
    
    Responsibilities:
        - Orchestrate velocity computation pipeline
        - Prepare data arrays
        - Initialize strategies
        - Compute velocities for all samples
        - Add debug columns
    """

    def __init__(self, cfg: OlsenVelocityConfig):
        self.cfg = cfg
        self.sampling_analyzer = SamplingAnalyzer(cfg)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute velocity for all samples in DataFrame.
        
        Args:
            df: Input DataFrame with gaze data
            
        Returns:
            DataFrame with added velocity columns
        """
        # Step 1: Prepare data
        df = self._prepare_dataframe(df)

        # Step 2: Analyze sampling
        times = df["time_ms"].to_numpy()
        dt_med, self.cfg = self.sampling_analyzer.analyze(times)

        # Step 3: Initialize strategies
        window_selector = make_window_selector(self.cfg)
        coord_rounding = _get_coordinate_rounding_strategy(self.cfg.coordinate_rounding)
        velocity_strategy = _get_velocity_calculation_strategy(self.cfg.velocity_method)

        self._print_strategy_info(window_selector, coord_rounding, velocity_strategy)

        # Step 4: Extract data arrays
        data_arrays = self._extract_data_arrays(df)

        # Step 5: Compute velocities
        df = self._compute_all_velocities(
            df, 
            data_arrays, 
            window_selector,
            coord_rounding,
            velocity_strategy,
            dt_med
        )

        return df

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with initial processing steps."""
        df = df.sort_values("time_ms").reset_index(drop=True)
        df = gap_fill_gaze(df, self.cfg)
        df = prepare_combined_columns(df, self.cfg)
        df = smooth_combined_gaze(df, self.cfg)

        # Add output columns
        df = df.copy()
        df["velocity_deg_per_sec"] = float("nan")
        df["window_width_samples"] = pd.NA
        
        # Debug columns
        df["env_has_invalid_above"] = pd.NA
        df["env_has_invalid_below"] = pd.NA
        df["env_rule_triggered"] = pd.NA
        df["gap_rule_triggered"] = pd.NA
        df["gap_left_invalid_idx"] = pd.NA
        df["gap_right_invalid_idx"] = pd.NA

        return df

    def _extract_data_arrays(self, df: pd.DataFrame) -> dict:
        """Extract numpy arrays from DataFrame for fast computation."""
        eye_mode = getattr(self.cfg, "eye_mode", "average")
        
        if eye_mode == "left":
            valid = df["left_eye_valid"].to_numpy()
        elif eye_mode == "right":
            valid = df["right_eye_valid"].to_numpy()
        else:
            valid = df["combined_valid"].to_numpy()

        return {
            "times": df["time_ms"].to_numpy(),
            "cx": df["smoothed_x_mm"].to_numpy(),
            "cy": df["smoothed_y_mm"].to_numpy(),
            "cex": df["eye_x_mm"].to_numpy(),
            "cey": df["eye_y_mm"].to_numpy(),
            "cz": df["eye_z_mm"].to_numpy(),
            "valid": valid,
            "left_valid": df["left_eye_valid"].to_numpy(),
            "right_valid": df["right_eye_valid"].to_numpy(),
            "lx": df["gaze_left_x_mm"].to_numpy(),
            "ly": df["gaze_left_y_mm"].to_numpy(),
            "rx": df["gaze_right_x_mm"].to_numpy(),
            "ry": df["gaze_right_y_mm"].to_numpy(),
        }

    def _print_strategy_info(self, window_selector, coord_rounding, velocity_strategy) -> None:
        """Print information about selected strategies."""
        print(f"[DEBUG] Window selector: {type(window_selector).__name__}")

        if self.cfg.coordinate_rounding != "none":
            print(f"[Rounding] Coordinate rounding: {coord_rounding.get_description()}")

        if self.cfg.velocity_method != "olsen2d":
            print(f"[Velocity] Calculation method: {self.cfg.velocity_method}")

        if isinstance(window_selector, FixedSampleSymmetricWindowSelector):
            print(f"[Window] Fixed sample window: {self.cfg.fixed_window_samples} samples")
        elif isinstance(window_selector, AsymmetricNeighborWindowSelector):
            print(f"[Window] Asymmetric neighbor window: 2 samples (backward/forward)")

    def _compute_all_velocities(
        self,
        df: pd.DataFrame,
        data_arrays: dict,
        window_selector,
        coord_rounding,
        velocity_strategy,
        dt_med: Optional[float],
    ) -> pd.DataFrame:
        """Compute velocities for all samples.
        
        This method would contain the main loop from the original function.
        For now, we delegate to the original implementation to maintain compatibility.
        """
        # This is where the main velocity computation loop would go
        # For now, import and call the original function to maintain backward compatibility
        from .velocity import compute_olsen_velocity as original_compute
        return original_compute(df, self.cfg)


def compute_velocity(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """Compute Olsen-style velocity (new interface).
    
    Args:
        df: DataFrame with gaze data
        cfg: Velocity configuration
        
    Returns:
        DataFrame with velocity columns added
    """
    computer = VelocityComputer(cfg)
    return computer.compute(df)
