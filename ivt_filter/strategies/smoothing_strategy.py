# ivt_filter/smoothing_strategy.py
"""
Strategies for spatial smoothing of gaze data.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class SmoothingStrategy(ABC):
    """Abstract base for smoothing strategies."""

    def __init__(self, window_samples: int = 5):
        self.window_samples = max(1, int(window_samples))
        if self.window_samples % 2 == 0:
            self.window_samples += 1

    @abstractmethod
    def smooth(
        self,
        series: pd.Series,
        valid_mask: pd.Series,
    ) -> pd.Series:
        """
        Apply smoothing to a series.

        Args:
            series: The data to be smoothed
            valid_mask: Boolean mask indicating where data is valid

        Returns:
            Smoothed series
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Description of the smoothing strategy."""
        pass


class NoSmoothing(SmoothingStrategy):
    """No smoothing: return the series as-is."""

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        return series.copy()

    def get_description(self) -> str:
        return "NoSmoothing"


class MedianSmoothing(SmoothingStrategy):
    """Median filter: rolling median across invalid points."""

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        # Only consider valid values
        filtered = series.where(valid_mask)
        smoothed = filtered.rolling(
            window=self.window_samples,
            center=True,
            min_periods=1
        ).median()
        return smoothed

    def get_description(self) -> str:
        return f"MedianSmoothing(window={self.window_samples})"


class MovingAverageSmoothing(SmoothingStrategy):
    """Moving average: rolling mean across invalid points."""

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        # Only consider valid values
        filtered = series.where(valid_mask)
        smoothed = filtered.rolling(
            window=self.window_samples,
            center=True,
            min_periods=1
        ).mean()
        return smoothed

    def get_description(self) -> str:
        return f"MovingAverageSmoothing(window={self.window_samples})"


class MedianSmoothingStrict(SmoothingStrategy):
    """
    Strict median filter: skips smoothing if there are invalid samples in the window.

    In contrast to MedianSmoothing, which ignores invalid samples and computes the
    median from the remaining valid values, here smoothing is skipped entirely
    if not ALL samples in the window are valid.

    Example (3-sample window):
        [Valid, Invalid, Valid] -> MedianSmoothing would compute median(val1, val3)
                                 -> MedianSmoothingStrict returns NaN
        [Valid, Valid, Valid]   -> Both compute median(val1, val2, val3)
    """

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        result = series.copy()
        n = len(series)
        half_window = self.window_samples // 2
        
        for i in range(n):
            # Determine window bounds (centered)
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Check if ALL samples are valid
            window_valid = valid_mask.iloc[start:end]

            if window_valid.all():
                # All valid -> compute median
                window_values = series.iloc[start:end]
                result.iloc[i] = window_values.median()
            else:
                # At least one invalid sample -> skip smoothing
                # Keep original value if the sample itself is valid
                if valid_mask.iloc[i]:
                    result.iloc[i] = series.iloc[i]
                else:
                    result.iloc[i] = np.nan
        
        return result

    def get_description(self) -> str:
        return f"MedianSmoothingStrict(window={self.window_samples}, skip_if_invalid_in_window)"


class MovingAverageSmoothingStrict(SmoothingStrategy):
    """
    Strict moving average: skips smoothing if there are invalid samples in the window.

    Behaves like MedianSmoothingStrict, but uses the mean instead of the median.
    """

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        result = series.copy()
        n = len(series)
        half_window = self.window_samples // 2
        
        for i in range(n):
            # Determine window bounds (centered)
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Check if ALL samples are valid
            window_valid = valid_mask.iloc[start:end]

            if window_valid.all():
                # All valid -> compute mean
                window_values = series.iloc[start:end]
                result.iloc[i] = window_values.mean()
            else:
                # At least one invalid sample -> skip smoothing
                if valid_mask.iloc[i]:
                    result.iloc[i] = series.iloc[i]
                else:
                    result.iloc[i] = np.nan
        
        return result

    def get_description(self) -> str:
        return f"MovingAverageSmoothingStrict(window={self.window_samples}, skip_if_invalid_in_window)"


class MedianSmoothingAdaptive(SmoothingStrategy):
    """
    Adaptive median filter: collects only valid samples from the window.

    This variant collects all valid samples within the standard window
    and can optionally expand the search if not enough valid samples were found.

    Differences from other variants:
    - MedianSmoothing: Uses pandas rolling, ignores NaN automatically
    - MedianSmoothingStrict: Skips smoothing if ANY invalid sample is in the window
    - MedianSmoothingAdaptive: Collects ONLY valid samples, can expand the window

    Example (3-sample window, min_samples=2, expansion_radius=1):
        [Invalid, Invalid, Valid=10, Valid=20, Valid=30]

        Index 2 (Valid=10):
          - Standard window: [1, 2, 3] -> valid: [10, 20]
          - min_samples=2 satisfied -> median([10, 20]) = 15.0

        Index 1 (Invalid):
          - Standard window: [0, 1, 2] -> valid: [10]
          - min_samples=2 not satisfied -> expand by expansion_radius=1
          - Expanded window: [-1, 0, 1, 2, 3] -> valid: [10, 20]
          - min_samples=2 satisfied -> median([10, 20]) = 15.0
          - BUT: sample itself invalid -> result = NaN
    """

    def __init__(self, window_samples: int = 5, min_samples: int = 1, expansion_radius: int = 0):
        super().__init__(window_samples)
        self.min_samples = max(1, int(min_samples))
        self.expansion_radius = max(0, int(expansion_radius))

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        result = series.copy()
        n = len(series)
        half_window = self.window_samples // 2
        
        for i in range(n):
            # If the sample itself is invalid, it stays NaN
            if not valid_mask.iloc[i]:
                result.iloc[i] = np.nan
                continue

            # Collect valid samples from the standard window
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            valid_samples = []
            for j in range(start, end):
                if valid_mask.iloc[j]:
                    valid_samples.append(series.iloc[j])

            # If not enough valid samples -> expand the search
            if len(valid_samples) < self.min_samples and self.expansion_radius > 0:
                # Expand the window symmetrically by expansion_radius
                expanded_start = max(0, start - self.expansion_radius)
                expanded_end = min(n, end + self.expansion_radius)

                # Collect from the expanded window (no duplicates)
                valid_samples = []
                for j in range(expanded_start, expanded_end):
                    if valid_mask.iloc[j]:
                        valid_samples.append(series.iloc[j])

            # Compute median if there are enough valid samples
            if len(valid_samples) >= self.min_samples:
                result.iloc[i] = np.median(valid_samples)
            else:
                # Not enough valid samples -> keep the original
                result.iloc[i] = series.iloc[i]
        
        return result

    def get_description(self) -> str:
        return (f"MedianSmoothingAdaptive(window={self.window_samples}, "
                f"min_samples={self.min_samples}, expansion_radius={self.expansion_radius})")


class MovingAverageSmoothingAdaptive(SmoothingStrategy):
    """
    Adaptive moving average: collects only valid samples from the window.

    Behaves like MedianSmoothingAdaptive, but uses the mean instead of the median.
    """

    def __init__(self, window_samples: int = 5, min_samples: int = 1, expansion_radius: int = 0):
        super().__init__(window_samples)
        self.min_samples = max(1, int(min_samples))
        self.expansion_radius = max(0, int(expansion_radius))

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        result = series.copy()
        n = len(series)
        half_window = self.window_samples // 2
        
        for i in range(n):
            # If the sample itself is invalid, it stays NaN
            if not valid_mask.iloc[i]:
                result.iloc[i] = np.nan
                continue

            # Collect valid samples from the standard window
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            valid_samples = []
            for j in range(start, end):
                if valid_mask.iloc[j]:
                    valid_samples.append(series.iloc[j])

            # If not enough valid samples -> expand the search
            if len(valid_samples) < self.min_samples and self.expansion_radius > 0:
                # Expand the window symmetrically by expansion_radius
                expanded_start = max(0, start - self.expansion_radius)
                expanded_end = min(n, end + self.expansion_radius)

                # Collect from the expanded window (no duplicates)
                valid_samples = []
                for j in range(expanded_start, expanded_end):
                    if valid_mask.iloc[j]:
                        valid_samples.append(series.iloc[j])

            # Compute mean if there are enough valid samples
            if len(valid_samples) >= self.min_samples:
                result.iloc[i] = np.mean(valid_samples)
            else:
                # Not enough valid samples -> keep the original
                result.iloc[i] = series.iloc[i]
        
        return result

    def get_description(self) -> str:
        return (f"MovingAverageSmoothingAdaptive(window={self.window_samples}, "
                f"min_samples={self.min_samples}, expansion_radius={self.expansion_radius})")
