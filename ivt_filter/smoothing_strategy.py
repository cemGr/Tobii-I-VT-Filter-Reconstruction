# ivt_filter/smoothing_strategy.py
"""
Strategien für räumliches Smoothing von Gaze-Daten.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class SmoothingStrategy(ABC):
    """Abstrakte Basis für Smoothing-Strategien."""

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
        Smoothing auf eine Serie anwenden.
        
        Args:
            series: Die zu glättenden Daten
            valid_mask: Boolean-Maske, wo Daten gültig sind
            
        Returns:
            Geglättete Serie
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Beschreibung der Smoothing-Strategie."""
        pass


class NoSmoothing(SmoothingStrategy):
    """Keine Glättung: gebe Serie wie sie ist zurück."""

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        return series.copy()

    def get_description(self) -> str:
        return "NoSmoothing"


class MedianSmoothing(SmoothingStrategy):
    """Median-Filter: rolling median über ungültige Punkte hinweg."""

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        # Nur gültige Werte berücksichtigen
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
    """Moving Average: rolling mean über ungültige Punkte hinweg."""

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        # Nur gültige Werte berücksichtigen
        filtered = series.where(valid_mask)
        smoothed = filtered.rolling(
            window=self.window_samples,
            center=True,
            min_periods=1
        ).mean()
        return smoothed

    def get_description(self) -> str:
        return f"MovingAverageSmoothing(window={self.window_samples})"
