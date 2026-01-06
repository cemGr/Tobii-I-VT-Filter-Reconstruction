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
    """Moving Average: rolling mean über ungültige Punkte hinweg."""

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
    Strict Median-Filter: Überspringt Smoothing wenn invalide Samples im Fenster.
    
    Im Gegensatz zu MedianSmoothing, das invalide Samples ignoriert und den Median
    aus den verbleibenden gültigen Werten berechnet, wird hier das Smoothing komplett
    übersprungen wenn nicht ALLE Samples im Fenster gültig sind.
    
    Beispiel (3er-Fenster):
        [Valid, Invalid, Valid] -> MedianSmoothing würde median(val1, val3) berechnen
                                 -> MedianSmoothingStrict gibt NaN zurück
        [Valid, Valid, Valid]   -> Beide berechnen median(val1, val2, val3)
    """

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        result = series.copy()
        n = len(series)
        half_window = self.window_samples // 2
        
        for i in range(n):
            # Bestimme Fenster-Grenzen (zentriert)
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            # Check if ALL samples are valid sind
            window_valid = valid_mask.iloc[start:end]
            
            if window_valid.all():
                # All valid -> berechne Median
                window_values = series.iloc[start:end]
                result.iloc[i] = window_values.median()
            else:
                # At least one invalid sample -> überspringe Smoothing
                # Keep original value wenn Sample selbst gültig ist
                if valid_mask.iloc[i]:
                    result.iloc[i] = series.iloc[i]
                else:
                    result.iloc[i] = np.nan
        
        return result

    def get_description(self) -> str:
        return f"MedianSmoothingStrict(window={self.window_samples}, skip_if_invalid_in_window)"


class MovingAverageSmoothingStrict(SmoothingStrategy):
    """
    Strict Moving Average: Überspringt Smoothing wenn invalide Samples im Fenster.
    
    Verhält sich wie MedianSmoothingStrict, verwendet aber den Mittelwert statt Median.
    """

    def smooth(self, series: pd.Series, valid_mask: pd.Series) -> pd.Series:
        result = series.copy()
        n = len(series)
        half_window = self.window_samples // 2
        
        for i in range(n):
            # Bestimme Fenster-Grenzen (zentriert)
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            # Check if ALL samples are valid sind
            window_valid = valid_mask.iloc[start:end]
            
            if window_valid.all():
                # All valid -> berechne Mittelwert
                window_values = series.iloc[start:end]
                result.iloc[i] = window_values.mean()
            else:
                # At least one invalid sample -> überspringe Smoothing
                if valid_mask.iloc[i]:
                    result.iloc[i] = series.iloc[i]
                else:
                    result.iloc[i] = np.nan
        
        return result

    def get_description(self) -> str:
        return f"MovingAverageSmoothingStrict(window={self.window_samples}, skip_if_invalid_in_window)"


class MedianSmoothingAdaptive(SmoothingStrategy):
    """
    Adaptive Median-Filter: Sammelt nur gültige Samples aus dem Fenster.
    
    Diese Variante sammelt alle gültigen Samples innerhalb des Standard-Fensters
    und kann optional die Suche erweitern, wenn nicht genug gültige Samples gefunden wurden.
    
    Unterschiede zu anderen Varianten:
    - MedianSmoothing: Verwendet pandas rolling, ignoriert NaN automatisch
    - MedianSmoothingStrict: Überspringt Smoothing wenn IRGENDEIN invalid Sample im Fenster
    - MedianSmoothingAdaptive: Sammelt NUR gültige Samples, kann Fenster erweitern
    
    Beispiel (3er-Fenster, min_samples=2, expansion_radius=1):
        [Invalid, Invalid, Valid=10, Valid=20, Valid=30]
        
        Index 2 (Valid=10):
          - Standard-Fenster: [1, 2, 3] -> gültige: [10, 20]
          - min_samples=2 erfüllt -> median([10, 20]) = 15.0
        
        Index 1 (Invalid):
          - Standard-Fenster: [0, 1, 2] -> gültige: [10]
          - min_samples=2 nicht erfüllt -> erweitere um expansion_radius=1
          - Erweitertes Fenster: [-1, 0, 1, 2, 3] -> gültige: [10, 20]
          - min_samples=2 erfüllt -> median([10, 20]) = 15.0
          - ABER: Sample selbst invalid -> result = NaN
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
            # Wenn Sample selbst invalid ist, bleibt es NaN
            if not valid_mask.iloc[i]:
                result.iloc[i] = np.nan
                continue
            
            # Collect valid samples aus Standard-Fenster
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            valid_samples = []
            for j in range(start, end):
                if valid_mask.iloc[j]:
                    valid_samples.append(series.iloc[j])
            
            # If not enough valid samples -> erweitere Suche
            if len(valid_samples) < self.min_samples and self.expansion_radius > 0:
                # Erweitere das Fenster symmetrisch um expansion_radius
                expanded_start = max(0, start - self.expansion_radius)
                expanded_end = min(n, end + self.expansion_radius)
                
                # Sammle aus erweitertem Fenster (ohne Duplikate)
                valid_samples = []
                for j in range(expanded_start, expanded_end):
                    if valid_mask.iloc[j]:
                        valid_samples.append(series.iloc[j])
            
            # Calculate median wenn genug gültige Samples
            if len(valid_samples) >= self.min_samples:
                result.iloc[i] = np.median(valid_samples)
            else:
                # Not enough valid samples -> behalte Original
                result.iloc[i] = series.iloc[i]
        
        return result

    def get_description(self) -> str:
        return (f"MedianSmoothingAdaptive(window={self.window_samples}, "
                f"min_samples={self.min_samples}, expansion_radius={self.expansion_radius})")


class MovingAverageSmoothingAdaptive(SmoothingStrategy):
    """
    Adaptive Moving Average: Sammelt nur gültige Samples aus dem Fenster.
    
    Verhält sich wie MedianSmoothingAdaptive, verwendet aber den Mittelwert statt Median.
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
            # Wenn Sample selbst invalid ist, bleibt es NaN
            if not valid_mask.iloc[i]:
                result.iloc[i] = np.nan
                continue
            
            # Collect valid samples aus Standard-Fenster
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            valid_samples = []
            for j in range(start, end):
                if valid_mask.iloc[j]:
                    valid_samples.append(series.iloc[j])
            
            # If not enough valid samples -> erweitere Suche
            if len(valid_samples) < self.min_samples and self.expansion_radius > 0:
                # Erweitere das Fenster symmetrisch um expansion_radius
                expanded_start = max(0, start - self.expansion_radius)
                expanded_end = min(n, end + self.expansion_radius)
                
                # Sammle aus erweitertem Fenster (ohne Duplikate)
                valid_samples = []
                for j in range(expanded_start, expanded_end):
                    if valid_mask.iloc[j]:
                        valid_samples.append(series.iloc[j])
            
            # Calculate mean wenn genug gültige Samples
            if len(valid_samples) >= self.min_samples:
                result.iloc[i] = np.mean(valid_samples)
            else:
                # Not enough valid samples -> behalte Original
                result.iloc[i] = series.iloc[i]
        
        return result

    def get_description(self) -> str:
        return (f"MovingAverageSmoothingAdaptive(window={self.window_samples}, "
                f"min_samples={self.min_samples}, expansion_radius={self.expansion_radius})")
