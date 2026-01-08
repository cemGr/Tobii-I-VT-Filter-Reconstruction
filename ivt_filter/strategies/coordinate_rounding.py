# ivt_filter/strategies/coordinate_rounding.py
"""
Strategien für das Runden von Gaze- und Eye-Koordinaten vor der Velocity-Berechnung.

Das Runden von Koordinaten kann die Velocity-Berechnung beeinflussen und wird oft
verwendet, um Rauschen zu reduzieren oder Diskretisierungseffekte zu simulieren.

Beispiel:
    >>> from coordinate_rounding import RoundToNearest, RoundHalfUp, FloorRounding
    >>> 
    >>> # Banker's Rounding (round half to even)
    >>> nearest = RoundToNearest()
    >>> x, y = nearest.round_gaze(201.5, 92.5)
    >>> print(x, y)  # 202.0, 92.0 (0.5 → zur geraden Zahl)
    >>> 
    >>> # Classical Rounding (0.5 always up)
    >>> halfup = RoundHalfUp()
    >>> x, y = halfup.round_gaze(201.5, 92.5)
    >>> print(x, y)  # 202.0, 93.0
    >>> 
    >>> # Floor Rounding
    >>> floor = FloorRounding()
    >>> x, y = floor.round_gaze(201.8, 92.3)
    >>> print(x, y)  # 201.0, 92.0

Rundungseffekte auf Velocity:
    - Kleine Koordinatenänderungen: Rounding kann Velocity auf 0 reduzieren
    - Große Saccaden: 2-5 deg/s Differenz typisch
    - Empfehlung: 'none' für präzise Messungen, 'nearest' oder 'halfup' für Tobii-ähnliche Filter
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class CoordinateRoundingStrategy(ABC):
    """
    Abstrakte Basis für Koordinaten-Rundung.
    
    Rundet Gaze (x, y) und Eye (x, y, z) Koordinaten vor der Winkelberechnung.
    """

    @abstractmethod
    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        """
        Rundet Gaze-Koordinaten (x, y).
        
        Args:
            x: X-Koordinate in mm
            y: Y-Koordinate in mm
            
        Returns:
            Tuple (x_rounded, y_rounded)
        """
        pass

    @abstractmethod
    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Rundet Eye-Koordinaten (x, y, z).
        
        Args:
            x: X-Koordinate in mm
            y: Y-Koordinate in mm
            z: Z-Koordinate (Distanz) in mm
            
        Returns:
            Tuple (x_rounded, y_rounded, z_rounded)
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Beschreibung der Rounding-Strategie."""
        pass


class NoRounding(CoordinateRoundingStrategy):
    """
    Keine Rundung: Koordinaten bleiben wie sie sind (Default).
    """

    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        return x, y

    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return x, y, z

    def get_description(self) -> str:
        return "NoRounding"


class RoundToNearest(CoordinateRoundingStrategy):
    """
    Rundet auf die nächste ganze Zahl (round-to-nearest).
    
    Nutzt Python's built-in round(), welches "Banker's Rounding" verwendet:
    Bei genau 0.5 wird zur nächsten GERADEN Zahl gerundet.
    
    Beispiele:
      - Gaze: (201.8, 92.0) → (202, 92)
      - Gaze: (202.8, 130.9) → (203, 131)
      - Gaze: (206.1, 125.7) → (206, 126)
      - Gaze: (2.5, 3.5) → (2, 4)  ← Banker's Rounding!
      - Eye: (201.8, 92.0, 560.6) → (202, 92, 561)
    """

    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        return round(x), round(y)

    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return round(x), round(y), round(z)

    def get_description(self) -> str:
        return "RoundToNearest (round to whole numbers, Banker's Rounding at 0.5)"


class RoundHalfUp(CoordinateRoundingStrategy):
    """
    Rundet auf die nächste ganze Zahl, bei 0.5 IMMER aufrunden.
    
    Klassisches "round half up" Verhalten.
    
    Beispiele:
      - Gaze: (2.5, 3.5) → (3, 4)  ← Immer aufrunden bei 0.5
      - Gaze: (201.8, 92.0) → (202, 92)
      - Gaze: (0.5, 1.5) → (1, 2)
    """

    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        return np.floor(x + 0.5), np.floor(y + 0.5)

    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return np.floor(x + 0.5), np.floor(y + 0.5), np.floor(z + 0.5)

    def get_description(self) -> str:
        return "RoundHalfUp (always round up at 0.5)"


class FloorRounding(CoordinateRoundingStrategy):
    """
    Rundet immer ab (floor).
    
    Beispiel:
      - Gaze: (201.8, 92.0) → (201, 92)
      - Eye: (201.8, 92.0, 560.6) → (201, 92, 560)
    """

    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        return np.floor(x), np.floor(y)

    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return np.floor(x), np.floor(y), np.floor(z)

    def get_description(self) -> str:
        return "FloorRounding (always round down)"


class CeilRounding(CoordinateRoundingStrategy):
    """
    Rundet immer auf (ceil).
    
    Beispiel:
      - Gaze: (201.8, 92.0) → (202, 92)
      - Eye: (201.8, 92.0, 560.6) → (202, 92, 561)
    """

    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        return np.ceil(x), np.ceil(y)

    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return np.ceil(x), np.ceil(y), np.ceil(z)

    def get_description(self) -> str:
        return "CeilRounding (always round up)"
