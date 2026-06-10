# ivt_filter/strategies/coordinate_rounding.py
"""
Strategies for rounding gaze and eye coordinates before velocity calculation.

Rounding coordinates can affect the velocity calculation and is often
used to reduce noise or to simulate discretization effects.

Example:
    >>> from coordinate_rounding import RoundToNearest, RoundHalfUp, FloorRounding
    >>> 
    >>> # Banker's Rounding (round half to even)
    >>> nearest = RoundToNearest()
    >>> x, y = nearest.round_gaze(201.5, 92.5)
    >>> print(x, y)  # 202.0, 92.0 (0.5 → to the even number)
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

Rounding effects on velocity:
    - Small coordinate changes: rounding can reduce velocity to 0
    - Large saccades: 2-5 deg/s difference is typical
    - Recommendation: 'none' for precise measurements, 'nearest' or 'halfup' for Tobii-like filters
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class CoordinateRoundingStrategy(ABC):
    """
    Abstract base for coordinate rounding.

    Rounds gaze (x, y) and eye (x, y, z) coordinates before the angle calculation.
    """

    @abstractmethod
    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        """
        Round gaze coordinates (x, y).

        Args:
            x: X coordinate in mm
            y: Y coordinate in mm

        Returns:
            Tuple (x_rounded, y_rounded)
        """
        pass

    @abstractmethod
    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Round eye coordinates (x, y, z).

        Args:
            x: X coordinate in mm
            y: Y coordinate in mm
            z: Z coordinate (distance) in mm

        Returns:
            Tuple (x_rounded, y_rounded, z_rounded)
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Description of the rounding strategy."""
        pass


class NoRounding(CoordinateRoundingStrategy):
    """
    No rounding: coordinates stay as they are (default).
    """

    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        return x, y

    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return x, y, z

    def get_description(self) -> str:
        return "NoRounding"


class RoundToNearest(CoordinateRoundingStrategy):
    """
    Rounds to the nearest integer (round-to-nearest).

    Uses Python's built-in round(), which applies "Banker's Rounding":
    at exactly 0.5 it rounds to the nearest EVEN number.

    Examples:
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
    Rounds to the nearest integer, ALWAYS rounding up at 0.5.

    Classic "round half up" behavior.

    Examples:
      - Gaze: (2.5, 3.5) → (3, 4)  ← Always round up at 0.5
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
    Always rounds down (floor).

    Example:
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
    Always rounds up (ceil).

    Example:
      - Gaze: (201.8, 92.0) → (202, 92)
      - Eye: (201.8, 92.0, 560.6) → (202, 92, 561)
    """

    def round_gaze(self, x: float, y: float) -> Tuple[float, float]:
        return np.ceil(x), np.ceil(y)

    def round_eye(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return np.ceil(x), np.ceil(y), np.ceil(z)

    def get_description(self) -> str:
        return "CeilRounding (always round up)"
