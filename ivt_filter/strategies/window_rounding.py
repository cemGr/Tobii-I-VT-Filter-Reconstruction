# ivt_filter/strategies/window_rounding.py
"""
Strategies for window rounding/sizing after selection.

Determine how the selected_size is converted into a half_size for
FixedSampleSymmetricWindowSelector.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class WindowRoundingStrategy(ABC):
    """
    Determines how a sample window width is converted to a half_size.
    """

    @abstractmethod
    def calculate_half_size(self, window_size: int) -> int:
        """
        Convert window_size to half_size.

        Args:
            window_size: The computed or provided window width in samples

        Returns:
            half_size: The size used for FixedSampleSymmetricWindowSelector
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Description of the rounding strategy."""
        pass


class StandardWindowRounding(WindowRoundingStrategy):
    """
    Standard rounding: ensure that window_size is odd,
    then compute half_size = (window_size - 1) // 2.

    Examples:
      - window_size=7 (odd) → half_size=3 → effective size=7
      - window_size=8 (even) → becomes 9 → half_size=4 → effective size=9
    """

    def calculate_half_size(self, window_size: int) -> int:
        n = int(window_size)
        if n % 2 == 0:
            n += 1
        return (n - 1) // 2

    def get_description(self) -> str:
        return "Standard: ensure odd size, then half_size=(n-1)//2"


class SymmetricRoundWindowStrategy(WindowRoundingStrategy):
    """
    Symmetric rounding: compute per_side = round(window_size / 2),
    then effective size = 2*per_side + 1.

    Can increase the window width:
      - window_size=7 → per_side=round(3.5)=4 → effective size=9
      - window_size=9 → per_side=round(4.5)=4 → effective size=9
      - window_size=8 → per_side=round(4.0)=4 → effective size=9

    The effective size can therefore GROW relative to the input.
    """

    def calculate_half_size(self, window_size: int) -> int:
        n = int(window_size)
        per_side = round(n / 2.0)
        return per_side

    def get_description(self) -> str:
        return "SymmetricRound: per_side=round(n/2), effective_size=2*per_side+1"
