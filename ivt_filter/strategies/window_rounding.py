# ivt_filter/strategies/window_rounding.py
"""
Strategien für Fenster-Rounding/Sizing nach Selection.

Bestimmen, wie die selected_size in eine half_size für FixedSampleSymmetricWindowSelector
umgerechnet wird.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class WindowRoundingStrategy(ABC):
    """
    Bestimmt, wie eine Sample-Fensterbreite zu einer half_size konvertiert wird.
    """

    @abstractmethod
    def calculate_half_size(self, window_size: int) -> int:
        """
        Konvertiert window_size zu half_size.
        
        Args:
            window_size: Die berechnete oder eingegebene Fensterbreite in Samples
            
        Returns:
            half_size: Die verwendete Größe für FixedSampleSymmetricWindowSelector
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Beschreibung der Rounding-Strategie."""
        pass


class StandardWindowRounding(WindowRoundingStrategy):
    """
    Standard-Rounding: Stelle sicher, dass window_size ungerade ist,
    dann berechne half_size = (window_size - 1) // 2.
    
    Beispiele:
      - window_size=7 (ungerade) → half_size=3 → effektive Größe=7
      - window_size=8 (gerade) → wird zu 9 → half_size=4 → effektive Größe=9
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
    Symmetrische Rundung: Berechne per_side = round(window_size / 2),
    dann effektive Größe = 2*per_side + 1.
    
    Kann die Fensterbreite erhöhen:
      - window_size=7 → per_side=round(3.5)=4 → effektive Größe=9
      - window_size=9 → per_side=round(4.5)=4 → effektive Größe=9
      - window_size=8 → per_side=round(4.0)=4 → effektive Größe=9
    
    Die Effektivgröße kann also WACHSEN gegenüber der Eingabe.
    """

    def calculate_half_size(self, window_size: int) -> int:
        n = int(window_size)
        per_side = round(n / 2.0)
        return per_side

    def get_description(self) -> str:
        return "SymmetricRound: per_side=round(n/2), effective_size=2*per_side+1"
