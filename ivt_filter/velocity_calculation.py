# ivt_filter/velocity_calculation.py
"""
Strategien für die Velocity-Berechnung.

Verschiedene Methoden zur Berechnung des visuellen Winkels zwischen zwei Gaze-Punkten.

Beispiel:
    >>> from velocity_calculation import Olsen2DApproximation, Ray3DAngle
    >>> 
    >>> # Olsen 2D: Schnell, nur eye_z nötig
    >>> olsen = Olsen2DApproximation()
    >>> angle = olsen.calculate_visual_angle(
    ...     x1_mm=516.4, y1_mm=293.0,
    ...     x2_mm=520.0, y2_mm=299.8,
    ...     eye_x_mm=None, eye_y_mm=None, eye_z_mm=582.4
    ... )
    >>> velocity = angle / 0.02  # dt = 20ms
    >>> print(f"Olsen 2D: {velocity:.2f} deg/s")  # ~37.84 deg/s
    >>> 
    >>> # Ray 3D: Physikalisch korrekt, benötigt eye_x, eye_y, eye_z
    >>> ray3d = Ray3DAngle()
    >>> angle = ray3d.calculate_visual_angle(
    ...     x1_mm=516.4, y1_mm=293.0,
    ...     x2_mm=520.0, y2_mm=299.8,
    ...     eye_x_mm=255.4, eye_y_mm=99.5, eye_z_mm=582.4
    ... )
    >>> velocity = angle / 0.02
    >>> print(f"Ray 3D: {velocity:.2f} deg/s")  # ~29.54 deg/s
    >>> 
    >>> # Unterschied: Olsen 2D ist ~22% höher bei off-center Positionen

Vergleich der Methoden:
    
    Olsen 2D Approximation:
        - Schnell (nur sqrt + atan2)
        - Benötigt nur eye_z (Distanz)
        - 2D-Näherung auf Screen-Ebene
        - Abweichung: 1-5% bei typischen Positionen, bis 22% bei extremen off-center
        
    Ray 3D Angle:
        - Physikalisch korrekt
        - Benötigt vollständige Eye-Position (x, y, z)
        - Berücksichtigt 3D-Geometrie
        - Etwas langsamer (vector math + acos)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import math
import numpy as np


class VelocityCalculationStrategy(ABC):
    """
    Abstrakte Basis für Velocity-Berechnungs-Strategien.
    
    Berechnet den visuellen Winkel zwischen Start- und End-Gaze-Punkt.
    """

    @abstractmethod
    def calculate_visual_angle(
        self,
        x1_mm: float,
        y1_mm: float,
        x2_mm: float,
        y2_mm: float,
        eye_x_mm: Optional[float],
        eye_y_mm: Optional[float],
        eye_z_mm: Optional[float],
    ) -> float:
        """
        Berechnet den visuellen Winkel in Grad zwischen zwei Punkten.
        
        Args:
            x1_mm, y1_mm: Start-Gaze-Position auf dem Screen (mm)
            x2_mm, y2_mm: End-Gaze-Position auf dem Screen (mm)
            eye_x_mm: X-Position des Auges (mm)
            eye_y_mm: Y-Position des Auges (mm)
            eye_z_mm: Z-Position (Distanz) des Auges zum Screen (mm)
            
        Returns:
            Visueller Winkel in Grad
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Beschreibung der Berechnungs-Strategie."""
        pass


class Olsen2DApproximation(VelocityCalculationStrategy):
    """
    Original Olsen-Style 2D-Näherung (Standard).
    
    Methode:
      1. Berechne 2D-Distanz auf dem Screen: s = √(Δx² + Δy²)
      2. Verwende Kleinwinkel-Näherung: θ ≈ arctan(s / d)
      
    Verwendet nur eye_z (Distanz), ignoriert eye_x und eye_y.
    Screen liegt in der XY-Ebene, Auge schaut orthogonal darauf.
    
    Schnell, aber weniger präzise bei:
      - Off-center Gaze (Auge nicht zentral vor Screen)
      - Großen Winkeln
    """

    def calculate_visual_angle(
        self,
        x1_mm: float,
        y1_mm: float,
        x2_mm: float,
        y2_mm: float,
        eye_x_mm: Optional[float],
        eye_y_mm: Optional[float],
        eye_z_mm: Optional[float],
    ) -> float:
        dx = float(x2_mm) - float(x1_mm)
        dy = float(y2_mm) - float(y1_mm)
        s_mm = math.hypot(dx, dy)

        # Distanz Auge-Screen (Fallback: 600mm)
        if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
            d_mm = 600.0
        else:
            d_mm = float(eye_z_mm)

        theta_rad = math.atan2(s_mm, d_mm)
        return math.degrees(theta_rad)

    def get_description(self) -> str:
        return "Olsen 2D Approximation: θ = atan(screen_distance / eye_z)"


class Ray3DAngle(VelocityCalculationStrategy):
    """
    Physikalisch korrekte 3D-Ray-Methode.
    
    Methode:
      1. Screen liegt bei z=0, Gaze-Punkte: G₀ = (gx₀, gy₀, 0), G₁ = (gx₁, gy₁, 0)
      2. Eye-Position: E = (ex, ey, ez)
      3. Ray-Vektoren: d₀ = G₀ - E, d₁ = G₁ - E
      4. Winkel zwischen Rays: θ = arccos(d₀ · d₁ / (|d₀| × |d₁|))
      
    Vorteile:
      - Physikalisch korrekt
      - Funktioniert auch off-center
      - Berücksichtigt volle 3D-Geometrie
      
    Benötigt vollständige Eye-Position (x, y, z).
    """

    def calculate_visual_angle(
        self,
        x1_mm: float,
        y1_mm: float,
        x2_mm: float,
        y2_mm: float,
        eye_x_mm: Optional[float],
        eye_y_mm: Optional[float],
        eye_z_mm: Optional[float],
    ) -> float:
        # Eye Position (Fallback auf Defaults wenn nicht verfügbar)
        if eye_x_mm is None or not math.isfinite(eye_x_mm):
            eye_x_mm = 0.0
        if eye_y_mm is None or not math.isfinite(eye_y_mm):
            eye_y_mm = 0.0
        if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
            eye_z_mm = 600.0

        ex, ey, ez = float(eye_x_mm), float(eye_y_mm), float(eye_z_mm)

        # Screen liegt bei z=0
        gx0, gy0, gz0 = float(x1_mm), float(y1_mm), 0.0
        gx1, gy1, gz1 = float(x2_mm), float(y2_mm), 0.0

        # Ray-Vektoren vom Auge zu den Gaze-Punkten
        d0x = gx0 - ex
        d0y = gy0 - ey
        d0z = gz0 - ez

        d1x = gx1 - ex
        d1y = gy1 - ey
        d1z = gz1 - ez

        # Skalarprodukt
        dot_product = d0x * d1x + d0y * d1y + d0z * d1z

        # Normen (Längen der Vektoren)
        norm0 = math.sqrt(d0x**2 + d0y**2 + d0z**2)
        norm1 = math.sqrt(d1x**2 + d1y**2 + d1z**2)

        # Numerische Stabilität: prüfe Division durch Null
        if norm0 == 0.0 or norm1 == 0.0:
            return 0.0

        # cos(θ) = dot / (norm0 * norm1)
        cos_theta = dot_product / (norm0 * norm1)

        # Clamp auf [-1, 1] für numerische Stabilität
        cos_theta = max(-1.0, min(1.0, cos_theta))

        # Winkel berechnen
        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)

        return theta_deg

    def get_description(self) -> str:
        return "3D Ray Angle: θ = acos(ray0 · ray1 / (|ray0| × |ray1|))"
