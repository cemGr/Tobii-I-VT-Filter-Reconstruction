# ivt_filter/velocity_calculation.py
"""
Strategies for velocity calculation.

Different methods for calculating visual angle between two gaze points.

Comparison of methods:
    
    Olsen 2D Approximation:
        - Fast (only sqrt + atan2)
        - Requires only eye_z (distance)
        - 2D approximation on screen plane
        - Deviation: 1-5% for typical positions, up to 22% for extreme off-center
        
    Ray 3D Angle:
        - Physically correct
        - Requires full eye position (x, y, z)
        - Accounts for 3D geometry
        - Slightly slower (vector math + acos)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

from .constants import PhysicalConstants


@dataclass
class VelocityContext:
    """Container for velocity calculation inputs."""

    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    eye_x_mm: Optional[float]
    eye_y_mm: Optional[float]
    eye_z_mm: Optional[float]
    dir1: Optional[np.ndarray] = None
    dir2: Optional[np.ndarray] = None


class VelocityCalculationStrategy(ABC):
    """Base class for velocity calculation strategies."""

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
        """Calculate visual angle in degrees between two points."""
        pass

    def calculate_visual_angle_ctx(self, ctx: VelocityContext) -> float:
        """Default adapter to use dataclass context."""
        return self.calculate_visual_angle(
            ctx.x1_mm,
            ctx.y1_mm,
            ctx.x2_mm,
            ctx.y2_mm,
            ctx.eye_x_mm,
            ctx.eye_y_mm,
            ctx.eye_z_mm,
        )

    @abstractmethod
    def get_description(self) -> str:
        """Description of calculation strategy."""
        pass


class Olsen2DApproximation(VelocityCalculationStrategy):
    """Original Olsen-style 2D approximation (default).
    
    Method:
      1. Calculate 2D distance on screen: s = √(Δx² + Δy²)
      2. Use small angle approximation: θ ≈ arctan(s / d)
      
    Uses only eye_z (distance), ignores eye_x and eye_y.
    Assumes screen in XY plane, eye looking orthogonally.
    
    Fast but less accurate for:
      - Off-center gaze (eye not centered in front of screen)
      - Large angles
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

        if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
            d_mm = PhysicalConstants.DEFAULT_EYE_SCREEN_DISTANCE_MM
        else:
            d_mm = float(eye_z_mm)

        theta_rad = math.atan2(s_mm, d_mm)
        return math.degrees(theta_rad)

    def get_description(self) -> str:
        return "Olsen 2D Approximation: θ = atan(screen_distance / eye_z)"


class Ray3DAngle(VelocityCalculationStrategy):
    """Physically correct 3D ray method.
    
    Method:
      1. Screen at z=0, gaze points: G₀ = (gx₀, gy₀, 0), G₁ = (gx₁, gy₁, 0)
      2. Eye position: E = (ex, ey, ez)
      3. Ray vectors: d₀ = G₀ - E, d₁ = G₁ - E
      4. Angle between rays: θ = arccos(d₀ · d₁ / (|d₀| × |d₁|))
      
    Advantages:
      - Physically correct
      - Works for off-center positions
      - Accounts for full 3D geometry
      
    Typically 1-5% lower velocities than Olsen 2D at typical viewing distances.
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

        # Ray vectors from eye to gaze points
        d0x = gx0 - ex
        d0y = gy0 - ey
        d0z = gz0 - ez

        d1x = gx1 - ex
        d1y = gy1 - ey
        d1z = gz1 - ez

        # Dot product
        dot_product = d0x * d1x + d0y * d1y + d0z * d1z

        # Norms (lengths of vectors)
        norm0 = math.sqrt(d0x**2 + d0y**2 + d0z**2)
        norm1 = math.sqrt(d1x**2 + d1y**2 + d1z**2)

        if norm0 == 0.0 or norm1 == 0.0:
            return 0.0

        cos_theta = dot_product / (norm0 * norm1)
        cos_theta = max(-1.0, min(1.0, cos_theta))

        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)

        return theta_deg

    def calculate_visual_angle_ctx(self, ctx: VelocityContext) -> float:
        return self.calculate_visual_angle(
            ctx.x1_mm,
            ctx.y1_mm,
            ctx.x2_mm,
            ctx.y2_mm,
            ctx.eye_x_mm,
            ctx.eye_y_mm,
            ctx.eye_z_mm,
        )

    def get_description(self) -> str:
        return "3D Ray Angle: θ = acos(ray0 · ray1 / (|ray0| × |ray1|))"


class Ray3DGazeDir(VelocityCalculationStrategy):
    """Velocity directly from normalized gaze direction vectors.

    Expects per-sample normalized gaze direction (x,y,z) for the selected eye.
    Uses only the direction change; ignores screen geometry and eye position.
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
        dir1: Optional[np.ndarray] = None,
        dir2: Optional[np.ndarray] = None,
    ) -> float:
        ctx = VelocityContext(
            x1_mm=x1_mm,
            y1_mm=y1_mm,
            x2_mm=x2_mm,
            y2_mm=y2_mm,
            eye_x_mm=eye_x_mm,
            eye_y_mm=eye_y_mm,
            eye_z_mm=eye_z_mm,
            dir1=dir1,
            dir2=dir2,
        )
        return self.calculate_visual_angle_ctx(ctx)

    def calculate_visual_angle_ctx(self, ctx: VelocityContext) -> float:
        dir1 = ctx.dir1
        dir2 = ctx.dir2
        if dir1 is None or dir2 is None:
            return 0.0

        v1 = np.array(dir1, dtype=float)
        v2 = np.array(dir2, dtype=float)

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0.0 or n2 == 0.0 or not np.isfinite(n1) or not np.isfinite(n2):
            return 0.0

        v1 /= n1
        v2 /= n2

        dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
        theta_rad = math.acos(dot)
        return math.degrees(theta_rad)

    def get_description(self) -> str:
        return "Ray3D using gaze direction vectors (acos(dir0·dir1))"
