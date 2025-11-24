"""Geometry helpers for gaze calculations."""
from __future__ import annotations

import math
from typing import Optional

from .config import OlsenVelocityConfig


class VisualAngleCalculator:
    """Convert pixel displacement to visual angles."""

    def __init__(self, config: OlsenVelocityConfig) -> None:
        self.config = config

    def visual_angle_deg(
        self,
        x1_px: float,
        y1_px: float,
        x2_px: float,
        y2_px: float,
        eye_z_mm: Optional[float],
    ) -> float:
        dx = float(x2_px) - float(x1_px)
        dy = float(y2_px) - float(y1_px)
        s_px = math.hypot(dx, dy)

        d_mm = (
            float(eye_z_mm)
            if eye_z_mm is not None and math.isfinite(eye_z_mm) and eye_z_mm > 0
            else self.config.default_eye_distance_mm
        )

        theta_rad = math.atan2(s_px, d_mm)
        return math.degrees(theta_rad)
