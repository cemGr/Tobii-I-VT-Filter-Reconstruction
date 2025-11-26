"""Geometry helpers for gaze calculations."""
from __future__ import annotations

import math

from .config import OlsenVelocityConfig


class VisualAngleCalculator:
    """Convert gaze displacement to visual angles."""

    def __init__(self, config: OlsenVelocityConfig) -> None:
        self.config = config

    def visual_angle_deg(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        eye_z_mm: float | None,
        is_mm: bool,
    ) -> float:
        """Compute visual angle between two gaze points and eye distance.

        If ``is_mm`` is True, x/y are interpreted as millimetres on the stimulus plane.
        If ``is_mm`` is False, x/y are interpreted as pixels and converted to mm using
        ``pixel_size_mm`` from the configuration.
        """

        dx = float(x2) - float(x1)
        dy = float(y2) - float(y1)

        if is_mm:
            s_mm = math.hypot(dx, dy)
        else:
            if self.config.pixel_size_mm is None:
                raise ValueError("pixel_size_mm must be provided if use_gaze_mm is False")
            s_px = math.hypot(dx, dy)
            s_mm = s_px * self.config.pixel_size_mm

        if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
            d_mm = 600.0
        else:
            d_mm = float(eye_z_mm)

        theta_rad = math.atan2(s_mm, d_mm)
        return math.degrees(theta_rad)
