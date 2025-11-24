"""Gaze/eye data combination utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .config import OlsenVelocityConfig


@dataclass
class CombinedGaze:
    x_px: Optional[float]
    y_px: Optional[float]
    eye_z_mm: Optional[float]
    is_valid: bool


class EyeCombiner:
    """Combine left/right eye data according to the chosen strategy."""

    def __init__(self, config: OlsenVelocityConfig) -> None:
        self.config = config

    @staticmethod
    def _parse_validity(value) -> int:
        if isinstance(value, str):
            v = value.strip().lower()
            if v == "valid":
                return 0
            if v == "invalid":
                return 999

        try:
            return int(value)
        except (TypeError, ValueError):
            return 999

    def _combine_row(self, row: pd.Series) -> CombinedGaze:
        cfg = self.config
        v_left = self._parse_validity(row.get("validity_left"))
        v_right = self._parse_validity(row.get("validity_right"))

        left_valid = (
            pd.notna(row.get("gaze_left_x_px"))
            and pd.notna(row.get("gaze_left_y_px"))
            and v_left <= cfg.max_validity
        )
        right_valid = (
            pd.notna(row.get("gaze_right_x_px"))
            and pd.notna(row.get("gaze_right_y_px"))
            and v_right <= cfg.max_validity
        )

        lx, ly = row.get("gaze_left_x_px"), row.get("gaze_left_y_px")
        rx, ry = row.get("gaze_right_x_px"), row.get("gaze_right_y_px")
        lz = row.get("eye_left_z_mm")
        rz = row.get("eye_right_z_mm")

        def use_left() -> CombinedGaze:
            return CombinedGaze(
                float(lx), float(ly), float(lz) if pd.notna(lz) else None, True
            )

        def use_right() -> CombinedGaze:
            return CombinedGaze(
                float(rx), float(ry), float(rz) if pd.notna(rz) else None, True
            )

        mode = cfg.eye_mode
        if mode == "left":
            return use_left() if left_valid else CombinedGaze(None, None, None, False)
        if mode == "right":
            return use_right() if right_valid else CombinedGaze(None, None, None, False)

        if left_valid and right_valid:
            gaze_x = (float(lx) + float(rx)) / 2.0
            gaze_y = (float(ly) + float(ry)) / 2.0
            if pd.notna(lz) and pd.notna(rz):
                eye_z = (float(lz) + float(rz)) / 2.0
            elif pd.notna(lz):
                eye_z = float(lz)
            elif pd.notna(rz):
                eye_z = float(rz)
            else:
                eye_z = None
            return CombinedGaze(gaze_x, gaze_y, eye_z, True)

        if left_valid:
            return use_left()
        if right_valid:
            return use_right()
        return CombinedGaze(None, None, None, False)

    def add_combined_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        combined = [self._combine_row(row) for _, row in df.iterrows()]
        df = df.copy()
        df["combined_x_px"] = [c.x_px for c in combined]
        df["combined_y_px"] = [c.y_px for c in combined]
        df["eye_z_mm"] = [c.eye_z_mm for c in combined]
        df["combined_valid"] = [c.is_valid for c in combined]
        return df
