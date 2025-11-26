"""Gaze/eye data combination utilities."""
from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import OlsenVelocityConfig


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned == "":
            return None
        cleaned = cleaned.replace(",", ".")
        try:
            return float(cleaned)
        except ValueError:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_validity(value: object) -> int:
    """Parse Tobii-style validity into a numeric code.

    Rules:
    - strings:
      - "Valid"   -> 0
      - "Invalid" -> 999
    - numeric strings like "0", "1", "2" -> int(value)
    - anything else -> 999 (treat as not trustworthy)
    """

    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() == "valid":
            return 0
        if normalized.lower() == "invalid":
            return 999
        try:
            return int(normalized)
        except ValueError:
            return 999

    try:
        return int(value)
    except (TypeError, ValueError):
        return 999


def combine_gaze_and_eye(row: pd.Series, cfg: OlsenVelocityConfig) -> Tuple[float | None, float | None, float | None, bool]:
    """Combine left/right gaze and eye position into a single gaze point and Z distance.

    Returns: (gaze_x, gaze_y, eye_z_mm, is_valid)
    - gaze_x / gaze_y are in mm if cfg.use_gaze_mm is True,
      otherwise in px.
    - eye_z_mm is always in millimetres.
    - is_valid is True if the combined gaze is trustworthy for velocity.
    """

    if cfg.use_gaze_mm:
        lx = _to_float(row.get("gaze_left_x_mm"))
        ly = _to_float(row.get("gaze_left_y_mm"))
        rx = _to_float(row.get("gaze_right_x_mm"))
        ry = _to_float(row.get("gaze_right_y_mm"))
    else:
        lx = _to_float(row.get("gaze_left_x_px"))
        ly = _to_float(row.get("gaze_left_y_px"))
        rx = _to_float(row.get("gaze_right_x_px"))
        ry = _to_float(row.get("gaze_right_y_px"))

    lz = _to_float(row.get("eye_left_z_mm"))
    rz = _to_float(row.get("eye_right_z_mm"))

    v_left = parse_validity(row.get("validity_left"))
    v_right = parse_validity(row.get("validity_right"))

    left_valid = v_left <= cfg.max_validity and lx is not None and ly is not None and not pd.isna(lx) and not pd.isna(ly)
    right_valid = v_right <= cfg.max_validity and rx is not None and ry is not None and not pd.isna(rx) and not pd.isna(ry)

    mode = cfg.eye_mode

    def select_eye(x, y, z, valid) -> Tuple[float | None, float | None, float | None, bool]:
        return (x if pd.notna(x) else None, y if pd.notna(y) else None, z if pd.notna(z) else None, valid and pd.notna(x) and pd.notna(y))

    if mode == "left":
        return select_eye(lx, ly, lz, left_valid)
    if mode == "right":
        return select_eye(rx, ry, rz, right_valid)

    if left_valid and right_valid:
        gaze_x = (lx + rx) / 2.0
        gaze_y = (ly + ry) / 2.0
        if pd.notna(lz) and pd.notna(rz):
            eye_z = (lz + rz) / 2.0
        elif pd.notna(lz):
            eye_z = lz
        elif pd.notna(rz):
            eye_z = rz
        else:
            eye_z = None
        return gaze_x, gaze_y, eye_z, True

    if left_valid:
        return select_eye(lx, ly, lz, True)
    if right_valid:
        return select_eye(rx, ry, rz, True)
    return None, None, None, False


class EyeCombiner:
    """Combine left/right eye data according to the chosen strategy."""

    def __init__(self, config: OlsenVelocityConfig) -> None:
        self.config = config

    def add_combined_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        gaze_x = []
        gaze_y = []
        eye_z = []
        valid = []
        for _, row in df.iterrows():
            gx, gy, gz, is_valid = combine_gaze_and_eye(row, self.config)
            gaze_x.append(gx)
            gaze_y.append(gy)
            eye_z.append(gz)
            valid.append(is_valid)

        if self.config.use_gaze_mm:
            df["combined_x_mm"] = gaze_x
            df["combined_y_mm"] = gaze_y
        else:
            df["combined_x_px"] = gaze_x
            df["combined_y_px"] = gaze_y
        df["eye_z_mm"] = eye_z
        df["combined_valid"] = valid
        return df
