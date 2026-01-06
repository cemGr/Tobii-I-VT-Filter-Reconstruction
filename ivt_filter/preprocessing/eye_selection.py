# ivt_filter/preprocessing/eye_selection.py
"""Eye selection: combines left/right eye data based on validity and configuration."""

from __future__ import annotations

from typing import Optional, Tuple, List

import pandas as pd

from ..config import OlsenVelocityConfig


def _parse_validity(value) -> int:
    """
    Tobii-Validity robust parsen.

    - "Valid"   -> 0
    - "Invalid" -> 999
    - numeric strings (0,1,2,...) -> int(value)
    - ints/floats -> int(value)
    - alles andere -> 999
    """
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


def _combine_gaze_and_eye(
    row: pd.Series,
    cfg: OlsenVelocityConfig,
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    bool,
    bool,
    bool,
]:
    """
    Linkes und rechtes Auge zu einem Gaze-Punkt + Eye-Position kombinieren.

    Rueckgabe:
      (combined_x_mm, combined_y_mm,
       combined_x_px, combined_y_px,
       eye_x_mm, eye_y_mm, eye_z_mm,
       combined_valid, left_valid, right_valid)
    """

    v_left = _parse_validity(row.get("validity_left"))
    v_right = _parse_validity(row.get("validity_right"))

    # mm gaze
    lx_mm = row.get("gaze_left_x_mm")
    ly_mm = row.get("gaze_left_y_mm")
    rx_mm = row.get("gaze_right_x_mm")
    ry_mm = row.get("gaze_right_y_mm")

    # px gaze (optional, fuer Debug/Plots)
    lx_px = row.get("gaze_left_x_px")
    ly_px = row.get("gaze_left_y_px")
    rx_px = row.get("gaze_right_x_px")
    ry_px = row.get("gaze_right_y_px")

    # eye position (mm)
    lex = row.get("eye_left_x_mm")
    ley = row.get("eye_left_y_mm")
    lz = row.get("eye_left_z_mm")
    rex = row.get("eye_right_x_mm")
    rey = row.get("eye_right_y_mm")
    rz = row.get("eye_right_z_mm")

    left_valid = (
        pd.notna(lx_mm)
        and pd.notna(ly_mm)
        and v_left <= cfg.max_validity
    )
    right_valid = (
        pd.notna(rx_mm)
        and pd.notna(ry_mm)
        and v_right <= cfg.max_validity
    )

    def use_left():
        return (
            float(lx_mm),
            float(ly_mm),
            float(lx_px) if pd.notna(lx_px) else None,
            float(ly_px) if pd.notna(ly_px) else None,
            float(lex) if pd.notna(lex) else None,
            float(ley) if pd.notna(ley) else None,
            float(lz) if pd.notna(lz) else None,
            True,
            bool(left_valid),
            bool(right_valid),
        )

    def use_right():
        return (
            float(rx_mm),
            float(ry_mm),
            float(rx_px) if pd.notna(rx_px) else None,
            float(ry_px) if pd.notna(ry_px) else None,
            float(rex) if pd.notna(rex) else None,
            float(rey) if pd.notna(rey) else None,
            float(rz) if pd.notna(rz) else None,
            True,
            bool(left_valid),
            bool(right_valid),
        )

    mode = cfg.eye_mode

    # Nur linkes Auge
    if mode == "left":
        if left_valid:
            return use_left()
        return None, None, None, None, None, None, None, False, bool(left_valid), bool(right_valid)

    # Nur rechtes Auge
    if mode == "right":
        if right_valid:
            return use_right()
        return None, None, None, None, None, None, None, False, bool(left_valid), bool(right_valid)

    # Durchschnitt beider Augen (Standard)
    if left_valid and right_valid:
        gaze_x_mm = (float(lx_mm) + float(rx_mm)) / 2.0
        gaze_y_mm = (float(ly_mm) + float(ry_mm)) / 2.0

        # px-Werte optional kombinieren
        if pd.notna(lx_px) and pd.notna(rx_px):
            gaze_x_px = (float(lx_px) + float(rx_px)) / 2.0
        elif pd.notna(lx_px):
            gaze_x_px = float(lx_px)
        elif pd.notna(rx_px):
            gaze_x_px = float(rx_px)
        else:
            gaze_x_px = None

        if pd.notna(ly_px) and pd.notna(ry_px):
            gaze_y_px = (float(ly_px) + float(ry_px)) / 2.0
        elif pd.notna(ly_px):
            gaze_y_px = float(ly_px)
        elif pd.notna(ry_px):
            gaze_y_px = float(ry_px)
        else:
            gaze_y_px = None

        # Eye position kombinieren (X, Y, Z)
        if pd.notna(lex) and pd.notna(rex):
            eye_x = (float(lex) + float(rex)) / 2.0
        elif pd.notna(lex):
            eye_x = float(lex)
        elif pd.notna(rex):
            eye_x = float(rex)
        else:
            eye_x = None

        if pd.notna(ley) and pd.notna(rey):
            eye_y = (float(ley) + float(rey)) / 2.0
        elif pd.notna(ley):
            eye_y = float(ley)
        elif pd.notna(rey):
            eye_y = float(rey)
        else:
            eye_y = None

        if pd.notna(lz) and pd.notna(rz):
            eye_z = (float(lz) + float(rz)) / 2.0
        elif pd.notna(lz):
            eye_z = float(lz)
        elif pd.notna(rz):
            eye_z = float(rz)
        else:
            eye_z = None

        return (
            gaze_x_mm,
            gaze_y_mm,
            gaze_x_px,
            gaze_y_px,
            eye_x,
            eye_y,
            eye_z,
            True,
            bool(left_valid),
            bool(right_valid),
        )

    # Fallback: nur ein gueltiges Auge
    if left_valid:
        return use_left()
    if right_valid:
        return use_right()

    # kein gueltiger Gaze
    return None, None, None, None, None, None, None, False, bool(left_valid), bool(right_valid)


def prepare_combined_columns(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """
    Fuegt kombinierte Gaze/Eye-Spalten hinzu:

      - combined_x_mm, combined_y_mm
      - combined_x_px, combined_y_px
      - eye_x_mm, eye_y_mm, eye_z_mm
      - combined_valid
      - left_eye_valid, right_eye_valid
    """
    combined_x_mm: List[Optional[float]] = []
    combined_y_mm: List[Optional[float]] = []
    combined_x_px: List[Optional[float]] = []
    combined_y_px: List[Optional[float]] = []
    combined_ex: List[Optional[float]] = []  # eye_x_mm
    combined_ey: List[Optional[float]] = []  # eye_y_mm
    combined_z: List[Optional[float]] = []   # eye_z_mm
    combined_valid: List[bool] = []
    left_eye_valid: List[bool] = []
    right_eye_valid: List[bool] = []

    for _, row in df.iterrows():
        (
            gx_mm,
            gy_mm,
            gx_px,
            gy_px,
            ex,
            ey,
            gz,
            valid,
            lv,
            rv,
        ) = _combine_gaze_and_eye(row, cfg)

        combined_x_mm.append(gx_mm)
        combined_y_mm.append(gy_mm)
        combined_x_px.append(gx_px)
        combined_y_px.append(gy_px)
        combined_ex.append(ex)
        combined_ey.append(ey)
        combined_z.append(gz)
        combined_valid.append(bool(valid))
        left_eye_valid.append(bool(lv))
        right_eye_valid.append(bool(rv))

    df = df.copy()
    df["combined_x_mm"] = combined_x_mm
    df["combined_y_mm"] = combined_y_mm
    df["combined_x_px"] = combined_x_px
    df["combined_y_px"] = combined_y_px
    df["eye_x_mm"] = combined_ex
    df["eye_y_mm"] = combined_ey
    df["eye_z_mm"] = combined_z
    df["combined_valid"] = combined_valid
    df["left_eye_valid"] = left_eye_valid
    df["right_eye_valid"] = right_eye_valid
    return df
