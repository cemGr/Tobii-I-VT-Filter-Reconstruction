# ivt_filter/preprocessing/eye_selection.py
"""Eye selection: combines left/right eye data based on validity and configuration."""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from ..config import OlsenVelocityConfig


from ..domain.validity import parse_tobii_validity

# Backward-compatible private alias; parsing rules live in domain.validity.
_parse_validity = parse_tobii_validity


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
    Combine the left and right eye into a single gaze point + eye position.

    Returns:
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

    # px gaze (optional, for debug/plots)
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

    # Left eye only
    if mode == "left":
        if left_valid:
            return use_left()
        return None, None, None, None, None, None, None, False, bool(left_valid), bool(right_valid)

    # Right eye only
    if mode == "right":
        if right_valid:
            return use_right()
        return None, None, None, None, None, None, None, False, bool(left_valid), bool(right_valid)

    # Strict average: no single-eye fallback
    if mode == "strict_average" and not (left_valid and right_valid):
        return None, None, None, None, None, None, None, False, bool(left_valid), bool(right_valid)

    # Average of both eyes (default)
    if left_valid and right_valid:
        gaze_x_mm = (float(lx_mm) + float(rx_mm)) / 2.0
        gaze_y_mm = (float(ly_mm) + float(ry_mm)) / 2.0

        # Optionally combine the px values
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

        # Combine the eye position (X, Y, Z)
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

    # Fallback: only one valid eye
    if left_valid:
        return use_left()
    if right_valid:
        return use_right()

    # No valid gaze
    return None, None, None, None, None, None, None, False, bool(left_valid), bool(right_valid)


def prepare_combined_columns(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """
    Adds combined gaze/eye columns:

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


def apply_tobii_eye_offset_interpolation(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
) -> pd.DataFrame:
    """Tobii-exact eye offset interpolation for missing eye data.

    Reconstructed from ``RemoteTrackerGazeDataToRecordedTwoEyedGazeDataConverter``
    in the decompiled Tobii C# implementation.

    When an eye is missing (invalid), the last known spatial offset
    between the left and right eye (gaze-point offset and eye-origin offset)
    is used to estimate the missing eye. This is more precise than a simple
    fallback to a single eye, since parallax effects are compensated.

    How it works:
    - ``BothEyesFound``:  offsets are updated, real data is used.
    - ``OnlyRightEyeFound``: left_gaze = right_gaze - last_gaze_offset
                             left_eye  = right_eye  - last_eye_offset
    - ``OnlyLeftEyeFound``:  right_gaze = left_gaze + last_gaze_offset
                             right_eye  = left_eye  + last_eye_offset
    - ``NoEyesFound``: no estimation possible (data stays NaN).

    Modifies the columns ``gaze_left_x_mm``, ``gaze_left_y_mm``,
    ``gaze_right_x_mm``, ``gaze_right_y_mm`` as well as the eye-origin columns
    ``eye_left_x_mm``, ``eye_left_y_mm``, ``eye_left_z_mm`` etc. in-place
    (on a copy).

    Args:
        df: DataFrame with raw Tobii data (must contain gaze_left_*, gaze_right_*,
            validity_left, validity_right).
        cfg: velocity configuration (for ``max_validity``).

    Returns:
        Copy of the DataFrame with interpolated gaze/eye columns.
    """
    df = df.copy()

    # Required gaze columns
    gaze_cols_left = ["gaze_left_x_mm", "gaze_left_y_mm"]
    gaze_cols_right = ["gaze_right_x_mm", "gaze_right_y_mm"]
    eye_cols_left = ["eye_left_x_mm", "eye_left_y_mm", "eye_left_z_mm"]
    eye_cols_right = ["eye_right_x_mm", "eye_right_y_mm", "eye_right_z_mm"]

    has_gaze_left = all(c in df.columns for c in gaze_cols_left)
    has_gaze_right = all(c in df.columns for c in gaze_cols_right)
    has_eye_left = all(c in df.columns for c in eye_cols_left)
    has_eye_right = all(c in df.columns for c in eye_cols_right)

    if not (has_gaze_left and has_gaze_right):
        # Without both gaze columns no interpolation is possible
        return df

    # Stored offsets (right - left); initialized with None
    last_gaze_offset: Optional[np.ndarray] = None   # shape (2,): [dx, dy]
    last_eye_offset: Optional[np.ndarray] = None    # shape (3,): [dx, dy, dz]

    # Numpy arrays for fast access
    vl_arr = df["validity_left"].to_numpy() if "validity_left" in df.columns else np.zeros(len(df))
    vr_arr = df["validity_right"].to_numpy() if "validity_right" in df.columns else np.zeros(len(df))

    # Validity arrays as object dtype for writing back (allows mixed types)
    vl_out = vl_arr.copy().astype(object)
    vr_out = vr_arr.copy().astype(object)

    # "Valid" marker: "Valid" (string) if the column contains string values, otherwise 0
    def _valid_marker(arr: np.ndarray) -> object:
        """Return the matching 'valid' value (string or int)."""
        for v in arr:
            if isinstance(v, str):
                return "Valid"
        return 0

    valid_marker_l = _valid_marker(vl_arr)
    valid_marker_r = _valid_marker(vr_arr)

    # Explicit copies (writable=True) – to_numpy() can return read-only arrays
    lx = df["gaze_left_x_mm"].to_numpy(dtype=float).copy()
    ly = df["gaze_left_y_mm"].to_numpy(dtype=float).copy()
    rx = df["gaze_right_x_mm"].to_numpy(dtype=float).copy()
    ry = df["gaze_right_y_mm"].to_numpy(dtype=float).copy()

    lex = df["eye_left_x_mm"].to_numpy(dtype=float).copy() if has_eye_left else None
    ley = df["eye_left_y_mm"].to_numpy(dtype=float).copy() if has_eye_left else None
    lez = df["eye_left_z_mm"].to_numpy(dtype=float).copy() if has_eye_left else None
    rex_ = df["eye_right_x_mm"].to_numpy(dtype=float).copy() if has_eye_right else None
    rey_ = df["eye_right_y_mm"].to_numpy(dtype=float).copy() if has_eye_right else None
    rez_ = df["eye_right_z_mm"].to_numpy(dtype=float).copy() if has_eye_right else None

    max_v = cfg.max_validity

    for i in range(len(df)):
        left_valid = (_parse_validity(vl_arr[i]) <= max_v
                      and np.isfinite(lx[i]) and np.isfinite(ly[i]))
        right_valid = (_parse_validity(vr_arr[i]) <= max_v
                       and np.isfinite(rx[i]) and np.isfinite(ry[i]))

        if left_valid and right_valid:
            # Both eyes valid -> update offsets
            last_gaze_offset = np.array([rx[i] - lx[i], ry[i] - ly[i]])
            if (has_eye_left and has_eye_right
                    and np.isfinite(lex[i]) and np.isfinite(rex_[i])):  # type: ignore[index]
                last_eye_offset = np.array([
                    rex_[i] - lex[i],   # type: ignore[index]
                    rey_[i] - ley[i],   # type: ignore[index]
                    rez_[i] - lez[i],   # type: ignore[index]
                ])

        elif right_valid and not left_valid and last_gaze_offset is not None:
            # Only right eye -> estimate left eye via the stored offset
            # left_gaze = right_gaze - offset  (offset = right - left)
            lx[i] = rx[i] - last_gaze_offset[0]
            ly[i] = ry[i] - last_gaze_offset[1]
            if (has_eye_left and has_eye_right and last_eye_offset is not None
                    and np.isfinite(rex_[i])):  # type: ignore[index]
                lex[i] = rex_[i] - last_eye_offset[0]   # type: ignore[index]
                ley[i] = rey_[i] - last_eye_offset[1]   # type: ignore[index]
                lez[i] = rez_[i] - last_eye_offset[2]   # type: ignore[index]
            # Set validity to the valid marker so that prepare_combined_columns
            # treats the interpolated eye as equivalent
            vl_out[i] = valid_marker_l

        elif left_valid and not right_valid and last_gaze_offset is not None:
            # Only left eye -> estimate right eye via the stored offset
            # right_gaze = left_gaze + offset
            rx[i] = lx[i] + last_gaze_offset[0]
            ry[i] = ly[i] + last_gaze_offset[1]
            if (has_eye_right and has_eye_left and last_eye_offset is not None
                    and np.isfinite(lex[i])):  # type: ignore[index]
                rex_[i] = lex[i] + last_eye_offset[0]   # type: ignore[index]
                rey_[i] = ley[i] + last_eye_offset[1]   # type: ignore[index]
                rez_[i] = lez[i] + last_eye_offset[2]   # type: ignore[index]
            # Set validity to the valid marker
            vr_out[i] = valid_marker_r
        # else: both invalid -> no estimation, data stays NaN

    # Write back
    df["gaze_left_x_mm"] = lx
    df["gaze_left_y_mm"] = ly
    df["gaze_right_x_mm"] = rx
    df["gaze_right_y_mm"] = ry
    if has_eye_left:
        df["eye_left_x_mm"] = lex
        df["eye_left_y_mm"] = ley
        df["eye_left_z_mm"] = lez
    if has_eye_right:
        df["eye_right_x_mm"] = rex_
        df["eye_right_y_mm"] = rey_
        df["eye_right_z_mm"] = rez_
    # Write back the validity flags (interpolated eyes are now "Valid")
    if "validity_left" in df.columns:
        df["validity_left"] = vl_out
    if "validity_right" in df.columns:
        df["validity_right"] = vr_out

    return df
