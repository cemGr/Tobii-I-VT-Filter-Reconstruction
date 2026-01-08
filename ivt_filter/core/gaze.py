# ivt_filter/core/gaze.py
from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from ..config import OlsenVelocityConfig
from ..smoothing_strategy import (
    SmoothingStrategy,
    NoSmoothing,
    MedianSmoothing,
    MovingAverageSmoothing,
    MedianSmoothingStrict,
    MovingAverageSmoothingStrict,
    MedianSmoothingAdaptive,
    MovingAverageSmoothingAdaptive,
)


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


def gap_fill_gaze(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """
    Zeitliches Gap-Filling pro Auge:
      - Kleine Luecken (bis gap_fill_max_gap_ms) werden linear interpoliert.
      - Funktioniert getrennt fuer linkes/rechtes Auge.
      - Aktualisiert auch validity_left / validity_right auf "gueltig"
        fuer die imputierten Samples.

    Wichtig:
      - Muss VOR prepare_combined_columns() aufgerufen werden.
      - Nutzt time_ms als Zeitachse.
    """
    if not cfg.gap_fill_enabled or cfg.gap_fill_max_gap_ms <= 0:
        return df

    if "time_ms" not in df.columns:
        raise ValueError("DataFrame must contain 'time_ms' for gap filling.")

    df = df.copy()
    times = df["time_ms"].to_numpy()
    n = len(df)
    max_gap_ms = float(cfg.gap_fill_max_gap_ms)

    for eye in ("left", "right"):
        x_col = f"gaze_{eye}_x_mm"
        y_col = f"gaze_{eye}_y_mm"
        val_col = f"validity_{eye}"
        z_col = f"eye_{eye}_z_mm"
        px_x_col = f"gaze_{eye}_x_px"
        px_y_col = f"gaze_{eye}_y_px"

        if x_col not in df.columns or y_col not in df.columns:
            continue

        # numerische Arrays fuer x/y
        x_vals = df[x_col].to_numpy()
        y_vals = df[y_col].to_numpy()

        # optionale Arrays
        z_vals = df[z_col].to_numpy() if z_col in df.columns else None
        px_x_vals = df[px_x_col].to_numpy() if px_x_col in df.columns else None
        px_y_vals = df[px_y_col].to_numpy() if px_y_col in df.columns else None

        # Validitaetscodes parsen (falls vorhanden)
        if val_col in df.columns:
            val_series = df[val_col].copy()
            v_codes = val_series.map(_parse_validity).to_numpy()
        else:
            val_series = None
            v_codes = None

        # "valide" Samples fuer dieses Auge:
        valid_mask = ~pd.isna(x_vals) & ~pd.isna(y_vals)
        if v_codes is not None:
            valid_mask &= v_codes <= cfg.max_validity

        valid_mask = valid_mask.astype(bool)

        idx = 0
        while idx < n:
            if valid_mask[idx]:
                idx += 1
                continue

            # Beginn eines Gaps
            gap_start = idx
            while idx < n and not valid_mask[idx]:
                idx += 1
            gap_end = idx - 1

            prev_idx = gap_start - 1
            next_idx = gap_end + 1

            # Gap am Rand -> nicht fuellen
            if prev_idx < 0 or next_idx >= n:
                continue

            if not valid_mask[prev_idx] or not valid_mask[next_idx]:
                continue

            # Gap-Größe: Zeit vom letzten validen Sample bis zum letzten invaliden Sample
            gap_ms = float(times[gap_end] - times[prev_idx])
            if gap_ms <= 0 or gap_ms > max_gap_ms:
                continue

            # Endpunkte
            x_prev, y_prev = x_vals[prev_idx], y_vals[prev_idx]
            x_next, y_next = x_vals[next_idx], y_vals[next_idx]

            if any(pd.isna(v) for v in (x_prev, y_prev, x_next, y_next)):
                continue

            # optional: Z- und Pixel-Endpunkte
            if z_vals is not None:
                z_prev, z_next = z_vals[prev_idx], z_vals[next_idx]
            else:
                z_prev = z_next = None

            if px_x_vals is not None and px_y_vals is not None:
                px_x_prev, px_x_next = px_x_vals[prev_idx], px_x_vals[next_idx]
                px_y_prev, px_y_next = px_y_vals[prev_idx], px_y_vals[next_idx]
            else:
                px_x_prev = px_x_next = px_y_prev = px_y_next = None

            dt_total = float(times[next_idx] - times[prev_idx])
            if dt_total <= 0:
                continue

            # Interpolation fuer alle Samples zwischen prev_idx und next_idx
            for j in range(prev_idx + 1, next_idx):
                t_j = float(times[j])
                alpha = (t_j - float(times[prev_idx])) / dt_total

                x_vals[j] = (1.0 - alpha) * float(x_prev) + alpha * float(x_next)
                y_vals[j] = (1.0 - alpha) * float(y_prev) + alpha * float(y_next)
                valid_mask[j] = True

                if z_vals is not None and not pd.isna(z_prev) and not pd.isna(z_next):
                    z_vals[j] = (1.0 - alpha) * float(z_prev) + alpha * float(z_next)

                if (
                    px_x_vals is not None
                    and px_y_vals is not None
                    and not pd.isna(px_x_prev)
                    and not pd.isna(px_x_next)
                    and not pd.isna(px_y_prev)
                    and not pd.isna(px_y_next)
                ):
                    px_x_vals[j] = (1.0 - alpha) * float(px_x_prev) + alpha * float(px_x_next)
                    px_y_vals[j] = (1.0 - alpha) * float(px_y_prev) + alpha * float(px_y_next)

                # Validitaetscode fuer imputierte Samples auf "gueltig" setzen
                if val_series is not None:
                    val_series.iloc[j] = 0  # numerisch "Valid"

        # Zurueck ins DataFrame schreiben
        df[x_col] = x_vals
        df[y_col] = y_vals
        if z_vals is not None:
            df[z_col] = z_vals
        if px_x_vals is not None and px_y_vals is not None:
            df[px_x_col] = px_x_vals
            df[px_y_col] = px_y_vals
        if val_series is not None:
            df[val_col] = val_series

    return df


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


def _get_smoothing_strategy(
    mode: str, 
    window_samples: int,
    min_samples: int = 1,
    expansion_radius: int = 0
) -> SmoothingStrategy:
    """Factory für Smoothing-Strategien."""
    if mode == "none":
        return NoSmoothing(window_samples)
    elif mode == "median":
        return MedianSmoothing(window_samples)
    elif mode == "moving_average":
        return MovingAverageSmoothing(window_samples)
    elif mode == "median_strict":
        return MedianSmoothingStrict(window_samples)
    elif mode == "moving_average_strict":
        return MovingAverageSmoothingStrict(window_samples)
    elif mode == "median_adaptive":
        return MedianSmoothingAdaptive(window_samples, min_samples, expansion_radius)
    elif mode == "moving_average_adaptive":
        return MovingAverageSmoothingAdaptive(window_samples, min_samples, expansion_radius)
    else:
        raise ValueError(f"Unknown smoothing mode: {mode}")


def smooth_combined_gaze(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """
    Optionales Smoothing auf combined_x_mm / combined_y_mm.

    - Wir smoothing nur dort, wo combined_valid == True.
    - Rolling-Window in Sample-Domaene (nicht Zeit).
    - Erzeugt Spalten:
        - smoothed_x_mm
        - smoothed_y_mm
        
    Nutzt SmoothingStrategy Pattern für verschiedene Methoden.
    """
    df = df.copy()

    valid_mask = df["combined_valid"]
    x_series = df["combined_x_mm"]
    y_series = df["combined_y_mm"]

    # Select strategy basierend auf Config
    strategy = _get_smoothing_strategy(
        cfg.smoothing_mode, 
        cfg.smoothing_window_samples,
        cfg.smoothing_min_samples,
        cfg.smoothing_expansion_radius
    )

    # Anwende Smoothing
    x_smooth = strategy.smooth(x_series, valid_mask)
    y_smooth = strategy.smooth(y_series, valid_mask)

    df["smoothed_x_mm"] = x_smooth
    df["smoothed_y_mm"] = y_smooth
    return df
