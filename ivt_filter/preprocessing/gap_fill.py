# ivt_filter/preprocessing/gap_fill.py
"""Gap fill-in interpolation: fills small temporal gaps in eye-tracking data."""

from __future__ import annotations

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
