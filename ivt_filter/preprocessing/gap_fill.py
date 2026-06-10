# ivt_filter/preprocessing/gap_fill.py
"""Gap fill-in interpolation: fills small temporal gaps in eye-tracking data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import OlsenVelocityConfig


from ..domain.validity import parse_tobii_validity


def gap_fill_gaze(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """
    Temporal gap filling per eye:
      - Small gaps (up to gap_fill_max_gap_ms) are interpolated linearly.
      - Operates separately for the left/right eye.
      - Also updates validity_left / validity_right to "valid"
        for the imputed samples.

    Important:
      - Must be called BEFORE prepare_combined_columns().
      - Uses time_ms as the time axis.
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
        dir_x_col = f"gaze_dir_{eye}_x"
        dir_y_col = f"gaze_dir_{eye}_y"
        dir_z_col = f"gaze_dir_{eye}_z"

        if x_col not in df.columns or y_col not in df.columns:
            continue

        # numeric arrays for x/y
        # .copy() ensures the array is writable –
        # pandas 2.0+ (copy-on-write) can return read-only arrays from to_numpy().
        x_vals = df[x_col].to_numpy().copy()
        y_vals = df[y_col].to_numpy().copy()

        # optional arrays
        z_vals = df[z_col].to_numpy().copy() if z_col in df.columns else None
        px_x_vals = df[px_x_col].to_numpy().copy() if px_x_col in df.columns else None
        px_y_vals = df[px_y_col].to_numpy().copy() if px_y_col in df.columns else None

        # Gaze direction vectors (normalized unit vectors, for ray3d_gaze_dir / tobii_gaze_dir)
        has_dir = all(c in df.columns for c in (dir_x_col, dir_y_col, dir_z_col))
        if has_dir:
            dir_x_vals = df[dir_x_col].to_numpy().copy()
            dir_y_vals = df[dir_y_col].to_numpy().copy()
            dir_z_vals = df[dir_z_col].to_numpy().copy()
        else:
            dir_x_vals = dir_y_vals = dir_z_vals = None

        # Parse the validity codes (if present)
        if val_col in df.columns:
            val_series = df[val_col].copy()
            v_codes = val_series.map(parse_tobii_validity).to_numpy()
        else:
            val_series = None
            v_codes = None

        # "valid" samples for this eye:
        valid_mask = ~pd.isna(x_vals) & ~pd.isna(y_vals)
        if v_codes is not None:
            valid_mask &= v_codes <= cfg.max_validity

        valid_mask = valid_mask.astype(bool)

        idx = 0
        while idx < n:
            if valid_mask[idx]:
                idx += 1
                continue

            # Start of a gap
            gap_start = idx
            while idx < n and not valid_mask[idx]:
                idx += 1
            gap_end = idx - 1

            prev_idx = gap_start - 1
            next_idx = gap_end + 1

            # Gap at the edge -> do not fill
            if prev_idx < 0 or next_idx >= n:
                continue

            if not valid_mask[prev_idx] or not valid_mask[next_idx]:
                continue

            # Gap size: span of the invalid run (first to last invalid sample).
            # This handles leading/trailing samples correctly: if an eye drops out a few
            # samples before the shared gap, that extends the boundary-to-boundary distance –
            # but not the actual gap length. Strictly < max_gap_ms (Tobii spec 3.1.1.1).
            # gap_ms == 0 (1-sample gap) is allowed.
            gap_ms = float(times[gap_end] - times[gap_start])
            if gap_ms < 0 or gap_ms >= max_gap_ms:
                continue

            # Endpoints
            x_prev, y_prev = x_vals[prev_idx], y_vals[prev_idx]
            x_next, y_next = x_vals[next_idx], y_vals[next_idx]

            if any(pd.isna(v) for v in (x_prev, y_prev, x_next, y_next)):
                continue

            # optional: Z and pixel endpoints
            if z_vals is not None:
                z_prev, z_next = z_vals[prev_idx], z_vals[next_idx]
            else:
                z_prev = z_next = None

            if px_x_vals is not None and px_y_vals is not None:
                px_x_prev, px_x_next = px_x_vals[prev_idx], px_x_vals[next_idx]
                px_y_prev, px_y_next = px_y_vals[prev_idx], px_y_vals[next_idx]
            else:
                px_x_prev = px_x_next = px_y_prev = px_y_next = None

            # Gaze direction endpoints
            dpx: tuple[float, float, float] | None
            dnx: tuple[float, float, float] | None
            if dir_x_vals is not None:
                dpx = (float(dir_x_vals[prev_idx]), float(dir_y_vals[prev_idx]), float(dir_z_vals[prev_idx]))
                dnx = (float(dir_x_vals[next_idx]), float(dir_y_vals[next_idx]), float(dir_z_vals[next_idx]))
                dir_endpoints_valid = all(not pd.isna(v) for v in dpx + dnx)
            else:
                dpx = dnx = None
                dir_endpoints_valid = False

            dt_total = float(times[next_idx] - times[prev_idx])
            if dt_total <= 0:
                continue

            # Interpolation for all samples between prev_idx and next_idx
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

                # Interpolate and renormalize the gaze direction vectors
                if dir_endpoints_valid and dpx is not None and dnx is not None:
                    ix = (1.0 - alpha) * dpx[0] + alpha * dnx[0]
                    iy = (1.0 - alpha) * dpx[1] + alpha * dnx[1]
                    iz = (1.0 - alpha) * dpx[2] + alpha * dnx[2]
                    norm = np.sqrt(ix * ix + iy * iy + iz * iz)
                    if norm > 0:
                        dir_x_vals[j] = ix / norm
                        dir_y_vals[j] = iy / norm
                        dir_z_vals[j] = iz / norm

                # Set the validity code to "valid" for imputed samples
                if val_series is not None:
                    val_series.iloc[j] = val_series.iloc[prev_idx]

        # Write back into the DataFrame
        df[x_col] = x_vals
        df[y_col] = y_vals
        if z_vals is not None:
            df[z_col] = z_vals
        if px_x_vals is not None and px_y_vals is not None:
            df[px_x_col] = px_x_vals
            df[px_y_col] = px_y_vals
        if has_dir:
            df[dir_x_col] = dir_x_vals
            df[dir_y_col] = dir_y_vals
            df[dir_z_col] = dir_z_vals
        if val_series is not None:
            df[val_col] = val_series

    return df
