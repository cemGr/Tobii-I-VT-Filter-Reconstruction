"""Velocity computation following Olsen/I-VT."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .combiner import EyeCombiner
from .config import OlsenVelocityConfig
from .geometry import VisualAngleCalculator


class VelocityCalculator:
    """Compute angular gaze velocity using a sliding window."""

    def __init__(self, config: Optional[OlsenVelocityConfig] = None) -> None:
        self.config = config or OlsenVelocityConfig()
        self.combiner = EyeCombiner(self.config)
        self.angle_calculator = VisualAngleCalculator(self.config)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = df.sort_values("time_ms").reset_index(drop=True)
        df = self.combiner.add_combined_columns(df)
        df["velocity_deg_per_sec"] = float("nan")

        half_window = cfg.window_length_ms / 2.0
        times = df["time_ms"].to_numpy()
        coord_x_col = "combined_x_mm" if cfg.use_gaze_mm else "combined_x_px"
        coord_y_col = "combined_y_mm" if cfg.use_gaze_mm else "combined_y_px"
        cx = df[coord_x_col].to_numpy()
        cy = df[coord_y_col].to_numpy()
        cz = df["eye_z_mm"].to_numpy()
        valid = df["combined_valid"].to_numpy()
        n = len(df)

        for i in range(n):
            if not valid[i]:
                continue

            t_center = float(times[i])
            eye_z = cz[i] if i < len(cz) else None

            first_idx = self._find_edge_index(times, valid, i, -1, half_window)
            last_idx = self._find_edge_index(times, valid, i, 1, half_window)

            if first_idx is None or last_idx is None or first_idx == last_idx:
                continue

            t_first = float(times[first_idx])
            t_last = float(times[last_idx])
            dt_ms = t_last - t_first
            if dt_ms < cfg.min_dt_ms:
                continue

            x1, y1 = cx[first_idx], cy[first_idx]
            x2, y2 = cx[last_idx], cy[last_idx]
            if any(pd.isna(v) for v in (x1, y1, x2, y2)):
                continue

            angle_deg = self.angle_calculator.visual_angle_deg(
                x1,
                y1,
                x2,
                y2,
                eye_z,
                is_mm=cfg.use_gaze_mm,
            )
            dt_s = dt_ms / 1000.0
            df.at[i, "velocity_deg_per_sec"] = angle_deg / dt_s if dt_s > 0 else float("nan")

        return df

    @staticmethod
    def _find_edge_index(times, valid, center_idx: int, step: int, half_window: float) -> Optional[int]:
        idx = center_idx
        last_valid: Optional[int] = None
        while 0 <= idx < len(times):
            if not valid[idx]:
                break
            delta = (times[center_idx] - times[idx]) if step < 0 else (times[idx] - times[center_idx])
            if delta > half_window:
                break
            last_valid = idx
            idx += step
        return last_valid

    def compute_from_file(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(input_path, sep="\t", decimal=",")
        result = self.compute(df)
        if output_path:
            result.to_csv(output_path, sep="\t", index=False, decimal=",")
        return result


def compute_olsen_velocity_from_slim_tsv(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Optional[OlsenVelocityConfig] = None,
) -> pd.DataFrame:
    calculator = VelocityCalculator(cfg)
    return calculator.compute_from_file(input_path, output_path)
