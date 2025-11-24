from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class OlsenVelocityConfig:
    """Configuration for Olsen-style velocity computation.

    Attributes
    ----------
    window_length_ms:
        Total time span of the velocity window centered on each sample
        (e.g. 20 ms as in typical I-VT setups).

    eye_mode:
        How to combine eyes:
          - "left":   use left eye only
          - "right":  use right eye only
          - "average": average valid eyes, fallback to valid single eye

    max_validity:
        Highest validity code still considered "valid" (Tobii/Olsen: 0 or 1).

    min_dt_ms:
        Minimum time difference in ms for a valid velocity computation. This
        avoids extreme spikes when timestamps are identical or nearly so.
    """

    window_length_ms: float = 20.0
    eye_mode: Literal["left", "right", "average"] = "average"
    max_validity: int = 1
    min_dt_ms: float = 0.1


# -----------------------------
# 1) Gaze/Eye combination
# -----------------------------


def _parse_validity(value) -> int:
    """
    Robustly parse Tobii validity:
    - 'Valid'   -> 0
    - 'Invalid' -> 999
    - numeric strings (0,1,2,...) -> int(value)
    - everything else -> 999 (treat as invalid)
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


def _combine_gaze_and_eye(row: pd.Series, cfg: OlsenVelocityConfig):
    """Combine left and right eye into a single gaze point and Z distance.

    Returns: (combined_x_px, combined_y_px, eye_z_mm, is_valid)
    """

    v_left = _parse_validity(row.get("validity_left"))
    v_right = _parse_validity(row.get("validity_right"))

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

    def use_left():
        return (
            float(lx),
            float(ly),
            float(lz) if pd.notna(lz) else None,
            True,
        )

    def use_right():
        return (
            float(rx),
            float(ry),
            float(rz) if pd.notna(rz) else None,
            True,
        )

    mode = cfg.eye_mode

    if mode == "left":
        if left_valid:
            return use_left()
        return None, None, None, False

    if mode == "right":
        if right_valid:
            return use_right()
        return None, None, None, False

    # mode == "average"
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

        return gaze_x, gaze_y, eye_z, True

    if left_valid:
        return use_left()

    if right_valid:
        return use_right()

    # no valid gaze
    return None, None, None, False


def _prepare_combined_columns(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """Add combined gaze/eye columns to the DataFrame."""

    combined_x: list[Optional[float]] = []
    combined_y: list[Optional[float]] = []
    combined_z: list[Optional[float]] = []
    combined_valid: list[bool] = []

    for _, row in df.iterrows():
        gx, gy, gz, valid = _combine_gaze_and_eye(row, cfg)
        combined_x.append(gx)
        combined_y.append(gy)
        combined_z.append(gz)
        combined_valid.append(valid)

    df = df.copy()
    df["combined_x_px"] = combined_x
    df["combined_y_px"] = combined_y
    df["eye_z_mm"] = combined_z
    df["combined_valid"] = combined_valid
    return df


# -----------------------------
# 2) Visual angle
# -----------------------------


def _visual_angle_deg(
    x1_px: float,
    y1_px: float,
    x2_px: float,
    y2_px: float,
    eye_z_mm: Optional[float],
) -> float:
    """Approximate the visual angle between two gaze points (deg)."""

    dx = float(x2_px) - float(x1_px)
    dy = float(y2_px) - float(y1_px)
    s_px = math.hypot(dx, dy)

    if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
        d_mm = 600.0  # heuristic: 60 cm
    else:
        d_mm = float(eye_z_mm)

    theta_rad = math.atan2(s_px, d_mm)
    theta_deg = math.degrees(theta_rad)
    return theta_deg


# -----------------------------
# 3) Olsen-style velocity
# -----------------------------


def compute_olsen_velocity_from_slim_tsv(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Optional[OlsenVelocityConfig] = None,
) -> pd.DataFrame:
    """Compute Olsen-style angular gaze velocity from an extracted TSV."""

    if cfg is None:
        cfg = OlsenVelocityConfig()

    df = pd.read_csv(input_path, sep="\t", decimal=",")

    # Ensure sorted by time
    df = df.sort_values("time_ms").reset_index(drop=True)

    # Prepare combined gaze/eye columns
    df = _prepare_combined_columns(df, cfg)

    # Initialize velocity column
    df["velocity_deg_per_sec"] = float("nan")

    half_window = cfg.window_length_ms / 2.0
    n = len(df)

    times = df["time_ms"].to_numpy()
    cx = df["combined_x_px"].to_numpy()
    cy = df["combined_y_px"].to_numpy()
    cz = df["eye_z_mm"].to_numpy()
    valid = df["combined_valid"].to_numpy()

    for i in range(n):
        if not valid[i]:
            continue

        t_center = float(times[i])
        eye_z = cz[i] if i < len(cz) else None

        # Search backwards from centre within half_window for first valid
        first_idx: Optional[int] = None
        j = i
        while j >= 0:
            if not valid[j]:
                break
            if t_center - float(times[j]) > half_window:
                break
            first_idx = j
            j -= 1

        # Search forwards from centre within half_window for last valid
        last_idx: Optional[int] = None
        k = i
        while k < n:
            if not valid[k]:
                break
            if float(times[k]) - t_center > half_window:
                break
            last_idx = k
            k += 1

        if first_idx is None or last_idx is None:
            continue
        if first_idx == last_idx:
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

        angle_deg = _visual_angle_deg(x1, y1, x2, y2, eye_z)
        dt_s = dt_ms / 1000.0
        velocity = angle_deg / dt_s if dt_s > 0 else float("nan")

        df.at[i, "velocity_deg_per_sec"] = velocity

    if output_path is not None:
        df.to_csv(output_path, sep="\t", index=False)

    return df


# -----------------------------
# 4) Plot helpers
# -----------------------------


def _plot_velocity_only(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> None:
    mask = df["velocity_deg_per_sec"].notna()
    times = df.loc[mask, "time_ms"]
    vels = df.loc[mask, "velocity_deg_per_sec"]

    plt.figure()
    plt.plot(times, vels)
    plt.xlabel("Time [ms]")
    plt.ylabel("Gaze velocity [deg/s]")
    plt.title(
        f"Olsen-style gaze velocity (window={cfg.window_length_ms} ms, eye_mode={cfg.eye_mode})"
    )
    plt.tight_layout()
    plt.show()


def _plot_velocity_and_classification(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> None:
    """Zwei einfache Darstellungen:
    - oben: Zeit vs. Geschwindigkeit
    - unten: Zeit vs. Klassifikation (diskrete Stufen)
    """

    # 1) Velocity
    mask = df["velocity_deg_per_sec"].notna()
    times_vel = df.loc[mask, "time_ms"]
    vels = df.loc[mask, "velocity_deg_per_sec"]

    # 2) Klassifikation (Fixation/Saccade/...)
    if "gt_event_type" in df.columns:
        type_col = "gt_event_type"
    elif "Eye movement type" in df.columns:
        type_col = "Eye movement type"
    else:
        raise ValueError("Keine Event-Typ-Spalte gefunden (gt_event_type / Eye movement type).")

    times_evt = df["time_ms"]
    events = df[type_col].fillna("Unknown").astype(str)

    # mapping label -> code
    unique_labels = list(dict.fromkeys(events))  # Reihenfolge beibehalten
    label_to_code = {lab: i for i, lab in enumerate(unique_labels)}
    codes = events.map(label_to_code)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    # oben: Geschwindigkeit
    ax1.plot(times_vel, vels)
    ax1.set_ylabel("Velocity [deg/s]")
    ax1.set_title(
        f"Olsen-style velocity + classification Cem "
        f"(window={cfg.window_length_ms} ms, eye_mode={cfg.eye_mode})"
    )

    # unten: Klassifikation als Stepplot
    ax2.step(times_evt, codes, where="post")
    ax2.set_xlabel("Time [ms]")
    ax2.set_ylabel("Class")
    ax2.set_yticks(list(label_to_code.values()))
    ax2.set_yticklabels(list(label_to_code.keys()))
    ax2.grid(True, axis="y", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.show()


# -----------------------------
# CLI entry point
# -----------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute Olsen-style angular velocity from extracted IVT TSV.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input TSV with columns like time_ms, gaze_left_x_px, gaze_left_y_px, "
            "gaze_right_x_px, gaze_right_y_px, validity_left, validity_right, "
            "eye_left_z_mm, eye_right_z_mm."
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        help=(
            "Optional output TSV path. If provided, the DataFrame with the added "
            "velocity_deg_per_sec column will be written there."
        ),
    )
    parser.add_argument(
        "--window",
        type=float,
        default=20.0,
        help="Total window length in ms (default: 20.0).",
    )
    parser.add_argument(
        "--eye",
        choices=["left", "right", "average"],
        default="average",
        help="Eye selection mode (default: average).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="If set, do not show any matplotlib plot (only compute/save TSV).",
    )
    parser.add_argument(
        "--with-events",
        action="store_true",
        help="If set, show two plots: time vs velocity and time vs classification.",
    )

    args = parser.parse_args()
    config = OlsenVelocityConfig(window_length_ms=args.window, eye_mode=args.eye)

    df_result = compute_olsen_velocity_from_slim_tsv(
        input_path=args.input,
        output_path=args.output,
        cfg=config,
    )

    if not args.no_plot:
        if args.with_events:
            _plot_velocity_and_classification(df_result, config)
        else:
            _plot_velocity_only(df_result, config)
