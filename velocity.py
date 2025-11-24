from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict

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


@dataclass
class IVTClassifierConfig:
    """Configuration for the I-VT velocity-threshold classifier."""

    velocity_threshold_deg_per_sec: float = 30.0


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


def _combine_gaze_and_eye(row: pd.Series, cfg: OlsenVelocityConfig) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
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
# 4) I-VT classifier (threshold)
# -----------------------------


def apply_ivt_classifier(
    df: pd.DataFrame,
    cfg: Optional[IVTClassifierConfig] = None,
) -> pd.DataFrame:
    """
    Apply a simple I-VT velocity-threshold classifier on a DataFrame
    that already contains 'velocity_deg_per_sec'.

    Adds:
      - ivt_sample_type  (Fixation / Saccade / Unclassified)
      - ivt_event_type   (Fixation / Saccade / Unclassified)
      - ivt_event_index  (int, increments for each new Fixation/Saccade event)
    """
    if cfg is None:
        cfg = IVTClassifierConfig()

    if "velocity_deg_per_sec" not in df.columns:
        raise ValueError("DataFrame must contain 'velocity_deg_per_sec' before classification.")

    df = df.copy()

    def classify_sample(v: float) -> str:
        if v is None or not math.isfinite(v):
            return "Unclassified"
        if v > cfg.velocity_threshold_deg_per_sec:
            return "Saccade"
        return "Fixation"

    df["ivt_sample_type"] = df["velocity_deg_per_sec"].apply(classify_sample)

    # Build event-level type/index from per-sample labels
    event_types: list[str] = []
    event_indices: list[Optional[int]] = []

    current_type: Optional[str] = None
    current_index: int = 0

    for label in df["ivt_sample_type"]:
        if label not in ("Fixation", "Saccade"):
            event_types.append(label)
            event_indices.append(None)
            current_type = None
            continue

        if label != current_type:
            current_index += 1
            current_type = label

        event_types.append(label)
        event_indices.append(current_index)

    df["ivt_event_type"] = event_types
    df["ivt_event_index"] = event_indices

    return df


# -----------------------------
# 5) Evaluation vs. ground truth 
# -----------------------------


def evaluate_ivt_vs_ground_truth(
    df: pd.DataFrame,
    gt_col: Optional[str] = None,
    pred_col: str = "ivt_sample_type",
) -> Dict[str, float]:
    """
    Compare I-VT classifier output against ground truth.

    The evaluation is done on a *sample level* and focuses on
    Fixation/Saccade agreement, as in RQ1.1 of the exposé
    ("percentage agreement").

    Returns a dict with summary statistics and also prints
    a small report to stdout, including Cohen's kappa.
    """

    if gt_col is None:
        if "gt_event_type" in df.columns:
            gt_col = "gt_event_type"
        elif "Eye movement type" in df.columns:
            gt_col = "Eye movement type"
        else:
            raise ValueError("No ground-truth event type column found.")

    if pred_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found. Run apply_ivt_classifier first.")

    # We focus on samples where the GT is Fixation or Saccade.
    mask = df[gt_col].isin(["Fixation", "Saccade"])
    gt = df.loc[mask, gt_col].astype(str)
    pred = df.loc[mask, pred_col].astype(str)

    total = len(gt)
    if total == 0:
        raise ValueError("No samples with ground truth Fixation/Saccade found.")

    # Observed agreement (sample-level)
    agree_mask = gt == pred
    agree = int(agree_mask.sum())
    agreement = agree / total

    # Confusion counts for Fixation/Saccade (für Recall)
    tp_fix = int(((gt == "Fixation") & (pred == "Fixation")).sum())
    fn_fix = int(((gt == "Fixation") & (pred == "Saccade")).sum())
    tp_sac = int(((gt == "Saccade") & (pred == "Saccade")).sum())
    fn_sac = int(((gt == "Saccade") & (pred == "Fixation")).sum())

    n_fix = int((gt == "Fixation").sum())
    n_sac = int((gt == "Saccade").sum())

    recall_fix = tp_fix / n_fix if n_fix > 0 else float("nan")
    recall_sac = tp_sac / n_sac if n_sac > 0 else float("nan")

    # ---- Cohen's kappa (multi-class, generisch) ----
    labels = sorted(set(gt) | set(pred))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)

    # Konfusionsmatrix aufbauen
    conf = [[0] * k for _ in range(k)]
    for g_val, p_val in zip(gt, pred):
        i = label_to_idx[g_val]
        j = label_to_idx[p_val]
        conf[i][j] += 1

    # Beobachtete Übereinstimmung Po
    po = sum(conf[i][i] for i in range(k)) / total

    # Erwartete Übereinstimmung Pe (Randverteilungen)
    row_marg = [sum(conf[i][j] for j in range(k)) for i in range(k)]
    col_marg = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(k)) / (total * total)

    if 1.0 - pe != 0.0:
        kappa = (po - pe) / (1.0 - pe)
    else:
        kappa = float("nan")

    stats = {
        "n_samples_gt_fix_or_sac": total,
        "n_fix_in_gt": n_fix,
        "n_sac_in_gt": n_sac,
        "n_agree": agree,
        "percentage_agreement": agreement * 100.0,
        "fixation_recall": recall_fix * 100.0,
        "saccade_recall": recall_sac * 100.0,
        "cohen_kappa": kappa,
    }

    # Print report
    print("=== I-VT classifier evaluation vs. ground truth ===")
    print(f"Total samples with GT Fixation/Saccade: {total}")
    print(f"  Fixation in GT: {n_fix}")
    print(f"  Saccade in GT:  {n_sac}")
    print()
    print(f"Agreement (sample-level): {agree} / {total} = {agreement*100:.2f}%")
    print()
    print("Confusion (rows = GT, cols = Pred):")
    print("             Pred: Fixation   Pred: Saccade")
    print(f"GT Fixation   {tp_fix:7d}        {fn_fix:7d}")
    print(f"GT Saccade    {fn_sac:7d}        {tp_sac:7d}")
    print()
    print(f"Fixation recall: {recall_fix*100:.2f}%")
    print(f"Saccade recall:  {recall_sac*100:.2f}%")
    print()
    print(f"Cohen's kappa:   {kappa:.3f}")
    print("==============================================")

    return stats



# -----------------------------
# 6) Plot helpers
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
    """Two simple plots:
    - top:  time vs. velocity
    - bottom: time vs. (ground-truth) classification as discrete steps
    """

    # 1) Velocity
    mask = df["velocity_deg_per_sec"].notna()
    times_vel = df.loc[mask, "time_ms"]
    vels = df.loc[mask, "velocity_deg_per_sec"]

    # 2) Classification (Fixation/Saccade/...)
    if "gt_event_type" in df.columns:
        type_col = "gt_event_type"
    elif "Eye movement type" in df.columns:
        type_col = "Eye movement type"
    else:
        raise ValueError("No event-type column found for plotting (gt_event_type / Eye movement type).")

    times_evt = df["time_ms"]
    events = df[type_col].fillna("Unknown").astype(str)

    # mapping label -> code
    unique_labels = list(dict.fromkeys(events))  # preserve order
    label_to_code = {lab: i for i, lab in enumerate(unique_labels)}
    codes = events.map(label_to_code)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    # top: velocity
    ax1.plot(times_vel, vels)
    ax1.set_ylabel("Velocity [deg/s]")
    ax1.set_title(
        f"Olsen-style velocity + GT classification "
        f"(window={cfg.window_length_ms} ms, eye_mode={cfg.eye_mode})"
    )

    # bottom: classification as step plot
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
            "velocity_deg_per_sec and possible classifier columns will be written there."
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
        help="If set, show two plots: time vs velocity and time vs GT classification.",
    )
    parser.add_argument(
        "--classify",
        action="store_true",
        help=(
            "If set, apply an I-VT velocity-threshold classifier and add "
            "ivt_sample_type / ivt_event_type / ivt_event_index columns."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Velocity threshold in deg/s for the I-VT classifier (default: 30).",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If set, compare classifier output with ground truth and print statistics.",
    )

    args = parser.parse_args()
    vel_config = OlsenVelocityConfig(window_length_ms=args.window, eye_mode=args.eye)

    # 1) Compute velocity
    df_result = compute_olsen_velocity_from_slim_tsv(
        input_path=args.input,
        output_path=None,  # we will handle writing at the end
        cfg=vel_config,
    )

    # 2) Optional: apply classifier
    if args.classify or args.evaluate:
        cls_cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=args.threshold)
        df_result = apply_ivt_classifier(df_result, cls_cfg)

    # 3) Optional: write TSV
    if args.output is not None:
        df_result.to_csv(args.output, sep="\t", index=False)

    # 4) Optional: evaluation report
    if args.evaluate:
        evaluate_ivt_vs_ground_truth(df_result)

    # 5) Optional: plots
    if not args.no_plot:
        if args.with_events:
            _plot_velocity_and_classification(df_result, vel_config)
        else:
            _plot_velocity_only(df_result, vel_config)
