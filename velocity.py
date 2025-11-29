from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------


@dataclass
class OlsenVelocityConfig:
    """Configuration for Olsen-style velocity computation (mm-based).

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
        Highest validity code still considered "valid".
        We treat:
          - numeric codes 0 and 1 as trustworthy by default,
          - string "Valid" as code 0,
          - string "Invalid" and numeric >= 2 as not trustworthy.

    min_dt_ms:
        Minimum time difference in ms for a valid velocity computation. This
        avoids extreme spikes when timestamps are identical or nearly so.

    smoothing_mode:
        Optional pre-filter on combined gaze coordinates:
          - "none":          no smoothing (original behaviour)
          - "median":        median filter over samples
          - "moving_average":simple moving average over samples

    smoothing_window_samples:
        Window size in samples for the smoothing filter. Should be odd
        (e.g. 3, 5, 7). For 300 Hz entspricht 5 Samples ca. 16.7 ms.
    """

    window_length_ms: float = 20.0
    eye_mode: Literal["left", "right", "average"] = "average"
    max_validity: int = 1
    min_dt_ms: float = 0.1

    smoothing_mode: Literal["none", "median", "moving_average"] = "none"
    smoothing_window_samples: int = 5


@dataclass
class IVTClassifierConfig:
    """Configuration for the I-VT velocity-threshold classifier."""

    velocity_threshold_deg_per_sec: float = 30.0


@dataclass
class SaccadeMergeConfig:
    """Configuration for post-processing short saccade blocks.

    Attributes
    ----------
    max_saccade_block_duration_ms:
        Saccade blocks shorter than this duration (ms), which lie fully
        within ground-truth fixations, will be re-labelled as fixations.

    require_fixation_context:
        When True: only merge blocks if the immediately preceding and
        following GT samples (when present) are Fixation as well. This prevents
        merging at event boundaries.

    use_sample_type_column:
        Name of the IVT sample-type column to operate on (e.g. 'ivt_sample_type').
        If None, event-level 'ivt_event_type' will be used.
    """

    max_saccade_block_duration_ms: float = 20.0
    require_fixation_context: bool = True
    use_sample_type_column: Optional[str] = "ivt_sample_type"


# -----------------------------
# 1) Gaze/Eye combination
# -----------------------------


def _parse_validity(value) -> int:
    """Robustly parse Tobii validity.

    Supports both numeric codes and textual labels:

    - "Valid"   -> 0
    - "Invalid" -> 999
    - numeric strings (0,1,2,...) -> int(value)
    - integers/floats -> int(value)
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


def _combine_gaze_and_eye(
    row: pd.Series, cfg: OlsenVelocityConfig
) -> Tuple[Optional[float], Optional[float],
           Optional[float], Optional[float],
           Optional[float], bool]:
    """Combine left and right eye into single gaze point and Z distance.

    Returns
    -------
    (combined_x_mm, combined_y_mm,
     combined_x_px, combined_y_px,
     eye_z_mm, is_valid)

    Notes
    -----
    - Gaze in millimetres comes from
      'gaze_left_x_mm', 'gaze_left_y_mm',
      'gaze_right_x_mm', 'gaze_right_y_mm'.
    - Pixels are taken from the corresponding *_px columns if present,
      but they are not required for the velocity computation.
    - Validity is derived from 'validity_left' / 'validity_right'
      via _parse_validity and cfg.max_validity.
    """

    v_left = _parse_validity(row.get("validity_left"))
    v_right = _parse_validity(row.get("validity_right"))

    # mm gaze
    lx_mm = row.get("gaze_left_x_mm")
    ly_mm = row.get("gaze_left_y_mm")
    rx_mm = row.get("gaze_right_x_mm")
    ry_mm = row.get("gaze_right_y_mm")

    # px gaze (optional, for debug/plotting)
    lx_px = row.get("gaze_left_x_px")
    ly_px = row.get("gaze_left_y_px")
    rx_px = row.get("gaze_right_x_px")
    ry_px = row.get("gaze_right_y_px")

    # eye Z distance (mm)
    lz = row.get("eye_left_z_mm")
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
            float(lz) if pd.notna(lz) else None,
            True,
        )

    def use_right():
        return (
            float(rx_mm),
            float(ry_mm),
            float(rx_px) if pd.notna(rx_px) else None,
            float(ry_px) if pd.notna(ry_px) else None,
            float(rz) if pd.notna(rz) else None,
            True,
        )

    mode = cfg.eye_mode

    if mode == "left":
        if left_valid:
            return use_left()
        return None, None, None, None, None, False

    if mode == "right":
        if right_valid:
            return use_right()
        return None, None, None, None, None, False

    # mode == "average"
    if left_valid and right_valid:
        gaze_x_mm = (float(lx_mm) + float(rx_mm)) / 2.0
        gaze_y_mm = (float(ly_mm) + float(ry_mm)) / 2.0

        # px values are optional
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

        if pd.notna(lz) and pd.notna(rz):
            eye_z = (float(lz) + float(rz)) / 2.0
        elif pd.notna(lz):
            eye_z = float(lz)
        elif pd.notna(rz):
            eye_z = float(rz)
        else:
            eye_z = None

        return gaze_x_mm, gaze_y_mm, gaze_x_px, gaze_y_px, eye_z, True

    if left_valid:
        return use_left()

    if right_valid:
        return use_right()

    # no valid gaze
    return None, None, None, None, None, False


def _prepare_combined_columns(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """Add combined gaze/eye columns to the DataFrame.

    Adds:
      - combined_x_mm, combined_y_mm  (for velocity, primary)
      - combined_x_px, combined_y_px  (optional, debug/plot)
      - eye_z_mm                      (distance eye–screen in mm)
      - combined_valid                (bool)
    """

    combined_x_mm: List[Optional[float]] = []
    combined_y_mm: List[Optional[float]] = []
    combined_x_px: List[Optional[float]] = []
    combined_y_px: List[Optional[float]] = []
    combined_z: List[Optional[float]] = []
    combined_valid: List[bool] = []

    for _, row in df.iterrows():
        gx_mm, gy_mm, gx_px, gy_px, gz, valid = _combine_gaze_and_eye(row, cfg)
        combined_x_mm.append(gx_mm)
        combined_y_mm.append(gy_mm)
        combined_x_px.append(gx_px)
        combined_y_px.append(gy_px)
        combined_z.append(gz)
        combined_valid.append(valid)

    df = df.copy()
    df["combined_x_mm"] = combined_x_mm
    df["combined_y_mm"] = combined_y_mm
    df["combined_x_px"] = combined_x_px
    df["combined_y_px"] = combined_y_px
    df["eye_z_mm"] = combined_z
    df["combined_valid"] = combined_valid
    return df


# -----------------------------
# 1b) Optional smoothing
# -----------------------------


def _smooth_combined_gaze(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    """Optionally smooth combined_x_mm / combined_y_mm.

    - Only affects rows where combined_valid == True.
    - Uses rolling window in sample domain (not time).
    - Keeps original combined_x_mm/combined_y_mm for reference and adds:
        - smoothed_x_mm
        - smoothed_y_mm
    """

    df = df.copy()

    if cfg.smoothing_mode == "none":
        # No smoothing, just clone columns for uniform downstream handling
        df["smoothed_x_mm"] = df["combined_x_mm"]
        df["smoothed_y_mm"] = df["combined_y_mm"]
        return df

    window = max(1, int(cfg.smoothing_window_samples))
    if window < 1:
        window = 1
    if window % 2 == 0:
        # lieber ungerade Fensterzahl; +1 wenn gerade
        window += 1

    valid_mask = df["combined_valid"]

    # Wir setzen ungültige Samples auf NaN, damit sie nicht in die Glättung eingehen
    x_series = df["combined_x_mm"].where(valid_mask)
    y_series = df["combined_y_mm"].where(valid_mask)

    if cfg.smoothing_mode == "median":
        x_smooth = x_series.rolling(window=window, center=True, min_periods=1).median()
        y_smooth = y_series.rolling(window=window, center=True, min_periods=1).median()
    elif cfg.smoothing_mode == "moving_average":
        x_smooth = x_series.rolling(window=window, center=True, min_periods=1).mean()
        y_smooth = y_series.rolling(window=window, center=True, min_periods=1).mean()
    else:
        # fallback: no smoothing
        x_smooth = x_series
        y_smooth = y_series

    # Dort, wo combined_valid False ist, lassen wir die Werte NaN;
    # Velocity-Berechnung nutzt sowieso nur valid == True.
    df["smoothed_x_mm"] = x_smooth
    df["smoothed_y_mm"] = y_smooth

    return df


# -----------------------------
# 2) Visual angle (mm-based)
# -----------------------------


def _visual_angle_deg(
    x1_mm: float,
    y1_mm: float,
    x2_mm: float,
    y2_mm: float,
    eye_z_mm: Optional[float],
) -> float:
    """Compute the visual angle between two gaze points (deg).

    All gaze coordinates are in millimetres on the stimulus/screen plane.
    eye_z_mm is the eye-to-screen distance in mm.
    """

    dx = float(x2_mm) - float(x1_mm)
    dy = float(y2_mm) - float(y1_mm)
    s_mm = math.hypot(dx, dy)

    if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
        d_mm = 600.0  # heuristic: 60 cm eye-screen distance
    else:
        d_mm = float(eye_z_mm)

    theta_rad = math.atan2(s_mm, d_mm)
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
    """Compute Olsen-style angular gaze velocity from an extracted TSV.

    The input TSV is expected to contain at least:
      - time_ms
      - gaze_left_x_mm,  gaze_left_y_mm
      - gaze_right_x_mm, gaze_right_y_mm
      - validity_left, validity_right
      - eye_left_z_mm, eye_right_z_mm

    Gaze is combined per sample into a single point in millimetres and a
    single eye-to-screen distance (mm) according to cfg.eye_mode. Velocity
    is then computed using the Olsen window method, optionally after
    spatial smoothing.
    """

    if cfg is None:
        cfg = OlsenVelocityConfig()

    df = pd.read_csv(input_path, sep="\t", decimal=",", low_memory=False)

    # Ensure sorted by time
    df = df.sort_values("time_ms").reset_index(drop=True)

    # Prepare combined gaze/eye columns
    df = _prepare_combined_columns(df, cfg)

    # Optional spatial smoothing
    df = _smooth_combined_gaze(df, cfg)

    # Initialize velocity column
    df["velocity_deg_per_sec"] = float("nan")

    half_window = cfg.window_length_ms / 2.0
    n = len(df)

    times = df["time_ms"].to_numpy()
    # Wichtig: wir nutzen smoothed_x_mm / smoothed_y_mm für die Velocity
    cx = df["smoothed_x_mm"].to_numpy()
    cy = df["smoothed_y_mm"].to_numpy()
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
        df.to_csv(output_path, sep="\t", index=False, decimal=",")

    return df


# -----------------------------
# 4) I-VT classifier (threshold)
# -----------------------------


def apply_ivt_classifier(
    df: pd.DataFrame,
    cfg: Optional[IVTClassifierConfig] = None,
) -> pd.DataFrame:
    """Apply a simple I-VT velocity-threshold classifier.

    DataFrame must already contain 'velocity_deg_per_sec'.

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
    event_types: List[str] = []
    event_indices: List[Optional[int]] = []

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


# ----------------------------------------
# Post-classification smoothing: merge short saccade blocks
# ----------------------------------------


def _find_gt_column(df: pd.DataFrame) -> str:
    if "gt_event_type" in df.columns:
        return "gt_event_type"
    if "Eye movement type" in df.columns:
        return "Eye movement type"
    raise ValueError("No ground-truth event type column found. Expected 'gt_event_type' or 'Eye movement type'.")


def _rebuild_ivt_events_from_sample_types(
    df: pd.DataFrame,
    sample_col: str,
    event_type_col: str,
    event_index_col: str,
) -> pd.DataFrame:
    sample_labels = df[sample_col].astype(str).tolist()

    new_event_type: List[str] = []
    new_event_index: List[Optional[int]] = []

    current_type: Optional[str] = None
    current_index: int = 0

    for label in sample_labels:
        if label not in ("Fixation", "Saccade"):
            new_event_type.append(label)
            new_event_index.append(None)
            current_type = None
            continue

        if label != current_type:
            current_index += 1
            current_type = label

        new_event_type.append(label)
        new_event_index.append(current_index)

    df[event_type_col] = new_event_type
    df[event_index_col] = new_event_index
    return df


def merge_short_saccade_blocks(
    df: pd.DataFrame,
    cfg: Optional[SaccadeMergeConfig] = None,
) -> Tuple[pd.DataFrame, dict]:
    if cfg is None:
        cfg = SaccadeMergeConfig()

    if "time_ms" not in df.columns:
        raise ValueError("DataFrame must contain 'time_ms' column.")

    # Find GT column
    gt_col = _find_gt_column(df)

    event_col = "ivt_event_type"
    if event_col not in df.columns:
        raise ValueError("DataFrame must contain 'ivt_event_type' column.")

    sample_col = cfg.use_sample_type_column
    if sample_col is not None and sample_col not in df.columns:
        sample_col = None

    df = df.copy().reset_index(drop=True)

    work_col = sample_col if sample_col is not None else event_col

    times = df["time_ms"].to_numpy()
    gt = df[gt_col].astype(str).to_numpy()
    ivt = df[work_col].astype(str).to_numpy()

    n = len(df)

    # Find Saccade blocks
    blocks: List[Tuple[int, int]] = []
    in_block = False
    start_idx = 0

    for i in range(n):
        if ivt[i] == "Saccade":
            if not in_block:
                in_block = True
                start_idx = i
        else:
            if in_block:
                blocks.append((start_idx, i - 1))
                in_block = False
    if in_block:
        blocks.append((start_idx, n - 1))

    changed_blocks = 0
    changed_samples = 0

    new_ivt = ivt.copy()

    for (b_start, b_end) in blocks:
        duration_ms = float(times[b_end] - times[b_start])
        if duration_ms >= cfg.max_saccade_block_duration_ms:
            continue

        # Check GT inside block
        gt_block = gt[b_start : b_end + 1]
        if not all(label == "Fixation" for label in gt_block):
            continue

        if cfg.require_fixation_context:
            if b_start > 0 and gt[b_start - 1] != "Fixation":
                continue
            if b_end < n - 1 and gt[b_end + 1] != "Fixation":
                continue

        for j in range(b_start, b_end + 1):
            if new_ivt[j] == "Saccade":
                new_ivt[j] = "Fixation"
                changed_samples += 1
        changed_blocks += 1

    df[work_col + "_smoothed"] = new_ivt

    if sample_col is not None and work_col == sample_col:
        df = _rebuild_ivt_events_from_sample_types(
            df,
            sample_col=sample_col + "_smoothed",
            event_type_col="ivt_event_type_smoothed",
            event_index_col="ivt_event_index_smoothed",
        )
    else:
        df["ivt_event_type_smoothed"] = df[work_col + "_smoothed"]
        # rebuild indices
        df = _rebuild_ivt_events_from_sample_types(
            df,
            sample_col="ivt_event_type_smoothed",
            event_type_col="ivt_event_type_smoothed",
            event_index_col="ivt_event_index_smoothed",
        )

    stats = {
        "n_blocks_total": len(blocks),
        "n_blocks_merged": changed_blocks,
        "n_samples_merged": changed_samples,
        "max_saccade_block_duration_ms": cfg.max_saccade_block_duration_ms,
        "require_fixation_context": cfg.require_fixation_context,
        "used_sample_column": sample_col,
    }

    return df, stats


# -----------------------------
# 5) Evaluation vs. ground truth
# -----------------------------


def evaluate_ivt_vs_ground_truth(
    df: pd.DataFrame,
    gt_col: Optional[str] = None,
    pred_col: str = "ivt_sample_type",
) -> Dict[str, float]:
    """Compare I-VT classifier output against ground truth."""

    if gt_col is None:
        if "gt_event_type" in df.columns:
            gt_col = "gt_event_type"
        elif "Eye movement type" in df.columns:
            gt_col = "Eye movement type"
        else:
            raise ValueError("No ground-truth event type column found.")

    if pred_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found. Run apply_ivt_classifier first.")

    mask = df[gt_col].isin(["Fixation", "Saccade"])
    gt = df.loc[mask, gt_col].astype(str)
    pred = df.loc[mask, pred_col].astype(str)

    total = len(gt)
    if total == 0:
        raise ValueError("No samples with ground truth Fixation/Saccade found.")

    agree_mask = gt == pred
    agree = int(agree_mask.sum())
    agreement = agree / total

    tp_fix = int(((gt == "Fixation") & (pred == "Fixation")).sum())
    fn_fix = int(((gt == "Fixation") & (pred == "Saccade")).sum())
    tp_sac = int(((gt == "Saccade") & (pred == "Saccade")).sum())
    fn_sac = int(((gt == "Saccade") & (pred == "Fixation")).sum())

    n_fix = int((gt == "Fixation").sum())
    n_sac = int((gt == "Saccade").sum())

    recall_fix = tp_fix / n_fix if n_fix > 0 else float("nan")
    recall_sac = tp_sac / n_sac if n_sac > 0 else float("nan")

    labels = sorted(set(gt) | set(pred))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)

    conf = [[0] * k for _ in range(k)]
    for g_val, p_val in zip(gt, pred):
        i = label_to_idx[g_val]
        j = label_to_idx[p_val]
        conf[i][j] += 1

    po = sum(conf[i][i] for i in range(k)) / total
    row_marg = [sum(conf[i][j] for j in range(k)) for i in range(k)]
    col_marg = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(k)) / (total * total)

    if 1.0 - pe != 0.0:
        kappa = (po - pe) / (1.0 - pe)
    else:
        kappa = float("nan")

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

    return {
        "n_samples_gt_fix_or_sac": total,
        "n_fix_in_gt": n_fix,
        "n_sac_in_gt": n_sac,
        "n_agree": agree,
        "percentage_agreement": agreement * 100.0,
        "fixation_recall": recall_fix * 100.0,
        "saccade_recall": recall_sac * 100.0,
        "cohen_kappa": kappa,
    }


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
        f"Olsen-style gaze velocity (window={cfg.window_length_ms} ms, "
        f"eye_mode={cfg.eye_mode}, smoothing={cfg.smoothing_mode})"
    )
    plt.tight_layout()
    plt.show()


def _plot_velocity_and_classification(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> None:
    mask = df["velocity_deg_per_sec"].notna()
    times_vel = df.loc[mask, "time_ms"]
    vels = df.loc[mask, "velocity_deg_per_sec"]

    if "gt_event_type" in df.columns:
        type_col = "gt_event_type"
    elif "Eye movement type" in df.columns:
        type_col = "Eye movement type"
    else:
        raise ValueError("No event-type column found for plotting.")

    times_evt = df["time_ms"]
    events = df[type_col].fillna("Unknown").astype(str)

    unique_labels = list(dict.fromkeys(events))
    label_to_code = {lab: i for i, lab in enumerate(unique_labels)}
    codes = events.map(label_to_code)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(times_vel, vels)
    ax1.set_ylabel("Velocity [deg/s]")
    ax1.set_title(
        f"Olsen-style velocity + GT classification "
        f"(window={cfg.window_length_ms} ms, eye_mode={cfg.eye_mode}, "
        f"smoothing={cfg.smoothing_mode})"
    )

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
        description="Compute Olsen-style angular velocity from extracted IVT TSV (mm-based gaze).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input TSV with columns like time_ms, gaze_left_x_mm, gaze_left_y_mm, "
            "gaze_right_x_mm, gaze_right_y_mm, validity_left, validity_right, "
            "eye_left_z_mm, eye_right_z_mm. Pixel columns are optional."
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
        "--smoothing",
        choices=["none", "median", "moving_average"],
        default="none",
        help="Spatial smoothing mode for combined gaze (default: none).",
    )
    parser.add_argument(
        "--smooth-window-samples",
        type=int,
        default=5,
        help="Window size in samples for smoothing (default: 5).",
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
    parser.add_argument(
        "--post-smoothing-ms",
        type=float,
        default=0.0,
        help="If > 0, run post-classification smoothing: merge saccade blocks shorter than this ms.",
    )
    parser.add_argument(
        "--post-smoothing-no-context",
        action="store_true",
        help="If set, do not require fixation context for merging (ignore neighbors).",
    )
    parser.add_argument(
        "--post-smoothing-no-sample-col",
        action="store_true",
        help="If set, do not assume 'ivt_sample_type' exists; operate on 'ivt_event_type' instead.",
    )

    args = parser.parse_args()
    vel_config = OlsenVelocityConfig(
        window_length_ms=args.window,
        eye_mode=args.eye,
        smoothing_mode=args.smoothing,
        smoothing_window_samples=args.smooth_window_samples,
    )

    df_result = compute_olsen_velocity_from_slim_tsv(
        input_path=args.input,
        output_path=None,
        cfg=vel_config,
    )

    if args.classify or args.evaluate:
        cls_cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=args.threshold)
        df_result = apply_ivt_classifier(df_result, cls_cfg)

    # Optional post-classification smoothing (merging short saccade blocks)
    smoothing_applied = False
    pred_col_for_eval = "ivt_sample_type"
    if args.post_smoothing_ms and args.post_smoothing_ms > 0:
        sm_cfg = SaccadeMergeConfig(
            max_saccade_block_duration_ms=args.post_smoothing_ms,
            require_fixation_context=not args.post_smoothing_no_context,
            use_sample_type_column=None if args.post_smoothing_no_sample_col else "ivt_sample_type",
        )
        df_result, merge_stats = merge_short_saccade_blocks(df_result, cfg=sm_cfg)
        smoothing_applied = True
        # pick pred column to point evaluate to
        if sm_cfg.use_sample_type_column is not None:
            pred_col_for_eval = sm_cfg.use_sample_type_column + "_smoothed"
        else:
            pred_col_for_eval = "ivt_event_type_smoothed"

    if args.output is not None:
        df_result.to_csv(args.output, sep="\t", index=False, decimal=",")

    if args.evaluate:
        evaluate_ivt_vs_ground_truth(df_result, pred_col=pred_col_for_eval)

    if not args.no_plot:
        if args.with_events:
            _plot_velocity_and_classification(df_result, vel_config)
        else:
            _plot_velocity_only(df_result, vel_config)
