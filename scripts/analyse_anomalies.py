#!/usr/bin/env python3
"""
Empirical analysis of two anomalies identified in the Bachelor's thesis:
  1. C5 anomaly — why Right-eye-occluded 120 Hz gives κ=0.797 vs C3 κ=0.943 and C6 κ=0.998
  2. Moving-average smoothing degradation — why MA-9 collapses saccade recall to 81%

This script runs the IVT pipeline on each relevant condition, computes metrics,
and exports disagreement data for further analysis.  All numerical claims in
the thesis section 9.5 must trace back to the output of this script.

Usage (from repo root, with venv active):
    python scripts/analyse_anomalies.py
"""
from __future__ import annotations

import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ivt_filter.config import (
    FixationPostConfig,
    IVTClassifierConfig,
    OlsenVelocityConfig,
    PipelineConfig,
)
from ivt_filter.evaluation.evaluation import compute_ivt_metrics
from ivt_filter.io.pipeline import IVTPipeline

INPUTS = REPO / "test_data" / "inputs"
OUTPUTS = REPO / "scripts" / "anomaly_analysis_outputs"
OUTPUTS.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: build a standard Tobii IVT preset pipeline config
# ---------------------------------------------------------------------------

def _make_std_config(
    eye_mode: str,
    *,
    smoothing_mode: str = "none",
    smoothing_window: int = 5,
    merge: bool = False,
    discard: bool = False,
    merge_gap_ms: float = 75.0,
    merge_angle_deg: float = 0.5,
    min_fixation_ms: float = 60.0,
    smoothing_mode_str: str | None = None,
) -> PipelineConfig:
    vel = OlsenVelocityConfig(
        window_length_ms=20.0,
        velocity_method="tobii_gaze_dir",
        time_column="time_ms",
        time_unit="ms",
        eye_mode=eye_mode,
        smoothing_mode=smoothing_mode,
        smoothing_window_samples=smoothing_window,
        gap_fill_enabled=False,
        gap_fill_max_gap_ms=75.0,
    )
    clf = IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0)

    fix_post: FixationPostConfig | None = None
    if merge or discard:
        fix_post = FixationPostConfig()
        if merge:
            fix_post.merge_adjacent_fixations = True
            fix_post.max_time_gap_ms = merge_gap_ms
            fix_post.max_angle_deg = merge_angle_deg
        if discard:
            fix_post.discard_short_fixations = True
            fix_post.min_fixation_duration_ms = min_fixation_ms
        fix_post.__post_init__()

    return PipelineConfig(
        velocity=vel,
        classifier=clf,
        classify=True,
        fixation_post=fix_post,
    )


def _pred_col(config: PipelineConfig) -> str:
    if config.fixation_post:
        return "ivt_event_type_post"
    return "ivt_sample_type"


def _run(input_path: Path, config: PipelineConfig) -> pd.DataFrame:
    return IVTPipeline(config).run(str(input_path), evaluate=False, plot=False)


def _metrics(df: pd.DataFrame, pred_col: str) -> dict:
    return compute_ivt_metrics(df, pred_col=pred_col)


# ===========================================================================
# PHASE 1: Baseline — C2 verification
# ===========================================================================

print("=" * 70)
print("PHASE 1: C2 BASELINE VERIFICATION")
print("=" * 70)

c2_path = INPUTS / "IVT30_extracted.tsv"
c2_config = _make_std_config("average")
c2_df = _run(c2_path, c2_config)
c2_metrics = _metrics(c2_df, _pred_col(c2_config))

print(f"C2 Bino 300Hz:")
print(f"  N samples     : {c2_metrics['n_samples_total']}")
print(f"  Agreement     : {c2_metrics['percentage_agreement']:.4f}%")
print(f"  Cohen's κ     : {c2_metrics['cohen_kappa']:.4f}")
print(f"  Fix Recall    : {c2_metrics['fixation_recall']:.4f}%")
print(f"  Sac Recall    : {c2_metrics['saccade_recall']:.4f}%")
print()

kappa_c2 = c2_metrics['cohen_kappa']
if 0.985 <= kappa_c2 <= 0.992:
    print(f"✓ C2 κ={kappa_c2:.4f} within expected range [0.985, 0.992] — baseline OK")
else:
    print(f"✗ WARNING: C2 κ={kappa_c2:.4f} outside expected range [0.985, 0.992]")
    print("  Stopping — fix baseline before proceeding.")
    sys.exit(1)
print()


# ===========================================================================
# PHASE 2: C5 ANOMALY ANALYSIS
# ===========================================================================

print("=" * 70)
print("PHASE 2: C5 ANOMALY — C3, C5, C6 METRICS")
print("=" * 70)

# C5: Right-eye occluded 120Hz — LeftFixation_extracted.tsv, left eye, median-3, merge+discard
c5_path = INPUTS / "LeftFixation_extracted.tsv"
c5_config = _make_std_config(
    "left",
    smoothing_mode="median",
    smoothing_window=3,
    merge=True,
    discard=True,
)
c5_df = _run(c5_path, c5_config)
c5_pred_col = _pred_col(c5_config)
c5_metrics = _metrics(c5_df, c5_pred_col)

# C6: Right-eye occluded 300Hz — left20ms30_extracted.tsv, left eye, no smoothing
c6_path = INPUTS / "left20ms30_extracted.tsv"
c6_config = _make_std_config("left")
c6_df = _run(c6_path, c6_config)
c6_pred_col = _pred_col(c6_config)
c6_metrics = _metrics(c6_df, c6_pred_col)

# C3 proxy: Left-eye occluded 120Hz — use RIGHT eye on binocular 120Hz recording
# The binocular 120Hz recording is I-VT-frequency120Fixation_input.tsv (or extracted version)
# C3 uses right eye on binocular 120Hz recording, median-3, merge+discard
c3_path = INPUTS / "I-VT-frequency120Fixation_input.tsv"
if not c3_path.exists():
    c3_path = INPUTS / "I-VT-frequency120Fixation export_extracted.tsv"
c3_config = _make_std_config(
    "right",
    smoothing_mode="median",
    smoothing_window=3,
    merge=True,
    discard=True,
)
c3_df = _run(c3_path, c3_config)
c3_pred_col = _pred_col(c3_config)
c3_metrics = _metrics(c3_df, c3_pred_col)

print(f"C3 Left-occ 120Hz (proxy):")
print(f"  N samples     : {c3_metrics['n_samples_total']}")
print(f"  Agreement     : {c3_metrics['percentage_agreement']:.4f}%")
print(f"  Cohen's κ     : {c3_metrics['cohen_kappa']:.4f}")
print(f"  Fix Recall    : {c3_metrics['fixation_recall']:.4f}%")
print(f"  Sac Recall    : {c3_metrics['saccade_recall']:.4f}%")
print()
print(f"C5 Right-occ 120Hz:")
print(f"  N samples     : {c5_metrics['n_samples_total']}")
print(f"  Agreement     : {c5_metrics['percentage_agreement']:.4f}%")
print(f"  Cohen's κ     : {c5_metrics['cohen_kappa']:.4f}")
print(f"  Fix Recall    : {c5_metrics['fixation_recall']:.4f}%")
print(f"  Sac Recall    : {c5_metrics['saccade_recall']:.4f}%")
print()
print(f"C6 Right-occ 300Hz:")
print(f"  N samples     : {c6_metrics['n_samples_total']}")
print(f"  Agreement     : {c6_metrics['percentage_agreement']:.4f}%")
print(f"  Cohen's κ     : {c6_metrics['cohen_kappa']:.4f}")
print(f"  Fix Recall    : {c6_metrics['fixation_recall']:.4f}%")
print(f"  Sac Recall    : {c6_metrics['saccade_recall']:.4f}%")
print()


# ---------------------------------------------------------------------------
# STEP 2.2: Disagreement localisation
# ---------------------------------------------------------------------------

print("-" * 60)
print("STEP 2.2: DISAGREEMENT LOCALISATION")
print("-" * 60)

def make_disagreement_df(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """Create per-sample comparison with match flag and validity columns."""
    # Determine GT column
    for col in ("gt_sample_type", "gt_event_type", "Eye movement type"):
        if col in df.columns:
            gt_col = col
            break
    else:
        raise ValueError("No GT column found")

    gt = df[gt_col].astype(str)
    pred = df[pred_col].astype(str)

    rows = []
    for i, idx in enumerate(df.index):
        row = df.loc[idx]
        rows.append({
            "sample_idx": i,
            "time_ms": row.get("time_ms", float("nan")),
            "gt": gt[idx],
            "pred": pred[idx],
            "match": gt[idx] == pred[idx],
            "validity_left": row.get("validity_left", float("nan")),
            "validity_right": row.get("validity_right", float("nan")),
        })
    return pd.DataFrame(rows)


print("Building per-sample comparison for C5...")
c5_compare = make_disagreement_df(c5_df, c5_pred_col)
c5_disagree = c5_compare[~c5_compare["match"]]
c5_disagree.to_csv(OUTPUTS / "disagreements_C5.csv", index=False)
print(f"  Total samples: {len(c5_compare)}")
print(f"  Disagreements: {len(c5_disagree)} ({100.0 * len(c5_disagree) / len(c5_compare):.2f}%)")

print("Building per-sample comparison for C3...")
c3_compare = make_disagreement_df(c3_df, c3_pred_col)
c3_disagree = c3_compare[~c3_compare["match"]]
c3_disagree.to_csv(OUTPUTS / "disagreements_C3.csv", index=False)
print(f"  Total samples: {len(c3_compare)}")
print(f"  Disagreements: {len(c3_disagree)} ({100.0 * len(c3_disagree) / len(c3_compare):.2f}%)")
print()

# Time distribution of disagreements
def analyse_time_distribution(compare_df: pd.DataFrame, label: str) -> dict:
    disagree = compare_df[~compare_df["match"]]
    n_total = len(compare_df)
    n_disagree = len(disagree)

    if n_total == 0 or n_disagree == 0:
        return {}

    t_min = compare_df["time_ms"].min()
    t_max = compare_df["time_ms"].max()
    duration_ms = t_max - t_min

    # Split into thirds
    t1 = t_min + duration_ms / 3
    t2 = t_min + 2 * duration_ms / 3

    d_first = len(disagree[disagree["time_ms"] < t1])
    d_mid = len(disagree[(disagree["time_ms"] >= t1) & (disagree["time_ms"] < t2)])
    d_last = len(disagree[disagree["time_ms"] >= t2])

    print(f"  {label} disagreement time distribution:")
    print(f"    First third  (< {t1:.0f}ms): {d_first} ({100.*d_first/n_disagree:.1f}%)")
    print(f"    Middle third               : {d_mid} ({100.*d_mid/n_disagree:.1f}%)")
    print(f"    Last third   (>= {t2:.0f}ms): {d_last} ({100.*d_last/n_disagree:.1f}%)")

    return {"first": d_first, "mid": d_mid, "last": d_last}

print("Time distribution:")
c5_time_dist = analyse_time_distribution(c5_compare, "C5")
c3_time_dist = analyse_time_distribution(c3_compare, "C3")
print()

# Disagreements by GT label
print("C5 disagreements by GT label:")
if len(c5_disagree) > 0:
    for lbl, cnt in c5_disagree["gt"].value_counts().items():
        print(f"  GT={lbl}: {cnt} ({100.*cnt/len(c5_disagree):.1f}%)")

print("C3 disagreements by GT label:")
if len(c3_disagree) > 0:
    for lbl, cnt in c3_disagree["gt"].value_counts().items():
        print(f"  GT={lbl}: {cnt} ({100.*cnt/len(c3_disagree):.1f}%)")
print()


# ---------------------------------------------------------------------------
# STEP 2.3: Validity pattern analysis
# ---------------------------------------------------------------------------

print("-" * 60)
print("STEP 2.3: VALIDITY PATTERN ANALYSIS")
print("-" * 60)

def analyse_validity(df: pd.DataFrame, compare_df: pd.DataFrame, label: str) -> dict:
    """Analyse validity patterns and correlation with disagreements."""

    # Validity columns
    if "validity_left" not in df.columns or "validity_right" not in df.columns:
        print(f"  {label}: validity columns missing")
        return {}

    # Convert validity to numeric, treating "Invalid" strings as invalid
    def parse_validity(col: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(col, errors="coerce")
        # "Invalid" string → NaN → treat as >1 (invalid)
        # In Tobii: 0=valid, 1=probably valid, 2+=invalid
        return numeric

    vl = parse_validity(df["validity_left"])
    vr = parse_validity(df["validity_right"])

    n_total = len(df)

    # Valid means validity ≤ 1 (0=valid, 1=probably valid)
    vl_valid = (vl <= 1).fillna(False)
    vr_valid = (vr <= 1).fillna(False)
    vl_invalid = (~vl_valid)
    vr_invalid = (~vr_valid)

    print(f"  {label} validity summary:")
    print(f"    N total          : {n_total}")
    print(f"    Left valid (≤1)  : {vl_valid.sum()} ({100.*vl_valid.sum()/n_total:.1f}%)")
    print(f"    Left invalid (>1): {vl_invalid.sum()} ({100.*vl_invalid.sum()/n_total:.1f}%)")
    print(f"    Right valid (≤1) : {vr_valid.sum()} ({100.*vr_valid.sum()/n_total:.1f}%)")
    print(f"    Right invalid(>1): {vr_invalid.sum()} ({100.*vr_invalid.sum()/n_total:.1f}%)")

    # Count validity transitions
    def count_transitions(valid_series: pd.Series) -> int:
        shifted = valid_series.shift(1)
        return int((valid_series != shifted).sum()) - 1  # subtract first-row non-transition

    vl_trans = count_transitions(vl_valid)
    vr_trans = count_transitions(vr_valid)
    print(f"    Left validity transitions : {vl_trans}")
    print(f"    Right validity transitions: {vr_trans}")

    # Average length of invalid sequences
    def avg_invalid_run_length(valid_series: pd.Series) -> float:
        runs = []
        count = 0
        for v in valid_series:
            if not v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                    count = 0
        if count > 0:
            runs.append(count)
        return float(np.mean(runs)) if runs else 0.0

    avg_vl_inv = avg_invalid_run_length(vl_valid)
    avg_vr_inv = avg_invalid_run_length(vr_valid)
    print(f"    Avg left invalid run length : {avg_vl_inv:.1f} samples")
    print(f"    Avg right invalid run length: {avg_vr_inv:.1f} samples")

    # Disagreement correlation with validity transitions
    # Find transition sample indices
    def transition_indices(valid_series: pd.Series, df_index) -> list[int]:
        s = valid_series.values
        indices = []
        for i in range(1, len(s)):
            if s[i] != s[i-1]:
                indices.append(i)
        return indices

    # Use compare_df which has 0-based sample_idx
    vl_valid_arr = vl_valid.values
    vr_valid_arr = vr_valid.values

    vl_trans_idx = [i for i in range(1, len(vl_valid_arr)) if vl_valid_arr[i] != vl_valid_arr[i-1]]
    vr_trans_idx = [i for i in range(1, len(vr_valid_arr)) if vr_valid_arr[i] != vr_valid_arr[i-1]]
    all_trans = sorted(set(vl_trans_idx + vr_trans_idx))

    # How many disagreements are within 100ms of a validity transition?
    # 100ms at 120Hz = 12 samples
    window_samples = 12  # ~100ms at 120Hz

    disagree_idx = set(compare_df.loc[~compare_df["match"], "sample_idx"].tolist())

    near_transition = 0
    far_from_transition = 0
    for d_idx in disagree_idx:
        is_near = any(abs(d_idx - t) <= window_samples for t in all_trans)
        if is_near:
            near_transition += 1
        else:
            far_from_transition += 1

    n_disagree = len(disagree_idx)
    print(f"    Disagreements near transition (±{window_samples} samples / ~100ms): "
          f"{near_transition} ({100.*near_transition/n_disagree:.1f}%)" if n_disagree > 0 else "    No disagreements")
    print(f"    Disagreements far from transition: "
          f"{far_from_transition} ({100.*far_from_transition/n_disagree:.1f}%)" if n_disagree > 0 else "")

    return {
        "n_total": n_total,
        "n_left_valid": int(vl_valid.sum()),
        "n_right_valid": int(vr_valid.sum()),
        "n_left_transitions": vl_trans,
        "n_right_transitions": vr_trans,
        "avg_left_invalid_run": avg_vl_inv,
        "avg_right_invalid_run": avg_vr_inv,
        "n_disagree_near_transition": near_transition,
        "n_disagree_far": far_from_transition,
        "n_disagree_total": n_disagree,
    }

c5_validity = analyse_validity(c5_df, c5_compare, "C5")
print()
c3_validity = analyse_validity(c3_df, c3_compare, "C3")
print()

# Also run C6 validity analysis for comparison
c6_compare = make_disagreement_df(c6_df, c6_pred_col)
c6_validity = analyse_validity(c6_df, c6_compare, "C6")
print()


# ---------------------------------------------------------------------------
# STEP 2.4: Eye offset analysis
# ---------------------------------------------------------------------------

print("-" * 60)
print("STEP 2.4: EYE OFFSET ANALYSIS")
print("-" * 60)

def analyse_eye_offset(df: pd.DataFrame, label: str) -> dict:
    """Analyse inter-eye gaze offset over time."""
    if "gaze_left_x_mm" not in df.columns or "gaze_right_x_mm" not in df.columns:
        print(f"  {label}: gaze_*_mm columns missing")
        return {}

    gl_x = pd.to_numeric(df["gaze_left_x_mm"], errors="coerce")
    gl_y = pd.to_numeric(df["gaze_left_y_mm"], errors="coerce")
    gr_x = pd.to_numeric(df["gaze_right_x_mm"], errors="coerce")
    gr_y = pd.to_numeric(df["gaze_right_y_mm"], errors="coerce")

    vl = pd.to_numeric(df.get("validity_left", pd.Series(dtype=float)), errors="coerce")
    vr = pd.to_numeric(df.get("validity_right", pd.Series(dtype=float)), errors="coerce")

    vl_ok = (vl <= 1).fillna(False)
    vr_ok = (vr <= 1).fillna(False)
    both_ok = vl_ok & vr_ok

    # Compute offset when both eyes are valid
    delta_x = gr_x - gl_x
    delta_y = gr_y - gl_y
    delta_mag = np.sqrt(delta_x**2 + delta_y**2)

    binocular = delta_mag[both_ok]
    n_bino = both_ok.sum()

    print(f"  {label} eye offset (both eyes valid, N={n_bino}):")
    if n_bino > 0:
        print(f"    Mean offset magnitude : {binocular.mean():.2f} mm")
        print(f"    Std offset magnitude  : {binocular.std():.2f} mm")
        print(f"    Min offset magnitude  : {binocular.min():.2f} mm")
        print(f"    Max offset magnitude  : {binocular.max():.2f} mm")

        # Drift over time: compare first 10% vs last 10%
        n_segment = max(1, n_bino // 10)
        first_offsets = binocular.iloc[:n_segment]
        last_offsets = binocular.iloc[-n_segment:]
        print(f"    First-10% mean offset : {first_offsets.mean():.2f} mm")
        print(f"    Last-10% mean offset  : {last_offsets.mean():.2f} mm")
        print(f"    Temporal drift        : {abs(last_offsets.mean() - first_offsets.mean()):.2f} mm")
    else:
        print(f"    No binocular samples found")

    return {
        "n_binocular": int(n_bino),
        "mean_offset_mm": float(binocular.mean()) if n_bino > 0 else float("nan"),
        "std_offset_mm": float(binocular.std()) if n_bino > 0 else float("nan"),
    }

c5_offset = analyse_eye_offset(c5_df, "C5 (Right-occ 120Hz)")
print()
c3_offset = analyse_eye_offset(c3_df, "C3 proxy (Left-occ 120Hz)")
print()
c6_offset = analyse_eye_offset(c6_df, "C6 (Right-occ 300Hz)")
print()

# Sampling rate comparison: binocular samples per second before occlusion
def count_bino_samples_per_sec(df: pd.DataFrame, label: str, window_ms: float = 1000.0) -> float:
    """Count binocular samples in the first window_ms of the recording."""
    vl = pd.to_numeric(df.get("validity_left", pd.Series(dtype=float)), errors="coerce")
    vr = pd.to_numeric(df.get("validity_right", pd.Series(dtype=float)), errors="coerce")
    t = pd.to_numeric(df.get("time_ms", pd.Series(dtype=float)), errors="coerce")

    vl_ok = (vl <= 1).fillna(False)
    vr_ok = (vr <= 1).fillna(False)
    both_ok = vl_ok & vr_ok

    t_start = t.min()
    window_mask = (t >= t_start) & (t < t_start + window_ms)
    n_bino_in_window = int(both_ok[window_mask].sum())
    rate = n_bino_in_window / (window_ms / 1000.0)
    print(f"  {label}: {n_bino_in_window} binocular samples in first {window_ms:.0f}ms "
          f"= {rate:.0f} Hz")
    return rate

print("Binocular samples per second (first 1000ms):")
c5_bino_rate = count_bino_samples_per_sec(c5_df, "C5 (120Hz)")
c6_bino_rate = count_bino_samples_per_sec(c6_df, "C6 (300Hz)")
print()


# ===========================================================================
# PHASE 3: MOVING AVERAGE ANOMALY
# ===========================================================================

print("=" * 70)
print("PHASE 3: MOVING AVERAGE ANOMALY ANALYSIS")
print("=" * 70)

# Use C2 binocular 300Hz data (IVT30_extracted.tsv)
# Run all 7 smoothing variants

ma_configs = {
    "no_smoothing":      _make_std_config("average", smoothing_mode="none"),
    "ma_3":              _make_std_config("average", smoothing_mode="moving_average", smoothing_window=3),
    "ma_5":              _make_std_config("average", smoothing_mode="moving_average", smoothing_window=5),
    "ma_9":              _make_std_config("average", smoothing_mode="moving_average", smoothing_window=9),
    "median_3":          _make_std_config("average", smoothing_mode="median", smoothing_window=3),
    "median_5":          _make_std_config("average", smoothing_mode="median", smoothing_window=5),
    "median_9":          _make_std_config("average", smoothing_mode="median", smoothing_window=9),
}

ma_results = {}
for name, config in ma_configs.items():
    print(f"  Running {name}...", end=" ", flush=True)
    df = _run(c2_path, config)
    m = _metrics(df, _pred_col(config))
    ma_results[name] = {
        "df": df,
        "metrics": m,
        "config": config,
    }
    print(f"κ={m['cohen_kappa']:.4f}, SacRecall={m['saccade_recall']:.1f}%")

print()
print("Smoothing comparison table:")
print(f"{'Name':<20} {'κ':>8} {'Agreement':>12} {'Fix Recall':>12} {'Sac Recall':>12}")
print("-" * 70)
for name, res in ma_results.items():
    m = res["metrics"]
    print(f"{name:<20} {m['cohen_kappa']:>8.4f} {m['percentage_agreement']:>12.2f}% "
          f"{m['fixation_recall']:>12.2f}% {m['saccade_recall']:>12.2f}%")
print()


# ---------------------------------------------------------------------------
# STEP 3.2: Velocity signal analysis
# ---------------------------------------------------------------------------

print("-" * 60)
print("STEP 3.2: VELOCITY SIGNAL ANALYSIS")
print("-" * 60)

# Extract velocity for no_smoothing, MA-9, Median-9 from output dataframes
def get_velocity(df: pd.DataFrame) -> pd.Series:
    for col in ("ivt_velocity_deg_per_sec", "velocity", "velocity_deg_per_sec",
                "ivt_velocity", "computed_velocity"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    # Try to find any column with 'velocity' in name
    vel_cols = [c for c in df.columns if "velocity" in c.lower() and "deg" in c.lower()]
    if vel_cols:
        return pd.to_numeric(df[vel_cols[0]], errors="coerce")
    vel_cols2 = [c for c in df.columns if "velocity" in c.lower()]
    if vel_cols2:
        return pd.to_numeric(df[vel_cols2[0]], errors="coerce")
    return pd.Series(dtype=float)

# Find the GT column
def get_gt(df: pd.DataFrame) -> pd.Series:
    for col in ("gt_sample_type", "gt_event_type", "Eye movement type"):
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series(dtype=str)

no_smooth_df = ma_results["no_smoothing"]["df"]
ma9_df = ma_results["ma_9"]["df"]
med9_df = ma_results["median_9"]["df"]

vel_no_smooth = get_velocity(no_smooth_df)
vel_ma9 = get_velocity(ma9_df)
vel_med9 = get_velocity(med9_df)

print(f"Velocity column found (no_smooth): {vel_no_smooth.name if hasattr(vel_no_smooth, 'name') else 'found'}")
print(f"  Non-NaN values: {vel_no_smooth.notna().sum()}")
print(f"  Velocity column available in MA-9: {vel_ma9.notna().sum() > 0}")
print(f"  Velocity column available in Median-9: {vel_med9.notna().sum() > 0}")
print()

if vel_no_smooth.notna().sum() > 0:
    gt_series = get_gt(no_smooth_df)

    # Identify saccade samples in GT
    gt_sac_mask = gt_series == "Saccade"
    n_gt_sac_samples = int(gt_sac_mask.sum())
    print(f"  GT saccade samples: {n_gt_sac_samples}")

    # Find saccade events (contiguous runs of GT saccade samples)
    saccade_events = []
    in_sac = False
    start_idx = 0
    for i, val in enumerate(gt_sac_mask.values):
        if val and not in_sac:
            in_sac = True
            start_idx = i
        elif not val and in_sac:
            in_sac = False
            saccade_events.append((start_idx, i - 1))
    if in_sac:
        saccade_events.append((start_idx, len(gt_sac_mask) - 1))

    n_sac_events = len(saccade_events)
    print(f"  GT saccade events (contiguous): {n_sac_events}")

    # For each saccade event, compute peak velocities under each smoothing
    sac_analysis = []
    THRESHOLD = 30.0

    vel_ns_vals = vel_no_smooth.values
    vel_ma9_vals = vel_ma9.values if vel_ma9.notna().sum() > 0 else None
    vel_med9_vals = vel_med9.values if vel_med9.notna().sum() > 0 else None

    n_missed_ma9 = 0
    n_missed_med9 = 0
    n_narrow_missed_ma9 = 0
    n_narrow_missed_med9 = 0
    n_wide_missed_ma9 = 0
    n_wide_missed_med9 = 0
    NARROW_THRESHOLD = 5  # samples

    for start, end in saccade_events:
        width = end - start + 1
        peak_ns = float(np.nanmax(vel_ns_vals[start:end+1])) if len(vel_ns_vals[start:end+1]) > 0 else float("nan")
        peak_ma9 = float(np.nanmax(vel_ma9_vals[start:end+1])) if vel_ma9_vals is not None and len(vel_ma9_vals[start:end+1]) > 0 else float("nan")
        peak_med9 = float(np.nanmax(vel_med9_vals[start:end+1])) if vel_med9_vals is not None and len(vel_med9_vals[start:end+1]) > 0 else float("nan")

        # "Missed" = peak velocity drops below threshold under smoothing
        missed_ma9 = (not math.isnan(peak_ma9)) and (peak_ma9 < THRESHOLD)
        missed_med9 = (not math.isnan(peak_med9)) and (peak_med9 < THRESHOLD)

        if missed_ma9:
            n_missed_ma9 += 1
            if width < NARROW_THRESHOLD:
                n_narrow_missed_ma9 += 1
            else:
                n_wide_missed_ma9 += 1
        if missed_med9:
            n_missed_med9 += 1
            if width < NARROW_THRESHOLD:
                n_narrow_missed_med9 += 1
            else:
                n_wide_missed_med9 += 1

        sac_analysis.append({
            "start": start,
            "end": end,
            "width_samples": width,
            "peak_no_smooth": peak_ns,
            "peak_ma9": peak_ma9,
            "peak_med9": peak_med9,
            "missed_ma9": missed_ma9,
            "missed_med9": missed_med9,
            "narrow": width < NARROW_THRESHOLD,
        })

    sac_df = pd.DataFrame(sac_analysis)
    sac_df.to_csv(OUTPUTS / "saccade_peak_analysis.csv", index=False)

    print(f"\nSaccade peak analysis ({n_sac_events} events):")
    print(f"  Saccades missed by MA-9  (peak < {THRESHOLD}°/s): {n_missed_ma9} ({100.*n_missed_ma9/n_sac_events:.1f}%)")
    print(f"  Saccades missed by Med-9 (peak < {THRESHOLD}°/s): {n_missed_med9} ({100.*n_missed_med9/n_sac_events:.1f}%)")
    print()

    # Narrow vs wide saccades
    n_narrow = sum(1 for r in sac_analysis if r["narrow"])
    n_wide = sum(1 for r in sac_analysis if not r["narrow"])
    print(f"  Narrow saccades (<{NARROW_THRESHOLD} samples): {n_narrow}")
    print(f"  Wide saccades   (≥{NARROW_THRESHOLD} samples): {n_wide}")
    print()

    if n_narrow > 0:
        print(f"  MA-9  miss rate for narrow saccades: {n_narrow_missed_ma9}/{n_narrow} "
              f"({100.*n_narrow_missed_ma9/n_narrow:.1f}%)")
    if n_wide > 0:
        print(f"  MA-9  miss rate for wide saccades  : {n_wide_missed_ma9}/{n_wide} "
              f"({100.*n_wide_missed_ma9/n_wide:.1f}%)")
    if n_narrow > 0:
        print(f"  Med-9 miss rate for narrow saccades: {n_narrow_missed_med9}/{n_narrow} "
              f"({100.*n_narrow_missed_med9/n_narrow:.1f}%)")
    if n_wide > 0:
        print(f"  Med-9 miss rate for wide saccades  : {n_wide_missed_med9}/{n_wide} "
              f"({100.*n_wide_missed_med9/n_wide:.1f}%)")
    print()

    # Peak attenuation: average peak reduction
    if vel_ma9_vals is not None:
        valid_rows = sac_df[sac_df["peak_no_smooth"].notna() & sac_df["peak_ma9"].notna() &
                            sac_df["peak_no_smooth"] > 0]
        if len(valid_rows) > 0:
            attenuation_ma9 = (valid_rows["peak_no_smooth"] - valid_rows["peak_ma9"]) / valid_rows["peak_no_smooth"]
            attenuation_med9 = (valid_rows["peak_no_smooth"] - valid_rows["peak_med9"]) / valid_rows["peak_no_smooth"]
            print(f"  Average peak attenuation MA-9  : {100.*attenuation_ma9.mean():.1f}%")
            print(f"  Average peak attenuation Med-9 : {100.*attenuation_med9.mean():.1f}%")
            print()

else:
    print("  WARNING: velocity column not found in output DataFrames")
    print(f"  Available columns: {list(no_smooth_df.columns[:20])}")
    print()


# ---------------------------------------------------------------------------
# STEP 3.4: Kernel asymmetry test (asymmetric MA)
# ---------------------------------------------------------------------------

print("-" * 60)
print("STEP 3.4: KERNEL ASYMMETRY TEST")
print("-" * 60)

# Test whether using a smaller effective window for MA gives better results
# The hypothesis: Tobii may use a non-symmetric or shorter kernel
# We test MA variants: MA-3, MA-5, MA-7 on C2 to see decay pattern

ma_extended_configs = {
    "ma_3":  _make_std_config("average", smoothing_mode="moving_average", smoothing_window=3),
    "ma_5":  _make_std_config("average", smoothing_mode="moving_average", smoothing_window=5),
    "ma_7":  _make_std_config("average", smoothing_mode="moving_average", smoothing_window=7),
    "ma_9":  _make_std_config("average", smoothing_mode="moving_average", smoothing_window=9),
}

print(f"{'MA window':<12} {'κ':>8} {'Sac Recall':>12}")
print("-" * 35)
for name, config in ma_extended_configs.items():
    # Already computed ma_3, ma_5, ma_9 above; run ma_7 fresh
    if name == "ma_7":
        df7 = _run(c2_path, config)
        m7 = _metrics(df7, _pred_col(config))
        ma_results["ma_7"] = {"df": df7, "metrics": m7}
        m = m7
    else:
        m = ma_results[name]["metrics"]
    print(f"{name:<12} {m['cohen_kappa']:>8.4f} {m['saccade_recall']:>12.2f}%")
print()


# ===========================================================================
# SAVE SUMMARY
# ===========================================================================

print("=" * 70)
print("SAVING SUMMARY")
print("=" * 70)

summary = {
    "phase1_baseline": {
        "C2_kappa": float(c2_metrics["cohen_kappa"]),
        "C2_agreement": float(c2_metrics["percentage_agreement"]),
        "C2_n_samples": int(c2_metrics["n_samples_total"]),
    },
    "phase2_c5_anomaly": {
        "C3": {
            "kappa": float(c3_metrics["cohen_kappa"]),
            "agreement": float(c3_metrics["percentage_agreement"]),
            "n_samples": int(c3_metrics["n_samples_total"]),
            "fix_recall": float(c3_metrics["fixation_recall"]),
            "sac_recall": float(c3_metrics["saccade_recall"]),
            "n_disagree": len(c3_disagree),
            "validity": c3_validity,
        },
        "C5": {
            "kappa": float(c5_metrics["cohen_kappa"]),
            "agreement": float(c5_metrics["percentage_agreement"]),
            "n_samples": int(c5_metrics["n_samples_total"]),
            "fix_recall": float(c5_metrics["fixation_recall"]),
            "sac_recall": float(c5_metrics["saccade_recall"]),
            "n_disagree": len(c5_disagree),
            "validity": c5_validity,
        },
        "C6": {
            "kappa": float(c6_metrics["cohen_kappa"]),
            "agreement": float(c6_metrics["percentage_agreement"]),
            "n_samples": int(c6_metrics["n_samples_total"]),
            "fix_recall": float(c6_metrics["fixation_recall"]),
            "sac_recall": float(c6_metrics["saccade_recall"]),
        },
    },
    "phase3_ma_anomaly": {
        name: {
            "kappa": float(res["metrics"]["cohen_kappa"]),
            "sac_recall": float(res["metrics"]["saccade_recall"]),
            "fix_recall": float(res["metrics"]["fixation_recall"]),
        }
        for name, res in ma_results.items()
    },
}

with open(OUTPUTS / "analysis_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"Summary saved to {OUTPUTS / 'analysis_summary.json'}")
print(f"Disagreements saved to {OUTPUTS / 'disagreements_C5.csv'}")
print(f"Disagreements saved to {OUTPUTS / 'disagreements_C3.csv'}")
print(f"Saccade analysis saved to {OUTPUTS / 'saccade_peak_analysis.csv'}")
print()
print("Analysis complete.")
