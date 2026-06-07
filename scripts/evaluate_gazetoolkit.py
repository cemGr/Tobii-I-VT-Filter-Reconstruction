#!/usr/bin/env python3
"""
Evaluate GazeToolkit (Konopka 2019) against Tobii Pro Lab ground truth.

Two variants are evaluated:
  1. no-gap-fill  — uses the pre-computed _output.tsv from the GazeToolkit
                    C# pipeline; ground truth from _input.csv.
  2. gap-fill 75ms — runs GazeToolkit's ivt_filter.py (Python) on the Tobii
                     gap-filled export.  GazeToolkit's Python script does not
                     implement gap fill itself; the comparison is therefore
                     "GazeToolkit on Tobii-gap-filled gaze data" vs "Tobii Pro
                     Lab classification with gap fill 75 ms" — see summary note.

Outputs (all in results/gazetoolkit/):
  gazetoolkit_eval_results.json
  gazetoolkit_raw_output_no_gap_fill.tsv
  gazetoolkit_raw_output_gap_fill.tsv
  gazetoolkit_summary.txt
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path as _Path

# Ensure the repo root is on sys.path so the ivt_filter package is importable
# when the script is run from scripts/ or any other working directory.
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
GAZETOOLKIT_DIR = Path("/home/cem/Documents/Gitprojekt/GazeToolkit")
RESULTS_DIR = REPO_ROOT / "results" / "gazetoolkit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NO_GAP_FILL_INPUT = GAZETOOLKIT_DIR / "I-VT-botheye20ms30threshold_input.csv"
NO_GAP_FILL_OUTPUT = GAZETOOLKIT_DIR / "I-VT-botheye20ms30threshold_output.tsv"
GAP_FILL_TOBII_TSV = (
    GAZETOOLKIT_DIR
    / "dataset"
    / "IVT-Interpolation75ms-Eyeboth-NoNoise-VelocityWindow20-VelocityTreshold30.tsv"
)

TOBII_COL_MAP: Dict[str, str] = {
    "Recording timestamp [ms]": "time_ms",
    "Gaze point left X [DACS mm]": "gaze_left_x_mm",
    "Gaze point left Y [DACS mm]": "gaze_left_y_mm",
    "Gaze point right X [DACS mm]": "gaze_right_x_mm",
    "Gaze point right Y [DACS mm]": "gaze_right_y_mm",
    "Gaze point left X [DACS px]": "gaze_left_x_px",
    "Gaze point left Y [DACS px]": "gaze_left_y_px",
    "Gaze point right X [DACS px]": "gaze_right_x_px",
    "Gaze point right Y [DACS px]": "gaze_right_y_px",
    "Validity left": "validity_left",
    "Validity right": "validity_right",
    "Eye position left X [DACS mm]": "eye_left_x_mm",
    "Eye position left Y [DACS mm]": "eye_left_y_mm",
    "Eye position left Z [DACS mm]": "eye_left_z_mm",
    "Eye position right X [DACS mm]": "eye_right_x_mm",
    "Eye position right Y [DACS mm]": "eye_right_y_mm",
    "Eye position right Z [DACS mm]": "eye_right_z_mm",
    "Eye movement type": "gt_event_type",
    "Eye movement type index": "gt_event_index",
}

# GazeToolkit movement type → evaluation framework label
LABEL_MAP: Dict[str, str] = {
    "Fixation": "Fixation",
    "Saccade": "Saccade",
    "Unknown": "Unclassified",
}


# ── loading ───────────────────────────────────────────────────────────────────

def load_gazetoolkit_input(path: Path) -> pd.DataFrame:
    """Load the slim GazeToolkit input CSV (auto-detect separator)."""
    df = pd.read_csv(path, sep=None, engine="python")
    if "time_ms" not in df.columns:
        raise ValueError(f"time_ms not found in {path.name}; columns: {list(df.columns)[:8]}")
    if "gt_event_type" not in df.columns:
        raise ValueError(f"gt_event_type not found in {path.name}")
    return df.sort_values("time_ms").reset_index(drop=True)


def _normalise_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename start_time_ms/end_time_ms → start_time/end_time if needed."""
    renames: Dict[str, str] = {}
    if "start_time_ms" in df.columns and "start_time" not in df.columns:
        renames["start_time_ms"] = "start_time"
    if "end_time_ms" in df.columns and "end_time" not in df.columns:
        renames["end_time_ms"] = "end_time"
    if renames:
        df = df.rename(columns=renames)
    required = {"movement_type", "start_time", "end_time"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Event output missing columns {missing}; got {list(df.columns)}")
    return df


def load_gazetoolkit_events(path: Path) -> pd.DataFrame:
    """Load GazeToolkit event-level output (auto-detect separator)."""
    df = pd.read_csv(path, sep=None, engine="python")
    df = _normalise_event_columns(df)
    return df.sort_values("start_time").reset_index(drop=True)


_TIMESTAMP_FALLBACKS = [
    ("Recording timestamp [ms]", 1.0),
    ("Recording timestamp [μs]", 1e-3),
    ("Eyetracker timestamp [μs]", 1e-3),
]


def extract_gap_fill_input(tobii_tsv: Path) -> pd.DataFrame:
    """Extract the slim input format from a 92-column Tobii Pro Lab export."""
    print(f"  Loading Tobii TSV: {tobii_tsv.name} ...")
    df_raw = pd.read_csv(tobii_tsv, sep="\t", decimal=",", low_memory=False)

    # Resolve timestamp column — prefer ms, fall back to µs with unit conversion
    ts_col: Optional[str] = None
    ts_scale: float = 1.0
    for col, scale in _TIMESTAMP_FALLBACKS:
        if col in df_raw.columns:
            ts_col = col
            ts_scale = scale
            break
    if ts_col is None:
        raise ValueError(
            f"No known timestamp column found. Available: {list(df_raw.columns)[:10]}"
        )
    print(f"  Using timestamp column '{ts_col}' (scale ×{ts_scale})")

    available = {k: v for k, v in TOBII_COL_MAP.items()
                 if k in df_raw.columns and k not in ("Recording timestamp [ms]",)}
    # Always include the resolved timestamp column
    available[ts_col] = "time_ms"

    missing_cols = (
        set(TOBII_COL_MAP.keys())
        - set(df_raw.columns)
        - {"Recording timestamp [ms]"}
    )
    if missing_cols:
        print(f"  Warning: {len(missing_cols)} Tobii columns not found "
              f"(first 5: {sorted(missing_cols)[:5]})")
    if "Eye movement type" not in available:
        raise ValueError("'Eye movement type' not found in Tobii export.")

    df_slim = df_raw[list(available.keys())].rename(columns=available)
    if ts_scale != 1.0:
        df_slim["time_ms"] = df_slim["time_ms"] * ts_scale
    return df_slim.sort_values("time_ms").reset_index(drop=True)


# ── GazeToolkit Python implementation ─────────────────────────────────────────

def _load_gazetoolkit_module():
    """Import GazeToolkit's ivt_filter.py under a non-conflicting name."""
    spec = importlib.util.spec_from_file_location(
        "gazetoolkit_ivt_filter",
        str(GAZETOOLKIT_DIR / "ivt_filter.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_gazetoolkit_python(input_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    """
    Run GazeToolkit's ivt_filter.py on *input_df*; save raw TSV to *output_path*.

    NOTE: ivt_filter.py's velocity calculation is O(n²) in pandas; for ~28 k
    samples expect 2–5 minutes of runtime.
    """
    gtmod = _load_gazetoolkit_module()

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".tsv")
    tmp_path = Path(tmp_name)
    try:
        os.close(tmp_fd)
        # Write with comma decimals so ivt_filter.py's decimal=',' parser works
        input_df.to_csv(tmp_path, sep="\t", index=False, decimal=",")
        print("  Running ivt_filter.py  (threshold=30 °/s, window=20 ms, eye=average)")
        print("  NOTE: O(n²) velocity loop — may take several minutes …")
        gtmod.run_ivt_filter(
            str(tmp_path),
            str(output_path),
            threshold=30.0,
            window=20.0,
            eye="average",
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return load_gazetoolkit_events(output_path)


# ── timestamp alignment ────────────────────────────────────────────────────────

def assign_labels(samples_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.Series:
    """
    For each sample in samples_df['time_ms'] assign the GazeToolkit label of
    the event where start_time ≤ time_ms ≤ end_time.  Samples not covered by
    any event receive 'Unclassified' (further refined to EyesNotFound in
    evaluate_variant based on validity columns).  Uses merge_asof (O(n log n)).
    """
    samples = samples_df[["time_ms"]].copy().sort_values("time_ms")
    events = (
        events_df[["start_time", "end_time", "movement_type"]]
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        samples,
        events,
        left_on="time_ms",
        right_on="start_time",
        direction="backward",
    )

    # Samples that precede every event start, or fall after the matched event's end
    not_covered = merged["end_time"].isna() | (merged["time_ms"] > merged["end_time"])
    merged.loc[not_covered, "movement_type"] = "Unknown"

    labels = merged["movement_type"].map(LABEL_MAP).fillna("Unclassified")
    labels.index = samples.index  # index from sorted samples (matches samples_df rows)
    return labels.reindex(samples_df.index)  # restore original row order


# ── metric serialisation ───────────────────────────────────────────────────────

def _json_safe(val: Any) -> Any:
    if isinstance(val, float) and math.isnan(val):
        return None
    if isinstance(val, (list, tuple)):
        return [_json_safe(v) for v in val]
    if isinstance(val, dict):
        return {k: _json_safe(v) for k, v in val.items()}
    return val


def _prf(tp: int, total_gt: int, total_pred: int) -> Dict[str, Optional[float]]:
    recall = tp / total_gt if total_gt > 0 else float("nan")
    precision = tp / total_pred if total_pred > 0 else float("nan")
    denom = precision + recall
    f1 = (2 * precision * recall / denom
          if (not math.isnan(precision) and not math.isnan(recall) and denom > 0)
          else float("nan"))
    return {
        "precision": _json_safe(round(precision * 100, 2)),
        "recall": _json_safe(round(recall * 100, 2)),
        "f1": _json_safe(round(f1 * 100, 2)),
    }


def serialise_sample_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Add per-class P/R/F1 and make compute_ivt_metrics output JSON-safe."""
    labels: List[str] = m["labels"]
    conf: List[List[int]] = m["confusion_matrix"]
    k = len(labels)

    per_class: Dict[str, Any] = {}
    for cls in ["Fixation", "Saccade"]:
        if cls not in labels:
            per_class[cls] = {"precision": None, "recall": None, "f1": None}
            continue
        idx = labels.index(cls)
        tp = conf[idx][idx]
        total_gt = sum(conf[idx])
        total_pred = sum(conf[i][idx] for i in range(k))
        per_class[cls] = _prf(tp, total_gt, total_pred)

    out = _json_safe({key: val for key, val in m.items() if key != "confusion_matrix"})
    out["confusion_matrix"] = _json_safe(conf)
    out["confusion_matrix_labels"] = labels
    out["per_class"] = per_class
    return out


def serialise_event_metrics(m: Dict[str, Any]) -> Dict[str, Any]:
    """Add per-class P/R/F1 and make compute_event_iou_metrics output JSON-safe."""
    # Drop non-serialisable MatchResult lists
    stripped = {k: v for k, v in m.items() if k not in ("matches", "unmatched_pred")}

    cm: Dict[str, Dict[str, int]] = stripped.get("confusion_matrix", {})
    event_counts: Dict[str, Dict[str, int]] = stripped.get("event_counts", {})

    per_class: Dict[str, Any] = {}
    for cls in ["Fixation", "Saccade"]:
        gt_row = cm.get(cls, {})
        tp = gt_row.get(cls, 0)
        fn = gt_row.get("FN", 0)
        wrong = sum(v for tag, v in gt_row.items() if tag not in (cls, "FN"))
        total_gt = tp + fn + wrong
        n_pred = event_counts.get(cls, {}).get("pred", 0)
        prf = _prf(tp, total_gt, n_pred)
        per_class[cls] = {"n_gt": total_gt, "n_pred": n_pred, "tp": tp, **prf}

    out = _json_safe(stripped)
    out["per_class"] = per_class
    return out


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_variant(
    samples_df: pd.DataFrame,
    events_df: pd.DataFrame,
    label: str,
) -> Dict[str, Any]:
    from ivt_filter.evaluation.evaluation import compute_ivt_metrics
    from ivt_filter.evaluation.event_iou import compute_event_iou_metrics

    print(f"\n[{label}] Assigning sample labels …")
    samples_df = samples_df.copy()
    # Fill empty GT cells (NaN in sparse Tobii exports) with Unclassified
    if "gt_event_type" in samples_df.columns:
        samples_df["gt_event_type"] = (
            samples_df["gt_event_type"].fillna("Unclassified").astype(str)
        )
    samples_df["gazetoolkit_sample_type"] = assign_labels(samples_df, events_df)

    # GazeToolkit has only Unknown (≈ no valid velocity), not separate EyesNotFound /
    # Unclassified.  Samples where BOTH eyes are invalid have no gaze data at all —
    # these correspond to Tobii's EyesNotFound, not Unclassified.  Refine the mapping:
    #   Both eyes Invalid  →  EyesNotFound
    #   At least one Valid →  Unclassified  (keep as is)
    if "validity_left" in samples_df.columns and "validity_right" in samples_df.columns:
        both_invalid = (
            (samples_df["validity_left"].astype(str) == "Invalid")
            & (samples_df["validity_right"].astype(str) == "Invalid")
        )
        predicted_uncl = samples_df["gazetoolkit_sample_type"] == "Unclassified"
        n_refined = int((both_invalid & predicted_uncl).sum())
        samples_df.loc[both_invalid & predicted_uncl, "gazetoolkit_sample_type"] = "EyesNotFound"
        if n_refined:
            print(f"  Refined {n_refined} Unknown→EyesNotFound "
                  f"(both eyes Invalid, not Unclassified)")
    vc = samples_df["gazetoolkit_sample_type"].value_counts()
    for lbl, cnt in vc.items():
        print(f"  {lbl}: {cnt} samples")

    print(f"[{label}] Computing sample-level metrics …")
    sample_m = compute_ivt_metrics(
        samples_df,
        gt_col="gt_event_type",
        pred_col="gazetoolkit_sample_type",
    )

    print(f"[{label}] Computing event-level IoU metrics …")
    event_m = compute_event_iou_metrics(
        samples_df,
        gt_col="gt_event_type",
        pred_col="gazetoolkit_sample_type",
        time_col="time_ms",
        event_types=["Fixation", "Saccade"],
    )

    return {
        "sample_level": serialise_sample_metrics(sample_m),
        "event_level": serialise_event_metrics(event_m),
    }


# ── summary text ───────────────────────────────────────────────────────────────

def _f(val: Optional[float], dec: int = 1) -> str:
    return "N/A" if val is None else f"{val:.{dec}f}"


def generate_summary(results: Dict[str, Any]) -> str:
    lines = [
        "GazeToolkit vs Tobii Pro Lab — Evaluation Summary",
        "=" * 55,
        "",
        "Settings: both-eye average, velocity threshold 30 °/s,",
        "          velocity window 20 ms.",
        "",
    ]

    variant_info = [
        ("no_gap_fill",
         "Variant 1: No Gap Fill  (C# pipeline output)"),
        ("gap_fill",
         "Variant 2: Gap Fill 75 ms  (Python ivt_filter.py)"),
    ]

    for key, title in variant_info:
        r = results.get(key, {})
        sl = r.get("sample_level", {})
        el = r.get("event_level", {})
        pc_sl = sl.get("per_class", {})
        pc_el = el.get("per_class", {})

        lines += [
            title,
            "-" * len(title),
            "",
            "Sample-level",
            f"  Agreement (GT Fix/Sac only):  {_f(sl.get('percentage_agreement'))} %",
            f"  Agreement (all samples):      {_f(sl.get('percentage_agreement_all'))} %",
            f"  Cohen's κ:                    {_f(sl.get('cohen_kappa'), 3)}",
            "",
            f"  {'Class':<12}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}",
            f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}",
        ]
        for cls in ["Fixation", "Saccade"]:
            pc = pc_sl.get(cls, {})
            lines.append(
                f"  {cls:<12}  {_f(pc.get('precision')):>10}  "
                f"{_f(pc.get('recall')):>10}  {_f(pc.get('f1')):>10}"
            )

        lines += [
            "",
            "Event-level  (max-IoU matching, min IoU > 0.0)",
            f"  {'Class':<12}  {'Precision':>10}  {'Recall':>10}  "
            f"{'F1':>10}  GT events  Pred events",
            f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*9}  {'-'*11}",
        ]
        for cls in ["Fixation", "Saccade"]:
            pc = pc_el.get(cls, {})
            lines.append(
                f"  {cls:<12}  {_f(pc.get('precision')):>10}  "
                f"{_f(pc.get('recall')):>10}  {_f(pc.get('f1')):>10}  "
                f"{str(pc.get('n_gt', '?')):>9}  {str(pc.get('n_pred', '?')):>11}"
            )
        lines.append("")

    lines += [
        "Note on gap-fill variant:",
        "  GazeToolkit's Python implementation (ivt_filter.py) does not",
        "  implement gap fill.  The gap-fill dataset (Tobii Pro Lab export",
        "  with interpolation 75 ms enabled) was fed directly to ivt_filter.py",
        "  without further interpolation.  The comparison is therefore",
        "  'GazeToolkit on Tobii-gap-filled gaze data' vs 'Tobii Pro Lab",
        "  classification with gap fill 75 ms', not 'GazeToolkit with gap fill'.",
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    results: Dict[str, Any] = {}

    # ── Variant 1: no-gap-fill ─────────────────────────────────────────────
    print("=" * 60)
    print("Variant 1: No-gap-fill  (pre-computed output)")
    print("=" * 60)

    samples_no_gf = load_gazetoolkit_input(NO_GAP_FILL_INPUT)
    print(f"Input: {len(samples_no_gf)} samples, "
          f"{samples_no_gf['time_ms'].min():.0f}–{samples_no_gf['time_ms'].max():.0f} ms")

    events_no_gf = load_gazetoolkit_events(NO_GAP_FILL_OUTPUT)
    print(f"Events: {len(events_no_gf)} rows, types: "
          f"{events_no_gf['movement_type'].value_counts().to_dict()}")

    raw_no_gf = RESULTS_DIR / "gazetoolkit_raw_output_no_gap_fill.tsv"
    shutil.copy(NO_GAP_FILL_OUTPUT, raw_no_gf)
    print(f"Raw output saved → {raw_no_gf.name}")

    results["no_gap_fill"] = evaluate_variant(samples_no_gf, events_no_gf, "no-gap-fill")

    # ── Variant 2: gap-fill ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Variant 2: Gap-fill 75 ms  (run ivt_filter.py)")
    print("=" * 60)

    samples_gf = extract_gap_fill_input(GAP_FILL_TOBII_TSV)
    print(f"Input: {len(samples_gf)} samples, "
          f"{samples_gf['time_ms'].min():.0f}–{samples_gf['time_ms'].max():.0f} ms")

    raw_gf = RESULTS_DIR / "gazetoolkit_raw_output_gap_fill.tsv"
    events_gf = run_gazetoolkit_python(samples_gf, raw_gf)
    print(f"Events: {len(events_gf)} rows, types: "
          f"{events_gf['movement_type'].value_counts().to_dict()}")
    print(f"Raw output saved → {raw_gf.name}")

    results["gap_fill"] = evaluate_variant(samples_gf, events_gf, "gap-fill")

    # ── Save JSON results ──────────────────────────────────────────────────
    json_out = RESULTS_DIR / "gazetoolkit_eval_results.json"
    with open(json_out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nResults → {json_out}")

    # ── Save summary ───────────────────────────────────────────────────────
    summary = generate_summary(results)
    summary_out = RESULTS_DIR / "gazetoolkit_summary.txt"
    with open(summary_out, "w", encoding="utf-8") as fh:
        fh.write(summary)
    print(f"Summary → {summary_out}\n")
    print(summary)


if __name__ == "__main__":
    main()
