#!/usr/bin/env python3
"""Tier-2 Sweep Benchmark: systematic one-at-a-time parameter sweep on a single input file.

For each filter dimension (eye mode, velocity method, smoothing, window policy, etc.),
all variants are evaluated against the same input file while every other parameter
is held at the Tobii IVT Fixation preset baseline.

Outputs:
  sweep_results.json   – full per-variant data
  sweep_metrics.csv    – flat table (one row per variant)
  sweep_summary.txt    – human-readable comparison with deltas vs baseline

Usage:
    python scripts/sweep_benchmark.py [options]

Options:
    --input-both PATH    Botheye input TSV (default: I-VT-botheye20ms30threshold_input.tsv)
    --input-left PATH    Left-eye input TSV (default: LeftV30W20_input.tsv)
    --results-dir PATH   Output directory (default: results/sweep_benchmark/)
    --dimensions NAME…   Run only specific dimensions
    --fail-fast          Stop at first error instead of recording it
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ivt_filter.config import (  # noqa: E402
    FixationPostConfig,
    IVTClassifierConfig,
    OlsenVelocityConfig,
    PipelineConfig,
)
from ivt_filter.evaluation.evaluation import compute_ivt_metrics  # noqa: E402
from ivt_filter.evaluation.event_iou import compute_event_iou_metrics  # noqa: E402
from ivt_filter.io.pipeline import IVTPipeline  # noqa: E402


DEFAULT_BOTH_INPUT_FILE = (
    REPO_ROOT / "test_data" / "inputs" / "I-VT-botheye20ms30threshold_input.tsv"
)
DEFAULT_LEFT_INPUT_FILE = (
    REPO_ROOT / "test_data" / "inputs" / "LeftV30W20_input.tsv"
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "sweep_benchmark"

# ---------------------------------------------------------------------------
# Per-dimension input file routing
# "both" → DEFAULT_BOTH_INPUT_FILE (botheye),  "left" → DEFAULT_LEFT_INPUT_FILE
# ---------------------------------------------------------------------------
DIMENSION_FILE: dict[str, str] = {
    "eye_mode":               "both",
    "velocity_method":        "both",
    "velocity_threshold":     "both",
    "window_length_ms":       "left",
    "smoothing":              "left",
    "gap_fill":               "both",
    "window_policy":          "left",
    "postprocessing":         "left",
    "tobii_eye_offset_interp": "both",
    "coordinate_rounding":     "both",
}

# ---------------------------------------------------------------------------
# Baseline: Tobii IVT Fixation preset
# ---------------------------------------------------------------------------
BASELINE: dict[str, Any] = {
    # Eye selection
    "eye_mode": "average",
    # Velocity
    "velocity_method": "tobii_gaze_dir",
    "velocity_threshold_deg_per_sec": 30.0,
    "window_length_ms": 20.0,
    # Smoothing
    "smoothing_mode": "median",
    "smoothing_window_samples": 3,
    # Gap fill
    "gap_fill_enabled": False,
    "gap_fill_max_gap_ms": 75.0,
    # Postprocessing
    "merge_adjacent_fixations": True,
    "merge_max_time_gap_ms": 75.0,
    "merge_max_angle_deg": 0.5,
    "discard_short_fixations": True,
    "min_fixation_duration_ms": 60.0,
    # Window policy flags (all off = time-symmetric default)
    "auto_fixed_window_from_ms": False,
    "shifted_valid_window": False,
    "asymmetric_neighbor_window": False,
    # Tobii-specific
    "tobii_eye_offset_interpolation": False,
    # Coordinate rounding
    "coordinate_rounding": "none",
}

# ---------------------------------------------------------------------------
# Sweep dimensions: (dimension_name, [(label, override_dict), ...])
# An empty override dict {} means "identical to baseline for this dimension".
# ---------------------------------------------------------------------------
SWEEP_DIMENSIONS: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = [
    ("eye_mode", [
        ("left",                {"eye_mode": "left"}),
        ("average",             {}),                                      # baseline
        ("right",               {"eye_mode": "right"}),
    ]),
    ("velocity_method", [
        ("olsen2d",             {"velocity_method": "olsen2d"}),
        ("tobii_gaze_dir",      {}),                                      # baseline
        ("ray3d_gaze_dir",      {"velocity_method": "ray3d_gaze_dir"}),
    ]),
    ("velocity_threshold", [
        ("30_deg_s",            {}),                                      # baseline
        ("100_deg_s",           {"velocity_threshold_deg_per_sec": 100.0}),
    ]),
    ("window_length_ms", [
        ("1ms",                 {"window_length_ms": 1.0}),
        ("20ms",                {}),                                      # baseline
    ]),
    ("smoothing", [
        ("none",                {"smoothing_mode": "none"}),
        ("median_3",            {}),                                      # baseline
        ("median_9",            {"smoothing_mode": "median",
                                 "smoothing_window_samples": 9}),
        ("moving_avg_3",        {"smoothing_mode": "moving_average",
                                 "smoothing_window_samples": 3}),
        ("moving_avg_9",        {"smoothing_mode": "moving_average",
                                 "smoothing_window_samples": 9}),
    ]),
    ("gap_fill", [
        ("disabled",            {}),                                      # baseline
        ("75ms",                {"gap_fill_enabled": True,
                                 "gap_fill_max_gap_ms": 75.0}),
    ]),
    ("window_policy", [
        ("time_symmetric",      {}),                                      # baseline
        ("auto_fixed_from_ms",  {"auto_fixed_window_from_ms": True}),
        ("shifted_valid",       {"auto_fixed_window_from_ms": True,
                                 "shifted_valid_window": True}),
        ("asymmetric_neighbor", {"asymmetric_neighbor_window": True}),
    ]),
    ("postprocessing", [
        ("none",                {"merge_adjacent_fixations": False,
                                 "discard_short_fixations": False}),
        ("merge_only",          {"discard_short_fixations": False}),
        ("discard_only",        {"merge_adjacent_fixations": False}),
        ("full_preset",         {}),                                      # baseline
    ]),
    ("tobii_eye_offset_interp", [
        ("disabled",            {}),                                      # baseline
        ("enabled",             {"tobii_eye_offset_interpolation": True}),
    ]),
    ("coordinate_rounding", [
        ("none",                {}),                                      # baseline
        ("nearest",             {"coordinate_rounding": "nearest"}),
        ("halfup",              {"coordinate_rounding": "halfup"}),
        ("floor",               {"coordinate_rounding": "floor"}),
        ("ceil",                {"coordinate_rounding": "ceil"}),
    ]),
]


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------

def _probe_time_settings(path: Path) -> tuple[str, str]:
    """Detect whether the file uses time_ms or time_us."""
    try:
        probe = pd.read_csv(
            path, sep="\t", decimal=",", nrows=25,
            usecols=lambda c: c in {"time_ms", "time_us"},
        )
    except Exception:
        return "time_ms", "ms"
    for col, unit in (("time_ms", "ms"), ("time_us", "us")):
        if col in probe.columns:
            vals = pd.to_numeric(probe[col], errors="coerce")
            if vals.notna().any():
                return col, unit
    return "time_ms", "ms"


def _build_pipeline_config(
    params: dict[str, Any], time_column: str, time_unit: str
) -> PipelineConfig:
    velocity = OlsenVelocityConfig(
        window_length_ms=params["window_length_ms"],
        velocity_method=params["velocity_method"],
        time_column=time_column,
        time_unit=time_unit,
        eye_mode=params["eye_mode"],
        smoothing_mode=params["smoothing_mode"],
        smoothing_window_samples=params["smoothing_window_samples"],
        gap_fill_enabled=params["gap_fill_enabled"],
        gap_fill_max_gap_ms=params["gap_fill_max_gap_ms"],
        auto_fixed_window_from_ms=params["auto_fixed_window_from_ms"],
        shifted_valid_window=params["shifted_valid_window"],
        asymmetric_neighbor_window=params["asymmetric_neighbor_window"],
        tobii_eye_offset_interpolation=params["tobii_eye_offset_interpolation"],
        coordinate_rounding=params.get("coordinate_rounding", "none"),
    )
    classifier = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=params["velocity_threshold_deg_per_sec"]
    )

    merge = params["merge_adjacent_fixations"]
    discard = params["discard_short_fixations"]
    if merge or discard:
        fp = FixationPostConfig()
        if merge:
            fp.merge_adjacent_fixations = True
            fp.max_time_gap_ms = params["merge_max_time_gap_ms"]
            fp.max_angle_deg = params["merge_max_angle_deg"]
        if discard:
            fp.discard_short_fixations = True
            fp.min_fixation_duration_ms = params["min_fixation_duration_ms"]
        fp.__post_init__()
        fixation_post: FixationPostConfig | None = fp
    else:
        fixation_post = None

    return PipelineConfig(
        velocity=velocity,
        classifier=classifier,
        classify=True,
        fixation_post=fixation_post,
    )


def _prediction_column(config: PipelineConfig) -> str:
    if config.fixation_post:
        return "ivt_event_type_post"
    return "ivt_sample_type"


# ---------------------------------------------------------------------------
# Running a single variant
# ---------------------------------------------------------------------------

def _run_variant(
    path: Path,
    params: dict[str, Any],
    time_column: str,
    time_unit: str,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    config = _build_pipeline_config(params, time_column, time_unit)
    df = IVTPipeline(config).run(str(path), evaluate=False, plot=False)
    pred_col = _prediction_column(config)
    metrics = compute_ivt_metrics(df, pred_col=pred_col, exclude_calibration=False)
    event = _compute_event_metrics(df, pred_col)
    return {
        "status": "ok",
        "duration_s": round(time.perf_counter() - t0, 3),
        "metrics": metrics,
        "event_iou": event,
    }


def _compute_event_metrics(df: Any, pred_col: str) -> dict[str, Any] | None:
    gt_col = "gt_sample_type" if "gt_sample_type" in df.columns else "gt_event_type"
    if gt_col not in df.columns or pred_col not in df.columns or "time_ms" not in df.columns:
        return None
    m = compute_event_iou_metrics(
        df, gt_col=gt_col, pred_col=pred_col, time_col="time_ms"
    )
    matches = m.get("matches", [])
    matched_ious = [x.iou for x in matches if x.pred_event is not None]
    correct = sum(1 for x in matches if x.is_correct_type)
    n_gt = m["n_gt_events"]
    return {
        "n_gt_events": n_gt,
        "n_pred_events": m["n_pred_events"],
        "n_matched": m["n_matched"],
        "n_fn": m["n_fn"],
        "n_fp": m["n_fp"],
        "mean_iou": statistics.fmean(matched_ious) if matched_ious else float("nan"),
        "correct_type_rate": correct / n_gt if n_gt else float("nan"),
    }


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    input_both: Path,
    input_left: Path,
    dimensions: list[tuple[str, list[tuple[str, dict[str, Any]]]]],
    *,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    tc_both, tu_both = _probe_time_settings(input_both)
    tc_left, tu_left = _probe_time_settings(input_left)

    print(f"[sweep] input-both:  {input_both.name}  ({tc_both}, {tu_both})", flush=True)
    print(f"[sweep] input-left:  {input_left.name}  ({tc_left}, {tu_left})", flush=True)
    print(f"[sweep] dimensions:  {[d for d, _ in dimensions]}", flush=True)

    # Baseline runs on the botheye file
    print("\n[sweep] baseline (Tobii IVT Fixation preset) ...", end=" ", flush=True)
    try:
        baseline = _run_variant(input_both, BASELINE, tc_both, tu_both)
        _print_metrics(baseline)
    except Exception as exc:
        baseline = {"status": "error", "error": str(exc)}
        print(f"ERROR: {exc}", flush=True)

    sweep: dict[str, list[dict[str, Any]]] = {}

    for dimension, variants in dimensions:
        # Pick the appropriate input file for this dimension
        file_key = DIMENSION_FILE.get(dimension, "both")
        if file_key == "left":
            dim_path, dim_tc, dim_tu = input_left, tc_left, tu_left
        else:
            dim_path, dim_tc, dim_tu = input_both, tc_both, tu_both

        print(f"\n[sweep] [{dimension}] using {dim_path.name}", flush=True)
        dim_results: list[dict[str, Any]] = []
        for label, overrides in variants:
            params = {**BASELINE, **overrides}
            is_base = not overrides
            marker = " (baseline)" if is_base else ""
            print(f"  {label}{marker} ...", end=" ", flush=True)
            try:
                result = _run_variant(dim_path, params, dim_tc, dim_tu)
                _print_metrics(result)
            except Exception as exc:
                result = {"status": "error", "error": str(exc)}
                print(f"ERROR: {exc}", flush=True)
                if not continue_on_error:
                    raise
            dim_results.append({
                "label": label,
                "is_baseline_variant": is_base,
                "overrides": overrides,
                "input_file": dim_path.name,
                **result,
            })
        sweep[dimension] = dim_results

    return {
        "input_both": input_both.name,
        "input_left": input_left.name,
        "baseline_params": BASELINE,
        "baseline": baseline,
        "sweep": sweep,
    }


def _print_metrics(result: dict) -> None:
    if result.get("status") != "ok":
        return
    agr = _get_metric(result, "percentage_agreement")
    kap = _get_metric(result, "cohen_kappa")
    print(f"agreement={_fmt(agr)}%  κ={_fmt(kap)}", flush=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _get_metric(result: dict, key: str) -> float | None:
    v = result.get("metrics", {}).get(key)
    if v is None:
        v = (result.get("event_iou") or {}).get(key)
    return v


def _fmt(v: Any, decimals: int = 4) -> str:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return f"{float(v):.{decimals}f}"
    return "n/a"


def _delta_str(v: Any, base: Any) -> str:
    if (
        isinstance(v, (int, float)) and math.isfinite(float(v))
        and isinstance(base, (int, float)) and math.isfinite(float(base))
    ):
        d = float(v) - float(base)
        return f"{'+' if d >= 0 else ''}{d:.4f}"
    return "—"


def format_text_report(data: dict[str, Any]) -> str:
    # Support both old (single input_file) and new (input_both/input_left) formats
    if "input_file" in data:
        input_header = f"Input: {data['input_file']}"
    else:
        input_header = (
            f"Input (botheye): {data.get('input_both', '—')}\n"
            f"Input (left):    {data.get('input_left', '—')}"
        )
    lines: list[str] = ["I-VT Sweep Benchmark", "=" * 72, input_header, ""]

    b = data["baseline"]
    lines.append("Baseline (Tobii IVT Fixation Preset):")
    if b.get("status") == "ok":
        lines.append(f"  Agreement Fix/Sac : {_fmt(_get_metric(b, 'percentage_agreement'))} %")
        lines.append(f"  Agreement all     : {_fmt(_get_metric(b, 'percentage_agreement_all'))} %")
        lines.append(f"  Cohen's kappa     : {_fmt(_get_metric(b, 'cohen_kappa'))}")
        lines.append(f"  Fixation recall   : {_fmt(_get_metric(b, 'fixation_recall'))} %")
        lines.append(f"  Saccade recall    : {_fmt(_get_metric(b, 'saccade_recall'))} %")
        lines.append(f"  Event mean IoU    : {_fmt((_get_metric(b, 'mean_iou')))}")
    else:
        lines.append(f"  ERROR: {b.get('error')}")
    lines.append("")

    b_agr = _get_metric(b, "percentage_agreement") if b.get("status") == "ok" else None
    b_kap = _get_metric(b, "cohen_kappa") if b.get("status") == "ok" else None
    b_iou = (_get_metric(b, "mean_iou")) if b.get("status") == "ok" else None

    for dimension, variants in data["sweep"].items():
        dim_file = variants[0].get("input_file", "?") if variants else "?"
        lines.append(f"[{dimension}]  ({dim_file})")
        col_w = max(len(v["label"]) for v in variants) + 3
        for v in variants:
            label = v["label"] + (" *" if v.get("is_baseline_variant") else "")
            if v.get("status") != "ok":
                lines.append(f"  {label:<{col_w}}  ERROR: {v.get('error', '?')}")
                continue
            agr = _get_metric(v, "percentage_agreement")
            kap = _get_metric(v, "cohen_kappa")
            iou = _get_metric(v, "mean_iou")
            lines.append(
                f"  {label:<{col_w}}"
                f"  agr={_fmt(agr)}% ({_delta_str(agr, b_agr)})"
                f"  κ={_fmt(kap)} ({_delta_str(kap, b_kap)})"
                f"  IoU={_fmt(iou)} ({_delta_str(iou, b_iou)})"
            )
        lines.append("")

    lines.append("* = baseline variant for this dimension")
    return "\n".join(lines) + "\n"


_CSV_FIELDS = [
    "dimension", "label", "is_baseline_variant", "status", "error",
    "percentage_agreement", "percentage_agreement_all",
    "fixation_recall", "saccade_recall", "cohen_kappa",
    "event_mean_iou", "event_correct_type_rate",
    "n_gt_events", "n_pred_events", "n_matched", "n_fn", "n_fp",
    "duration_s",
]


def _round(v: Any) -> Any:
    if isinstance(v, float) and math.isfinite(v):
        return round(v, 6)
    return v


def _csv_row(dimension: str, label: str, is_baseline: bool, result: dict) -> dict:
    metrics = result.get("metrics", {})
    event = result.get("event_iou") or {}
    return {
        "dimension": dimension,
        "label": label,
        "is_baseline_variant": is_baseline,
        "status": result.get("status", ""),
        "error": result.get("error", ""),
        "percentage_agreement": _round(metrics.get("percentage_agreement")),
        "percentage_agreement_all": _round(metrics.get("percentage_agreement_all")),
        "fixation_recall": _round(metrics.get("fixation_recall")),
        "saccade_recall": _round(metrics.get("saccade_recall")),
        "cohen_kappa": _round(metrics.get("cohen_kappa")),
        "event_mean_iou": _round(event.get("mean_iou")),
        "event_correct_type_rate": _round(event.get("correct_type_rate")),
        "n_gt_events": event.get("n_gt_events"),
        "n_pred_events": event.get("n_pred_events"),
        "n_matched": event.get("n_matched"),
        "n_fn": event.get("n_fn"),
        "n_fp": event.get("n_fp"),
        "duration_s": result.get("duration_s"),
    }


def write_reports(data: dict[str, Any], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "sweep_results.json").write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (results_dir / "sweep_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        b = data["baseline"]
        writer.writerow(_csv_row("baseline", "baseline", True, b))
        for dimension, variants in data["sweep"].items():
            for v in variants:
                writer.writerow(_csv_row(dimension, v["label"], v.get("is_baseline_variant", False), v))

    (results_dir / "sweep_summary.txt").write_text(
        format_text_report(data), encoding="utf-8"
    )


def _json_safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    available = [d for d, _ in SWEEP_DIMENSIONS]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-both", type=Path, default=DEFAULT_BOTH_INPUT_FILE,
        dest="input_both",
        help=(
            "Botheye input TSV (used for: eye_mode, velocity_method, velocity_threshold, "
            "gap_fill, tobii_eye_offset_interp). Default: I-VT-botheye20ms30threshold_input.tsv"
        ),
    )
    parser.add_argument(
        "--input-left", type=Path, default=DEFAULT_LEFT_INPUT_FILE,
        dest="input_left",
        help=(
            "Left-eye input TSV (used for: window_length_ms, smoothing, window_policy, "
            "postprocessing). Default: LeftV30W20_input.tsv"
        ),
    )
    # Keep --input as a deprecated alias for --input-both
    parser.add_argument(
        "--input", type=Path, default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Directory for output files (default: results/sweep_benchmark/).",
    )
    parser.add_argument(
        "--dimensions", nargs="+", metavar="DIM", default=None,
        help=f"Run only these dimensions. Available: {available}",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Abort on first variant error instead of recording it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # --input is a deprecated alias for --input-both
    input_both: Path = args.input if args.input is not None else args.input_both
    input_left: Path = args.input_left

    for label, path in (("--input-both", input_both), ("--input-left", input_left)):
        if not path.exists():
            raise SystemExit(f"Input file not found ({label}): {path}")

    dimensions = SWEEP_DIMENSIONS
    if args.dimensions:
        requested = set(args.dimensions)
        dimensions = [(d, v) for d, v in SWEEP_DIMENSIONS if d in requested]
        unknown = requested - {d for d, _ in dimensions}
        if unknown:
            available = [d for d, _ in SWEEP_DIMENSIONS]
            raise SystemExit(
                f"Unknown dimensions: {sorted(unknown)}\n"
                f"Available: {available}"
            )

    data = run_sweep(
        input_both, input_left, dimensions, continue_on_error=not args.fail_fast
    )
    write_reports(data, args.results_dir)

    print("\n" + format_text_report(data))
    print(f"[sweep] results written to {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
