#!/usr/bin/env python3
"""Benchmark IVT filter settings inferred from files in test_data/inputs."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
import json
import math
from numbers import Number
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterable

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


DEFAULT_INPUT_DIR = REPO_ROOT / "test_data" / "inputs"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "filter_benchmark"

CSV_FIELDS = [
    "file",
    "status",
    "error_type",
    "error_message",
    "duration_seconds",
    "n_samples_total",
    "n_samples_gt_fix_or_sac",
    "percentage_agreement",
    "percentage_agreement_all",
    "fixation_recall",
    "saccade_recall",
    "cohen_kappa",
    "event_n_gt_events",
    "event_n_pred_events",
    "event_n_matched",
    "event_n_fn",
    "event_n_fp",
    "event_mean_iou",
    "event_correct_type_rate",
    "eye_mode",
    "velocity_method",
    "threshold_deg_per_sec",
    "window_length_ms",
    "smoothing_mode",
    "smoothing_window_samples",
    "gap_fill_enabled",
    "gap_fill_max_gap_ms",
    "merge_adjacent_fixations",
    "discard_short_fixations",
    "inference_notes",
]


def _parse_float(value: str) -> float:
    return float(value.replace(",", ".").replace("-", "."))


def _compact_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ("_input", "_extracted", "_slim", "_small"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def infer_pipeline_config(path: Path) -> tuple[PipelineConfig, dict[str, Any]]:
    """Infer the filter configuration encoded in an input filename.

    The repository's historical input names are not a formal schema, so this
    function records every inferred value in ``notes`` for auditability.
    """
    stem = _compact_stem(path)
    lower = stem.lower()
    notes: dict[str, Any] = {"source_name": path.name}

    eye_mode = "average"
    if "right" in lower:
        eye_mode = "right"
    elif "left" in lower:
        eye_mode = "left"
    if "both" in lower or lower.startswith("lr") or "-lr" in lower:
        eye_mode = "average"
    notes["eye_mode"] = eye_mode

    threshold = _infer_threshold(stem)
    notes["threshold_deg_per_sec"] = threshold

    window_ms = _infer_window_ms(stem)
    notes["window_length_ms"] = window_ms

    smoothing_mode, smoothing_window = _infer_smoothing(stem)
    notes["smoothing_mode"] = smoothing_mode
    notes["smoothing_window_samples"] = smoothing_window

    gap_fill_enabled, gap_fill_ms = _infer_gap_fill(stem)
    notes["gap_fill_enabled"] = gap_fill_enabled
    notes["gap_fill_max_gap_ms"] = gap_fill_ms

    fixation_post = _infer_fixation_post(stem, notes)
    time_column, time_unit = _infer_time_settings(path, notes)

    velocity = OlsenVelocityConfig(
        window_length_ms=window_ms,
        velocity_method="olsen2d",
        time_column=time_column,
        time_unit=time_unit,
        eye_mode=eye_mode,
        smoothing_mode=smoothing_mode,
        smoothing_window_samples=smoothing_window,
        gap_fill_enabled=gap_fill_enabled,
        gap_fill_max_gap_ms=gap_fill_ms,
    )
    classifier = IVTClassifierConfig(velocity_threshold_deg_per_sec=threshold)
    config = PipelineConfig(
        velocity=velocity,
        classifier=classifier,
        classify=True,
        fixation_post=fixation_post,
    )
    return config, notes


def load_file_configs(configs_path: Path) -> dict[str, Any]:
    """Load explicit per-file configurations from a JSON file.

    Returns a dict with keys ``files`` (mapping filename → config entry) and
    ``skip`` (list of filenames to exclude from benchmarking).
    """
    if not configs_path.exists():
        return {"files": {}, "skip": []}
    with configs_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data


def build_pipeline_config_from_entry(
    entry: dict[str, Any], path: Path
) -> tuple[PipelineConfig, dict[str, Any]]:
    """Build a PipelineConfig from an explicit JSON config entry.

    Time column/unit are still probed from the file itself since that
    information is not encoded in the config entry.
    """
    notes: dict[str, Any] = {
        "source_name": path.name,
        "config_source": "configs.json",
        "entry_notes": entry.get("notes", ""),
    }
    time_column, time_unit = _infer_time_settings(path, notes)

    velocity_method = entry.get("velocity_method", "tobii_gaze_dir")
    eye_mode = entry.get("eye_mode", "average")
    window_ms = float(entry.get("window_length_ms", 20.0))
    smoothing_mode = entry.get("smoothing_mode", "none")
    smoothing_window = int(entry.get("smoothing_window_samples", 5))
    gap_fill_enabled = bool(entry.get("gap_fill_enabled", False))
    gap_fill_ms = float(entry.get("gap_fill_max_gap_ms", 75.0))
    threshold = float(entry.get("velocity_threshold_deg_per_sec", 30.0))

    velocity = OlsenVelocityConfig(
        window_length_ms=window_ms,
        velocity_method=velocity_method,
        time_column=time_column,
        time_unit=time_unit,
        eye_mode=eye_mode,
        smoothing_mode=smoothing_mode,
        smoothing_window_samples=smoothing_window,
        gap_fill_enabled=gap_fill_enabled,
        gap_fill_max_gap_ms=gap_fill_ms,
    )
    classifier = IVTClassifierConfig(velocity_threshold_deg_per_sec=threshold)

    merge = bool(entry.get("merge_adjacent_fixations", False))
    discard = bool(entry.get("discard_short_fixations", False))
    if merge or discard:
        fixation_post = FixationPostConfig()
        if merge:
            fixation_post.merge_adjacent_fixations = True
            fixation_post.max_time_gap_ms = float(entry.get("merge_max_time_gap_ms", 75.0))
            fixation_post.max_angle_deg = float(entry.get("merge_max_angle_deg", 0.5))
        if discard:
            fixation_post.discard_short_fixations = True
            fixation_post.min_fixation_duration_ms = float(entry.get("min_fixation_duration_ms", 60.0))
            if "discard_target" in entry:
                fixation_post.discard_target = entry["discard_target"]
        fixation_post.__post_init__()
    else:
        fixation_post = None

    notes.update({
        "eye_mode": eye_mode,
        "velocity_method": velocity_method,
        "threshold_deg_per_sec": threshold,
        "window_length_ms": window_ms,
        "smoothing_mode": smoothing_mode,
        "smoothing_window_samples": smoothing_window,
        "gap_fill_enabled": gap_fill_enabled,
        "gap_fill_max_gap_ms": gap_fill_ms,
        "merge_adjacent_fixations": merge,
        "discard_short_fixations": discard,
    })

    config = PipelineConfig(
        velocity=velocity,
        classifier=classifier,
        classify=True,
        fixation_post=fixation_post,
    )
    return config, notes


def _infer_threshold(stem: str) -> float:
    match = re.search(r"onlyvelocityT?(\d+(?:[.,]\d+)?)", stem, flags=re.IGNORECASE)
    if match:
        return _parse_float(match.group(1))

    patterns = [
        (r"IVT(\d+(?:[.,]\d+)?)", 0),
        (r"T(\d+(?:[.,]\d+)?)", 0),
        (r"(?<!A)V(\d+(?:[.,]\d+)?)", 0),
        (r"ms(\d+(?:[.,]\d+)?)", re.IGNORECASE),
        (r"(\d+(?:[.,]\d+)?)threshold", re.IGNORECASE),
    ]
    for pattern, flags in patterns:
        match = re.search(pattern, stem, flags=flags)
        if match:
            return _parse_float(match.group(1))
    return 30.0


def _infer_window_ms(stem: str) -> float:
    patterns = [
        r"Win(\d+(?:[.,]\d+)?)",
        r"W(\d+(?:[.,]\d+)?)",
        r"(\d+(?:[.,]\d+)?)ms",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return _parse_float(match.group(1))
    return 20.0


def _infer_smoothing(stem: str) -> tuple[str, int]:
    if re.search(r"NoNoise", stem, flags=re.IGNORECASE):
        return "none", 5

    median = re.search(r"Noise(?:Med|M)W?(\d+)", stem, flags=re.IGNORECASE)
    if median:
        return "median", _ensure_odd(int(median.group(1)))

    average = re.search(r"Noise(?:Av|Avg|Average)W?(\d+)", stem, flags=re.IGNORECASE)
    if average:
        return "moving_average", _ensure_odd(int(average.group(1)))

    return "none", 5


def _ensure_odd(value: int) -> int:
    if value < 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _infer_gap_fill(stem: str) -> tuple[bool, float]:
    match = re.search(
        r"(?:Interp|Interpolation|Int|In)(\d+(?:[.,]\d+)?)",
        stem,
        flags=re.IGNORECASE,
    )
    if match:
        return True, _parse_float(match.group(1))
    return False, 75.0


def _infer_time_settings(path: Path, notes: dict[str, Any]) -> tuple[str, str]:
    """Prefer time_ms, but fall back to time_us when time_ms is empty."""
    try:
        probe = pd.read_csv(
            path,
            sep="\t",
            decimal=",",
            nrows=25,
            usecols=lambda col: col in {"time_ms", "time_us"},
        )
    except Exception as exc:
        notes["time_probe_error"] = f"{type(exc).__name__}: {exc}"
        notes["time_column"] = "time_ms"
        notes["time_unit"] = "ms"
        return "time_ms", "ms"

    for column, unit in (("time_ms", "ms"), ("time_us", "us")):
        if column not in probe.columns:
            continue
        values = pd.to_numeric(probe[column], errors="coerce")
        if values.notna().any():
            notes["time_column"] = column
            notes["time_unit"] = unit
            return column, unit

    notes["time_column"] = "time_ms"
    notes["time_unit"] = "ms"
    notes["time_probe_warning"] = "No numeric time_ms or time_us values found in probe rows."
    return "time_ms", "ms"


def _infer_fixation_post(stem: str, notes: dict[str, Any]) -> FixationPostConfig | None:
    merge = re.search(
        r"MergeF(\d+(?:[.,]\d+)?)ms(?:(\d+)[-_](\d+)angle)?",
        stem,
        flags=re.IGNORECASE,
    )
    discard = re.search(r"Discard(\d+(?:[.,]\d+)?)", stem, flags=re.IGNORECASE)

    if not merge and not discard:
        notes["merge_adjacent_fixations"] = False
        notes["discard_short_fixations"] = False
        return None

    cfg = FixationPostConfig()
    if merge:
        cfg.merge_adjacent_fixations = True
        cfg.max_time_gap_ms = _parse_float(merge.group(1))
        if merge.group(2) and merge.group(3):
            cfg.max_angle_deg = float(f"{merge.group(2)}.{merge.group(3)}")
    if discard:
        cfg.discard_short_fixations = True
        cfg.min_fixation_duration_ms = _parse_float(discard.group(1))
        if "nosaccade" in stem.lower():
            cfg.discard_target = "Unclassified"

    notes["merge_adjacent_fixations"] = cfg.merge_adjacent_fixations
    notes["merge_fix_max_gap_ms"] = cfg.max_time_gap_ms
    notes["merge_fix_max_angle_deg"] = cfg.max_angle_deg
    notes["discard_short_fixations"] = cfg.discard_short_fixations
    notes["min_fixation_duration_ms"] = cfg.min_fixation_duration_ms
    notes["discard_target"] = cfg.discard_target
    cfg.__post_init__()
    return cfg


def prediction_column(config: PipelineConfig) -> str:
    if config.fixation_post:
        return "ivt_event_type_post"
    if config.saccade_merge:
        sample_col = config.saccade_merge.use_sample_type_column
        return f"{sample_col}_smoothed" if sample_col else "ivt_event_type_smoothed"
    return "ivt_sample_type"


def discover_inputs(input_dir: Path, pattern: str, limit: int | None = None) -> list[Path]:
    files = [
        path
        for path in sorted(input_dir.glob(pattern))
        if path.is_file() and not path.name.startswith(".~lock")
    ]
    if limit is not None:
        return files[:limit]
    return files


def run_benchmark(
    input_files: Iterable[Path],
    results_dir: Path,
    *,
    configs_path: Path | None = None,
    write_outputs: bool = False,
    exclude_calibration: bool = False,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    file_configs = load_file_configs(configs_path) if configs_path else {"files": {}, "skip": []}
    skip_set = set(file_configs.get("skip", []))
    explicit_configs = file_configs.get("files", {})

    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = results_dir / "outputs"
    if write_outputs:
        outputs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in input_files:
        if path.name in skip_set:
            print(f"[benchmark] skipping {path.name} (skip list)", flush=True)
            continue
        started = time.perf_counter()
        print(f"[benchmark] {path.name}", flush=True)
        try:
            if path.name in explicit_configs:
                config, notes = build_pipeline_config_from_entry(explicit_configs[path.name], path)
            else:
                config, notes = infer_pipeline_config(path)
            output_path = outputs_dir / f"{path.stem}_benchmark.tsv" if write_outputs else None
            df = IVTPipeline(config).run(
                str(path),
                output_path=str(output_path) if output_path else None,
                evaluate=False,
                plot=False,
            )
            pred_col = prediction_column(config)
            metrics = compute_ivt_metrics(
                df,
                pred_col=pred_col,
                exclude_calibration=exclude_calibration,
            )
            event_summary = _compute_event_summary(df, pred_col)
            result = {
                "file": path.name,
                "status": "ok",
                "duration_seconds": time.perf_counter() - started,
                "config": config_to_dict(config),
                "inference_notes": notes,
                "metrics": metrics,
                "event_iou": event_summary,
                "output_path": str(output_path) if output_path else None,
            }
        except Exception as exc:
            result = {
                "file": path.name,
                "status": "error",
                "duration_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            if not continue_on_error:
                raise
        results.append(result)
    return results


def _compute_event_summary(df: Any, pred_col: str) -> dict[str, Any] | None:
    gt_col = "gt_sample_type" if "gt_sample_type" in df.columns else "gt_event_type"
    if gt_col not in df.columns or pred_col not in df.columns or "time_ms" not in df.columns:
        return None

    metrics = compute_event_iou_metrics(
        df,
        gt_col=gt_col,
        pred_col=pred_col,
        time_col="time_ms",
    )
    matches = metrics.get("matches", [])
    correct_type = sum(1 for match in matches if match.is_correct_type)
    matched = [match for match in matches if match.pred_event is not None]
    ious = [match.iou for match in matched]
    n_gt = metrics["n_gt_events"]

    return {
        "n_gt_events": n_gt,
        "n_pred_events": metrics["n_pred_events"],
        "n_matched": metrics["n_matched"],
        "n_fn": metrics["n_fn"],
        "n_fp": metrics["n_fp"],
        "n_correct_type": correct_type,
        "match_rate": metrics["n_matched"] / n_gt if n_gt else float("nan"),
        "correct_type_rate": correct_type / n_gt if n_gt else float("nan"),
        "mean_iou": statistics.fmean(ious) if ious else float("nan"),
        "event_counts": metrics["event_counts"],
        "confusion_matrix": metrics["confusion_matrix"],
        "timing": metrics["timing"],
    }


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return _json_safe(asdict(config))


def build_aggregate_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_results = [result for result in results if result.get("status") == "ok"]
    error_results = [result for result in results if result.get("status") != "ok"]

    totals = {
        "files_total": len(results),
        "files_ok": len(ok_results),
        "files_error": len(error_results),
        "duration_seconds": sum(_metric(result, "duration_seconds") or 0.0 for result in results),
        "n_samples_total": sum(_metric(result, "n_samples_total") or 0 for result in ok_results),
        "n_samples_gt_fix_or_sac": sum(
            _metric(result, "n_samples_gt_fix_or_sac") or 0 for result in ok_results
        ),
        "n_agree": sum(_metric(result, "n_agree") or 0 for result in ok_results),
        "n_agree_all": sum(_metric(result, "n_agree_all") or 0 for result in ok_results),
        "n_fix_in_gt": sum(_metric(result, "n_fix_in_gt") or 0 for result in ok_results),
        "n_sac_in_gt": sum(_metric(result, "n_sac_in_gt") or 0 for result in ok_results),
        "tp_fix": sum(_metric(result, "tp_fix") or 0 for result in ok_results),
        "tp_sac": sum(_metric(result, "tp_sac") or 0 for result in ok_results),
        "event_n_gt_events": sum(_event_metric(result, "n_gt_events") or 0 for result in ok_results),
        "event_n_pred_events": sum(_event_metric(result, "n_pred_events") or 0 for result in ok_results),
        "event_n_matched": sum(_event_metric(result, "n_matched") or 0 for result in ok_results),
        "event_n_fn": sum(_event_metric(result, "n_fn") or 0 for result in ok_results),
        "event_n_fp": sum(_event_metric(result, "n_fp") or 0 for result in ok_results),
        "event_n_correct_type": sum(
            _event_metric(result, "n_correct_type") or 0 for result in ok_results
        ),
    }

    weighted = {
        "percentage_agreement": _percent(totals["n_agree"], totals["n_samples_gt_fix_or_sac"]),
        "percentage_agreement_all": _percent(totals["n_agree_all"], totals["n_samples_total"]),
        "fixation_recall": _percent(totals["tp_fix"], totals["n_fix_in_gt"]),
        "saccade_recall": _percent(totals["tp_sac"], totals["n_sac_in_gt"]),
        "event_match_rate": _ratio(totals["event_n_matched"], totals["event_n_gt_events"]),
        "event_correct_type_rate": _ratio(
            totals["event_n_correct_type"], totals["event_n_gt_events"]
        ),
    }

    per_file = {
        "mean_percentage_agreement": _mean_metric(ok_results, "percentage_agreement"),
        "mean_percentage_agreement_all": _mean_metric(ok_results, "percentage_agreement_all"),
        "mean_fixation_recall": _mean_metric(ok_results, "fixation_recall"),
        "mean_saccade_recall": _mean_metric(ok_results, "saccade_recall"),
        "mean_cohen_kappa": _mean_metric(ok_results, "cohen_kappa"),
        "mean_event_iou": _mean_event_metric(ok_results, "mean_iou"),
    }

    return {
        "totals": totals,
        "weighted": weighted,
        "per_file_mean": per_file,
        "pooled_cohen_kappa": _pooled_kappa(ok_results),
        "best_files": _rank_files(ok_results, reverse=True),
        "worst_files": _rank_files(ok_results, reverse=False),
        "errors": [
            {
                "file": result["file"],
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
            for result in error_results
        ],
    }


def _metric(result: dict[str, Any], key: str) -> Any:
    if key in result:
        return result[key]
    return result.get("metrics", {}).get(key)


def _event_metric(result: dict[str, Any], key: str) -> Any:
    event = result.get("event_iou") or {}
    return event.get(key)


def _percent(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100.0 if denominator else float("nan")


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else float("nan")


def _valid_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool) and math.isfinite(float(value))


def _mean_metric(results: list[dict[str, Any]], key: str) -> float:
    values = [float(_metric(result, key)) for result in results if _valid_number(_metric(result, key))]
    return statistics.fmean(values) if values else float("nan")


def _mean_event_metric(results: list[dict[str, Any]], key: str) -> float:
    values = [
        float(_event_metric(result, key))
        for result in results
        if _valid_number(_event_metric(result, key))
    ]
    return statistics.fmean(values) if values else float("nan")


def _pooled_kappa(results: list[dict[str, Any]]) -> float:
    labels: list[str] = []
    pooled: dict[str, dict[str, int]] = {}
    for result in results:
        metrics = result.get("metrics", {})
        result_labels = metrics.get("labels", [])
        conf = metrics.get("confusion_matrix", [])
        for label in result_labels:
            if label not in labels:
                labels.append(label)
        for gt_label, row in zip(result_labels, conf):
            pooled.setdefault(gt_label, {})
            for pred_label, count in zip(result_labels, row):
                pooled[gt_label][pred_label] = pooled[gt_label].get(pred_label, 0) + int(count)

    labels.sort()
    total = sum(sum(row.values()) for row in pooled.values())
    if total == 0:
        return float("nan")

    po = sum(pooled.get(label, {}).get(label, 0) for label in labels) / total
    row_marg = {label: sum(pooled.get(label, {}).values()) for label in labels}
    col_marg = {
        label: sum(pooled.get(gt_label, {}).get(label, 0) for gt_label in labels)
        for label in labels
    }
    pe = sum(row_marg[label] * col_marg[label] for label in labels) / (total * total)
    return (po - pe) / (1.0 - pe) if 1.0 - pe else float("nan")


def _rank_files(results: list[dict[str, Any]], *, reverse: bool) -> list[dict[str, Any]]:
    ranked = [
        result
        for result in results
        if _valid_number(_metric(result, "percentage_agreement"))
    ]
    ranked.sort(key=lambda item: float(_metric(item, "percentage_agreement")), reverse=reverse)
    return [
        {
            "file": result["file"],
            "percentage_agreement": _metric(result, "percentage_agreement"),
            "percentage_agreement_all": _metric(result, "percentage_agreement_all"),
            "cohen_kappa": _metric(result, "cohen_kappa"),
        }
        for result in ranked[:5]
    ]


def write_reports(results: list[dict[str, Any]], summary: dict[str, Any], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {"summary": summary, "results": results}
    (results_dir / "benchmark_results.json").write_text(
        json.dumps(_json_safe(json_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(results, results_dir / "benchmark_metrics.csv")
    (results_dir / "benchmark_summary.txt").write_text(
        format_summary(summary),
        encoding="utf-8",
    )


def _write_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(_csv_row(result))


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config", {})
    velocity = config.get("velocity", {})
    classifier = config.get("classifier", {})
    fixation_post = config.get("fixation_post") or {}
    event = result.get("event_iou") or {}
    row = {
        "file": result.get("file"),
        "status": result.get("status"),
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message"),
        "duration_seconds": _round(result.get("duration_seconds")),
        "event_n_gt_events": event.get("n_gt_events"),
        "event_n_pred_events": event.get("n_pred_events"),
        "event_n_matched": event.get("n_matched"),
        "event_n_fn": event.get("n_fn"),
        "event_n_fp": event.get("n_fp"),
        "event_mean_iou": _round(event.get("mean_iou")),
        "event_correct_type_rate": _round(event.get("correct_type_rate")),
        "eye_mode": velocity.get("eye_mode"),
        "velocity_method": velocity.get("velocity_method"),
        "threshold_deg_per_sec": classifier.get("velocity_threshold_deg_per_sec"),
        "window_length_ms": velocity.get("window_length_ms"),
        "smoothing_mode": velocity.get("smoothing_mode"),
        "smoothing_window_samples": velocity.get("smoothing_window_samples"),
        "gap_fill_enabled": velocity.get("gap_fill_enabled"),
        "gap_fill_max_gap_ms": velocity.get("gap_fill_max_gap_ms"),
        "merge_adjacent_fixations": fixation_post.get("merge_adjacent_fixations", False),
        "discard_short_fixations": fixation_post.get("discard_short_fixations", False),
        "inference_notes": json.dumps(_json_safe(result.get("inference_notes", {})), sort_keys=True),
    }
    for key in (
        "n_samples_total",
        "n_samples_gt_fix_or_sac",
        "percentage_agreement",
        "percentage_agreement_all",
        "fixation_recall",
        "saccade_recall",
        "cohen_kappa",
    ):
        row[key] = _round(_metric(result, key))
    return row


def format_summary(summary: dict[str, Any]) -> str:
    totals = summary["totals"]
    weighted = summary["weighted"]
    means = summary["per_file_mean"]
    lines = [
        "I-VT Filter Benchmark Gesamtstatistik",
        "====================================",
        "",
        f"Dateien: {totals['files_ok']} erfolgreich / {totals['files_total']} gesamt",
        f"Fehlerhafte Dateien: {totals['files_error']}",
        f"Laufzeit gesamt: {_fmt(totals['duration_seconds'])} s",
        f"Samples gesamt: {totals['n_samples_total']}",
        f"GT Fixation/Saccade Samples: {totals['n_samples_gt_fix_or_sac']}",
        "",
        "Gewichtete Gesamtmetriken",
        f"- Agreement Fix/Sac: {_fmt(weighted['percentage_agreement'])} %",
        f"- Agreement alle Labels: {_fmt(weighted['percentage_agreement_all'])} %",
        f"- Fixation Recall: {_fmt(weighted['fixation_recall'])} %",
        f"- Saccade Recall: {_fmt(weighted['saccade_recall'])} %",
        f"- Cohen's Kappa gepoolt: {_fmt(summary['pooled_cohen_kappa'])}",
        f"- Event Match Rate: {_fmt(weighted['event_match_rate'])}",
        f"- Event Correct-Type Rate: {_fmt(weighted['event_correct_type_rate'])}",
        "",
        "Ungewichtete Mittelwerte pro Datei",
        f"- Agreement Fix/Sac: {_fmt(means['mean_percentage_agreement'])} %",
        f"- Agreement alle Labels: {_fmt(means['mean_percentage_agreement_all'])} %",
        f"- Fixation Recall: {_fmt(means['mean_fixation_recall'])} %",
        f"- Saccade Recall: {_fmt(means['mean_saccade_recall'])} %",
        f"- Cohen's Kappa: {_fmt(means['mean_cohen_kappa'])}",
        f"- Event Mean IoU: {_fmt(means['mean_event_iou'])}",
        "",
        "Beste Dateien nach Fix/Sac Agreement",
        *_format_ranked(summary["best_files"]),
        "",
        "Schlechteste Dateien nach Fix/Sac Agreement",
        *_format_ranked(summary["worst_files"]),
    ]
    if summary["errors"]:
        lines.extend(["", "Fehler"])
        lines.extend(
            f"- {item['file']}: {item['error_type']}: {item['error_message']}"
            for item in summary["errors"]
        )
    return "\n".join(lines) + "\n"


def _format_ranked(items: list[dict[str, Any]]) -> list[str]:
    if not items:
        return ["- Keine erfolgreichen Dateien"]
    return [
        f"- {item['file']}: {_fmt(item['percentage_agreement'])} % "
        f"(alle Labels {_fmt(item['percentage_agreement_all'])} %, "
        f"kappa {_fmt(item['cohen_kappa'])})"
        for item in items
    ]


def _round(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if _valid_number(value):
        return round(float(value), 6)
    return value


def _fmt(value: Any) -> str:
    if _valid_number(value):
        return f"{float(value):.4f}"
    return "n/a"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Number) and not isinstance(value, bool):
        numeric = float(value)
        if not math.isfinite(numeric):
            return None
        if isinstance(value, int):
            return int(value)
        return numeric
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run all IVT filter input files with settings inferred from their "
            "filenames and write aggregate benchmark statistics."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--configs",
        type=Path,
        default=None,
        help=(
            "Path to a configs.json file with explicit per-file filter settings. "
            "Defaults to <input-dir>/configs.json when that file exists."
        ),
    )
    parser.add_argument("--pattern", default="*.tsv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-outputs", action="store_true")
    parser.add_argument("--exclude-calibration", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first file error instead of recording it and continuing.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Return exit code 0 even when individual files are recorded as errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_files = discover_inputs(args.input_dir, args.pattern, args.limit)
    if not input_files:
        raise SystemExit(f"No input files matched {args.input_dir / args.pattern}")

    # Resolve configs path: explicit arg > auto-detect default location
    configs_path: Path | None = args.configs
    if configs_path is None:
        default_configs = args.input_dir / "configs.json"
        if default_configs.exists():
            configs_path = default_configs
            print(f"[benchmark] using configs: {configs_path}", flush=True)

    results = run_benchmark(
        input_files,
        args.results_dir,
        configs_path=configs_path,
        write_outputs=args.write_outputs,
        exclude_calibration=args.exclude_calibration,
        continue_on_error=not args.fail_fast,
    )
    summary = build_aggregate_summary(results)
    write_reports(results, summary, args.results_dir)
    print(format_summary(summary))
    print(f"[benchmark] wrote reports to {args.results_dir}")
    return 0 if args.allow_errors or summary["totals"]["files_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
