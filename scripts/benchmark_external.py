#!/usr/bin/env python3
"""External benchmark on the Lund2013 / Andersson 2017 hand-annotated data (WS-C).

This is the third evaluation axis: instead of measuring fidelity to Tobii Pro
Lab, it measures how the reconstruction's I-VT agrees with two human coders (MN,
RA) on a community benchmark the closed Tobii software could never be run on.

For each annotated ``.mat`` the reconstruction runs with a fixed I-VT
configuration, then sample-level (Cohen's kappa, agreement, recall) and
event-level (IoU matching) metrics are computed against the human labels under a
label-mapping protocol. P0 is primary; S1/S2 vary PSO and pursuit handling.

Outputs (results/external_benchmark/):
  external_results.json   per-file + pooled metrics per protocol and coder
  external_metrics.csv    flat per-file table
  external_summary.txt    human-readable report
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ivt_filter.config import (  # noqa: E402
    IVTClassifierConfig,
    OlsenVelocityConfig,
    PipelineConfig,
)
from ivt_filter.evaluation.evaluation import compute_ivt_metrics  # noqa: E402
from ivt_filter.evaluation.event_iou import compute_event_iou_metrics  # noqa: E402
from ivt_filter.io.lund2013 import LUND_PROTOCOLS, load_lund2013_mat  # noqa: E402
from ivt_filter.io.pipeline import IVTPipeline  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "test_data" / "external" / "lund2013" / "img"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "external_benchmark"
SCORED_EVENT_TYPES = ["Fixation", "Saccade"]


def build_config(threshold: float, window_ms: float) -> PipelineConfig:
    """A plain I-VT configuration. olsen2d needs only gaze-mm + eye-z, which the
    Lund loader synthesises (the dataset has no gaze-direction vectors)."""
    velocity = OlsenVelocityConfig(
        window_length_ms=window_ms,
        velocity_method="olsen2d",
        eye_mode="average",
        sampling_rate_method="all_samples",
    )
    return PipelineConfig(
        velocity=velocity,
        classifier=IVTClassifierConfig(velocity_threshold_deg_per_sec=threshold),
    )


def coder_of(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_RA"):
        return "RA"
    if stem.endswith("_MN"):
        return "MN"
    return "?"


def discover_mats(data_dir: Path, coder: str) -> list[Path]:
    mats = sorted(data_dir.glob("*.mat"))
    if coder != "both":
        mats = [m for m in mats if coder_of(m) == coder]
    return mats


def _event_summary(out: pd.DataFrame) -> dict[str, float]:
    """Derive match_rate / mean_iou / correct_type_rate from the IoU matches.

    compute_event_iou_metrics returns raw counts plus the per-event ``matches``
    list; the ratios are derived here exactly as the internal benchmark does.
    """
    metrics = compute_event_iou_metrics(
        out,
        gt_col="gt_event_type",
        pred_col="ivt_sample_type",
        time_col="time_ms",
        event_types=SCORED_EVENT_TYPES,
    )
    matches = metrics.get("matches", [])
    matched = [m for m in matches if m.pred_event is not None]
    ious = [m.iou for m in matched]
    correct = sum(1 for m in matches if m.is_correct_type)
    n_gt = metrics["n_gt_events"]
    return {
        "n_gt_events": n_gt,
        "n_matched": metrics["n_matched"],
        "match_rate": metrics["n_matched"] / n_gt if n_gt else float("nan"),
        "correct_type_rate": correct / n_gt if n_gt else float("nan"),
        "mean_iou": (sum(ious) / len(ious)) if ious else float("nan"),
    }


def evaluate_file(path: Path, protocol: str, config: PipelineConfig) -> dict[str, Any]:
    df = load_lund2013_mat(path, protocol=protocol)
    out = IVTPipeline(config).process_dataframe(df.copy(), config=config)
    scored = out[out["gt_event_type"] != ""].copy()
    sample = compute_ivt_metrics(scored, pred_col="ivt_sample_type", gt_col="gt_event_type")
    event = _event_summary(out)
    return {"scored": scored, "sample": sample, "event": event, "n": len(df)}


def pooled_sample_metrics(scored_frames: list[pd.DataFrame]) -> dict[str, Any]:
    if not scored_frames:
        return {}
    combined = pd.concat(scored_frames, ignore_index=True)
    return compute_ivt_metrics(combined, pred_col="ivt_sample_type", gt_col="gt_event_type")


def mean_event_metrics(events: list[dict]) -> dict[str, float]:
    keys = ("match_rate", "mean_iou", "correct_type_rate")
    out: dict[str, float] = {}
    for key in keys:
        values = [e[key] for e in events if isinstance(e.get(key), (int, float)) and not math.isnan(e[key])]
        out[key] = sum(values) / len(values) if values else float("nan")
    return out


def _safe(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def run(data_dir: Path, protocols: list[str], coder: str, threshold: float, window_ms: float) -> dict:
    config = build_config(threshold, window_ms)
    coders = ["MN", "RA"] if coder == "both" else [coder]
    result: dict[str, Any] = {
        "config": {"velocity_threshold_deg_per_sec": threshold, "window_length_ms": window_ms,
                   "velocity_method": "olsen2d"},
        "data_dir": str(data_dir),
        "protocols": {},
    }
    for protocol in protocols:
        result["protocols"][protocol] = {}
        for cdr in coders:
            mats = discover_mats(data_dir, cdr)
            files: dict[str, Any] = {}
            scored_frames, events = [], []
            for path in mats:
                ev = evaluate_file(path, protocol, config)
                scored_frames.append(ev["scored"])
                events.append(ev["event"])
                files[path.name] = {
                    "n_samples": ev["n"],
                    "cohen_kappa": _safe(ev["sample"]["cohen_kappa"]),
                    "percentage_agreement": _safe(ev["sample"]["percentage_agreement"]),
                    "fixation_recall": _safe(ev["sample"]["fixation_recall"]),
                    "saccade_recall": _safe(ev["sample"]["saccade_recall"]),
                    "event_match_rate": _safe(ev["event"]["match_rate"]),
                    "event_mean_iou": _safe(ev["event"]["mean_iou"]),
                }
            pooled = pooled_sample_metrics(scored_frames)
            result["protocols"][protocol][cdr] = {
                "n_files": len(mats),
                "pooled_sample": {
                    "cohen_kappa": _safe(pooled.get("cohen_kappa")),
                    "percentage_agreement": _safe(pooled.get("percentage_agreement")),
                    "fixation_recall": _safe(pooled.get("fixation_recall")),
                    "saccade_recall": _safe(pooled.get("saccade_recall")),
                },
                "mean_event": {k: _safe(v) for k, v in mean_event_metrics(events).items()},
                "files": files,
            }
    return result


def write_outputs(result: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "external_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    rows = []
    for protocol, by_coder in result["protocols"].items():
        for cdr, block in by_coder.items():
            for fname, m in block["files"].items():
                rows.append({"protocol": protocol, "coder": cdr, "file": fname, **m})
    if rows:
        with (results_dir / "external_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = ["External benchmark (Lund2013 / Andersson 2017)",
             f"  config: {result['config']}", ""]
    for protocol, by_coder in result["protocols"].items():
        for cdr, block in by_coder.items():
            ps = block["pooled_sample"]
            me = block["mean_event"]
            kappa = ps.get("cohen_kappa")
            agree = ps.get("percentage_agreement")
            lines.append(
                f"  {protocol} vs {cdr} ({block['n_files']} files): "
                f"pooled kappa={kappa:.3f} agree={agree:.1f}% "
                f"fixR={ps.get('fixation_recall'):.0f} sacR={ps.get('saccade_recall'):.0f} | "
                f"event match={me.get('match_rate'):.3f} IoU={me.get('mean_iou'):.3f}"
                if kappa is not None else f"  {protocol} vs {cdr}: no data"
            )
    (results_dir / "external_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--protocol", default="P0", choices=sorted(LUND_PROTOCOLS))
    parser.add_argument("--all-protocols", action="store_true")
    parser.add_argument("--coder", default="both", choices=["MN", "RA", "both"])
    parser.add_argument("--velocity-threshold", type=float, default=30.0)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[external] data dir not found: {data_dir}\n"
              f"Provide the Lund2013 img/*.mat files (see docs/external_data.md).", file=sys.stderr)
        return 1
    if not list(data_dir.glob("*.mat")):
        print(f"[external] no .mat files under {data_dir}", file=sys.stderr)
        return 1

    protocols = sorted(LUND_PROTOCOLS) if args.all_protocols else [args.protocol]
    result = run(data_dir, protocols, args.coder, args.velocity_threshold, args.window_ms)
    write_outputs(result, Path(args.results_dir))
    print(f"[external] protocols={protocols} coder={args.coder} -> {args.results_dir}")
    for protocol, by_coder in result["protocols"].items():
        for cdr, block in by_coder.items():
            ps = block["pooled_sample"]
            if ps.get("cohen_kappa") is not None:
                print(f"  {protocol} vs {cdr}: pooled kappa={ps['cohen_kappa']:.3f} "
                      f"agree={ps['percentage_agreement']:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
