#!/usr/bin/env python3
"""Attribute per-sample disagreements vs Tobii to a cause category (WS-A).

For each benchmark condition the script runs the reconstruction pipeline with
that file's configuration, attributes every sample that disagrees with the Tobii
ground truth to one of the discrepancy categories, and writes a per-condition
breakdown plus a pooled summary. This turns the thesis's aggregate-only
diagnosis into per-sample numbers (the gap WS-A fills) and prioritises WS-B.

Outputs (results/attribution/):
  attribution_<stem>.csv      disagreeing samples with their category
  attribution_summary.json    per-condition + pooled category counts
  attribution_summary.txt      human-readable report
  attribution_overview.png    optional stacked-bar of category fractions

Examples:
  python scripts/attribute_discrepancies.py
  python scripts/attribute_discrepancies.py --files LeftFixation --max-rows 4000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_filter_settings as bench  # noqa: E402
from ivt_filter.evaluation.discrepancy import (  # noqa: E402
    CATEGORIES,
    DISCREPANCY_CATEGORY,
    AttributionParams,
    attribute_discrepancies,
    format_attribution_report,
)
from ivt_filter.io.io import read_tsv  # noqa: E402
from ivt_filter.io.pipeline import IVTPipeline  # noqa: E402

DEFAULT_INPUT_DIR = REPO_ROOT / "test_data" / "inputs"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "attribution"


def select_inputs(
    input_dir: Path,
    pattern: str,
    files_filter: list[str] | None,
    limit: int | None,
    skip_set: set[str],
) -> list[Path]:
    paths = [p for p in bench.discover_inputs(input_dir, pattern) if p.name not in skip_set]
    if files_filter:
        needles = [f.lower() for f in files_filter]
        paths = [p for p in paths if any(n in p.name.lower() for n in needles)]
    if limit is not None:
        paths = paths[:limit]
    return paths


def build_config(path: Path, explicit_configs: dict[str, Any]):
    if path.name in explicit_configs:
        return bench.build_pipeline_config_from_entry(explicit_configs[path.name], path)
    return bench.infer_pipeline_config(path)


def attribute_file(path: Path, config, eps: float, max_rows: int | None):
    """Run the pipeline on one file and attribute its disagreements."""
    df_raw = read_tsv(str(path))
    if max_rows is not None:
        df_raw = df_raw.head(max_rows).copy()
    df = IVTPipeline(config).process_dataframe(df_raw, config=config)
    pred_col = bench.prediction_column(config)
    threshold = float(config.classifier.velocity_threshold_deg_per_sec)
    params = AttributionParams(near_threshold_eps_deg=eps)
    attributed, summary = attribute_discrepancies(
        df, config.velocity, threshold, pred_col=pred_col, params=params
    )
    return attributed, summary, pred_col


def write_condition_csv(attributed, pred_col: str, gt_col: str, out_path: Path) -> None:
    mask = attributed[DISCREPANCY_CATEGORY] != ""
    disagreements = attributed.loc[mask]
    columns = {
        "sample_idx": disagreements.index,
        "time_ms": disagreements.get("time_ms"),
        "gt": disagreements[gt_col],
        "pred": disagreements[pred_col],
        "velocity_deg_per_sec": disagreements.get("velocity_deg_per_sec"),
        "velocity_eye_used": disagreements.get("velocity_eye_used"),
        DISCREPANCY_CATEGORY: disagreements[DISCREPANCY_CATEGORY],
    }
    import pandas as pd

    pd.DataFrame(columns).to_csv(out_path, index=False)


def pooled_summary(per_condition: dict[str, dict]) -> dict:
    counts = {cat: 0 for cat in CATEGORIES}
    n_disagreements = 0
    n_total = 0
    for summary in per_condition.values():
        n_total += summary["n_total"]
        n_disagreements += summary["n_disagreements"]
        for cat in CATEGORIES:
            counts[cat] += summary["counts"][cat]
    fraction = {
        cat: (counts[cat] / n_disagreements if n_disagreements else 0.0) for cat in CATEGORIES
    }
    return {
        "n_total": n_total,
        "n_disagreements": n_disagreements,
        "counts": counts,
        "fraction": fraction,
        "n_conditions": len(per_condition),
    }


def maybe_plot(per_condition: dict[str, dict], out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    import numpy as np

    conditions = list(per_condition)
    bar_cats = [c for c in CATEGORIES if c != "other"] + ["other"]
    bottoms = np.zeros(len(conditions))
    fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 0.9), 5))
    for cat in bar_cats:
        values = np.array([per_condition[c]["fraction"][cat] for c in conditions])
        ax.bar(conditions, values, bottom=bottoms, label=cat)
        bottoms += values
    ax.set_ylabel("fraction of disagreements")
    ax.set_title("Discrepancy attribution by condition")
    ax.legend(loc="upper right", fontsize="small")
    plt.xticks(rotation=45, ha="right", fontsize="small")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--configs", default=None)
    parser.add_argument("--pattern", default="*.tsv")
    parser.add_argument("--files", nargs="*", default=None, help="filename substrings to include")
    parser.add_argument("--limit", type=int, default=None, help="max number of files")
    parser.add_argument("--max-rows", type=int, default=None, help="truncate each file (quick runs)")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--eps", type=float, default=0.05, help="near-threshold band (deg/s)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    configs_path = Path(args.configs) if args.configs else input_dir / "configs.json"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    file_configs = bench.load_file_configs(configs_path)
    skip_set = set(file_configs.get("skip", []))
    explicit_configs = file_configs.get("files", {})

    paths = select_inputs(input_dir, args.pattern, args.files, args.limit, skip_set)
    if not paths:
        print(f"[attribution] no input files under {input_dir} matching the filters", file=sys.stderr)
        return 1

    per_condition: dict[str, dict] = {}
    for path in paths:
        print(f"[attribution] {path.name}", flush=True)
        config, _notes = build_config(path, explicit_configs)
        attributed, summary, pred_col = attribute_file(path, config, args.eps, args.max_rows)
        stem = bench._compact_stem(path)
        write_condition_csv(attributed, pred_col, summary["gt_col"], results_dir / f"attribution_{stem}.csv")
        per_condition[path.name] = summary

    pooled = pooled_summary(per_condition)
    payload = {
        "params": {"near_threshold_eps_deg": args.eps},
        "conditions": per_condition,
        "pooled": pooled,
    }
    (results_dir / "attribution_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    report_lines = []
    for name, summary in per_condition.items():
        report_lines.append(f"== {name} ==")
        report_lines.append(format_attribution_report(summary))
        report_lines.append("")
    report_lines.append("== POOLED ==")
    report_lines.append(format_attribution_report({**pooled, "threshold": float("nan"),
                                                   "eye_mode": "mixed", "smoothing_mode": "mixed"}))
    (results_dir / "attribution_summary.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    if args.plot:
        if maybe_plot(per_condition, results_dir / "attribution_overview.png"):
            print("[attribution] wrote attribution_overview.png")
        else:
            print("[attribution] matplotlib unavailable; skipped plot", file=sys.stderr)

    print(f"[attribution] {len(per_condition)} conditions, "
          f"{pooled['n_disagreements']} disagreements -> {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
