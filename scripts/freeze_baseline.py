#!/usr/bin/env python3
"""Freeze the per-condition golden baseline for the WS-B no-regression gate.

Reads the latest benchmark output (results/filter_benchmark/benchmark_results.json,
produced by scripts/benchmark_filter_settings.py) and writes a compact, sorted
snapshot of the gate metrics per condition. Every WS-B fix is checked against
this snapshot: no condition's Cohen's kappa may drop below its frozen value.

This is a deliberate, rarely-run action. Run `make benchmark` first, then
`make baseline`. The frozen file is committed and only regenerated on purpose.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = REPO_ROOT / "results" / "filter_benchmark" / "benchmark_results.json"
DEFAULT_BASELINE = REPO_ROOT / "results" / "filter_benchmark" / "baseline_v1.0.0.json"

GATE_METRIC = "cohen_kappa"
TOLERANCE = 1e-6


def extract_conditions(results: list[dict]) -> dict[str, dict]:
    """Map condition file name -> frozen gate metrics."""
    conditions: dict[str, dict] = {}
    for entry in results:
        if entry.get("status") != "ok":
            continue
        metrics = entry.get("metrics") or {}
        event = entry.get("event_iou") or {}
        conditions[entry["file"]] = {
            "cohen_kappa": metrics.get("cohen_kappa"),
            "percentage_agreement": metrics.get("percentage_agreement"),
            "percentage_agreement_all": metrics.get("percentage_agreement_all"),
            "fixation_recall": metrics.get("fixation_recall"),
            "saccade_recall": metrics.get("saccade_recall"),
            "event_match_rate": event.get("match_rate"),
            "event_correct_type_rate": event.get("correct_type_rate"),
            "event_mean_iou": event.get("mean_iou"),
        }
    return dict(sorted(conditions.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--output", default=str(DEFAULT_BASELINE))
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    if not results_path.exists():
        parser.error(f"benchmark results not found: {results_path}\nRun `make benchmark` first.")

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    conditions = extract_conditions(payload["results"])

    baseline = {
        "version": args.version,
        "source": results_path.relative_to(REPO_ROOT).as_posix(),
        "gate_metric": GATE_METRIC,
        "tolerance": TOLERANCE,
        "n_conditions": len(conditions),
        "conditions": conditions,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(f"froze {len(conditions)} conditions to {output_path}")
    for name, metrics in conditions.items():
        print(f"  {name:52s} {GATE_METRIC}={metrics[GATE_METRIC]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
