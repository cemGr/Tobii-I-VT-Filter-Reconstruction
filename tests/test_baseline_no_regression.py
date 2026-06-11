"""WS-B regression gate: no benchmark condition may lose fidelity.

This is the single go/no-go signal for every evidence-guided fix. It compares
the latest benchmark output against the frozen golden baseline
(results/filter_benchmark/baseline_v1.0.0.json) and fails if any condition's
Cohen's kappa drops below its frozen value, or if a condition errors or
disappears.

The A/B loop is: change code -> `make benchmark` (regenerates
benchmark_results.json) -> run this test. The test skips when the benchmark
output or the baseline is absent (e.g. CI without the gitignored study data).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "results" / "filter_benchmark" / "baseline_v1.0.0.json"
CURRENT_PATH = REPO_ROOT / "results" / "filter_benchmark" / "benchmark_results.json"


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        pytest.skip(f"baseline snapshot not found: {BASELINE_PATH} (run `make baseline`)")
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _load_current_kappa() -> dict[str, float]:
    if not CURRENT_PATH.exists():
        pytest.skip(f"benchmark output not found: {CURRENT_PATH} (run `make benchmark`)")
    payload = json.loads(CURRENT_PATH.read_text(encoding="utf-8"))
    current: dict[str, float] = {}
    for entry in payload["results"]:
        metrics = entry.get("metrics") or {}
        current[entry["file"]] = {
            "status": entry.get("status"),
            "cohen_kappa": metrics.get("cohen_kappa"),
        }
    return current


def test_no_condition_kappa_regression() -> None:
    """Every frozen condition keeps its Cohen's kappa (within tolerance)."""
    baseline = _load_baseline()
    current = _load_current_kappa()
    tol = float(baseline.get("tolerance", 1e-6))

    failures: list[str] = []
    for name, frozen in baseline["conditions"].items():
        frozen_kappa = frozen["cohen_kappa"]
        if name not in current:
            failures.append(f"{name}: missing from current benchmark output")
            continue
        entry = current[name]
        if entry["status"] != "ok":
            failures.append(f"{name}: status={entry['status']!r} (was ok in baseline)")
            continue
        now = entry["cohen_kappa"]
        if now is None:
            failures.append(f"{name}: cohen_kappa is None in current output")
            continue
        if now < frozen_kappa - tol:
            failures.append(
                f"{name}: kappa regressed {frozen_kappa:.6f} -> {now:.6f} "
                f"(drop {frozen_kappa - now:.6f} > tol {tol})"
            )

    assert not failures, "Baseline regression detected:\n" + "\n".join(failures)


def test_baseline_snapshot_is_well_formed() -> None:
    """The frozen baseline has the expected structure and gate metric."""
    baseline = _load_baseline()
    assert baseline["gate_metric"] == "cohen_kappa"
    assert baseline["conditions"], "baseline has no conditions"
    assert baseline["n_conditions"] == len(baseline["conditions"])
    for name, metrics in baseline["conditions"].items():
        assert metrics["cohen_kappa"] is not None, f"{name} has no frozen kappa"
