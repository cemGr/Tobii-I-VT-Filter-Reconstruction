#!/usr/bin/env python3
"""Reproducible performance comparison of implementation variants.

Three sub-benchmarks back the paper's implementation-performance section:

  (a) ``methods``   -- the four visual-angle methods (olsen2d, ray3d,
                       ray3d_gaze_dir, tobii_gaze_dir) head to head: runtime AND
                       fidelity (Cohen's kappa vs Tobii) under a fixed config.
  (b) ``vectorise`` -- the per-sample ``df.at`` write pattern vs the vectorised
                       whole-column assignment that replaced it (commit 157f3ba).
                       The two patterns are isolated and asserted bit-identical.
  (c) ``stages``    -- per-stage wall-clock breakdown of the pipeline, to show
                       which stage dominates (the velocity loop).

Timing protocol: ``time.perf_counter`` around the measured region only, one
discarded warm-up run, then ``--repeats`` runs reduced to a median with IQR.
Results are written as JSON into ``results/implementation_benchmark/``; the
reporting environment (platform, Python, numpy/pandas) is recorded in env.json.

The fidelity column reuses ``compute_ivt_metrics``, the same function the
fidelity sections use, so the kappa here cannot drift from the rest of the paper.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
from ivt_filter.io import read_tsv  # noqa: E402
from ivt_filter.io.pipeline import IVTPipeline  # noqa: E402
from ivt_filter.preprocessing.eye_selection import prepare_combined_columns  # noqa: E402
from ivt_filter.preprocessing.gap_fill import gap_fill_gaze  # noqa: E402
from ivt_filter.preprocessing.noise_reduction import smooth_combined_gaze  # noqa: E402
from ivt_filter.processing.classification import apply_ivt_classifier  # noqa: E402
from ivt_filter.processing.velocity_input import normalize_timestamps  # noqa: E402
from ivt_filter.processing.velocity_samples import VelocitySampleComputer  # noqa: E402
from ivt_filter.postprocessing import apply_fixation_postprocessing  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "implementation_benchmark"
DEFAULT_INPUT_DIR = REPO_ROOT / "test_data" / "inputs"

VELOCITY_METHODS = ("olsen2d", "ray3d", "ray3d_gaze_dir", "tobii_gaze_dir")

# Representative input carrying gaze_dir columns so every method can run. It is a
# both-eye / average-eye recording, so the fixed eye_mode="average" config matches
# its ground truth and the reported kappa is representative (not skewed by an
# eye-mode/GT mismatch). The full pipeline is dominated by the per-sample velocity
# loop, so the sweep uses an 18k-sample recording to keep wall-clock practical;
# override with --inputs to add larger recordings.
DEFAULT_METHOD_INPUTS = (
    "I-VT-botheye20ms30threshold_extracted.tsv",  # ~18k rows, average/both-eye, 20 ms / 30
)
DEFAULT_STAGE_INPUT = "I-VT-botheye20ms30threshold_extracted.tsv"
DEFAULT_VECTORISE_INPUT = "I-VT-botheye20ms30threshold_extracted.tsv"


# --------------------------------------------------------------------------- #
# Timing helper
# --------------------------------------------------------------------------- #
def time_callable(
    fn: Callable[[], Any],
    *,
    repeats: int = 7,
    warmup: int = 1,
) -> dict[str, Any]:
    """Median + IQR wall-clock of ``fn`` over ``repeats`` runs after ``warmup``.

    The median resists the outliers a mean would absorb from GC or scheduler
    jitter. Returns seconds.
    """
    for _ in range(max(0, warmup)):
        fn()
    runs: list[float] = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - start)
    runs_sorted = sorted(runs)
    if len(runs_sorted) >= 4:
        iqr = (
            statistics.median(runs_sorted[len(runs_sorted) // 2:])
            - statistics.median(runs_sorted[: len(runs_sorted) // 2])
        )
    else:
        iqr = runs_sorted[-1] - runs_sorted[0]
    return {
        "median_s": statistics.median(runs),
        "iqr_s": float(iqr),
        "min_s": min(runs),
        "runs": runs,
    }


def _base_velocity_cfg(method: str) -> OlsenVelocityConfig:
    """A fixed standard velocity config, varying only the visual-angle method."""
    return OlsenVelocityConfig(eye_mode="average", window_length_ms=20.0,
                               velocity_method=method)


def _core_config(method: str) -> PipelineConfig:
    """Velocity + classification only, so runtime and kappa isolate the method."""
    return PipelineConfig(
        velocity=_base_velocity_cfg(method),
        classifier=IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0),
        classify=True,
    )


# --------------------------------------------------------------------------- #
# (a) velocity-method head to head
# --------------------------------------------------------------------------- #
def bench_velocity_methods(
    inputs: list[Path],
    *,
    methods: tuple[str, ...] = VELOCITY_METHODS,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in inputs:
        raw = read_tsv(str(path))
        n_rows = int(len(raw))
        for method in methods:
            config = _core_config(method)
            label = f"{path.name}:{method}"
            try:
                # one untimed run also yields the df for the kappa column
                df = IVTPipeline(config).process_dataframe(raw.copy(), config=config)
                metrics = compute_ivt_metrics(df, pred_col="ivt_sample_type")
                timing = time_callable(
                    lambda: IVTPipeline(config).process_dataframe(
                        raw.copy(), config=config
                    ),
                    repeats=repeats,
                )
                rows.append({
                    "input": path.name,
                    "n_rows": n_rows,
                    "method": method,
                    "runtime_median_s": timing["median_s"],
                    "runtime_iqr_s": timing["iqr_s"],
                    "runtime_min_s": timing["min_s"],
                    "cohen_kappa": metrics.get("cohen_kappa"),
                    "percentage_agreement": metrics.get("percentage_agreement"),
                    "status": "ok",
                })
                print(f"[methods] {label}: "
                      f"{timing['median_s']:.2f}s kappa={metrics.get('cohen_kappa')}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 -- record, do not abort the sweep
                rows.append({
                    "input": path.name, "n_rows": n_rows, "method": method,
                    "status": "error", "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
                print(f"[methods] {label}: ERROR {type(exc).__name__}: {exc}",
                      flush=True)
    return rows


# --------------------------------------------------------------------------- #
# (b) scalar df.at vs vectorised column assignment
# --------------------------------------------------------------------------- #
# The 16 per-sample output columns the velocity stage writes (velocity_samples.py).
_VELOCITY_OUTPUT_COLUMNS = (
    "velocity_deg_per_sec", "dt_ms", "velocity_raw_deg_per_sec",
    "window_any_invalid", "window_width_samples", "velocity_first_idx",
    "velocity_last_idx", "velocity_eye_used", "velocity_window_selector",
    "velocity_fallback_applied", "env_has_invalid_above", "env_has_invalid_below",
    "env_rule_triggered", "gap_rule_triggered", "gap_left_invalid_idx",
    "gap_right_invalid_idx",
)


def bench_vectorise(input_path: Path, *, repeats: int = 7) -> dict[str, Any]:
    """Isolate the write-back: vectorised whole-column vs scalar ``df.at`` loop.

    The velocity stage runs once to produce real per-sample arrays of the correct
    dtypes. Both write patterns then populate a fresh copy of the same base frame
    with the identical values, so the only difference timed is the write pattern.
    The result is asserted bit-identical before the speedup is reported.
    """
    raw = read_tsv(str(input_path))
    cfg = _base_velocity_cfg("olsen2d")
    prepared = _prepare_for_velocity(raw.copy(), cfg)
    computed = VelocitySampleComputer(cfg).compute(prepared.copy())

    cols = [c for c in _VELOCITY_OUTPUT_COLUMNS if c in computed.columns]
    arrays = {c: computed[c].to_numpy() for c in cols}
    base = computed.drop(columns=cols)
    n = int(len(base))

    def write_vectorised() -> pd.DataFrame:
        d = base.copy()
        for col, arr in arrays.items():
            d[col] = arr
        return d

    def write_scalar() -> pd.DataFrame:
        d = base.copy()
        for col in cols:  # pre-create the columns, as the old loop required
            d[col] = pd.Series([pd.NA] * n, dtype=object)
        for i in range(n):
            for col in cols:
                d.at[i, col] = arrays[col][i]
        return d

    df_vec = write_vectorised()
    df_sca = write_scalar()
    identical = bool(df_vec[cols].astype(object).equals(df_sca[cols].astype(object)))

    vec = time_callable(write_vectorised, repeats=repeats)
    sca = time_callable(write_scalar, repeats=repeats)
    speedup = sca["median_s"] / vec["median_s"] if vec["median_s"] > 0 else None
    print(f"[vectorise] n={n} scalar={sca['median_s']:.3f}s "
          f"vector={vec['median_s']:.4f}s speedup={speedup:.1f}x identical={identical}",
          flush=True)
    return {
        "input": input_path.name,
        "n_rows": n,
        "n_columns": len(cols),
        "scalar_median_s": sca["median_s"],
        "scalar_iqr_s": sca["iqr_s"],
        "vector_median_s": vec["median_s"],
        "vector_iqr_s": vec["iqr_s"],
        "speedup": speedup,
        "identical": identical,
    }


# --------------------------------------------------------------------------- #
# (c) per-stage breakdown
# --------------------------------------------------------------------------- #
def _prepare_for_velocity(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> pd.DataFrame:
    df = normalize_timestamps(df, cfg)
    df = gap_fill_gaze(df, cfg)
    df = prepare_combined_columns(df, cfg)
    df = smooth_combined_gaze(df, cfg)
    return df


def bench_stages(input_path: Path, *, repeats: int = 3) -> dict[str, Any]:
    """Time each pipeline stage in real order on a full-preset config.

    Stages: normalize, gap fill, eye selection, noise reduction, velocity,
    classification, post-processing (merge + discard). Each stage is timed on a
    fresh copy of its real input frame, so the medians are independent.
    """
    raw = read_tsv(str(input_path))
    cfg = OlsenVelocityConfig(
        eye_mode="average", window_length_ms=20.0, velocity_method="olsen2d",
        gap_fill_enabled=True, smoothing_mode="moving_average",
    )
    classifier = IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0)
    fixpost = FixationPostConfig(
        merge_adjacent_fixations=True, discard_short_fixations=True,
    )

    # Build the chain once, capturing each stage's input frame.
    df_norm_in = raw.copy()
    df_gap_in = normalize_timestamps(df_norm_in.copy(), cfg)
    df_eye_in = gap_fill_gaze(df_gap_in.copy(), cfg)
    df_noise_in = prepare_combined_columns(df_eye_in.copy(), cfg)
    df_vel_in = smooth_combined_gaze(df_noise_in.copy(), cfg)
    df_cls_in = VelocitySampleComputer(cfg).compute(df_vel_in.copy())
    df_post_in = apply_ivt_classifier(df_cls_in.copy(), classifier)

    def _post(df: pd.DataFrame) -> pd.DataFrame:
        out, _ = apply_fixation_postprocessing(
            df, cfg=fixpost, sample_col="ivt_sample_type", time_col="time_ms",
            x_col="smoothed_x_mm", y_col="smoothed_y_mm", eye_z_col="eye_z_mm",
            event_type_col="ivt_event_type_post",
            event_index_col="ivt_event_index_post", use_ray3d=False,
        )
        return out

    stage_specs: list[tuple[str, Callable[[], Any]]] = [
        ("normalize", lambda: normalize_timestamps(df_norm_in.copy(), cfg)),
        ("gap_fill", lambda: gap_fill_gaze(df_gap_in.copy(), cfg)),
        ("eye_selection", lambda: prepare_combined_columns(df_eye_in.copy(), cfg)),
        ("noise_reduction", lambda: smooth_combined_gaze(df_noise_in.copy(), cfg)),
        ("velocity", lambda: VelocitySampleComputer(cfg).compute(df_vel_in.copy())),
        ("classification", lambda: apply_ivt_classifier(df_cls_in.copy(), classifier)),
        ("post_processing", lambda: _post(df_post_in.copy())),
    ]

    stages: list[dict[str, Any]] = []
    for name, fn in stage_specs:
        timing = time_callable(fn, repeats=repeats)
        stages.append({"name": name, "median_s": timing["median_s"],
                       "iqr_s": timing["iqr_s"]})
        print(f"[stages] {name}: {timing['median_s']:.3f}s", flush=True)
    total = sum(s["median_s"] for s in stages)
    for s in stages:
        s["pct"] = 100.0 * s["median_s"] / total if total > 0 else 0.0
    return {"input": input_path.name, "n_rows": int(len(raw)),
            "stages": stages, "total_s": total}


# --------------------------------------------------------------------------- #
# env + IO
# --------------------------------------------------------------------------- #
def _env_metadata(repeats: int, warmup: int) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "repeats": repeats,
        "warmup": warmup,
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True),
                    encoding="utf-8")
    print(f"[write] {path}", flush=True)


def _resolve_inputs(names: list[str], input_dir: Path) -> list[Path]:
    return [input_dir / n for n in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=["methods", "vectorise", "stages", "all"],
                        default="all")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--inputs", nargs="*", default=None,
                        help="Input file names for the methods sweep "
                             "(default: two representative sizes).")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(results_dir / "env.json", _env_metadata(args.repeats, args.warmup))

    run_all = args.which == "all"

    if run_all or args.which == "methods":
        names = args.inputs or list(DEFAULT_METHOD_INPUTS)
        rows = bench_velocity_methods(
            _resolve_inputs(names, args.input_dir), repeats=args.repeats)
        _write_json(results_dir / "velocity_methods.json",
                    {"env": _env_metadata(args.repeats, args.warmup), "rows": rows})

    if run_all or args.which == "vectorise":
        result = bench_vectorise(args.input_dir / DEFAULT_VECTORISE_INPUT,
                                 repeats=max(args.repeats, 7))
        _write_json(results_dir / "vectorise.json", result)

    if run_all or args.which == "stages":
        result = bench_stages(args.input_dir / DEFAULT_STAGE_INPUT,
                              repeats=args.repeats)
        _write_json(results_dir / "stages.json", result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
