#!/usr/bin/env python3
"""Per-component fidelity of the reconstruction vs Tobii (Phase 1).

The 14-condition agreement and the per-sample attribution both mix all filter
stages together. This script isolates *one component at a time* using minimal
pairs: the same recording exported by Tobii with exactly one setting changed
(baseline B vs variant V). It then asks how faithfully the reconstruction
reproduces Tobii's marginal effect of that component.

Metric (B and V aligned on ``time_ms``; both pipeline configs run on B's raw gaze
so only the component differs):

  T_delta = samples where Tobii's label changes between V and B
  M_delta = samples where our label changes between config V and config B
  K_dir   = co-located changes that land on the same destination label
            (direction-faithful); K_colo = plain co-location
  recall = K / |T_delta|, precision = K / |M_delta|, jaccard = K / |T_delta u M_delta|

The primary basis is the dense study_a1 dose-response sweep (same A1 recording);
the seven test_data/inputs pairs add a cross-recording replication.

Outputs (results/component_fidelity/):
  component_fidelity.csv        one row per variant (the table)
  component_fidelity_summary.json  per-variant + per-component pooled metrics
  component_fidelity_summary.txt   human-readable report
  component_fidelity_curves.png    optional recall-vs-dose curves (--plot)

Examples:
  python scripts/component_fidelity.py
  python scripts/component_fidelity.py --max-rows 4000 --plot
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import benchmark_filter_settings as bench  # noqa: E402
from ivt_filter.io.io import read_tsv  # noqa: E402
from ivt_filter.io.pipeline import IVTPipeline  # noqa: E402

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "component_fidelity"
STUDY_A1_DIR = REPO_ROOT / "test_data" / "study_a1"
INPUTS_DIR = REPO_ROOT / "test_data" / "inputs"

GT_COL = "gt_event_type"
TIME_COL = "time_ms"

# Components whose marginal effect drifts with the numerical environment; flagged
# in the summary so readers do not over-interpret their absolute fidelity.
ENV_DRIFT_COMPONENTS = {"smoothing"}

# --------------------------------------------------------------------------- #
# Minimal-pair specification                                                  #
# --------------------------------------------------------------------------- #
STUDY_A1_BASELINE = "A1averageWindow20T30.tsv"

# component -> [(variant_filename, dose_label), ...] (one setting changed vs baseline)
STUDY_A1_FAMILIES: dict[str, list[tuple[str, str]]] = {
    "smoothing": [
        ("A1averageWindow20T30SmoothMed3.tsv", "median3"),
        ("A1averageWindow20T30SmoothMed5.tsv", "median5"),
        ("A1averageWindow20T30SmoothMed9.tsv", "median9"),
        ("A1averageWindow20T30SmoothAvg3.tsv", "average3"),
        ("A1averageWindow20T30SmoothAvg5.tsv", "average5"),
        ("A1averageWindow20T30SmoothAvg9.tsv", "average9"),
    ],
    "merge": [
        ("A1averageWindow20T30MergeFix25ms0_5.tsv", "25ms_0.5deg"),
        ("A1averageWindow20T30MergeFix75ms0_5.tsv", "75ms_0.5deg"),
        ("A1averageWindow20T30MergeFix150ms0_5.tsv", "150ms_0.5deg"),
        ("A1averageWindow20T30MergeFix75ms1.tsv", "75ms_1deg"),
        ("A1averageWindow20T30MergeFix75ms2.tsv", "75ms_2deg"),
        ("A1averageWindow20T30MergeFix75ms5.tsv", "75ms_5deg"),
    ],
    "discard": [
        ("A1averageWindow20T30Discard40ms.tsv", "40ms"),
        ("A1averageWindow20T30Discard60ms.tsv", "60ms"),
        ("A1averageWindow20T30Discard100ms.tsv", "100ms"),
        ("A1averageWindow20T30Discard140ms.tsv", "140ms"),
    ],
    "gap_fill": [
        ("A1averageWindow20T30Interpolation25ms.tsv", "25ms"),
        ("A1averageWindow20T30Interpolation75ms.tsv", "75ms"),
        ("A1averageWindow20T30Interpolation100ms.tsv", "100ms"),
        ("A1averageWindow20T30Interpolation150ms.tsv", "150ms"),
        ("A1averageWindow20T30Interpolation250ms.tsv", "250ms"),
    ],
    "window": [
        ("A1averageWindow4T30.tsv", "4ms"),
        ("A1averageWindow8T30.tsv", "8ms"),
        ("A1AverageWindow10T30.tsv", "10ms"),
        ("A1averageWindow12T30.tsv", "12ms"),
        ("A1averageWindow16T30.tsv", "16ms"),
        ("A1averageWindow32T30.tsv", "32ms"),
        ("A1AverageWindow50T30.tsv", "50ms"),
        ("A1averageWindow52T30.tsv", "52ms"),
    ],
    "threshold": [
        ("A1averageWindow20T50.tsv", "50dps"),
        ("A1averageWindow20T75.tsv", "75dps"),
        ("A1averageWindow20T100.tsv", "100dps"),
    ],
    "eye": [
        ("A1EyeLeftWindow20T30.tsv", "left"),
        ("A1RightWindow20T30.tsv", "right"),
        ("A1StrictAverageWindow20T30.tsv", "strict_average"),
    ],
}

# Cross-recording replication from test_data/inputs:
# (component, dose, baseline_file, variant_file)
INPUT_PAIRS: list[tuple[str, str, str, str]] = [
    ("smoothing", "median9", "LeftV30W20_extracted.tsv", "LeftV30W20NoiseMedW9_extracted.tsv"),
    ("smoothing", "average9", "LeftV30W20_extracted.tsv", "LeftV30W20NoiseAvW9_extracted.tsv"),
    ("merge", "75ms_0.5deg", "LeftV30W20_extracted.tsv", "LeftV30W20MergeF75ms0-5angle_extracted.tsv"),
    ("discard", "60ms", "LeftV30W20_extracted.tsv", "LeftV30W20Discard60NoSaccade_extracted.tsv"),
    ("window", "1ms", "LeftV30W20_extracted.tsv", "LeftV30W1_extracted.tsv"),
    ("smoothing", "average3", "IVT30_extracted.tsv", "IVT-Eyeboth-NoiseAv3-Win20-V30_extracted.tsv"),
    ("gap_fill", "75ms", "IVT30_extracted.tsv", "IVT-Interp75-EyeBoth-NoNoise-W20-V30.tsv"),
]


@dataclass
class PairSpec:
    component: str
    dose: str
    baseline: str
    variant: str
    source: str  # "study_a1" or "inputs"


def build_pair_specs() -> list[PairSpec]:
    specs: list[PairSpec] = []
    for component, variants in STUDY_A1_FAMILIES.items():
        for variant, dose in variants:
            specs.append(PairSpec(component, dose, STUDY_A1_BASELINE, variant, "study_a1"))
    for component, dose, base, variant in INPUT_PAIRS:
        specs.append(PairSpec(component, dose, base, variant, "inputs"))
    return specs


# --------------------------------------------------------------------------- #
# Core metric (pure)                                                          #
# --------------------------------------------------------------------------- #
def _ratio(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def fidelity_metrics(
    t_delta: Sequence[bool],
    m_delta: Sequence[bool],
    dir_ok: Optional[Sequence[bool]] = None,
) -> dict[str, Any]:
    """Recall/precision/Jaccard of M_delta reproducing T_delta.

    ``dir_ok[i]`` marks samples where our changed label lands on the same
    destination as Tobii's (direction-faithful). When omitted, K_dir collapses to
    plain co-location. Recall is ``None`` when T_delta is empty (undefined, not a
    silent zero); precision is ``None`` when M_delta is empty.
    """
    t = np.asarray(t_delta, dtype=bool)
    m = np.asarray(m_delta, dtype=bool)
    inter = t & m
    k_colo = int(inter.sum())
    if dir_ok is None:
        k_dir = k_colo
    else:
        d = np.asarray(dir_ok, dtype=bool)
        k_dir = int((inter & d).sum())
    t_count = int(t.sum())
    m_count = int(m.sum())
    union = int((t | m).sum())
    return {
        "t_delta": t_count,
        "m_delta": m_count,
        "k_dir": k_dir,
        "k_colo": k_colo,
        "recall_dir": _ratio(k_dir, t_count),
        "precision_dir": _ratio(k_dir, m_count),
        "jaccard_dir": _ratio(k_dir, union),
        "recall_colo": _ratio(k_colo, t_count),
        "precision_colo": _ratio(k_colo, m_count),
        "jaccard_colo": _ratio(k_colo, union),
    }


# --------------------------------------------------------------------------- #
# Sample alignment                                                            #
# --------------------------------------------------------------------------- #
def align_pair(
    df_b: pd.DataFrame,
    df_v: pd.DataFrame,
    *,
    time_col: str = TIME_COL,
    round_decimals: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Align two minimal-pair frames on ``time_ms``.

    Fast path: equal length and element-wise equal timestamps -> identity. Else
    intersect on rounded timestamps (same recording, same rate) and keep the
    common samples in order. Dropped counts are reported, never silently lost.
    """
    n_b, n_v = len(df_b), len(df_v)
    if (
        n_b == n_v
        and n_b > 0
        and np.allclose(df_b[time_col].to_numpy(), df_v[time_col].to_numpy())
    ):
        info = {
            "n_b": n_b,
            "n_v": n_v,
            "time_match": True,
            "n_aligned": n_b,
            "n_dropped_b": 0,
            "n_dropped_v": 0,
            "method": "identity",
        }
        return df_b.reset_index(drop=True), df_v.reset_index(drop=True), info

    key = "__t_key__"
    b = df_b.copy()
    v = df_v.copy()
    b[key] = b[time_col].round(round_decimals)
    v[key] = v[time_col].round(round_decimals)
    b = b.drop_duplicates(key)
    v = v.drop_duplicates(key)
    common = pd.Index(b[key]).intersection(pd.Index(v[key]))
    b_aligned = b.set_index(key).loc[common].reset_index(drop=True)
    v_aligned = v.set_index(key).loc[common].reset_index(drop=True)
    info = {
        "n_b": n_b,
        "n_v": n_v,
        "time_match": False,
        "n_aligned": int(len(common)),
        "n_dropped_b": int(n_b - len(common)),
        "n_dropped_v": int(n_v - len(common)),
        "method": "rounded_time_intersection",
    }
    return b_aligned, v_aligned, info


# --------------------------------------------------------------------------- #
# Config resolution + one-pair computation                                    #
# --------------------------------------------------------------------------- #
def resolve_config(name: str, path: Path, explicit_configs: dict[str, Any]):
    """Same resolution order as benchmark run_benchmark: explicit then inferred."""
    if name in explicit_configs:
        return bench.build_pipeline_config_from_entry(explicit_configs[name], path)
    return bench.infer_pipeline_config(path)


def _predict(raw: pd.DataFrame, config) -> np.ndarray:
    df = IVTPipeline(config).process_dataframe(raw.copy(), config=config)
    return df[bench.prediction_column(config)].to_numpy()


def _apply_merge_absorb(config) -> None:
    """Opt-in: enable Tobii-faithful saccade absorption on a merge config."""
    fp = getattr(config, "fixation_post", None)
    if fp is not None and getattr(fp, "merge_adjacent_fixations", False):
        fp.merge_absorb_saccades = True


def compute_pair(
    spec: PairSpec,
    data_dir: Path,
    explicit_configs: dict[str, Any],
    max_rows: Optional[int] = None,
    baseline_cache: Optional[dict[tuple, np.ndarray]] = None,
    merge_absorb_saccades: bool = False,
) -> dict[str, Any]:
    base_path = data_dir / spec.baseline
    var_path = data_dir / spec.variant
    row: dict[str, Any] = {
        "component": spec.component,
        "dose": spec.dose,
        "source": spec.source,
        "baseline": spec.baseline,
        "variant": spec.variant,
        "env_drift_flag": spec.component in ENV_DRIFT_COMPONENTS,
    }
    if not base_path.exists() or not var_path.exists():
        row["status"] = "missing"
        return row

    df_b = read_tsv(str(base_path))
    df_v = read_tsv(str(var_path))
    if max_rows is not None:
        df_b = df_b.head(max_rows).copy()
        df_v = df_v.head(max_rows).copy()

    ab, av, align_info = align_pair(df_b, df_v)
    row["alignment"] = align_info
    if align_info["n_aligned"] == 0:
        row["status"] = "skipped"
        return row

    cfg_b, _ = resolve_config(spec.baseline, base_path, explicit_configs)
    cfg_v, _ = resolve_config(spec.variant, var_path, explicit_configs)
    if merge_absorb_saccades:
        _apply_merge_absorb(cfg_v)
        row["merge_absorb_saccades"] = True

    raw = ab  # B's raw gaze, aligned to the common samples
    # The baseline prediction is identical across every variant that shares the
    # same baseline file and the same aligned index (the study_a1 sweep is fully
    # sample-identical, so alignment is "identity" and the key is stable). Cache
    # it so the heavy pipeline runs once per baseline instead of once per pair.
    cache_key = (spec.source, spec.baseline, align_info["method"], len(ab), max_rows)
    if baseline_cache is not None and cache_key in baseline_cache:
        pred_b = baseline_cache[cache_key]
    else:
        pred_b = _predict(raw, cfg_b)
        if baseline_cache is not None:
            baseline_cache[cache_key] = pred_b
    pred_v = _predict(raw, cfg_v)

    gt_b = ab[GT_COL].to_numpy()
    gt_v = av[GT_COL].to_numpy()
    t_delta = gt_v != gt_b
    m_delta = pred_v != pred_b
    dir_ok = pred_v == gt_v

    row.update(fidelity_metrics(t_delta, m_delta, dir_ok))
    row["status"] = "ok"
    return row


# --------------------------------------------------------------------------- #
# Aggregation + reporting                                                     #
# --------------------------------------------------------------------------- #
def pool_component(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool counts across all ok variants of a component (micro-average)."""
    ok = [r for r in rows if r.get("status") == "ok"]
    t = sum(r["t_delta"] for r in ok)
    m = sum(r["m_delta"] for r in ok)
    k_dir = sum(r["k_dir"] for r in ok)
    k_colo = sum(r["k_colo"] for r in ok)
    # union is not additive; approximate pooled union as t + m - k_colo
    union = t + m - k_colo
    return {
        "n_variants": len(ok),
        "t_delta": t,
        "m_delta": m,
        "k_dir": k_dir,
        "k_colo": k_colo,
        "recall_dir": _ratio(k_dir, t),
        "precision_dir": _ratio(k_dir, m),
        "jaccard_dir": _ratio(k_dir, union),
        "recall_colo": _ratio(k_colo, t),
        "precision_colo": _ratio(k_colo, m),
    }


def _fmt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def format_fidelity_report(rows: list[dict[str, Any]], pooled: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("Component-fidelity (direction-faithful recall = how well we")
    lines.append("reproduce Tobii's marginal effect of each component)\n")
    header = f"{'component':12} {'dose':14} {'src':8} {'|Tdelta|':>9} {'|Mdelta|':>9} {'recall':>7} {'prec':>6} {'jacc':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"{r['component']:12} {r['dose']:14} {r['source']:8} -- {r.get('status')}")
            continue
        lines.append(
            f"{r['component']:12} {r['dose']:14} {r['source']:8} "
            f"{r['t_delta']:9d} {r['m_delta']:9d} "
            f"{_fmt(r['recall_dir']):>7} {_fmt(r['precision_dir']):>6} {_fmt(r['jaccard_dir']):>6}"
        )
    lines.append("\n== pooled per component (micro-averaged) ==")
    for comp, agg in pooled.items():
        drift = "  [env-drift]" if comp in ENV_DRIFT_COMPONENTS else ""
        lines.append(
            f"{comp:12} n={agg['n_variants']:2d}  recall_dir={_fmt(agg['recall_dir'])}  "
            f"precision_dir={_fmt(agg['precision_dir'])}  "
            f"(colo recall={_fmt(agg['recall_colo'])}){drift}"
        )
    return "\n".join(lines)


def maybe_plot(rows: list[dict[str, Any]], out_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    ok = [r for r in rows if r.get("status") == "ok" and r["source"] == "study_a1"]
    components = sorted({r["component"] for r in ok})
    fig, ax = plt.subplots(figsize=(max(8, len(components) * 1.2), 5))
    for comp in components:
        variants = [r for r in ok if r["component"] == comp]
        xs = list(range(len(variants)))
        ys = [r["recall_dir"] if r["recall_dir"] is not None else np.nan for r in variants]
        ax.plot(xs, ys, marker="o", label=comp)
    ax.set_ylabel("direction-faithful recall")
    ax.set_xlabel("dose index (within component)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Component fidelity: recall of Tobii's marginal effect (study_a1)")
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run(
    specs: list[PairSpec],
    *,
    max_rows: Optional[int] = None,
    configs_by_source: Optional[dict[str, dict[str, Any]]] = None,
    merge_absorb_saccades: bool = False,
) -> dict[str, Any]:
    if configs_by_source is None:
        configs_by_source = {
            "study_a1": bench.load_file_configs(STUDY_A1_DIR / "configs.json").get("files", {}),
            "inputs": bench.load_file_configs(INPUTS_DIR / "configs.json").get("files", {}),
        }
    dir_by_source = {"study_a1": STUDY_A1_DIR, "inputs": INPUTS_DIR}

    baseline_cache: dict[tuple, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[component_fidelity] {spec.component}/{spec.dose} ({spec.source}) {spec.variant}", flush=True)
        row = compute_pair(
            spec,
            dir_by_source[spec.source],
            configs_by_source.get(spec.source, {}),
            max_rows=max_rows,
            baseline_cache=baseline_cache,
            merge_absorb_saccades=merge_absorb_saccades,
        )
        rows.append(row)

    pooled = {}
    for comp in {r["component"] for r in rows}:
        pooled[comp] = pool_component([r for r in rows if r["component"] == comp])
    return {"variants": rows, "pooled": pooled}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--max-rows", type=int, default=None, help="truncate each file (quick runs)")
    parser.add_argument("--component", nargs="*", default=None, help="restrict to these components")
    parser.add_argument("--source", choices=["study_a1", "inputs"], default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--merge-absorb-saccades",
        action="store_true",
        help="opt-in: enable the Tobii-faithful merge (absorb the inter-fixation "
        "saccade) to A/B the flag-gated fix against the default capped merge",
    )
    args = parser.parse_args(argv)

    specs = build_pair_specs()
    if args.component:
        specs = [s for s in specs if s.component in set(args.component)]
    if args.source:
        specs = [s for s in specs if s.source == args.source]
    if not specs:
        print("[component_fidelity] no pairs match the filters", file=sys.stderr)
        return 1

    result = run(specs, max_rows=args.max_rows, merge_absorb_saccades=args.merge_absorb_saccades)
    rows = result["variants"]
    pooled = result["pooled"]

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # CSV: one row per variant (the table)
    csv_cols = [
        "component", "dose", "source", "baseline", "variant", "status",
        "env_drift_flag", "t_delta", "m_delta", "k_dir", "k_colo",
        "recall_dir", "precision_dir", "jaccard_dir", "recall_colo", "precision_colo",
    ]
    csv_rows = [{c: r.get(c) for c in csv_cols} for r in rows]
    pd.DataFrame(csv_rows, columns=csv_cols).to_csv(results_dir / "component_fidelity.csv", index=False)

    payload = _json_safe({"variants": rows, "pooled": pooled})
    (results_dir / "component_fidelity_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (results_dir / "component_fidelity_summary.txt").write_text(
        format_fidelity_report(rows, pooled) + "\n", encoding="utf-8"
    )

    if args.plot:
        if maybe_plot(rows, results_dir / "component_fidelity_curves.png"):
            print("[component_fidelity] wrote component_fidelity_curves.png")
        else:
            print("[component_fidelity] matplotlib unavailable; skipped plot", file=sys.stderr)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[component_fidelity] {n_ok}/{len(rows)} pairs ok -> {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
