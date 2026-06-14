"""Generate the implementation-performance figures for the paper.

Reads the JSON written by ``scripts/benchmark_implementations.py`` and renders:

  * perf_velocity_methods.{png,pdf} -- runtime and fidelity (kappa) of the four
    visual-angle methods, one figure with two panels.
  * perf_stage_breakdown.{png,pdf}  -- per-stage share of pipeline wall-clock.

Figures are written into ``IVT-Paper/figures/`` as both PNG (preview) and PDF
(vector, for the camera-ready include).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not available. Install with: pip install matplotlib")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "implementation_benchmark"
PAPER_FIG_DIR = REPO_ROOT.parent / "IVT-Paper" / "figures"

METHOD_LABELS = {
    "olsen2d": "Olsen 2D",
    "ray3d": "Ray 3D",
    "ray3d_gaze_dir": "Ray 3D\n(gaze dir)",
    "tobii_gaze_dir": "Tobii\n(gaze dir)",
}
METHOD_ORDER = ["olsen2d", "ray3d", "ray3d_gaze_dir", "tobii_gaze_dir"]
STAGE_LABELS = {
    "normalize": "Normalise",
    "gap_fill": "Gap fill",
    "eye_selection": "Eye selection",
    "noise_reduction": "Noise reduction",
    "velocity": "Velocity",
    "classification": "Classification",
    "post_processing": "Post-processing",
}


def _save(fig: plt.Figure, name: str) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = PAPER_FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"[figure] {path}")
    plt.close(fig)


def fig_velocity_methods() -> None:
    data = json.loads((RESULTS_DIR / "velocity_methods.json").read_text())
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    if not rows:
        print("[figure] no ok rows in velocity_methods.json; skipping")
        return
    sizes = sorted({r["n_rows"] for r in rows})
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]

    def value(metric: str, method: str, n_rows: int):
        for r in rows:
            if r["method"] == method and r["n_rows"] == n_rows:
                return r.get(metric)
        return None

    x = range(len(methods))
    width = 0.8 / max(1, len(sizes))
    fig, (ax_rt, ax_k) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    for si, n_rows in enumerate(sizes):
        offs = [xi + (si - (len(sizes) - 1) / 2) * width for xi in x]
        rts = [value("runtime_median_s", m, n_rows) or 0.0 for m in methods]
        ks = [value("cohen_kappa", m, n_rows) or 0.0 for m in methods]
        label = f"{n_rows // 1000}k rows"
        ax_rt.bar(offs, rts, width=width, label=label)
        ax_k.bar(offs, ks, width=width, label=label)

    for ax, ylabel, title in (
        (ax_rt, "runtime (s, median)", "(a) Runtime"),
        (ax_k, r"Cohen's $\kappa$ vs Tobii", "(b) Fidelity"),
    ):
        ax.set_xticks(list(x))
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax_k.set_ylim(0.0, 1.0)
    ax_rt.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "perf_velocity_methods")


def fig_stage_breakdown() -> None:
    data = json.loads((RESULTS_DIR / "stages.json").read_text())
    stages = data["stages"]
    names = [s["name"] for s in stages]
    pcts = [s["pct"] for s in stages]
    labels = [STAGE_LABELS.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    left = 0.0
    cmap = plt.get_cmap("tab10")
    for i, (label, pct) in enumerate(zip(labels, pcts)):
        ax.barh(0, pct, left=left, color=cmap(i % 10), edgecolor="white", label=label)
        if pct >= 5.0:
            ax.text(left + pct / 2, 0, f"{pct:.0f}%", ha="center", va="center",
                    fontsize=8, color="white")
        left += pct
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("share of pipeline wall-clock (%)", fontsize=9)
    n_rows = data.get("n_rows")
    ax.set_title(f"Per-stage runtime breakdown ({n_rows // 1000}k samples)"
                 if n_rows else "Per-stage runtime breakdown", fontsize=10)
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35),
              frameon=False)
    fig.tight_layout()
    _save(fig, "perf_stage_breakdown")


def write_latex_macros() -> None:
    """Emit figures/perf_values.tex so every table number regenerates from data.

    The methods table reports the largest recording in the sweep. Macros are
    consumed by sections/06b-performance.tex via \\input in the preamble.
    """
    methods = json.loads((RESULTS_DIR / "velocity_methods.json").read_text())["rows"]
    ok = [r for r in methods if r.get("status") == "ok"]
    vec = json.loads((RESULTS_DIR / "vectorise.json").read_text())

    lines: list[str] = ["% Auto-generated by scripts/generate_performance_figures.py.",
                        "% Do not edit by hand; regenerate with `make perf`."]

    macro_keys = {
        "olsen2d": "Olsen", "ray3d": "Ray",
        "ray3d_gaze_dir": "Rgd", "tobii_gaze_dir": "Tgd",
    }
    if ok:
        n_table = max(r["n_rows"] for r in ok)
        lines.append(f"% methods table recording: {n_table} rows")
        for method, key in macro_keys.items():
            row = next((r for r in ok if r["method"] == method
                        and r["n_rows"] == n_table), None)
            rt = f"{row['runtime_median_s']:.2f}" if row else "{-}"
            k = f"{row['cohen_kappa']:.4f}" if row and row.get("cohen_kappa") is not None else "{-}"
            ag = f"{row['percentage_agreement']:.2f}" if row and row.get("percentage_agreement") is not None else "{-}"
            lines.append(f"\\newcommand{{\\perfMethod{key}Rt}}{{{rt}}}")
            lines.append(f"\\newcommand{{\\perfMethod{key}K}}{{{k}}}")
            lines.append(f"\\newcommand{{\\perfMethod{key}A}}{{{ag}}}")

    lines.append(f"\\newcommand{{\\perfVecScalar}}{{{vec['scalar_median_s']:.2f}}}")
    lines.append(f"\\newcommand{{\\perfVecVector}}{{{vec['vector_median_s']:.2f}}}")
    speedup = vec.get("speedup")
    lines.append(f"\\newcommand{{\\perfVecSpeedup}}{{{speedup:.1f}}}"
                 if speedup else "\\newcommand{\\perfVecSpeedup}{{-}}")

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIG_DIR / "perf_values.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[macros] {out}")


def main() -> int:
    fig_velocity_methods()
    fig_stage_breakdown()
    write_latex_macros()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
