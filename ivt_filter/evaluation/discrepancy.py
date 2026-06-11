"""Per-sample discrepancy attribution against the Tobii ground truth (WS-A).

The thesis diagnoses the residual disagreements between the reconstruction and
Tobii Pro Lab only in aggregate. This module assigns every disagreeing sample to
one cause, turning the aggregate diagnosis into per-sample evidence.

A disagreement is a sample where the predicted label differs from the ground
truth (both present). Five orthogonal mechanism predicates are evaluated per
disagreeing sample:

  * ``gap_edge``             within the Stage-4 half-window of a recording edge,
                            a gap the pipeline flagged, or an edge-fallback
                            velocity.
  * ``event_boundary``      within a couple of samples of a ground-truth event
                            onset/offset, i.e. an off-by-one timing difference at
                            a fixation/saccade boundary (the cascade).
  * ``eye_offset_avg``      inside a single-eye run under Average mode, where
                            Tobii falls back to the remaining eye.
  * ``near_threshold_fp``   velocity within a narrow band of the threshold, where
                            arithmetic-ordering differences can flip the label.
  * ``smoothing_attenuation`` smoothing is active and removing it moves the
                            sample across the threshold (the MA case).

Resolution: collect the mechanism set; two or more true gives ``ambiguous``,
exactly one gives that category. When no mechanism fires the disagreement is
characterised by velocity: a finite velocity well away from the threshold is a
``velocity_magnitude`` difference (the reconstruction confidently computes a
velocity on the opposite side of the threshold from Tobii's label), otherwise
``other``. Every disagreeing sample maps to exactly one category, so the
per-category counts sum to the total number of disagreements.

This evidence-based set extends the thesis's three aggregate causes: on the clean
conditions the dominant residual is event-boundary timing and confident
velocity-magnitude differences, not the threshold or smoothing effects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..config import OlsenVelocityConfig

# Mechanism predicates, in the fixed order used for single-cause resolution.
_MECHANISMS = (
    "gap_edge",
    "event_boundary",
    "eye_offset_avg",
    "near_threshold_fp",
    "smoothing_attenuation",
)
# Full category set: mechanisms, the velocity-magnitude fallback, then the
# overlap and unexplained buckets.
CATEGORIES = (*_MECHANISMS, "velocity_magnitude", "ambiguous", "other")
DISCREPANCY_CATEGORY = "discrepancy_category"
UNSMOOTHED_VELOCITY_COL = "velocity_unsmoothed_deg_per_sec"

# Ground-truth column auto-detection order, matching evaluation.compute_ivt_metrics.
_GT_CANDIDATES = ("gt_sample_type", "gt_event_type", "Eye movement type")
_AVERAGE_MODES = ("average", "strict_average")
_SINGLE_EYE_VALUES = ("left", "right")


@dataclass(frozen=True)
class AttributionParams:
    """Tunable thresholds for the attribution predicates.

    ``near_threshold_eps_deg`` defaults to 0.05 deg/s, half the 0.1 deg/s
    quantisation step at which velocities are stored, so the band is the
    tightest that still captures a single-step flip around the threshold. It is
    a parameter so the paper can report a sensitivity sweep (0.01 / 0.05 / 0.1).

    ``confident_margin_deg`` is the distance from the threshold beyond which an
    unexplained finite-velocity disagreement is a confident velocity-magnitude
    difference rather than borderline noise.
    """

    near_threshold_eps_deg: float = 0.05
    boundary_extra_samples: int = 0
    event_boundary_samples: int = 2
    min_single_eye_run: int = 3
    confident_margin_deg: float = 2.0
    smoothing_recheck: bool = True


def _detect_gt_col(df: pd.DataFrame, gt_col: Optional[str]) -> str:
    if gt_col is not None:
        if gt_col not in df.columns:
            raise ValueError(f"ground-truth column {gt_col!r} not in frame")
        return gt_col
    for candidate in _GT_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"no ground-truth column found; tried {_GT_CANDIDATES}")


def _nonempty(series: pd.Series) -> np.ndarray:
    """True where a label cell is present and not blank."""
    text = series.astype(str).str.strip()
    return (series.notna() & (text != "") & (text.str.lower() != "nan")).to_numpy()


def _median_dt_ms(time_ms: pd.Series) -> float:
    values = pd.to_numeric(time_ms, errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _half_window_samples(
    cfg: "OlsenVelocityConfig", df: pd.DataFrame, params: AttributionParams
) -> int:
    """Half-window width in samples for the gap-edge exclusion test."""
    if "window_width_samples" in df.columns:
        widths = pd.to_numeric(df["window_width_samples"], errors="coerce").to_numpy(dtype=float)
        widths = widths[np.isfinite(widths) & (widths > 0)]
        if widths.size:
            return max(1, int(round(float(np.median(widths)) / 2.0)))
    median_dt = _median_dt_ms(df["time_ms"]) if "time_ms" in df.columns else 1.0
    half_ms = float(cfg.window_length_ms) / 2.0
    return max(1, int(round(half_ms / median_dt)))


def _distance_to_positions(n: int, positions: np.ndarray) -> np.ndarray:
    """For each index 0..n-1, the distance to the nearest given position."""
    if positions.size == 0:
        return np.full(n, np.inf)
    positions = np.sort(positions)
    idx = np.arange(n)
    insert = np.searchsorted(positions, idx)
    left = np.clip(insert - 1, 0, positions.size - 1)
    right = np.clip(insert, 0, positions.size - 1)
    return np.minimum(np.abs(idx - positions[left]), np.abs(idx - positions[right])).astype(float)


def _run_mask(flags: np.ndarray, min_len: int) -> np.ndarray:
    """True where a position is inside a run of consecutive True of length >= min_len."""
    n = flags.size
    out = np.zeros(n, dtype=bool)
    start = 0
    while start < n:
        if not flags[start]:
            start += 1
            continue
        end = start
        while end < n and flags[end]:
            end += 1
        if end - start >= min_len:
            out[start:end] = True
        start = end
    return out


def _gap_edge_predicate(
    df: pd.DataFrame, n: int, half_window: int, params: AttributionParams
) -> np.ndarray:
    """Half-window exclusion near a recording edge or a pipeline-flagged gap.

    Proximity is measured to recording edges and to the windows the pipeline
    itself flagged as gap-affected (``gap_rule_triggered``), not to every invalid
    raw sample, which would paint a noisy recording entirely as a gap effect.
    Samples whose velocity came from the edge fallback are gap-edge by
    construction.
    """
    idx = np.arange(n)
    dist = np.minimum(idx, (n - 1) - idx).astype(float)
    if "gap_rule_triggered" in df.columns:
        gap = np.where(df["gap_rule_triggered"].fillna(False).astype(bool).to_numpy())[0]
        dist = np.minimum(dist, _distance_to_positions(n, gap))
    predicate = dist <= (half_window + params.boundary_extra_samples)
    if "velocity_fallback_applied" in df.columns:
        predicate = predicate | df["velocity_fallback_applied"].fillna(False).astype(bool).to_numpy()
    return predicate


def _event_boundary_predicate(
    gt_labels: np.ndarray, n: int, params: AttributionParams
) -> np.ndarray:
    """Within a few samples of a ground-truth event onset/offset.

    The off-by-one cascade appears at fixation/saccade boundaries: Tobii ends or
    starts an event one sample before or after the reconstruction. Proximity is
    measured to ground-truth transitions only (independent of the prediction),
    so the test is not circular with the disagreement itself.
    """
    if n == 0:
        return np.zeros(0, dtype=bool)
    change = np.zeros(n, dtype=bool)
    change[1:] = gt_labels[1:] != gt_labels[:-1]
    positions = np.where(change)[0]
    dist = _distance_to_positions(n, positions)
    return dist <= params.event_boundary_samples


def _eye_offset_predicate(
    df: pd.DataFrame, cfg: "OlsenVelocityConfig", n: int, params: AttributionParams
) -> np.ndarray:
    if cfg.eye_mode not in _AVERAGE_MODES or "velocity_eye_used" not in df.columns:
        return np.zeros(n, dtype=bool)
    used = df["velocity_eye_used"].astype(str).str.lower()
    single = used.isin(_SINGLE_EYE_VALUES).to_numpy()
    return _run_mask(single, params.min_single_eye_run)


def _velocity_array(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _unsmoothed_velocity(
    df: pd.DataFrame, cfg: "OlsenVelocityConfig"
) -> Optional[np.ndarray]:
    if UNSMOOTHED_VELOCITY_COL in df.columns:
        return _velocity_array(df, UNSMOOTHED_VELOCITY_COL)
    try:
        from dataclasses import replace

        from ..processing.velocity import compute_olsen_velocity

        cfg_unsmoothed = replace(cfg, smoothing_mode="none")
        recomputed = compute_olsen_velocity(df.copy(), cfg_unsmoothed)
        return _velocity_array(recomputed, "velocity_deg_per_sec")
    except Exception:
        return None


def _smoothing_predicate(
    df: pd.DataFrame,
    cfg: "OlsenVelocityConfig",
    n: int,
    threshold: float,
    params: AttributionParams,
) -> np.ndarray:
    if cfg.smoothing_mode == "none" or not params.smoothing_recheck:
        return np.zeros(n, dtype=bool)
    if "velocity_deg_per_sec" not in df.columns:
        return np.zeros(n, dtype=bool)
    v_unsmoothed = _unsmoothed_velocity(df, cfg)
    if v_unsmoothed is None:
        return np.zeros(n, dtype=bool)
    v_smoothed = _velocity_array(df, "velocity_deg_per_sec")
    finite = np.isfinite(v_smoothed) & np.isfinite(v_unsmoothed)
    return finite & ((v_smoothed >= threshold) != (v_unsmoothed >= threshold))


def _near_threshold_predicate(
    v: np.ndarray, threshold: float, params: AttributionParams
) -> np.ndarray:
    return np.isfinite(v) & (np.abs(v - threshold) < params.near_threshold_eps_deg)


def _velocity_magnitude_predicate(
    v: np.ndarray, threshold: float, params: AttributionParams
) -> np.ndarray:
    return np.isfinite(v) & (np.abs(v - threshold) >= params.confident_margin_deg)


def attribute_discrepancies(
    df: pd.DataFrame,
    cfg: "OlsenVelocityConfig",
    threshold: float,
    *,
    gt_col: Optional[str] = None,
    pred_col: str = "ivt_sample_type",
    params: AttributionParams = AttributionParams(),
) -> tuple[pd.DataFrame, dict]:
    """Attribute every disagreeing sample to one cause category.

    Returns ``(df_with_category, summary)``. The returned frame is a copy of
    ``df`` with an added :data:`DISCREPANCY_CATEGORY` column ("" for agreeing
    samples, one of :data:`CATEGORIES` for each disagreement). ``summary`` holds
    per-category counts that sum to the total number of disagreements.
    """
    if pred_col not in df.columns:
        raise ValueError(f"prediction column {pred_col!r} not in frame")
    resolved_gt = _detect_gt_col(df, gt_col)

    n = len(df)
    gt_text = df[resolved_gt].astype(str).to_numpy()
    pred_text = df[pred_col].astype(str).to_numpy()
    gt_present = _nonempty(df[resolved_gt])
    pred_present = _nonempty(df[pred_col])
    disagree = gt_present & pred_present & (gt_text != pred_text)

    velocity = (
        _velocity_array(df, "velocity_deg_per_sec")
        if "velocity_deg_per_sec" in df.columns
        else np.full(n, np.nan)
    )
    half_window = _half_window_samples(cfg, df, params)
    mechanisms = np.vstack(
        [
            _gap_edge_predicate(df, n, half_window, params),
            _event_boundary_predicate(gt_text, n, params),
            _eye_offset_predicate(df, cfg, n, params),
            _near_threshold_predicate(velocity, threshold, params),
            _smoothing_predicate(df, cfg, n, threshold, params),
        ]
    )
    confident = _velocity_magnitude_predicate(velocity, threshold, params)

    n_true = mechanisms.sum(axis=0)
    which = mechanisms.argmax(axis=0)
    names = np.array(_MECHANISMS, dtype=object)

    resolved = np.where(confident, "velocity_magnitude", "other").astype(object)
    single = n_true == 1
    resolved[single] = names[which[single]]
    resolved[n_true >= 2] = "ambiguous"

    category = np.full(n, "", dtype=object)
    category[disagree] = resolved[disagree]

    out = df.copy()
    out[DISCREPANCY_CATEGORY] = category

    counts = {cat: int(np.count_nonzero(category == cat)) for cat in CATEGORIES}
    n_disagreements = int(disagree.sum())
    fraction = {
        cat: (counts[cat] / n_disagreements if n_disagreements else 0.0) for cat in CATEGORIES
    }
    summary = {
        "n_total": n,
        "n_disagreements": n_disagreements,
        "counts": counts,
        "fraction": fraction,
        "params": asdict(params),
        "threshold": float(threshold),
        "half_window_samples": half_window,
        "gt_col": resolved_gt,
        "pred_col": pred_col,
        "eye_mode": cfg.eye_mode,
        "smoothing_mode": cfg.smoothing_mode,
    }
    return out, summary


def format_attribution_report(summary: dict) -> str:
    """Human-readable breakdown of an attribution summary."""
    lines = [
        "Discrepancy attribution",
        f"  threshold: {summary['threshold']:.3f} deg/s   "
        f"eye_mode: {summary.get('eye_mode')}   smoothing: {summary.get('smoothing_mode')}",
        f"  samples: {summary['n_total']}   disagreements: {summary['n_disagreements']}",
        "  category                 count    fraction",
    ]
    for cat in CATEGORIES:
        count = summary["counts"][cat]
        frac = summary["fraction"][cat]
        lines.append(f"  {cat:22s} {count:7d}   {frac * 100:6.2f}%")
    return "\n".join(lines)
