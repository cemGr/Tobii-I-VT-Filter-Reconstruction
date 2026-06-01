"""
Maximum IoU Event-Level Evaluation for I-VT Filter output.

Implements the event matching and timing evaluation approach recommended by:
    Startsev, M., & Zemblys, R. (2022).
    Evaluating eye movement event detection: A review of the state of the art.
    Behavior Research Methods, 54, 1653–1714.
    https://doi.org/10.3758/s13428-021-01763-7

Algorithm summary (Section 3.5 / Table 2):
    1. Extract contiguous event runs from the sample-level label sequence.
    2. Compute Intersection-over-Union (IoU) for all (GT, Pred) event pairs.
    3. Sort pairs by IoU descending.
    4. Greedily match: if both the GT event and the Pred event are still
       unmatched, assign them as a matched pair.
    5. Unmatched GT events → False Negatives (FN).
       Unmatched Pred events → False Positives (FP).
    6. Report an event-level confusion matrix and timing quality measures
       (onset / offset deviation) for correctly matched event types.

Rationale for this approach over simpler "run-based agreement":
    - Majority Voting (the run-based approach) is over-optimistic and insensitive
      to fragmentation (one wrong sample splits a ground-truth event into 3 runs).
    - Maximum IoU matching is a proper 1-to-1 assignment, robust to fragmentation
      and merging, and allows informative timing statistics per event class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """A single eye-movement event (contiguous run of the same label).

    Attributes
    ----------
    event_type:     Class label (e.g. "Fixation", "Saccade", "EyesNotFound").
    start_idx:      0-based positional index of the first sample (inclusive).
    end_idx:        0-based positional index of the last sample (inclusive).
    start_time_ms:  Timestamp of the first sample in milliseconds.
    end_time_ms:    Timestamp of the last sample in milliseconds.
    n_samples:      Number of samples = end_idx - start_idx + 1.
    """

    event_type: str
    start_idx: int
    end_idx: int
    start_time_ms: float
    end_time_ms: float
    n_samples: int


@dataclass
class MatchResult:
    """Result of matching a single GT event to a Pred event.

    Attributes
    ----------
    gt_event:        The ground-truth event.
    pred_event:      The matched predicted event, or None if unmatched (FN).
    iou:             Intersection-over-Union of the matched pair (0 if FN).
    onset_dev_ms:    pred.start_time_ms - gt.start_time_ms  (NaN if FN).
                     Positive  → pred starts *later*  than GT.
                     Negative  → pred starts *earlier* than GT.
    offset_dev_ms:   pred.end_time_ms - gt.end_time_ms  (NaN if FN).
                     Positive  → pred ends *later*   than GT.
                     Negative  → pred ends *earlier*  than GT.
    is_correct_type: True when gt_event.event_type == pred_event.event_type.
    """

    gt_event: Event
    pred_event: Optional[Event]
    iou: float
    onset_dev_ms: float
    offset_dev_ms: float
    is_correct_type: bool


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def extract_events(
    sample_types: pd.Series,
    times: pd.Series,
) -> List[Event]:
    """Extract contiguous event runs from a sample-level label sequence.

    Parameters
    ----------
    sample_types:   Series of string labels, one per sample.
    times:          Series of timestamps (ms), aligned with sample_types.

    Returns
    -------
    List of :class:`Event` objects in chronological order.
    An empty Series returns an empty list.
    """
    if len(sample_types) == 0:
        return []

    # Reindex so we can use positional access safely
    types = sample_types.reset_index(drop=True)
    t = times.reset_index(drop=True)

    events: List[Event] = []
    current_type: str = str(types.iloc[0])
    start_pos: int = 0

    for i in range(1, len(types)):
        lbl = str(types.iloc[i])
        if lbl != current_type:
            events.append(
                Event(
                    event_type=current_type,
                    start_idx=start_pos,
                    end_idx=i - 1,
                    start_time_ms=float(t.iloc[start_pos]),
                    end_time_ms=float(t.iloc[i - 1]),
                    n_samples=i - start_pos,
                )
            )
            current_type = lbl
            start_pos = i

    # Append the final (possibly only) event
    events.append(
        Event(
            event_type=current_type,
            start_idx=start_pos,
            end_idx=len(types) - 1,
            start_time_ms=float(t.iloc[start_pos]),
            end_time_ms=float(t.iloc[-1]),
            n_samples=len(types) - start_pos,
        )
    )
    return events


def compute_iou(e1: Event, e2: Event) -> float:
    """Compute Intersection-over-Union between two events by sample index.

    Overlap and union are measured in numbers of samples (not time), so the
    result is robust to irregular sampling rates.

    Returns
    -------
    Float in [0, 1].  0.0 if there is no overlap.
    """
    overlap_start = max(e1.start_idx, e2.start_idx)
    overlap_end = min(e1.end_idx, e2.end_idx)
    overlap = max(0, overlap_end - overlap_start + 1)

    if overlap == 0:
        return 0.0

    union = e1.n_samples + e2.n_samples - overlap
    return overlap / union if union > 0 else 0.0


def match_events_max_iou(
    gt_events: List[Event],
    pred_events: List[Event],
    min_iou: float = 0.0,
) -> Tuple[List[MatchResult], List[Event]]:
    """Match GT events to Pred events using the Maximum IoU greedy algorithm.

    This implements the matching procedure from Startsev & Zemblys (2022),
    Table 2 (Maximum IoU matcher):

    1. Compute IoU for every (GT, Pred) pair.
    2. Sort pairs by IoU in descending order (stable sort preserves insertion
       order for ties — earlier events get priority).
    3. Iterate: if both the GT event and the Pred event are still unmatched,
       record a match and mark both as matched.
    4. After iteration:
       - GT events without a match → False Negatives, represented as
         :class:`MatchResult` with ``pred_event = None``.
       - Pred events without a match → False Positives, returned in the
         second return value.

    Note on IoU threshold:
    The paper notes that an IoU > 0.5 threshold guarantees at most one Pred
    candidate per GT event. By default (``min_iou=0.0``) *any* overlap
    constitutes a candidate match; callers can tighten this if needed.

    Parameters
    ----------
    gt_events:   Ground-truth events.
    pred_events: Predicted events.
    min_iou:     Minimum IoU required for a pair to be considered
                 (exclusive: ``iou > min_iou`` must hold).

    Returns
    -------
    matches:        One :class:`MatchResult` per GT event (matched or FN).
    unmatched_pred: Pred events not assigned to any GT event (FP).
    """
    # Build all candidate (iou, gt_idx, pred_idx) triples
    candidates: List[Tuple[float, int, int]] = []
    for gi, ge in enumerate(gt_events):
        for pi, pe in enumerate(pred_events):
            iou = compute_iou(ge, pe)
            if iou > min_iou:
                candidates.append((iou, gi, pi))

    # Sort by IoU descending; Python's sort is stable so equal-IoU pairs keep
    # their original enumeration order (earlier GT / Pred events win ties).
    candidates.sort(key=lambda x: -x[0])

    matched_gt: set = set()
    matched_pred: set = set()
    pair_matches: List[MatchResult] = []

    for iou, gi, pi in candidates:
        if gi not in matched_gt and pi not in matched_pred:
            ge = gt_events[gi]
            pe = pred_events[pi]
            pair_matches.append(
                MatchResult(
                    gt_event=ge,
                    pred_event=pe,
                    iou=iou,
                    onset_dev_ms=pe.start_time_ms - ge.start_time_ms,
                    offset_dev_ms=pe.end_time_ms - ge.end_time_ms,
                    is_correct_type=(ge.event_type == pe.event_type),
                )
            )
            matched_gt.add(gi)
            matched_pred.add(pi)

    # False Negatives: GT events with no match
    for gi, ge in enumerate(gt_events):
        if gi not in matched_gt:
            pair_matches.append(
                MatchResult(
                    gt_event=ge,
                    pred_event=None,
                    iou=0.0,
                    onset_dev_ms=float("nan"),
                    offset_dev_ms=float("nan"),
                    is_correct_type=False,
                )
            )

    # False Positives: Pred events with no match
    unmatched_pred = [
        pred_events[pi]
        for pi in range(len(pred_events))
        if pi not in matched_pred
    ]

    return pair_matches, unmatched_pred


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def compute_event_iou_metrics(
    df: pd.DataFrame,
    gt_col: str = "gt_sample_type",
    pred_col: str = "ivt_sample_type",
    time_col: str = "time_ms",
    min_iou: float = 0.0,
    event_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute event-level evaluation metrics using Maximum IoU matching.

    This is the recommended evaluation methodology from Startsev & Zemblys
    (2022, Behavior Research Methods).  It provides:

    * **Event counts** per class (GT and Pred).
    * **Event-level confusion matrix**: rows = GT class, columns = matched
      Pred class; ``"FN"`` column for unmatched GT events.
    * **Timing quality measures** per class (Startsev & Zemblys, 2022,
      Best Practice #5): mean ± std of onset and offset deviation in ms,
      computed only for correctly typed matched pairs.
    * **Summary counts**: n_gt_events, n_pred_events, n_matched, n_fn, n_fp.

    Parameters
    ----------
    df:           DataFrame with sample-level classification columns.
    gt_col:       Column name for ground-truth labels.
    pred_col:     Column name for predicted labels.
    time_col:     Column name for timestamps in milliseconds.
    min_iou:      Minimum IoU for a valid match (default 0 → any overlap).
    event_types:  If given, restrict analysis to these event types only.

    Returns
    -------
    Dictionary with keys:
        ``n_gt_events``, ``n_pred_events``, ``n_matched``, ``n_fn``, ``n_fp``,
        ``event_counts``, ``confusion_matrix``, ``timing``,
        ``matches``, ``unmatched_pred``.
    """
    if df.empty or len(df) == 0:
        return {
            "n_gt_events": 0,
            "n_pred_events": 0,
            "n_matched": 0,
            "n_fn": 0,
            "n_fp": 0,
            "event_counts": {},
            "confusion_matrix": {},
            "timing": {},
            "matches": [],
            "unmatched_pred": [],
        }

    # ---- Extract events --------------------------------------------------
    gt_events = extract_events(df[gt_col], df[time_col])
    pred_events = extract_events(df[pred_col], df[time_col])

    if event_types is not None:
        gt_events = [e for e in gt_events if e.event_type in event_types]
        pred_events = [e for e in pred_events if e.event_type in event_types]

    # ---- Match -----------------------------------------------------------
    matches, unmatched_pred = match_events_max_iou(gt_events, pred_events, min_iou=min_iou)

    # ---- Event counts per class ------------------------------------------
    all_types = sorted(
        set(e.event_type for e in gt_events) | set(e.event_type for e in pred_events)
    )
    event_counts: Dict[str, Dict[str, int]] = {}
    for t in all_types:
        event_counts[t] = {
            "gt": sum(1 for e in gt_events if e.event_type == t),
            "pred": sum(1 for e in pred_events if e.event_type == t),
        }

    # ---- Event-level confusion matrix ------------------------------------
    # conf[gt_type][pred_type] = number of GT events of gt_type matched to pred_type
    # conf[gt_type]["FN"]      = unmatched GT events of gt_type
    confusion: Dict[str, Dict[str, int]] = {}

    for m in matches:
        gt_type = m.gt_event.event_type
        pred_type = m.pred_event.event_type if m.pred_event is not None else "FN"
        confusion.setdefault(gt_type, {})
        confusion[gt_type][pred_type] = confusion[gt_type].get(pred_type, 0) + 1

    # FP events are not reflected in confusion (no GT row), kept in unmatched_pred

    # ---- Timing statistics -----------------------------------------------
    # Only for correctly typed matched pairs (gt_type == pred_type)
    timing_raw: Dict[str, Dict[str, List[float]]] = {}
    for m in matches:
        if m.pred_event is not None and m.is_correct_type:
            t = m.gt_event.event_type
            timing_raw.setdefault(t, {"onset": [], "offset": []})
            timing_raw[t]["onset"].append(m.onset_dev_ms)
            timing_raw[t]["offset"].append(m.offset_dev_ms)

    timing: Dict[str, Dict[str, Any]] = {}
    for t, devs in timing_raw.items():
        onset = devs["onset"]
        offset = devs["offset"]
        timing[t] = {
            "n_events": len(onset),
            "onset_dev_mean_ms": float(np.mean(onset)) if onset else float("nan"),
            "onset_dev_std_ms": (
                float(np.std(onset, ddof=1)) if len(onset) > 1 else float("nan")
            ),
            "onset_dev_abs_mean_ms": float(np.mean(np.abs(onset))) if onset else float("nan"),
            "offset_dev_mean_ms": float(np.mean(offset)) if offset else float("nan"),
            "offset_dev_std_ms": (
                float(np.std(offset, ddof=1)) if len(offset) > 1 else float("nan")
            ),
            "offset_dev_abs_mean_ms": float(np.mean(np.abs(offset))) if offset else float("nan"),
        }

    # ---- Summary --------------------------------------------------------
    n_matched = sum(1 for m in matches if m.pred_event is not None)
    n_fn = sum(1 for m in matches if m.pred_event is None)
    n_fp = len(unmatched_pred)

    return {
        "n_gt_events": len(gt_events),
        "n_pred_events": len(pred_events),
        "n_matched": n_matched,
        "n_fn": n_fn,
        "n_fp": n_fp,
        "event_counts": event_counts,
        "confusion_matrix": confusion,
        "timing": timing,
        "matches": matches,
        "unmatched_pred": unmatched_pred,
    }


# ---------------------------------------------------------------------------
# Report formatter (optional, for human-readable output)
# ---------------------------------------------------------------------------

def format_event_iou_report(metrics: Dict[str, Any]) -> str:
    """Format compute_event_iou_metrics() output as a human-readable string.

    Suitable for embedding in evaluate_ivt_vs_ground_truth().

    Parameters
    ----------
    metrics:  Output dict from :func:`compute_event_iou_metrics`.

    Returns
    -------
    Multi-line string with event counts, confusion matrix, and timing stats.
    """
    lines: List[str] = []
    a = lines.append

    a("=== Event-Level Evaluation (Maximum IoU – Startsev & Zemblys, 2022) ===")
    a(f"GT events:   {metrics['n_gt_events']}")
    a(f"Pred events: {metrics['n_pred_events']}")
    a(f"Matched:     {metrics['n_matched']}")
    a(f"False Neg:   {metrics['n_fn']}")
    a(f"False Pos:   {metrics['n_fp']}")
    a("")

    # Event counts per class
    a("Event counts by class:")
    counts = metrics["event_counts"]
    if counts:
        col = max(len(t) for t in counts) + 2
        a(f"  {'Class'.ljust(col)}  {'GT':>6}  {'Pred':>6}")
        a(f"  {'-' * col}  {'------'}  {'------'}")
        for t in sorted(counts):
            gt_n = counts[t]["gt"]
            pred_n = counts[t]["pred"]
            diff = pred_n - gt_n
            diff_str = f"({'+' if diff >= 0 else ''}{diff})" if diff != 0 else "     "
            a(f"  {t.ljust(col)}  {gt_n:>6}  {pred_n:>6}  {diff_str}")
    a("")

    # Confusion matrix
    a("Event confusion matrix (rows = GT, cols = matched Pred / FN):")
    conf = metrics["confusion_matrix"]
    if conf:
        all_pred_labels = sorted(
            set(k for row in conf.values() for k in row) | {"FN"}
        )
        # Header
        col_w = max(14, max(len(lbl) for lbl in list(conf.keys()) + all_pred_labels) + 2)
        header = "  " + "".ljust(col_w) + "".join(lbl.center(col_w) for lbl in all_pred_labels)
        a(header)
        sep = "  " + "-" * (col_w * (len(all_pred_labels) + 1))
        a(sep)
        for gt_type in sorted(conf.keys()):
            row_str = "  " + f"GT:{gt_type}".ljust(col_w)
            for pred_type in all_pred_labels:
                val = conf[gt_type].get(pred_type, 0)
                row_str += str(val).center(col_w)
            a(row_str)
        a(sep)
    a("")

    # Timing
    timing = metrics["timing"]
    if timing:
        a("Timing quality (correctly matched pairs, onset/offset vs. GT):")
        a(f"  {'Class'.ljust(18)} {'N':>5}  {'Onset µ':>9}  {'Onset σ':>9}"
          f"  {'Offset µ':>9}  {'Offset σ':>9}  (ms)")
        a("  " + "-" * 75)
        for t in sorted(timing):
            s = timing[t]

            def _fmt(v: float) -> str:
                return f"{v:+.2f}" if not math.isnan(v) else "   N/A "

            a(
                f"  {t.ljust(18)} {s['n_events']:>5}"
                f"  {_fmt(s['onset_dev_mean_ms']):>9}"
                f"  {_fmt(s['onset_dev_std_ms']):>9}"
                f"  {_fmt(s['offset_dev_mean_ms']):>9}"
                f"  {_fmt(s['offset_dev_std_ms']):>9}"
            )
    else:
        a("Timing quality: no correctly matched same-type pairs found.")

    a("=" * 72)
    return "\n".join(lines)
