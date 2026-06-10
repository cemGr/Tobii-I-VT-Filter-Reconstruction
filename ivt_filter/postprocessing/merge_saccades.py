"""Merge short saccade blocks into surrounding fixations."""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

from ..config import SaccadeMergeConfig
from ..domain.durations import event_duration_ms, estimate_sample_interval_ms
from ..domain.events import iter_contiguous_events, rebuild_events_from_sample_labels


# Backward-compatible private alias; event rules live in domain.events.
_rebuild_ivt_events_from_sample_types = rebuild_events_from_sample_labels


def merge_short_saccade_blocks(
    df: pd.DataFrame,
    cfg: Optional[SaccadeMergeConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Relabel short saccade blocks that lie entirely within GT fixations
    to Fixation (optional GT-assisted smoothing).

    - Operates on either 'ivt_sample_type' or 'ivt_event_type',
      depending on cfg.use_sample_type_column.
    - Creates an additional column <work_col>_smoothed.

    Returns:
      - df (with *additional* columns; the original is preserved)
      - stats dict with a few summary metrics
    """
    if cfg is None:
        cfg = SaccadeMergeConfig()

    if "time_ms" not in df.columns:
        raise ValueError("DataFrame must contain 'time_ms' column.")

    # The GT column is no longer needed - post-smoothing relies only on IVT context

    event_col = "ivt_event_type"
    if event_col not in df.columns:
        raise ValueError("DataFrame must contain 'ivt_event_type' column.")

    sample_col = cfg.use_sample_type_column
    if sample_col is not None and sample_col not in df.columns:
        sample_col = None

    df = df.copy().reset_index(drop=True)

    # Work column: either sample-level or event-level
    work_col = sample_col if sample_col is not None else event_col

    times = df["time_ms"].to_numpy()
    sample_interval_ms = estimate_sample_interval_ms(times)
    # GT no longer needed - post-smoothing relies only on IVT context and duration
    ivt = df[work_col].astype(str).to_numpy()

    n = len(df)

    # Find saccade blocks (contiguous "Saccade" segments)
    blocks: List[Tuple[int, int]] = [
        (event.start, event.end)
        for event in iter_contiguous_events(ivt)
        if event.label == "Saccade"
    ]

    changed_blocks = 0
    changed_samples = 0

    new_ivt = ivt.copy()

    for (b_start, b_end) in blocks:
        duration_ms = event_duration_ms(
            times[b_start : b_end + 1], sample_interval_ms=sample_interval_ms
        )
        if duration_ms >= cfg.max_saccade_block_duration_ms:
            continue

        # GT check skipped - merge is based only on duration and IVT context
        # (Original: checked whether GT within the block was entirely Fixation)

        if cfg.require_fixation_context:
            # Check IVT context (not GT): left and right neighbors must be IVT Fixation
            if b_start > 0 and ivt[b_start - 1] != "Fixation":
                continue
            if b_end < n - 1 and ivt[b_end + 1] != "Fixation":
                continue

        for j in range(b_start, b_end + 1):
            if new_ivt[j] == "Saccade":
                new_ivt[j] = "Fixation"
                changed_samples += 1
        changed_blocks += 1

    df[work_col + "_smoothed"] = new_ivt

    # Rebuild events
    if sample_col is not None and work_col == sample_col:
        # Sample-based smoothing -> rebuild events from the sample column
        df = _rebuild_ivt_events_from_sample_types(
            df,
            sample_col=sample_col + "_smoothed",
            event_type_col="ivt_event_type_smoothed",
            event_index_col="ivt_event_index_smoothed",
        )
    else:
        # Event-based smoothing -> directly on the event column
        df["ivt_event_type_smoothed"] = df[work_col + "_smoothed"]
        df = _rebuild_ivt_events_from_sample_types(
            df,
            sample_col="ivt_event_type_smoothed",
            event_type_col="ivt_event_type_smoothed",
            event_index_col="ivt_event_index_smoothed",
        )

    stats: Dict[str, Any] = {
        "n_blocks_total": len(blocks),
        "n_blocks_merged": changed_blocks,
        "n_samples_merged": changed_samples,
        "max_saccade_block_duration_ms": cfg.max_saccade_block_duration_ms,
        "require_fixation_context": cfg.require_fixation_context,
        "used_sample_column": sample_col,
    }

    return df, stats
