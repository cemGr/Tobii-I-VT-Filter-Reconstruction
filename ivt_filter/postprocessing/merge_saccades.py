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
    Kurze Saccaden-Bloecke, die komplett innerhalb von GT-Fixationen liegen,
    zu Fixation umlabeln (optionale GT-gestützte Glättung).

    - Operiert entweder auf 'ivt_sample_type' oder auf 'ivt_event_type',
      abhängig von cfg.use_sample_type_column.
    - Erzeugt eine zusätzliche Spalte <work_col>_smoothed.

    Rueckgabe:
      - df (mit *zusätzlichen* Spalten, Original bleibt erhalten)
      - stats-Dict mit ein paar Kennzahlen
    """
    if cfg is None:
        cfg = SaccadeMergeConfig()

    if "time_ms" not in df.columns:
        raise ValueError("DataFrame must contain 'time_ms' column.")

    # GT-Spalte wird nicht mehr benötigt - Post-Smoothing basiert nur auf IVT-Kontext

    event_col = "ivt_event_type"
    if event_col not in df.columns:
        raise ValueError("DataFrame must contain 'ivt_event_type' column.")

    sample_col = cfg.use_sample_type_column
    if sample_col is not None and sample_col not in df.columns:
        sample_col = None

    df = df.copy().reset_index(drop=True)

    # Arbeits-Spalte: entweder Sample-Level oder Event-Level
    work_col = sample_col if sample_col is not None else event_col

    times = df["time_ms"].to_numpy()
    sample_interval_ms = estimate_sample_interval_ms(times)
    # GT no longer needed - Post-Smoothing basiert nur auf IVT-Kontext und Dauer
    ivt = df[work_col].astype(str).to_numpy()

    n = len(df)

    # Find saccade blocks (zusammenhängende "Saccade"-Segmente)
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

        # GT check skipped - merge basierend nur auf Dauer und IVT-Kontext
        # (Original: prüfte ob GT innerhalb des Blocks komplett Fixation ist)

        if cfg.require_fixation_context:
            # Prüfe IVT-Kontext (nicht GT): linker und rechter Nachbar müssen IVT-Fixation sein
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

    # Events neu rekonstruieren
    if sample_col is not None and work_col == sample_col:
        # Sample-basierte Glättung -> Events aus Sample-Spalte neu bauen
        df = _rebuild_ivt_events_from_sample_types(
            df,
            sample_col=sample_col + "_smoothed",
            event_type_col="ivt_event_type_smoothed",
            event_index_col="ivt_event_index_smoothed",
        )
    else:
        # Event-basierte Glättung -> direkt auf Event-Spalte
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
