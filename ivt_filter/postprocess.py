# ivt_filter/postprocess.py
"""Legacy postprocessing module - imports from new postprocessing package."""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any

import pandas as pd

from .config import SaccadeMergeConfig, FixationPostConfig
from .postprocessing.merge_fixations import merge_adjacent_fixations as _merge_adjacent_fixations_internal
from .postprocessing.discard_short_fixations import discard_short_fixations as _discard_short_fixations_internal


def _rebuild_ivt_events_from_sample_types(
    df: pd.DataFrame,
    sample_col: str,
    event_type_col: str,
    event_index_col: str,
) -> pd.DataFrame:
    """
    Aus einer Sample-Label-Spalte (Fixation/Saccade/Unclassified)
    wieder Events (zusammenhängende Blöcke) aufbauen.

    Verantwortlichkeit:
      - nimmt eine Sample-Spalte (z.B. 'ivt_sample_type_smoothed')
      - erzeugt Event-Type/Index-Spalten
    """
    sample_labels = df[sample_col].astype(str).tolist()

    new_event_type: List[str] = []
    new_event_index: List[Optional[int]] = []

    current_type: Optional[str] = None
    current_index: int = 0

    for label in sample_labels:
        if label not in ("Fixation", "Saccade"):
            new_event_type.append(label)
            new_event_index.append(None)
            current_type = None
            continue

        if label != current_type:
            current_index += 1
            current_type = label

        new_event_type.append(label)
        new_event_index.append(current_index)

    df[event_type_col] = new_event_type
    df[event_index_col] = new_event_index
    return df



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
    # GT no longer needed - Post-Smoothing basiert nur auf IVT-Kontext und Dauer
    ivt = df[work_col].astype(str).to_numpy()

    n = len(df)

    # Find saccade blocks (zusammenhängende "Saccade"-Segmente)
    blocks: List[Tuple[int, int]] = []
    in_block = False
    start_idx = 0

    for i in range(n):
        if ivt[i] == "Saccade":
            if not in_block:
                in_block = True
                start_idx = i
        else:
            if in_block:
                blocks.append((start_idx, i - 1))
                in_block = False
    if in_block:
        blocks.append((start_idx, n - 1))

    changed_blocks = 0
    changed_samples = 0

    new_ivt = ivt.copy()

    for (b_start, b_end) in blocks:
        duration_ms = float(times[b_end] - times[b_start])
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


def apply_fixation_postprocessing(
    df: pd.DataFrame,
    cfg: FixationPostConfig,
    sample_col: str = "ivt_sample_type_smoothed",
    time_col: str = "time_ms",
    x_col: str = "smoothed_x_mm",
    y_col: str = "smoothed_y_mm",
    eye_z_col: str = "eye_z_mm",
    event_type_col: str = "ivt_event_type_post",
    event_index_col: str = "ivt_event_index_post",
    use_ray3d: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Tobii-aehnliches Fixations-Postprocessing auf IVT-Predictions:

      1) benachbarte Fixationen mergen (zeit-/raum-nahe)
      2) kurze Fixationen verwerfen

    Single Responsibility:
      - arbeitet nur auf einer gegebenen Sample-Spalte (z.B. 'ivt_sample_type')
      - erzeugt neue Event-Spalten (event_type_col / event_index_col)
      - gibt Stats zurueck, Drucken ist Aufgabe der CLI/UI-Schicht.
    """
    if not (cfg.merge_adjacent_fixations or cfg.discard_short_fixations):
        return df, {
            "merged_pairs": 0,
            "gap_samples_to_fixation": 0,
            "original_fixation_events": 0,
            "discarded_fixations": 0,
            "discarded_samples": 0,
        }

    df = df.copy()
    merge_stats: Dict[str, Any] = {
        "merged_pairs": 0,
        "gap_samples_to_fixation": 0,
        "original_fixation_events": 0,
    }
    discard_stats: Dict[str, Any] = {
        "discarded_fixations": 0,
        "discarded_samples": 0,
    }

    # 1) Merge naher Fixationen
    if cfg.merge_adjacent_fixations:
        df, merge_stats = _merge_adjacent_fixations_internal(
            df,
            cfg,
            sample_col=sample_col,
            time_col=time_col,
            x_col=x_col,
            y_col=y_col,
            eye_z_col=eye_z_col,
            use_ray3d=use_ray3d,
        )

    # 2) Events aus Sample-Spalte aufbauen (für Discard benötigt)
    df = _rebuild_ivt_events_from_sample_types(
        df,
        sample_col=sample_col,
        event_type_col=event_type_col,
        event_index_col=event_index_col,
    )

    # 3) Kurze Fixationen verwerfen (arbeitet auf Events)
    if cfg.discard_short_fixations:
        df, discard_stats = _discard_short_fixations_internal(
            df,
            cfg,
            event_type_col=event_type_col,
            event_index_col=event_index_col,
            time_col=time_col,
            discard_target=cfg.discard_target,
        )

    stats: Dict[str, Any] = {}
    stats.update(merge_stats)
    stats.update(discard_stats)
    return df, stats
