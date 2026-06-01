"""Orchestrate fixation postprocessing stages."""

from __future__ import annotations

from typing import Tuple, Dict, Any

import pandas as pd

from ..config import FixationPostConfig
from ..domain.events import rebuild_events_from_sample_labels
from .merge_fixations import merge_adjacent_fixations as _merge_adjacent_fixations_internal
from .discard_short_fixations import discard_short_fixations as _discard_short_fixations_internal


# Backward-compatible private alias; event rules live in domain.events.
_rebuild_ivt_events_from_sample_types = rebuild_events_from_sample_labels


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
