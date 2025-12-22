# ivt_filter/classification.py
from __future__ import annotations

from typing import Optional, List
import math
import pandas as pd

from .config import IVTClassifierConfig


def expand_gt_events_to_samples(
    df: pd.DataFrame,
    event_type_col: str = "gt_event_type",
    event_index_col: str = "gt_event_index",
    sample_type_col: str = "gt_sample_type",
) -> pd.DataFrame:
    """
    Expandiert Event-basierte Ground Truth Labels zu Sample-basierten Labels.
    
    Wenn ein Sample zum selben Event Index gehört wie sein Vorgänger,
    bekommt es das gleiche Label. Dies ist das inverse der rebuild-Operation.
    """
    if event_type_col not in df.columns:
        return df  # Keine GT Events vorhanden
    
    df = df.copy()
    df[sample_type_col] = df[event_type_col].astype(str)
    return df


def apply_ivt_classifier(
    df: pd.DataFrame,
    cfg: Optional[IVTClassifierConfig] = None,
) -> pd.DataFrame:
    """
    Einfacher I VT Velocity Threshold Klassifikator.

    Erwartet:
      - Spalte "velocity_deg_per_sec"

    Fuegt hinzu:
      - ivt_sample_type  (Fixation / Saccade / Unclassified)
      - ivt_event_type   (Fixation / Saccade / Unclassified)
      - ivt_event_index  (laufender Index pro Event)
    """
    if cfg is None:
        cfg = IVTClassifierConfig()

    if "velocity_deg_per_sec" not in df.columns:
        raise ValueError("DataFrame must contain 'velocity_deg_per_sec' before classification.")

    df = df.copy()



    def is_invalid(val):
        # Treat None, NaN, False, 0 as invalid
        if val is None:
            return True
        try:
            # Handle numpy.nan and floats
            if isinstance(val, float) and not math.isfinite(val):
                return True
        except Exception:
            pass
        if val is False or val == 0:
            return True
        return False

    def classify_sample_with_eyes(row) -> str:
        v = row["velocity_deg_per_sec"]
        # Primär: combined_valid steuert EyesNotFound eindeutig für alle Modi
        combined_valid = row.get("combined_valid", True)
        if is_invalid(combined_valid) or (isinstance(combined_valid, bool) and combined_valid is False):
            return "EyesNotFound"
        # Sekundär: Velocity-Prüfung für Unclassified
        # Prüfe auf None, NaN (float oder string), oder nicht-endliche Werte
        try:
            if v is None:
                return "Unclassified"
            # Handle string "nan" oder andere string-representations
            if isinstance(v, str):
                v_str = v.strip().lower()
                if v_str in ("nan", "none", "n/a", ""):
                    return "Unclassified"
                # Replace German decimal separator (comma) with English (period)
                v_str = v_str.replace(",", ".")
                v = float(v_str)  # Konvertiere zu float
            v_float = float(v)
            if not math.isfinite(v_float):
                return "Unclassified"
            if v_float > cfg.velocity_threshold_deg_per_sec:
                return "Saccade"
            return "Fixation"
        except (ValueError, TypeError):
            # Irgendein Konvertierungsfehler -> Unclassified
            return "Unclassified"

    # Use .apply with axis=1 to access both velocity and eye validity
    df["ivt_sample_type"] = df.apply(classify_sample_with_eyes, axis=1)

    # Event Labels aus Sample Labels bauen
    df = rebuild_ivt_events_from_sample_types(
        df,
        sample_col="ivt_sample_type",
        event_type_col="ivt_event_type",
        event_index_col="ivt_event_index",
    )

    return df


def rebuild_ivt_events_from_sample_types(
    df: pd.DataFrame,
    sample_col: str,
    event_type_col: str,
    event_index_col: str,
) -> pd.DataFrame:
    """
    Bilde Event Typen und Indizes aus sample basierten Labels.

    - Fixation/Saccade Bloecke bekommen fortlaufende Indizes.
    - Alles andere (z B Unclassified) bekommt None als Index.
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
