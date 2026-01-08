# ivt_filter/postprocessing/discard_short_fixations.py
"""Discard short fixations: removes fixations below minimum duration threshold."""

from __future__ import annotations

from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from ..config import FixationPostConfig


def discard_short_fixations(
    df: pd.DataFrame,
    cfg: FixationPostConfig,
    event_type_col: str,
    event_index_col: str,
    time_col: str,
    discard_target: str = "Unclassified",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Methodisch korrekte Implementierung von "Discard Short Fixations".
    
    Arbeitet auf Event-Level (nicht Sample-Level):
    - Nutzt event_type und event_index um Fixation-Events zu identifizieren
    - Berechnet Duration pro Event
    - Verwirft kurze Events durch Umlabeln aller Samples
    
    Definitionen:
    1) Fixation-Event: zusammenhängende Samples mit gleichem event_index und event_type == "Fixation"
    
    2) Robust dt aus time_ms:
       - dt_ms = median(positive_diffs)
       - Nicht hardcoded
    
    3) Eventdauer (inklusive Endsample):
       - duration_ms = (t_last - t_first) + dt_ms
       - +dt_ms ist verpflichtend (verhindert Off-by-one)
    
    Args:
        df: DataFrame mit Events
        cfg: FixationPostConfig mit min_fixation_duration_ms
        event_type_col: Name der Event-Type-Spalte
        event_index_col: Name der Event-Index-Spalte
        time_col: Name der Zeit-Spalte
        discard_target: Ziel-Label für verworfene Fixationen
    
    Returns:
        (df, stats): Modifizierter DataFrame und Statistiken
    """
    df = df.copy()
    
    if event_type_col not in df.columns:
        return df, {
            "discarded_fixations": 0,
            "discarded_samples": 0,
            "fixation_events": [],
        }
    
    if event_index_col not in df.columns:
        return df, {
            "discarded_fixations": 0,
            "discarded_samples": 0,
            "fixation_events": [],
        }
    
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")
    
    # Validate discard_target
    if discard_target not in ("Unclassified", "Saccade"):
        raise ValueError(f"discard_target must be 'Unclassified' or 'Saccade', got '{discard_target}'")
    
    event_types = df[event_type_col].astype(str).to_numpy()
    event_indices = df[event_index_col].to_numpy()
    times = df[time_col].to_numpy()
    n = len(df)
    
    # 1) Berechne robustes dt_ms aus time_ms (Median der positiven Differenzen)
    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0]
    
    if len(positive_diffs) == 0:
        dt_ms = 1.0
    else:
        dt_ms = float(np.median(positive_diffs))
    
    # 2) Finde alle Fixation-Events anhand von event_type und event_index
    fixation_events: List[Dict[str, Any]] = []
    
    # Gruppiere Fixation-Events
    current_event_index = None
    start_idx = None
    
    for i in range(n):
        is_fixation = event_types[i] == "Fixation"
        event_idx = event_indices[i]
        
        if is_fixation and event_idx is not None and not pd.isna(event_idx):
            event_idx_int = int(event_idx)
            
            if current_event_index is None or current_event_index != event_idx_int:
                # Neues Event beginnt
                if start_idx is not None:
                    # Vorheriges Event abschließen
                    end_idx = i - 1
                    t_first = float(times[start_idx])
                    t_last = float(times[end_idx])
                    duration_ms = (t_last - t_first) + dt_ms
                    
                    fixation_events.append({
                        "event_index": current_event_index,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "t_first": t_first,
                        "t_last": t_last,
                        "duration_ms": duration_ms,
                        "num_samples": end_idx - start_idx + 1,
                    })
                
                # Neues Event starten
                current_event_index = event_idx_int
                start_idx = i
        else:
            # Kein Fixation-Sample oder kein Index
            if start_idx is not None:
                # Vorheriges Event abschließen
                end_idx = i - 1
                t_first = float(times[start_idx])
                t_last = float(times[end_idx])
                duration_ms = (t_last - t_first) + dt_ms
                
                fixation_events.append({
                    "event_index": current_event_index,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "t_first": t_first,
                    "t_last": t_last,
                    "duration_ms": duration_ms,
                    "num_samples": end_idx - start_idx + 1,
                })
                
                current_event_index = None
                start_idx = None
    
    # Letztes Event behandeln
    if start_idx is not None:
        end_idx = n - 1
        t_first = float(times[start_idx])
        t_last = float(times[end_idx])
        duration_ms = (t_last - t_first) + dt_ms
        
        fixation_events.append({
            "event_index": current_event_index,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "t_first": t_first,
            "t_last": t_last,
            "duration_ms": duration_ms,
            "num_samples": end_idx - start_idx + 1,
        })
    
    # 3) Verwerfe kurze Fixationen
    discarded_fixations = 0
    discarded_samples = 0
    threshold_ms = cfg.min_fixation_duration_ms
    
    for event in fixation_events:
        if event["duration_ms"] < threshold_ms:
            # Verwerfe dieses Event: setze alle Samples auf discard_target
            start = event["start_idx"]
            end = event["end_idx"]
            
            for j in range(start, end + 1):
                if event_types[j] == "Fixation":
                    event_types[j] = discard_target
                    event_indices[j] = None  # Event-Index entfernen
                    discarded_samples += 1
            
            discarded_fixations += 1
            event["discarded"] = True
        else:
            event["discarded"] = False
    
    # 4) Aktualisiere DataFrame
    df[event_type_col] = event_types
    df[event_index_col] = event_indices
    
    # 5) Statistiken
    stats = {
        "discarded_fixations": discarded_fixations,
        "discarded_samples": discarded_samples,
        "total_fixation_events": len(fixation_events),
        "dt_ms": dt_ms,
        "threshold_ms": threshold_ms,
        "discard_target": discard_target,
        "fixation_events": fixation_events,
    }
    
    return df, stats
