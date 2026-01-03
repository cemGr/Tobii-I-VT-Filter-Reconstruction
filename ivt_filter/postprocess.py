# ivt_filter/postprocess.py
from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .config import SaccadeMergeConfig, FixationPostConfig
from .velocity import visual_angle_deg
from .velocity_calculation import Ray3DAngle, Olsen2DApproximation


# -------------------------------------------------------------------
# Hilfsfunktionen fuer Event-Rekonstruktion & GT-Handling
# -------------------------------------------------------------------


def _find_gt_column(df: pd.DataFrame) -> str:
    """
    Geeignete Ground-Truth-Spalte finden.

    Erwartet entweder:
      - 'gt_event_type' oder
      - 'Eye movement type'
    """
    if "gt_event_type" in df.columns:
        return "gt_event_type"
    if "Eye movement type" in df.columns:
        return "Eye movement type"
    raise ValueError(
        "No ground-truth event type column found. "
        "Expected 'gt_event_type' or 'Eye movement type'."
    )


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


# -------------------------------------------------------------------
# 1) GT-gestütztes Saccaden-Postprocessing
# -------------------------------------------------------------------


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


# -------------------------------------------------------------------
# 2) Tobii-aehnliches Fixations-Postprocessing (ohne GT)
#     - benachbarte Fixationen mergen (zeit-/raumnahe)
#     - kurze Fixationen verwerfen
# -------------------------------------------------------------------


def _merge_adjacent_fixations_internal(
    df: pd.DataFrame,
    cfg: FixationPostConfig,
    sample_col: str,
    time_col: str,
    x_col: str,
    y_col: str,
    eye_z_col: str,
    use_ray3d: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Benachbarte Fixationen mergen, wenn:

      - Zeitabstand zwischen Ende Fix1 und Start Fix2 <= max_time_gap_ms
      - visueller Winkel zwischen Fix1-Zentrum und Fix2-Zentrum
        <= max_angle_deg

    Umsetzung:
      - Luecke zwischen zwei Fixationen wird als Fixation umlabelt.
    """
    df = df.copy()
    if sample_col not in df.columns:
        return df, {
            "merged_pairs": 0,
            "gap_samples_to_fixation": 0,
            "original_fixation_events": 0,
        }
    labels = df[sample_col].astype(str).to_numpy()
    times = df[time_col].to_numpy()
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    eye_z = df[eye_z_col].to_numpy() if eye_z_col in df.columns else None
    
    # Get velocity for intelligent gap-filling (optional but recommended)
    velocity = None
    if "velocity_deg_per_sec" in df.columns:
        velocity = df["velocity_deg_per_sec"].to_numpy()
    
    # For Ray3D we need auch eye_x und eye_y
    eye_x = None
    eye_y = None
    if use_ray3d:
        if "eye_x_mm" in df.columns:
            eye_x = df["eye_x_mm"].to_numpy()
        if "eye_y_mm" in df.columns:
            eye_y = df["eye_y_mm"].to_numpy()

    n = len(df)
    
    # Select strategy
    if use_ray3d:
        angle_calculator = Ray3DAngle()
        print("[MergeFixations] Using Ray3D angle calculation")
    else:
        angle_calculator = Olsen2DApproximation()
        print("[MergeFixations] Using Olsen 2D approximation")

    # Fixations-Bloecke aus den Labels bestimmen
    events: List[Tuple[int, int]] = []
    in_fix = False
    start = 0
    for i in range(n):
        if labels[i] == "Fixation":
            if not in_fix:
                in_fix = True
                start = i
        else:
            if in_fix:
                events.append((start, i - 1))
                in_fix = False
    if in_fix:
        events.append((start, n - 1))

    merged_pairs = 0
    gap_samples_to_fix = 0

    for idx in range(len(events) - 1):
        s1, e1 = events[idx]
        s2, e2 = events[idx + 1]

        # Zeitabstand zwischen Ende Fix1 und Start Fix2
        time_gap = float(times[s2]) - float(times[e1])
        if time_gap <= 0 or time_gap > cfg.max_time_gap_ms:
            continue

        # Zentren der Fixationen (gemitteltes smoothed_x_mm / smoothed_y_mm)
        x1 = float(np.nanmean(x[s1 : e1 + 1]))
        y1 = float(np.nanmean(y[s1 : e1 + 1]))
        x2 = float(np.nanmean(x[s2 : e2 + 1]))
        y2 = float(np.nanmean(y[s2 : e2 + 1]))

        if any(pd.isna(v) for v in (x1, y1, x2, y2)):
            continue

        # Calculate angle mit gewählter Strategie
        if use_ray3d and eye_x is not None and eye_y is not None and eye_z is not None:
            # Ray3D: Verwende volle 3D-Geometrie
            ex1 = float(np.nanmean(eye_x[s1 : e1 + 1]))
            ey1 = float(np.nanmean(eye_y[s1 : e1 + 1]))
            ez1 = float(np.nanmean(eye_z[s1 : e1 + 1]))
            ex2 = float(np.nanmean(eye_x[s2 : e2 + 1]))
            ey2 = float(np.nanmean(eye_y[s2 : e2 + 1]))
            ez2 = float(np.nanmean(eye_z[s2 : e2 + 1]))
            
            # Mittelwerte der Eye-Position
            ex = float(np.nanmean([ex1, ex2]))
            ey = float(np.nanmean([ey1, ey2]))
            ez = float(np.nanmean([ez1, ez2]))
            
            angle = angle_calculator.calculate_visual_angle(x1, y1, x2, y2, ex, ey, ez)
        else:
            # Olsen 2D: Verwende nur Z-Distanz
            if eye_z is not None:
                z1 = float(np.nanmean(eye_z[s1 : e1 + 1]))
                z2 = float(np.nanmean(eye_z[s2 : e2 + 1]))
                z = float(np.nanmean([z1, z2]))
            else:
                z = None
            
            angle = angle_calculator.calculate_visual_angle(x1, y1, x2, y2, None, None, z)
        
        if angle > cfg.max_angle_deg:
            continue

        # Luecke zwischen den beiden Fixationen als Fixation umlabeln
        # WICHTIG: EyesNotFound darf NICHT zu Fixation werden
        # OPTIMIERUNG: Keine Saccade-Samples einbeziehen (Velocity-Check)
        if s2 > e1 + 1:
            for j in range(e1 + 1, s2):
                if labels[j] == "Fixation" or labels[j] == "EyesNotFound":
                    continue
                
                # Velocity-basiertes Gap-Filling:
                # Nur Samples mit niedriger Velocity (<= 35°/s) werden gemerged
                # Rational: IVT-Threshold ist 30°/s, +5°/s Puffer für Grenzfälle
                # Verhindert echte Saccade-Samples (> 35°/s) in Fixations
                if velocity is not None and not pd.isna(velocity[j]):
                    if velocity[j] > 35.0:
                        continue
                
                labels[j] = "Fixation"
                gap_samples_to_fix += 1

        merged_pairs += 1

    df[sample_col] = labels
    stats = {
        "merged_pairs": merged_pairs,
        "gap_samples_to_fixation": gap_samples_to_fix,
        "original_fixation_events": len(events),
    }
    return df, stats


def _discard_short_fixations_internal(
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
