#!/usr/bin/env python3
"""
Akzeptanztest für Discard Short Fixations (60 ms) auf Ground Truth Daten.

Testziel:
    Validierung dass die Discard-Logik auf GT-Labels exakt das gleiche 
    Ergebnis erzeugt wie die Tobii Ground Truth mit aktiviertem Discard.

Testdaten:
    - Input:  LeftV30W20_output.tsv (GT ohne Discard)
    - Referenz: LeftV30W20Discard60NoSaccade_output.tsv (GT mit Discard)

Erfolgskriterium:
    100% Sample-genaue Identität (0 mismatches)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def _rebuild_events_from_sample_types(
    df: pd.DataFrame,
    sample_col: str,
    event_type_col: str,
    event_index_col: str,
) -> pd.DataFrame:
    """
    Rekonstruiere Event-Spalten aus Sample-Labels.
    
    Args:
        df: DataFrame mit Sample-Labels
        sample_col: Spalte mit Sample-Labels ("Fixation", "Saccade", "Unclassified")
        event_type_col: Ziel-Spalte für Event-Typen
        event_index_col: Ziel-Spalte für Event-Indizes
    
    Returns:
        DataFrame mit hinzugefügten Event-Spalten
    """
    df = df.copy()
    
    labels = df[sample_col].astype(str).to_numpy()
    n = len(df)
    
    event_types = np.empty(n, dtype=object)
    event_indices = np.empty(n, dtype=object)
    
    current_type = None
    current_index = 0
    
    for i in range(n):
        label = labels[i]
        
        if label != current_type:
            # Neues Event beginnt
            current_type = label
            current_index += 1
        
        event_types[i] = current_type
        event_indices[i] = current_index
    
    df[event_type_col] = event_types
    df[event_index_col] = event_indices
    
    return df


def discard_short_fixations_on_events(
    df: pd.DataFrame,
    event_type_col: str,
    event_index_col: str,
    time_col: str = "time_ms",
    discard_threshold_ms: float = 60.0,
    discard_target: str = "Unclassified",
) -> pd.DataFrame:
    """
    Wende Discard Short Fixations auf Event-Spalten an.
    
    Methodisch korrekte Implementierung gemäß Spezifikation:
    1) Nutze event_type und event_index um Fixation-Events zu identifizieren
    2) Robust dt_ms = median(positive_diffs)
    3) Duration = (t_last - t_first) + dt_ms (mit +dt_ms!)
    4) Wenn duration < threshold: alle Samples des Events → discard_target
    
    Args:
        df: DataFrame mit Events
        event_type_col: Spalte mit Event-Typen
        event_index_col: Spalte mit Event-Indizes
        time_col: Name der Zeit-Spalte
        discard_threshold_ms: Mindestdauer für Fixationen
        discard_target: Ziel-Label für verworfene Fixationen
    
    Returns:
        Modifizierter DataFrame
    """
    df = df.copy()
    
    if event_type_col not in df.columns:
        raise ValueError(f"Column '{event_type_col}' not found in DataFrame")
    
    if event_index_col not in df.columns:
        raise ValueError(f"Column '{event_index_col}' not found in DataFrame")
    
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")
    
    event_types = df[event_type_col].astype(str).to_numpy()
    event_indices = df[event_index_col].to_numpy()
    times = df[time_col].to_numpy()
    n = len(df)
    
    # 1) Berechne robustes dt_ms (Median der positiven Zeitdifferenzen)
    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0]
    
    if len(positive_diffs) == 0:
        dt_ms = 1.0  # Fallback
    else:
        dt_ms = float(np.median(positive_diffs))
    
    print(f"[Discard] Robust dt_ms: {dt_ms:.3f} ms")
    
    # 2) Finde alle Fixation-Events anhand von event_type und event_index
    fixation_events: List[Dict[str, Any]] = []
    
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
            "duration_ms": duration_ms,
            "num_samples": end_idx - start_idx + 1,
        })
    
    # 3) Verwerfe kurze Fixationen
    discarded_fixations = 0
    discarded_samples = 0
    
    for event in fixation_events:
        if event["duration_ms"] < discard_threshold_ms:
            start = event["start_idx"]
            end = event["end_idx"]
            
            for j in range(start, end + 1):
                if event_types[j] == "Fixation":
                    event_types[j] = discard_target
                    event_indices[j] = None
                    discarded_samples += 1
            
            discarded_fixations += 1
            event["discarded"] = True
        else:
            event["discarded"] = False
    
    print(f"[Discard] Total fixation events: {len(fixation_events)}")
    print(f"[Discard] Discarded fixations: {discarded_fixations}")
    print(f"[Discard] Discarded samples: {discarded_samples}")
    
    df[event_type_col] = event_types
    df[event_index_col] = event_indices
    
    return df


def run_acceptance_test():
    """
    Führe den Akzeptanztest durch:
    Discard Short Fixations auf GT-Daten anwenden und mit Referenz vergleichen.
    """
    print("=" * 80)
    print("AKZEPTANZTEST: Discard Short Fixations (60 ms) auf Ground Truth (Event-Type)")
    print("=" * 80)
    print()
    # 1) Lade Input (GT ohne Discard)
    print("[1/5] Lade Input-Datei (GT ohne Discard)...")
    input_path = "./test_data/outputs/LeftV30W20_output.tsv"
    
    try:
        df_input = pd.read_csv(input_path, sep="\t", decimal=",", low_memory=False)
        print(f"      ✓ Geladen: {len(df_input)} samples")
    except FileNotFoundError:
        print(f"      ✗ FEHLER: Datei nicht gefunden: {input_path}")
        print("      Bitte zuerst Pipeline ausführen ohne --discard-short-fixations")
        return False
    
    if "gt_sample_type" not in df_input.columns:
        print("      ✗ FEHLER: Spalte 'gt_sample_type' nicht gefunden")
        return False
    
    print()
    
    # 2) Baue Event-Spalten aus GT-Sample-Types
    print("[2/5] Rekonstruiere Event-Spalten aus gt_sample_type...")
    
    df_with_events = _rebuild_events_from_sample_types(
        df_input,
        sample_col="gt_sample_type",
        event_type_col="gt_event_type",
        event_index_col="gt_event_index",
    )
    
    print(f"      ✓ Event-Spalten erstellt")
    print()
    
    # 3) Wende Discard auf Event-Spalten an
    print("[3/5] Wende Discard Short Fixations auf Event-Spalten an...")
    print(f"      Parameter: threshold=60 ms, target=Unclassified")
    print()
    
    df_simulated = discard_short_fixations_on_events(
        df_with_events,
        event_type_col="gt_event_type",
        event_index_col="gt_event_index",
        time_col="time_ms",
        discard_threshold_ms=60.0,
        discard_target="Unclassified",
    )
    
    # Simulierte Spalte speichern
    df_simulated["gt_sample_type_discard60_sim"] = df_simulated["gt_event_type"]
    print()
    
    # 4) Lade Referenz (GT mit Discard)
    print("[4/5] Lade Referenz-Datei (GT mit Discard)...")
    reference_path = "./test_data/outputs/LeftV30W20Discard60NoSaccade_output.tsv"
    
    try:
        df_reference = pd.read_csv(reference_path, sep="\t", decimal=",", low_memory=False)
        print(f"      ✓ Geladen: {len(df_reference)} samples")
    except FileNotFoundError:
        print(f"      ✗ FEHLER: Datei nicht gefunden: {reference_path}")
        print("      Bitte zuerst Pipeline ausführen mit --discard-short-fixations")
        return False
    
    if "gt_sample_type" not in df_reference.columns:
        print("      ✗ FEHLER: Spalte 'gt_sample_type' nicht gefunden in Referenz")
        return False
    
    print()
    
    # 5) Vergleiche Sample-genau
    print("[5/5] Vergleiche simulierte GT mit Referenz-GT...")
    
    # Sicherstellen dass beide gleiche Länge haben
    if len(df_simulated) != len(df_reference):
        print(f"      ✗ FEHLER: Unterschiedliche Anzahl Samples!")
        print(f"         Simuliert: {len(df_simulated)}")
        print(f"         Referenz:  {len(df_reference)}")
        return False
    
    # Sample-weiser Vergleich
    gt_sim = df_simulated["gt_sample_type_discard60_sim"].astype(str).to_numpy()
    gt_ref = df_reference["gt_sample_type"].astype(str).to_numpy()
    
    matches = gt_sim == gt_ref
    num_matches = matches.sum()
    num_mismatches = (~matches).sum()
    match_rate = (num_matches / len(gt_sim)) * 100
    
    print(f"      Total samples:    {len(gt_sim)}")
    print(f"      Matches:          {num_matches}")
    print(f"      Mismatches:       {num_mismatches}")
    print(f"      Match rate:       {match_rate:.2f}%")
    print()
    
    # Erfolgskriterium: 100% Match (0 mismatches)
    if num_mismatches == 0:
        print("=" * 80)
        print("✓ AKZEPTANZTEST BESTANDEN")
        print("=" * 80)
        print()
        print("Die Discard-Logik erzeugt sample-genau identische Ergebnisse")
        print("wie die Ground Truth mit aktiviertem Discard Short Fixations.")
        print()
        return True
    else:
        print("=" * 80)
        print("✗ AKZEPTANZTEST FEHLGESCHLAGEN")
        print("=" * 80)
        print()
        print(f"Es wurden {num_mismatches} Abweichungen gefunden.")
        print()
        
        # Zeige erste Mismatches
        mismatch_indices = np.where(~matches)[0]
        print("Erste 10 Mismatches:")
        print("-" * 80)
        for i, idx in enumerate(mismatch_indices[:10]):
            print(f"  Sample {idx}:")
            print(f"    Simuliert: {gt_sim[idx]}")
            print(f"    Referenz:  {gt_ref[idx]}")
            if "time_ms" in df_simulated.columns:
                print(f"    Zeit:      {df_simulated['time_ms'].iloc[idx]:.1f} ms")
            print()
        
        return False


if __name__ == "__main__":
    success = run_acceptance_test()
    exit(0 if success else 1)
