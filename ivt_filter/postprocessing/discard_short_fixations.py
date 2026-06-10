# ivt_filter/postprocessing/discard_short_fixations.py
"""Discard short fixations: removes fixations below minimum duration threshold."""

from __future__ import annotations

from typing import Tuple, Dict, Any, List

import pandas as pd

from ..config import FixationPostConfig
from ..domain.durations import event_duration_ms, estimate_sample_interval_ms


def discard_short_fixations(
    df: pd.DataFrame,
    cfg: FixationPostConfig,
    event_type_col: str,
    event_index_col: str,
    time_col: str,
    discard_target: str = "Unclassified",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Methodically correct implementation of "Discard Short Fixations".

    Operates at the event level (not the sample level):
    - Uses event_type and event_index to identify fixation events
    - Computes the duration per event
    - Discards short events by relabeling all of their samples

    Definitions:
    1) Fixation event: contiguous samples with the same event_index and event_type == "Fixation"

    2) Robust dt from time_ms:
       - dt_ms = median(positive_diffs)
       - Not hardcoded

    3) Event duration (including the end sample):
       - duration_ms = event_duration_ms(times[start_idx : end_idx + 1], sample_interval_ms=dt_ms)
       - +dt_ms is mandatory (prevents off-by-one)

    Args:
        df: DataFrame with events
        cfg: FixationPostConfig with min_fixation_duration_ms
        event_type_col: Name of the event type column
        event_index_col: Name of the event index column
        time_col: Name of the time column
        discard_target: Target label for discarded fixations

    Returns:
        (df, stats): Modified DataFrame and statistics
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
    
    event_types = df[event_type_col].astype(str).to_numpy().copy()
    event_indices = df[event_index_col].to_numpy(dtype=object).copy()
    times = df[time_col].to_numpy().copy()
    n = len(df)
    
    # 1) Compute a robust dt_ms from time_ms (median of positive differences)
    dt_ms = estimate_sample_interval_ms(times)

    # 2) Find all fixation events using event_type and event_index
    fixation_events: List[Dict[str, Any]] = []

    # Group fixation events
    current_event_index = None
    start_idx = None
    
    for i in range(n):
        is_fixation = event_types[i] == "Fixation"
        event_idx = event_indices[i]
        
        if is_fixation and event_idx is not None and not pd.isna(event_idx):
            event_idx_int = int(event_idx)
            
            if current_event_index is None or current_event_index != event_idx_int:
                # A new event begins
                if start_idx is not None:
                    # Finalize the previous event
                    end_idx = i - 1
                    t_first = float(times[start_idx])
                    t_last = float(times[end_idx])
                    duration_ms = event_duration_ms(times[start_idx : end_idx + 1], sample_interval_ms=dt_ms)
                    
                    fixation_events.append({
                        "event_index": current_event_index,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "t_first": t_first,
                        "t_last": t_last,
                        "duration_ms": duration_ms,
                        "num_samples": end_idx - start_idx + 1,
                    })
                
                # Start the new event
                current_event_index = event_idx_int
                start_idx = i
        else:
            # Not a fixation sample or no index
            if start_idx is not None:
                # Finalize the previous event
                end_idx = i - 1
                t_first = float(times[start_idx])
                t_last = float(times[end_idx])
                duration_ms = event_duration_ms(times[start_idx : end_idx + 1], sample_interval_ms=dt_ms)
                
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
    
    # Handle the last event
    if start_idx is not None:
        end_idx = n - 1
        t_first = float(times[start_idx])
        t_last = float(times[end_idx])
        duration_ms = event_duration_ms(times[start_idx : end_idx + 1], sample_interval_ms=dt_ms)
        
        fixation_events.append({
            "event_index": current_event_index,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "t_first": t_first,
            "t_last": t_last,
            "duration_ms": duration_ms,
            "num_samples": end_idx - start_idx + 1,
        })
    
    # 3) Discard short fixations
    discarded_fixations = 0
    discarded_samples = 0
    threshold_ms = cfg.min_fixation_duration_ms

    for event in fixation_events:
        if event["duration_ms"] < threshold_ms:
            # Discard this event: set all samples to discard_target
            start = event["start_idx"]
            end = event["end_idx"]

            for j in range(start, end + 1):
                if event_types[j] == "Fixation":
                    event_types[j] = discard_target
                    event_indices[j] = None  # Remove the event index
                    discarded_samples += 1
            
            discarded_fixations += 1
            event["discarded"] = True
        else:
            event["discarded"] = False
    
    # 4) Update the DataFrame
    df[event_type_col] = event_types
    df[event_index_col] = event_indices

    # 5) Statistics
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
