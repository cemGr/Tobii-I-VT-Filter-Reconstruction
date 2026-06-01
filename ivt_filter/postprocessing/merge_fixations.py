# ivt_filter/postprocessing/merge_fixations.py
"""Merge adjacent fixations: combines nearby fixations based on time and angle."""

from __future__ import annotations

import logging
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd

from ..config import FixationPostConfig
from ..strategies import Ray3DAngle, Olsen2DApproximation

logger = logging.getLogger(__name__)


def _weighted_fixation_center(
    arr: np.ndarray,
    start: int,
    end: int,
    n_samples: int,
    total_samples: int,
    weighting: str,
) -> float:
    """Berechnet das gewichtete Zentrum einer Fixation.

    Args:
        arr: Koordinaten-Array (x oder y).
        start: Startindex des Fixationsblocks.
        end: Endindex des Fixationsblocks (inklusive).
        n_samples: Anzahl Samples in diesem Block (für sample_count-Gewichtung).
        total_samples: Gesamtzahl Samples über alle gemergten Blöcke.
        weighting: "uniform" für np.nanmean, "sample_count" für gewichtetes Mittel.

    Returns:
        Gewichtetes Zentrum (float).
    """
    if weighting == "sample_count":
        block_mean = float(np.nanmean(arr[start : end + 1]))
        return block_mean * n_samples / total_samples
    return float(np.nanmean(arr[start : end + 1]))


def merge_adjacent_fixations(
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
        logger.info("[MergeFixations] Using Ray3D angle calculation")
    else:
        angle_calculator = Olsen2DApproximation()  # type: ignore[assignment]
        logger.info("[MergeFixations] Using Olsen 2D approximation")

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

        # Zeitabstand zwischen Ende Fix1 und Start Fix2. Dies ist der Abstand
        # zwischen Event-Grenzen, keine inklusive Eventdauer.
        time_gap = float(times[s2]) - float(times[e1])
        if time_gap <= 0 or time_gap > cfg.max_time_gap_ms:
            continue

        # Zentren der Fixationen (gemittelt nach gewählter Strategie)
        weighting = getattr(cfg, "merge_weighting", "uniform")
        n1_samples = e1 - s1 + 1
        n2_samples = e2 - s2 + 1
        total_samples = n1_samples + n2_samples

        if weighting == "sample_count":
            # Tobii-Referenz: Sample-Anzahl-gewichtetes Mittel
            # x_merged = (mean(x_fix1) * n1 + mean(x_fix2) * n2) / (n1 + n2)
            x1_mean = float(np.nanmean(x[s1 : e1 + 1]))
            y1_mean = float(np.nanmean(y[s1 : e1 + 1]))
            x2_mean = float(np.nanmean(x[s2 : e2 + 1]))
            y2_mean = float(np.nanmean(y[s2 : e2 + 1]))
            x1 = (x1_mean * n1_samples + x2_mean * n2_samples) / total_samples
            y1 = (y1_mean * n1_samples + y2_mean * n2_samples) / total_samples
            # Für den Winkeltest repräsentiert x1/y1 nun das gemischte Zentrum;
            # x2/y2 verwenden wir als zweites Fixationszentrum (ungemittelt)
            x2 = x2_mean
            y2 = y2_mean
        else:
            # Standard: einfaches nanmean
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
                # Nur Samples bis zum konfigurierten Velocity-Cap werden gemerged.
                # Verhindert echte Saccade-Samples in Fixations.
                if velocity is not None and not pd.isna(velocity[j]):
                    if velocity[j] > cfg.max_gap_velocity_deg_per_sec:
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
