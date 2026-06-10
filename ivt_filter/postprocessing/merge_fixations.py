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
    """Compute the weighted center of a fixation.

    Args:
        arr: Coordinate array (x or y).
        start: Start index of the fixation block.
        end: End index of the fixation block (inclusive).
        n_samples: Number of samples in this block (for sample_count weighting).
        total_samples: Total number of samples across all merged blocks.
        weighting: "uniform" for np.nanmean, "sample_count" for a weighted mean.

    Returns:
        Weighted center (float).
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
    Merge adjacent fixations when:

      - Time gap between end of Fix1 and start of Fix2 <= max_time_gap_ms
      - Visual angle between the Fix1 center and the Fix2 center
        <= max_angle_deg

    Implementation:
      - The gap between two fixations is relabeled as Fixation.
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
    
    # For Ray3D we also need eye_x and eye_y
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

    # Determine fixation blocks from the labels
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

        # Time gap between end of Fix1 and start of Fix2. This is the distance
        # between event boundaries, not an inclusive event duration.
        time_gap = float(times[s2]) - float(times[e1])
        if time_gap <= 0 or time_gap > cfg.max_time_gap_ms:
            continue

        # Fixation centers (averaged according to the chosen strategy)
        weighting = getattr(cfg, "merge_weighting", "uniform")
        n1_samples = e1 - s1 + 1
        n2_samples = e2 - s2 + 1
        total_samples = n1_samples + n2_samples

        if weighting == "sample_count":
            # Tobii reference: sample-count-weighted mean
            # x_merged = (mean(x_fix1) * n1 + mean(x_fix2) * n2) / (n1 + n2)
            x1_mean = float(np.nanmean(x[s1 : e1 + 1]))
            y1_mean = float(np.nanmean(y[s1 : e1 + 1]))
            x2_mean = float(np.nanmean(x[s2 : e2 + 1]))
            y2_mean = float(np.nanmean(y[s2 : e2 + 1]))
            x1 = (x1_mean * n1_samples + x2_mean * n2_samples) / total_samples
            y1 = (y1_mean * n1_samples + y2_mean * n2_samples) / total_samples
            # For the angle test, x1/y1 now represents the blended center;
            # x2/y2 are used as the second fixation center (not blended)
            x2 = x2_mean
            y2 = y2_mean
        else:
            # Standard: simple nanmean
            x1 = float(np.nanmean(x[s1 : e1 + 1]))
            y1 = float(np.nanmean(y[s1 : e1 + 1]))
            x2 = float(np.nanmean(x[s2 : e2 + 1]))
            y2 = float(np.nanmean(y[s2 : e2 + 1]))

        if any(pd.isna(v) for v in (x1, y1, x2, y2)):
            continue

        # Calculate angle with the chosen strategy
        if use_ray3d and eye_x is not None and eye_y is not None and eye_z is not None:
            # Ray3D: use full 3D geometry
            ex1 = float(np.nanmean(eye_x[s1 : e1 + 1]))
            ey1 = float(np.nanmean(eye_y[s1 : e1 + 1]))
            ez1 = float(np.nanmean(eye_z[s1 : e1 + 1]))
            ex2 = float(np.nanmean(eye_x[s2 : e2 + 1]))
            ey2 = float(np.nanmean(eye_y[s2 : e2 + 1]))
            ez2 = float(np.nanmean(eye_z[s2 : e2 + 1]))
            
            # Means of the eye position
            ex = float(np.nanmean([ex1, ex2]))
            ey = float(np.nanmean([ey1, ey2]))
            ez = float(np.nanmean([ez1, ez2]))
            
            angle = angle_calculator.calculate_visual_angle(x1, y1, x2, y2, ex, ey, ez)
        else:
            # Olsen 2D: use only the Z distance
            if eye_z is not None:
                z1 = float(np.nanmean(eye_z[s1 : e1 + 1]))
                z2 = float(np.nanmean(eye_z[s2 : e2 + 1]))
                z = float(np.nanmean([z1, z2]))
            else:
                z = None
            
            angle = angle_calculator.calculate_visual_angle(x1, y1, x2, y2, None, None, z)
        
        if angle > cfg.max_angle_deg:
            continue

        # Relabel the gap between the two fixations as Fixation
        # IMPORTANT: EyesNotFound must NOT become Fixation
        # OPTIMIZATION: do not include saccade samples (velocity check)
        if s2 > e1 + 1:
            for j in range(e1 + 1, s2):
                if labels[j] == "Fixation" or labels[j] == "EyesNotFound":
                    continue
                
                # Velocity-based gap filling:
                # Only samples up to the configured velocity cap are merged.
                # Prevents real saccade samples from ending up in fixations.
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
