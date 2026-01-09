# ivt_filter/core/classification.py
from __future__ import annotations

from typing import Optional, List
import math
import pandas as pd
import numpy as np

from ..config import OlsenVelocityConfig as IVTClassifierConfig
from ..strategies import Ray3DAngle
from ..strategies.coordinate_rounding import NoRounding


class SampleValidator:
    """Validates eye tracking sample validity.
    
    Checks if combined validity flags indicate valid eye tracking data.
    """

    @staticmethod
    def is_invalid(val) -> bool:
        """Check if a validity value indicates invalid data."""
        if val is None or val is False or val == 0:
            return True
        
        try:
            if isinstance(val, float) and not math.isfinite(val):
                return True
        except Exception:
            pass
        
        return False

    def is_valid(self, row: pd.Series) -> bool:
        """Check if sample has valid eye tracking data."""
        combined_valid = row.get("combined_valid", True)
        
        if self.is_invalid(combined_valid):
            return False
        
        if isinstance(combined_valid, bool) and not combined_valid:
            return False
        
        return True


class VelocityValidator:
    """Validates and parses velocity values.
    
    Handles various representations of invalid/missing velocity data.
    """

    @staticmethod
    def parse_velocity(value) -> Optional[float]:
        """Parse velocity value from various formats.
        
        Returns:
            Parsed float value or None if invalid/missing
        """
        if value is None:
            return None
        
        if isinstance(value, str):
            v_str = value.strip().lower()
            if v_str in ("nan", "none", "n/a", ""):
                return None
            v_str = v_str.replace(",", ".")
            try:
                return float(v_str)
            except (ValueError, TypeError):
                return None
        
        try:
            v_float = float(value)
            return v_float if math.isfinite(v_float) else None
        except (ValueError, TypeError):
            return None


class IVTClassifier:
    """Classifies eye tracking samples using velocity thresholding.
    
    Implements the I-VT (Velocity-Threshold Identification) algorithm
    with optional near-threshold hybrid strategy and eye-position jump rule.
    """

    def __init__(self, cfg: Optional[IVTClassifierConfig] = None):
        self.cfg = cfg or IVTClassifierConfig()
        self.sample_validator = SampleValidator()
        self.velocity_validator = VelocityValidator()
        self.velocity_strategy = Ray3DAngle()  # For alternative velocity
        self.coord_rounding = NoRounding()  # Match velocity computation
        self._df_context = None  # Store df for alternative velocity computation

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify samples as Fixation, Saccade, or Unclassified.
        
        Args:
            df: DataFrame with 'velocity_deg_per_sec' column
            
        Returns:
            DataFrame with added classification columns
        """
        if "velocity_deg_per_sec" not in df.columns:
            raise ValueError("DataFrame must contain 'velocity_deg_per_sec'")

        df = df.copy()
        self._df_context = df  # Store for alternative velocity access

        # Pre-compute neighbor support for high velocities in invalid windows
        # (prevents single-sample spikes from invalid-eye windows turning into saccades)
        df["window_any_invalid"] = df.get("window_any_invalid", False)
        df["velocity_neighbor_support"] = False
        velocities = df["velocity_deg_per_sec"].to_numpy()
        invalid_flags = df["window_any_invalid"].fillna(False).to_numpy().astype(bool)
        threshold = float(self.cfg.velocity_threshold_deg_per_sec)
        support = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            v = velocities[i]
            if not invalid_flags[i] or not np.isfinite(v) or v < threshold:
                continue
            prev_ok = i > 0 and np.isfinite(velocities[i - 1]) and velocities[i - 1] >= threshold
            next_ok = i + 1 < len(df) and np.isfinite(velocities[i + 1]) and velocities[i + 1] >= threshold
            support[i] = prev_ok or next_ok
        df["velocity_neighbor_support"] = support
        
        # Add diagnostic columns for refinement
        if self.cfg.enable_near_threshold_hybrid or self.cfg.enable_eye_jump_rule:
            df["velocity_refined"] = pd.NA
            df["refinement_applied"] = ""
            
            # Pre-compute alternative velocities where needed
            df = self._precompute_refinements(df)
        
        df["ivt_sample_type"] = df.apply(self._classify_sample, axis=1)

        df = rebuild_ivt_events_from_sample_types(
            df,
            sample_col="ivt_sample_type",
            event_type_col="ivt_event_type",
            event_index_col="ivt_event_index",
        )
        
        self._df_context = None  # Clean up
        return df
    
    def _precompute_refinements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute alternative velocities and eye-jump detection for all samples."""
        n = len(df)
        
        # Get required columns
        try:
            times = df["time_ms"].to_numpy()
            gaze_x = df["smoothed_x_mm"].to_numpy()
            gaze_y = df["smoothed_y_mm"].to_numpy()
            eye_x = df["eye_x_mm"].to_numpy()
            eye_y = df["eye_y_mm"].to_numpy()
            eye_z = df["eye_z_mm"].to_numpy()
            valid = df["combined_valid"].to_numpy()
            velocities_base = df["velocity_deg_per_sec"].to_numpy()
        except KeyError:
            # Missing required columns, skip refinement
            return df
        
        # Arrays to store results
        velocities_alt = np.full(n, np.nan)
        eye_jump_flags = np.zeros(n, dtype=bool)
        refinement_reasons = [""] * n
        
        # Check if we have window_width_samples column to infer window structure
        has_window_width = "window_width_samples" in df.columns
        
        for i in range(n):
            if not valid[i]:
                continue
            
            velocity_base = velocities_base[i]
            if not np.isfinite(velocity_base):
                continue
            
            # Infer window indices (symmetric around i based on window_width_samples)
            if has_window_width:
                window_width = df.at[i, "window_width_samples"]
                if pd.notna(window_width):
                    half_width = int(window_width) // 2
                    first_idx = max(0, i - half_width)
                    last_idx = min(n - 1, i + half_width)
                else:
                    continue
            else:
                # Default: use 3-sample window (i-1, i, i+1)
                first_idx = max(0, i - 1)
                last_idx = min(n - 1, i + 1)
            
            if first_idx >= last_idx:
                continue
            
            # Compute alternative velocity using per-sample eye positions
            try:
                x1, y1 = float(gaze_x[first_idx]), float(gaze_y[first_idx])
                x2, y2 = float(gaze_x[last_idx]), float(gaze_y[last_idx])
                
                # Use eye positions AT THE ENDPOINT SAMPLES (not middle)
                eye_x1 = float(eye_x[first_idx]) if np.isfinite(eye_x[first_idx]) else 0.0
                eye_y1 = float(eye_y[first_idx]) if np.isfinite(eye_y[first_idx]) else 0.0
                eye_z1 = float(eye_z[first_idx]) if np.isfinite(eye_z[first_idx]) and eye_z[first_idx] > 0 else 600.0
                
                eye_x2 = float(eye_x[last_idx]) if np.isfinite(eye_x[last_idx]) else 0.0
                eye_y2 = float(eye_y[last_idx]) if np.isfinite(eye_y[last_idx]) else 0.0
                eye_z2 = float(eye_z[last_idx]) if np.isfinite(eye_z[last_idx]) and eye_z[last_idx] > 0 else 600.0
                
                # Compute ray from first eye position to first gaze point
                ray1_x = x1 - eye_x1
                ray1_y = y1 - eye_y1
                ray1_z = 0.0 - eye_z1
                
                # Compute ray from second eye position to second gaze point
                ray2_x = x2 - eye_x2
                ray2_y = y2 - eye_y2
                ray2_z = 0.0 - eye_z2
                
                # Dot product and norms
                dot_product = ray1_x * ray2_x + ray1_y * ray2_y + ray1_z * ray2_z
                norm1 = math.sqrt(ray1_x**2 + ray1_y**2 + ray1_z**2)
                norm2 = math.sqrt(ray2_x**2 + ray2_y**2 + ray2_z**2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_theta = dot_product / (norm1 * norm2)
                    cos_theta = max(-1.0, min(1.0, cos_theta))
                    angle_rad = math.acos(cos_theta)
                    angle_deg = math.degrees(angle_rad)
                    
                    t1 = float(times[first_idx])
                    t2 = float(times[last_idx])
                    dt_ms = t2 - t1
                    if dt_ms > 0.1:
                        dt_s = dt_ms / 1000.0
                        velocity_alt = angle_deg / dt_s
                        velocities_alt[i] = round(velocity_alt, 1)
                    
            except (ValueError, IndexError):
                pass
            
            # Detect eye-position jump
            if self.cfg.enable_eye_jump_rule:
                try:
                    eye_x_first = float(eye_x[first_idx])
                    eye_y_first = float(eye_y[first_idx])
                    eye_z_first = float(eye_z[first_idx])
                    eye_x_last = float(eye_x[last_idx])
                    eye_y_last = float(eye_y[last_idx])
                    eye_z_last = float(eye_z[last_idx])
                    
                    if all(np.isfinite([eye_x_first, eye_y_first, eye_z_first, 
                                       eye_x_last, eye_y_last, eye_z_last])):
                        eye_displacement = math.sqrt(
                            (eye_x_last - eye_x_first)**2 +
                            (eye_y_last - eye_y_first)**2 +
                            (eye_z_last - eye_z_first)**2
                        )
                        if eye_displacement >= self.cfg.eye_jump_threshold_mm:
                            eye_jump_flags[i] = True
                except (ValueError, IndexError):
                    pass
        
        # Determine which samples need refinement
        for i in range(n):
            if not np.isfinite(velocities_base[i]):
                continue
            
            velocity_base = velocities_base[i]
            velocity_alt = velocities_alt[i]
            
            if not np.isfinite(velocity_alt):
                continue  # No alternative available
            
            # Stage 1: Near-threshold hybrid
            if self.cfg.enable_near_threshold_hybrid:
                # Check asymmetric or symmetric band
                threshold = self.cfg.velocity_threshold_deg_per_sec
                if self.cfg.near_threshold_band_lower is not None and self.cfg.near_threshold_band_upper is not None:
                    # Asymmetric band
                    lower_bound = threshold - self.cfg.near_threshold_band_lower
                    upper_bound = threshold + self.cfg.near_threshold_band_upper
                    near_threshold = lower_bound <= velocity_base <= upper_bound
                else:
                    # Symmetric band
                    near_threshold = abs(velocity_base - threshold) <= self.cfg.near_threshold_band
                
                if near_threshold:
                    # Choose strategy
                    if self.cfg.near_threshold_strategy == 'inverse':
                        # Inverse hybrid: use velocity farther from threshold (more confident)
                        dist_base = abs(velocity_base - self.cfg.velocity_threshold_deg_per_sec)
                        dist_alt = abs(velocity_alt - self.cfg.velocity_threshold_deg_per_sec)
                        margin = self.cfg.near_threshold_confidence_margin
                        # Direction guard: require same side if configured
                        same_side = (velocity_base - self.cfg.velocity_threshold_deg_per_sec) * (
                            velocity_alt - self.cfg.velocity_threshold_deg_per_sec
                        ) >= 0
                        max_delta_ok = abs(velocity_alt - velocity_base) <= self.cfg.near_threshold_max_delta

                        neighbor_support = True
                        if self.cfg.near_threshold_neighbor_check and (velocity_alt >= self.cfg.velocity_threshold_deg_per_sec) != (velocity_base >= self.cfg.velocity_threshold_deg_per_sec):
                            # Evaluate neighbor base velocities for majority support of alt side
                            alt_side = velocity_alt >= self.cfg.velocity_threshold_deg_per_sec
                            supports = []
                            for j in (i - 1, i + 1):
                                if 0 <= j < len(velocities_base):
                                    vb = velocities_base[j]
                                    if np.isfinite(vb):
                                        supports.append(vb >= self.cfg.velocity_threshold_deg_per_sec)
                            if supports:
                                neighbor_support = sum(1 for s in supports if s == alt_side) >= (len(supports) + 1) // 2
                            else:
                                neighbor_support = False

                        if (not self.cfg.near_threshold_require_same_side or same_side) and max_delta_ok and neighbor_support:
                            # Switch only if alternative is sufficiently farther from threshold
                            if dist_alt >= dist_base + margin:
                                df.at[i, "velocity_refined"] = velocity_alt
                                refinement_reasons[i] = "near_threshold_inverse"
                            # else keep base velocity
                    else:
                        # Default/replace: always use alternative velocity
                        df.at[i, "velocity_refined"] = velocity_alt
                        refinement_reasons[i] = "near_threshold"
                    continue
            
            # Rule A: Eye-position jump correction
            if self.cfg.enable_eye_jump_rule:
                is_clear_saccade = velocity_base >= self.cfg.eye_jump_velocity_threshold
                if eye_jump_flags[i] and is_clear_saccade:
                    df.at[i, "velocity_refined"] = velocity_alt
                    refinement_reasons[i] = "eye_jump"
        
        df["refinement_applied"] = refinement_reasons
        return df

    def _classify_sample(self, row: pd.Series) -> str:
        """Classify a single sample with optional refinement."""
        if not self.sample_validator.is_valid(row):
            return "EyesNotFound"
        
        velocity_base = self.velocity_validator.parse_velocity(
            row["velocity_deg_per_sec"]
        )
        
        if velocity_base is None:
            return "Unclassified"
        
        # Use refined velocity if available
        velocity_final = velocity_base
        if "velocity_refined" in row and pd.notna(row["velocity_refined"]):
            velocity_final = float(row["velocity_refined"])
        
        # Dynamic threshold: lower for invalid windows only in low-velocity band (<35)
        threshold = self.cfg.velocity_threshold_deg_per_sec
        invalid_window = bool(row.get("window_any_invalid", False))
        neighbor_support = bool(row.get("velocity_neighbor_support", True))

        # Confident mismatch switch: if base is far from threshold and alternative disagrees, adopt alternative
        if getattr(self.cfg, "enable_confident_switch", False):
            alt_v = row.get("velocity_alt_deg_per_sec", None)
            alt_v = self.velocity_validator.parse_velocity(alt_v)
            if alt_v is not None and math.isfinite(alt_v):
                base_side = velocity_final >= threshold
                alt_side = alt_v >= threshold
                if invalid_window and alt_side and not neighbor_support:
                    alt_side = False
                dist_base = abs(velocity_final - threshold)
                dist_alt = abs(alt_v - threshold)
                margin = getattr(self.cfg, "confident_switch_margin_deg", 4.0)
                if invalid_window:
                    # In invalid windows, switch on disagreement; require neighbor support for alt saccades
                    if alt_side != base_side and not (alt_side and not neighbor_support):
                        velocity_final = alt_v
                else:
                    # Default confident switch: alt must disagree, be confident, and farther from threshold than base
                    if dist_alt >= margin and alt_side != base_side and dist_alt > dist_base:
                        velocity_final = alt_v
        if invalid_window:
            if velocity_final < 35.0:
                threshold = min(threshold, 27.0)

        # Hysteresis: require stronger drop to switch to Fixation
        hysteresis_delta = 2.0
        saccade_cut = threshold
        fixation_cut = max(0.0, threshold - hysteresis_delta)

        if velocity_final >= saccade_cut:
            if invalid_window and not neighbor_support:
                # Require neighbor confirmation when window contains invalid eyes
                pass
            else:
                return "Saccade"
        if velocity_final < fixation_cut:
            return "Fixation"
        # In-between band: default to Fixation to avoid propagating stale labels
        return "Fixation"


def expand_gt_events_to_samples(
    df: pd.DataFrame,
    event_type_col: str = "gt_event_type",
    event_index_col: str = "gt_event_index",
    sample_type_col: str = "gt_sample_type",
) -> pd.DataFrame:
    """Expand event-based ground truth labels to sample-based labels."""
    if event_type_col not in df.columns:
        return df
    
    df = df.copy()
    df[sample_type_col] = df[event_type_col].astype(str)
    return df


def apply_ivt_classifier(
    df: pd.DataFrame,
    cfg: Optional[IVTClassifierConfig] = None,
) -> pd.DataFrame:
    """Apply I-VT velocity threshold classifier.
    
    Legacy function wrapper for backward compatibility.
    """
    classifier = IVTClassifier(cfg)
    return classifier.classify(df)


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
