# ivt_filter/config.py
"""
Configuration classes for the I-VT filter pipeline.

This module defines all configuration parameters for:
  - Velocity calculation (Olsen-style with extensions)
  - I-VT classification (fixation/saccade detection)
  - Post-processing (saccade merging, fixation filtering)

Example:
    >>> from config import OlsenVelocityConfig, IVTClassifierConfig
    >>>
    >>> # Standard configuration
    >>> vel_cfg = OlsenVelocityConfig(
    ...     window_length_ms=20.0,
    ...     velocity_method="olsen2d",
    ...     eye_mode="average"
    ... )
    >>>
    >>> # 3D ray method with coordinate rounding
    >>> vel_cfg = OlsenVelocityConfig(
    ...     window_length_ms=20.0,
    ...     velocity_method="ray3d",
    ...     coordinate_rounding="halfup",
    ...     smoothing_mode="median",
    ...     smoothing_window_samples=5
    ... )
    >>>
    >>> # Classifier
    >>> clf_cfg = IVTClassifierConfig(
    ...     velocity_threshold_deg_per_sec=30.0
    ... )
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal, Optional

from .window_policy import (
    FixedSampleWindowPolicy,
    ShiftedValidWindowPolicy,
    TobiiWindowPolicy,
    WindowPolicy,
    translate_legacy_window_flags,
    window_policy_from_dict,
)


def _require_choice(name: str, value: object, choices: tuple[object, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _require_non_negative(name: str, value: float) -> None:
    if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(name: str, value: float) -> None:
    if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_odd_samples(name: str, value: int) -> None:
    _require_positive_integer(name, value)
    if value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")


def _require_fixed_window_samples(name: str, value: int, *, allow_even: bool) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 3:
        raise ValueError(f"{name} must be an integer >= 3")
    if not allow_even and value % 2 == 0:
        raise ValueError(f"{name} must be odd unless allow_asymmetric_window=True")


@dataclass
class OlsenVelocityConfig:
    """
    Configuration for the Olsen-style velocity calculation.
    """

    # Time window in milliseconds (Olsen-style)
    window_length_ms: float = 20.0

    # Timestamp column and unit (allows microsecond precision when available)
    time_column: str = "time_ms"
    time_unit: Literal["ms", "us", "ns"] = "ms"

    # Eye mode
    # - "left":           left eye only
    # - "right":          right eye only
    # - "average":        average of both eyes (default, falls back to one eye)
    # - "strict_average": average only if BOTH eyes are valid, otherwise NaN
    eye_mode: Literal["left", "right", "average", "strict_average"] = "average"

    # Validity threshold for the Tobii codes
    max_validity: int = 1

    # Minimum time difference for a valid velocity calculation
    min_dt_ms: float = 0.1

    # Spatial smoothing on the combined coordinates
    # - "none": no smoothing
    # - "median": median filter (ignores invalid samples)
    # - "moving_average": moving average (ignores invalid samples)
    # - "median_strict": Median only if ALL samples in window are valid
    # - "moving_average_strict": Average only if ALL samples in window are valid
    # - "median_adaptive": Collects only valid samples, expands search if needed
    # - "moving_average_adaptive": Same as median_adaptive, but with mean
    smoothing_mode: Literal[
        "none", "median", "moving_average", 
        "median_strict", "moving_average_strict",
        "median_adaptive", "moving_average_adaptive"
    ] = "none"
    smoothing_window_samples: int = 5
    
    # Adaptive Smoothing Optionen
    # min_samples: Minimum number of valid samples for smoothing (default: 1)
    # expansion_radius: Search samples beyond standard window (default: 0)
    smoothing_min_samples: int = 1
    smoothing_expansion_radius: int = 0

    # Normalized, tagged selector policy. Legacy flags below remain temporarily
    # supported for direct callers and are normalized during initialization.
    window_policy: Optional[WindowPolicy] = None

    # Sample-symmetric window within the time window
    # (equal number of samples left/right, but still bounded by window_length_ms)
    sample_symmetric_window: bool = False

    # Fixed window size in samples (optional, odd >= 3).
    # When set, a pure sample-window strategy is used
    # (FixedSampleSymmetricWindowSelector) and window_length_ms only serves
    # as a reference for the dt minimum etc.
    fixed_window_samples: Optional[int] = None

    # If True: automatically compute a suitable fixed window size
    # in samples from `window_length_ms` and the observed sampling interval.
    auto_fixed_window_from_ms: bool = False

    # Asymmetric window: if True, per_side = round(window_size / 2) is used
    # instead of (window_size - 1) / 2. Also allows even window sizes.
    # Example: window_size=8 -> per_side=4 (instead of rounding up to 9)
    allow_asymmetric_window: bool = False

    # Shifted valid window: keep the window length constant and shift the window
    # to find a continuously valid block (no invalids inside the window).
    # Requires fixed_window_samples.
    shifted_valid_window: bool = False
    # Fallback when no valid window of the fixed length is found:
    # - "shrink": falls back to the previous shrink behavior
    # - "unclassified": returns no window (sample is later marked unclassified)
    shifted_valid_fallback: Literal["shrink", "unclassified"] = "shrink"

    # Asymmetric neighbor window (only 2 directly adjacent samples):
    # If True: use AsymmetricNeighborWindowSelector
    # - Priority: backward (i-1 -> i)
    # - Fallback: forward (i -> i+1)
    # - Gap rule: 2 samples = 1 sample radius
    asymmetric_neighbor_window: bool = False

    # Use fixed dt from the sampling rate (for asymmetric_neighbor_window):
    # If True: dt = 1/sampling_rate (precise, without time_ms jitter)
    # If False: dt from time_ms differences (can jitter due to rounding)
    use_fixed_dt: bool = False

    # Symmetric rounding logic: determine per_side = round(window_size / 2)
    # and then use 2*per_side + 1 as the effective window size (symmetric around the center).
    # This logic can increase the total size (e.g. 7 -> 9), but is independent
    # of the unclassified (gap) rule, which still uses the original size.
    symmetric_round_window: bool = False

    # Method for determining the sampling rate
    # - "all_samples": use all samples (default, more robust)
    # - "first_100": use only the first 100 samples (as in the Tobii paper)
    sampling_rate_method: Literal["all_samples", "first_100"] = "first_100"

    # Method for computing the time differences
    # - "median": more robust against outliers (default)
    # - "mean": arithmetic mean (as mentioned in the Tobii paper)
    dt_calculation_method: Literal["median", "mean"] = "mean"

    # For invalid first/last samples in the window: use the nearest valid sample
    use_fallback_valid_samples: bool = True

    # For fixed_window_samples: if the window edge has invalid samples,
    # use the velocity from the nearest sample with a valid window
    fixed_window_edge_fallback: bool = False

    # Strategies for eye_mode="average"
    # - average_window_single_eye:
    #       for mixed mono/binocular cases, the eye with more stable validity
    #       is used for the start/end of the window
    # - average_window_impute_neighbor:
    #       impute a missing eye coordinate at the window edge from the nearest
    #       neighbor (with a valid eye)
    # - average_fallback_single_eye:
    #       If only one eye is valid across the whole window (start to end) or
    #       at the middle sample, only that single eye is used CONSISTENTLY
    #       (no average). Prevents parallax effects when switching eyes.
    average_window_single_eye: bool = False
    average_window_impute_neighbor: bool = False
    average_fallback_single_eye: bool = False

    # Gap filling: temporally interpolate short gaps in the eye tracks
    # (per eye) before the eyes are combined.
    gap_fill_enabled: bool = False
    gap_fill_max_gap_ms: float = 75.0  # e.g. fill gaps up to 75 ms linearly

    # Coordinate rounding before the velocity calculation
    # - "none": no rounding (default)
    # - "nearest": banker's rounding (round, ties to even)
    # - "halfup": always round up on 0.5 (classic rounding)
    # - "floor": always round down
    # - "ceil": always round up
    coordinate_rounding: Literal["none", "nearest", "halfup", "floor", "ceil"] = "none"

    # Method for computing the visual angle
    # - "olsen2d": original Olsen 2D approximation: theta = atan(screen_distance / eye_z)
    #              fast, requires only eye_z, 2D approximation
    #              default for backward compatibility
    # - "ray3d": physically correct 3D ray method:
    #            theta = acos(ray0 . ray1 / (|ray0| x |ray1|))
    #            more precise, requires the full eye position (x, y, z)
    #            typically 1-5% lower velocities than olsen2d
    # - "ray3d_gaze_dir": uses the normalized gaze direction vectors (DACS norm)
    #            theta = acos(dir0 . dir1); needs no screen or eye position
    # - "tobii_gaze_dir": Tobii-exact formula from decompiled source code
    #            theta = 2*asin(||v1-v2||/2) -- numerically more stable than acos(dot product)
    #            requires normalized direction vectors (DACS norm), like ray3d_gaze_dir
    velocity_method: Literal["olsen2d", "ray3d", "ray3d_gaze_dir", "tobii_gaze_dir"] = "olsen2d"

    # Tobii-exact window (GazeVelocityCalculator):
    # If True, TobiiGazeVelocityWindowSelector is used:
    #   window_samples = floor(window_length_ms / tobii_sample_interval_ms * 1.01) + 1
    # Overrides all other window selectors (highest priority).
    tobii_window_mode: bool = False
    # Nominal sampling interval in ms for the Tobii window calculation.
    # Examples: 16.67 = 60 Hz, 8.33 = 120 Hz, 4.17 = 240 Hz
    # Computed automatically from the sampling rate when not set (None).
    tobii_sample_interval_ms: Optional[float] = None

    # Tobii-exact eye offset interpolation:
    # If True, the last known L->R gaze/eye-position offset is stored
    # and used to estimate the missing eye (instead of a simple fallback).
    # Corresponds to: RemoteTrackerGazeDataToRecordedTwoEyedGazeDataConverter (Tobii C#)
    tobii_eye_offset_interpolation: bool = False

    def __post_init__(self) -> None:
        """Validate values and normalize deprecated selector flags into one policy."""
        _require_positive("window_length_ms", self.window_length_ms)
        _require_positive("min_dt_ms", self.min_dt_ms)
        _require_non_negative_integer("max_validity", self.max_validity)
        _require_positive_odd_samples("smoothing_window_samples", self.smoothing_window_samples)
        _require_positive_integer("smoothing_min_samples", self.smoothing_min_samples)
        _require_non_negative_integer("smoothing_expansion_radius", self.smoothing_expansion_radius)
        _require_non_negative("gap_fill_max_gap_ms", self.gap_fill_max_gap_ms)
        _require_choice("time_unit", self.time_unit, ("ms", "us", "ns"))
        _require_choice("eye_mode", self.eye_mode, ("left", "right", "average", "strict_average"))
        _require_choice("smoothing_mode", self.smoothing_mode, (
            "none", "median", "moving_average", "median_strict",
            "moving_average_strict", "median_adaptive", "moving_average_adaptive",
        ))
        _require_choice("sampling_rate_method", self.sampling_rate_method, ("all_samples", "first_100"))
        _require_choice("dt_calculation_method", self.dt_calculation_method, ("median", "mean"))
        _require_choice("coordinate_rounding", self.coordinate_rounding, ("none", "nearest", "halfup", "floor", "ceil"))
        _require_choice("velocity_method", self.velocity_method, ("olsen2d", "ray3d", "ray3d_gaze_dir", "tobii_gaze_dir"))
        _require_choice("shifted_valid_fallback", self.shifted_valid_fallback, ("shrink", "unclassified"))
        if self.fixed_window_samples is not None:
            _require_fixed_window_samples(
                "fixed_window_samples", self.fixed_window_samples,
                allow_even=self.allow_asymmetric_window,
            )

        legacy_policy = translate_legacy_window_flags(
            time_symmetric_window=False,
            sample_symmetric_window=self.sample_symmetric_window,
            fixed_window_samples=self.fixed_window_samples,
            auto_fixed_window_from_ms=self.auto_fixed_window_from_ms,
            symmetric_round_window=self.symmetric_round_window,
            asymmetric_neighbor_window=self.asymmetric_neighbor_window,
            shifted_valid_window=self.shifted_valid_window,
            shifted_valid_fallback=self.shifted_valid_fallback,
            tobii_window_mode=self.tobii_window_mode,
            tobii_sample_interval_ms=self.tobii_sample_interval_ms,
        )
        if isinstance(self.window_policy, dict):
            self.window_policy = window_policy_from_dict(self.window_policy)
        if self.window_policy is None:
            self.window_policy = legacy_policy or TobiiWindowPolicy(
                sample_interval_ms=self.tobii_sample_interval_ms
            )
        elif legacy_policy is not None and self.window_policy != legacy_policy:
            raise ValueError(
                "window_policy cannot be combined with contradictory deprecated legacy window flags."
            )

        policy = self.window_policy
        if isinstance(policy, (FixedSampleWindowPolicy, ShiftedValidWindowPolicy)):
            _require_choice("window_policy.fallback", getattr(policy, "fallback", "shrink"), ("shrink", "unclassified"))
            if policy.samples is not None:
                _require_fixed_window_samples(
                    "window_policy.samples", policy.samples,
                    allow_even=self.allow_asymmetric_window,
                )
        if isinstance(policy, TobiiWindowPolicy):
            if policy.sample_interval_ms is not None:
                _require_positive(
                    "tobii_sample_interval_ms/sample_interval_ms",
                    policy.sample_interval_ms,
                )


@dataclass
class IVTClassifierConfig:
    """
    Configuration for the I-VT threshold classifier.
    """

    velocity_threshold_deg_per_sec: float = 30.0

    # Optional reconstruction heuristics. Disabled by default so the classifier
    # behaves as a strict I-VT velocity-threshold classifier.
    # Require an adjacent above-threshold velocity before classifying an
    # invalid-window sample as a saccade.
    enable_invalid_window_neighbor_confirmation: bool = False
    # Retain the previous motion label while velocity is in the band immediately
    # below the saccade threshold.
    enable_hysteresis: bool = False
    hysteresis_width_deg_per_sec: float = 2.0
    
    # Stage 1: Near-threshold hybrid strategy
    # Enable hybrid classification near threshold using alternative velocity
    enable_near_threshold_hybrid: bool = False
    # Band around threshold (deg/s) where alternative velocity is used
    near_threshold_band: float = 5.0
    # Asymmetric band: lower band (below threshold) - None means use symmetric band
    near_threshold_band_lower: float | None = None
    # Asymmetric band: upper band (above threshold) - None means use symmetric band
    near_threshold_band_upper: float | None = None
    # Hybrid strategy: 'replace' (always use alt), 'inverse' (use velocity farther from threshold)
    near_threshold_strategy: str = 'inverse'
    # Minimum confidence margin (deg/s) required to switch in inverse strategy
    near_threshold_confidence_margin: float = 0.4
    # Require base and alternative velocity to be on the same side of threshold
    near_threshold_require_same_side: bool = True
    # Maximum allowed difference |alt - base| (deg/s) to switch
    near_threshold_max_delta: float = 2.0
    # Neighbor consensus: require majority of neighbors to support alt side when flip occurs
    near_threshold_neighbor_check: bool = False
    
    # Rule A: Eye-position jump correction
    # Enable eye-position jump detection and correction
    enable_eye_jump_rule: bool = False
    # Eye position displacement threshold (mm) to trigger jump rule
    eye_jump_threshold_mm: float = 10.0
    # Velocity threshold for "clear saccade" (deg/s) to apply jump rule
    eye_jump_velocity_threshold: float = 50.0

    # Confident mismatch switch: use alternative velocity method when base is far from threshold
    enable_confident_switch: bool = False
    confident_switch_margin_deg: float = 4.0
    confident_switch_method: Literal["olsen2d", "ray3d", "ray3d_gaze_dir"] = "ray3d_gaze_dir"

    def __post_init__(self) -> None:
        _require_positive("velocity_threshold_deg_per_sec", self.velocity_threshold_deg_per_sec)
        for name in (
            "hysteresis_width_deg_per_sec", "near_threshold_band",
            "near_threshold_confidence_margin", "near_threshold_max_delta",
            "eye_jump_threshold_mm", "eye_jump_velocity_threshold",
            "confident_switch_margin_deg",
        ):
            _require_non_negative(name, getattr(self, name))
        for name in ("near_threshold_band_lower", "near_threshold_band_upper"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(name, value)
        _require_choice("near_threshold_strategy", self.near_threshold_strategy, ("replace", "inverse"))
        _require_choice("confident_switch_method", self.confident_switch_method, ("olsen2d", "ray3d", "ray3d_gaze_dir"))


@dataclass
class SaccadeMergeConfig:
    """
    Configuration for the post-processing of short saccade blocks.
    """

    # Saccade blocks shorter than this duration (ms) that lie completely within
    # GT fixations can be relabeled as fixations.
    max_saccade_block_duration_ms: float = 20.0

    # If True: only merge when the direct GT neighbors (if present)
    # are also fixations.
    require_fixation_context: bool = True

    # Which column to operate on:
    # - "ivt_sample_type": sample-based, then rebuild events
    # - None: directly on "ivt_event_type"
    use_sample_type_column: Optional[str] = "ivt_sample_type"

    def __post_init__(self) -> None:
        _require_positive("max_saccade_block_duration_ms", self.max_saccade_block_duration_ms)


@dataclass
class FixationPostConfig:
    """
    Configuration for the Tobii-like fixation filters:

      1) Merge adjacent fixations
         (temporally close + small angular distance)

      2) Discard short fixations
    """

    # Step 1: merge adjacent fixations
    merge_adjacent_fixations: bool = False
    max_time_gap_ms: float = 75.0   # e.g. 75 ms as in the Tobii paper
    max_angle_deg: float = 0.5      # e.g. 0.5 degrees between fixation centers
    max_gap_velocity_deg_per_sec: float = 35.0  # maximum velocity for gap relabeling

    # Weighting strategy for the averaged fixation position during merge:
    # - "uniform": np.nanmean of all samples (previous behavior)
    # - "sample_count": sample-count-weighted mean
    #   (corresponds to Tobii MergeFixationsFilter: vector2Df / num)
    merge_weighting: Literal["uniform", "sample_count"] = "uniform"

    # Step 2: discard fixations that are too short
    discard_short_fixations: bool = False
    min_fixation_duration_ms: float = 60.0  # e.g. 60 ms minimum duration
    discard_target: Literal["Unclassified", "Saccade"] = "Unclassified"  # target label for discarded fixations

    def __post_init__(self) -> None:
        for name in (
            "max_time_gap_ms", "max_angle_deg", "max_gap_velocity_deg_per_sec",
            "min_fixation_duration_ms",
        ):
            _require_non_negative(name, getattr(self, name))
        _require_choice("merge_weighting", self.merge_weighting, ("uniform", "sample_count"))
        _require_choice("discard_target", self.discard_target, ("Unclassified", "Saccade"))


@dataclass(frozen=True)
class PipelineConfig:
    """Single source of truth for the active core processing stages.

    File I/O, evaluation, and plotting are deliberately runner concerns.  A
    post-processing stage is active exactly when its optional configuration is
    present.
    """

    velocity: OlsenVelocityConfig
    classifier: IVTClassifierConfig
    classify: bool = True
    saccade_merge: Optional[SaccadeMergeConfig] = None
    fixation_post: Optional[FixationPostConfig] = None

    def __post_init__(self) -> None:
        if not self.classify and (self.saccade_merge or self.fixation_post):
            raise ValueError("Post-processing stages require classification")
        if self.fixation_post and not (
            self.fixation_post.merge_adjacent_fixations
            or self.fixation_post.discard_short_fixations
        ):
            raise ValueError("Fixation post-processing config must enable at least one operation")
