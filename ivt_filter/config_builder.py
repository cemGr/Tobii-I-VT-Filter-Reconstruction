# ivt_filter/config_builder.py
"""Build configuration objects from CLI arguments.

Separates configuration construction from argument parsing (SRP).
"""
from __future__ import annotations

import argparse

from .config import (
    OlsenVelocityConfig,
    IVTClassifierConfig,
    SaccadeMergeConfig,
    FixationPostConfig,
)


class ConfigBuilder:
    """Builds configuration objects from parsed CLI arguments.
    
    Responsibilities:
        - Map CLI arguments to configuration dataclasses
        - Handle default values and validation
        - Provide single source of truth for config construction
    """

    @staticmethod
    def build_velocity_config(args: argparse.Namespace) -> OlsenVelocityConfig:
        """Build velocity configuration from CLI arguments."""
        return OlsenVelocityConfig(
            window_length_ms=args.window,
            eye_mode=args.eye,
            smoothing_mode=args.smoothing,
            smoothing_window_samples=args.smooth_window_samples,
            smoothing_min_samples=args.smoothing_min_samples,
            smoothing_expansion_radius=args.smoothing_expansion_radius,
            sample_symmetric_window=args.sample_symmetric_window,
            fixed_window_samples=args.fixed_window_samples,
            auto_fixed_window_from_ms=args.auto_fixed_window_from_ms,
            fixed_window_edge_fallback=args.fixed_window_edge_fallback,
            symmetric_round_window=args.symmetric_round_window,
            allow_asymmetric_window=args.allow_asymmetric_window,
            asymmetric_neighbor_window=args.asymmetric_neighbor_window,
            shifted_valid_window=args.shifted_valid_window,
            shifted_valid_fallback=args.shifted_valid_fallback,
            use_fixed_dt=args.use_fixed_dt,
            sampling_rate_method=args.sampling_rate_method,
            dt_calculation_method=args.dt_calculation_method,
            use_fallback_valid_samples=not args.no_fallback_valid_samples,
            average_window_single_eye=args.average_window_single_eye,
            average_window_impute_neighbor=args.average_window_impute_neighbor,
            average_fallback_single_eye=args.average_fallback_single_eye,
            gap_fill_enabled=args.gap_fill,
            gap_fill_max_gap_ms=args.gap_fill_max_ms,
            coordinate_rounding=args.coordinate_rounding,
            velocity_method=args.velocity_method,
        )

    @staticmethod
    def build_classifier_config(args: argparse.Namespace) -> IVTClassifierConfig:
        """Build classifier configuration from CLI arguments."""
        return IVTClassifierConfig(
            velocity_threshold_deg_per_sec=args.threshold,
            enable_near_threshold_hybrid=args.enable_near_threshold_hybrid,
            near_threshold_band=args.near_threshold_band,
            near_threshold_band_lower=args.near_threshold_band_lower,
            near_threshold_band_upper=args.near_threshold_band_upper,
            near_threshold_strategy=args.near_threshold_strategy,
            near_threshold_confidence_margin=args.near_threshold_confidence_margin,
            near_threshold_require_same_side=args.near_threshold_require_same_side,
            near_threshold_max_delta=args.near_threshold_max_delta,
            near_threshold_neighbor_check=args.near_threshold_neighbor_check,
            enable_eye_jump_rule=args.enable_eye_jump_rule,
            eye_jump_threshold_mm=args.eye_jump_threshold,
            eye_jump_velocity_threshold=args.eye_jump_velocity_threshold,
        )

    @staticmethod
    def build_saccade_merge_config(args: argparse.Namespace) -> SaccadeMergeConfig:
        """Build saccade merge configuration from CLI arguments."""
        return SaccadeMergeConfig(
            max_saccade_block_duration_ms=args.post_smoothing_ms,
            require_fixation_context=True,
            use_sample_type_column="ivt_sample_type",
        )

    @staticmethod
    def build_fixation_post_config(args: argparse.Namespace) -> FixationPostConfig:
        """Build fixation post-processing configuration from CLI arguments."""
        return FixationPostConfig(
            min_fixation_duration_ms=args.min_fixation_duration_ms,
            max_time_gap_ms=args.merge_fix_max_gap_ms,
            max_angle_deg=args.merge_fix_max_angle_deg,
            merge_adjacent_fixations=args.merge_close_fixations,
            discard_short_fixations=args.discard_short_fixations,
            discard_target=args.discard_fixation_target,
        )

    @classmethod
    def build_all_configs(
        cls, 
        args: argparse.Namespace
    ) -> tuple[OlsenVelocityConfig, IVTClassifierConfig, SaccadeMergeConfig, FixationPostConfig]:
        """Build all configuration objects from CLI arguments.
        
        Returns:
            Tuple of (velocity_config, classifier_config, saccade_config, fixation_config)
        """
        return (
            cls.build_velocity_config(args),
            cls.build_classifier_config(args),
            cls.build_saccade_merge_config(args),
            cls.build_fixation_post_config(args),
        )
