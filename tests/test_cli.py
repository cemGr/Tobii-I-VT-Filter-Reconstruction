from __future__ import annotations

from ivt_filter.cli import build_arg_parser
from ivt_filter.config.config_builder import ConfigBuilder


CENTRAL_CLI_FLAGS = [
    "--input",
    "--output",
    "--time-column",
    "--time-unit",
    "--window",
    "--eye",
    "--smoothing",
    "--smooth-window-samples",
    "--fixed-window-samples",
    "--gap-fill",
    "--gap-fill-max-ms",
    "--velocity-method",
    "--classify",
    "--threshold",
    "--enable-near-threshold-hybrid",
    "--confident-switch-enabled",
    "--post-smoothing-ms",
    "--merge-close-fixations",
    "--discard-short-fixations",
    "--evaluate",
    "--exclude-calibration",
    "--no-plot",
    "--with-events",
]


def _parser_flags() -> set[str]:
    return {
        option
        for action in build_arg_parser()._actions
        for option in action.option_strings
    }


def test_build_arg_parser_exposes_central_flags_snapshot() -> None:
    flags = _parser_flags()

    assert [flag for flag in CENTRAL_CLI_FLAGS if flag not in flags] == []


def test_config_builder_accepts_cli_namespace_after_refactor() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--input",
            "input.tsv",
            "--output",
            "output.tsv",
            "--time-column",
            "time_us",
            "--time-unit",
            "us",
            "--window",
            "24",
            "--eye",
            "left",
            "--smoothing",
            "median",
            "--smooth-window-samples",
            "7",
            "--smoothing-min-samples",
            "2",
            "--smoothing-expansion-radius",
            "3",
            "--fixed-window-samples",
            "5",
            "--fixed-window-edge-fallback",
            "--allow-asymmetric-window",
            "--use-fixed-dt",
            "--sampling-rate-method",
            "all_samples",
            "--dt-calculation-method",
            "mean",
            "--no-fallback-valid-samples",
            "--average-window-single-eye",
            "--average-window-impute-neighbor",
            "--average-fallback-single-eye",
            "--gap-fill",
            "--gap-fill-max-ms",
            "42",
            "--coordinate-rounding",
            "halfup",
            "--velocity-method",
            "ray3d_gaze_dir",
            "--tobii-eye-offset-interpolation",
            "--classify",
            "--threshold",
            "31",
            "--enable-invalid-window-neighbor-confirmation",
            "--enable-hysteresis",
            "--hysteresis-width",
            "1.5",
            "--enable-near-threshold-hybrid",
            "--near-threshold-band",
            "6",
            "--near-threshold-band-lower",
            "4",
            "--near-threshold-band-upper",
            "8",
            "--near-threshold-strategy",
            "replace",
            "--near-threshold-confidence-margin",
            "0.4",
            "--near-threshold-require-same-side",
            "--near-threshold-max-delta",
            "3",
            "--near-threshold-neighbor-check",
            "--confident-switch-enabled",
            "--confident-switch-margin-deg",
            "5",
            "--confident-switch-method",
            "ray3d",
            "--enable-eye-jump-rule",
            "--eye-jump-threshold",
            "12",
            "--eye-jump-velocity-threshold",
            "55",
            "--post-smoothing-ms",
            "10",
            "--merge-close-fixations",
            "--discard-short-fixations",
        ]
    )

    pipeline_config = ConfigBuilder.build_pipeline_config(args)

    assert pipeline_config.classify is True
    assert pipeline_config.saccade_merge is not None
    assert pipeline_config.fixation_post is not None
    assert pipeline_config.velocity.window_length_ms == 24
    assert pipeline_config.velocity.time_column == "time_us"
    assert pipeline_config.velocity.time_unit == "us"
    assert pipeline_config.velocity.eye_mode == "left"
    assert pipeline_config.velocity.smoothing_mode == "median"
    assert pipeline_config.velocity.smoothing_window_samples == 7
    assert pipeline_config.velocity.smoothing_min_samples == 2
    assert pipeline_config.velocity.smoothing_expansion_radius == 3
    assert pipeline_config.velocity.fixed_window_edge_fallback is True
    assert pipeline_config.velocity.allow_asymmetric_window is True
    assert pipeline_config.velocity.use_fixed_dt is True
    assert pipeline_config.velocity.sampling_rate_method == "all_samples"
    assert pipeline_config.velocity.dt_calculation_method == "mean"
    assert pipeline_config.velocity.use_fallback_valid_samples is False
    assert pipeline_config.velocity.average_window_single_eye is True
    assert pipeline_config.velocity.average_window_impute_neighbor is True
    assert pipeline_config.velocity.average_fallback_single_eye is True
    assert pipeline_config.velocity.gap_fill_enabled is True
    assert pipeline_config.velocity.gap_fill_max_gap_ms == 42
    assert pipeline_config.velocity.coordinate_rounding == "halfup"
    assert pipeline_config.velocity.velocity_method == "ray3d_gaze_dir"
    assert pipeline_config.velocity.tobii_eye_offset_interpolation is True
    assert pipeline_config.classifier.velocity_threshold_deg_per_sec == 31
    assert pipeline_config.classifier.enable_invalid_window_neighbor_confirmation is True
    assert pipeline_config.classifier.enable_hysteresis is True
    assert pipeline_config.classifier.hysteresis_width_deg_per_sec == 1.5
    assert pipeline_config.classifier.enable_near_threshold_hybrid is True
    assert pipeline_config.classifier.near_threshold_band == 6
    assert pipeline_config.classifier.near_threshold_band_lower == 4
    assert pipeline_config.classifier.near_threshold_band_upper == 8
    assert pipeline_config.classifier.near_threshold_strategy == "replace"
    assert pipeline_config.classifier.near_threshold_confidence_margin == 0.4
    assert pipeline_config.classifier.near_threshold_require_same_side is True
    assert pipeline_config.classifier.near_threshold_max_delta == 3
    assert pipeline_config.classifier.near_threshold_neighbor_check is True
    assert pipeline_config.classifier.enable_confident_switch is True
    assert pipeline_config.classifier.confident_switch_margin_deg == 5
    assert pipeline_config.classifier.confident_switch_method == "ray3d"
    assert pipeline_config.classifier.enable_eye_jump_rule is True
    assert pipeline_config.classifier.eye_jump_threshold_mm == 12
    assert pipeline_config.classifier.eye_jump_velocity_threshold == 55
