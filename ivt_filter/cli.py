# ivt_filter/cli.py
"""Command-line interface for IVT filter pipeline.

Entry point for the IVT processing pipeline. Delegates to specialized modules:
    - arg_parser: CLI argument definitions
    - config_builder: Configuration object construction
    - pipeline: Processing orchestration
"""
from __future__ import annotations

import argparse

from .config.config_builder import ConfigBuilder
from .io.pipeline import IVTPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Responsibility: Only parsing and describing options (SRP).
    No business logic.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute Olsen-style angular velocity from extracted TSV (mm-based gaze) "
            "and apply an I-VT velocity-threshold classifier plus optional "
            "post-processing (gap-filling, smoothing, Tobii-like fixation filters)."
        ),
    )
    _add_io_arguments(parser)
    _add_velocity_arguments(parser)
    _add_window_arguments(parser)
    _add_smoothing_arguments(parser)
    _add_classifier_arguments(parser)
    _add_refinement_arguments(parser)
    _add_postprocessing_arguments(parser)
    _add_evaluation_arguments(parser)
    _add_plotting_arguments(parser)
    return parser


def _add_io_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input TSV with time_ms, gaze_left/right_x_mm, gaze_left/right_y_mm, "
            "validity_left/right, eye_left/right_z_mm. Pixel coordinates optional."
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        help=(
            "Optional output TSV path. If set, the result DataFrame "
            "with velocity/IVT/postprocessing columns is written there."
        ),
    )

    # Timestamp options
    parser.add_argument(
        "--time-column",
        choices=["time_ms", "time_us"],
        default="time_ms",
        help=(
            "Column with timestamps. time_us allows microsecond precision."
        ),
    )
    parser.add_argument(
        "--time-unit",
        choices=["ms", "us", "ns"],
        default="ms",
        help="Unit of the selected time column (default: ms).",
    )


def _add_velocity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eye",
        choices=["left", "right", "average", "strict_average"],
        default="average",
        help="Eye selection mode (default: average).",
    )
    parser.add_argument(
        "--sampling-rate-method",
        choices=["all_samples", "first_100"],
        default="first_100",
        help=(
            "Method for determining the sampling rate. "
            "'all_samples' (default): use all samples. "
            "'first_100': (default) use only the first 100 samples (as in the Tobii paper)."
        ),
    )
    parser.add_argument(
        "--dt-calculation-method",
        choices=["median", "mean"],
        default="median",
        help=(
            "Method for computing time differences. "
            "'median' (default): more robust against outliers. "
            "'mean': arithmetic mean (as mentioned in the Tobii paper)."
        ),
    )
    parser.add_argument(
        "--no-fallback-valid-samples",
        action="store_true",
        help="On invalid first/last samples: use the nearest valid sample (default: on).",
    )
    # Average-eye strategies
    parser.add_argument(
        "--average-window-single-eye",
        action="store_true",
        help=(
            "On mixed mono/binocular within a window, use the eye with "
            "more stable validity for start/end."
        ),
    )
    parser.add_argument(
        "--average-window-impute-neighbor",
        action="store_true",
        help=(
            "Impute a missing eye coordinate at the window edge from the nearest "
            "neighbor with a valid eye (only eye_mode=average)."
        ),
    )
    parser.add_argument(
        "--average-fallback-single-eye",
        action="store_true",
        help=(
            "If only one eye is valid in the window or at the center sample, "
            "consistently use ONLY that eye (no average). "
            "Prevents parallax effects when switching eyes."
        ),
    )
    parser.add_argument(
        "--coordinate-rounding",
        choices=["none", "nearest", "halfup", "floor", "ceil"],
        default="none",
        help=(
            "Rounds gaze/eye coordinates to integers before the velocity computation. "
            "'none': no rounding (default), "
            "'nearest': banker's rounding (round half to even), "
            "'halfup': always round up at 0.5, "
            "'floor': always round down, "
            "'ceil': always round up."
        ),
    )
    parser.add_argument(
        "--tobii-eye-offset-interpolation",
        action="store_true",
        help=(
            "Reconstructs a missing eye via the last known L→R offset "
            "(Tobii reference logic). Prevents phantom velocities at gap edges "
            "when one eye drops out briefly."
        ),
    )
    parser.add_argument(
        "--velocity-method",
        choices=["olsen2d", "ray3d", "ray3d_gaze_dir", "tobii_gaze_dir"],
        default="olsen2d",
        help=(
            "Method for computing the visual angle between two gaze points. "
            "'olsen2d': Olsen's 2D approximation (tan(θ)=s/d, only eye_z needed, fast), "
            "'ray3d': physically correct 3D angle method (acos(ray0·ray1), needs eye_x/y/z, more precise), "
            "'ray3d_gaze_dir': uses normalized gaze-direction vectors (DACS norm), acos(dir0·dir1); needs no screen or eye position."
        ),
    )


def _add_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window",
        type=float,
        default=20.0,
        help="Time-window length in ms for the Olsen window (default: 20.0).",
    )
    # Window strategies
    parser.add_argument(
        "--time-symmetric-window",
        action="store_true",
        help=(
            "Use the classic Olsen time window. Without an explicit window strategy, "
            "Tobii-compatible windowing is used by default."
        ),
    )
    parser.add_argument(
        "--sample-symmetric-window",
        action="store_true",
        help="Use a sample-symmetric window within the time window.",
    )
    parser.add_argument(
        "--fixed-window-samples",
        type=int,
        default=None,
        help=(
            "Fixed window width in samples (odd >= 3). "
            "If set, a pure sample-window strategy is used."
        ),
    )
    parser.add_argument(
        "--auto-fixed-window-from-ms",
        action="store_true",
        help=(
            "Derive the fixed window width automatically from window_length_ms and "
            "the sampling rate."
        ),
    )
    parser.add_argument(
        "--fixed-window-edge-fallback",
        action="store_true",
        help=(
            "With fixed-window-samples: if the window edge has invalid samples, "
            "use the velocity from the nearest sample with a valid window."
        ),
    )
    parser.add_argument(
        "--symmetric-round-window",
        action="store_true",
        help=(
            "Symmetric rounding logic: per_side = round(window_size / 2), "
            "effective size = 2*per_side + 1. Increases window size (e.g. 7 -> 9). "
            "The gap rule stays at the original size."
        ),
    )
    parser.add_argument(
        "--allow-asymmetric-window",
        action="store_true",
        help=(
            "Allow asymmetric window width: per_side = round(window_size / 2). "
            "Also allows even window sizes without rounding up to odd."
        ),
    )
    parser.add_argument(
        "--asymmetric-neighbor-window",
        action="store_true",
        help=(
            "Use an asymmetric 2-sample neighbor window. "
            "Priority: backward (i-1 → i), fallback: forward (i → i+1). "
            "Gap rule: 2 samples = radius 1."
        ),
    )

    # Shifted valid window (constant window length, shift on invalids)
    parser.add_argument(
        "--shifted-valid-window",
        action="store_true",
        help=(
            "Keep a fixed window length (fixed_window_samples) and shift the window "
            "until a contiguous block of valid samples is found."
        ),
    )
    parser.add_argument(
        "--shifted-valid-fallback",
        choices=["shrink", "unclassified"],
        default="shrink",
        help=(
            "Fallback if no valid window of constant length exists: "
            "'shrink' = use the old shrink behavior; 'unclassified' = no window."
        ),
    )
    parser.add_argument(
        "--use-fixed-dt",
        action="store_true",
        help=(
            "Use a fixed dt from the sampling rate (dt = 1/Hz) instead of time_ms differences. "
            "Avoids jitter from rounding. Only with --asymmetric-neighbor-window."
        ),
    )


def _add_smoothing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--smoothing",
        choices=[
            "none", "median", "moving_average", 
            "median_strict", "moving_average_strict",
            "median_adaptive", "moving_average_adaptive"
        ],
        default="none",
        help="Spatial smoothing on combined gaze coordinates. "
             "_strict variants skip smoothing when invalid samples are in the window, "
             "_adaptive collects only valid samples and can widen the search (default: none).",
    )
    parser.add_argument(
        "--smooth-window-samples",
        type=int,
        default=5,
        help="Window width in samples for smoothing (default: 5).",
    )
    parser.add_argument(
        "--smoothing-min-samples",
        type=int,
        default=1,
        help="(Adaptive only) Minimum number of valid samples for smoothing (default: 1).",
    )
    parser.add_argument(
        "--smoothing-expansion-radius",
        type=int,
        default=0,
        help="(Adaptive only) Search samples beyond the standard window (default: 0).",
    )


def _add_classifier_arguments(parser: argparse.ArgumentParser) -> None:
    # Classification
    parser.add_argument(
        "--classify",
        action="store_true",
        help=(
            "Apply an I-VT velocity-threshold classifier and produce "
            "ivt_sample_type / ivt_event_type / ivt_event_index."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Velocity threshold in deg/s for I-VT (default: 30).",
    )

    # Optional classifier reconstruction heuristics
    parser.add_argument(
        "--enable-invalid-window-neighbor-confirmation",
        action="store_true",
        help=(
            "Require an adjacent above-threshold velocity before classifying an "
            "invalid-window sample as a saccade. Disabled for strict I-VT baseline runs."
        ),
    )
    parser.add_argument(
        "--enable-hysteresis",
        action="store_true",
        help=(
            "Retain the previous motion label in a band immediately below the threshold. "
            "Disabled for strict I-VT baseline runs."
        ),
    )
    parser.add_argument(
        "--hysteresis-width",
        type=float,
        default=2.0,
        help="Width of the optional hysteresis band in deg/s (default: 2.0).",
    )


def _add_refinement_arguments(parser: argparse.ArgumentParser) -> None:
    # Near-threshold hybrid strategy
    parser.add_argument(
        "--enable-near-threshold-hybrid",
        action="store_true",
        help=(
            "Enable hybrid near-threshold classification using alternative velocity. "
            "Applies when |v_base - threshold| <= band."
        ),
    )
    parser.add_argument(
        "--near-threshold-band",
        type=float,
        default=5.0,
        help="Band around threshold (deg/s) for hybrid classification (default: 5.0).",
    )
    parser.add_argument(
        "--near-threshold-band-lower",
        type=float,
        default=None,
        help="Asymmetric lower band (below threshold). If not set, uses symmetric band.",
    )
    parser.add_argument(
        "--near-threshold-band-upper",
        type=float,
        default=None,
        help="Asymmetric upper band (above threshold). If not set, uses symmetric band.",
    )
    parser.add_argument(
        "--near-threshold-strategy",
        type=str,
        choices=['replace', 'inverse'],
        default='inverse',
        help=(
            "Hybrid strategy: 'replace' always uses alternative velocity, "
            "'inverse' uses velocity farther from threshold (default: inverse)."
        ),
    )
    parser.add_argument(
        "--near-threshold-confidence-margin",
        type=float,
        default=0.3,
        help=(
            "Only switch to alternative velocity in inverse strategy if it is at least "
            "this many deg/s farther from the threshold (default: 0.3)."
        ),
    )
    parser.add_argument(
        "--near-threshold-require-same-side",
        action="store_true",
        help=(
            "When set, only switch if base and alternative velocity are on the same side of the threshold."
        ),
    )
    parser.add_argument(
        "--near-threshold-max-delta",
        type=float,
        default=2.0,
        help=(
            "Maximum allowed |alt-base| (deg/s) to switch in inverse strategy (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--near-threshold-neighbor-check",
        action="store_true",
        help=(
            "Require neighbor majority support (previous/next base velocity side) when alt crosses the threshold."
        ),
    )

    # Confident mismatch switch (far from threshold)
    parser.add_argument(
        "--confident-switch-enabled",
        action="store_true",
        help=(
            "Enable confident mismatch switch: when base velocity is far from threshold (|v-th| >= margin) "
            "and alternative method disagrees, use the alternative label."
        ),
    )
    parser.add_argument(
        "--confident-switch-margin-deg",
        type=float,
        default=4.0,
        help="Margin in deg/s away from threshold to consider a mismatch confident (default: 4.0).",
    )
    parser.add_argument(
        "--confident-switch-method",
        choices=["olsen2d", "ray3d", "ray3d_gaze_dir"],
        default="ray3d_gaze_dir",
        help="Alternative velocity method to consult for confident switches (default: ray3d_gaze_dir).",
    )

    # Eye-position jump rule
    parser.add_argument(
        "--enable-eye-jump-rule",
        action="store_true",
        help=(
            "Enable eye-position jump correction. Uses alternative velocity when "
            "eye position shifts significantly within the window."
        ),
    )
    parser.add_argument(
        "--eye-jump-threshold",
        type=float,
        default=10.0,
        help="Eye position displacement threshold (mm) to trigger jump rule (default: 10.0).",
    )
    parser.add_argument(
        "--eye-jump-velocity-threshold",
        type=float,
        default=50.0,
        help="Velocity threshold (deg/s) for 'clear saccade' in jump rule (default: 50.0).",
    )


def _add_postprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    # Gap filling
    parser.add_argument(
        "--gap-fill",
        action="store_true",
        help="Enable temporal gap filling (interpolation per eye).",
    )
    parser.add_argument(
        "--gap-fill-max-ms",
        type=float,
        default=75.0,
        help="Maximum gap duration in ms that is filled by interpolation (default: 75).",
    )
    # GT-based saccade smoothing
    parser.add_argument(
        "--post-smoothing-ms",
        type=float,
        default=0.0,
        help=(
            "If > 0, merge short saccade blocks within GT fixations "
            "(duration < value in ms)."
        ),
    )
    parser.add_argument(
        "--post-smoothing-no-context",
        action="store_true",
        help="Ignore GT neighbor context on saccade merge (no fixation neighbors required).",
    )
    parser.add_argument(
        "--post-smoothing-no-sample-col",
        action="store_true",
        help="Do not operate on 'ivt_sample_type', but directly on 'ivt_event_type'.",
    )

    # Fixation postprocessing (Tobii-like)
    parser.add_argument(
        "--merge-close-fixations",
        action="store_true",
        help="Merge adjacent fixations when temporally/spatially close (Tobii-like).",
    )
    parser.add_argument(
        "--merge-fix-max-gap-ms",
        type=float,
        default=75.0,
        help="Maximum time gap in ms between fixations for merging (default: 75).",
    )
    parser.add_argument(
        "--merge-fix-max-angle-deg",
        type=float,
        default=0.5,
        help="Maximum visual angle in degrees between fixation centers for merging (default: 0.5).",
    )
    parser.add_argument(
        "--merge-fix-max-gap-velocity-deg-per-sec",
        type=float,
        default=35.0,
        help="Maximum velocity in deg/s for relabeling gap samples (default: 35).",
    )
    parser.add_argument(
        "--discard-short-fixations",
        action="store_true",
        help="Discard fixations whose duration is smaller than min-fixation-duration-ms.",
    )
    parser.add_argument(
        "--min-fixation-duration-ms",
        type=float,
        default=60.0,
        help="Minimum fixation duration in ms (default: 60).",
    )
    parser.add_argument(
        "--discard-fixation-target",
        choices=["Unclassified", "Saccade"],
        default="Unclassified",
        help="Target label for discarded short fixations (default: Unclassified).",
    )


def _add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Compare classification against ground truth and print metrics.",
    )
    parser.add_argument(
        "--exclude-calibration",
        action="store_true",
        help="Exclude samples whose presented stimulus name is 'Eyetracker Calibration' during evaluation.",
    )


def _add_plotting_arguments(parser: argparse.ArgumentParser) -> None:
    # Plotting
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display matplotlib plots.",
    )
    parser.add_argument(
        "--with-events",
        action="store_true",
        help="Display velocity + GT-event plot (otherwise velocity only).",
    )


def main() -> None:
    """Main entry point for CLI.
    
    Orchestrates:
        1. Parse arguments
        2. Build configurations
        3. Create and run pipeline
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Build all configurations
    pipeline_config = ConfigBuilder.build_pipeline_config(args)
    
    # Create pipeline
    pipeline = IVTPipeline(pipeline_config)
    
    # Run pipeline
    pipeline.run(
        input_path=args.input,
        output_path=args.output,
        evaluate=args.evaluate,
        plot=not args.no_plot,
        with_events=args.with_events,
        evaluate_exclude_calibration=args.exclude_calibration,
    )


if __name__ == "__main__":
    main()
