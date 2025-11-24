"""Command line interface for IVT processing."""
from __future__ import annotations

import argparse

from .analyzer import IVTAnalyzer, PlotConfig
from .classifier import IVTClassifier
from .config import IVTClassifierConfig, OlsenVelocityConfig
from .extractor import convert_tobii_tsv_to_ivt_tsv
from .metrics import evaluate_ivt_vs_ground_truth
from .velocity import VelocityCalculator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IVT processing pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Convert Tobii export to slim IVT TSV")
    extract.add_argument("input", help="Path to raw Tobii TSV export")
    extract.add_argument("output", help="Path to write slim TSV")

    velocity = sub.add_parser("velocity", help="Compute Olsen-style velocity")
    velocity.add_argument("input", help="Slim TSV with gaze columns")
    velocity.add_argument("output", help="Path to write TSV with velocity column")
    velocity.add_argument("--window", type=float, default=20.0, help="Window length in ms")
    velocity.add_argument(
        "--eye",
        choices=["left", "right", "average"],
        default="average",
        help="Eye selection strategy",
    )

    classify = sub.add_parser("classify", help="Apply IVT classifier to a velocity TSV")
    classify.add_argument("input", help="TSV containing velocity_deg_per_sec")
    classify.add_argument("output", help="Path to write TSV with classifier columns")
    classify.add_argument("--threshold", type=float, default=30.0, help="Velocity threshold deg/s")

    analyze = sub.add_parser("analyze", help="Plot velocity and gaze position; optionally show events")
    analyze.add_argument("input", help="TSV containing velocity and gaze position columns")
    analyze.add_argument("output", help="Path to write the generated plot (png or pdf)")
    analyze.add_argument("--threshold", type=float, default=30.0, help="Velocity threshold deg/s")
    analyze.add_argument(
        "--show-events",
        action="store_true",
        help="Include event index step plot when ivt_event_index is present",
    )
    analyze.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(10.0, 6.0),
        help="Figure size in inches (width height)",
    )
    analyze.add_argument("--dpi", type=float, default=None, help="Optional DPI override for the figure")
    analyze.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the file (uses your default backend)",
    )

    evaluate = sub.add_parser("evaluate", help="Evaluate classifier output against ground truth")
    evaluate.add_argument("input", help="TSV containing classifier output and GT")
    evaluate.add_argument("--gt-col", help="Ground truth column name")
    evaluate.add_argument(
        "--pred-col",
        default="ivt_sample_type",
        help="Prediction column name (default: ivt_sample_type)",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "extract":
        convert_tobii_tsv_to_ivt_tsv(args.input, args.output)
        return

    if args.command == "velocity":
        cfg = OlsenVelocityConfig(window_length_ms=args.window, eye_mode=args.eye)
        calculator = VelocityCalculator(cfg)
        calculator.compute_from_file(args.input, args.output)
        return

    if args.command == "classify":
        cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=args.threshold)
        classifier = IVTClassifier(cfg)
        classifier.classify_from_file(args.input, args.output)
        return

    if args.command == "analyze":
        cfg = PlotConfig(
            threshold_deg_per_sec=args.threshold,
            show_event_index=args.show_events,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            show=args.show,
        )
        IVTAnalyzer(cfg).plot_from_file(args.input, args.output)
        return

    if args.command == "evaluate":
        import pandas as pd

        df = pd.read_csv(args.input, sep="\t")
        evaluate_ivt_vs_ground_truth(df, gt_col=args.gt_col, pred_col=args.pred_col)
        return


if __name__ == "__main__":
    main()
