# ivt_filter/cli.py
from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from .config import (
    OlsenVelocityConfig,
    IVTClassifierConfig,
    SaccadeMergeConfig,
    FixationPostConfig,
)
from .io import read_tsv, write_tsv
from .velocity import compute_olsen_velocity
from .classification import apply_ivt_classifier, expand_gt_events_to_samples
from .postprocess import merge_short_saccade_blocks, apply_fixation_postprocessing
from .evaluation import evaluate_ivt_vs_ground_truth
from .plotting import (
    plot_velocity_only,
    plot_velocity_and_classification,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """
    CLI-Parser fuer die I-VT-Pipeline.

    Verantwortlichkeit:
      - Nur Parsing & Beschreibung der Optionen.
      - Keine Business-Logik (SOLID: SRP).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute Olsen-style angular velocity from extracted TSV (mm-based gaze) "
            "and apply an I-VT velocity-threshold classifier plus optional "
            "post-processing (gap-filling, smoothing, Tobii-like fixation filters)."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input TSV mit time_ms, gaze_left/right_x_mm, gaze_left/right_y_mm, "
            "validity_left/right, eye_left/right_z_mm. Pixel-Koordinaten optional."
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        help=(
            "Optionaler Output-TSV-Pfad. Falls gesetzt, wird das Ergebnis-DataFrame "
            "mit Velocity/IVT/Postprocessing-Spalten dort geschrieben."
        ),
    )

    # Velocity / Fenster-Konfiguration
    parser.add_argument(
        "--window",
        type=float,
        default=20.0,
        help="Zeitfenster-Laenge in ms fuer Olsen-Window (default: 20.0).",
    )
    parser.add_argument(
        "--eye",
        choices=["left", "right", "average"],
        default="average",
        help="Eye-Selection-Mode (default: average).",
    )
    parser.add_argument(
        "--smoothing",
        choices=["none", "median", "moving_average"],
        default="none",
        help="Raeumliches Smoothing auf kombinierten Gaze-Koordinaten (default: none).",
    )
    parser.add_argument(
        "--smooth-window-samples",
        type=int,
        default=5,
        help="Fensterbreite in Samples fuer Smoothing (default: 5).",
    )

    # Fenster-Strategien
    parser.add_argument(
        "--sample-symmetric-window",
        action="store_true",
        help="Nutze sample-symmetrisches Fenster innerhalb des Zeitfensters.",
    )
    parser.add_argument(
        "--fixed-window-samples",
        type=int,
        default=None,
        help=(
            "Feste Fensterbreite in Samples (ungerade >= 3). "
            "Falls gesetzt, wird eine reine Sample-Fenster-Strategie verwendet."
        ),
    )
    parser.add_argument(
        "--auto-fixed-window-from-ms",
        action="store_true",
        help=(
            "Leite feste Fensterbreite automatisch aus window_length_ms und "
            "Sampling-Rate ab."
        ),
    )
    parser.add_argument(
        "--symmetric-round-window",
        action="store_true",
        help=(
            "Symmetrische Rundungs-Logik: per_side = round(window_size / 2), "
            "effektive Größe = 2*per_side + 1. Erhöht Fenstergröße (z.B. 7 -> 9). "
            "Gap-Regel bleibt auf ursprünglicher Größe."
        ),
    )
    parser.add_argument(
        "--allow-asymmetric-window",
        action="store_true",
        help=(
            "Asymmetrische Fensterbreite erlauben: per_side = round(window_size / 2). "
            "Erlaubt auch gerade Fenstergrößen ohne Aufrunden auf ungerade."
        ),
    )
    parser.add_argument(
        "--sampling-rate-method",
        choices=["all_samples", "first_100"],
        default="first_100",
        help=(
            "Methode zur Bestimmung der Sampling-Rate. "
            "'all_samples' (Standard): Alle Samples verwenden. "
            "'first_100': (standarsyd) Nur die ersten 100 Samples verwenden (wie im Tobii-Paper)."
        ),
    )
    parser.add_argument(
        "--dt-calculation-method",
        choices=["median", "mean"],
        default="median",
        help=(
            "Methode zur Berechnung der Zeitdifferenzen. "
            "'median' (Standard): Robuster gegenüber Ausreißern. "
            "'mean': Arithmetisches Mittel (wie im Tobii-Paper erwähnt)."
        ),
    )
    parser.add_argument(
        "--no-fallback-valid-samples",
        action="store_true",
        help="Bei ungültigen first/last Samples: nächstes gültiges Sample verwenden (Standard: an).",
    )

    # Average-Auge Strategien
    parser.add_argument(
        "--average-window-single-eye",
        action="store_true",
        help=(
            "Bei Mixed mono/binokular innerhalb eines Fensters das Auge mit "
            "stabilerer Validitaet fuer Start/Ende verwenden."
        ),
    )
    parser.add_argument(
        "--average-window-impute-neighbor",
        action="store_true",
        help=(
            "Fehlende Augenkoordinate am Fensterrand anhand naechstem Nachbarn "
            "mit gueltigem Auge imputieren (nur eye_mode=average)."
        ),
    )

    # Gap-Filling
    parser.add_argument(
        "--gap-fill",
        action="store_true",
        help="Aktiviere zeitliches Gap-Filling (Interpolation pro Auge).",
    )
    parser.add_argument(
        "--gap-fill-max-ms",
        type=float,
        default=75.0,
        help="Maximale Luecken-Dauer in ms, die per Interpolation gefuellt wird (default: 75).",
    )
    parser.add_argument(
        "--coordinate-rounding",
        choices=["none", "nearest", "halfup", "floor", "ceil"],
        default="none",
        help=(
            "Rundet Gaze/Eye-Koordinaten vor der Velocity-Berechnung auf ganze Zahlen. "
            "'none': keine Rundung (default), "
            "'nearest': Banker's Rounding (bei 0.5 zur geraden Zahl), "
            "'halfup': bei 0.5 immer aufrunden, "
            "'floor': immer abrunden, "
            "'ceil': immer aufrunden."
        ),
    )
    parser.add_argument(
        "--velocity-method",
        choices=["olsen2d", "ray3d"],
        default="olsen2d",
        help=(
            "Methode zur Berechnung des visuellen Winkels zwischen zwei Gaze-Punkten. "
            "'olsen2d': Olsen's 2D-Approximation (tan(θ)=s/d, nur eye_z nötig, schnell), "
            "'ray3d': physikalisch korrekte 3D-Winkel-Methode (acos(ray0·ray1), benötigt eye_x/y/z, präziser)."
        ),
    )

    # Klassifikation
    parser.add_argument(
        "--classify",
        action="store_true",
        help=(
            "Wende einen I-VT Velocity-Threshold-Klassifikator an und erzeuge "
            "ivt_sample_type / ivt_event_type / ivt_event_index."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Velocity-Threshold in deg/s fuer I-VT (default: 30).",
    )

    # GT-basierte Saccaden-Glättung
    parser.add_argument(
        "--post-smoothing-ms",
        type=float,
        default=0.0,
        help=(
            "Wenn > 0, merge kurze Saccaden-Bloecke innerhalb von GT-Fixationen "
            "(Dauer < Wert in ms)."
        ),
    )
    parser.add_argument(
        "--post-smoothing-no-context",
        action="store_true",
        help="Bei Saccaden-Merge GT-Nachbar-Kontext ignorieren (keine Fixations-Nachbarn erforderlich).",
    )
    parser.add_argument(
        "--post-smoothing-no-sample-col",
        action="store_true",
        help="Nicht auf 'ivt_sample_type' operieren, sondern direkt auf 'ivt_event_type'.",
    )

    # Fixations-Postprocessing (Tobii-like)
    parser.add_argument(
        "--merge-close-fixations",
        action="store_true",
        help="Benachbarte Fixationen mergen, wenn zeitlich/raeumlich nah (Tobii-aehnlich).",
    )
    parser.add_argument(
        "--merge-fix-max-gap-ms",
        type=float,
        default=75.0,
        help="Maximale Zeitluecke in ms zwischen Fixationen fuer Merge (default: 75).",
    )
    parser.add_argument(
        "--merge-fix-max-angle-deg",
        type=float,
        default=0.5,
        help="Maximaler visueller Winkel in Grad zwischen Fixationszentren fuer Merge (default: 0.5).",
    )
    parser.add_argument(
        "--discard-short-fixations",
        action="store_true",
        help="Fixationen verwerfen, deren Dauer kleiner als min-fixation-duration-ms ist.",
    )
    parser.add_argument(
        "--min-fixation-duration-ms",
        type=float,
        default=60.0,
        help="Minimale Fixationsdauer in ms (default: 60).",
    )

    # Evaluation
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Vergleiche Klassifikation gegen Ground Truth und gib Metriken aus.",
    )

    # Plotting
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Keine Matplotlib-Plots anzeigen.",
    )
    parser.add_argument(
        "--with-events",
        action="store_true",
        help="Velocity + GT-Event-Plot anzeigen (sonst nur Velocity).",
    )

    return parser


def build_velocity_config(args: argparse.Namespace) -> OlsenVelocityConfig:
    """
    Velocity-Konfiguration aus CLI-Argumenten ableiten.

    (GRASP: Information Expert - diese Funktion kennt die Mapping-Details)
    """
    return OlsenVelocityConfig(
        window_length_ms=args.window,
        eye_mode=args.eye,
        smoothing_mode=args.smoothing,
        smoothing_window_samples=args.smooth_window_samples,
        sample_symmetric_window=args.sample_symmetric_window,
        fixed_window_samples=args.fixed_window_samples,
        auto_fixed_window_from_ms=args.auto_fixed_window_from_ms,
        symmetric_round_window=args.symmetric_round_window,
        allow_asymmetric_window=args.allow_asymmetric_window,
        sampling_rate_method=args.sampling_rate_method,
        dt_calculation_method=args.dt_calculation_method,
        use_fallback_valid_samples=not args.no_fallback_valid_samples,
        average_window_single_eye=args.average_window_single_eye,
        average_window_impute_neighbor=args.average_window_impute_neighbor,
        gap_fill_enabled=args.gap_fill,
        gap_fill_max_gap_ms=args.gap_fill_max_ms,
        coordinate_rounding=args.coordinate_rounding,
        velocity_method=args.velocity_method,
    )


def main() -> None:
    """
    Orchestriert die komplette Pipeline:

      1) TSV laden
      2) Velocity berechnen
      3) optional: klassifizieren
      4) optional: Saccaden-Postprocessing mit GT
      5) optional: Fixations-Postprocessing ohne GT (Tobii-like)
      6) optional: Evaluation
      7) optional: Plotting
      8) optional: TSV schreiben

    Die Logik pro Step liegt in eigenen Modulen/Funktionen (SOLID).
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1) Daten laden
    df = read_tsv(args.input)

    # 2) Velocity berechnen
    vel_cfg = build_velocity_config(args)
    df = compute_olsen_velocity(df, vel_cfg)

    pred_sample_col: Optional[str] = None
    pred_col_for_eval: Optional[str] = None

    # 3) I-VT Klassifikation (Sample + Events)
    if args.classify or args.evaluate or args.merge_close_fixations or args.discard_short_fixations:
        cls_cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=args.threshold)
        df = apply_ivt_classifier(df, cls_cfg)
        
        # Expandiere GT Events zu GT Samples (für Sample-Level Evaluation)
        df = expand_gt_events_to_samples(df)
        
        pred_sample_col = "ivt_sample_type"
        pred_col_for_eval = pred_sample_col

    # 4) GT-gestuetzte Saccaden-Glättung (optional)
    if (
        args.post_smoothing_ms
        and args.post_smoothing_ms > 0
        and pred_sample_col is not None
    ):
        sm_cfg = SaccadeMergeConfig(
            max_saccade_block_duration_ms=args.post_smoothing_ms,
            require_fixation_context=not args.post_smoothing_no_context,
            use_sample_type_column=None if args.post_smoothing_no_sample_col else "ivt_sample_type",
        )
        df, merge_stats = merge_short_saccade_blocks(df, cfg=sm_cfg)

        # welche Spalte als Prediction benutzen?
        if sm_cfg.use_sample_type_column is not None:
            # sample-basiert -> <sample_col>_smoothed
            pred_sample_col = sm_cfg.use_sample_type_column + "_smoothed"
        else:
            # event-basiert -> 'ivt_event_type_smoothed'
            pred_sample_col = "ivt_event_type_smoothed"

        pred_col_for_eval = pred_sample_col

        print(
            "[Post-Processing] merged short saccade blocks: "
            f"{merge_stats['n_blocks_merged']} / {merge_stats['n_blocks_total']} blocks, "
            f"{merge_stats['n_samples_merged']} samples."
        )

    # 5) Tobii-aehnliches Fixations-Postprocessing (optional, ohne GT)
    if (
        pred_sample_col is not None
        and (args.merge_close_fixations or args.discard_short_fixations)
    ):
        fix_cfg = FixationPostConfig(
            merge_adjacent_fixations=args.merge_close_fixations,
            max_time_gap_ms=args.merge_fix_max_gap_ms,
            max_angle_deg=args.merge_fix_max_angle_deg,
            discard_short_fixations=args.discard_short_fixations,
            min_fixation_duration_ms=args.min_fixation_duration_ms,
        )
        df, fix_stats = apply_fixation_postprocessing(
            df,
            cfg=fix_cfg,
            sample_col=pred_sample_col,
            time_col="time_ms",
            x_col="smoothed_x_mm",
            y_col="smoothed_y_mm",
            eye_z_col="eye_z_mm",
            event_type_col="ivt_event_type_post",
            event_index_col="ivt_event_index_post",
        )

        # Pred-Spalte fuer Evaluation bleibt pred_sample_col,
        # das wurde in-place modifiziert.
        pred_col_for_eval = pred_sample_col

        print(
            "[FixationPost] merged_pairs="
            f"{fix_stats.get('merged_pairs', 0)}, "
            "gap_samples_to_fixation="
            f"{fix_stats.get('gap_samples_to_fixation', 0)}, "
            "discarded_fixations="
            f"{fix_stats.get('discarded_fixations', 0)}, "
            "discarded_samples="
            f"{fix_stats.get('discarded_samples', 0)}"
        )

    # 6) TSV schreiben (falls gewuenscht)
    if args.output is not None:
        write_tsv(df, args.output)

    # 7) Evaluation (falls gewuenscht)
    if args.evaluate and pred_col_for_eval is not None:
        evaluate_ivt_vs_ground_truth(df, pred_col=pred_col_for_eval)

    # 8) Plotting (nur, wenn nicht --no-plot)
    if not args.no_plot:
        if args.with_events:
            # Velocity + GT-Ereignisse
            plot_velocity_and_classification(df, vel_cfg)
        else:
            # Nur Velocity
            plot_velocity_only(df, vel_cfg)


if __name__ == "__main__":
    main()
