# ivt_filter/cli.py
"""Command-line interface for IVT filter pipeline.

Entry point for the IVT processing pipeline. Delegates to specialized modules:
    - arg_parser: CLI argument definitions
    - config_builder: Configuration object construction
    - pipeline: Processing orchestration
"""
from __future__ import annotations

import argparse

from .config_builder import ConfigBuilder
from .pipeline import IVTPipeline


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
        choices=[
            "none", "median", "moving_average", 
            "median_strict", "moving_average_strict",
            "median_adaptive", "moving_average_adaptive"
        ],
        default="none",
        help="Raeumliches Smoothing auf kombinierten Gaze-Koordinaten. "
             "_strict Varianten ueberspringen Smoothing wenn invalide Samples im Fenster, "
             "_adaptive sammelt nur gueltige Samples und kann Suche erweitern (default: none).",
    )
    parser.add_argument(
        "--smooth-window-samples",
        type=int,
        default=5,
        help="Fensterbreite in Samples fuer Smoothing (default: 5).",
    )
    parser.add_argument(
        "--smoothing-min-samples",
        type=int,
        default=1,
        help="(Nur adaptive) Mindestanzahl gueltiger Samples fuer Smoothing (default: 1).",
    )
    parser.add_argument(
        "--smoothing-expansion-radius",
        type=int,
        default=0,
        help="(Nur adaptive) Samples ueber Standard-Fenster hinaus durchsuchen (default: 0).",
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
        "--fixed-window-edge-fallback",
        action="store_true",
        help=(
            "Bei fixed-window-samples: wenn Fensterrand ungültige Samples hat, "
            "verwende Geschwindigkeit vom nächsten Sample mit gültigem Fenster."
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
        "--asymmetric-neighbor-window",
        action="store_true",
        help=(
            "Nutze asymmetrisches 2-Sample Nachbar-Fenster. "
            "Priorität: Backward (i-1 → i), Fallback: Forward (i → i+1). "
            "Gap-Regel: 2 samples = radius 1."
        ),
    )

    # Shifted valid window (konstante Fensterlaenge, verschieben bei Invalids)
    parser.add_argument(
        "--shifted-valid-window",
        action="store_true",
        help=(
            "Halte feste Fensterlaenge (fixed_window_samples) und verschiebe das Fenster, "
            "bis ein zusammenhaengender Block gueltiger Samples gefunden wird."
        ),
    )
    parser.add_argument(
        "--shifted-valid-fallback",
        choices=["shrink", "unclassified"],
        default="shrink",
        help=(
            "Fallback falls kein gueltiges Fenster konstanter Laenge existiert: "
            "'shrink' = altes Shrink-Verhalten nutzen; 'unclassified' = kein Fenster."
        ),
    )
    parser.add_argument(
        "--use-fixed-dt",
        action="store_true",
        help=(
            "Nutze fixed dt aus Sampling-Rate (dt = 1/Hz) statt time_ms-Differenzen. "
            "Vermeidet Jitter durch Rundung. Nur mit --asymmetric-neighbor-window."
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
    parser.add_argument(
        "--average-fallback-single-eye",
        action="store_true",
        help=(
            "Wenn im Fenster oder mittleren Sample nur ein Auge valide ist, "
            "verwende durchgehend NUR dieses Auge (kein Average). "
            "Verhindert Parallaxe-Effekte bei Augen-Wechsel."
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
        choices=["olsen2d", "ray3d", "ray3d_gaze_dir"],
        default="olsen2d",
        help=(
            "Methode zur Berechnung des visuellen Winkels zwischen zwei Gaze-Punkten. "
            "'olsen2d': Olsen's 2D-Approximation (tan(θ)=s/d, nur eye_z nötig, schnell), "
            "'ray3d': physikalisch korrekte 3D-Winkel-Methode (acos(ray0·ray1), benötigt eye_x/y/z, präziser), "
            "'ray3d_gaze_dir': nutzt normalisierte Blickrichtungs-Vektoren (DACS norm), acos(dir0·dir1); benötigt keine Bildschirm- oder Eye-Position."
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

    # GT-based Saccaden-Glättung
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
    parser.add_argument(
        "--discard-fixation-target",
        choices=["Unclassified", "Saccade"],
        default="Unclassified",
        help="Ziel-Label fuer verworfene kurze Fixationen (default: Unclassified).",
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
    vel_cfg, cls_cfg, sac_cfg, fix_cfg = ConfigBuilder.build_all_configs(args)
    
    # Create pipeline
    pipeline = IVTPipeline(
        velocity_config=vel_cfg,
        classifier_config=cls_cfg,
        saccade_merge_config=sac_cfg if args.post_smoothing_ms else None,
        fixation_post_config=fix_cfg if (args.merge_close_fixations or args.discard_short_fixations) else None,
    )
    
    # Run pipeline
    pipeline.run(
        input_path=args.input,
        output_path=args.output,
        classify=args.classify,
        evaluate=args.evaluate,
        post_smoothing_ms=args.post_smoothing_ms,
        merge_close_fixations=args.merge_close_fixations,
        discard_short_fixations=args.discard_short_fixations,
        plot=not args.no_plot,
        with_events=args.with_events,
    )


if __name__ == "__main__":
    main()

