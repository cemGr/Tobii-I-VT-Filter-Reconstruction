#!/usr/bin/env python3
"""
Parameter sweep for different window lengths (sample windows).

Tests different velocity calculation windows to find optimal configuration.

Usage:
    python example_window_sweep.py
"""
from pathlib import Path

from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.observers import ConsoleReporter, MetricsLogger, ExperimentTracker
from ivt_filter.pipeline import IVTPipeline


def window_length_sweep():
    """
    Test verschiedene Window-Größen (Sample-Fenster).
    
    Typische Werte:
    - 1 ms: Sehr kleines Fenster (kaum Glättung)
    - 10 ms: Kleines Fenster
    - 20 ms: Standard (Olsen Paper)
    - 40 ms: Größeres Fenster
    - 60 ms: Sehr groß
    """
    print("\n" + "="*80)
    print("WINDOW LENGTH SWEEP - Testing Different Sample Windows")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    # Test different window sizes (in Millisekunden)
    window_sizes = [1.0, 10.0, 20.0, 40.0, 60.0]
    
    # Fixed threshold for fair comparison
    fixed_threshold = 30.0
    
    for window_ms in window_sizes:
        print(f"\n{'─'*80}")
        print(f"Testing window: {window_ms} ms")
        print(f"{'─'*80}")
        
        # Velocity Configuration mit verschiedenen Fenstergrößen
        velocity_config = OlsenVelocityConfig(
            window_length_ms=window_ms,
            velocity_method="olsen2d",
            eye_mode="average",
            smoothing_mode="none",  # No smoothing für fairen Vergleich
        )
        
        classifier_config = IVTClassifierConfig(
            velocity_threshold_deg_per_sec=fixed_threshold,
        )
        
        # Experiment Config
        exp_config = ExperimentConfig(
            name=f"window_sweep_{int(window_ms)}ms",
            description=f"Window length {window_ms} ms, threshold {fixed_threshold} deg/s",
            velocity_config=velocity_config,
            classifier_config=classifier_config,
            tags=["window_sweep", "olsen2d", f"window_{int(window_ms)}ms"],
            metadata={
                "window_ms": window_ms,
                "method": "olsen2d",
                "smoothing": "none",
            }
        )
        
        # Pipeline mit Observers
        pipeline = IVTPipeline(velocity_config, classifier_config)
        pipeline.register_observer(ConsoleReporter(verbose=False))
        pipeline.register_observer(MetricsLogger("experiments/window_sweep_metrics.csv"))
        pipeline.register_observer(ExperimentTracker("experiments"))
        
        try:
            df = pipeline.run_with_tracking(
                input_path=input_file,
                config=exp_config,
                evaluate=True,
            )
            print(f"✅ Window {window_ms} ms completed ({len(df)} samples)")
        except Exception as e:
            print(f"❌ Window {window_ms} ms failed: {e}")


def window_with_smoothing_sweep():
    """
    Test verschiedene Window-Größen MIT Smoothing.
    
    Kombiniert Window-Größe mit verschiedenen Smoothing-Methoden.
    """
    print("\n" + "="*80)
    print("WINDOW + SMOOTHING SWEEP")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    # Window-Größen
    window_sizes = [10.0, 20.0, 40.0]
    
    # Smoothing-Methoden
    smoothing_methods = ["none", "median", "moving_average"]
    
    fixed_threshold = 30.0
    
    for window_ms in window_sizes:
        for smoothing in smoothing_methods:
            print(f"\n{'─'*40}")
            print(f"Testing: {window_ms} ms + {smoothing}")
            print(f"{'─'*40}")
            
            velocity_config = OlsenVelocityConfig(
                window_length_ms=window_ms,
                velocity_method="olsen2d",
                eye_mode="average",
                smoothing_mode=smoothing if smoothing != "none" else None,
                smoothing_window_samples=3 if smoothing != "none" else 1,
            )
            
            classifier_config = IVTClassifierConfig(
                velocity_threshold_deg_per_sec=fixed_threshold,
            )
            
            exp_config = ExperimentConfig(
                name=f"window_{int(window_ms)}ms_smooth_{smoothing}",
                description=f"Window {window_ms} ms with {smoothing} smoothing",
                velocity_config=velocity_config,
                classifier_config=classifier_config,
                tags=["window_smoothing_sweep", smoothing, f"window_{int(window_ms)}ms"],
                metadata={
                    "window_ms": window_ms,
                    "smoothing": smoothing,
                }
            )
            
            pipeline = IVTPipeline(velocity_config, classifier_config)
            pipeline.register_observer(ConsoleReporter(verbose=False))
            pipeline.register_observer(MetricsLogger("experiments/window_smoothing_metrics.csv"))
            pipeline.register_observer(ExperimentTracker("experiments"))
            
            try:
                df = pipeline.run_with_tracking(
                    input_path=input_file,
                    config=exp_config,
                    evaluate=True,
                )
                print(f"✅ Completed")
            except Exception as e:
                print(f"❌ Failed: {e}")


def window_method_comparison():
    """
    Vergleicht Olsen2D vs. Ray3D bei verschiedenen Window-Größen.
    """
    print("\n" + "="*80)
    print("WINDOW SIZE: Olsen2D vs Ray3D Comparison")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    window_sizes = [10.0, 20.0, 40.0]
    methods = ["olsen2d", "ray3d"]
    fixed_threshold = 30.0
    
    for window_ms in window_sizes:
        for method in methods:
            print(f"\n{'─'*40}")
            print(f"Testing: {method} @ {window_ms} ms")
            print(f"{'─'*40}")
            
            velocity_config = OlsenVelocityConfig(
                window_length_ms=window_ms,
                velocity_method=method,
                eye_mode="average",
            )
            
            classifier_config = IVTClassifierConfig(
                velocity_threshold_deg_per_sec=fixed_threshold,
            )
            
            exp_config = ExperimentConfig(
                name=f"{method}_window_{int(window_ms)}ms",
                description=f"{method} method with {window_ms} ms window",
                velocity_config=velocity_config,
                classifier_config=classifier_config,
                tags=["method_comparison", method, f"window_{int(window_ms)}ms"],
                metadata={
                    "window_ms": window_ms,
                    "method": method,
                }
            )
            
            pipeline = IVTPipeline(velocity_config, classifier_config)
            pipeline.register_observer(ConsoleReporter(verbose=False))
            pipeline.register_observer(MetricsLogger("experiments/method_window_metrics.csv"))
            pipeline.register_observer(ExperimentTracker("experiments"))
            
            try:
                df = pipeline.run_with_tracking(
                    input_path=input_file,
                    config=exp_config,
                    evaluate=True,
                )
                print(f"✅ Completed")
            except Exception as e:
                print(f"❌ Failed: {e}")


def analyze_window_results():
    """
    Analysiert die Ergebnisse des Window Sweeps.
    """
    print("\n" + "="*80)
    print("ANALYZING WINDOW SWEEP RESULTS")
    print("="*80)
    
    manager = ExperimentManager("experiments")
    
    # Window Sweep Experimente finden
    window_experiments = manager.list_experiments(tags=["window_sweep"])
    
    if not window_experiments:
        print("⚠️  No window sweep experiments found. Run window_length_sweep() first!")
        return
    
    print(f"\nFound {len(window_experiments)} window sweep experiments")
    
    # Experimente vergleichen
    exp_names = [e["name"] for e in window_experiments]
    comparison = manager.compare_experiments(exp_names)
    
    # Sort by Window-Größe
    comparison = comparison.sort_values("window_ms")
    
    print("\n" + "─"*80)
    print("WINDOW SIZE COMPARISON")
    print("─"*80)
    print(comparison[[
        "experiment", 
        "window_ms", 
        "percentage_agreement", 
        "fixation_recall",
        "saccade_recall",
        "cohen_kappa"
    ]].to_string(index=False))
    
    # Beste Konfiguration finden
    try:
        best_name, best_value, best_config = manager.get_best_configuration(
            metric="percentage_agreement",
            tags=["window_sweep"]
        )
        
        print("\n" + "="*80)
        print("🏆 BEST WINDOW SIZE CONFIGURATION")
        print("="*80)
        print(f"Experiment: {best_name}")
        print(f"Agreement: {best_value:.2f}%")
        print(f"Window: {best_config.velocity_config.window_length_ms} ms")
        print(f"Method: {best_config.velocity_config.velocity_method}")
        print(f"Threshold: {best_config.classifier_config.velocity_threshold_deg_per_sec} deg/s")
    except ValueError as e:
        print(f"\n⚠️  Could not find best: {e}")


def sampling_rate_adaptive_window():
    """
    Test: Window-Größe automatisch an Sampling-Rate anpassen.
    
    Idee: Bei höherer Sampling-Rate können kleinere Fenster verwendet werden.
    120 Hz → 8.33 ms Sample-Abstand → Window sollte mindestens 2-3 Samples sein
    """
    print("\n" + "="*80)
    print("SAMPLING RATE ADAPTIVE WINDOW SIZING")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    # At 120 Hz:
    # 1 Sample = ~8.33 ms
    # 2 Samples = ~16.67 ms
    # 3 Samples = ~25 ms
    # 5 Samples = ~41.67 ms
    
    window_configs = [
        {"samples": 2, "window_ms": 16.67, "name": "2_samples"},
        {"samples": 3, "window_ms": 25.0, "name": "3_samples"},
        {"samples": 4, "window_ms": 33.33, "name": "4_samples"},
        {"samples": 5, "window_ms": 41.67, "name": "5_samples"},
    ]
    
    fixed_threshold = 30.0
    
    for cfg in window_configs:
        print(f"\n{'─'*40}")
        print(f"Testing: {cfg['samples']} samples (~{cfg['window_ms']:.1f} ms)")
        print(f"{'─'*40}")
        
        velocity_config = OlsenVelocityConfig(
            window_length_ms=cfg["window_ms"],
            velocity_method="olsen2d",
            eye_mode="average",
        )
        
        classifier_config = IVTClassifierConfig(
            velocity_threshold_deg_per_sec=fixed_threshold,
        )
        
        exp_config = ExperimentConfig(
            name=f"adaptive_window_{cfg['name']}",
            description=f"Window size: {cfg['samples']} samples at 120Hz (~{cfg['window_ms']:.1f} ms)",
            velocity_config=velocity_config,
            classifier_config=classifier_config,
            tags=["adaptive_window", "120hz", f"{cfg['samples']}_samples"],
            metadata={
                "window_ms": cfg["window_ms"],
                "samples_at_120hz": cfg["samples"],
                "sampling_rate": "120hz",
            }
        )
        
        pipeline = IVTPipeline(velocity_config, classifier_config)
        pipeline.register_observer(ConsoleReporter(verbose=False))
        pipeline.register_observer(MetricsLogger("experiments/adaptive_window_metrics.csv"))
        pipeline.register_observer(ExperimentTracker("experiments"))
        
        try:
            df = pipeline.run_with_tracking(
                input_path=input_file,
                config=exp_config,
                evaluate=True,
            )
            print(f"✅ Completed")
        except Exception as e:
            print(f"❌ Failed: {e}")


def main():
    """Run all window sweep experiments."""
    print("\n" + "="*80)
    print("WINDOW SIZE PARAMETER SWEEP")
    print("Testing different velocity calculation windows")
    print("="*80)
    
    # 1. Basis Window Sweep
    window_length_sweep()
    
    # 2. Analyse der Ergebnisse
    analyze_window_results()
    
    # Optional: Weitere Tests
    print("\n\nWeitere Tests verfügbar:")
    print("- window_with_smoothing_sweep(): Window + Smoothing Kombinationen")
    print("- window_method_comparison(): Olsen2D vs Ray3D bei versch. Windows")
    print("- sampling_rate_adaptive_window(): Sample-basierte Window-Größen")
    
    # Uncomment to run additional tests:
    # window_with_smoothing_sweep()
    # window_method_comparison()
    # sampling_rate_adaptive_window()
    
    print("\n" + "="*80)
    print("✅ Window sweep completed!")
    print("="*80)
    print("\nErgebnisse:")
    print("  - experiments/window_sweep_metrics.csv")
    print("  - experiments/{experiment_name}/")
    print()


if __name__ == "__main__":
    main()
