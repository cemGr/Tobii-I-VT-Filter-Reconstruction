#!/usr/bin/env python3
"""
Parameter sweep for different window lengths (sample windows).

Tests different velocity calculation windows to find optimal configuration.

Usage:
    python example_window_sweep.py
"""
from pathlib import Path

from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.evaluation.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.io.observers import ConsoleReporter, MetricsLogger, ExperimentTracker
from ivt_filter.io.pipeline import IVTPipeline


def window_length_sweep():
    """
    Test different window sizes (sample windows).

    Typical values:
    - 1 ms: very small window (almost no smoothing)
    - 10 ms: small window
    - 20 ms: standard (Olsen paper)
    - 40 ms: larger window
    - 60 ms: very large
    """
    print("\n" + "="*80)
    print("WINDOW LENGTH SWEEP - Testing Different Sample Windows")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    # Test different window sizes (in milliseconds)
    window_sizes = [1.0, 10.0, 20.0, 40.0, 60.0]
    
    # Fixed threshold for fair comparison
    fixed_threshold = 30.0
    
    for window_ms in window_sizes:
        print(f"\n{'─'*80}")
        print(f"Testing window: {window_ms} ms")
        print(f"{'─'*80}")
        
        # Velocity configuration with different window sizes
        velocity_config = OlsenVelocityConfig(
            window_length_ms=window_ms,
            velocity_method="olsen2d",
            eye_mode="average",
            smoothing_mode="none",  # No smoothing for a fair comparison
        )
        
        classifier_config = IVTClassifierConfig(
            velocity_threshold_deg_per_sec=fixed_threshold,
        )
        
        # Experiment config
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
        
        # Pipeline with observers
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
    Test different window sizes WITH smoothing.

    Combines window size with different smoothing methods.
    """
    print("\n" + "="*80)
    print("WINDOW + SMOOTHING SWEEP")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    # Window sizes
    window_sizes = [10.0, 20.0, 40.0]

    # Smoothing methods
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
    Compares Olsen2D vs. Ray3D at different window sizes.
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
    Analyzes the results of the window sweep.
    """
    print("\n" + "="*80)
    print("ANALYZING WINDOW SWEEP RESULTS")
    print("="*80)
    
    manager = ExperimentManager("experiments")
    
    # Find window sweep experiments
    window_experiments = manager.list_experiments(tags=["window_sweep"])
    
    if not window_experiments:
        print("⚠️  No window sweep experiments found. Run window_length_sweep() first!")
        return
    
    print(f"\nFound {len(window_experiments)} window sweep experiments")
    
    # Compare experiments
    exp_names = [e["name"] for e in window_experiments]
    comparison = manager.compare_experiments(exp_names)

    # Sort by window size
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
    
    # Find the best configuration
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
    Test: automatically adapt the window size to the sampling rate.

    Idea: at a higher sampling rate, smaller windows can be used.
    120 Hz -> 8.33 ms sample spacing -> window should be at least 2-3 samples
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
    
    # 1. Basic window sweep
    window_length_sweep()

    # 2. Analyze the results
    analyze_window_results()

    # Optional: further tests
    print("\n\nFurther tests available:")
    print("- window_with_smoothing_sweep(): window + smoothing combinations")
    print("- window_method_comparison(): Olsen2D vs Ray3D across different windows")
    print("- sampling_rate_adaptive_window(): sample-based window sizes")
    
    # Uncomment to run additional tests:
    # window_with_smoothing_sweep()
    # window_method_comparison()
    # sampling_rate_adaptive_window()
    
    print("\n" + "="*80)
    print("✅ Window sweep completed!")
    print("="*80)
    print("\nResults:")
    print("  - experiments/window_sweep_metrics.csv")
    print("  - experiments/{experiment_name}/")
    print()


if __name__ == "__main__":
    main()
