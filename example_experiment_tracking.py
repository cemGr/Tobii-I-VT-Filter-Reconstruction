#!/usr/bin/env python3
"""
Example script demonstrating the new Experiment Tracking and Observer Pattern.

This script shows how to:
1. Create experiment configurations
2. Run experiments with automatic tracking
3. Compare multiple experiments
4. Find the best configuration

Usage:
    python example_experiment_tracking.py
"""
from pathlib import Path

from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.observers import ConsoleReporter, MetricsLogger, ExperimentTracker
from ivt_filter.pipeline import IVTPipeline


def run_single_experiment():
    """Example 1: Run a single experiment with tracking."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Experiment with Automatic Tracking")
    print("="*80)
    
    # Create configuration
    velocity_config = OlsenVelocityConfig(
        window_length_ms=20.0,
        velocity_method="olsen2d",
        eye_mode="average",
        smoothing_mode="median",
        smoothing_window_samples=3,
    )
    
    classifier_config = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=30.0,
    )
    
    # Create experiment config
    exp_config = ExperimentConfig(
        name="baseline_olsen2d_median",
        description="Baseline: Olsen 2D with median smoothing, 20ms window, 30 deg/s threshold",
        velocity_config=velocity_config,
        classifier_config=classifier_config,
        tags=["baseline", "olsen2d", "median"],
    )
    
    # Create pipeline with observers
    pipeline = IVTPipeline(velocity_config, classifier_config)
    
    # Register observers for automatic tracking
    pipeline.register_observer(ConsoleReporter(verbose=True))
    pipeline.register_observer(MetricsLogger("experiments/metrics_log.csv"))
    pipeline.register_observer(ExperimentTracker("experiments"))
    
    # Run with tracking
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if Path(input_file).exists():
        df = pipeline.run_with_tracking(
            input_path=input_file,
            config=exp_config,
            evaluate=True,
        )
        print(f"\n✅ Experiment completed! Processed {len(df)} samples.")
    else:
        print(f"\n⚠️  Input file '{input_file}' not found. Skipping execution.")


def run_parameter_sweep():
    """Example 2: Run multiple experiments to compare different parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Parameter Sweep - Testing Different Thresholds")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found. Skipping parameter sweep.")
        return
    
    # Test different thresholds
    thresholds = [20.0, 30.0, 40.0, 50.0]
    
    for threshold in thresholds:
        velocity_config = OlsenVelocityConfig(
            window_length_ms=20.0,
            velocity_method="olsen2d",
            eye_mode="average",
            smoothing_mode="median",
        )
        
        classifier_config = IVTClassifierConfig(
            velocity_threshold_deg_per_sec=threshold,
        )
        
        exp_config = ExperimentConfig(
            name=f"threshold_sweep_{int(threshold)}deg",
            description=f"Testing threshold {threshold} deg/s",
            velocity_config=velocity_config,
            classifier_config=classifier_config,
            tags=["parameter_sweep", "threshold"],
        )
        
        pipeline = IVTPipeline(velocity_config, classifier_config)
        pipeline.register_observer(ConsoleReporter(verbose=False))
        pipeline.register_observer(MetricsLogger("experiments/metrics_log.csv"))
        pipeline.register_observer(ExperimentTracker("experiments"))
        
        try:
            df = pipeline.run_with_tracking(
                input_path=input_file,
                config=exp_config,
                evaluate=True,
            )
        except Exception as e:
            print(f"❌ Experiment failed: {e}")


def compare_experiments():
    """Example 3: Compare multiple experiments."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Comparing Experiments")
    print("="*80)
    
    manager = ExperimentManager("experiments")
    
    # List all experiments
    experiments = manager.list_experiments()
    
    if not experiments:
        print("⚠️  No experiments found. Run some experiments first!")
        return
    
    print(f"\nFound {len(experiments)} experiments:")
    for exp in experiments[:5]:  # Show first 5
        print(f"  - {exp['name']}: {exp['description']}")
    
    # Compare experiments
    if len(experiments) >= 2:
        exp_names = [e["name"] for e in experiments[:4]]  # Compare up to 4
        print(f"\nComparing experiments: {', '.join(exp_names)}")
        
        comparison = manager.compare_experiments(exp_names)
        print("\nComparison Results:")
        print(comparison.to_string(index=False))


def find_best_configuration():
    """Example 4: Find the best configuration based on a metric."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Finding Best Configuration")
    print("="*80)
    
    manager = ExperimentManager("experiments")
    
    experiments = manager.list_experiments()
    if not experiments:
        print("⚠️  No experiments found. Run some experiments first!")
        return
    
    # Find best by different metrics
    metrics_to_test = [
        ("percentage_agreement", True),
        ("fixation_recall", True),
        ("cohen_kappa", True),
    ]
    
    for metric, maximize in metrics_to_test:
        try:
            best_name, best_value, best_config = manager.get_best_configuration(
                metric=metric,
                maximize=maximize,
            )
            
            print(f"\n🏆 Best by {metric}: {best_name}")
            print(f"   Value: {best_value:.2f}")
            print(f"   Window: {best_config.velocity_config.window_length_ms} ms")
            print(f"   Method: {best_config.velocity_config.velocity_method}")
            print(f"   Threshold: {best_config.classifier_config.velocity_threshold_deg_per_sec} deg/s")
        except ValueError as e:
            print(f"\n⚠️  Could not find best for {metric}: {e}")


def run_experiment_with_tags():
    """Example 5: Use tags to organize experiments."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Organizing Experiments with Tags")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    # Create experiments with different tags
    configs = [
        {
            "name": "ray3d_experiment",
            "method": "ray3d",
            "tags": ["3d", "ray3d", "high-accuracy"],
        },
        {
            "name": "olsen2d_fast",
            "method": "olsen2d",
            "tags": ["2d", "fast", "baseline"],
        },
    ]
    
    for cfg_def in configs:
        velocity_config = OlsenVelocityConfig(
            window_length_ms=20.0,
            velocity_method=cfg_def["method"],
            eye_mode="average",
        )
        
        classifier_config = IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0)
        
        exp_config = ExperimentConfig(
            name=cfg_def["name"],
            description=f"Testing {cfg_def['method']} method",
            velocity_config=velocity_config,
            classifier_config=classifier_config,
            tags=cfg_def["tags"],
        )
        
        pipeline = IVTPipeline(velocity_config, classifier_config)
        pipeline.register_observer(ConsoleReporter(verbose=False))
        pipeline.register_observer(ExperimentTracker("experiments"))
        
        try:
            df = pipeline.run_with_tracking(input_path=input_file, config=exp_config)
        except Exception as e:
            print(f"❌ {cfg_def['name']} failed: {e}")
    
    # Query by tags
    manager = ExperimentManager("experiments")
    print("\nExperiments with '3d' tag:")
    for exp in manager.list_experiments(tags=["3d"]):
        print(f"  - {exp['name']}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("IVT Filter - Experiment Tracking & Observer Pattern Examples")
    print("="*80)
    
    # Run examples
    run_single_experiment()
    run_parameter_sweep()
    compare_experiments()
    find_best_configuration()
    run_experiment_with_tags()
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("="*80)
    print("\nCheck the following locations:")
    print("  - experiments/             : Saved experiment data")
    print("  - experiments/metrics_log.csv : Metrics log")
    print("  - experiments/experiments_index.json : Experiment index")
    print()


if __name__ == "__main__":
    main()
