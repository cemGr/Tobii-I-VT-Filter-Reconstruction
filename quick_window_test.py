#!/usr/bin/env python3
"""
Quick example: Test different window sizes in a few lines.

Usage:
    python quick_window_test.py
"""
from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.observers import ConsoleReporter, MetricsLogger, ExperimentTracker
from ivt_filter.pipeline import IVTPipeline

# Input file
INPUT_FILE = "I-VT-frequency120Fixation_input.tsv"

# Window sizes to test (in milliseconds)
WINDOW_SIZES = [1.0, 10.0, 20.0, 40.0, 60.0]

# Fixed parameters
THRESHOLD = 30.0
METHOD = "olsen2d"

print(f"\n🔬 Testing {len(WINDOW_SIZES)} different window sizes...")

for window_ms in WINDOW_SIZES:
    # Create configs
    velocity_config = OlsenVelocityConfig(
        window_length_ms=window_ms,
        velocity_method=METHOD,
        eye_mode="average",
    )
    
    classifier_config = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=THRESHOLD,
    )
    
    exp_config = ExperimentConfig(
        name=f"quick_window_{int(window_ms)}ms",
        description=f"Quick test: {window_ms} ms window",
        velocity_config=velocity_config,
        classifier_config=classifier_config,
        tags=["quick_test", f"window_{int(window_ms)}ms"],
    )
    
    # Create pipeline with observers
    pipeline = IVTPipeline(velocity_config, classifier_config)
    pipeline.register_observer(ConsoleReporter(verbose=False))
    pipeline.register_observer(MetricsLogger("experiments/quick_metrics.csv"))
    pipeline.register_observer(ExperimentTracker("experiments"))
    
    # Run
    try:
        df = pipeline.run_with_tracking(INPUT_FILE, exp_config, evaluate=True)
        print(f"  ✅ {window_ms} ms - Done!")
    except Exception as e:
        print(f"  ❌ {window_ms} ms - Error: {e}")

# Compare results
print(f"\n📊 Comparing results...")
manager = ExperimentManager("experiments")
quick_exps = [e["name"] for e in manager.list_experiments(tags=["quick_test"])]
comparison = manager.compare_experiments(quick_exps)

print("\nResults:")
print(comparison[["experiment", "window_ms", "percentage_agreement", 
                 "fixation_recall", "saccade_recall"]].to_string(index=False))

# Find best
best_name, best_value, _ = manager.get_best_configuration("percentage_agreement", tags=["quick_test"])
print(f"\n🏆 Best window: {best_name} with {best_value:.2f}% agreement")
