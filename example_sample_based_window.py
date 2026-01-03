#!/usr/bin/env python3
"""
Sample-based window size testing (instead of time-based).

Instead of using fixed milliseconds (e.g., 20 ms), this uses a fixed
number of samples (e.g., 3 samples), which adapts to different sampling rates.

Example:
    3 samples at 60 Hz  = 50 ms
    3 samples at 120 Hz = 25 ms
    3 samples at 300 Hz = 10 ms

Usage:
    python example_sample_based_window.py
"""
from pathlib import Path
import pandas as pd

from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.observers import ConsoleReporter, MetricsLogger, ExperimentTracker
from ivt_filter.pipeline import IVTPipeline
from ivt_filter.sampling import estimate_sampling_rate
from ivt_filter.io import read_tsv


def detect_sampling_rate_from_file(file_path: str) -> float:
    """
    Detect sampling rate from input file.
    
    Returns:
        Nominal sampling rate in Hz (e.g., 120.0, 300.0)
    """
    df = read_tsv(file_path)
    
    if "time_ms" not in df.columns:
        raise ValueError("File must contain 'time_ms' column")
    
    sampling_info = estimate_sampling_rate(df, time_col="time_ms")
    nominal_hz = sampling_info["nominal_hz"]
    
    print(f"\n📊 Detected sampling rate: {nominal_hz} Hz")
    print(f"   Median dt: {sampling_info['median_dt_ms']:.2f} ms")
    print(f"   Mean dt: {sampling_info['mean_dt_ms']:.2f} ms")
    
    return nominal_hz


def samples_to_milliseconds(n_samples: int, sampling_rate_hz: float) -> float:
    """
    Convert number of samples to milliseconds.
    
    Args:
        n_samples: Number of samples
        sampling_rate_hz: Sampling rate in Hz
    
    Returns:
        Time duration in milliseconds
    
    Example:
        >>> samples_to_milliseconds(3, 120.0)
        25.0  # 3 samples at 120 Hz = 25 ms
    """
    return (n_samples / sampling_rate_hz) * 1000.0


def sample_based_window_sweep():
    """
    Test different SAMPLE-BASED window sizes.
    
    Instead of fixed milliseconds, uses fixed number of samples.
    This is more robust across different sampling rates.
    """
    print("\n" + "="*80)
    print("SAMPLE-BASED WINDOW SWEEP")
    print("Using fixed number of samples instead of fixed milliseconds")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    # Detect sampling rate
    sampling_rate_hz = detect_sampling_rate_from_file(input_file)
    
    # Different sample counts to test
    sample_counts = [2, 3, 4, 5, 7, 10]
    
    print(f"\n🔬 Testing {len(sample_counts)} different sample window sizes:")
    print(f"   At {sampling_rate_hz} Hz:")
    for n in sample_counts:
        ms = samples_to_milliseconds(n, sampling_rate_hz)
        print(f"   - {n} samples = {ms:.2f} ms")
    
    fixed_threshold = 30.0
    
    for n_samples in sample_counts:
        # Convert samples to milliseconds for current sampling rate
        window_ms = samples_to_milliseconds(n_samples, sampling_rate_hz)
        
        print(f"\n{'─'*80}")
        print(f"Testing: {n_samples} samples ({window_ms:.2f} ms at {sampling_rate_hz} Hz)")
        print(f"{'─'*80}")
        
        velocity_config = OlsenVelocityConfig(
            window_length_ms=window_ms,
            velocity_method="olsen2d",
            eye_mode="average",
            smoothing_mode=None,
        )
        
        classifier_config = IVTClassifierConfig(
            velocity_threshold_deg_per_sec=fixed_threshold,
        )
        
        exp_config = ExperimentConfig(
            name=f"sample_based_{n_samples}samples",
            description=f"Window: {n_samples} samples ({window_ms:.2f} ms at {sampling_rate_hz} Hz)",
            velocity_config=velocity_config,
            classifier_config=classifier_config,
            tags=["sample_based", f"{n_samples}_samples", f"{int(sampling_rate_hz)}hz"],
            metadata={
                "n_samples": n_samples,
                "window_ms": window_ms,
                "sampling_rate_hz": sampling_rate_hz,
                "window_type": "sample_based",
            }
        )
        
        pipeline = IVTPipeline(velocity_config, classifier_config)
        pipeline.register_observer(ConsoleReporter(verbose=False))
        pipeline.register_observer(MetricsLogger("experiments/sample_based_metrics.csv"))
        pipeline.register_observer(ExperimentTracker("experiments"))
        
        try:
            df = pipeline.run_with_tracking(
                input_path=input_file,
                config=exp_config,
                evaluate=True,
            )
            print(f"✅ {n_samples} samples completed ({len(df)} data points)")
        except Exception as e:
            print(f"❌ {n_samples} samples failed: {e}")


def compare_time_vs_sample_based():
    """
    Compare time-based windows vs sample-based windows.
    
    Shows difference between:
    - Fixed 20 ms (varies in samples at different rates)
    - Fixed 3 samples (varies in ms at different rates)
    """
    print("\n" + "="*80)
    print("TIME-BASED vs SAMPLE-BASED COMPARISON")
    print("="*80)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    if not Path(input_file).exists():
        print(f"⚠️  Input file '{input_file}' not found.")
        return
    
    sampling_rate_hz = detect_sampling_rate_from_file(input_file)
    
    # Strategy 1: Fixed time (20 ms)
    fixed_time_ms = 20.0
    samples_at_this_rate = int((fixed_time_ms / 1000.0) * sampling_rate_hz)
    
    print(f"\n📊 Strategy Comparison at {sampling_rate_hz} Hz:")
    print(f"   Time-based: 20 ms = {samples_at_this_rate} samples at this rate")
    print(f"   Sample-based: 3 samples = {samples_to_milliseconds(3, sampling_rate_hz):.2f} ms at this rate")
    
    configs = [
        {
            "name": "time_based_20ms",
            "description": "Fixed time: 20 ms",
            "window_ms": 20.0,
            "strategy": "time_based",
            "tags": ["comparison", "time_based"],
        },
        {
            "name": "sample_based_3samples",
            "description": "Fixed samples: 3 samples",
            "window_ms": samples_to_milliseconds(3, sampling_rate_hz),
            "strategy": "sample_based",
            "tags": ["comparison", "sample_based"],
        },
    ]
    
    fixed_threshold = 30.0
    
    for cfg in configs:
        print(f"\n{'─'*40}")
        print(f"Testing: {cfg['description']}")
        print(f"         Window: {cfg['window_ms']:.2f} ms")
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
            name=cfg["name"],
            description=cfg["description"],
            velocity_config=velocity_config,
            classifier_config=classifier_config,
            tags=cfg["tags"],
            metadata={
                "window_ms": cfg["window_ms"],
                "strategy": cfg["strategy"],
                "sampling_rate_hz": sampling_rate_hz,
            }
        )
        
        pipeline = IVTPipeline(velocity_config, classifier_config)
        pipeline.register_observer(ConsoleReporter(verbose=False))
        pipeline.register_observer(MetricsLogger("experiments/comparison_metrics.csv"))
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


def multi_rate_sample_consistency():
    """
    Test consistency of sample-based approach across different datasets.
    
    If you have multiple datasets with different sampling rates,
    sample-based windows ensure consistent behavior.
    """
    print("\n" + "="*80)
    print("MULTI-RATE SAMPLE CONSISTENCY TEST")
    print("="*80)
    
    # List of files with potentially different sampling rates
    test_files = [
        "I-VT-frequency120Fixation_input.tsv",  # 120 Hz
        # Add more files if available:
        # "data_60hz.tsv",    # 60 Hz
        # "data_300hz.tsv",   # 300 Hz
    ]
    
    # Fixed sample count - should give consistent behavior across rates
    fixed_samples = 3
    fixed_threshold = 30.0
    
    for input_file in test_files:
        if not Path(input_file).exists():
            print(f"⚠️  Skipping '{input_file}' (not found)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing file: {input_file}")
        print(f"{'='*80}")
        
        # Detect sampling rate for this file
        sampling_rate_hz = detect_sampling_rate_from_file(input_file)
        window_ms = samples_to_milliseconds(fixed_samples, sampling_rate_hz)
        
        print(f"\n📊 Configuration:")
        print(f"   Sampling rate: {sampling_rate_hz} Hz")
        print(f"   Fixed samples: {fixed_samples}")
        print(f"   Window size: {window_ms:.2f} ms")
        
        velocity_config = OlsenVelocityConfig(
            window_length_ms=window_ms,
            velocity_method="olsen2d",
            eye_mode="average",
        )
        
        classifier_config = IVTClassifierConfig(
            velocity_threshold_deg_per_sec=fixed_threshold,
        )
        
        dataset_name = Path(input_file).stem
        exp_config = ExperimentConfig(
            name=f"consistency_{dataset_name}_{fixed_samples}samples",
            description=f"{fixed_samples} samples on {dataset_name} ({sampling_rate_hz} Hz)",
            velocity_config=velocity_config,
            classifier_config=classifier_config,
            tags=["multi_rate", "consistency", f"{fixed_samples}_samples"],
            metadata={
                "n_samples": fixed_samples,
                "window_ms": window_ms,
                "sampling_rate_hz": sampling_rate_hz,
                "dataset": dataset_name,
            }
        )
        
        pipeline = IVTPipeline(velocity_config, classifier_config)
        pipeline.register_observer(ConsoleReporter(verbose=False))
        pipeline.register_observer(MetricsLogger("experiments/multi_rate_metrics.csv"))
        pipeline.register_observer(ExperimentTracker("experiments"))
        
        try:
            df = pipeline.run_with_tracking(
                input_path=input_file,
                config=exp_config,
                evaluate=True,
            )
            print(f"✅ Completed: {len(df)} samples processed")
        except Exception as e:
            print(f"❌ Failed: {e}")


def analyze_sample_based_results():
    """
    Analyze results from sample-based window experiments.
    """
    print("\n" + "="*80)
    print("ANALYZING SAMPLE-BASED RESULTS")
    print("="*80)
    
    manager = ExperimentManager("experiments")
    
    # Find sample-based experiments
    sample_experiments = manager.list_experiments(tags=["sample_based"])
    
    if not sample_experiments:
        print("⚠️  No sample-based experiments found. Run sample_based_window_sweep() first!")
        return
    
    print(f"\nFound {len(sample_experiments)} sample-based experiments")
    
    # Compare experiments
    exp_names = [e["name"] for e in sample_experiments]
    comparison = manager.compare_experiments(exp_names)
    
    # Sort by number of samples
    if "metadata" in comparison.columns:
        print("\n" + "─"*80)
        print("SAMPLE-BASED WINDOW COMPARISON")
        print("─"*80)
        print(comparison[[
            "experiment",
            "window_ms",
            "percentage_agreement",
            "fixation_recall",
            "saccade_recall",
            "cohen_kappa"
        ]].to_string(index=False))
    
    # Find best configuration
    try:
        best_name, best_value, best_config = manager.get_best_configuration(
            metric="percentage_agreement",
            tags=["sample_based"]
        )
        
        n_samples = best_config.metadata.get("n_samples", "?")
        sampling_rate = best_config.metadata.get("sampling_rate_hz", "?")
        
        print("\n" + "="*80)
        print("🏆 BEST SAMPLE-BASED CONFIGURATION")
        print("="*80)
        print(f"Experiment: {best_name}")
        print(f"Agreement: {best_value:.2f}%")
        print(f"Number of samples: {n_samples}")
        print(f"Window size: {best_config.velocity_config.window_length_ms:.2f} ms")
        print(f"At sampling rate: {sampling_rate} Hz")
        print(f"\n💡 This means using {n_samples} samples is optimal at {sampling_rate} Hz")
        print(f"   For other rates, adjust window to maintain {n_samples} samples")
    except ValueError as e:
        print(f"\n⚠️  Could not find best: {e}")


def create_sample_window_config_helper():
    """
    Helper function to create sample-based window configs.
    
    Usage example in your own code:
    
        from example_sample_based_window import create_sample_based_config
        
        # Create config for 3 samples at 120 Hz
        config = create_sample_based_config(
            n_samples=3,
            sampling_rate_hz=120.0,
            method="olsen2d"
        )
    """
    print("\n" + "="*80)
    print("HELPER FUNCTION FOR SAMPLE-BASED CONFIGS")
    print("="*80)
    
    print("""
def create_sample_based_config(
    n_samples: int,
    sampling_rate_hz: float,
    method: str = "olsen2d",
    eye_mode: str = "average",
    threshold: float = 30.0
) -> tuple[OlsenVelocityConfig, IVTClassifierConfig]:
    '''
    Create configurations with sample-based window sizing.
    
    Args:
        n_samples: Number of samples for window
        sampling_rate_hz: Sampling rate in Hz
        method: Velocity calculation method
        eye_mode: Eye selection mode
        threshold: Velocity threshold
    
    Returns:
        Tuple of (velocity_config, classifier_config)
    
    Example:
        >>> vel_cfg, clf_cfg = create_sample_based_config(3, 120.0)
        >>> # Uses 3 samples = 25 ms at 120 Hz
    '''
    window_ms = (n_samples / sampling_rate_hz) * 1000.0
    
    velocity_config = OlsenVelocityConfig(
        window_length_ms=window_ms,
        velocity_method=method,
        eye_mode=eye_mode,
    )
    
    classifier_config = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=threshold,
    )
    
    return velocity_config, classifier_config

# Usage:
vel_cfg, clf_cfg = create_sample_based_config(
    n_samples=3,
    sampling_rate_hz=120.0
)
print(f"Window size: {vel_cfg.window_length_ms} ms")
# Output: Window size: 25.0 ms
""")


def main():
    """Run sample-based window experiments."""
    print("\n" + "="*80)
    print("SAMPLE-BASED WINDOW SIZE TESTING")
    print("Testing fixed number of samples (adaptive to sampling rate)")
    print("="*80)
    
    # 1. Sample-based window sweep
    sample_based_window_sweep()
    
    # 2. Analyze results
    analyze_sample_based_results()
    
    # 3. Compare time-based vs sample-based
    compare_time_vs_sample_based()
    
    # Optional: Multi-rate consistency test
    # multi_rate_sample_consistency()
    
    # Show helper
    create_sample_window_config_helper()
    
    print("\n" + "="*80)
    print("✅ Sample-based window testing completed!")
    print("="*80)
    print("\nErgebnisse:")
    print("  - experiments/sample_based_metrics.csv")
    print("  - experiments/comparison_metrics.csv")
    print("  - experiments/{experiment_name}/")
    print()


if __name__ == "__main__":
    main()
