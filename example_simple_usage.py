#!/usr/bin/env python3
"""
Simple usage examples for users WITHOUT ground truth data.

This demonstrates how easy it is to use the I-VT filter
for normal eye tracking analysis.
"""

from ivt_filter.simple_api import (
    process_eye_tracking,
    process_eye_tracking_adaptive,
    print_statistics
)


def example_1_minimal():
    """Simplest possible usage - just process the data."""
    print("=" * 60)
    print("Example 1: Minimal Usage")
    print("=" * 60)
    
    # Just one line!
    df = process_eye_tracking(
        "I-VT-frequency120Fixation_input.tsv",
        show_plot=False  # Disable plot for this example
    )
    
    print(f"\n✅ Processed {len(df)} samples")
    print(f"✅ Columns added: {[c for c in df.columns if c not in ['time_ms', 'gaze_left_x_mm']][:5]}")
    print(f"✅ Fixations detected: {(df['ivt_sample_type'] == 'Fixation').sum()}")


def example_2_with_output():
    """Save results to file."""
    print("\n" + "=" * 60)
    print("Example 2: Save Results to File")
    print("=" * 60)
    
    df = process_eye_tracking(
        input_path="I-VT-frequency120Fixation_input.tsv",
        output_path="simple_api_output.tsv",  # Save here!
        show_plot=False
    )
    
    print(f"\n✅ Results saved to: simple_api_output.tsv")
    print(f"✅ File contains {len(df)} rows with velocity and classification")


def example_3_custom_threshold():
    """Try different velocity thresholds."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Velocity Threshold")
    print("=" * 60)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    
    # Test different thresholds
    for threshold in [20.0, 30.0, 40.0]:
        df = process_eye_tracking(
            input_file,
            velocity_threshold=threshold,
            show_plot=False
        )
        
        fixations = (df['ivt_sample_type'] == 'Fixation').sum()
        percentage = (fixations / len(df)) * 100
        
        print(f"\nThreshold {threshold} deg/s:")
        print(f"  Fixations: {fixations} ({percentage:.1f}%)")


def example_4_with_statistics():
    """Process and show statistics."""
    print("\n" + "=" * 60)
    print("Example 4: With Statistics")
    print("=" * 60)
    
    df = process_eye_tracking(
        "I-VT-frequency120Fixation_input.tsv",
        show_plot=False
    )
    
    # Print nice statistics
    print_statistics(df)


def example_5_adaptive_sampling():
    """Adaptive sample-based window (recommended)."""
    print("\n" + "=" * 60)
    print("Example 5: Adaptive Sample-Based (Recommended)")
    print("=" * 60)
    
    # Automatically detects sampling rate and uses fixed samples
    df = process_eye_tracking_adaptive(
        input_path="I-VT-frequency120Fixation_input.tsv",
        n_samples=3,  # Use 3 samples regardless of sampling rate
        velocity_threshold=30.0,
        show_plot=False
    )
    
    print(f"\n✅ Processed with adaptive sample-based window")
    print(f"✅ Window automatically adjusted to sampling rate")
    
    print_statistics(df)


def example_6_left_eye_only():
    """Use only left eye with 3D method."""
    print("\n" + "=" * 60)
    print("Example 6: Left Eye Only + Ray 3D Method")
    print("=" * 60)
    
    df = process_eye_tracking(
        input_path="I-VT-frequency120Fixation_input.tsv",
        eye_selection="left",  # Only left eye
        velocity_method="ray3d",  # 3D calculation
        show_plot=False
    )
    
    print(f"\n✅ Processed using LEFT eye only")
    print(f"✅ Used physically correct 3D ray method")
    
    print_statistics(df)


def example_7_compare_methods():
    """Compare different configurations."""
    print("\n" + "=" * 60)
    print("Example 7: Compare Different Methods")
    print("=" * 60)
    
    input_file = "I-VT-frequency120Fixation_input.tsv"
    
    configs = [
        ("Time-based 20ms", {"window_ms": 20.0}),
        ("Time-based 40ms", {"window_ms": 40.0}),
    ]
    
    print("\nComparison:")
    print("-" * 50)
    
    for name, kwargs in configs:
        df = process_eye_tracking(
            input_file,
            show_plot=False,
            **kwargs
        )
        
        fixations = (df['ivt_sample_type'] == 'Fixation').sum()
        percentage = (fixations / len(df)) * 100
        
        # Get velocity column
        vel_col = 'velocity_deg_per_sec'
        avg_vel = df[vel_col].mean() if vel_col in df.columns else 0.0
        
        print(f"\n{name}:")
        print(f"  Fixations: {percentage:.1f}%")
        print(f"  Avg velocity: {avg_vel:.1f} deg/s")


def example_8_production_workflow():
    """Typical production workflow."""
    print("\n" + "=" * 60)
    print("Example 8: Production Workflow")
    print("=" * 60)
    
    # Step 1: Process with adaptive method
    print("\n📊 Step 1: Processing eye tracking data...")
    df = process_eye_tracking_adaptive(
        input_path="I-VT-frequency120Fixation_input.tsv",
        output_path="production_results.tsv",
        n_samples=3,
        velocity_threshold=30.0,
        show_plot=False
    )
    
    # Step 2: Get statistics
    print("\n📊 Step 2: Computing statistics...")
    print_statistics(df)
    
    # Step 3: Extract fixations only
    print("\n📊 Step 3: Extracting fixations...")
    fixations = df[df['ivt_sample_type'] == 'Fixation'].copy()
    print(f"   Found {len(fixations)} fixation samples")
    
    # Step 4: Analyze fixation durations
    print("\n📊 Step 4: Analyzing fixation durations...")
    fixations['fixation_id'] = (
        fixations['ivt_sample_type'] != fixations['ivt_sample_type'].shift()
    ).cumsum()
    
    fixation_durations = (
        fixations.groupby('fixation_id')['time_ms']
        .apply(lambda x: x.max() - x.min())
    )
    
    print(f"   Average fixation duration: {fixation_durations.mean():.1f} ms")
    print(f"   Median fixation duration: {fixation_durations.median():.1f} ms")
    print(f"   Number of fixations: {len(fixation_durations)}")
    
    print("\n✅ Production workflow complete!")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("SIMPLE API EXAMPLES - NO GROUND TRUTH NEEDED")
    print("=" * 60)
    
    # Run examples
    example_1_minimal()
    example_2_with_output()
    example_3_custom_threshold()
    example_4_with_statistics()
    example_5_adaptive_sampling()
    example_6_left_eye_only()
    example_7_compare_methods()
    example_8_production_workflow()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 60)
    print("\n💡 Key Takeaway:")
    print("   The simple API requires NO ground truth data!")
    print("   Just call process_eye_tracking() with your TSV file.")
    print("\n🚀 Ready to use in production!")


if __name__ == "__main__":
    main()
