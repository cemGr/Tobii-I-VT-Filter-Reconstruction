#!/usr/bin/env python3
"""Quick test of simple API"""

from ivt_filter.simple_api import process_eye_tracking, print_statistics

print("Testing Simple API...")
print("=" * 60)

# Test 1: Basic processing
print("\n1. Basic Processing:")
df = process_eye_tracking(
    "I-VT-frequency120Fixation_input.tsv",
    show_plot=False
)
print(f"✅ Processed {len(df)} samples")
print(f"✅ Fixations: {(df['ivt_sample_type'] == 'Fixation').sum()}")

# Test 2: With statistics
print("\n2. With Statistics:")
print_statistics(df)

# Test 3: Save output
print("\n3. Save Results:")
df = process_eye_tracking(
    "I-VT-frequency120Fixation_input.tsv",
    output_path="test_simple_output.tsv",
    show_plot=False
)
print("✅ Saved to: test_simple_output.tsv")

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("\n💡 Simple API works WITHOUT ground truth!")
