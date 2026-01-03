#!/usr/bin/env python3
"""
Debug-Skript zur Prüfung der kompletten Pipeline:
Windowing -> Velocity -> Classification

Szenario: Fenster mit 3-5 Samples, davon etwa 50% ungültig
"""
import pandas as pd
import numpy as np
from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.velocity import compute_olsen_velocity
from ivt_filter.classification import apply_ivt_classifier

# Test-Case: 7 Samples, Sample 3-4 sind ungültig (50% im 3er Fenster um Sample 4)
df = pd.DataFrame({
    'time_ms': [0.0, 8.33, 16.67, 25.0, 33.33, 41.67, 50.0],  # 120 Hz
    'gaze_left_x_mm': [100.0, 101.0, 102.0, np.nan, np.nan, 105.0, 106.0],
    'gaze_left_y_mm': [100.0, 101.0, 102.0, np.nan, np.nan, 105.0, 106.0],
    'gaze_right_x_mm': [100.0, 101.0, 102.0, np.nan, np.nan, 105.0, 106.0],
    'gaze_right_y_mm': [100.0, 101.0, 102.0, np.nan, np.nan, 105.0, 106.0],
    'validity_left': [0, 0, 0, 999, 999, 0, 0],
    'validity_right': [0, 0, 0, 999, 999, 0, 0],
    'eye_left_z_mm': [600.0]*7,
    'eye_right_z_mm': [600.0]*7,
})

print("=== Input DataFrame ===")
print(df[['time_ms', 'gaze_left_x_mm', 'validity_left']])
print("\nSzenario: Sample 3-4 sind ungültig (beide Augen)")

# Config: 20ms Fenster (ca. 3 Samples bei 120 Hz)
cfg = OlsenVelocityConfig(
    window_length_ms=20,
    eye_mode='average',
    max_validity=1,
)

print("\n=== Nach Velocity-Berechnung ===")
df = compute_olsen_velocity(df, cfg)
print(df[['time_ms', 'combined_valid', 'velocity_deg_per_sec']])

clf_cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=30)
print("\n=== Nach Klassifikation ===")
df = apply_ivt_classifier(df, cfg=clf_cfg)
print(df[['time_ms', 'combined_valid', 'velocity_deg_per_sec', 'ivt_sample_type']])

print("\n=== Prüfung: Was sollte passieren? ===")
for i in range(len(df)):
    sample = df.iloc[i]
    expected = "EyesNotFound" if not sample['combined_valid'] else "?"
    actual = sample['ivt_sample_type']
    status = "✓" if (not sample['combined_valid'] and actual == "EyesNotFound") or (sample['combined_valid']) else "✗"
    print(f"Sample {i} (t={sample['time_ms']}): combined_valid={sample['combined_valid']}, "
          f"velocity={sample['velocity_deg_per_sec']:.2f if pd.notna(sample['velocity_deg_per_sec']) else 'NaN'}, "
          f"label={actual} {status}")
