#!/usr/bin/env python3
"""
Debug-Analyse: Warum werden GT Unclassified/EyesNotFound nicht korrekt erkannt?
"""
import pandas as pd
import sys
from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.io import read_tsv
from ivt_filter.velocity import compute_olsen_velocity
from ivt_filter.classification import apply_ivt_classifier

# Datei laden
input_file = "./I-VT-botheye20ms30threshold_input.tsv"
df = read_tsv(input_file)

print(f"Loaded {len(df)} samples from {input_file}")
print(f"Columns: {list(df.columns)}")

# Pipeline mit gleichen Settings wie CLI
cfg = OlsenVelocityConfig(
    window_length_ms=20,
    eye_mode='average',
    max_validity=1,
    gap_fill_enabled=False,
)

print("\n=== Vor Velocity ===")
print(f"unique combined_valid (falls vorhanden): {df['combined_valid'].unique() if 'combined_valid' in df.columns else 'N/A'}")

# Velocity berechnen
df = compute_olsen_velocity(df, cfg)

print("\n=== Nach Velocity ===")
print(f"Samples mit combined_valid=False: {(df['combined_valid'] == False).sum()}")
print(f"Samples mit velocity=NaN: {df['velocity_deg_per_sec'].isna().sum()}")

# Klassifikation
clf_cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=30)
df = apply_ivt_classifier(df, cfg=clf_cfg)

print("\n=== Nach Klassifikation ===")
print(f"Samples mit label 'EyesNotFound': {(df['ivt_sample_type'] == 'EyesNotFound').sum()}")
print(f"Samples mit label 'Unclassified': {(df['ivt_sample_type'] == 'Unclassified').sum()}")

# Problem-Analyse: Samples mit combined_valid=False aber label != EyesNotFound
problem_mask = (df['combined_valid'] == False) & (df['ivt_sample_type'] != 'EyesNotFound')
problem_count = problem_mask.sum()

if problem_count > 0:
    print(f"\n✗ FEHLER: {problem_count} Samples haben combined_valid=False aber nicht 'EyesNotFound'!")
    print("\nBeispiele (erste 10):")
    problem_samples = df[problem_mask][['time_ms', 'combined_valid', 'velocity_deg_per_sec', 'ivt_sample_type']].head(10)
    print(problem_samples)
else:
    print(f"\n✓ Alle Samples mit combined_valid=False sind korrekt als 'EyesNotFound' gelabelt")

# Weitere Analyse: Samples mit combined_valid=True aber label='EyesNotFound'
problem_mask2 = (df['combined_valid'] == True) & (df['ivt_sample_type'] == 'EyesNotFound')
problem_count2 = problem_mask2.sum()

if problem_count2 > 0:
    print(f"\n✗ FEHLER: {problem_count2} Samples haben combined_valid=True aber 'EyesNotFound'!")
    print("\nBeispiele (erste 10):")
    problem_samples2 = df[problem_mask2][['time_ms', 'combined_valid', 'velocity_deg_per_sec', 'ivt_sample_type']].head(10)
    print(problem_samples2)
else:
    print(f"\n✓ Alle Samples mit label='EyesNotFound' haben combined_valid=False")

# Samples mit velocity aber combined_valid=False
problem_mask3 = (df['combined_valid'] == False) & (df['velocity_deg_per_sec'].notna())
problem_count3 = problem_mask3.sum()

if problem_count3 > 0:
    print(f"\n✗ FEHLER: {problem_count3} Samples haben combined_valid=False aber velocity!=NaN!")
    print("\nBeispiele (erste 10):")
    problem_samples3 = df[problem_mask3][['time_ms', 'combined_valid', 'velocity_deg_per_sec', 'ivt_sample_type']].head(10)
    print(problem_samples3)
else:
    print(f"\n✓ Alle Samples mit combined_valid=False haben velocity=NaN")

print("\n=== Summary ===")
print(f"Total samples: {len(df)}")
print(f"combined_valid=True: {(df['combined_valid'] == True).sum()}")
print(f"combined_valid=False: {(df['combined_valid'] == False).sum()}")
print(f"ivt_sample_type='EyesNotFound': {(df['ivt_sample_type'] == 'EyesNotFound').sum()}")
print(f"ivt_sample_type='Unclassified': {(df['ivt_sample_type'] == 'Unclassified').sum()}")
