#!/usr/bin/env python3
"""
Analyse der IVTDEMO3.tsv: Was passiert mit GT-Unclassified Samples?
"""
import pandas as pd

# Output-Datei laden (mit GT + Pred Labels)
output_file = "./IVTDEMO3.tsv"
df = pd.read_csv(output_file, sep='\t')

# Convert velocity to numeric (handle 'NaN' strings)
df['velocity_deg_per_sec'] = pd.to_numeric(df['velocity_deg_per_sec'], errors='coerce')

print(f"Loaded {len(df)} samples from {output_file}")
print(f"\nColumns: {list(df.columns)}")

# GT Unclassified Samples
gt_uncl = df[df['gt_event_type'] == 'Unclassified']
print(f"\n=== GT Unclassified: {len(gt_uncl)} Samples ===")

# Wie werden sie klassifiziert?
print(f"\nPred labels für GT Unclassified:")
print(gt_uncl['ivt_sample_type'].value_counts())

# Detaillierte Analyse: Was ist die Velocity dieser Samples?
print(f"\n=== Velocity-Analyse für GT Unclassified ===")
print(f"Velocity min: {gt_uncl['velocity_deg_per_sec'].min():.2f}")
print(f"Velocity max: {gt_uncl['velocity_deg_per_sec'].max():.2f}")
print(f"Velocity mean: {gt_uncl['velocity_deg_per_sec'].mean():.2f}")
print(f"Velocity NaN count: {gt_uncl['velocity_deg_per_sec'].isna().sum()}")

# Grouped by pred label
print(f"\n=== Velocity grouped by Pred Label ===")
for label in ['Fixation', 'Saccade', 'Unclassified', 'EyesNotFound']:
    subset = gt_uncl[gt_uncl['ivt_sample_type'] == label]
    if len(subset) > 0:
        vel_mean = subset['velocity_deg_per_sec'].mean()
        vel_nan = subset['velocity_deg_per_sec'].isna().sum()
        print(f"Pred '{label}': {len(subset)} samples, mean velocity={vel_mean:.2f}, NaN count={vel_nan}")

# Beispiele: GT Unclassified -> Pred Saccade (594 Samples!)
print(f"\n=== Beispiele: GT Unclassified -> Pred Saccade ===")
gt_uncl_pred_sac = gt_uncl[gt_uncl['ivt_sample_type'] == 'Saccade'].head(20)
print(gt_uncl_pred_sac[['time_ms', 'combined_valid', 'velocity_deg_per_sec', 'gt_event_type', 'ivt_sample_type']])

# Beispiele: GT Unclassified -> Pred Unclassified (korrekt: 52 Samples)
print(f"\n=== Beispiele: GT Unclassified -> Pred Unclassified (KORREKT) ===")
gt_uncl_pred_uncl = gt_uncl[gt_uncl['ivt_sample_type'] == 'Unclassified'].head(10)
print(gt_uncl_pred_uncl[['time_ms', 'combined_valid', 'velocity_deg_per_sec', 'gt_event_type', 'ivt_sample_type']])

# Frage: Sind die GT Unclassified vielleicht nur Samples ohne valide Velocity?
print(f"\n=== Vermutung: GT Unclassified = Samples ohne valide Velocity? ===")
gt_uncl_with_vel = gt_uncl[gt_uncl['velocity_deg_per_sec'].notna()]
print(f"GT Unclassified mit velocity!=NaN: {len(gt_uncl_with_vel)}")
gt_uncl_without_vel = gt_uncl[gt_uncl['velocity_deg_per_sec'].isna()]
print(f"GT Unclassified mit velocity=NaN: {len(gt_uncl_without_vel)}")

if len(gt_uncl_with_vel) > 0:
    print(f"\nVon den GT Unclassified MIT velocity:")
    print(gt_uncl_with_vel['ivt_sample_type'].value_counts())
    print(f"Mean velocity: {gt_uncl_with_vel['velocity_deg_per_sec'].mean():.2f}")

if len(gt_uncl_without_vel) > 0:
    print(f"\nVon den GT Unclassified OHNE velocity (NaN):")
    print(gt_uncl_without_vel['ivt_sample_type'].value_counts())
