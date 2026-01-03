#!/usr/bin/env python3
"""
Debug-Evaluation: Warum ist der Recall von Unclassified/EyesNotFound so niedrig?
"""
import pandas as pd
from ivt_filter.io import read_tsv
from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig
from ivt_filter.velocity import compute_olsen_velocity
from ivt_filter.classification import apply_ivt_classifier
from ivt_filter.evaluation import compute_ivt_metrics

# Pipeline
input_file = "./I-VT-botheye20ms30threshold_input.tsv"
df = read_tsv(input_file)

cfg = OlsenVelocityConfig(window_length_ms=20, eye_mode='average', max_validity=1, gap_fill_enabled=False)
df = compute_olsen_velocity(df, cfg)

clf_cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=30)
df = apply_ivt_classifier(df, cfg=clf_cfg)

# GT-Labels prüfen
print("=== Ground Truth Labels ===")
if 'gt_event_type' in df.columns:
    gt_col = 'gt_event_type'
    print(f"GT column found: {gt_col}")
    print(f"Unique GT labels: {df[gt_col].unique()}")
    print(f"GT label counts:\n{df[gt_col].value_counts()}")
    
    print(f"\n=== Predicted Labels ===")
    print(f"Unique Pred labels: {df['ivt_sample_type'].unique()}")
    print(f"Pred label counts:\n{df['ivt_sample_type'].value_counts()}")
    
    # Evaluation
    print("\n=== Evaluation ===")
    metrics = compute_ivt_metrics(df, gt_col=gt_col, pred_col='ivt_sample_type')
    
    print(f"\nMetrics:")
    print(f"  n_gt_uncl: {metrics['n_gt_uncl']} (GT Unclassified)")
    print(f"  n_correct_uncl: {metrics['n_correct_uncl']} (korrekt als Unclassified erkannt)")
    print(f"  recall_uncl: {metrics['recall_uncl']:.1f}%")
    
    print(f"\n  n_gt_eynf: {metrics['n_gt_eynf']} (GT EyesNotFound)")
    print(f"  n_correct_eynf: {metrics['n_correct_eynf']} (korrekt als EyesNotFound erkannt)")
    print(f"  recall_eynf: {metrics['recall_eynf']:.1f}%")
    
    # Detaillierte Confusion für Unclassified/EyesNotFound
    print(f"\n=== Confusion für GT Unclassified ===")
    gt_uncl = df[df[gt_col] == 'Unclassified']
    print(f"GT Unclassified -> Pred labels:")
    print(gt_uncl['ivt_sample_type'].value_counts())
    
    print(f"\n=== Confusion für GT EyesNotFound ===")
    if 'EyesNotFound' in df[gt_col].values:
        gt_eynf = df[df[gt_col] == 'EyesNotFound']
        print(f"GT EyesNotFound -> Pred labels:")
        print(gt_eynf['ivt_sample_type'].value_counts())
    else:
        print("Keine GT EyesNotFound in Daten!")
        
else:
    print("No gt_event_type column found!")
    print(f"Available columns: {list(df.columns)}")
