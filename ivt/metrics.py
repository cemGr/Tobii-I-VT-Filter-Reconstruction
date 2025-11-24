"""Evaluation helpers."""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def evaluate_ivt_vs_ground_truth(
    df: pd.DataFrame,
    gt_col: Optional[str] = None,
    pred_col: str = "ivt_sample_type",
) -> Dict[str, float]:
    if gt_col is None:
        if "gt_event_type" in df.columns:
            gt_col = "gt_event_type"
        elif "Eye movement type" in df.columns:
            gt_col = "Eye movement type"
        else:
            raise ValueError("No ground-truth event type column found.")

    if pred_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found. Run apply_ivt_classifier first.")

    mask = df[gt_col].isin(["Fixation", "Saccade"])
    gt = df.loc[mask, gt_col].astype(str)
    pred = df.loc[mask, pred_col].astype(str)

    total = len(gt)
    if total == 0:
        raise ValueError("No samples with ground truth Fixation/Saccade found.")

    agree_mask = gt == pred
    agree = int(agree_mask.sum())
    agreement = agree / total

    tp_fix = int(((gt == "Fixation") & (pred == "Fixation")).sum())
    fn_fix = int(((gt == "Fixation") & (pred == "Saccade")).sum())
    tp_sac = int(((gt == "Saccade") & (pred == "Saccade")).sum())
    fn_sac = int(((gt == "Saccade") & (pred == "Fixation")).sum())

    n_fix = int((gt == "Fixation").sum())
    n_sac = int((gt == "Saccade").sum())

    recall_fix = tp_fix / n_fix if n_fix > 0 else float("nan")
    recall_sac = tp_sac / n_sac if n_sac > 0 else float("nan")

    labels = sorted(set(gt) | set(pred))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)

    conf = [[0] * k for _ in range(k)]
    for g_val, p_val in zip(gt, pred):
        i = label_to_idx[g_val]
        j = label_to_idx[p_val]
        conf[i][j] += 1

    po = sum(conf[i][i] for i in range(k)) / total
    row_marg = [sum(conf[i][j] for j in range(k)) for i in range(k)]
    col_marg = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(k)) / (total * total)
    kappa = (po - pe) / (1.0 - pe) if 1.0 - pe != 0.0 else float("nan")

    stats = {
        "n_samples_gt_fix_or_sac": total,
        "n_fix_in_gt": n_fix,
        "n_sac_in_gt": n_sac,
        "n_agree": agree,
        "percentage_agreement": agreement * 100.0,
        "fixation_recall": recall_fix * 100.0,
        "saccade_recall": recall_sac * 100.0,
        "cohen_kappa": kappa,
    }

    print("=== I-VT classifier evaluation vs. ground truth ===")
    print(f"Total samples with GT Fixation/Saccade: {total}")
    print(f"  Fixation in GT: {n_fix}")
    print(f"  Saccade in GT:  {n_sac}")
    print()
    print(f"Agreement (sample-level): {agree} / {total} = {agreement*100:.2f}%")
    print()
    print("Confusion (rows = GT, cols = Pred):")
    print("             Pred: Fixation   Pred: Saccade")
    print(f"GT Fixation   {tp_fix:7d}        {fn_fix:7d}")
    print(f"GT Saccade    {fn_sac:7d}        {tp_sac:7d}")
    print()
    print(f"Fixation recall: {recall_fix*100:.2f}%")
    print(f"Saccade recall:  {recall_sac*100:.2f}%")
    print()
    print(f"Cohen's kappa:   {kappa:.3f}")
    print("==============================================")

    return stats
