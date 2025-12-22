# ivt_filter/evaluation.py
from __future__ import annotations

from typing import Optional, List, Dict, Any
import math
import pandas as pd


def compute_ivt_metrics(
    df: pd.DataFrame,
    gt_col: Optional[str] = None,
    pred_col: str = "ivt_sample_type",
) -> Dict[str, Any]:
    """
    Metriken zwischen Ground Truth und IVT Vorhersage berechnen,
    ohne etwas zu drucken.
    """

    if gt_col is None:
        # Prefer sample-level GT (expanded from events)
        if "gt_sample_type" in df.columns:
            gt_col = "gt_sample_type"
        elif "gt_event_type" in df.columns:
            gt_col = "gt_event_type"
        elif "Eye movement type" in df.columns:
            gt_col = "Eye movement type"
        else:
            raise ValueError("No ground-truth event type column found.")

    if pred_col not in df.columns:
        raise ValueError(f"Prediction column '{pred_col}' not found. Run apply_ivt_classifier first.")

    # Alle Samples verwenden (nicht nur GT Fixation/Saccade)
    gt = df[gt_col].astype(str)
    pred = df[pred_col].astype(str)

    total_all = len(gt)
    
    # Für traditionelle Metriken: nur GT Fixation/Saccade berücksichtigen
    mask_traditional = gt.isin(["Fixation", "Saccade"])
    gt_trad = gt.loc[mask_traditional]
    pred_trad = pred.loc[mask_traditional]

    total_traditional = len(gt_trad)
    if total_traditional == 0:
        raise ValueError("No samples with ground truth Fixation/Saccade found.")

    # Basis-Agreement für traditionelle Metriken (nur GT Fix/Sac)
    agree_mask = gt_trad == pred_trad
    agree = int(agree_mask.sum())
    agreement = agree / total_traditional
    
    # TOTAL AGREEMENT: alle Samples und alle Klassifikationstypen
    agree_all = int((gt == pred).sum())
    agreement_all = agree_all / total_all if total_all > 0 else float("nan")

    # True Positives / False Negatives für Fixation/Saccade
    tp_fix = int(((gt_trad == "Fixation") & (pred_trad == "Fixation")).sum())
    fn_fix = int(((gt_trad == "Fixation") & (pred_trad == "Saccade")).sum())
    tp_sac = int(((gt_trad == "Saccade") & (pred_trad == "Saccade")).sum())
    fn_sac = int(((gt_trad == "Saccade") & (pred_trad == "Fixation")).sum())

    n_fix = int((gt_trad == "Fixation").sum())
    n_sac = int((gt_trad == "Saccade").sum())

    recall_fix = tp_fix / n_fix if n_fix > 0 else float("nan")
    recall_sac = tp_sac / n_sac if n_sac > 0 else float("nan")

    # Volle Konfusionsmatrix über alle Labels und alle Samples
    labels: List[str] = sorted(set(gt) | set(pred))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)

    conf: List[List[int]] = [[0] * k for _ in range(k)]
    for g_val, p_val in zip(gt, pred):
        i = label_to_idx[g_val]
        j = label_to_idx[p_val]
        conf[i][j] += 1

    # Cohen's kappa (über alle Samples)
    po = sum(conf[i][i] for i in range(k)) / total_all
    row_marg = [sum(conf[i][j] for j in range(k)) for i in range(k)]
    col_marg = [sum(conf[i][j] for i in range(k)) for j in range(k)]
    pe = sum(row_marg[i] * col_marg[i] for i in range(k)) / (total_all * total_all)

    if 1.0 - pe != 0.0:
        kappa = (po - pe) / (1.0 - pe)
    else:
        kappa = float("nan")

    # Statistiken für Unclassified und EyesNotFound getrennt
    pred_is_uncl = pred == "Unclassified"
    pred_is_eynf = pred == "EyesNotFound"
    gt_is_uncl = gt == "Unclassified"
    gt_is_eynf = gt == "EyesNotFound"

    n_pred_uncl = int(pred_is_uncl.sum())
    n_pred_eynf = int(pred_is_eynf.sum())
    n_gt_uncl = int(gt_is_uncl.sum())
    n_gt_eynf = int(gt_is_eynf.sum())

    correct_uncl = int((gt_is_uncl & pred_is_uncl).sum())
    correct_eynf = int((gt_is_eynf & pred_is_eynf).sum())
    accuracy_uncl = correct_uncl / n_gt_uncl if n_gt_uncl > 0 else float("nan")
    accuracy_eynf = correct_eynf / n_gt_eynf if n_gt_eynf > 0 else float("nan")

    # Recall für Unclassified und EyesNotFound (wie viele GT wurden korrekt erkannt)
    recall_uncl = correct_uncl / n_gt_uncl if n_gt_uncl > 0 else float("nan")
    recall_eynf = correct_eynf / n_gt_eynf if n_gt_eynf > 0 else float("nan")

    # Traditionelle Counts für GT Fix/Sac → Pred Uncl/EyeNF
    n_fix_to_uncl = int(((gt_trad == "Fixation") & pred_is_uncl.loc[mask_traditional]).sum())
    n_sac_to_uncl = int(((gt_trad == "Saccade") & pred_is_uncl.loc[mask_traditional]).sum())
    n_fix_to_eynf = int(((gt_trad == "Fixation") & pred_is_eynf.loc[mask_traditional]).sum())
    n_sac_to_eynf = int(((gt_trad == "Saccade") & pred_is_eynf.loc[mask_traditional]).sum())

    return {
        "n_samples_total": total_all,
        "n_samples_gt_fix_or_sac": total_traditional,
        "n_fix_in_gt": n_fix,
        "n_sac_in_gt": n_sac,
        "n_gt_uncl": n_gt_uncl,
        "n_pred_uncl": n_pred_uncl,
        "n_correct_uncl": correct_uncl,
        "accuracy_uncl": accuracy_uncl * 100.0 if not math.isnan(accuracy_uncl) else float("nan"),
        "recall_uncl": recall_uncl * 100.0 if not math.isnan(recall_uncl) else float("nan"),
        "n_gt_eynf": n_gt_eynf,
        "n_pred_eynf": n_pred_eynf,
        "n_correct_eynf": correct_eynf,
        "accuracy_eynf": accuracy_eynf * 100.0 if not math.isnan(accuracy_eynf) else float("nan"),
        "recall_eynf": recall_eynf * 100.0 if not math.isnan(recall_eynf) else float("nan"),
        "n_agree": agree,
        "percentage_agreement": agreement * 100.0,
        "n_agree_all": agree_all,
        "percentage_agreement_all": agreement_all * 100.0 if not math.isnan(agreement_all) else float("nan"),
        "fixation_recall": recall_fix * 100.0,
        "saccade_recall": recall_sac * 100.0,
        "cohen_kappa": kappa,
        "labels": labels,
        "confusion_matrix": conf,
        "tp_fix": tp_fix,
        "fn_fix": fn_fix,
        "tp_sac": tp_sac,
        "fn_sac": fn_sac,
        # GT Fix/Sac → Pred Unclassified/EyesNotFound
        "n_fix_to_uncl": n_fix_to_uncl,
        "n_sac_to_uncl": n_sac_to_uncl,
        "n_fix_to_eynf": n_fix_to_eynf,
        "n_sac_to_eynf": n_sac_to_eynf,
    }


def compute_event_agreement(
    df: pd.DataFrame,
    event_start_col: str = "event_start_idx",
    event_end_col: str = "event_end_idx",
    event_type_col: str = "ivt_event_type",
    gt_event_type_col: str = "gt_event_type",
) -> Dict[str, Any]:
    """
    Event-basiertes Agreement berechnen.
    
    Vergleicht Events (nicht Samples): Stimmen GT und Prediction für den gleichen Event überein?
    Unterstützt ALLE Klassifikationen: Fixation, Saccade, Unclassified, EyesNotFound
    """
    
    # Events extrahieren - verwende event_type wenn vorhanden, sonst nutze ivt_event_type
    if "event_type" in df.columns:
        # Event-basiert Daten (von events expandiert)
        events = df.drop_duplicates(subset=["event_start_idx"], keep="first").copy()
    else:
        # Sample-basierte Daten - Events müssen rekonstruiert werden
        events = []
        current_event = None
        
        for idx, row in df.iterrows():
            pred_type = row.get(event_type_col, row.get("ivt_sample_type", None))
            gt_type = row.get(gt_event_type_col, row.get("gt_sample_type", None))
            
            if current_event is None:
                current_event = {
                    "event_start_idx": idx,
                    "pred_type": pred_type,
                    "gt_type": gt_type,
                    "samples": 1,
                }
            elif current_event["pred_type"] == pred_type and current_event["gt_type"] == gt_type:
                current_event["samples"] += 1
            else:
                current_event["event_end_idx"] = idx - 1
                events.append(current_event)
                current_event = {
                    "event_start_idx": idx,
                    "pred_type": pred_type,
                    "gt_type": gt_type,
                    "samples": 1,
                }
        
        if current_event:
            current_event["event_end_idx"] = len(df) - 1
            events.append(current_event)
        
        events = pd.DataFrame(events)
    
    if events.empty:
        return {
            "n_events": 0,
            "event_agreement": float("nan"),
            "agreement_by_type": {},
        }
    
    # Prediction vs GT
    pred_types = events[event_type_col].astype(str) if event_type_col in events.columns else events["pred_type"].astype(str)
    gt_types = events[gt_event_type_col].astype(str) if gt_event_type_col in events.columns else events["gt_type"].astype(str)
    
    # Gesamtes Agreement
    agree_mask = pred_types == gt_types
    total_agreement = int(agree_mask.sum())
    total_events = len(events)
    event_agreement_pct = (total_agreement / total_events * 100.0) if total_events > 0 else float("nan")
    
    # Agreement pro Klassifikationstyp
    agreement_by_type = {}
    all_types = set(gt_types.unique()) | set(pred_types.unique())
    
    for evt_type in sorted(all_types):
        mask = gt_types == evt_type
        if mask.sum() > 0:
            correct = int(((gt_types == evt_type) & (pred_types == evt_type)).sum())
            total = int(mask.sum())
            pct = (correct / total * 100.0) if total > 0 else float("nan")
            agreement_by_type[evt_type] = {
                "n_gt_events": total,
                "n_correct": correct,
                "agreement_pct": pct,
            }
    
    return {
        "n_events": total_events,
        "n_agreement": total_agreement,
        "event_agreement_pct": event_agreement_pct,
        "agreement_by_type": agreement_by_type,
    }


def evaluate_ivt_vs_ground_truth(
    df: pd.DataFrame,
    gt_col: Optional[str] = None,
    pred_col: str = "ivt_sample_type",
) -> Dict[str, Any]:
    """
    Wrapper: berechnet Metriken und druckt einen kurzen Report auf stdout.
    """
    metrics = compute_ivt_metrics(df, gt_col=gt_col, pred_col=pred_col)

    print("=== I-VT classifier evaluation vs. ground truth ===")
    print(f"Total samples: {metrics['n_samples_total']}")
    print(f"Samples with GT Fixation/Saccade: {metrics['n_samples_gt_fix_or_sac']}")
    print(f"  Fixation in GT: {metrics['n_fix_in_gt']}")
    print(f"  Saccade in GT:  {metrics['n_sac_in_gt']}")
    print(f"  Unclassified in GT: {metrics['n_gt_uncl']}")
    print(f"  EyesNotFound in GT: {metrics['n_gt_eynf']}")
    print()
    
    # Event-basierte Metriken berechnen und ausgeben
    event_metrics = compute_event_agreement(df, event_type_col="ivt_event_type", gt_event_type_col="gt_event_type")
    if event_metrics['n_events'] > 0:
        print("=== Event-level Agreement (all event types) ===")
        print(f"Total events: {event_metrics['n_events']}")
        print(f"Agreement: {event_metrics['n_agreement']} / {event_metrics['n_events']} = "
              f"{event_metrics['event_agreement_pct']:.2f}%")
        print()
        print("Agreement by event type:")
        for evt_type in sorted(event_metrics['agreement_by_type'].keys()):
            stats = event_metrics['agreement_by_type'][evt_type]
            print(f"  {evt_type}: {stats['n_correct']}/{stats['n_gt_events']} = {stats['agreement_pct']:.2f}%")
        print()
    
    print(
        "Agreement (sample-level, GT Fix/Sac only): "
        f"{metrics['n_agree']} / {metrics['n_samples_gt_fix_or_sac']} = "
        f"{metrics['percentage_agreement']:.2f}%"
    )
    print()
    
    print(
        "Agreement (sample-level, ALL types): "
        f"{metrics['n_agree_all']} / {metrics['n_samples_total']} = "
        f"{metrics['percentage_agreement_all']:.2f}%"
    )
    print()

    print("Confusion Matrix (rows = GT, columns = Pred):")
    labels = metrics['labels']
    conf = metrics['confusion_matrix']
    # Calculate column width for best alignment
    col_width = max(14, max(len(f"Pred: {lab}") for lab in labels) + 2)
    # Header
    header_cells = ["".ljust(col_width)] + [f"Pred: {lab}".center(col_width) for lab in labels]
    header = "|".join(header_cells) + "|"
    sep = "+".join(["-" * col_width for _ in range(len(labels) + 1)]) + "+"
    print(sep)
    print(header)
    print(sep)
    # Rows
    for i, gt_lab in enumerate(labels):
        row_cells = [f"GT: {gt_lab}".ljust(col_width)]
        for j in range(len(labels)):
            row_cells.append(f"{conf[i][j]:^{col_width}d}")
        row = "|".join(row_cells) + "|"
        print(row)
        print(sep)
    print()
    print(f"Fixation recall: {metrics['fixation_recall']:.2f}%")
    print(f"Saccade recall:  {metrics['saccade_recall']:.2f}%")
    if not math.isnan(metrics['accuracy_uncl']):
        print(f"Unclassified accuracy: {metrics['accuracy_uncl']:.2f}% "
              f"({metrics['n_correct_uncl']}/{metrics['n_gt_uncl']} correctly classified as Unclassified)")
        print(f"Unclassified recall:   {metrics['recall_uncl']:.2f}%")
    if not math.isnan(metrics['accuracy_eynf']):
        print(f"EyesNotFound accuracy: {metrics['accuracy_eynf']:.2f}% "
              f"({metrics['n_correct_eynf']}/{metrics['n_gt_eynf']} correctly classified as EyesNotFound)")
        print(f"EyesNotFound recall:   {metrics['recall_eynf']:.2f}%")
    print()
    print(f"Cohen's kappa (all samples): {metrics['cohen_kappa']:.3f}")
    print("==============================================")

    return metrics
