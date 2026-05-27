# ivt_filter/evaluation/plotting.py
from __future__ import annotations

from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

from ..config import OlsenVelocityConfig


def plot_velocity_only(df: pd.DataFrame, cfg: OlsenVelocityConfig) -> None:
    """
    Zeitverlauf der Winkelgeschwindigkeit plotten.
    """
    mask = df["velocity_deg_per_sec"].notna()
    times = df.loc[mask, "time_ms"]
    vels = df.loc[mask, "velocity_deg_per_sec"]

    plt.figure()
    plt.plot(times, vels)
    plt.xlabel("Time [ms]")
    plt.ylabel("Gaze velocity [deg/s]")
    plt.title(
        f"Olsen-style gaze velocity (window={cfg.window_length_ms} ms, "
        f"eye_mode={cfg.eye_mode}, smoothing={cfg.smoothing_mode})"
    )
    plt.tight_layout()
    plt.show()


def plot_velocity_and_classification(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
    type_col: Optional[str] = None,
) -> None:
    """
    Geschwindigkeit + Ereignis Labels plotten.

    type_col:
      - wenn None: versucht zuerst postprocessed Spalten, dann "ivt_event_type", dann GT
      - Zeigt sowohl Ground Truth als auch IVT-Klassifikation an, falls beide vorhanden
    """
    mask = df["velocity_deg_per_sec"].notna()
    times_vel = df.loc[mask, "time_ms"]
    vels = df.loc[mask, "velocity_deg_per_sec"]

    # Bestimme Klassifikationsspalten
    gt_col = None
    ivt_col = None
    
    if "gt_event_type" in df.columns:
        gt_col = "gt_event_type"
    elif "Eye movement type" in df.columns:
        gt_col = "Eye movement type"
    
    # Versuche zuerst postprocessed Spalten zu nutzen (falls vorhanden)
    if "ivt_event_type_post" in df.columns:
        ivt_col = "ivt_event_type_post"
    elif "ivt_event_type" in df.columns:
        ivt_col = "ivt_event_type"
    elif type_col is not None and type_col in df.columns:
        ivt_col = type_col

    if gt_col is None and ivt_col is None:
        raise ValueError("No event-type column found for plotting.")

    times_evt = df["time_ms"]

    # Erstelle Subplots basierend auf verfügbaren Spalten
    num_class_plots = (1 if gt_col else 0) + (1 if ivt_col else 0)
    if num_class_plots == 0:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
        ax1.plot(times_vel, vels)
        ax1.set_ylabel("Velocity [deg/s]")
        ax1.set_xlabel("Time [ms]")
        ax1.set_title(
            f"Olsen-style velocity "
            f"(window={cfg.window_length_ms} ms, eye_mode={cfg.eye_mode}, "
            f"smoothing={cfg.smoothing_mode})"
        )
        plt.tight_layout()
        plt.show()
        return

    height_ratios = [3] + [1] * num_class_plots
    fig, axes = plt.subplots(
        num_class_plots + 1, 1, sharex=True, figsize=(10, 4 + 2*num_class_plots), 
        gridspec_kw={"height_ratios": height_ratios}
    )
    
    if num_class_plots + 1 == 1:
        axes = [axes]  # falls nur ein Plot
    
    ax1 = axes[0]
    ax1.plot(times_vel, vels)
    ax1.set_ylabel("Velocity [deg/s]")
    ax1.set_title(
        f"Olsen-style velocity + classification "
        f"(window={cfg.window_length_ms} ms, eye_mode={cfg.eye_mode}, "
        f"smoothing={cfg.smoothing_mode})"
    )

    plot_idx = 1
    label_to_codes = {}
    
    # Ground Truth Plot
    if gt_col:
        events_gt = df[gt_col].fillna("Unknown").astype(str)
        unique_labels_gt = list(dict.fromkeys(events_gt))
        label_to_code_gt = {lab: i for i, lab in enumerate(unique_labels_gt)}
        codes_gt = events_gt.map(label_to_code_gt)
        label_to_codes["Ground Truth"] = label_to_code_gt
        
        ax_gt = axes[plot_idx]
        ax_gt.step(times_evt, codes_gt, where="post", label="Ground Truth")
        ax_gt.set_ylabel("GT Class")
        ax_gt.set_yticks(list(label_to_code_gt.values()))
        ax_gt.set_yticklabels(list(label_to_code_gt.keys()))
        ax_gt.grid(True, axis="y", linestyle=":", linewidth=0.5)
        plot_idx += 1
    
    # IVT Classification Plot
    if ivt_col:
        events_ivt = df[ivt_col].fillna("Unknown").astype(str)
        unique_labels_ivt = list(dict.fromkeys(events_ivt))
        label_to_code_ivt = {lab: i for i, lab in enumerate(unique_labels_ivt)}
        codes_ivt = events_ivt.map(label_to_code_ivt)
        label_to_codes["IVT"] = label_to_code_ivt
        
        ax_ivt = axes[plot_idx]
        ax_ivt.step(times_evt, codes_ivt, where="post", label="IVT Classification")
        ax_ivt.set_ylabel("IVT Class")
        ax_ivt.set_yticks(list(label_to_code_ivt.values()))
        ax_ivt.set_yticklabels(list(label_to_code_ivt.keys()))
        ax_ivt.grid(True, axis="y", linestyle=":", linewidth=0.5)
        ax_ivt.set_xlabel("Time [ms]")
    
    plt.tight_layout()
    plt.show()
