# ivt_filter/velocity.py
from __future__ import annotations

from typing import Optional, List
import math
import dataclasses

import numpy as np
import pandas as pd

from .config import OlsenVelocityConfig
from .gaze import prepare_combined_columns, smooth_combined_gaze, gap_fill_gaze

from .windowing import (
    WindowSelector,
    TimeSymmetricWindowSelector,
    SampleSymmetricWindowSelector,
    FixedSampleSymmetricWindowSelector,
)

from .window_rounding import (
    WindowRoundingStrategy,
    StandardWindowRounding,
    SymmetricRoundWindowStrategy,
)

from .coordinate_rounding import (
    CoordinateRoundingStrategy,
    NoRounding,
    RoundToNearest,
    RoundHalfUp,
    FloorRounding,
    CeilRounding,
)

from .velocity_calculation import (
    VelocityCalculationStrategy,
    Olsen2DApproximation,
    Ray3DAngle,
)


def _get_velocity_calculation_strategy(method: str) -> VelocityCalculationStrategy:
    """Factory für Velocity-Berechnungs-Strategien."""
    if method == "olsen2d":
        return Olsen2DApproximation()
    elif method == "ray3d":
        return Ray3DAngle()
    else:
        raise ValueError(f"Unknown velocity calculation method: {method}")


# Legacy function for backward compatibility (used in some places)
def visual_angle_deg(
    x1_mm: float,
    y1_mm: float,
    x2_mm: float,
    y2_mm: float,
    eye_z_mm: Optional[float],
) -> float:
    """
    Visueller Winkel in Grad zwischen zwei Punkten (mm Koordinaten).
    
    Legacy wrapper - nutzt Olsen 2D Approximation.
    """
    dx = float(x2_mm) - float(x1_mm)
    dy = float(y2_mm) - float(y1_mm)
    s_mm = math.hypot(dx, dy)

    if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
        d_mm = 600.0
    else:
        d_mm = float(eye_z_mm)

    theta_rad = math.atan2(s_mm, d_mm)
    return math.degrees(theta_rad)


def make_window_selector(cfg: OlsenVelocityConfig) -> WindowSelector:
    """
    Waehlt die passende Fenster-Strategie basierend auf der Config.
    Prioritaet:
      1) fixed_window_samples (reines Sample-Fenster)
      2) sample_symmetric_window (Zeit + sample-symmetrisch)
      3) reines Zeitfenster
    
    Nutzt WindowRoundingStrategy zur Bestimmung der half_size.
    """
    # Wähle Rounding-Strategie
    rounding_strategy: WindowRoundingStrategy
    if cfg.symmetric_round_window:
        rounding_strategy = SymmetricRoundWindowStrategy()
    else:
        rounding_strategy = StandardWindowRounding()
    
    if cfg.fixed_window_samples is not None:
        n = int(cfg.fixed_window_samples)
        if n < 3:
            raise ValueError("fixed_window_samples must be >= 3.")
        
        half_size = rounding_strategy.calculate_half_size(n)
        return FixedSampleSymmetricWindowSelector(half_size=half_size)

    if cfg.sample_symmetric_window:
        return SampleSymmetricWindowSelector()

    return TimeSymmetricWindowSelector()


def _get_coordinate_rounding_strategy(mode: str) -> CoordinateRoundingStrategy:
    """Factory für Koordinaten-Rounding-Strategien."""
    if mode == "none":
        return NoRounding()
    elif mode == "nearest":
        return RoundToNearest()
    elif mode == "halfup":
        return RoundHalfUp()
    elif mode == "floor":
        return FloorRounding()
    elif mode == "ceil":
        return CeilRounding()
    else:
        raise ValueError(f"Unknown coordinate rounding mode: {mode}")


def compute_olsen_velocity(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
) -> pd.DataFrame:
    """
    Olsen-Style Geschwindigkeit berechnen (mm basierte Gaze Daten).
    """
    df = df.sort_values("time_ms").reset_index(drop=True)

    # NEU: Gap Filling vor der Augen-Kombination
    df = gap_fill_gaze(df, cfg)

    df = prepare_combined_columns(df, cfg)
    df = smooth_combined_gaze(df, cfg)

    df = df.copy()
    df["velocity_deg_per_sec"] = float("nan")
    df["window_width_samples"] = pd.NA  # Neue Spalte für Fensterbreite
    # Debug/Transparenz: Umgebung-Validitäts-Flags
    df["env_has_invalid_above"] = pd.NA
    df["env_has_invalid_below"] = pd.NA
    df["env_rule_triggered"] = pd.NA
    # Debug für Abstand-Regel (Invalid-Gap)
    df["gap_rule_triggered"] = pd.NA
    df["gap_left_invalid_idx"] = pd.NA
    df["gap_right_invalid_idx"] = pd.NA

    times = df["time_ms"].to_numpy()
    cx = df["smoothed_x_mm"].to_numpy()
    cy = df["smoothed_y_mm"].to_numpy()
    cex = df["eye_x_mm"].to_numpy()
    cey = df["eye_y_mm"].to_numpy()
    cz = df["eye_z_mm"].to_numpy()
    eye_mode = getattr(cfg, "eye_mode", "average")
    left_valid = df["left_eye_valid"].to_numpy()
    right_valid = df["right_eye_valid"].to_numpy()
    # combined_valid je nach Modus setzen
    if eye_mode == "left":
        valid = left_valid
    elif eye_mode == "right":
        valid = right_valid
    else:
        valid = df["combined_valid"].to_numpy()
    lx = df["gaze_left_x_mm"].to_numpy()
    ly = df["gaze_left_y_mm"].to_numpy()
    rx = df["gaze_right_x_mm"].to_numpy()
    ry = df["gaze_right_y_mm"].to_numpy()

    n = len(df)
    half_window = cfg.window_length_ms / 2.0
    selector = make_window_selector(cfg)

    # Sampling-Analyse (median dt, effektive Hz)
    dt_med = None
    if n >= 2:
        if cfg.sampling_rate_method == "first_100":
            # Tobii-Paper Methode: nur erste 100 Samples verwenden
            n_samples_for_rate = min(100, n - 1)  # max 100, aber nicht mehr als verfügbar
            dt = np.diff(times[:n_samples_for_rate + 1])
            method_desc = f"first {n_samples_for_rate} samples"
        else:
            # Standard: alle Samples verwenden
            dt = np.diff(times)
            method_desc = "all samples"
        
        dt = dt[np.isfinite(dt)]
        if dt.size > 0:
            if cfg.dt_calculation_method == "median":
                dt_med = float(np.median(dt))
            else:  # mean
                dt_med = float(np.mean(dt))
            hz_measured = 1000.0 / dt_med if dt_med > 0 else float("nan")
            method_name = "median" if cfg.dt_calculation_method == "median" else "mean"
            print(f"[Sampling] {method_name} dt = {dt_med:.3f} ms -> measured ~{hz_measured:.1f} Hz (using {method_desc})")
            if math.isfinite(hz_measured):
                nominal_candidates = [30.0, 50.0, 60.0, 120.0, 150.0, 250.0, 300.0, 500.0, 600.0, 1000.0]
                nearest_nom = min(nominal_candidates, key=lambda f: abs(f - hz_measured))
                print(f"[Sampling] nearest nominal rate: {nearest_nom:.1f} Hz")

            # Auto-Konvertierung ms -> Sample-Fenster, falls gewuenscht
            # ODER wenn symmetric_round_window gesetzt ist (impliziert Sample-Fenster)
            should_auto_convert = (cfg.auto_fixed_window_from_ms or cfg.symmetric_round_window)
            if should_auto_convert and cfg.fixed_window_samples is None and dt_med > 0:
                n_intervals = max(1, int(round(cfg.window_length_ms / dt_med)))
                n_samples = n_intervals + 1
                if n_samples < 3:
                    n_samples = 3
                if n_samples % 2 == 0:
                    n_samples += 1
                effective_ms = (n_samples - 1) * dt_med
                per_side = (n_samples - 1) / 2.0
                cfg = dataclasses.replace(cfg, fixed_window_samples=n_samples)
                print(
                    f"[Window] auto sample window: {n_samples} samples total "
                    f"(~{per_side:.1f} pro Seite um das Zentrum, "
                    f"effektive Spannweite ~{effective_ms:.2f} ms)"
                )

    half_window = cfg.window_length_ms / 2.0
    selector = make_window_selector(cfg)
    print("[DEBUG] Window selector:", type(selector).__name__)
    
    # Koordinaten-Rounding-Strategie
    coord_rounding = _get_coordinate_rounding_strategy(cfg.coordinate_rounding)
    if cfg.coordinate_rounding != "none":
        print(f"[Rounding] Coordinate rounding enabled: {coord_rounding.get_description()}")
    
    # Velocity-Calculation-Strategie
    velocity_strategy = _get_velocity_calculation_strategy(cfg.velocity_method)
    if cfg.velocity_method != "olsen2d":
        print(f"[Velocity] Using calculation method: {cfg.velocity_method}")

    # Fensterbreite in Samples berechnen für Transparenz
    if isinstance(selector, FixedSampleSymmetricWindowSelector):
        # Bei FixedSample: Fensterbreite ist bekannt
        fixed_samples = cfg.fixed_window_samples
        print(f"[Window] Fixed sample window: {fixed_samples} samples")
    else:
        # Bei Zeit-basierten Fenstern: Schätzung basierend auf dt_med
        if dt_med is not None and dt_med > 0:
            estimated_samples = int(round(cfg.window_length_ms / dt_med)) + 1
            print(f"[Window] Time-based window (~{cfg.window_length_ms} ms): estimated ~{estimated_samples} samples (based on dt={dt_med:.2f} ms)")
        else:
            print(f"[Window] Time-based window (~{cfg.window_length_ms} ms): sample count varies per location")

    # Fallback: wenn der Selector sample-symmetrisch ist und kein gueltiges Fenster findet,
    # kann auf ein Zeitfenster zurueckgefallen werden.
    fallback_selector: Optional[WindowSelector] = None
    if isinstance(selector, (SampleSymmetricWindowSelector, FixedSampleSymmetricWindowSelector)):
        fallback_selector = TimeSymmetricWindowSelector()

    # Abstand-basierte Unclassified-Regel vorbereiten
    # max_gap_samples = ursprüngliche Fensterbreite - 1 (OHNE Asymmetrie-Aufrundung)
    # Beispiel: window=7 -> gap=6, auch wenn Velocity-Fenster auf 8 aufgerundet wird
    gap_max: Optional[int] = None
    if isinstance(selector, FixedSampleSymmetricWindowSelector) and cfg.fixed_window_samples is not None:
        # Ursprüngliche Fenstergröße ohne Asymmetrie-Anpassung
        original_window_size = int(cfg.fixed_window_samples)
        gap_max = max(0, original_window_size - 1)
    elif dt_med is not None and dt_med > 0:
        # Ursprüngliche Zeit-basierte Schätzung ohne Asymmetrie-Anpassung
        original_est = int(round(cfg.window_length_ms / dt_med)) + 1
        gap_max = max(0, original_est - 1)
    if gap_max is not None:
        print(f"[Rule] Gap-based Unclassified enabled: max_gap_samples={gap_max}")

    # Precompute nächster ungültiger Index links/rechts für jedes Sample
    invalid = ~valid
    prev_invalid_idx = np.full(n, -1, dtype=int)
    next_invalid_idx = np.full(n, -1, dtype=int)
    last_inv = -1
    for ii in range(n):
        if not bool(valid[ii]):
            last_inv = ii
        prev_invalid_idx[ii] = last_inv
    next_inv = -1
    for ii in range(n - 1, -1, -1):
        if not bool(valid[ii]):
            next_inv = ii
        next_invalid_idx[ii] = next_inv

    def impute_avg_with_neighbor(idx: int, first_idx: int, last_idx: int):
        lv = left_valid[idx]
        rv = right_valid[idx]
        if (lv and rv) or (not lv and not rv):
            return cx[idx], cy[idx]

        missing_eye = "right" if lv and not rv else "left"
        t_idx = float(times[idx])
        best_j: Optional[int] = None
        best_dt: Optional[float] = None

        for j in range(first_idx, last_idx + 1):
            if j == idx:
                continue
            if missing_eye == "right":
                if not right_valid[j]:
                    continue
            else:
                if not left_valid[j]:
                    continue

            dt_j = abs(float(times[j]) - t_idx)
            if best_dt is None or dt_j < best_dt:
                best_dt = dt_j
                best_j = j

        if best_j is None:
            return cx[idx], cy[idx]

        if missing_eye == "right":
            lx_idx, ly_idx = lx[idx], ly[idx]
            rx_idx, ry_idx = rx[best_j], ry[best_j]
        else:
            lx_idx, ly_idx = lx[best_j], ly[best_j]
            rx_idx, ry_idx = rx[idx], ry[idx]

        if any(pd.isna(v) for v in (lx_idx, ly_idx, rx_idx, ry_idx)):
            return cx[idx], cy[idx]

        x = (float(lx_idx) + float(rx_idx)) / 2.0
        y = (float(ly_idx) + float(ry_idx)) / 2.0
        return x, y

    for i in range(n):
        # Explizit prüfen: nur wenn das Sample selbst gültig ist, berechne Velocity
        if not bool(valid[i]):
            continue

        # 1. Versuch: aktueller Selector (z.B. SampleSymmetric oder FixedSample)
        first_idx, last_idx = selector.select(i, times, valid, half_window)

        # 2. Fallback: wenn kein sinnvolles Fenster gefunden wurde,
        #    versuche noch das klassische Zeitfenster
        if (first_idx is None or last_idx is None or first_idx == last_idx) and fallback_selector is not None:
            first_idx_fb, last_idx_fb = fallback_selector.select(i, times, valid, half_window)
            if first_idx_fb is not None and last_idx_fb is not None and first_idx_fb != last_idx_fb:
                first_idx, last_idx = first_idx_fb, last_idx_fb

        # Wenn immer noch kein Fenster: Velocity bleibt NaN -> Sample bleibt Unclassified
        if first_idx is None or last_idx is None or first_idx == last_idx:
            continue

        # Neue Regel (Abstand): Liegt das Sample zwischen zwei invaliden Samples,
        # deren Abstand (exklusive) <= gap_max ist? Dann Unclassified.
        if gap_max is not None and gap_max >= 0:
            L = int(prev_invalid_idx[i])
            R = int(next_invalid_idx[i])
            triggered = False
            if L != -1 and R != -1 and L < i < R:
                gap = R - L - 1
                if gap <= gap_max:
                    triggered = True
            df.at[i, "gap_rule_triggered"] = bool(triggered)
            df.at[i, "gap_left_invalid_idx"] = None if L == -1 else int(L)
            df.at[i, "gap_right_invalid_idx"] = None if R == -1 else int(R)
            # Für Rückwärtskompatibilität die alten env-Flags auf False setzen
            df.at[i, "env_has_invalid_above"] = False
            df.at[i, "env_has_invalid_below"] = False
            df.at[i, "env_rule_triggered"] = False
            if triggered:
                continue

        # Fallback: nächstes gültiges Sample verwenden, falls first/last ungültig
        if cfg.use_fallback_valid_samples:
            if not valid[first_idx]:
                # Finde nächstes gültiges Sample nach first_idx
                for j in range(first_idx + 1, last_idx + 1):
                    if valid[j]:
                        first_idx = j
                        break
                else:
                    continue  # Kein gültiges Sample gefunden
            
            if not valid[last_idx]:
                # Finde nächstes gültiges Sample vor last_idx
                for j in range(last_idx - 1, first_idx - 1, -1):
                    if valid[j]:
                        last_idx = j
                        break
                else:
                    continue  # Kein gültiges Sample gefunden

        t_first = float(times[first_idx])
        t_last = float(times[last_idx])
        dt_ms = t_last - t_first
        if dt_ms < cfg.min_dt_ms:
            continue

        x1, y1 = cx[first_idx], cy[first_idx]
        x2, y2 = cx[last_idx], cy[last_idx]

        # Strategien nur im average Modus
        if cfg.eye_mode == "average":
            use_single_eye = cfg.average_window_single_eye
            use_neighbor = cfg.average_window_impute_neighbor

            if use_single_eye or use_neighbor:
                window_lv = left_valid[first_idx:last_idx + 1]
                window_rv = right_valid[first_idx:last_idx + 1]
                both_valid = window_lv & window_rv
                single_valid = window_lv ^ window_rv

                if both_valid.any() and single_valid.any():
                    if use_neighbor:
                        x1, y1 = impute_avg_with_neighbor(first_idx, first_idx, last_idx)
                        x2, y2 = impute_avg_with_neighbor(last_idx, first_idx, last_idx)
                    elif use_single_eye:
                        candidates: List[tuple[str, int]] = []
                        if left_valid[first_idx] and left_valid[last_idx]:
                            candidates.append(("left", int(window_lv.sum())))
                        if right_valid[first_idx] and right_valid[last_idx]:
                            candidates.append(("right", int(window_rv.sum())))
                        if candidates:
                            candidates.sort(key=lambda t: t[1], reverse=True)
                            chosen_eye = candidates[0][0]
                            if chosen_eye == "left":
                                x1, y1 = lx[first_idx], ly[first_idx]
                                x2, y2 = lx[last_idx], ly[last_idx]
                            else:
                                x1, y1 = rx[first_idx], ry[first_idx]
                                x2, y2 = rx[last_idx], ry[last_idx]

        if any(pd.isna(v) for v in (x1, y1, x2, y2)):
            continue

        # Koordinaten-Rounding anwenden (optional) - nutze die bereits erstellte Strategie
        x1, y1 = coord_rounding.round_gaze(x1, y1)
        x2, y2 = coord_rounding.round_gaze(x2, y2)
        
        # Eye Position extrahieren und runden
        eye_x = cex[i] if i < len(cex) else None
        eye_y = cey[i] if i < len(cey) else None
        eye_z = cz[i] if i < len(cz) else None
        if eye_x is not None and eye_y is not None and eye_z is not None:
            eye_x, eye_y, eye_z = coord_rounding.round_eye(eye_x, eye_y, eye_z)
        
        # Velocity-Calculation-Strategie verwenden
        angle_deg = velocity_strategy.calculate_visual_angle(
            x1, y1, x2, y2, eye_x, eye_y, eye_z
        )
        dt_s = dt_ms / 1000.0
        velocity = angle_deg / dt_s if dt_s > 0 else float("nan")
        df.at[i, "velocity_deg_per_sec"] = velocity
        
        # Speichere die tatsächliche Fensterbreite (in Samples)
        final_window_size = last_idx - first_idx + 1
        df.at[i, "window_width_samples"] = final_window_size

    # Transparenz: Fenster-Statistik ausgeben
    computed = (~df["velocity_deg_per_sec"].isna()).sum()
    if computed > 0:
        print(f"[Velocity] Computed velocity for {computed}/{n} samples")
        print(f"[Velocity] Window configuration: window_length_ms={cfg.window_length_ms}, selector={type(selector).__name__}")

    return df


def compute_olsen_velocity_from_slim_tsv(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Optional[OlsenVelocityConfig] = None,
) -> pd.DataFrame:

    from .io import read_tsv, write_tsv

    if cfg is None:
        cfg = OlsenVelocityConfig()

    df = read_tsv(input_path)
    df = compute_olsen_velocity(df, cfg)

    if output_path is not None:
        write_tsv(df, output_path)

    return df
