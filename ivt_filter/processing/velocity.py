# ivt_filter/processing/velocity.py
from __future__ import annotations

from typing import Optional, List
from decimal import Decimal, ROUND_HALF_UP
import logging
import math
import dataclasses

import numpy as np
import pandas as pd

from ..config import OlsenVelocityConfig, PhysicalConstants
from ..preprocessing import prepare_combined_columns, smooth_combined_gaze, gap_fill_gaze

from ..strategies import (
    WindowSelector,
    TimeSymmetricWindowSelector,
    SampleSymmetricWindowSelector,
    FixedSampleSymmetricWindowSelector,
    AsymmetricNeighborWindowSelector,
    ShiftedValidWindowSelector,
    TimeBasedShiftedValidWindowSelector,
    WindowRoundingStrategy,
    CoordinateRoundingStrategy,
    VelocityCalculationStrategy,
    VelocityContext,
    Olsen2DApproximation,
    Ray3DAngle,
    Ray3DGazeDir,
)
from ..strategies.window_rounding import StandardWindowRounding, SymmetricRoundWindowStrategy
from ..strategies.coordinate_rounding import (
    NoRounding,
    RoundToNearest,
    RoundHalfUp,
    FloorRounding,
    CeilRounding,
)


logger = logging.getLogger(__name__)


def _get_velocity_calculation_strategy(method: str) -> VelocityCalculationStrategy:
    """Factory for velocity calculation strategies."""
    if method == "olsen2d":
        return Olsen2DApproximation()
    elif method == "ray3d":
        return Ray3DAngle()
    elif method == "ray3d_gaze_dir":
        return Ray3DGazeDir()
    else:
        raise ValueError(f"Unknown velocity calculation method: {method}")


def visual_angle_deg(
    x1_mm: float,
    y1_mm: float,
    x2_mm: float,
    y2_mm: float,
    eye_z_mm: Optional[float],
) -> float:
    """Calculate visual angle between two points.
    
    Legacy wrapper using Olsen 2D approximation.
    """
    dx = float(x2_mm) - float(x1_mm)
    dy = float(y2_mm) - float(y1_mm)
    s_mm = math.hypot(dx, dy)

    if eye_z_mm is None or not math.isfinite(eye_z_mm) or eye_z_mm <= 0:
        d_mm = PhysicalConstants.DEFAULT_EYE_SCREEN_DISTANCE_MM
    else:
        d_mm = float(eye_z_mm)

    theta_rad = math.atan2(s_mm, d_mm)
    return math.degrees(theta_rad)


def make_window_selector(cfg: OlsenVelocityConfig) -> WindowSelector:
    """
    Waehlt die passende Fenster-Strategie basierend auf der Config.
    Prioritaet:
      0) asymmetric_neighbor_window (2-Sample asymmetrisch, Backward/Forward)
      1) fixed_window_samples (reines Sample-Fenster)
      2) sample_symmetric_window (Zeit + sample-symmetrisch)
      3) reines Zeitfenster
    
    Nutzt WindowRoundingStrategy zur Bestimmung der half_size.
    """
    # Asymmetrisches Nachbar-Fenster (2 Samples)
    if cfg.asymmetric_neighbor_window:
        return AsymmetricNeighborWindowSelector()
    
    # Select strategy
    rounding_strategy: WindowRoundingStrategy
    if cfg.symmetric_round_window:
        rounding_strategy = SymmetricRoundWindowStrategy()
    else:
        rounding_strategy = StandardWindowRounding()
    
    if cfg.shifted_valid_window:
        if cfg.fixed_window_samples is not None:
            # Sample-based shifted valid window
            n = int(cfg.fixed_window_samples)
            if n < 3:
                raise ValueError("fixed_window_samples must be >= 3 for shifted_valid_window.")
            half_size = rounding_strategy.calculate_half_size(n)
            return ShiftedValidWindowSelector(half_size=half_size, fallback_mode=cfg.shifted_valid_fallback)
        else:
            # Time-based shifted valid window
            return TimeBasedShiftedValidWindowSelector(fallback_mode=cfg.shifted_valid_fallback)

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


def _calculate_dt_ms(
    first_idx: int,
    last_idx: int,
    times: np.ndarray,
    selector: WindowSelector,
    hz_measured: Optional[float],
    use_fixed_dt: bool = False,
) -> float:
    """
    Berechne Zeitdifferenz dt_ms basierend auf Fenster-Selector-Typ.
    
    - AsymmetricNeighbor: nutze fixed dt aus Sampling-Rate
    - FixedSampleSymmetric: nutze nominalen dt (ohne Timestamp-Jitter)
    - ShiftedValid: nutze tatsächliche Zeitstempel-Differenzen
    - Andere: normale dt-Berechnung aus time_ms
    """
    if use_fixed_dt and isinstance(selector, AsymmetricNeighborWindowSelector):
        if hz_measured is not None and hz_measured > 0:
            return 1000.0 / hz_measured
        else:
            return float(times[last_idx]) - float(times[first_idx])
    
    if isinstance(selector, FixedSampleSymmetricWindowSelector):
        if hz_measured is not None and hz_measured > 0:
            window_size = last_idx - first_idx + 1
            window_spans = window_size - 1
            return window_spans * (1000.0 / hz_measured)
        else:
            return float(times[last_idx]) - float(times[first_idx])
    
    if isinstance(selector, ShiftedValidWindowSelector):
        return float(times[last_idx]) - float(times[first_idx])
    
    # Default: normale dt-Berechnung
    return float(times[last_idx]) - float(times[first_idx])


def _get_direction_vectors(
    first_idx: int,
    last_idx: int,
    eye_mode: str,
    used_eye: str,
    eye_consistent_override: bool,
    ldx: np.ndarray,
    ldy: np.ndarray,
    ldz: np.ndarray,
    rdx: np.ndarray,
    rdy: np.ndarray,
    rdz: np.ndarray,
    combined_dir_x: np.ndarray,
    combined_dir_y: np.ndarray,
    combined_dir_z: np.ndarray,
) -> tuple:
    """
    Extrahiere Richtungsvektoren für beide Endpunkte.
    
    Rückgabe: (dir_first_tuple, dir_last_tuple)
    """
    if eye_consistent_override and used_eye == "left":
        dir_first = (ldx[first_idx], ldy[first_idx], ldz[first_idx])
        dir_last = (ldx[last_idx], ldy[last_idx], ldz[last_idx])
    elif eye_consistent_override and used_eye == "right":
        dir_first = (rdx[first_idx], rdy[first_idx], rdz[first_idx])
        dir_last = (rdx[last_idx], rdy[last_idx], rdz[last_idx])
    elif eye_mode == "left":
        dir_first = (ldx[first_idx], ldy[first_idx], ldz[first_idx])
        dir_last = (ldx[last_idx], ldy[last_idx], ldz[last_idx])
    elif eye_mode == "right":
        dir_first = (rdx[first_idx], rdy[first_idx], rdz[first_idx])
        dir_last = (rdx[last_idx], rdy[last_idx], rdz[last_idx])
    else:
        # average mode
        dir_first = (
            combined_dir_x[first_idx],
            combined_dir_y[first_idx],
            combined_dir_z[first_idx],
        )
        dir_last = (
            combined_dir_x[last_idx],
            combined_dir_y[last_idx],
            combined_dir_z[last_idx],
        )
    
    return dir_first, dir_last


def _apply_eye_consistent_override(
    velocity_strategy: VelocityCalculationStrategy,
    eye_mode: str,
    first_idx: int,
    last_idx: int,
    times: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    rx: np.ndarray,
    ry: np.ndarray,
) -> tuple:
    """
    Wende Eye-Consistent-Override für Ray3DGazeDir an (3-sample window).
    
    Wählt ein Auge, das an beiden Endpunkten gültig ist, oder überspringt Velocity.
    
    Rückgabe: (x1, y1, x2, y2, used_eye, override_applied, should_skip)
    - should_skip: True wenn keine konsistente Augen-Wahl möglich
    """
    if not (
        isinstance(velocity_strategy, Ray3DGazeDir)
        and eye_mode == "average"
        and (last_idx - first_idx) == 2
    ):
        return None, None, None, None, eye_mode, False, False
    
    left_both_valid = bool(left_valid[first_idx]) and bool(left_valid[last_idx])
    right_both_valid = bool(right_valid[first_idx]) and bool(right_valid[last_idx])
    
    if left_both_valid:
        x1, y1 = lx[first_idx], ly[first_idx]
        x2, y2 = lx[last_idx], ly[last_idx]
        return x1, y1, x2, y2, "left", True, False
    elif right_both_valid:
        x1, y1 = rx[first_idx], ry[first_idx]
        x2, y2 = rx[last_idx], ry[last_idx]
        return x1, y1, x2, y2, "right", True, False
    else:
        # Kein einzelnes Auge an beiden Endpunkten gültig
        return None, None, None, None, eye_mode, False, True


def compute_olsen_velocity(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
) -> pd.DataFrame:
    """
    Olsen-Style Geschwindigkeit berechnen (mm basierte Gaze Daten).
    """
    df = df.copy()

    # Normalize time column to milliseconds (allows micro/nano inputs)
    time_col = getattr(cfg, "time_column", "time_ms")
    time_unit = getattr(cfg, "time_unit", "ms")
    if time_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{time_col}' column")

    unit_divisor = {"ms": 1.0, "us": 1000.0, "ns": 1_000_000.0}.get(time_unit, 1.0)
    df["time_ms"] = df[time_col].astype(float) / unit_divisor

    df = df.sort_values("time_ms").reset_index(drop=True)

    # NEU: Gap Filling vor der Augen-Kombination
    df = gap_fill_gaze(df, cfg)

    df = prepare_combined_columns(df, cfg)
    df = smooth_combined_gaze(df, cfg)

    df = df.copy()
    df["velocity_deg_per_sec"] = float("nan")
    df["dt_ms"] = float("nan")  # Time difference used for velocity calculation
    df["window_width_samples"] = pd.NA  # New column for Fensterbreite
    df["window_any_invalid"] = False  # Track if any eye invalid in the used window
    df["velocity_eye_used"] = "average"  # Diagnostic: which eye contributed to velocity
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

    # Optional: Blickrichtungs-Vektoren (DACS norm)
    ldx = df.get("gaze_dir_left_x", pd.Series([np.nan] * len(df))).to_numpy()
    ldy = df.get("gaze_dir_left_y", pd.Series([np.nan] * len(df))).to_numpy()
    ldz = df.get("gaze_dir_left_z", pd.Series([np.nan] * len(df))).to_numpy()
    rdx = df.get("gaze_dir_right_x", pd.Series([np.nan] * len(df))).to_numpy()
    rdy = df.get("gaze_dir_right_y", pd.Series([np.nan] * len(df))).to_numpy()
    rdz = df.get("gaze_dir_right_z", pd.Series([np.nan] * len(df))).to_numpy()

    # Kombiniere Richtungen für Eye-Mode "average": Mittelwert der validen Richtungen, sonst fallback
    combined_dir_x = np.full(len(df), np.nan)
    combined_dir_y = np.full(len(df), np.nan)
    combined_dir_z = np.full(len(df), np.nan)
    for i in range(len(df)):
        lv = bool(left_valid[i])
        rv = bool(right_valid[i])
        if lv and rv:
            combined_dir_x[i] = np.mean([ldx[i], rdx[i]])
            combined_dir_y[i] = np.mean([ldy[i], rdy[i]])
            combined_dir_z[i] = np.mean([ldz[i], rdz[i]])
        elif lv:
            combined_dir_x[i] = ldx[i]
            combined_dir_y[i] = ldy[i]
            combined_dir_z[i] = ldz[i]
        elif rv:
            combined_dir_x[i] = rdx[i]
            combined_dir_y[i] = rdy[i]
            combined_dir_z[i] = rdz[i]

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
    logger.debug("Window selector: %s", type(selector).__name__)
    
    # Koordinaten-Rounding-Strategie
    coord_rounding = _get_coordinate_rounding_strategy(cfg.coordinate_rounding)
    if cfg.coordinate_rounding != "none":
        logger.info("Coordinate rounding enabled: %s", coord_rounding.get_description())
    
    # Velocity-Calculation-Strategie
    velocity_strategy = _get_velocity_calculation_strategy(cfg.velocity_method)
    if cfg.velocity_method != "olsen2d":
        logger.info("Velocity calculation method: %s", cfg.velocity_method)

    # Fensterbreite in Samples berechnen für Transparenz
    if isinstance(selector, FixedSampleSymmetricWindowSelector):
        # Bei FixedSample: Fensterbreite ist bekannt
        fixed_samples = cfg.fixed_window_samples
        logger.info("Fixed sample window: %s samples", fixed_samples)
    elif isinstance(selector, AsymmetricNeighborWindowSelector):
        # Bei Asymmetrischem Nachbar-Fenster: 2 Samples
        logger.info("Asymmetric neighbor window: 2 samples (backward/forward)")
        if cfg.use_fixed_dt and hz_measured is not None:
            logger.info("Using fixed dt: %.4f ms (from %.1f Hz)", 1000.0 / hz_measured, hz_measured)
    else:
        # Bei Zeit-basierten Fenstern: Schätzung basierend auf dt_med
        if dt_med is not None and dt_med > 0:
            estimated_samples = int(round(cfg.window_length_ms / dt_med)) + 1
            logger.info(
                "Time-based window (~%.1f ms): estimated ~%s samples (based on dt=%.2f ms)",
                cfg.window_length_ms,
                estimated_samples,
                dt_med,
            )
        else:
            logger.info(
                "Time-based window (~%.1f ms): sample count varies per location",
                cfg.window_length_ms,
            )

    # Fallback: wenn der Selector sample-symmetrisch ist und kein gueltiges Fenster findet,
    # kann auf ein Zeitfenster zurueckgefallen werden.
    fallback_selector: Optional[WindowSelector] = None
    if isinstance(selector, (SampleSymmetricWindowSelector, FixedSampleSymmetricWindowSelector)):
        fallback_selector = TimeSymmetricWindowSelector()

    # Abstand-basierte Unclassified-Regel vorbereiten
    # max_gap_samples = ursprüngliche Fensterbreite - 1 (OHNE Asymmetrie-Aufrundung)
    # Beispiel: window=7 -> gap=6, auch wenn Velocity-Fenster auf 8 aufgerundet wird
    # Bei AsymmetricNeighborWindowSelector: 2 Samples -> gap=1 (radius=1)
    gap_max: Optional[int] = None
    if isinstance(selector, AsymmetricNeighborWindowSelector):
        # 2-Sample Fenster: radius = 1
        gap_max = 1
    elif isinstance(selector, FixedSampleSymmetricWindowSelector) and cfg.fixed_window_samples is not None:
        # Ursprüngliche Fenstergröße ohne Asymmetrie-Anpassung
        original_window_size = int(cfg.fixed_window_samples)
        gap_max = max(0, original_window_size - 1)
    elif dt_med is not None and dt_med > 0:
        # Ursprüngliche Zeit-basierte Schätzung ohne Asymmetrie-Anpassung
        original_est = int(round(cfg.window_length_ms / dt_med)) + 1
        gap_max = max(0, original_est - 1)
    if gap_max is not None:
        logger.info("Gap-based Unclassified enabled: max_gap_samples=%s", gap_max)

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

        used_eye = eye_mode
        eye_consistent_override = False

        # Diagnose: Validität im aktuell gewählten Fenster
        window_lv = left_valid[first_idx:last_idx + 1]
        window_rv = right_valid[first_idx:last_idx + 1]
        window_any_invalid = (~window_lv | ~window_rv).any()
        df.at[i, "window_any_invalid"] = bool(window_any_invalid)

        # Berechne Zeitdifferenz (nutzt Window-Selector-Typ)
        dt_ms = _calculate_dt_ms(first_idx, last_idx, times, selector, hz_measured, cfg.use_fixed_dt)
        
        # Track the actual endpoints used for velocity and direction lookup
        actual_first_idx = first_idx
        actual_last_idx = last_idx

        # Eye-consistent override for gaze-dir velocities (3-sample symmetric window)
        override_result = _apply_eye_consistent_override(
            velocity_strategy, cfg.eye_mode, first_idx, last_idx, times,
            left_valid, right_valid, lx, ly, rx, ry
        )
        x1, y1, x2, y2, chosen_eye, override_applied, should_skip = override_result
        
        if should_skip:
            # No single eye valid at both endpoints -> velocity missing
            continue
        
        if override_applied:
            eye_consistent_override = True
            used_eye = chosen_eye
            # dt from the chosen endpoints' timestamps (time_ms array already normalized)
            dt_ms = _calculate_dt_ms(actual_first_idx, actual_last_idx, times, selector, hz_measured, cfg.use_fixed_dt)
        else:
            x1, y1 = cx[first_idx], cy[first_idx]
            x2, y2 = cx[last_idx], cy[last_idx]

        # Speichere finalen dt und wende min_dt-Prüfung an (außer beim späteren Single-Eye-Fallback)
        original_dt_ms = dt_ms
        skip_dt_check = cfg.eye_mode == "average" and cfg.average_fallback_single_eye and not eye_consistent_override
        if not skip_dt_check:
            if dt_ms < cfg.min_dt_ms:
                continue

        # Strategien nur im average Modus (ohne Override)
        use_single_eye = False
        use_neighbor = False
        use_fallback_single = False
        if cfg.eye_mode == "average" and not eye_consistent_override:
            use_single_eye = cfg.average_window_single_eye
            use_neighbor = cfg.average_window_impute_neighbor
            use_fallback_single = cfg.average_fallback_single_eye

            # NEW: average_fallback_single_eye - use only valid eyes for velocity when any invalids are present
            if use_fallback_single:
                window_lv = left_valid[first_idx:last_idx + 1]
                window_rv = right_valid[first_idx:last_idx + 1]
                single_valid = window_lv ^ window_rv
                window_any_invalid = (~window_lv | ~window_rv).any()

                # Check middle sample validity too
                mid_idx = i
                mid_left_valid = left_valid[mid_idx]
                mid_right_valid = right_valid[mid_idx]
                mid_mixed = (mid_left_valid and not mid_right_valid) or (not mid_left_valid and mid_right_valid)

                # Trigger fallback whenever the window is not fully valid for both eyes
                if window_any_invalid or single_valid.any() or mid_mixed:
                    # Evaluate both eyes: prefer the one with wider valid span, then more valid samples, then center validity
                    left_count = int(window_lv.sum())
                    right_count = int(window_rv.sum())

                    def _endpoints(valid_arr):
                        first = None
                        last = None
                        for j in range(first_idx, last_idx + 1):
                            if valid_arr[j]:
                                first = j
                                break
                        for j in range(last_idx, first_idx - 1, -1):
                            if valid_arr[j]:
                                last = j
                                break
                        return first, last

                    left_first, left_last = _endpoints(left_valid)
                    right_first, right_last = _endpoints(right_valid)
                    left_span = (left_last - left_first) if left_first is not None and left_last is not None else -1
                    right_span = (right_last - right_first) if right_first is not None and right_last is not None else -1

                    if mid_left_valid and not mid_right_valid:
                        left_count += 1
                    elif mid_right_valid and not mid_left_valid:
                        right_count += 1

                    score_left = (left_span, left_count)
                    score_right = (right_span, right_count)
                    if score_left == score_right:
                        chosen_eye = "left" if mid_left_valid or not mid_right_valid else "right"
                    else:
                        chosen_eye = "left" if score_left > score_right else "right"

                    # Find valid endpoints for chosen eye (with fallback search inside window)
                    if chosen_eye == "left":
                        if left_first is not None:
                            actual_first_idx = left_first
                        if left_last is not None:
                            actual_last_idx = left_last

                        if left_first is None or left_last is None or actual_first_idx >= actual_last_idx:
                            continue

                        x1, y1 = lx[actual_first_idx], ly[actual_first_idx]
                        x2, y2 = lx[actual_last_idx], ly[actual_last_idx]
                        used_eye = "left"

                    else:  # right eye
                        if right_first is not None:
                            actual_first_idx = right_first
                        if right_last is not None:
                            actual_last_idx = right_last

                        if right_first is None or right_last is None or actual_first_idx >= actual_last_idx:
                            continue

                        x1, y1 = rx[actual_first_idx], ry[actual_first_idx]
                        x2, y2 = rx[actual_last_idx], ry[actual_last_idx]
                        used_eye = "right"

                    # Recalculate dt using the chosen eye indices
                    if cfg.use_fixed_dt and isinstance(selector, AsymmetricNeighborWindowSelector):
                        if hz_measured is not None and hz_measured > 0:
                            dt_ms = 1000.0 / hz_measured
                        else:
                            t_first = float(times[actual_first_idx])
                            t_last = float(times[actual_last_idx])
                            dt_ms = t_last - t_first
                    elif isinstance(selector, FixedSampleSymmetricWindowSelector) and hz_measured is not None and hz_measured > 0:
                        window_size = actual_last_idx - actual_first_idx + 1
                        window_spans = window_size - 1
                        dt_ms = window_spans * (1000.0 / hz_measured)
                    elif isinstance(selector, ShiftedValidWindowSelector):
                        # Shifted valid window: use actual timestamps
                        t_first = float(times[actual_first_idx])
                        t_last = float(times[actual_last_idx])
                        dt_ms = t_last - t_first
                    else:
                        t_first = float(times[actual_first_idx])
                        t_last = float(times[actual_last_idx])
                        dt_ms = t_last - t_first

        # Original strategies (only if fallback_single not active)
        elif use_single_eye or use_neighbor:
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
                            used_eye = "left"
                        else:
                            x1, y1 = rx[first_idx], ry[first_idx]
                            x2, y2 = rx[last_idx], ry[last_idx]
                            used_eye = "right"

        if any(pd.isna(v) for v in (x1, y1, x2, y2)):
            continue

        # Optional direction vectors for gaze-dir strategy
        dir_first = dir_last = None
        if isinstance(velocity_strategy, Ray3DGazeDir):
            dir_first, dir_last = _get_direction_vectors(
                actual_first_idx, actual_last_idx, eye_mode, used_eye, eye_consistent_override,
                ldx, ldy, ldz, rdx, rdy, rdz, combined_dir_x, combined_dir_y, combined_dir_z
            )

        # Coordinate rounding (optional)
        x1, y1 = coord_rounding.round_gaze(x1, y1)
        x2, y2 = coord_rounding.round_gaze(x2, y2)
        
        # Eye position rounding
        eye_x = cex[i] if i < len(cex) else None
        eye_y = cey[i] if i < len(cey) else None
        eye_z = cz[i] if i < len(cz) else None
        if eye_x is not None and eye_y is not None and eye_z is not None:
            eye_x, eye_y, eye_z = coord_rounding.round_eye(eye_x, eye_y, eye_z)
        
        ctx = VelocityContext(
            x1_mm=x1,
            y1_mm=y1,
            x2_mm=x2,
            y2_mm=y2,
            eye_x_mm=eye_x,
            eye_y_mm=eye_y,
            eye_z_mm=eye_z,
            dir1=dir_first,
            dir2=dir_last,
        )

        angle_deg = velocity_strategy.calculate_visual_angle_ctx(ctx)
        dt_s = dt_ms / 1000.0
        raw_velocity = angle_deg / dt_s if dt_s > 0 else float("nan")
        # Rundung auf 2 Nachkommastellen mit ROUND_HALF_UP, um Banker's Rounding zu vermeiden
        if not pd.isna(raw_velocity):
            velocity = float(Decimal(raw_velocity).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        else:
            velocity = raw_velocity
        df.at[i, "velocity_deg_per_sec"] = velocity
        df.at[i, "dt_ms"] = dt_ms  # Store the time difference used for velocity
        # Optional: rohen Wert speichern, falls später benötigt
        df.at[i, "velocity_raw_deg_per_sec"] = raw_velocity
        df.at[i, "velocity_eye_used"] = used_eye
        
        # Speichere die tatsächliche Fensterbreite (in Samples)
        final_window_size = last_idx - first_idx + 1
        df.at[i, "window_width_samples"] = final_window_size

    # Transparenz: Fenster-Statistik ausgeben
    computed = (~df["velocity_deg_per_sec"].isna()).sum()
    if computed > 0:
        logger.info("Computed velocity for %s/%s samples", computed, n)
        logger.info(
            "Window configuration: window_length_ms=%.1f, selector=%s",
            cfg.window_length_ms,
            type(selector).__name__,
        )

    # fixed_window_edge_fallback: Wenn Fensterrand invalide Samples hat,
    # kopiere Velocity vom nächsten Sample mit gültigem Fenster
    # WICHTIG: Unclassified Samples (durch gap_rule) dürfen NICHT überschrieben werden
    if cfg.fixed_window_edge_fallback and isinstance(selector, FixedSampleSymmetricWindowSelector):
        logger.info("Applying fixed_window_edge_fallback strategy...")
        half_size = selector.half_size
        fallback_count = 0
        
        for i in range(n):
            # Nur für valide Samples ohne berechnete Velocity
            if not bool(valid[i]) or not pd.isna(df.at[i, "velocity_deg_per_sec"]):
                continue
            
            # NICHT überschreiben, wenn gap_rule_triggered (= absichtlich Unclassified)
            if "gap_rule_triggered" in df.columns:
                if bool(df.at[i, "gap_rule_triggered"]):
                    continue
            
            # Prüfe ob das Problem invalide Fensterränder sind
            window_start = max(0, i - half_size)
            window_end = min(n - 1, i + half_size)
            
            # Hat das Fenster invalide Samples am Rand?
            has_invalid_edge = False
            if window_start < i:  # Links-Rand prüfen
                if not bool(valid[window_start]):
                    has_invalid_edge = True
            if window_end > i:  # Rechts-Rand prüfen
                if not bool(valid[window_end]):
                    has_invalid_edge = True
            
            if not has_invalid_edge:
                continue
            
            # Suche nächstes Sample mit gültiger Velocity (links und rechts)
            found_velocity = None
            search_radius = min(50, n)  # Max 50 Samples suchen
            
            for offset in range(1, search_radius):
                # Erst rechts suchen
                if i + offset < n and bool(valid[i + offset]):
                    vel = df.at[i + offset, "velocity_deg_per_sec"]
                    if not pd.isna(vel):
                        found_velocity = vel
                        break
                # Dann links suchen
                if i - offset >= 0 and bool(valid[i - offset]):
                    vel = df.at[i - offset, "velocity_deg_per_sec"]
                    if not pd.isna(vel):
                        found_velocity = vel
                        break
            
            if found_velocity is not None:
                df.at[i, "velocity_deg_per_sec"] = found_velocity
                fallback_count += 1
        
        if fallback_count > 0:
            logger.info("Applied fallback velocity to %s samples", fallback_count)

    return df



def compute_olsen_velocity_from_slim_tsv(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Optional[OlsenVelocityConfig] = None,
) -> pd.DataFrame:

    from ..io import read_tsv, write_tsv

    if cfg is None:
        cfg = OlsenVelocityConfig()

    df = read_tsv(input_path)
    df = compute_olsen_velocity(df, cfg)

    if output_path is not None:
        write_tsv(df, output_path)

    return df
