"""Sample-level velocity calculation and endpoint handling."""
from __future__ import annotations

import dataclasses
from decimal import Decimal, ROUND_HALF_UP
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from ..config import OlsenVelocityConfig, PhysicalConstants
from ..domain.schema import validate_preprocessed_frame
from ..strategies import (
    AsymmetricNeighborWindowSelector,
    FixedSampleSymmetricWindowSelector,
    Ray3DGazeDir,
    SampleSymmetricWindowSelector,
    ShiftedValidWindowSelector,
    TimeSymmetricWindowSelector,
    TobiiGazeDirAngle,
    VelocityCalculationStrategy,
    VelocityContext,
    WindowSelector,
)
from .velocity_factory import (
    VelocityStrategyFactory,
    WindowSelectorFactory,
    _get_coordinate_rounding_strategy,
)
from .velocity_input import SamplingAnalyzer, VelocityInputArrays

logger = logging.getLogger("ivt_filter.processing.velocity")

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



def find_single_eye_endpoints(
    valid: np.ndarray, first_idx: int, last_idx: int
) -> tuple[Optional[int], Optional[int]]:
    """Return the first and last valid endpoint for one eye inside a window."""
    first = next((idx for idx in range(first_idx, last_idx + 1) if valid[idx]), None)
    last = next((idx for idx in range(last_idx, first_idx - 1, -1) if valid[idx]), None)
    return first, last


@dataclasses.dataclass(frozen=True)
class AverageNeighborImputer:
    """Impute a missing-eye endpoint from the closest valid neighbor in a window."""

    arrays: VelocityInputArrays

    @classmethod
    def from_arrays(cls, arrays: VelocityInputArrays) -> "AverageNeighborImputer":
        return cls(arrays)

    def impute(self, idx: int, first_idx: int, last_idx: int) -> tuple[float, float]:
        a = self.arrays
        left_valid, right_valid = bool(a.left_valid[idx]), bool(a.right_valid[idx])
        if left_valid == right_valid:
            return a.combined_x[idx], a.combined_y[idx]
        missing_right = left_valid and not right_valid
        candidates = range(first_idx, last_idx + 1)
        neighbor = min(
            (
                j
                for j in candidates
                if j != idx
                and bool(a.right_valid[j] if missing_right else a.left_valid[j])
            ),
            key=lambda j: abs(float(a.times[j]) - float(a.times[idx])),
            default=None,
        )
        if neighbor is None:
            return a.combined_x[idx], a.combined_y[idx]
        if missing_right:
            values = (
                a.left_x[idx],
                a.left_y[idx],
                a.right_x[neighbor],
                a.right_y[neighbor],
            )
        else:
            values = (
                a.left_x[neighbor],
                a.left_y[neighbor],
                a.right_x[idx],
                a.right_y[idx],
            )
        if any(pd.isna(value) for value in values):
            return a.combined_x[idx], a.combined_y[idx]
        return (float(values[0]) + float(values[2])) / 2.0, (
            float(values[1]) + float(values[3])
        ) / 2.0


@dataclasses.dataclass(frozen=True)
class FixedWindowEdgeFallbackContext:
    cfg: OlsenVelocityConfig
    selector: WindowSelector
    valid: np.ndarray


def apply_fixed_window_edge_fallback(
    df: pd.DataFrame, context: FixedWindowEdgeFallbackContext
) -> int:
    """Copy a nearby velocity into fixed-window samples blocked by invalid edges."""
    if not context.cfg.fixed_window_edge_fallback or not isinstance(
        context.selector, FixedSampleSymmetricWindowSelector
    ):
        return 0
    logger.info("Applying fixed_window_edge_fallback strategy...")
    valid, half_size, count = context.valid, context.selector.half_size, 0
    for i in range(len(df)):
        if not bool(valid[i]) or not pd.isna(df.at[i, "velocity_deg_per_sec"]):
            continue
        if "gap_rule_triggered" in df.columns and bool(df.at[i, "gap_rule_triggered"]):
            continue
        start, end = max(0, i - half_size), min(len(df) - 1, i + half_size)
        if not (
            (start < i and not bool(valid[start])) or (end > i and not bool(valid[end]))
        ):
            continue
        found = None
        for offset in range(1, min(50, len(df))):
            for candidate in (i + offset, i - offset):
                if 0 <= candidate < len(df) and bool(valid[candidate]):
                    velocity = df.at[candidate, "velocity_deg_per_sec"]
                    if not pd.isna(velocity):
                        found = candidate, velocity
                        break
            if found is not None:
                break
        if found is not None:
            source_idx, velocity = found
            df.at[i, "velocity_deg_per_sec"] = velocity
            for column in (
                "dt_ms",
                "velocity_raw_deg_per_sec",
                "window_width_samples",
                "velocity_first_idx",
                "velocity_last_idx",
                "velocity_eye_used",
                "velocity_window_selector",
            ):
                if column in df.columns:
                    df.at[i, column] = df.at[source_idx, column]
            if "velocity_fallback_applied" in df.columns:
                df.at[i, "velocity_fallback_applied"] = True
            count += 1
    if count:
        logger.info("Applied fallback velocity to %s samples", count)
    return count


@dataclasses.dataclass(frozen=True)
class ComputedVelocitySample:
    """Result values written for one successfully computed sample."""

    velocity_deg_per_sec: float
    raw_velocity_deg_per_sec: float
    dt_ms: float


@dataclasses.dataclass(frozen=True)
class VelocityDirectionArrays:
    """Optional gaze-direction vector arrays used by 3D velocity strategies."""

    left_x: np.ndarray
    left_y: np.ndarray
    left_z: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    right_z: np.ndarray
    combined_x: np.ndarray
    combined_y: np.ndarray
    combined_z: np.ndarray


@dataclasses.dataclass(frozen=True)
class PreparedVelocityArrays:
    """NumPy arrays required by the sample-level velocity computation."""

    input: VelocityInputArrays
    eye_x: np.ndarray
    eye_y: np.ndarray
    eye_z: np.ndarray
    eye_mode: str
    directions: VelocityDirectionArrays

    @property
    def times(self) -> np.ndarray:
        return self.input.times

    @property
    def valid(self) -> np.ndarray:
        return self.input.valid


def _initialize_velocity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with initialized output and diagnostic columns."""
    result = df.copy()
    result["velocity_deg_per_sec"] = float("nan")
    result["dt_ms"] = float("nan")
    result["window_width_samples"] = pd.NA
    result["window_any_invalid"] = False
    result["velocity_first_idx"] = pd.NA
    result["velocity_last_idx"] = pd.NA
    result["velocity_eye_used"] = pd.NA
    result["velocity_window_selector"] = pd.NA
    result["velocity_fallback_applied"] = pd.NA
    result["env_has_invalid_above"] = pd.NA
    result["env_has_invalid_below"] = pd.NA
    result["env_rule_triggered"] = pd.NA
    result["gap_rule_triggered"] = pd.NA
    result["gap_left_invalid_idx"] = pd.NA
    result["gap_right_invalid_idx"] = pd.NA
    return result


def _optional_array(df: pd.DataFrame, column: str) -> np.ndarray:
    """Return a column as NumPy array, or a NaN-filled array if absent."""
    if column in df.columns:
        return df[column].to_numpy()
    return np.full(len(df), np.nan)


def _combine_direction_vectors(
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    ldx: np.ndarray,
    ldy: np.ndarray,
    ldz: np.ndarray,
    rdx: np.ndarray,
    rdy: np.ndarray,
    rdz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine left/right gaze-direction vectors according to per-eye validity."""
    combined_dir_x = np.full(len(left_valid), np.nan)
    combined_dir_y = np.full(len(left_valid), np.nan)
    combined_dir_z = np.full(len(left_valid), np.nan)

    both_valid = left_valid.astype(bool) & right_valid.astype(bool)
    left_only = left_valid.astype(bool) & ~right_valid.astype(bool)
    right_only = ~left_valid.astype(bool) & right_valid.astype(bool)

    combined_dir_x[both_valid] = np.mean([ldx[both_valid], rdx[both_valid]], axis=0)
    combined_dir_y[both_valid] = np.mean([ldy[both_valid], rdy[both_valid]], axis=0)
    combined_dir_z[both_valid] = np.mean([ldz[both_valid], rdz[both_valid]], axis=0)

    combined_dir_x[left_only] = ldx[left_only]
    combined_dir_y[left_only] = ldy[left_only]
    combined_dir_z[left_only] = ldz[left_only]

    combined_dir_x[right_only] = rdx[right_only]
    combined_dir_y[right_only] = rdy[right_only]
    combined_dir_z[right_only] = rdz[right_only]
    return combined_dir_x, combined_dir_y, combined_dir_z


def _prepare_velocity_arrays(
    df: pd.DataFrame, cfg: OlsenVelocityConfig
) -> PreparedVelocityArrays:
    """Load all NumPy views used during velocity computation."""
    eye_mode = getattr(cfg, "eye_mode", "average")
    left_valid = df["left_eye_valid"].to_numpy()
    right_valid = df["right_eye_valid"].to_numpy()
    if eye_mode == "left":
        valid = left_valid
    elif eye_mode == "right":
        valid = right_valid
    else:
        valid = df["combined_valid"].to_numpy()

    input_arrays = VelocityInputArrays.from_dataframe(df, valid=valid)

    ldx = _optional_array(df, "gaze_dir_left_x")
    ldy = _optional_array(df, "gaze_dir_left_y")
    ldz = _optional_array(df, "gaze_dir_left_z")
    rdx = _optional_array(df, "gaze_dir_right_x")
    rdy = _optional_array(df, "gaze_dir_right_y")
    rdz = _optional_array(df, "gaze_dir_right_z")
    combined_dir_x, combined_dir_y, combined_dir_z = _combine_direction_vectors(
        left_valid, right_valid, ldx, ldy, ldz, rdx, rdy, rdz
    )

    return PreparedVelocityArrays(
        input=input_arrays,
        eye_x=df["eye_x_mm"].to_numpy(),
        eye_y=df["eye_y_mm"].to_numpy(),
        eye_z=df["eye_z_mm"].to_numpy(),
        eye_mode=eye_mode,
        directions=VelocityDirectionArrays(
            left_x=ldx,
            left_y=ldy,
            left_z=ldz,
            right_x=rdx,
            right_y=rdy,
            right_z=rdz,
            combined_x=combined_dir_x,
            combined_y=combined_dir_y,
            combined_z=combined_dir_z,
        ),
    )


@dataclasses.dataclass(frozen=True)
class VelocityComputationContext:
    """Sampling, window, strategy, and diagnostic state for velocity computation."""

    cfg: OlsenVelocityConfig
    arrays: PreparedVelocityArrays
    dt_med: Optional[float]
    hz_measured: Optional[float]
    half_window: float
    selector: WindowSelector
    fallback_selector: Optional[WindowSelector]
    coord_rounding: object
    velocity_strategy: VelocityCalculationStrategy
    gap_max: Optional[int]
    prev_invalid_idx: np.ndarray
    next_invalid_idx: np.ndarray
    neighbor_imputer: AverageNeighborImputer

    @classmethod
    def create(
        cls, arrays: PreparedVelocityArrays, cfg: OlsenVelocityConfig
    ) -> "VelocityComputationContext":
        sampling = SamplingAnalyzer().analyze(arrays.times, cfg)
        dt_med, hz_measured, effective_cfg = (
            sampling.dt_ms,
            sampling.hz_measured,
            sampling.config,
        )
        half_window = effective_cfg.window_length_ms / 2.0
        selector = WindowSelectorFactory.create(effective_cfg)
        logger.debug("Window selector: %s", type(selector).__name__)

        coord_rounding = _get_coordinate_rounding_strategy(
            effective_cfg.coordinate_rounding
        )
        if effective_cfg.coordinate_rounding != "none":
            logger.info(
                "Coordinate rounding enabled: %s", coord_rounding.get_description()
            )

        velocity_strategy = VelocityStrategyFactory.create(effective_cfg.velocity_method)
        if effective_cfg.velocity_method != "olsen2d":
            logger.info("Velocity calculation method: %s", effective_cfg.velocity_method)

        _log_window_configuration(selector, effective_cfg, dt_med, hz_measured)
        fallback_selector = _make_fallback_selector(selector)
        gap_max = _calculate_gap_max(selector, effective_cfg, dt_med)
        if gap_max is not None:
            logger.info("Gap-based Unclassified enabled: max_gap_samples=%s", gap_max)
        prev_invalid_idx, next_invalid_idx = _precompute_invalid_neighbors(
            arrays.valid
        )

        return cls(
            cfg=effective_cfg,
            arrays=arrays,
            dt_med=dt_med,
            hz_measured=hz_measured,
            half_window=half_window,
            selector=selector,
            fallback_selector=fallback_selector,
            coord_rounding=coord_rounding,
            velocity_strategy=velocity_strategy,
            gap_max=gap_max,
            prev_invalid_idx=prev_invalid_idx,
            next_invalid_idx=next_invalid_idx,
            neighbor_imputer=AverageNeighborImputer.from_arrays(arrays.input),
        )


def _log_window_configuration(
    selector: WindowSelector,
    cfg: OlsenVelocityConfig,
    dt_med: Optional[float],
    hz_measured: Optional[float],
) -> None:
    """Log the selected window configuration for transparent diagnostics."""
    if isinstance(selector, FixedSampleSymmetricWindowSelector):
        fixed_samples = getattr(cfg.window_policy, "samples", None)
        logger.info("Fixed sample window: %s samples", fixed_samples)
    elif isinstance(selector, AsymmetricNeighborWindowSelector):
        logger.info("Asymmetric neighbor window: 2 samples (backward/forward)")
        if cfg.use_fixed_dt and hz_measured is not None:
            logger.info(
                "Using fixed dt: %.4f ms (from %.1f Hz)",
                1000.0 / hz_measured,
                hz_measured,
            )
    elif dt_med is not None and dt_med > 0:
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


def _make_fallback_selector(selector: WindowSelector) -> Optional[WindowSelector]:
    """Return the time-window fallback for sample-symmetric selectors."""
    if isinstance(
        selector, (SampleSymmetricWindowSelector, FixedSampleSymmetricWindowSelector)
    ):
        return TimeSymmetricWindowSelector()
    return None


def _calculate_gap_max(
    selector: WindowSelector, cfg: OlsenVelocityConfig, dt_med: Optional[float]
) -> Optional[int]:
    """Calculate the invalid-gap radius for Unclassified diagnostics."""
    if isinstance(selector, AsymmetricNeighborWindowSelector):
        return 1
    if isinstance(selector, FixedSampleSymmetricWindowSelector) and isinstance(
        getattr(cfg.window_policy, "samples", None), int
    ):
        original_window_size = int(getattr(cfg.window_policy, "samples"))
        return max(0, original_window_size - 1)
    if dt_med is not None and dt_med > 0:
        original_est = int(round(cfg.window_length_ms / dt_med)) + 1
        return max(0, original_est - 1)
    return None


def _precompute_invalid_neighbors(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute nearest invalid sample indices to the left and right."""
    n = len(valid)
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
    return prev_invalid_idx, next_invalid_idx


class VelocitySampleComputer:
    """Compute individual velocity values and coordinate the sample iteration."""

    def __init__(self, cfg: OlsenVelocityConfig):
        self.cfg = cfg

    @staticmethod
    def compute_sample(
        context: VelocityContext,
        dt_ms: float,
        strategy: VelocityCalculationStrategy,
    ) -> ComputedVelocitySample:
        """Calculate one rounded velocity sample from an explicit context."""
        angle_deg = strategy.calculate_visual_angle_ctx(context)
        raw_velocity = angle_deg / (dt_ms / 1000.0) if dt_ms > 0 else float("nan")
        if pd.isna(raw_velocity):
            velocity = raw_velocity
        else:
            velocity = float(
                Decimal(raw_velocity).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            )
        return ComputedVelocitySample(velocity, raw_velocity, dt_ms)

    def compute(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        validate_preprocessed_frame(prepared_df)
        return _compute_olsen_velocity_impl(prepared_df, self.cfg)


def _compute_olsen_velocity_impl(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
) -> pd.DataFrame:
    """Olsen-Style Geschwindigkeit berechnen (mm basierte Gaze Daten)."""
    df = _initialize_velocity_columns(df)
    arrays = _prepare_velocity_arrays(df, cfg)
    if len(df) == 0:
        return df

    context = VelocityComputationContext.create(arrays, cfg)
    _compute_velocity_samples(df, context)
    apply_fixed_window_edge_fallback(
        df,
        FixedWindowEdgeFallbackContext(
            cfg=context.cfg, selector=context.selector, valid=arrays.valid
        ),
    )
    return df


def _compute_velocity_samples(
    df: pd.DataFrame, context: VelocityComputationContext
) -> None:
    """Compute velocity values for all samples using prepared context state."""
    cfg = context.cfg
    prepared = context.arrays
    arrays = prepared.input
    times = arrays.times
    cx = arrays.combined_x
    cy = arrays.combined_y
    cex = prepared.eye_x
    cey = prepared.eye_y
    cz = prepared.eye_z
    eye_mode = prepared.eye_mode
    valid = arrays.valid
    left_valid = arrays.left_valid
    right_valid = arrays.right_valid
    lx = arrays.left_x
    ly = arrays.left_y
    rx = arrays.right_x
    ry = arrays.right_y
    dirs = prepared.directions
    ldx = dirs.left_x
    ldy = dirs.left_y
    ldz = dirs.left_z
    rdx = dirs.right_x
    rdy = dirs.right_y
    rdz = dirs.right_z
    combined_dir_x = dirs.combined_x
    combined_dir_y = dirs.combined_y
    combined_dir_z = dirs.combined_z
    n = len(df)
    selector = context.selector
    fallback_selector = context.fallback_selector
    half_window = context.half_window
    hz_measured = context.hz_measured
    coord_rounding = context.coord_rounding
    velocity_strategy = context.velocity_strategy
    gap_max = context.gap_max
    prev_invalid_idx = context.prev_invalid_idx
    next_invalid_idx = context.next_invalid_idx
    neighbor_imputer = context.neighbor_imputer

    for i in range(n):
        # Explizit prüfen: nur wenn das Sample selbst gültig ist, berechne Velocity
        if not bool(valid[i]):
            continue

        # 1. Versuch: aktueller Selector (z.B. SampleSymmetric oder FixedSample)
        selected_selector = selector
        first_idx, last_idx = selector.select(i, times, valid, half_window)
        fallback_applied = bool(getattr(selector, "last_fallback_applied", False))

        # 2. Fallback: wenn kein sinnvolles Fenster gefunden wurde,
        #    versuche noch das klassische Zeitfenster
        if (
            first_idx is None or last_idx is None or first_idx == last_idx
        ) and fallback_selector is not None:
            first_idx_fb, last_idx_fb = fallback_selector.select(
                i, times, valid, half_window
            )
            if (
                first_idx_fb is not None
                and last_idx_fb is not None
                and first_idx_fb != last_idx_fb
            ):
                first_idx, last_idx = first_idx_fb, last_idx_fb
                selected_selector = fallback_selector
                fallback_applied = True

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
                        fallback_applied = True
                        break
                else:
                    continue  # Kein gültiges Sample gefunden

            if not valid[last_idx]:
                # Finde nächstes gültiges Sample vor last_idx
                for j in range(last_idx - 1, first_idx - 1, -1):
                    if valid[j]:
                        last_idx = j
                        fallback_applied = True
                        break
                else:
                    continue  # Kein gültiges Sample gefunden

        used_eye = eye_mode
        eye_consistent_override = False

        # Diagnose: Validität im aktuell gewählten Fenster
        window_lv = left_valid[first_idx : last_idx + 1]
        window_rv = right_valid[first_idx : last_idx + 1]
        window_any_invalid = (~window_lv | ~window_rv).any()
        df.at[i, "window_any_invalid"] = bool(window_any_invalid)

        # Berechne Zeitdifferenz (nutzt Window-Selector-Typ)
        dt_ms = _calculate_dt_ms(
            first_idx, last_idx, times, selected_selector, hz_measured, cfg.use_fixed_dt
        )

        # Track the actual endpoints used for velocity and direction lookup
        actual_first_idx = first_idx
        actual_last_idx = last_idx

        # Eye-consistent override for gaze-dir velocities (3-sample symmetric window)
        override_result = _apply_eye_consistent_override(
            velocity_strategy,
            cfg.eye_mode,
            first_idx,
            last_idx,
            times,
            left_valid,
            right_valid,
            lx,
            ly,
            rx,
            ry,
        )
        x1, y1, x2, y2, chosen_eye, override_applied, should_skip = override_result

        if should_skip:
            # No single eye valid at both endpoints -> velocity missing
            continue

        if override_applied:
            eye_consistent_override = True
            used_eye = chosen_eye
            # dt from the chosen endpoints' timestamps (time_ms array already normalized)
            dt_ms = _calculate_dt_ms(
                actual_first_idx,
                actual_last_idx,
                times,
                selected_selector,
                hz_measured,
                cfg.use_fixed_dt,
            )
        else:
            x1, y1 = cx[first_idx], cy[first_idx]
            x2, y2 = cx[last_idx], cy[last_idx]

        # Speichere finalen dt und wende min_dt-Prüfung an (außer beim späteren Single-Eye-Fallback)
        skip_dt_check = (
            cfg.eye_mode == "average"
            and cfg.average_fallback_single_eye
            and not eye_consistent_override
        )
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
                window_lv = left_valid[first_idx : last_idx + 1]
                window_rv = right_valid[first_idx : last_idx + 1]
                single_valid = window_lv ^ window_rv
                window_any_invalid = (~window_lv | ~window_rv).any()

                # Check middle sample validity too
                mid_idx = i
                mid_left_valid = left_valid[mid_idx]
                mid_right_valid = right_valid[mid_idx]
                mid_mixed = (mid_left_valid and not mid_right_valid) or (
                    not mid_left_valid and mid_right_valid
                )

                # Trigger fallback whenever the window is not fully valid for both eyes
                if window_any_invalid or single_valid.any() or mid_mixed:
                    # Evaluate both eyes: prefer the one with wider valid span, then more valid samples, then center validity
                    left_count = int(window_lv.sum())
                    right_count = int(window_rv.sum())

                    left_first, left_last = find_single_eye_endpoints(
                        left_valid, first_idx, last_idx
                    )
                    right_first, right_last = find_single_eye_endpoints(
                        right_valid, first_idx, last_idx
                    )
                    left_span = (
                        (left_last - left_first)
                        if left_first is not None and left_last is not None
                        else -1
                    )
                    right_span = (
                        (right_last - right_first)
                        if right_first is not None and right_last is not None
                        else -1
                    )

                    if mid_left_valid and not mid_right_valid:
                        left_count += 1
                    elif mid_right_valid and not mid_left_valid:
                        right_count += 1

                    score_left = (left_span, left_count)
                    score_right = (right_span, right_count)
                    if score_left == score_right:
                        chosen_eye = (
                            "left" if mid_left_valid or not mid_right_valid else "right"
                        )
                    else:
                        chosen_eye = "left" if score_left > score_right else "right"

                    # Find valid endpoints for chosen eye (with fallback search inside window)
                    fallback_applied = True
                    if chosen_eye == "left":
                        if left_first is not None:
                            actual_first_idx = left_first
                        if left_last is not None:
                            actual_last_idx = left_last

                        if (
                            left_first is None
                            or left_last is None
                            or actual_first_idx >= actual_last_idx
                        ):
                            continue

                        x1, y1 = lx[actual_first_idx], ly[actual_first_idx]
                        x2, y2 = lx[actual_last_idx], ly[actual_last_idx]
                        used_eye = "left"

                    else:  # right eye
                        if right_first is not None:
                            actual_first_idx = right_first
                        if right_last is not None:
                            actual_last_idx = right_last

                        if (
                            right_first is None
                            or right_last is None
                            or actual_first_idx >= actual_last_idx
                        ):
                            continue

                        x1, y1 = rx[actual_first_idx], ry[actual_first_idx]
                        x2, y2 = rx[actual_last_idx], ry[actual_last_idx]
                        used_eye = "right"

                    # Recalculate dt using the chosen eye indices
                    if cfg.use_fixed_dt and isinstance(
                        selected_selector, AsymmetricNeighborWindowSelector
                    ):
                        if hz_measured is not None and hz_measured > 0:
                            dt_ms = 1000.0 / hz_measured
                        else:
                            t_first = float(times[actual_first_idx])
                            t_last = float(times[actual_last_idx])
                            dt_ms = t_last - t_first
                    elif (
                        isinstance(selected_selector, FixedSampleSymmetricWindowSelector)
                        and hz_measured is not None
                        and hz_measured > 0
                    ):
                        window_size = actual_last_idx - actual_first_idx + 1
                        window_spans = window_size - 1
                        dt_ms = window_spans * (1000.0 / hz_measured)
                    elif isinstance(selected_selector, ShiftedValidWindowSelector):
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
                    x1, y1 = neighbor_imputer.impute(first_idx, first_idx, last_idx)
                    x2, y2 = neighbor_imputer.impute(last_idx, first_idx, last_idx)
                elif use_single_eye:
                    candidates: list[tuple[str, int]] = []
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
        if isinstance(velocity_strategy, (Ray3DGazeDir, TobiiGazeDirAngle)):
            dir_first, dir_last = _get_direction_vectors(
                actual_first_idx,
                actual_last_idx,
                eye_mode,
                used_eye,
                eye_consistent_override,
                ldx,
                ldy,
                ldz,
                rdx,
                rdy,
                rdz,
                combined_dir_x,
                combined_dir_y,
                combined_dir_z,
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

        sample_result = VelocitySampleComputer.compute_sample(
            ctx, dt_ms, velocity_strategy
        )
        df.at[i, "velocity_deg_per_sec"] = sample_result.velocity_deg_per_sec
        df.at[i, "dt_ms"] = sample_result.dt_ms
        df.at[i, "velocity_raw_deg_per_sec"] = sample_result.raw_velocity_deg_per_sec
        df.at[i, "velocity_first_idx"] = actual_first_idx
        df.at[i, "velocity_last_idx"] = actual_last_idx
        df.at[i, "velocity_eye_used"] = used_eye
        df.at[i, "velocity_window_selector"] = type(selected_selector).__name__
        df.at[i, "velocity_fallback_applied"] = fallback_applied

        # Speichere die tatsächliche Fensterbreite (in Samples)
        final_window_size = actual_last_idx - actual_first_idx + 1
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
