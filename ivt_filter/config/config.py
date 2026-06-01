# ivt_filter/config.py
"""
Konfigurationsklassen für die I-VT Filter Pipeline.

Dieses Modul definiert alle Konfigurationsparameter für:
  - Velocity-Berechnung (Olsen-Style mit Erweiterungen)
  - I-VT Klassifikation (Fixation/Saccade Detection)
  - Post-Processing (Saccade Merging, Fixation Filtering)

Beispiel:
    >>> from config import OlsenVelocityConfig, IVTClassifierConfig
    >>> 
    >>> # Standard-Konfiguration
    >>> vel_cfg = OlsenVelocityConfig(
    ...     window_length_ms=20.0,
    ...     velocity_method="olsen2d",
    ...     eye_mode="average"
    ... )
    >>> 
    >>> # 3D Ray Methode mit Koordinaten-Rounding
    >>> vel_cfg = OlsenVelocityConfig(
    ...     window_length_ms=20.0,
    ...     velocity_method="ray3d",
    ...     coordinate_rounding="halfup",
    ...     smoothing_mode="median",
    ...     smoothing_window_samples=5
    ... )
    >>> 
    >>> # Klassifikator
    >>> clf_cfg = IVTClassifierConfig(
    ...     velocity_threshold_deg_per_sec=30.0
    ... )
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Literal, Optional

from .window_policy import (
    FixedSampleWindowPolicy,
    ShiftedValidWindowPolicy,
    TobiiWindowPolicy,
    WindowPolicy,
    translate_legacy_window_flags,
    window_policy_from_dict,
)


def _require_choice(name: str, value: object, choices: tuple[object, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _require_non_negative(name: str, value: object) -> None:
    if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive(name: str, value: object) -> None:
    if not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: int) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_odd_samples(name: str, value: int) -> None:
    _require_positive_integer(name, value)
    if value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer")


def _require_fixed_window_samples(name: str, value: int, *, allow_even: bool) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 3:
        raise ValueError(f"{name} must be an integer >= 3")
    if not allow_even and value % 2 == 0:
        raise ValueError(f"{name} must be odd unless allow_asymmetric_window=True")


@dataclass
class OlsenVelocityConfig:
    """
    Konfiguration fuer die Olsen-Style-Geschwindigkeitsberechnung.
    """

    # Zeitfenster in Millisekunden (Olsen-Style)
    window_length_ms: float = 20.0

    # Timestamp column and unit (allows microsecond precision when available)
    time_column: str = "time_ms"
    time_unit: Literal["ms", "us", "ns"] = "ms"

    # Augenmodus
    # - "left":   nur linkes Auge
    # - "right":  nur rechtes Auge
    # - "average": Mittelwert der Augen (Standard)
    eye_mode: Literal["left", "right", "average"] = "average"

    # Validitaet der Tobii-Codes
    max_validity: int = 1

    # Minimaler Zeitabstand fuer eine gueltige Velocity-Berechnung
    min_dt_ms: float = 0.1

    # Raeumliches Smoothing auf kombinierten Koordinaten
    # - "none": Kein Smoothing
    # - "median": Median-Filter (ignoriert invalide Samples)
    # - "moving_average": Moving Average (ignoriert invalide Samples)
    # - "median_strict": Median only if ALL samples in window are valid
    # - "moving_average_strict": Average only if ALL samples in window are valid
    # - "median_adaptive": Collects only valid samples, expands search if needed
    # - "moving_average_adaptive": Same as median_adaptive, but with mean
    smoothing_mode: Literal[
        "none", "median", "moving_average", 
        "median_strict", "moving_average_strict",
        "median_adaptive", "moving_average_adaptive"
    ] = "none"
    smoothing_window_samples: int = 5
    
    # Adaptive Smoothing Optionen
    # min_samples: Minimum number of valid samples for smoothing (default: 1)
    # expansion_radius: Search samples beyond standard window (default: 0)
    smoothing_min_samples: int = 1
    smoothing_expansion_radius: int = 0

    # Normalized, tagged selector policy. Legacy flags below remain temporarily
    # supported for direct callers and are normalized during initialization.
    window_policy: Optional[WindowPolicy] = None

    # Sample-symmetrisches Fenster innerhalb des Zeitfensters
    # (gleich viele Samples links/rechts, aber immer noch durch window_length_ms begrenzt)
    sample_symmetric_window: bool = False

    # Feste Fenstergroesse in Samples (optional, odd >= 3).
    # Wenn gesetzt, wird eine reine Sample-Fenster-Strategie verwendet
    # (FixedSampleSymmetricWindowSelector) und window_length_ms dient nur noch
    # als Referenz fuer dt-Minimum etc.
    fixed_window_samples: Optional[int] = None

    # Wenn True: berechne automatisch eine passende feste Fenstergröße
    # in Samples aus `window_length_ms` und beobachtetem Sampling-Intervall.
    auto_fixed_window_from_ms: bool = False

    # Asymmetrisches Fenster: wenn True, wird per_side = round(window_size / 2)
    # verwendet statt (window_size - 1) / 2. Erlaubt auch gerade Fenstergrößen.
    # Beispiel: window_size=8 -> per_side=4 (statt zu 9 aufgerundet)
    allow_asymmetric_window: bool = False

    # Shifted valid window: halte Fensterlaenge konstant und verschiebe das Fenster,
    # um einen durchgaengig gueltigen Block zu finden (keine Invalids im Fenster).
    # Erfordert fixed_window_samples.
    shifted_valid_window: bool = False
    # Fallback, wenn kein gueltiges Fenster der festen Laenge gefunden wird:
    # - "shrink": faellt auf das bisherige Shrink-Verhalten zurueck
    # - "unclassified": liefert kein Fenster (Sample wird spaeter unclassified)
    shifted_valid_fallback: Literal["shrink", "unclassified"] = "shrink"
    
    # Asymmetrisches Nachbar-Fenster (nur 2 direkt angrenzende Samples):
    # Wenn True: Nutze AsymmetricNeighborWindowSelector
    # - Priorität: Backward (i-1 → i)
    # - Fallback: Forward (i → i+1)
    # - Gap-Regel: 2 samples = 1 sample radius
    asymmetric_neighbor_window: bool = False
    
    # Fixed dt aus Sampling-Rate verwenden (für asymmetric_neighbor_window):
    # Wenn True: dt = 1/sampling_rate (präzise, ohne time_ms-Jitter)
    # Wenn False: dt aus time_ms-Differenzen (kann durch Rundung jittern)
    use_fixed_dt: bool = False

    # Symmetrische Rundungs-Logik: Bestimme per_side = round(window_size / 2)
    # und verwende dann 2*per_side + 1 als effektive Fenstergröße (symmetrisch um das Zentrum).
    # Diese Logik kann die Gesamtgröße erhöhen (z.B. 7 -> 9), ist aber unabhängig
    # von der Unclassified(Gap)-Regel, die weiterhin die ursprüngliche Größe nutzt.
    symmetric_round_window: bool = False

    # Methode zur Bestimmung der Sampling-Rate
    # - "all_samples": Verwende alle Samples (Standard, robuster)
    # - "first_100": Verwende nur die ersten 100 Samples (wie im Tobii-Paper)
    sampling_rate_method: Literal["all_samples", "first_100"] = "first_100"

    # Methode zur Berechnung der Zeitdifferenzen
    # - "median": Robuster gegenüber Ausreißern (Standard)
    # - "mean": Arithmetisches Mittel (wie im Tobii-Paper erwähnt)
    dt_calculation_method: Literal["median", "mean"] = "mean"

    # Bei ungültigen first/last Samples im Fenster: nächstes gültiges Sample verwenden
    use_fallback_valid_samples: bool = True

    # Bei fixed_window_samples: wenn Fensterrand ungültige Samples hat,
    # verwende Geschwindigkeit vom nächsten Sample mit gültigem Fenster
    fixed_window_edge_fallback: bool = False

    # Strategien fuer eye_mode="average"
    # - average_window_single_eye:
    #       bei mixed mono/binokular wird das Auge mit stabilerer Validitaet
    #       fuer Start/Ende des Fensters verwendet
    # - average_window_impute_neighbor:
    #       fehlende Augenkoordinate am Fensterrand mit naechstem Nachbarn
    #       (mit gueltigem Auge) imputieren
    # - average_fallback_single_eye:
    #       Wenn im gesamten Fenster (Start bis Ende) oder mittleren Sample
    #       nur ein Auge valide ist, wird DURCHGEHEND nur dieses eine Auge
    #       verwendet (kein Average). Verhindert Parallaxe-Effekte bei Augen-Wechsel.
    average_window_single_eye: bool = False
    average_window_impute_neighbor: bool = False
    average_fallback_single_eye: bool = False

    # Gap-Filling: kurze Luecken in den Eye-Tracks zeitlich interpolieren
    # (pro Auge), bevor die Augen kombiniert werden.
    gap_fill_enabled: bool = False
    gap_fill_max_gap_ms: float = 75.0  # z.B. bis 75 ms Luecke linear fuellen

    # Koordinaten-Rundung vor der Velocity-Berechnung
    # - "none": Keine Rundung (Standard)
    # - "nearest": Banker's Rounding (round, bei 0.5 zur geraden Zahl)
    # - "halfup": Bei 0.5 immer aufrunden (klassisches Rounding)
    # - "floor": Immer abrunden
    # - "ceil": Immer aufrunden
    coordinate_rounding: Literal["none", "nearest", "halfup", "floor", "ceil"] = "none"

    # Methode zur Berechnung des visuellen Winkels
    # - "olsen2d": Original Olsen 2D-Näherung: θ = atan(screen_distance / eye_z)
    #              Schnell, benötigt nur eye_z, 2D-Approximation
    #              Standard für Backward-Compatibility
    # - "ray3d": Physikalisch korrekte 3D-Ray-Methode:
    #            θ = acos(ray0 · ray1 / (|ray0| × |ray1|))
    #            Präziser, benötigt vollständige Eye-Position (x, y, z)
    #            Typisch 1-5% niedrigere Velocities als olsen2d
    # - "ray3d_gaze_dir": Nutzt die normalisierten Blickrichtungs-Vektoren (DACS norm)
    #            θ = acos(dir0 · dir1); benötigt keine Bildschirm- oder Eye-Position
    # - "tobii_gaze_dir": Tobii-exakte Formel aus dekompiliertem Quellcode
    #            θ = 2·asin(‖v₁−v₂‖/2) — numerisch stabiler als acos(dot product)
    #            Benötigt normierte Richtungsvektoren (DACS norm), wie ray3d_gaze_dir
    velocity_method: Literal["olsen2d", "ray3d", "ray3d_gaze_dir", "tobii_gaze_dir"] = "olsen2d"

    # Tobii-exaktes Fenster (GazeVelocityCalculator):
    # Wenn True, wird TobiiGazeVelocityWindowSelector verwendet:
    #   window_samples = floor(window_length_ms / tobii_sample_interval_ms * 1.01) + 1
    # Überschreibt alle anderen Fenster-Selektoren (höchste Priorität).
    tobii_window_mode: bool = False
    # Nominelles Abtastintervall in ms für Tobii-Fensterberechnung.
    # Beispiele: 16.67 = 60 Hz, 8.33 = 120 Hz, 4.17 = 240 Hz
    # Wird automatisch aus Sampling-Rate berechnet, wenn nicht gesetzt (None).
    tobii_sample_interval_ms: Optional[float] = None

    # Tobii-exakte Auge-Offset-Interpolation:
    # Wenn True, wird der zuletzt bekannte L→R Gaze-/Eye-Positions-Versatz gespeichert
    # und verwendet, um das fehlende Auge zu schätzen (statt einfachem Fallback).
    # Entspricht: RemoteTrackerGazeDataToRecordedTwoEyedGazeDataConverter (Tobii C#)
    tobii_eye_offset_interpolation: bool = False

    def __post_init__(self) -> None:
        """Validate values and normalize deprecated selector flags into one policy."""
        _require_positive("window_length_ms", self.window_length_ms)
        _require_positive("min_dt_ms", self.min_dt_ms)
        _require_non_negative_integer("max_validity", self.max_validity)
        _require_positive_odd_samples("smoothing_window_samples", self.smoothing_window_samples)
        _require_positive_integer("smoothing_min_samples", self.smoothing_min_samples)
        _require_non_negative_integer("smoothing_expansion_radius", self.smoothing_expansion_radius)
        _require_non_negative("gap_fill_max_gap_ms", self.gap_fill_max_gap_ms)
        _require_choice("time_unit", self.time_unit, ("ms", "us", "ns"))
        _require_choice("eye_mode", self.eye_mode, ("left", "right", "average"))
        _require_choice("smoothing_mode", self.smoothing_mode, (
            "none", "median", "moving_average", "median_strict",
            "moving_average_strict", "median_adaptive", "moving_average_adaptive",
        ))
        _require_choice("sampling_rate_method", self.sampling_rate_method, ("all_samples", "first_100"))
        _require_choice("dt_calculation_method", self.dt_calculation_method, ("median", "mean"))
        _require_choice("coordinate_rounding", self.coordinate_rounding, ("none", "nearest", "halfup", "floor", "ceil"))
        _require_choice("velocity_method", self.velocity_method, ("olsen2d", "ray3d", "ray3d_gaze_dir", "tobii_gaze_dir"))
        _require_choice("shifted_valid_fallback", self.shifted_valid_fallback, ("shrink", "unclassified"))
        if self.fixed_window_samples is not None:
            _require_fixed_window_samples(
                "fixed_window_samples", self.fixed_window_samples,
                allow_even=self.allow_asymmetric_window,
            )

        legacy_policy = translate_legacy_window_flags(
            sample_symmetric_window=self.sample_symmetric_window,
            fixed_window_samples=self.fixed_window_samples,
            auto_fixed_window_from_ms=self.auto_fixed_window_from_ms,
            symmetric_round_window=self.symmetric_round_window,
            asymmetric_neighbor_window=self.asymmetric_neighbor_window,
            shifted_valid_window=self.shifted_valid_window,
            shifted_valid_fallback=self.shifted_valid_fallback,
            tobii_window_mode=self.tobii_window_mode,
            tobii_sample_interval_ms=self.tobii_sample_interval_ms,
        )
        if isinstance(self.window_policy, dict):
            self.window_policy = window_policy_from_dict(self.window_policy)
        if self.window_policy is None:
            self.window_policy = legacy_policy
        elif legacy_policy.kind != "time_symmetric" and self.window_policy != legacy_policy:
            raise ValueError(
                "window_policy cannot be combined with contradictory deprecated legacy window flags."
            )

        policy = self.window_policy
        if isinstance(policy, (FixedSampleWindowPolicy, ShiftedValidWindowPolicy)):
            _require_choice("window_policy.fallback", getattr(policy, "fallback", "shrink"), ("shrink", "unclassified"))
            if policy.samples is not None:
                _require_fixed_window_samples(
                    "window_policy.samples", policy.samples,
                    allow_even=self.allow_asymmetric_window,
                )
        if isinstance(policy, TobiiWindowPolicy):
            if policy.sample_interval_ms is None:
                raise ValueError("Tobii window mode requires tobii_sample_interval_ms/sample_interval_ms")
            _require_positive("tobii_sample_interval_ms/sample_interval_ms", policy.sample_interval_ms)


@dataclass
class IVTClassifierConfig:
    """
    Konfiguration fuer den I-VT Threshold Klassifikator.
    """

    velocity_threshold_deg_per_sec: float = 30.0

    # Optional reconstruction heuristics. Disabled by default so the classifier
    # behaves as a strict I-VT velocity-threshold classifier.
    # Require an adjacent above-threshold velocity before classifying an
    # invalid-window sample as a saccade.
    enable_invalid_window_neighbor_confirmation: bool = False
    # Retain the previous motion label while velocity is in the band immediately
    # below the saccade threshold.
    enable_hysteresis: bool = False
    hysteresis_width_deg_per_sec: float = 2.0
    
    # Stage 1: Near-threshold hybrid strategy
    # Enable hybrid classification near threshold using alternative velocity
    enable_near_threshold_hybrid: bool = False
    # Band around threshold (deg/s) where alternative velocity is used
    near_threshold_band: float = 5.0
    # Asymmetric band: lower band (below threshold) - None means use symmetric band
    near_threshold_band_lower: float | None = None
    # Asymmetric band: upper band (above threshold) - None means use symmetric band
    near_threshold_band_upper: float | None = None
    # Hybrid strategy: 'replace' (always use alt), 'inverse' (use velocity farther from threshold)
    near_threshold_strategy: str = 'inverse'
    # Minimum confidence margin (deg/s) required to switch in inverse strategy
    near_threshold_confidence_margin: float = 0.4
    # Require base and alternative velocity to be on the same side of threshold
    near_threshold_require_same_side: bool = True
    # Maximum allowed difference |alt - base| (deg/s) to switch
    near_threshold_max_delta: float = 2.0
    # Neighbor consensus: require majority of neighbors to support alt side when flip occurs
    near_threshold_neighbor_check: bool = False
    
    # Rule A: Eye-position jump correction
    # Enable eye-position jump detection and correction
    enable_eye_jump_rule: bool = False
    # Eye position displacement threshold (mm) to trigger jump rule
    eye_jump_threshold_mm: float = 10.0
    # Velocity threshold for "clear saccade" (deg/s) to apply jump rule
    eye_jump_velocity_threshold: float = 50.0

    # Confident mismatch switch: use alternative velocity method when base is far from threshold
    enable_confident_switch: bool = False
    confident_switch_margin_deg: float = 4.0
    confident_switch_method: Literal["olsen2d", "ray3d", "ray3d_gaze_dir"] = "ray3d_gaze_dir"

    def __post_init__(self) -> None:
        _require_positive("velocity_threshold_deg_per_sec", self.velocity_threshold_deg_per_sec)
        for name in (
            "hysteresis_width_deg_per_sec", "near_threshold_band",
            "near_threshold_confidence_margin", "near_threshold_max_delta",
            "eye_jump_threshold_mm", "eye_jump_velocity_threshold",
            "confident_switch_margin_deg",
        ):
            _require_non_negative(name, getattr(self, name))
        for name in ("near_threshold_band_lower", "near_threshold_band_upper"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative(name, value)
        _require_choice("near_threshold_strategy", self.near_threshold_strategy, ("replace", "inverse"))
        _require_choice("confident_switch_method", self.confident_switch_method, ("olsen2d", "ray3d", "ray3d_gaze_dir"))


@dataclass
class SaccadeMergeConfig:
    """
    Konfiguration fuer das Post-Processing von kurzen Saccade-Bloecken.
    """

    # Saccade-Bloecke kuerzer als diese Dauer (ms), die komplett innerhalb
    # von GT-Fixationen liegen, koennen zu Fixationen umgelabelt werden.
    max_saccade_block_duration_ms: float = 20.0

    # Wenn True: nur mergen, wenn direkte GT-Nachbarn (sofern vorhanden)
    # ebenfalls Fixation sind.
    require_fixation_context: bool = True

    # Auf welcher Spalte operiert werden soll:
    # - "ivt_sample_type": sample-basiert, danach Events neu bauen
    # - None: direkt auf "ivt_event_type"
    use_sample_type_column: Optional[str] = "ivt_sample_type"

    def __post_init__(self) -> None:
        _require_positive("max_saccade_block_duration_ms", self.max_saccade_block_duration_ms)


@dataclass
class FixationPostConfig:
    """
    Konfiguration fuer die Tobii-aehnlichen Fixations-Filter:

      1) Zusammenfuehren benachbarter Fixationen
         (zeitlich nah + kleiner Winkelabstand)

      2) Verwerfen kurzer Fixationen
    """

    # Schritt 1: benachbarte Fixationen mergen
    merge_adjacent_fixations: bool = False
    max_time_gap_ms: float = 75.0   # z.B. 75 ms wie im Tobii-Paper
    max_angle_deg: float = 0.5      # z.B. 0.5 Grad zwischen Fixationszentren
    max_gap_velocity_deg_per_sec: float = 35.0  # maximale Velocity fuer Gap-Relabeling

    # Gewichtungsstrategie für gemittelte Fixations-Position beim Merge:
    # - "uniform": np.nanmean aller Samples (bisheriges Verhalten)
    # - "sample_count": Sample-Anzahl-gewichteter Mittelwert
    #   (entspricht Tobii MergeFixationsFilter: vector2Df / num)
    merge_weighting: Literal["uniform", "sample_count"] = "uniform"

    # Schritt 2: zu kurze Fixationen verwerfen
    discard_short_fixations: bool = False
    min_fixation_duration_ms: float = 60.0  # z.B. 60 ms Mindestdauer
    discard_target: Literal["Unclassified", "Saccade"] = "Unclassified"  # Ziel-Label für verworfene Fixationen

    def __post_init__(self) -> None:
        for name in (
            "max_time_gap_ms", "max_angle_deg", "max_gap_velocity_deg_per_sec",
            "min_fixation_duration_ms",
        ):
            _require_non_negative(name, getattr(self, name))
        _require_choice("merge_weighting", self.merge_weighting, ("uniform", "sample_count"))
        _require_choice("discard_target", self.discard_target, ("Unclassified", "Saccade"))


@dataclass(frozen=True)
class PipelineConfig:
    """Single source of truth for the active core processing stages.

    File I/O, evaluation, and plotting are deliberately runner concerns.  A
    post-processing stage is active exactly when its optional configuration is
    present.
    """

    velocity: OlsenVelocityConfig
    classifier: IVTClassifierConfig
    classify: bool = True
    saccade_merge: Optional[SaccadeMergeConfig] = None
    fixation_post: Optional[FixationPostConfig] = None

    def __post_init__(self) -> None:
        if not self.classify and (self.saccade_merge or self.fixation_post):
            raise ValueError("Post-processing stages require classification")
        if self.fixation_post and not (
            self.fixation_post.merge_adjacent_fixations
            or self.fixation_post.discard_short_fixations
        ):
            raise ValueError("Fixation post-processing config must enable at least one operation")
