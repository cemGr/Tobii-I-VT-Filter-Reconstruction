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
from typing import Literal, Optional


@dataclass
class OlsenVelocityConfig:
    """
    Konfiguration fuer die Olsen-Style-Geschwindigkeitsberechnung.
    """

    # Zeitfenster in Millisekunden (Olsen-Style)
    window_length_ms: float = 20.0

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
    smoothing_mode: Literal["none", "median", "moving_average"] = "none"
    smoothing_window_samples: int = 5

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
    # Beispiel: window_size=8 -> per_side=4 (statt zu 9 aufgerunden)
    allow_asymmetric_window: bool = False

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
    dt_calculation_method: Literal["median", "mean"] = "median"

    # Bei ungültigen first/last Samples im Fenster: nächstes gültiges Sample verwenden
    use_fallback_valid_samples: bool = True

    # Strategien fuer eye_mode="average"
    # - average_window_single_eye:
    #       bei mixed mono/binokular wird das Auge mit stabilerer Validitaet
    #       fuer Start/Ende des Fensters verwendet
    # - average_window_impute_neighbor:
    #       fehlende Augenkoordinate am Fensterrand mit naechstem Nachbarn
    #       (mit gueltigem Auge) imputieren
    average_window_single_eye: bool = False
    average_window_impute_neighbor: bool = False

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
    velocity_method: Literal["olsen2d", "ray3d"] = "olsen2d"


@dataclass
class IVTClassifierConfig:
    """
    Konfiguration fuer den I-VT Threshold Klassifikator.
    """

    velocity_threshold_deg_per_sec: float = 30.0


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

    # Schritt 2: zu kurze Fixationen verwerfen
    discard_short_fixations: bool = False
    min_fixation_duration_ms: float = 60.0  # z.B. 60 ms Mindestdauer
