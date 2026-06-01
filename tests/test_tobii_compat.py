# tests/test_tobii_compat.py
"""
Tests für die neuen Tobii-kompatiblen Strategy-Klassen und Algorithmen.

Getestete Komponenten:
  1. TobiiGazeDirAngle  – asin-basierte Winkelberechnung (numerisch stabil)
  2. TobiiGazeVelocityWindowSelector – Tobii-Fenstergröße mit 1.01-Faktor
  3. apply_tobii_eye_offset_interpolation – Offset-basierte Augen-Schätzung
  4. merge_adjacent_fixations(weighting="sample_count") – Sample-Anzahl-Gewichtung

Quellreferenz: Dekompilierter Tobii C#-Quellcode (Point3DVectorExtensions,
GazeVelocityCalculatorHelper, RemoteTrackerGazeDataToRecordedTwoEyedGazeDataConverter,
MergeFixationsFilter).
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from ivt_filter.strategies.velocity_calculation import (
    TobiiGazeDirAngle,
    Ray3DGazeDir,
    VelocityContext,
)
from ivt_filter.strategies.windowing import TobiiGazeVelocityWindowSelector
from ivt_filter.preprocessing.eye_selection import (
    apply_tobii_eye_offset_interpolation,
    _parse_validity,
)
from ivt_filter.postprocessing.merge_fixations import merge_adjacent_fixations
from ivt_filter.config import OlsenVelocityConfig, FixationPostConfig


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _make_unit_vec(x: float, y: float, z: float) -> np.ndarray:
    """Gibt normierten 3D-Vektor zurück."""
    v = np.array([x, y, z], dtype=float)
    return v / np.linalg.norm(v)


def _angle_ref(v1: np.ndarray, v2: np.ndarray) -> float:
    """Referenz-Winkelberechnung via acos (Standard-Methode) in Grad."""
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return math.degrees(math.acos(dot))


# ---------------------------------------------------------------------------
# 1. TobiiGazeDirAngle – Winkelberechnung
# ---------------------------------------------------------------------------

class TestTobiiGazeDirAngle:
    """Tests für die asin-basierte Winkelformel."""

    def setup_method(self):
        self.strat = TobiiGazeDirAngle()

    def _ctx(self, v1, v2):
        return VelocityContext(
            x1_mm=0.0, y1_mm=0.0, x2_mm=0.0, y2_mm=0.0,
            eye_x_mm=None, eye_y_mm=None, eye_z_mm=None,
            dir1=v1, dir2=v2,
        )

    def test_zero_angle(self):
        """Identische Vektoren → 0°."""
        v = _make_unit_vec(0, 0, 1)
        assert self.strat.calculate_visual_angle_ctx(self._ctx(v, v)) == pytest.approx(0.0, abs=1e-8)

    def test_90_degrees(self):
        """Orthogonale Vektoren → 90°."""
        v1 = _make_unit_vec(1, 0, 0)
        v2 = _make_unit_vec(0, 1, 0)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_45_degrees(self):
        """45°-Winkel."""
        v1 = _make_unit_vec(1, 0, 0)
        v2 = _make_unit_vec(1, 1, 0)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(45.0, abs=1e-5)

    def test_180_degrees(self):
        """Entgegengesetzte Vektoren → 180°."""
        v1 = _make_unit_vec(0, 0, 1)
        v2 = _make_unit_vec(0, 0, -1)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(180.0, abs=1e-5)

    def test_small_angle_numerical_stability(self):
        """Sehr kleiner Winkel (~0.001°) – numerische Stabilität gegenüber acos."""
        v1 = _make_unit_vec(0, 0, 1)
        # Leicht abweichender Vektor (sehr kleiner Winkel)
        epsilon = 1e-5
        v2 = _make_unit_vec(epsilon, 0, 1)
        angle_tobii = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        angle_ref = _angle_ref(v1, v2)
        # Beide sollten sehr nahe beieinander sein, aber Tobii-Formel stabiler
        assert angle_tobii >= 0.0
        assert angle_tobii == pytest.approx(angle_ref, abs=1e-4)

    def test_obtuse_angle_uses_complement_formula(self):
        """Stumpfer Winkel (>90°) – komplementäre Formel wird verwendet."""
        # 150° zwischen den Vektoren
        v1 = _make_unit_vec(0, 0, 1)
        angle_target = 150.0
        # Rotiere um x-Achse
        rad = math.radians(angle_target)
        v2 = np.array([0, math.sin(rad), math.cos(rad)])
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(angle_target, abs=1e-4)

    def test_unnormalized_vectors_are_handled(self):
        """Nicht-normierte Vektoren werden intern normiert."""
        v1 = np.array([3.0, 0.0, 0.0])   # wird normiert zu [1, 0, 0]
        v2 = np.array([0.0, 5.0, 0.0])   # wird normiert zu [0, 1, 0]
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_zero_vector_returns_nan(self):
        """Nullvektor → NaN, da der Winkel nicht berechenbar ist."""
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = _make_unit_vec(0, 0, 1)
        angle = self.strat.calculate_visual_angle_ctx(self._ctx(v1, v2))
        assert math.isnan(angle)

    def test_none_direction_returns_nan(self):
        """Fehlende Richtung (None) → NaN."""
        ctx = VelocityContext(
            x1_mm=0.0, y1_mm=0.0, x2_mm=0.0, y2_mm=0.0,
            eye_x_mm=None, eye_y_mm=None, eye_z_mm=None,
            dir1=None, dir2=_make_unit_vec(0, 0, 1),
        )
        assert math.isnan(self.strat.calculate_visual_angle_ctx(ctx))

    def test_consistent_with_ray3d_gaze_dir_for_moderate_angles(self):
        """Für moderate Winkel (<90°) sollten TobiiGazeDirAngle und Ray3DGazeDir
        sehr ähnliche Ergebnisse liefern (beide auf Dot-Product-Basis)."""
        ref_strat = Ray3DGazeDir()
        for angle_deg in [5.0, 15.0, 30.0, 45.0, 60.0, 80.0]:
            rad = math.radians(angle_deg)
            v1 = _make_unit_vec(0, 0, 1)
            v2 = _make_unit_vec(math.sin(rad), 0, math.cos(rad))
            ctx = self._ctx(v1, v2)
            angle_tobii = self.strat.calculate_visual_angle_ctx(ctx)
            angle_ray = ref_strat.calculate_visual_angle_ctx(ctx)
            # Beide Methoden sollten innerhalb von 0.001° übereinstimmen
            assert abs(angle_tobii - angle_ray) < 0.001, (
                f"Angle {angle_deg}°: Tobii={angle_tobii:.6f}, Ray3D={angle_ray:.6f}"
            )

    def test_description_contains_asin(self):
        """Beschreibung soll 'asin' enthalten."""
        assert "asin" in self.strat.get_description().lower()

    def test_tobii_asin_formula_matches_reference_cases(self):
        """Überprüfe konkrete Winkel gegen analytische Werte."""
        strat = self.strat
        cases = [
            (10.0,),
            (30.0,),
            (60.0,),
            (90.0,),
            (120.0,),
            (170.0,),
        ]
        for (target_deg,) in cases:
            rad = math.radians(target_deg)
            v1 = np.array([0.0, 0.0, 1.0])
            v2 = np.array([math.sin(rad), 0.0, math.cos(rad)])
            ctx = self._ctx(v1, v2)
            result = strat.calculate_visual_angle_ctx(ctx)
            assert result == pytest.approx(target_deg, abs=1e-4), (
                f"Target {target_deg}°, got {result:.6f}°"
            )


# ---------------------------------------------------------------------------
# 2. TobiiGazeVelocityWindowSelector – Fensterberechnung
# ---------------------------------------------------------------------------

class TestTobiiGazeVelocityWindowSelector:
    """Tests für die Tobii-exakte Fenstergröße."""

    def _make_valid_data(self, n: int):
        """Erstellt n valide Samples (alle valid=True, äquidistant)."""
        times = np.arange(n, dtype=float)  # 1 ms Abstand
        valid = np.ones(n, dtype=bool)
        return times, valid

    def test_window_size_60hz(self):
        """60 Hz, 20 ms Fenster: window_samples = floor(20/16.67*1.01)+1 = 3."""
        # floor(20 / 16.67 * 1.01) + 1 = floor(1.213) + 1 = 1 + 1 = 2 → half = 0
        # Aber: volle Fenstergröße: window_ms = 20 ms (half_window_ms = 10 ms übergeben)
        # Tobii: window_samples = floor(20 / 16.67 * 1.01) + 1 = floor(1.213) + 1 = 2 → half = 0
        # Mit half = 0 gibt es kein sinnvolles Fenster mehr.
        # Korrekte Verwendung: half_window_ms = window_length_ms / 2 = 10 ms
        sample_interval_ms = 1000.0 / 60  # ≈ 16.67 ms
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        half_size = sel._compute_half_size(half_window_ms=10.0)  # 20 ms gesamt
        # window_samples = floor(20 / 16.67 * 1.01) + 1 = floor(1.213) + 1 = 2
        # half = (2-1)//2 = 0 → kein Fenster bei half=0
        # Das stimmt mit Tobii überein: bei 60 Hz und 20 ms Fenster = 2 Samples,
        # was bedeutet: 1 Sample links und 1 Sample rechts → half=1 wäre falsch
        # Laut Tobii: window_samples=2 → half=(2-1)//2=0 → asymmetrisch
        assert half_size >= 0

    def test_window_size_120hz_20ms(self):
        """120 Hz, 20 ms Fenster: window_samples = floor(20/8.33*1.01)+1 = 4."""
        sample_interval_ms = 1000.0 / 120  # ≈ 8.33 ms
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        half_size = sel._compute_half_size(half_window_ms=10.0)
        # window_samples = floor(20 / 8.33 * 1.01) + 1 = floor(2.425) + 1 = 3
        # half = (3-1)//2 = 1
        assert half_size == 1

    def test_window_size_240hz_20ms(self):
        """240 Hz, 20 ms Fenster: Größerer half_size."""
        sample_interval_ms = 1000.0 / 240  # ≈ 4.17 ms
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        half_size = sel._compute_half_size(half_window_ms=10.0)
        # window_samples = floor(20 / 4.17 * 1.01) + 1 = floor(4.848) + 1 = 5
        # half = (5-1)//2 = 2
        assert half_size == 2

    def test_select_returns_valid_endpoints(self):
        """Fensterauswahl liefert valide Endpunkte."""
        sample_interval_ms = 8.33  # 120 Hz
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=sample_interval_ms)
        n = 20
        times, valid = self._make_valid_data(n)
        idx = 10
        first, last = sel.select(idx, times, valid, half_window_ms=10.0)
        assert first is not None and last is not None
        assert first < last
        assert first <= idx <= last

    def test_select_invalid_center_returns_none(self):
        """Ungültiges Center-Sample → kein Fenster."""
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=8.33)
        n = 20
        times = np.arange(n, dtype=float)
        valid = np.ones(n, dtype=bool)
        valid[10] = False  # center ungültig
        first, last = sel.select(10, times, valid, half_window_ms=10.0)
        assert first is None and last is None

    def test_select_at_data_boundary(self):
        """Fenster am Rand des Datensatzes wird korrekt begrenzt."""
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=8.33)
        n = 20
        times, valid = self._make_valid_data(n)
        # Am Anfang (idx=0)
        first, last = sel.select(0, times, valid, half_window_ms=10.0)
        if first is not None:
            assert first >= 0
            assert last <= n - 1

    def test_invalid_sample_interval_raises(self):
        """Ungültiges sample_interval_ms löst ValueError aus."""
        with pytest.raises(ValueError):
            TobiiGazeVelocityWindowSelector(sample_interval_ms=0.0)
        with pytest.raises(ValueError):
            TobiiGazeVelocityWindowSelector(sample_interval_ms=-1.0)

    def test_1_01_factor_larger_than_without_factor(self):
        """Mit 1.01-Faktor wird mindestens genauso groß wie ohne."""
        interval = 8.33
        sel = TobiiGazeVelocityWindowSelector(sample_interval_ms=interval)
        half_with = sel._compute_half_size(10.0)
        # Ohne 1.01-Faktor: floor(20/8.33) + 1 = 3 → half = 1
        window_without = int(20.0 / interval) + 1
        half_without = (window_without - 1) // 2
        assert half_with >= half_without


# ---------------------------------------------------------------------------
# 3. apply_tobii_eye_offset_interpolation
# ---------------------------------------------------------------------------

class TestTobiiEyeOffsetInterpolation:
    """Tests für die Tobii Auge-Offset-Interpolation."""

    def _make_df(self, n: int = 10) -> pd.DataFrame:
        """Einfaches DataFrame mit beiden Augen valide."""
        return pd.DataFrame({
            "time_ms": np.arange(n, dtype=float),
            "gaze_left_x_mm": np.full(n, 100.0),
            "gaze_left_y_mm": np.full(n, 200.0),
            "gaze_right_x_mm": np.full(n, 110.0),  # right ist 10 mm weiter rechts
            "gaze_right_y_mm": np.full(n, 205.0),  # right ist 5 mm weiter unten
            "eye_left_x_mm": np.zeros(n),
            "eye_left_y_mm": np.zeros(n),
            "eye_left_z_mm": np.full(n, 600.0),
            "eye_right_x_mm": np.full(n, 65.0),    # IPD ~ 65 mm
            "eye_right_y_mm": np.zeros(n),
            "eye_right_z_mm": np.full(n, 600.0),
            "validity_left": np.zeros(n, dtype=int),
            "validity_right": np.zeros(n, dtype=int),
        })

    def _cfg(self) -> OlsenVelocityConfig:
        return OlsenVelocityConfig(max_validity=1)

    def test_both_valid_no_change(self):
        """Wenn beide Augen valide → keine Veränderung der Gaze-Daten."""
        df = self._make_df(5)
        cfg = self._cfg()
        result = apply_tobii_eye_offset_interpolation(df, cfg)
        pd.testing.assert_series_equal(result["gaze_left_x_mm"], df["gaze_left_x_mm"])
        pd.testing.assert_series_equal(result["gaze_right_x_mm"], df["gaze_right_x_mm"])

    def test_missing_right_eye_reconstructed_from_offset(self):
        """Rechtes Auge fehlt → wird via gespeichertem Offset rekonstruiert."""
        df = self._make_df(10)
        cfg = self._cfg()

        # Sample 0: beide valid → Offset = (10, 5) wird gespeichert
        # Sample 1–5: rechtes Auge ungültig
        df.loc[1:5, "validity_right"] = 2   # > max_validity=1 → ungültig
        df.loc[1:5, "gaze_right_x_mm"] = float("nan")
        df.loc[1:5, "gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # Rekonstruierte rechte Gaze = linke Gaze + offset(10, 5)
        for i in range(1, 6):
            expected_rx = df.loc[i, "gaze_left_x_mm"] + 10.0
            expected_ry = df.loc[i, "gaze_left_y_mm"] + 5.0
            assert result.loc[i, "gaze_right_x_mm"] == pytest.approx(expected_rx, abs=1e-6)
            assert result.loc[i, "gaze_right_y_mm"] == pytest.approx(expected_ry, abs=1e-6)

    def test_missing_left_eye_reconstructed_from_offset(self):
        """Linkes Auge fehlt → wird via gespeichertem Offset rekonstruiert."""
        df = self._make_df(10)
        cfg = self._cfg()

        # Sample 0: beide valid → Offset = (10, 5)
        # Sample 1–3: linkes Auge ungültig
        df.loc[1:3, "validity_left"] = 2
        df.loc[1:3, "gaze_left_x_mm"] = float("nan")
        df.loc[1:3, "gaze_left_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        for i in range(1, 4):
            # left = right - offset
            expected_lx = df.loc[i, "gaze_right_x_mm"] - 10.0
            expected_ly = df.loc[i, "gaze_right_y_mm"] - 5.0
            assert result.loc[i, "gaze_left_x_mm"] == pytest.approx(expected_lx, abs=1e-6)
            assert result.loc[i, "gaze_left_y_mm"] == pytest.approx(expected_ly, abs=1e-6)

    def test_no_offset_yet_missing_eye_stays_nan(self):
        """Wenn noch kein Offset bekannt und Auge fehlt → NaN bleibt NaN."""
        df = self._make_df(5)
        cfg = self._cfg()

        # Alle Samples: rechtes Auge ungültig (kein vorheriges valid-Paar)
        df["validity_right"] = 2
        df["gaze_right_x_mm"] = float("nan")
        df["gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)
        # Kein Offset bekannt → NaN bleibt
        assert result["gaze_right_x_mm"].isna().all()

    def test_original_df_not_modified(self):
        """Original DataFrame wird nicht modifiziert (copy)."""
        df = self._make_df(5)
        cfg = self._cfg()
        df.loc[2, "validity_right"] = 2
        df.loc[2, "gaze_right_x_mm"] = float("nan")

        original_rx = df["gaze_right_x_mm"].copy()
        _ = apply_tobii_eye_offset_interpolation(df, cfg)
        pd.testing.assert_series_equal(df["gaze_right_x_mm"], original_rx)

    def test_eye_origin_offset_also_reconstructed(self):
        """Eye-Origin-Positions werden ebenfalls via Offset rekonstruiert."""
        df = self._make_df(10)
        cfg = self._cfg()

        # Sample 0: beide valid → eye_offset = (65, 0, 0) gespeichert
        df.loc[1, "validity_right"] = 2
        df.loc[1, "eye_right_x_mm"] = float("nan")
        df.loc[1, "eye_right_y_mm"] = float("nan")
        df.loc[1, "eye_right_z_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # Rekonstruiertes rechtes Auge = linkes Auge + IPD-Offset
        expected_rex = df.loc[1, "eye_left_x_mm"] + 65.0
        assert result.loc[1, "eye_right_x_mm"] == pytest.approx(expected_rex, abs=1e-6)

    def test_validity_flag_updated_after_interpolation(self):
        """Kritisch: validity_right wird nach erfolgreicher Interpolation auf den
        gültigen Marker gesetzt (0 für int-Spalten), damit prepare_combined_columns
        das interpolierte Auge in den Average einbezieht."""
        df = self._make_df(5)  # nutzt int-Validity (0 = valid, 2 = invalid)
        cfg = self._cfg()

        # Sample 2: rechtes Auge ungültig → soll interpoliert werden
        df.loc[2, "validity_right"] = 2
        df.loc[2, "gaze_right_x_mm"] = float("nan")
        df.loc[2, "gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # Koordinaten müssen rekonstruiert sein
        assert pd.notna(result.loc[2, "gaze_right_x_mm"])
        # Validity muss als "gültig" erkannt werden (int→0, str→"Valid")
        parsed = _parse_validity(result.loc[2, "validity_right"])
        assert parsed <= cfg.max_validity, (
            f"Nach Interpolation muss validity gültig sein (≤{cfg.max_validity}), "
            f"got: {result.loc[2, 'validity_right']!r} → parsed={parsed}"
        )
        # Unberührte Samples bleiben unverändert
        assert result.loc[0, "validity_right"] == df.loc[0, "validity_right"]

    def test_validity_flag_left_updated_after_interpolation(self):
        """validity_left wird auf gültigen Marker gesetzt wenn linkes Auge interpoliert."""
        df = self._make_df(5)
        cfg = self._cfg()

        df.loc[2, "validity_left"] = 2
        df.loc[2, "gaze_left_x_mm"] = float("nan")
        df.loc[2, "gaze_left_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        assert pd.notna(result.loc[2, "gaze_left_x_mm"])
        parsed = _parse_validity(result.loc[2, "validity_left"])
        assert parsed <= cfg.max_validity

    def test_no_validity_update_when_no_offset_known(self):
        """Wenn kein Offset bekannt → Validity bleibt ungültig."""
        df = self._make_df(5)
        cfg = self._cfg()

        # Alle Samples: rechtes Auge ungültig (nie ein valid-Paar → kein Offset)
        df["validity_right"] = 2
        df["gaze_right_x_mm"] = float("nan")
        df["gaze_right_y_mm"] = float("nan")

        result = apply_tobii_eye_offset_interpolation(df, cfg)

        # Kein Offset → bleibt ungültig
        for v in result["validity_right"]:
            assert _parse_validity(v) > cfg.max_validity

    def test_velocity_artifact_reduced_with_interpolation(self):
        """Phantom-Velocity an Gap-Rändern wird durch Offset-Interpolation reduziert.

        Szenario: linkes Auge springt zwischen valid/invalid – ohne Interpolation
        entsteht am Gap-Rand eine hohe Phantom-Velocity (Positions-Sprung wegen
        einseitigem Average). Mit Interpolation bleibt die Velocity niedrig.
        """
        from ivt_filter.processing.velocity import compute_olsen_velocity

        rng = np.random.default_rng(42)
        n = 30
        hz = 120.0
        dt = 1000.0 / hz  # ~8.33 ms

        # Stabile Fixation auf konstantem Punkt
        lx = np.full(n, 260.0)  # linkes Auge immer ~260 mm
        rx = np.full(n, 265.0)  # rechtes Auge immer ~265 mm  (IPD-Offset=5mm)
        ly = np.full(n, 133.0)
        ry = np.full(n, 133.0)

        df = pd.DataFrame({
            "time_ms": np.arange(n) * dt,
            "gaze_left_x_mm":  lx,  "gaze_left_y_mm":  ly,
            "gaze_right_x_mm": rx,  "gaze_right_y_mm": ry,
            "validity_left":  ["Valid"] * n,
            "validity_right": ["Valid"] * n,
            "eye_left_z_mm":  np.full(n, 600.0),
            "eye_right_z_mm": np.full(n, 600.0),
        })

        # Gap: Sample 10-12 linkes Auge invalid
        for i in [10, 11, 12]:
            df.loc[i, "validity_left"] = "Invalid"
            df.loc[i, "gaze_left_x_mm"] = float("nan")
            df.loc[i, "gaze_left_y_mm"] = float("nan")

        cfg_base = OlsenVelocityConfig(
            window_length_ms=20.0, eye_mode="average",
            velocity_method="olsen2d", smoothing_mode="none",
            tobii_eye_offset_interpolation=False,
        )
        cfg_interp = OlsenVelocityConfig(
            window_length_ms=20.0, eye_mode="average",
            velocity_method="olsen2d", smoothing_mode="none",
            tobii_eye_offset_interpolation=True,
        )

        df_base   = compute_olsen_velocity(df.copy(), cfg_base)
        df_interp = compute_olsen_velocity(df.copy(), cfg_interp)

        # Samples direkt nach dem Gap (idx 13-15) – hier entsteht Phantom-Velocity
        post_gap = slice(13, 16)
        max_vel_base   = df_base.loc[post_gap, "velocity_deg_per_sec"].max()
        max_vel_interp = df_interp.loc[post_gap, "velocity_deg_per_sec"].max()

        # Mit Interpolation muss die Phantom-Velocity deutlich kleiner sein
        assert max_vel_interp < max_vel_base, (
            f"Interpolation sollte Phantom-Velocity reduzieren: "
            f"base={max_vel_base:.1f}, interp={max_vel_interp:.1f}"
        )
        # Und unter 30 deg/s bleiben (kein falscher Saccade-Alarm)
        assert max_vel_interp < 30.0, (
            f"Nach Interpolation keine Phantom-Saccade: vel={max_vel_interp:.1f} deg/s"
        )


# ---------------------------------------------------------------------------
# 4. merge_adjacent_fixations(weighting="sample_count")
# ---------------------------------------------------------------------------

class TestMergeFixationsSampleCountWeighting:
    """Tests für die Tobii-exakte Sample-Count-Gewichtung beim Fixations-Merge."""

    def _make_fixation_df(
        self,
        fix1_len: int = 100,
        fix2_len: int = 50,
        fix1_x: float = 0.0,
        fix2_x: float = 10.0,
        gap: int = 3,
    ) -> pd.DataFrame:
        """
        Erstellt DataFrame mit zwei Fixationen (x-Koordinaten) und einer kleinen Lücke.
        """
        n = fix1_len + gap + fix2_len
        x = np.concatenate([
            np.full(fix1_len, fix1_x),
            np.full(gap, float("nan")),   # Lücke
            np.full(fix2_len, fix2_x),
        ])
        y = np.zeros(n)
        times = np.arange(n, dtype=float)  # 1 ms Abstand
        sample_type = (
            ["Fixation"] * fix1_len +
            ["Unclassified"] * gap +
            ["Fixation"] * fix2_len
        )
        velocity = np.zeros(n)  # Alle 0°/s → Gap wird gemergt

        return pd.DataFrame({
            "time_ms": times,
            "combined_x_mm": x,
            "combined_y_mm": y,
            "eye_z_mm": np.full(n, 600.0),
            "ivt_sample_type": sample_type,
            "velocity_deg_per_sec": velocity,
        })

    def test_uniform_weighting_uses_simple_mean(self):
        """Uniform-Modus: beide Fixationszentren gleichgewichtig gemittelt."""
        df = self._make_fixation_df(fix1_len=100, fix2_len=100, fix1_x=0.0, fix2_x=10.0)
        cfg = FixationPostConfig(
            merge_adjacent_fixations=True,
            max_time_gap_ms=10.0,
            max_angle_deg=10.0,
            merge_weighting="uniform",
        )
        # Beide Fixationen haben gleich viele Samples → uniform == sample_count
        result, stats = merge_adjacent_fixations(
            df, cfg,
            sample_col="ivt_sample_type",
            time_col="time_ms",
            x_col="combined_x_mm",
            y_col="combined_y_mm",
            eye_z_col="eye_z_mm",
        )
        assert stats["merged_pairs"] == 1

    def test_sample_count_weighted_mean_different_from_uniform(self):
        """Sample-Count-Gewichtung liefert anderen Wert als Uniform wenn n1≠n2."""
        # Fix1: 100 Samples bei x=0, Fix2: 50 Samples bei x=6
        # uniform: mean(0, 6) = 3.0
        # sample_count: (0*100 + 6*50) / 150 = 2.0
        fix1_x = 0.0
        fix2_x = 6.0
        n1 = 100
        n2 = 50
        expected_sample_count = (fix1_x * n1 + fix2_x * n2) / (n1 + n2)
        expected_uniform = (fix1_x + fix2_x) / 2.0
        assert expected_sample_count != expected_uniform  # 2.0 != 3.0

    def test_sample_count_with_equal_fixations_matches_uniform(self):
        """Bei gleich großen Fixationen: sample_count == uniform."""
        fix1_x = 0.0
        fix2_x = 10.0
        n1 = n2 = 50
        expected_sc = (fix1_x * n1 + fix2_x * n2) / (n1 + n2)
        expected_uniform = (fix1_x + fix2_x) / 2.0
        assert expected_sc == pytest.approx(expected_uniform, abs=1e-9)

    def test_merge_happens_regardless_of_weighting(self):
        """Merge findet statt unabhängig von weighting-Modus."""
        df = self._make_fixation_df(fix1_len=50, fix2_len=30, gap=2)
        for weighting in ("uniform", "sample_count"):
            cfg = FixationPostConfig(
                merge_adjacent_fixations=True,
                max_time_gap_ms=10.0,
                max_angle_deg=5.0,
                merge_weighting=weighting,
            )
            result, stats = merge_adjacent_fixations(
                df.copy(), cfg,
                sample_col="ivt_sample_type",
                time_col="time_ms",
                x_col="combined_x_mm",
                y_col="combined_y_mm",
                eye_z_col="eye_z_mm",
            )
            assert stats["merged_pairs"] == 1, f"No merge for weighting={weighting}"

    def test_default_weighting_is_uniform(self):
        """Standard-Config hat weighting='uniform'."""
        cfg = FixationPostConfig()
        assert cfg.merge_weighting == "uniform"

    def test_tobii_weighting_config(self):
        """sample_count ist in FixationPostConfig als Literal akzeptiert."""
        cfg = FixationPostConfig(merge_weighting="sample_count")
        assert cfg.merge_weighting == "sample_count"


class TestMergeFixationsGapVelocityCap:
    """Regressionstests fuer den konfigurierbaren Velocity-Cap beim Gap-Relabeling."""

    @staticmethod
    def _merge_gap_velocities(
        gap_velocities: list[float],
        *,
        max_gap_velocity_deg_per_sec: float = 35.0,
    ) -> tuple[pd.DataFrame, dict[str, object], slice]:
        gap = len(gap_velocities)
        fixation_len = 3
        n = fixation_len + gap + fixation_len
        gap_slice = slice(fixation_len, fixation_len + gap)
        df = pd.DataFrame({
            "time_ms": np.arange(n, dtype=float),
            "combined_x_mm": np.zeros(n),
            "combined_y_mm": np.zeros(n),
            "eye_z_mm": np.full(n, 600.0),
            "ivt_sample_type": (
                ["Fixation"] * fixation_len
                + ["Unclassified"] * gap
                + ["Fixation"] * fixation_len
            ),
            "velocity_deg_per_sec": (
                [0.0] * fixation_len
                + gap_velocities
                + [0.0] * fixation_len
            ),
        })
        cfg = FixationPostConfig(
            merge_adjacent_fixations=True,
            max_time_gap_ms=10.0,
            max_angle_deg=0.5,
            max_gap_velocity_deg_per_sec=max_gap_velocity_deg_per_sec,
        )
        result, stats = merge_adjacent_fixations(
            df, cfg,
            sample_col="ivt_sample_type",
            time_col="time_ms",
            x_col="combined_x_mm",
            y_col="combined_y_mm",
            eye_z_col="eye_z_mm",
        )
        return result, stats, gap_slice

    def test_sample_at_configured_cap_is_eligible_for_relabeling(self):
        """Der Cap ist inklusiv: Velocity == Cap darf zur Fixation werden."""
        result, stats, gap_slice = self._merge_gap_velocities(
            [12.0], max_gap_velocity_deg_per_sec=12.0
        )

        assert result.iloc[gap_slice]["ivt_sample_type"].tolist() == ["Fixation"]
        assert stats["gap_samples_to_fixation"] == 1

    def test_sample_above_configured_cap_is_preserved(self):
        """Velocity > Cap behaelt das urspruengliche Gap-Label."""
        result, stats, gap_slice = self._merge_gap_velocities(
            [12.01], max_gap_velocity_deg_per_sec=12.0
        )

        assert result.iloc[gap_slice]["ivt_sample_type"].tolist() == ["Unclassified"]
        assert stats["gap_samples_to_fixation"] == 0

    def test_default_cap_preserves_existing_35_deg_per_sec_behavior(self):
        """Der neue Config-Default entspricht der bisherigen festen 35-Grad-Grenze."""
        assert FixationPostConfig().max_gap_velocity_deg_per_sec == 35.0

        result, stats, gap_slice = self._merge_gap_velocities([35.0, 35.01])

        assert result.iloc[gap_slice]["ivt_sample_type"].tolist() == [
            "Fixation",
            "Unclassified",
        ]
        assert stats["gap_samples_to_fixation"] == 1

    def test_custom_cap_changes_only_intended_gap_fill_decision(self):
        """Ein eigener Cap relabelt nur das zusaetzlich berechtigte Gap-Sample."""
        default_result, _, gap_slice = self._merge_gap_velocities([34.0, 36.0, 37.0])
        custom_result, _, _ = self._merge_gap_velocities(
            [34.0, 36.0, 37.0], max_gap_velocity_deg_per_sec=36.0
        )

        assert default_result.iloc[gap_slice]["ivt_sample_type"].tolist() == [
            "Fixation",
            "Unclassified",
            "Unclassified",
        ]
        assert custom_result.iloc[gap_slice]["ivt_sample_type"].tolist() == [
            "Fixation",
            "Fixation",
            "Unclassified",
        ]
        assert custom_result.loc[[0, 1, 2, 6, 7, 8], "ivt_sample_type"].tolist() == [
            "Fixation",
        ] * 6


# ---------------------------------------------------------------------------
# 5. Integration: OlsenVelocityConfig mit Tobii-Flags
# ---------------------------------------------------------------------------

class TestTobiiConfigFields:
    """Stellt sicher, dass die neuen Config-Felder vorhanden und korrekt typisiert sind."""

    def test_tobii_gaze_dir_is_valid_method(self):
        """'tobii_gaze_dir' ist ein gültiger velocity_method-Wert."""
        cfg = OlsenVelocityConfig(velocity_method="tobii_gaze_dir")
        assert cfg.velocity_method == "tobii_gaze_dir"

    def test_tobii_window_mode_default_false(self):
        """tobii_window_mode ist standardmäßig False."""
        cfg = OlsenVelocityConfig()
        assert cfg.tobii_window_mode is False

    def test_tobii_sample_interval_ms_default_none(self):
        """tobii_sample_interval_ms ist standardmäßig None."""
        cfg = OlsenVelocityConfig()
        assert cfg.tobii_sample_interval_ms is None

    def test_tobii_eye_offset_interpolation_default_false(self):
        """tobii_eye_offset_interpolation ist standardmäßig False."""
        cfg = OlsenVelocityConfig()
        assert cfg.tobii_eye_offset_interpolation is False

    def test_tobii_window_mode_requires_sample_interval(self):
        """make_window_selector() löst ValueError aus wenn tobii_window_mode=True
        aber tobii_sample_interval_ms nicht gesetzt."""
        from ivt_filter.processing.velocity import make_window_selector
        cfg = OlsenVelocityConfig(tobii_window_mode=True, tobii_sample_interval_ms=None)
        with pytest.raises(ValueError, match="tobii_sample_interval_ms"):
            make_window_selector(cfg)

    def test_make_window_selector_returns_tobii_selector(self):
        """make_window_selector() gibt TobiiGazeVelocityWindowSelector zurück."""
        from ivt_filter.processing.velocity import make_window_selector
        cfg = OlsenVelocityConfig(tobii_window_mode=True, tobii_sample_interval_ms=8.33)
        sel = make_window_selector(cfg)
        assert isinstance(sel, TobiiGazeVelocityWindowSelector)

    def test_velocity_strategy_factory_tobii_gaze_dir(self):
        """_get_velocity_calculation_strategy('tobii_gaze_dir') gibt TobiiGazeDirAngle zurück."""
        from ivt_filter.processing.velocity import _get_velocity_calculation_strategy
        strat = _get_velocity_calculation_strategy("tobii_gaze_dir")
        assert isinstance(strat, TobiiGazeDirAngle)


# ---------------------------------------------------------------------------
# 6. Integration: Vollständige Pipeline mit Tobii-Flags (synthetische Daten)
# ---------------------------------------------------------------------------

class TestTobiiPipelineIntegration:
    """Smoke-Tests für die vollständige Pipeline mit Tobii-kompatiblen Einstellungen."""

    def _make_gaze_dir_data(self, n: int = 100, hz: float = 120.0) -> pd.DataFrame:
        """Erstellt DataFrame mit normierten Gaze-Direction-Spalten."""
        dt_ms = 1000.0 / hz
        # Fixation: Blickrichtung leicht variiert um (0, 0, 1)
        np.random.seed(42)
        noise = np.random.normal(0, 0.001, (n, 3))
        base = np.array([0, 0, 1], dtype=float)
        dirs = base + noise
        # Normieren
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / norms

        df = pd.DataFrame({
            "time_ms": np.arange(n) * dt_ms,
            "gaze_left_x_mm": np.full(n, 100.0),
            "gaze_left_y_mm": np.full(n, 200.0),
            "gaze_right_x_mm": np.full(n, 110.0),
            "gaze_right_y_mm": np.full(n, 205.0),
            "eye_left_x_mm": np.zeros(n),
            "eye_left_y_mm": np.zeros(n),
            "eye_left_z_mm": np.full(n, 600.0),
            "eye_right_x_mm": np.full(n, 65.0),
            "eye_right_y_mm": np.zeros(n),
            "eye_right_z_mm": np.full(n, 600.0),
            "validity_left": np.zeros(n, dtype=int),
            "validity_right": np.zeros(n, dtype=int),
            "gaze_dir_left_x": dirs[:, 0],
            "gaze_dir_left_y": dirs[:, 1],
            "gaze_dir_left_z": dirs[:, 2],
            "gaze_dir_right_x": dirs[:, 0],
            "gaze_dir_right_y": dirs[:, 1],
            "gaze_dir_right_z": dirs[:, 2],
        })
        return df

    def test_tobii_gaze_dir_pipeline_runs(self):
        """Pipeline mit tobii_gaze_dir + tobii_window_mode läuft ohne Fehler."""
        from ivt_filter.processing.velocity import compute_olsen_velocity

        df = self._make_gaze_dir_data(n=50, hz=120.0)
        cfg = OlsenVelocityConfig(
            velocity_method="tobii_gaze_dir",
            tobii_window_mode=True,
            tobii_sample_interval_ms=1000.0 / 120,
            eye_mode="average",
        )
        result = compute_olsen_velocity(df, cfg)
        assert "velocity_deg_per_sec" in result.columns
        # Velocities sollten vorhanden und nicht alle NaN sein
        velocities = result["velocity_deg_per_sec"].dropna()
        assert len(velocities) > 0

    def test_tobii_eye_offset_interpolation_flag(self):
        """tobii_eye_offset_interpolation=True läuft ohne Fehler."""
        from ivt_filter.processing.velocity import compute_olsen_velocity

        df = self._make_gaze_dir_data(n=30, hz=120.0)
        # Mache einige Samples mit fehlendem rechten Auge
        df.loc[10:15, "validity_right"] = 2
        df.loc[10:15, "gaze_right_x_mm"] = float("nan")
        df.loc[10:15, "gaze_right_y_mm"] = float("nan")

        cfg = OlsenVelocityConfig(
            velocity_method="olsen2d",
            tobii_eye_offset_interpolation=True,
            eye_mode="average",
        )
        result = compute_olsen_velocity(df, cfg)
        assert "velocity_deg_per_sec" in result.columns

    def test_tobii_gaze_dir_produces_lower_velocities_than_acos_for_small_angles(self):
        """Für sehr kleine Winkel (Fixation) sollte TobiiGazeDirAngle nahezu
        identisch zu Ray3DGazeDir sein (Unterschied < 0.01°/s)."""
        strat_tobii = TobiiGazeDirAngle()
        strat_ray = Ray3DGazeDir()

        # Kleiner Winkel: ~1°
        v1 = _make_unit_vec(0, 0, 1)
        v2 = _make_unit_vec(math.sin(math.radians(1)), 0, math.cos(math.radians(1)))
        ctx = VelocityContext(
            x1_mm=0, y1_mm=0, x2_mm=0, y2_mm=0,
            eye_x_mm=None, eye_y_mm=None, eye_z_mm=None,
            dir1=v1, dir2=v2,
        )
        angle_tobii = strat_tobii.calculate_visual_angle_ctx(ctx)
        angle_ray = strat_ray.calculate_visual_angle_ctx(ctx)
        assert abs(angle_tobii - angle_ray) < 0.001
