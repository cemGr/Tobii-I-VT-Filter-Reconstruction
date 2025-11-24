import math

import pandas as pd
import pytest

from ivt.config import OlsenVelocityConfig
from ivt.velocity import VelocityCalculator


def test_velocity_computation_uses_window_and_distance():
    df = pd.DataFrame(
        {
            "time_ms": [0, 10, 20],
            "gaze_left_x_px": [0.0, 1.0, 2.0],
            "gaze_left_y_px": [0.0, 0.0, 0.0],
            "gaze_right_x_px": [0.0, 1.0, 2.0],
            "gaze_right_y_px": [0.0, 0.0, 0.0],
            "validity_left": [0, 0, 0],
            "validity_right": [0, 0, 0],
            "eye_left_z_mm": [600.0, 600.0, 600.0],
            "eye_right_z_mm": [600.0, 600.0, 600.0],
        }
    )

    calc = VelocityCalculator(OlsenVelocityConfig(window_length_ms=20, eye_mode="average"))
    result = calc.compute(df)

    expected_angle = math.degrees(math.atan2(2.0, 600.0))
    expected_velocity = expected_angle / 0.02

    assert result.loc[0, "velocity_deg_per_sec"] == pytest.approx(expected_velocity, rel=1e-3)
    assert result.loc[2, "velocity_deg_per_sec"] == pytest.approx(expected_velocity, rel=1e-3)
    assert result.loc[1, "velocity_deg_per_sec"] == pytest.approx(expected_velocity, rel=1e-3)


def test_velocity_respects_minimum_time_delta():
    df = pd.DataFrame(
        {
            "time_ms": [0, 0],
            "gaze_left_x_px": [0.0, 1.0],
            "gaze_left_y_px": [0.0, 0.0],
            "gaze_right_x_px": [0.0, 1.0],
            "gaze_right_y_px": [0.0, 0.0],
            "validity_left": [0, 0],
            "validity_right": [0, 0],
            "eye_left_z_mm": [600.0, 600.0],
            "eye_right_z_mm": [600.0, 600.0],
        }
    )

    calc = VelocityCalculator(OlsenVelocityConfig(min_dt_ms=1.0))
    result = calc.compute(df)
    assert result["velocity_deg_per_sec"].isna().all()


def test_velocity_accepts_comma_decimal_strings():
    df = pd.DataFrame(
        {
            "time_ms": [0, 10, 20],
            "gaze_left_x_px": ["0,0", "1,0", "2,0"],
            "gaze_left_y_px": ["0,0", "0,0", "0,0"],
            "gaze_right_x_px": ["0,0", "1,0", "2,0"],
            "gaze_right_y_px": ["0,0", "0,0", "0,0"],
            "validity_left": [0, 0, 0],
            "validity_right": [0, 0, 0],
            "eye_left_z_mm": ["573,4", "573,4", "573,4"],
            "eye_right_z_mm": ["573,4", "573,4", "573,4"],
        }
    )

    calc = VelocityCalculator(OlsenVelocityConfig(window_length_ms=20, eye_mode="average"))
    result = calc.compute(df)

    assert result["velocity_deg_per_sec"].notna().all()
