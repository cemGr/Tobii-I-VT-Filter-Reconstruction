from __future__ import annotations

import math
from typing import List


from converter import prepare_recording_for_velocity
from domain import EyeData, IVTVelocityConfig, Recording, Sample
from ivt_velocity import IVTVelocityCalculator


def _make_sample(time_ms: float, gaze_x: float, gaze_y: float, eye_pos=(0.0, 0.0, 600.0)) -> Sample:
    left = EyeData(gaze_x_px=gaze_x, gaze_y_px=gaze_y, eye_pos_x_mm=eye_pos[0], eye_pos_y_mm=eye_pos[1], eye_pos_z_mm=eye_pos[2])
    right = EyeData(gaze_x_px=gaze_x, gaze_y_px=gaze_y, eye_pos_x_mm=eye_pos[0], eye_pos_y_mm=eye_pos[1], eye_pos_z_mm=eye_pos[2])
    return Sample(time_ms=time_ms, left=left, right=right, validity_left=0, validity_right=0)


def _prepare_recording(samples: List[Sample]) -> Recording:
    recording = Recording(id="synthetic", samples=samples)
    prepare_recording_for_velocity(recording)
    return recording


def test_constant_gaze_produces_near_zero_velocity() -> None:
    samples = [_make_sample(time_ms=i * 10.0, gaze_x=100.0, gaze_y=200.0) for i in range(20)]
    recording = _prepare_recording(samples)

    calculator = IVTVelocityCalculator(IVTVelocityConfig(window_length_ms=20.0))
    calculator.compute_velocities(recording)

    velocities = [s.velocity_deg_per_sec for s in recording.samples if s.velocity_deg_per_sec is not None]
    assert velocities, "Expected some velocity values"
    for v in velocities:
        assert math.isclose(v, 0.0, abs_tol=1e-3)


def test_linear_motion_produces_constant_positive_velocity() -> None:
    samples = [_make_sample(time_ms=i * 10.0, gaze_x=50.0 + i * 5.0, gaze_y=100.0) for i in range(20)]
    recording = _prepare_recording(samples)

    calculator = IVTVelocityCalculator(IVTVelocityConfig(window_length_ms=20.0))
    calculator.compute_velocities(recording)

    velocities = [s.velocity_deg_per_sec for s in recording.samples if s.velocity_deg_per_sec is not None]
    assert velocities, "Expected velocity values"
    first = velocities[0]
    for v in velocities:
        assert v > 0
        assert math.isfinite(v)
        assert math.isclose(v, first, rel_tol=0.1)


def test_window_edges_leave_none_velocity() -> None:
    samples = [_make_sample(time_ms=i * 10.0, gaze_x=10.0 * i, gaze_y=0.0) for i in range(7)]
    recording = _prepare_recording(samples)

    calculator = IVTVelocityCalculator(IVTVelocityConfig(window_length_ms=40.0))
    calculator.compute_velocities(recording)

    velocities = [s.velocity_deg_per_sec for s in recording.samples]
    assert velocities[0] is None
    assert velocities[1] is None
    assert velocities[-1] is None
    assert velocities[-2] is None
    assert any(v is not None for v in velocities[2:-2])


def test_real_data_computes_plausible_velocities() -> None:
    from converter import load_tobii_tsv_to_recording

    recording = load_tobii_tsv_to_recording("I-VT-normal Data export_short.tsv")
    prepare_recording_for_velocity(recording)

    calculator = IVTVelocityCalculator(IVTVelocityConfig(window_length_ms=20.0))
    calculator.compute_velocities(recording)

    velocities = [s.velocity_deg_per_sec for s in recording.samples if s.velocity_deg_per_sec is not None]
    assert len(velocities) > len(recording.samples) * 0.3
    assert all(math.isfinite(v) and v >= 0 for v in velocities)
    assert max(velocities) < 5000
