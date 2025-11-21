from datetime import datetime, timedelta

import pytest

from ivt_filter.engine import IVTFilterEngine
from ivt_filter.config import IVTFilterConfiguration, EyeSelectionMode
from ivt_filter.domain.dataset import EyeData, Recording, Sample
from ivt_filter.domain.events import GazeEventType


def build_constant_recording(length: int = 20, gaze_x: float = 100.0) -> Recording:
    base = datetime.fromtimestamp(0)
    samples = []
    for i in range(length):
        ts = base + timedelta(milliseconds=i * 10)
        samples.append(
            Sample(
                timestamp=ts,
                left_validity=0,
                right_validity=0,
                left_eye=EyeData(gaze_x=gaze_x, gaze_y=gaze_x),
                right_eye=EyeData(gaze_x=gaze_x, gaze_y=gaze_x),
            )
        )
    return Recording(id="const", start_time=samples[0].timestamp, end_time=samples[-1].timestamp, samples=samples)


def test_constant_gaze_yields_fixation():
    recording = build_constant_recording()
    engine = IVTFilterEngine()
    result = engine.run(recording, IVTFilterConfiguration())
    assert len(result.events) == 1
    assert result.events[0].event_type == GazeEventType.FIXATION


def test_jump_creates_saccade():
    base = datetime.fromtimestamp(0)
    samples = []
    for i in range(12):
        ts = base + timedelta(milliseconds=i * 10)
        gaze = 100.0 if i < 4 or i > 7 else 400.0
        samples.append(
            Sample(
                timestamp=ts,
                left_validity=0,
                right_validity=0,
                left_eye=EyeData(gaze_x=gaze, gaze_y=gaze),
                right_eye=EyeData(gaze_x=gaze, gaze_y=gaze),
            )
        )
    rec = Recording(id="jump", start_time=samples[0].timestamp, end_time=samples[-1].timestamp, samples=samples)
    result = IVTFilterEngine().run(rec, IVTFilterConfiguration(minimum_fixation_duration_ms=0))
    types = [e.event_type for e in result.events]
    assert GazeEventType.SACCADE in types
    assert types[0] == GazeEventType.FIXATION


def test_strict_average_marks_invalid_when_one_eye_missing():
    base = datetime.fromtimestamp(0)
    ts = base
    sample = Sample(
        timestamp=ts,
        left_validity=0,
        right_validity=4,
        left_eye=EyeData(gaze_x=10.0, gaze_y=10.0),
        right_eye=EyeData(gaze_x=200.0, gaze_y=200.0),
    )
    rec = Recording(id="single", start_time=ts, end_time=ts, samples=[sample])
    config = IVTFilterConfiguration(eye_selection_mode=EyeSelectionMode.STRICT_AVERAGE)
    res = IVTFilterEngine().run(rec, config)
    assert res.recording.samples[0].combined_valid is False
