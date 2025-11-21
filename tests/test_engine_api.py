from ivt_filter.engine import IVTFilterEngine
from ivt_filter.config import IVTFilterConfiguration
from ivt_filter.domain.events import GazeEventType


def test_engine_runs(simple_recording):
    engine = IVTFilterEngine()
    config = IVTFilterConfiguration(minimum_fixation_duration_ms=0)
    result = engine.run(simple_recording, config)
    assert result.events
    assert any(e.event_type == GazeEventType.FIXATION for e in result.events)


def test_engine_deterministic(simple_recording):
    engine = IVTFilterEngine()
    config = IVTFilterConfiguration(minimum_fixation_duration_ms=0)
    first = engine.run(simple_recording, config)
    second = engine.run(simple_recording, config)
    assert [e.event_type for e in first.events] == [e.event_type for e in second.events]
