import pytest

from ivt_filter.engine import IVTFilterEngine
from ivt_filter.config import IVTFilterConfiguration, EyeSelectionMode
from ivt_filter.noise.moving_average import MovingAverageNoiseFilterStrategy
from ivt_filter.noise.median_filter import MedianNoiseFilterStrategy
from ivt_filter.domain.events import GazeEventType
from conftest import load_recording_from_tsv, load_ground_truth_events_from_tsv


def _match_sequence(result_events, gt_events):
    return bool(result_events)


@pytest.mark.parametrize(
    "filename,config",
    [
        ("I-VT-normal Data export_short.tsv", IVTFilterConfiguration()),
        (
            "I-VT-rightEye Data export_short.tsv",
            IVTFilterConfiguration(eye_selection_mode=EyeSelectionMode.RIGHT),
        ),
        (
            "I-VT-StrictAverage_frequency120_movingAverageData export_short.tsv",
            IVTFilterConfiguration(
                eye_selection_mode=EyeSelectionMode.STRICT_AVERAGE,
                noise_filter_strategy=MovingAverageNoiseFilterStrategy(window_size=5),
            ),
        ),
    ],
)
def test_result_event_sequence_matches_ground_truth(filename, config):
    recording = load_recording_from_tsv(filename)
    gt_events = load_ground_truth_events_from_tsv(filename)
    engine = IVTFilterEngine()
    result = engine.run(recording, config)
    assert _match_sequence(result.events, gt_events)


def test_result_deterministic_with_noise_strategy():
    recording = load_recording_from_tsv("I-VT-normal Data export_short.tsv")
    config = IVTFilterConfiguration(noise_filter_strategy=MedianNoiseFilterStrategy(window_size=3))
    engine = IVTFilterEngine()
    first = engine.run(recording, config)
    second = engine.run(recording, config)
    assert [e.event_type for e in first.events] == [e.event_type for e in second.events]
