from datetime import datetime

from ivt_filter.noise.moving_average import MovingAverageNoiseFilterStrategy
from ivt_filter.noise.median_filter import MedianNoiseFilterStrategy
from ivt_filter.domain.dataset import EyeData, Sample


def build_samples(values):
    base = datetime.fromtimestamp(0)
    samples = []
    for idx, v in enumerate(values):
        ts = base
        sample = Sample(
            timestamp=ts,
            left_validity=0,
            right_validity=0,
            left_eye=EyeData(gaze_x=v, gaze_y=v),
            right_eye=EyeData(gaze_x=v, gaze_y=v),
            combined_gaze_x=v,
            combined_gaze_y=v,
            combined_valid=True,
        )
        samples.append(sample)
    return samples


def test_moving_average_smooths():
    samples = build_samples([0, 10, 0])
    strategy = MovingAverageNoiseFilterStrategy(window_size=3)
    smoothed = strategy.apply(samples)
    assert smoothed[1].combined_gaze_x == 10 / 3


def test_median_filter_smooths():
    samples = build_samples([0, 10, 0])
    strategy = MedianNoiseFilterStrategy(window_size=3)
    smoothed = strategy.apply(samples)
    assert smoothed[1].combined_gaze_x == 0
