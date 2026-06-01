import math

import numpy as np
import pandas as pd

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.processing.velocity import (
    SamplingAnalyzer as facade_sampling_analyzer,
    VelocitySampleComputer as facade_sample_computer,
    make_window_selector as facade_make_window_selector,
)
from ivt_filter.processing.velocity_factory import (
    VelocityStrategyFactory,
    WindowSelectorFactory,
    make_window_selector,
)
from ivt_filter.processing.velocity_input import (
    SamplingAnalyzer,
    normalize_timestamps,
)
from ivt_filter.processing.velocity_samples import (
    VelocitySampleComputer,
    find_single_eye_endpoints,
)
from ivt_filter.strategies import Olsen2DApproximation, VelocityContext


def test_factory_boundary_creates_requested_window_and_velocity_strategies():
    cfg = OlsenVelocityConfig(fixed_window_samples=5)

    selector = WindowSelectorFactory.create(cfg)
    strategy = VelocityStrategyFactory.create("olsen2d")

    assert type(selector).__name__ == "FixedSampleSymmetricWindowSelector"
    assert isinstance(strategy, Olsen2DApproximation)


def test_input_boundary_normalizes_timestamp_copy_and_analyzes_sampling():
    original = pd.DataFrame({"timestamp": [30_000, 10_000, 20_000]})
    cfg = OlsenVelocityConfig(time_column="timestamp", time_unit="us")

    normalized = normalize_timestamps(original, cfg)
    sampling = SamplingAnalyzer().analyze(normalized["time_ms"].to_numpy(), cfg)

    assert normalized["time_ms"].tolist() == [10.0, 20.0, 30.0]
    assert "time_ms" not in original
    assert sampling.dt_ms == 10.0
    assert sampling.hz_measured == 100.0


def test_samples_boundary_handles_endpoints_and_computes_one_sample():
    assert find_single_eye_endpoints(np.array([False, True, True, False]), 0, 3) == (
        1,
        2,
    )

    context = VelocityContext(0, 0, 10, 0, 0, 0, 600)
    result = VelocitySampleComputer.compute_sample(
        context, 10.0, Olsen2DApproximation()
    )

    expected = math.degrees(math.atan(10 / 600)) / 0.01
    assert result.velocity_deg_per_sec == round(expected, 2)
    assert result.dt_ms == 10.0


def test_velocity_facade_reexports_historical_internal_imports():
    assert facade_make_window_selector is make_window_selector
    assert facade_sampling_analyzer is SamplingAnalyzer
    assert facade_sample_computer is VelocitySampleComputer
