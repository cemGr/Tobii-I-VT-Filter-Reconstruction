import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.processing import velocity_parallel


def test_compute_olsen_velocity_parallel_delegates_to_canonical_implementation(
    monkeypatch,
):
    df = pd.DataFrame({"sample": [1]})
    cfg = OlsenVelocityConfig()
    expected = pd.DataFrame({"velocity_deg_per_sec": [12.3]})
    calls = []

    def canonical(received_df, received_cfg):
        calls.append((received_df, received_cfg))
        return expected

    monkeypatch.setattr(velocity_parallel, "compute_olsen_velocity", canonical)

    with pytest.warns(DeprecationWarning, match="Velocity computation is sequential"):
        result = velocity_parallel.compute_olsen_velocity_parallel(
            df, cfg, n_jobs=8, chunk_size=1000
        )

    assert result is expected
    assert calls == [(df, cfg)]


@pytest.mark.parametrize("sample_count", [1, 50_001])
@pytest.mark.parametrize("parallel", [False, True])
def test_compute_velocity_auto_delegates_to_canonical_implementation(
    monkeypatch, sample_count, parallel
):
    df = pd.DataFrame({"sample": range(sample_count)})
    cfg = OlsenVelocityConfig()
    expected = pd.DataFrame({"velocity_deg_per_sec": [45.6]})
    calls = []

    def canonical(received_df, received_cfg):
        calls.append((received_df, received_cfg))
        return expected

    monkeypatch.setattr(velocity_parallel, "compute_olsen_velocity", canonical)

    result = velocity_parallel.compute_velocity_auto(df, cfg, parallel=parallel, n_jobs=8)

    assert result is expected
    assert calls == [(df, cfg)]
