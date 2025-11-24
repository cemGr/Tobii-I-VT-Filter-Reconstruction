import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
from ivt.analyzer import IVTAnalyzer, PlotConfig


def test_plot_from_file_creates_image(tmp_path):
    df = pd.DataFrame(
        {
            "time_ms": [0, 10, 20, 30],
            "velocity_deg_per_sec": [0.0, 5.0, 15.0, 25.0],
            "combined_x_px": [100.0, 101.0, 102.0, 103.0],
            "ivt_event_index": [0, 0, 1, 1],
        }
    )
    tsv_path = tmp_path / "ivt.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    out_path = tmp_path / "plot.png"
    IVTAnalyzer().plot_from_file(tsv_path, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_requires_time_and_velocity_columns():
    df = pd.DataFrame({"velocity_deg_per_sec": [1, 2, 3], "combined_x_px": [1, 2, 3]})
    analyzer = IVTAnalyzer()

    with pytest.raises(ValueError):
        analyzer.plot(df)


def test_plot_respects_custom_threshold(tmp_path):
    df = pd.DataFrame(
        {
            "time_ms": [0, 10],
            "velocity_deg_per_sec": [1.0, 2.0],
            "combined_x_px": [50.0, 55.0],
            "ivt_event_index": [0, 1],
        }
    )
    out_path = tmp_path / "plot.png"
    analyzer = IVTAnalyzer(
        PlotConfig(threshold_deg_per_sec=10, show_event_index=True, figsize=(4.0, 3.0), dpi=150)
    )
    analyzer.plot(df, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_can_show(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "time_ms": [0, 10],
            "velocity_deg_per_sec": [1.0, 2.0],
            "combined_x_px": [50.0, 55.0],
        }
    )

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))

    out_path = tmp_path / "plot.png"
    analyzer = IVTAnalyzer(PlotConfig(show=True))
    analyzer.plot(df, out_path)

    assert out_path.exists()
    assert called == [True]
