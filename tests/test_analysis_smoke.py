from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from analysis import plot_velocity_from_tobii_tsv


def test_plot_velocity_smoke() -> None:
    plot_velocity_from_tobii_tsv("I-VT-normal Data export_short.tsv", window_length_ms=20.0)
