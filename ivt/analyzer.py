"""Visualization helpers for IVT outputs."""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class PlotConfig:
    """Configuration for IVT plot generation."""

    threshold_deg_per_sec: float | None = None
    velocity_column: str = "velocity_deg_per_sec"
    time_column: str = "time_ms"
    event_index_column: str = "ivt_event_index"
    figsize: tuple[float, float] = (10.0, 6.0)
    tight_layout: bool = True

    def ensure_columns(self, columns: Iterable[str]) -> None:
        """Validate that required columns exist."""
        missing = {self.time_column, self.velocity_column} - set(columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


class IVTAnalyzer:
    """Generate plots for velocity and I-VT classification outputs."""

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()

    def plot(self, df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
        """Plot velocity and event indices; save to the provided path."""

        self.config.ensure_columns(df.columns)
        cfg = self.config

        try:
            matplotlib = import_module("matplotlib")
            matplotlib.use("Agg")
            plt = import_module("matplotlib.pyplot")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency is optional in CI
            raise ModuleNotFoundError(
                "matplotlib is required for plotting; install via `pip install matplotlib`."
            ) from exc

        fig, ax = plt.subplots(2, 1, figsize=cfg.figsize, sharex=True)

        ax[0].plot(df[cfg.time_column], df[cfg.velocity_column], label="velocity")
        if cfg.threshold_deg_per_sec is not None:
            ax[0].axhline(cfg.threshold_deg_per_sec, color="red", linestyle="--", label="threshold")
        ax[0].set_ylabel("deg/sec")
        ax[0].legend()

        if cfg.event_index_column in df.columns:
            ax[1].plot(df[cfg.time_column], df[cfg.event_index_column], drawstyle="steps-post")
            ax[1].set_ylabel("event index")
        else:
            ax[1].text(0.5, 0.5, "No event index column", ha="center", va="center", transform=ax[1].transAxes)
            ax[1].set_ylabel("event index")
        ax[1].set_xlabel("time (ms)")

        if cfg.tight_layout:
            plt.tight_layout()

        output_path = Path(output_path or "ivt_plot.png")
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def plot_from_file(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        """Load TSV and plot the IVT outputs."""

        df = pd.read_csv(input_path, sep="\t")
        return self.plot(df, output_path=output_path)


__all__ = ["IVTAnalyzer", "PlotConfig"]
