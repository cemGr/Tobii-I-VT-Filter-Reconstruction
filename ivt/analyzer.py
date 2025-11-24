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
    gaze_x_column: str = "combined_x_px"
    event_index_column: str = "ivt_event_index"
    show_event_index: bool = False
    figsize: tuple[float, float] = (10.0, 6.0)
    dpi: float | None = None
    tight_layout: bool = True
    show: bool = False

    def ensure_columns(self, columns: Iterable[str]) -> None:
        """Validate that required columns exist."""
        missing = {self.time_column, self.velocity_column, self.gaze_x_column} - set(columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


class IVTAnalyzer:
    """Generate plots for velocity and I-VT classification outputs."""

    def __init__(self, config: PlotConfig | None = None) -> None:
        self.config = config or PlotConfig()

    def plot(self, df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
        """Plot velocity and gaze position; optionally include event indices."""

        self.config.ensure_columns(df.columns)
        cfg = self.config

        try:
            matplotlib = import_module("matplotlib")
            if not cfg.show:
                matplotlib.use("Agg")
            plt = import_module("matplotlib.pyplot")
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency is optional in CI
            raise ModuleNotFoundError(
                "matplotlib is required for plotting; install via `pip install matplotlib`."
            ) from exc

        fig, axes = plt.subplots(
            2 + int(cfg.show_event_index), 1, figsize=cfg.figsize, dpi=cfg.dpi, sharex=True
        )
        ax = list(axes) if hasattr(axes, "__iter__") else [axes]

        ax[0].plot(df[cfg.time_column], df[cfg.velocity_column], label="velocity")
        if cfg.threshold_deg_per_sec is not None:
            ax[0].axhline(cfg.threshold_deg_per_sec, color="red", linestyle="--", label="threshold")
        ax[0].set_ylabel("deg/sec")
        ax[0].legend()

        ax[1].plot(df[cfg.time_column], df[cfg.gaze_x_column], label="gaze x")
        ax[1].set_ylabel("gaze x (px)")
        ax[1].legend()

        next_idx = 2
        if cfg.show_event_index:
            if cfg.event_index_column in df.columns:
                ax[next_idx].plot(
                    df[cfg.time_column], df[cfg.event_index_column], drawstyle="steps-post"
                )
                ax[next_idx].set_ylabel("event index")
            else:
                ax[next_idx].text(
                    0.5,
                    0.5,
                    "No event index column",
                    ha="center",
                    va="center",
                    transform=ax[next_idx].transAxes,
                )
                ax[next_idx].set_ylabel("event index")
            next_idx += 1

        ax[next_idx - 1].set_xlabel("time (ms)")

        if cfg.tight_layout:
            plt.tight_layout()

        output_path = Path(output_path or "ivt_plot.png")
        fig.savefig(output_path)
        if cfg.show:  # pragma: no cover - UI-driven choice
            plt.show()
        plt.close(fig)
        return output_path

    def plot_from_file(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        """Load TSV and plot the IVT outputs."""

        df = pd.read_csv(input_path, sep="\t")
        return self.plot(df, output_path=output_path)


__all__ = ["IVTAnalyzer", "PlotConfig"]
