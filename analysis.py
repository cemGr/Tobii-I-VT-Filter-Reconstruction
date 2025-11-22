from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt

from converter import load_tobii_tsv_to_recording, prepare_recording_for_velocity
from domain import IVTVelocityConfig
from ivt_velocity import IVTVelocityCalculator


def plot_velocity_from_tobii_tsv(path: str, window_length_ms: float = 20.0) -> None:
    recording = load_tobii_tsv_to_recording(path)
    prepare_recording_for_velocity(recording)

    config = IVTVelocityConfig(window_length_ms=window_length_ms)
    calculator = IVTVelocityCalculator(config)
    calculator.compute_velocities(recording)

    times: List[float] = []
    velocities: List[float] = []
    for sample in recording.samples:
        if sample.velocity_deg_per_sec is not None:
            times.append(sample.time_ms)
            velocities.append(sample.velocity_deg_per_sec)

    plt.figure()
    plt.plot(times, velocities)
    plt.xlabel("Time (ms)")
    plt.ylabel("Velocity (deg/s)")
    plt.title(f"Velocity over time (window={window_length_ms} ms) for {os.path.basename(path)}")
    plt.tight_layout()
    plt.show()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot IVT velocity from Tobii TSV export")
    parser.add_argument("--input", required=True, help="Path to Tobii TSV export")
    parser.add_argument("--window", type=float, default=20.0, help="Window length in milliseconds")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    plot_velocity_from_tobii_tsv(args.input, args.window)
