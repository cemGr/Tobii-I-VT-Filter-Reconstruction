"""Extract Tobii TSV exports into a slim IVT-friendly schema."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

RAW_COLUMNS: Dict[str, str] = {
    "time_ms": "Recording timestamp [ms]",
    "gaze_left_x_mm": "Gaze point left X [DACS mm]",
    "gaze_left_y_mm": "Gaze point left Y [DACS mm]",
    "gaze_right_x_mm": "Gaze point right X [DACS mm]",
    "gaze_right_y_mm": "Gaze point right Y [DACS mm]",
    "gaze_left_x_px": "Gaze point left X [DACS px]",
    "gaze_left_y_px": "Gaze point left Y [DACS px]",
    "gaze_right_x_px": "Gaze point right X [DACS px]",
    "gaze_right_y_px": "Gaze point right Y [DACS px]",
    "validity_left": "Validity left",
    "validity_right": "Validity right",
    "eye_left_x_mm": "Eye position left X [DACS mm]",
    "eye_left_y_mm": "Eye position left Y [DACS mm]",
    "eye_left_z_mm": "Eye position left Z [DACS mm]",
    "eye_right_x_mm": "Eye position right X [DACS mm]",
    "eye_right_y_mm": "Eye position right Y [DACS mm]",
    "eye_right_z_mm": "Eye position right Z [DACS mm]",
    "gt_event_type": "Eye movement type",
    "gt_event_index": "Eye movement type index",
}


class TobiiTSVExtractor:
    """Slim down Tobii exports to the minimal IVT schema."""

    def __init__(self, raw_columns: Dict[str, str] | None = None) -> None:
        self.raw_columns = raw_columns or RAW_COLUMNS

    def convert(self, input_path: str, output_path: str) -> None:
        input_path = Path(input_path)
        wanted = set(self.raw_columns.values()) | {"Sensor"}
        df = pd.read_csv(
            input_path,
            sep="\t",
            decimal=",",
            low_memory=False,
            usecols=lambda c: c in wanted,
        )

        if "Sensor" in df.columns:
            df = df[df["Sensor"] == "Eye Tracker"]

        slim = pd.DataFrame()
        for new_name, old_name in self.raw_columns.items():
            slim[new_name] = df[old_name] if old_name in df.columns else pd.NA

        slim = slim.sort_values("time_ms").reset_index(drop=True)
        slim.to_csv(output_path, sep="\t", index=False, decimal=",")


def convert_tobii_tsv_to_ivt_tsv(input_path: str, output_path: str) -> None:
    TobiiTSVExtractor().convert(input_path, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert Tobii TSV exports to slim IVT TSV")
    parser.add_argument("--input", required=True, help="Raw Tobii TSV path")
    parser.add_argument("--output", required=True, help="Output path for slim TSV")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    convert_tobii_tsv_to_ivt_tsv(args.input, args.output)
