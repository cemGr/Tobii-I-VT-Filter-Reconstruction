from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import pandas as pd


# Mapping from slim TSV column names to original Tobii TSV headers.
RAW_COLUMNS = {
    "time_ms": "Recording timestamp [ms]",
    "gaze_left_x_mm": "Gaze point left X [DACS mm]",
    "gaze_left_y_mm": "Gaze point left Y [DACS mm]",
    "gaze_right_x_mm": "Gaze point right X [DACS mm]",
    "gaze_right_y_mm": "Gaze point right Y [DACS mm]",
    "gaze_left_x_px": "Gaze point left X [DACS px]",
    "gaze_left_y_px": "Gaze point left Y [DACS px]",
    "gaze_right_x_px": "Gaze point right X [DACS px]",
    "gaze_right_y_px": "Gaze point right Y [DACS px]",
    # Normalized gaze direction vectors (needed for optional ray3d_gaze_dir)
    "gaze_dir_left_x": "Gaze direction left X [DACS norm]",
    "gaze_dir_left_y": "Gaze direction left Y [DACS norm]",
    "gaze_dir_left_z": "Gaze direction left Z [DACS norm]",
    "gaze_dir_right_x": "Gaze direction right X [DACS norm]",
    "gaze_dir_right_y": "Gaze direction right Y [DACS norm]",
    "gaze_dir_right_z": "Gaze direction right Z [DACS norm]",
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
    "pupil_left_mm": "Pupil diameter left [mm]",
    "pupil_right_mm": "Pupil diameter right [mm]",
}

TIMESTAMP_ALTERNATIVES = [
    "Recording timestamp [ms]",
    "Recording timestamp [μs]",
    "Recording timestamp [us]",
    "Recording timestamp [ns]",
]


class TimestampUnitDetector:
    """Detects and converts timestamp units from Tobii exports."""

    def __init__(self, alternatives: list[str] = TIMESTAMP_ALTERNATIVES):
        self.alternatives = alternatives

    def detect_column(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """Find timestamp column and detect its unit.
        
        Returns:
            (column_name, unit) where unit is one of: 'ms', 'us', 'ns', None
        """
        for col_name in self.alternatives:
            if col_name in df.columns:
                unit = self._parse_unit_from_column_name(col_name)
                return col_name, unit
        return None, None

    def _parse_unit_from_column_name(self, col_name: str) -> str:
        """Extract time unit from column name."""
        if "[ms]" in col_name:
            return "ms"
        elif "[μs]" in col_name or "[us]" in col_name:
            return "us"
        elif "[ns]" in col_name:
            return "ns"
        return "ms"

    def convert_to_milliseconds(
        self, 
        values: pd.Series, 
        unit: str
    ) -> pd.Series:
        """Convert timestamps to milliseconds."""
        if unit == "ms":
            return values
        elif unit == "us":
            return values / 1000.0
        elif unit == "ns":
            return values / 1_000_000.0
        else:
            return values


class TobiiDataExtractor:
    """Extracts IVT-relevant data from Tobii Pro Lab TSV exports.
    
    Responsibilities:
        - Read Tobii TSV files with correct format settings
        - Filter eye tracker data
        - Convert timestamps to milliseconds
        - Map columns to standardized schema
        - Write slim TSV for IVT processing
    """

    def __init__(
        self, 
        column_mapping: Dict[str, str] = RAW_COLUMNS,
        timestamp_detector: Optional[TimestampUnitDetector] = None
    ):
        self.column_mapping = column_mapping
        self.timestamp_detector = timestamp_detector or TimestampUnitDetector()

    def extract(
        self,
        input_path: str | Path,
        output_path: str | Path,
        timestamp_unit: str = "auto"
    ) -> None:
        """Extract and convert Tobii TSV to slim IVT format.
        
        Args:
            input_path: Path to input Tobii TSV file
            output_path: Path to output slim TSV file
            timestamp_unit: Unit override ('auto', 'ms', 'us', 'ns')
        """
        df = self._read_data(input_path)
        df = self._filter_sensor(df)
        slim = self._build_slim_dataframe(df, timestamp_unit)
        slim = self._sort_by_time(slim)
        self._write_data(slim, output_path)

    def _read_data(self, path: str | Path) -> pd.DataFrame:
        """Read Tobii TSV with comma as decimal separator."""
        return pd.read_csv(path, sep="\t", decimal=",", low_memory=False)

    def _filter_sensor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only eye tracker rows if Sensor column exists."""
        if "Sensor" in df.columns:
            return df[df["Sensor"] == "Eye Tracker"].copy()
        return df

    def _build_slim_dataframe(
        self, 
        df: pd.DataFrame, 
        timestamp_unit_override: str
    ) -> pd.DataFrame:
        """Build slim DataFrame with standardized column schema."""
        slim = pd.DataFrame()
        
        slim = self._convert_timestamps(df, slim, timestamp_unit_override)
        slim = self._map_columns(df, slim)
        
        return slim

    def _convert_timestamps(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        unit_override: str
    ) -> pd.DataFrame:
        """Detect and convert timestamp column to milliseconds."""
        timestamp_col, detected_unit = self.timestamp_detector.detect_column(source)
        
        if timestamp_col is None:
            print("[Extractor] Warning: No timestamp column found")
            target["time_ms"] = pd.NA
            return target
        
        unit_to_use = unit_override if unit_override != "auto" else detected_unit
        
        if unit_override != "auto":
            print(f"[Extractor] Using manual timestamp unit: {unit_to_use}")
        else:
            print(f"[Extractor] Auto-detected timestamp unit: {unit_to_use}")
        
        converted = self.timestamp_detector.convert_to_milliseconds(
            source[timestamp_col], 
            unit_to_use
        )
        
        if unit_to_use != "ms":
            print(f"[Extractor] Converting from {unit_to_use} to milliseconds")
        
        target["time_ms"] = converted
        return target

    def _map_columns(
        self, 
        source: pd.DataFrame, 
        target: pd.DataFrame
    ) -> pd.DataFrame:
        """Map Tobii columns to standardized names."""
        for new_name, old_name in self.column_mapping.items():
            if new_name == "time_ms":
                continue
            
            if old_name in source.columns:
                target[new_name] = source[old_name]
            else:
                target[new_name] = pd.NA
        
        return target

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort DataFrame by timestamp."""
        if "time_ms" in df.columns:
            return df.sort_values("time_ms").reset_index(drop=True)
        return df

    def _write_data(self, df: pd.DataFrame, path: str | Path) -> None:
        """Write DataFrame to TSV with Tobii format."""
        df.to_csv(path, sep="\t", index=False, decimal=",")


def convert_tobii_tsv_to_ivt_tsv(
    input_path: str | Path, 
    output_path: str | Path,
    timestamp_unit: str = "auto"
) -> None:
    """Legacy wrapper for backward compatibility.
    
    Args:
        input_path: Path to input Tobii TSV file
        output_path: Path to output slim TSV file
        timestamp_unit: Unit of timestamps ('auto', 'ms', 'us', 'ns')
    """
    extractor = TobiiDataExtractor()
    extractor.extract(input_path, output_path, timestamp_unit)


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract and convert Tobii TSV archive to slim IVT format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input Tobii TSV archive file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output slim IVT TSV file"
    )
    parser.add_argument(
        "--timestamp-unit",
        default="auto",
        choices=["auto", "ms", "us", "ns"],
        help="Timestamp unit override (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    try:
        convert_tobii_tsv_to_ivt_tsv(
            args.input,
            args.output,
            timestamp_unit=args.timestamp_unit
        )
        print(f"[Extractor] Successfully wrote output to: {args.output}")
    except Exception as e:
        print(f"[Extractor] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

