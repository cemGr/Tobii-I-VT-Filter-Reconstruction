from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import pandas as pd


# Mapping from slim TSV column names to original Tobii TSV headers.
RAW_COLUMNS = {
    "time_ms": "Recording timestamp [ms]",
    # Optional microsecond precision export
    # Will be populated automatically during conversion
    # even if the source file only contains millisecond timestamps.
    "time_us": "Recording timestamp [μs]",
    "presented_stimulus_name": "Presented Stimulus name",
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

# Native timestamp columns - no conversion needed
RECORDING_TIMESTAMP_MS_ALTERNATIVES = [
    "Recording timestamp [ms]",
]

EYETRACKER_TIMESTAMP_US_ALTERNATIVES = [
    "Eyetracker timestamp [μs]",
    "Eyetracker timestamp [us]",
]


class TimestampUnitDetector:
    """Detects and extracts timestamp columns from Tobii exports (no conversion)."""

    def detect_columns(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """Find recording (ms) and eyetracker (us) timestamp columns.
        
        Returns:
            (recording_ms_col, eyetracker_us_col) where either can be None
        """
        recording_col = None
        eyetracker_col = None
        
        for col_name in RECORDING_TIMESTAMP_MS_ALTERNATIVES:
            if col_name in df.columns:
                recording_col = col_name
                break
        
        for col_name in EYETRACKER_TIMESTAMP_US_ALTERNATIVES:
            if col_name in df.columns:
                eyetracker_col = col_name
                break
        
        return recording_col, eyetracker_col


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
        timestamp_unit: str = "auto",
        exclude_calibration: bool = True,
        exclude_empty_stimulus: bool = True,
    ) -> None:
        """Extract and convert Tobii TSV to slim IVT format.
        
        Args:
            input_path: Path to input Tobii TSV file
            output_path: Path to output slim TSV file
            timestamp_unit: Unit override ('auto', 'ms', 'us', 'ns')
            exclude_calibration: Drop calibration samples if possible (default: True)
            exclude_empty_stimulus: Drop rows without presented stimulus name (default: True)
        """
        df = self._read_data(input_path)
        df = self._filter_sensor(df)
        df = self._filter_calibration(df, exclude_calibration)
        df = self._filter_empty_stimulus(df, exclude_empty_stimulus)
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

    def _filter_calibration(
        self, df: pd.DataFrame, enabled: bool = True
    ) -> pd.DataFrame:
        """Drop calibration samples using Event/Stimulus columns when available."""
        if not enabled:
            return df

        masks = []
        if "Event" in df.columns:
            masks.append(~df["Event"].astype(str).str.contains("calibration", case=False, na=False))
        if "Presented Stimulus name" in df.columns:
            masks.append(~df["Presented Stimulus name"].astype(str).str.contains("calibration", case=False, na=False))

        if not masks:
            return df

        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask &= m

        removed = len(df) - combined_mask.sum()
        if removed > 0:
            print(f"[Extractor] Dropped {removed} calibration samples")
        return df[combined_mask].copy()

    def _filter_empty_stimulus(
        self, df: pd.DataFrame, enabled: bool = True
    ) -> pd.DataFrame:
        """Drop rows where presented_stimulus_name is missing/empty."""
        if not enabled:
            return df

        col = "Presented Stimulus name"
        if col not in df.columns:
            return df

        mask = df[col].notna() & (df[col].astype(str).str.strip() != "")
        removed = len(df) - mask.sum()
        if removed > 0:
            print(f"[Extractor] Dropped {removed} rows without presented stimulus name")
        return df[mask].copy()

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
        unit_override: str,  # Not used anymore, kept for API compatibility
    ) -> pd.DataFrame:
        """Extract native timestamp columns without conversion.
        
        - time_ms: From Recording timestamp [ms] (native milliseconds)
        - time_us: From Eyetracker timestamp [μs] (native microseconds)
        """
        recording_col, eyetracker_col = self.timestamp_detector.detect_columns(source)

        # Extract time_ms from Recording timestamp [ms]
        if recording_col is not None:
            target["time_ms"] = source[recording_col].copy()
            print(f"[Extractor] Using Recording timestamp [ms] for time_ms")
        else:
            print("[Extractor] Warning: Recording timestamp [ms] not found")
            target["time_ms"] = pd.NA

        # Extract time_us from Eyetracker timestamp [μs]
        if eyetracker_col is not None:
            target["time_us"] = source[eyetracker_col].copy()
            print(f"[Extractor] Using Eyetracker timestamp [μs] for time_us")
        else:
            print("[Extractor] Warning: Eyetracker timestamp [μs] not found")
            target["time_us"] = pd.NA

        return target

    def _map_columns(
        self, 
        source: pd.DataFrame, 
        target: pd.DataFrame
    ) -> pd.DataFrame:
        """Map Tobii columns to standardized names."""
        for new_name, old_name in self.column_mapping.items():
            if new_name in {"time_ms", "time_us"}:
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
    timestamp_unit: str = "auto",
    exclude_calibration: bool = True,
    exclude_empty_stimulus: bool = True,
) -> None:
    """Legacy wrapper for backward compatibility.
    
    Args:
        input_path: Path to input Tobii TSV file
        output_path: Path to output slim TSV file
        timestamp_unit: Unit of timestamps ('auto', 'ms', 'us', 'ns')
    """
    extractor = TobiiDataExtractor()
    extractor.extract(
        input_path,
        output_path,
        timestamp_unit,
        exclude_calibration,
        exclude_empty_stimulus,
    )


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
    parser.add_argument(
        "--keep-calibration",
        action="store_true",
        help="Keep calibration samples instead of dropping them during extraction."
    )
    parser.add_argument(
        "--keep-empty-stimulus",
        action="store_true",
        help="Keep rows without presented stimulus name instead of dropping them."
    )
    
    args = parser.parse_args()
    
    try:
        convert_tobii_tsv_to_ivt_tsv(
            args.input,
            args.output,
            timestamp_unit=args.timestamp_unit,
            exclude_calibration=not args.keep_calibration,
            exclude_empty_stimulus=not args.keep_empty_stimulus,
        )
        print(f"[Extractor] Successfully wrote output to: {args.output}")
    except Exception as e:
        print(f"[Extractor] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

