from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict
import pandas as pd

from ivt_filter.utils.sampling import sort_by_time_with_source_row_id


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

# Timestamp headers that can be detected in Tobii exports.  Recording timestamps
# are preferred when a manual unit override requires choosing one source column.
TIMESTAMP_ALTERNATIVES = [
    ("Recording timestamp [ms]", "ms"),
    ("Recording timestamp [μs]", "us"),
    ("Recording timestamp [us]", "us"),
    ("Recording timestamp [ns]", "ns"),
    ("Eyetracker timestamp [μs]", "us"),
    ("Eyetracker timestamp [us]", "us"),
]


class TimestampUnitDetector:
    """Detect timestamp columns and convert their values between supported units."""

    def detect_columns(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """Return the first native millisecond and microsecond timestamp columns."""
        native_ms_col = self._find_column(df, "ms")
        native_us_col = self._find_column(df, "us")
        return native_ms_col, native_us_col

    def detect_column(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        """Return the preferred timestamp column and the unit declared by its header."""
        for column_name, unit in TIMESTAMP_ALTERNATIVES:
            if column_name in df.columns:
                return column_name, unit
        return None, None

    def _find_column(self, df: pd.DataFrame, unit: str) -> Optional[str]:
        """Return the first timestamp column whose header declares ``unit``."""
        for column_name, column_unit in TIMESTAMP_ALTERNATIVES:
            if column_unit == unit and column_name in df.columns:
                return column_name
        return None

    def convert(self, values: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
        """Convert timestamp ``values`` from one supported unit to another."""
        scale_in_nanoseconds = {"ms": 1_000_000, "us": 1_000, "ns": 1}
        try:
            scale = scale_in_nanoseconds[from_unit] / scale_in_nanoseconds[to_unit]
        except KeyError as exc:
            raise ValueError(
                "timestamp unit must be one of: 'auto', 'ms', 'us', 'ns'"
            ) from exc
        return values * scale


class TobiiDataExtractor:
    """Extracts IVT-relevant data from Tobii Pro Lab TSV exports.
    
    Responsibilities:
        - Read Tobii TSV files with correct format settings
        - Filter eye tracker data
        - Detect timestamps and populate millisecond and microsecond columns
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
        unit_override: str,
    ) -> pd.DataFrame:
        """Populate ``time_ms`` and ``time_us`` using detection or an explicit unit.

        In ``auto`` mode, native millisecond and microsecond timestamps are
        preserved when both exist.  When only one timestamp exists, the other
        output column is derived from it.  For an explicit ``ms``, ``us``, or
        ``ns`` override, the preferred detected source column is interpreted as
        that unit and converted into both output units.
        """
        if unit_override not in {"auto", "ms", "us", "ns"}:
            raise ValueError("timestamp unit must be one of: 'auto', 'ms', 'us', 'ns'")

        if unit_override == "auto":
            native_ms_col, native_us_col = self.timestamp_detector.detect_columns(source)
            if native_ms_col is not None or native_us_col is not None:
                if native_ms_col is not None:
                    target["time_ms"] = source[native_ms_col].copy()
                    print(f"[Extractor] Using {native_ms_col} for time_ms")
                else:
                    target["time_ms"] = self.timestamp_detector.convert(
                        source[native_us_col], "us", "ms"
                    )
                    print(f"[Extractor] Deriving time_ms from {native_us_col}")

                if native_us_col is not None:
                    target["time_us"] = source[native_us_col].copy()
                    print(f"[Extractor] Using {native_us_col} for time_us")
                else:
                    target["time_us"] = self.timestamp_detector.convert(
                        source[native_ms_col], "ms", "us"
                    )
                    print(f"[Extractor] Deriving time_us from {native_ms_col}")
                return target

            timestamp_col, detected_unit = self.timestamp_detector.detect_column(source)
            if timestamp_col is None or detected_unit is None:
                raise ValueError("No usable timestamp column found")
            unit_to_use = detected_unit
            print(f"[Extractor] Auto-detected {timestamp_col} as {unit_to_use}")
        else:
            timestamp_col, _ = self.timestamp_detector.detect_column(source)
            if timestamp_col is None:
                raise ValueError("No usable timestamp column found")
            unit_to_use = unit_override
            print(f"[Extractor] Interpreting {timestamp_col} as {unit_to_use}")

        target["time_ms"] = self.timestamp_detector.convert(
            source[timestamp_col], unit_to_use, "ms"
        )
        target["time_us"] = self.timestamp_detector.convert(
            source[timestamp_col], unit_to_use, "us"
        )
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
        """Validate and stably sort timestamps while retaining source-row provenance."""
        return sort_by_time_with_source_row_id(df)

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

