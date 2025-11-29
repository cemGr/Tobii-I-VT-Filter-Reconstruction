from __future__ import annotations

from pathlib import Path
import pandas as pd

# Mapping from slim TSV column names to original Tobii TSV headers.
RAW_COLUMNS = {
    # timing
    "time_ms": "Recording timestamp [ms]",

    # gaze on stimulus plane in millimetres (primary for velocity)
    "gaze_left_x_mm": "Gaze point left X [DACS mm]",
    "gaze_left_y_mm": "Gaze point left Y [DACS mm]",
    "gaze_right_x_mm": "Gaze point right X [DACS mm]",
    "gaze_right_y_mm": "Gaze point right Y [DACS mm]",

    # gaze in pixels (optional, for plotting/debugging)
    "gaze_left_x_px": "Gaze point left X [DACS px]",
    "gaze_left_y_px": "Gaze point left Y [DACS px]",
    "gaze_right_x_px": "Gaze point right X [DACS px]",
    "gaze_right_y_px": "Gaze point right Y [DACS px]",

    # validity per eye
    "validity_left": "Validity left",
    "validity_right": "Validity right",

    # eye position in mm (for eye–screen distance and optional analyses)
    "eye_left_x_mm": "Eye position left X [DACS mm]",
    "eye_left_y_mm": "Eye position left Y [DACS mm]",
    "eye_left_z_mm": "Eye position left Z [DACS mm]",
    "eye_right_x_mm": "Eye position right X [DACS mm]",
    "eye_right_y_mm": "Eye position right Y [DACS mm]",
    "eye_right_z_mm": "Eye position right Z [DACS mm]",

    # ground-truth events (for evaluation)
    "gt_event_type": "Eye movement type",
    "gt_event_index": "Eye movement type index",
}


def convert_tobii_tsv_to_ivt_tsv(input_path: str | Path, output_path: str | Path) -> None:
    """Convert a full Tobii Pro Lab TSV export to a slim IVT TSV.

    The slim TSV keeps only the columns needed for:
      - Olsen-style velocity computation (using gaze in mm),
      - I-VT threshold classification,
      - evaluation against ground-truth events.

    Notes
    -----
    - Reads with sep="\t" and decimal="," because Tobii exports
      often use a comma as decimal separator.
    - Filters to Sensor == "Eye Tracker" if a 'Sensor' column exists.
    - Writes again with sep="\t" and decimal="," so that downstream
      code can safely read with decimal=",".
    """
    in_path = Path(input_path)
    out_path = Path(output_path)

    df = pd.read_csv(in_path, sep="\t", decimal=",", low_memory=False)

    # Keep only eye-tracker rows if Sensor column is present
    if "Sensor" in df.columns:
        df = df[df["Sensor"] == "Eye Tracker"]

    # Build slim DataFrame with the desired schema
    slim = pd.DataFrame()

    for new_name, old_name in RAW_COLUMNS.items():
        if old_name in df.columns:
            slim[new_name] = df[old_name]
        else:
            # Column missing in this export; fill with NA so the schema stays stable
            slim[new_name] = pd.NA

    # Sort by time just to be sure
    if "time_ms" in slim.columns:
        slim = slim.sort_values("time_ms").reset_index(drop=True)

    slim.to_csv(out_path, sep="\t", index=False, decimal=",")


if __name__ == "__main__":
    # Example call – adapt paths as needed
    convert_tobii_tsv_to_ivt_tsv(
        "IVT30.tsv",
        "IVT30_input.tsv",
    )

