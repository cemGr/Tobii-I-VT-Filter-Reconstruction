#!/usr/bin/env python3
"""Batch-extract Tobii TSV exports and generate configs.json for IVT benchmark (study A1)."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = Path("/data/cem/Downloads/cem(danke)")
OUTPUT_DIR = REPO_ROOT / "test_data" / "study_a1"
XLSX_PATH = Path("/home/cem/Documents/Gitprojekt/IVT-Thesis/filter_settings_table.xlsx")
EXTRACTOR = REPO_ROOT / "extractor.py"
GITIGNORE = REPO_ROOT / ".gitignore"

SKIP_FILES = {
    "A1EyeAverageW20T30.tsv",       # duplicate of baseline (different export name)
    "A1averageWindow50T20.tsv",      # confounded: W=50 and T=20 both non-default
    "A1averageWindow50T50.tsv",      # confounded
    "A1averageWindow50T75.tsv",      # confounded
    "A1averageWindow50T100.tsv",     # confounded
}

EMDASH = "\u2013"  # U+2013 en-dash used as placeholder in xlsx

COL_MAP = {
    "Eye Mode":                  "eye_mode",
    "Window (ms)":               "window_length_ms",
    "Threshold (deg/s)":         "velocity_threshold_deg_per_sec",
    "Smoothing Mode":            "smoothing_mode",
    "Smoothing Samples":         "smoothing_window_samples",
    "Interpolation":             "gap_fill_enabled",
    "Interp Max Gap (ms)":       "gap_fill_max_gap_ms",
    "Merge Adjacent":            "merge_adjacent_fixations",
    "Merge Time Gap (ms)":       "merge_max_time_gap_ms",
    "Merge Angle (\u00b0)":      "merge_max_angle_deg",
    "Discard Short":             "discard_short_fixations",
    "Discard Min Duration (ms)": "min_fixation_duration_ms",
}


def convert_value(val):
    if val == EMDASH or (isinstance(val, float) and pd.isna(val)):
        return None
    if val == "on":
        return True
    if val == "off":
        return False
    if val == "moving_avg":
        return "moving_average"
    if isinstance(val, (int, float)):
        return val
    return str(val)


def task1_setup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[1] Output directory: {OUTPUT_DIR}")

    gitignore_line = "test_data/study_a1/"
    text = GITIGNORE.read_text()
    if gitignore_line not in text:
        GITIGNORE.write_text(text.rstrip("\n") + f"\n{gitignore_line}\n")
        print(f"[1] Added '{gitignore_line}' to .gitignore")
    else:
        print(f"[1] '{gitignore_line}' already in .gitignore")


def task2_extract():
    source_files = sorted(SOURCE_DIR.glob("*.tsv"))
    extracted = []
    skipped = []

    print(f"\n[2] Extracting from {SOURCE_DIR} ({len(source_files)} .tsv files found)")
    for src in source_files:
        if src.name in SKIP_FILES:
            skipped.append(src.name)
            continue
        dest = OUTPUT_DIR / src.name
        print(f"    {src.name} ...", end=" ", flush=True)
        result = subprocess.run(
            [
                sys.executable, str(EXTRACTOR),
                "--input", str(src),
                "--output", str(dest),
                "--timestamp-unit", "us",
                "--keep-empty-stimulus",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"FAILED\n{result.stderr}")
            raise RuntimeError(f"Extraction failed for {src.name}")
        print("OK")
        extracted.append(src.name)

    return extracted, skipped


def task3_generate_configs(extracted_set: set) -> dict:
    df = pd.read_excel(XLSX_PATH, sheet_name="Filter Settings")

    configs: dict = {"files": {}, "skip": []}

    for _, row in df.iterrows():
        fname = str(row["Filename"])
        if fname not in extracted_set:
            continue

        entry: dict = {}
        for col, key in COL_MAP.items():
            val = convert_value(row[col])
            if val is not None:
                entry[key] = val

        configs["files"][fname] = entry

    out_path = OUTPUT_DIR / "configs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

    print(f"\n[3] Written {out_path} ({len(configs['files'])} entries)")
    return configs


def task4_verify(extracted: list, skipped: list, configs: dict):
    print("\n[4] Verification")

    tsv_files = list(OUTPUT_DIR.glob("*.tsv"))
    n_tsv = len(tsv_files)
    n_cfg = len(configs["files"])
    status = "OK" if n_tsv == n_cfg else "MISMATCH"
    print(f"    Files in output dir : {n_tsv}  |  Entries in configs.json : {n_cfg}  [{status}]")
    assert n_tsv == n_cfg, f"Count mismatch: {n_tsv} files vs {n_cfg} config entries"

    spot_checks = [
        "A1averageWindow20T30.tsv",
        "A1averageWindow20T30SmoothMed3.tsv",
        "A1averageWindow20T30Interpolation75ms.tsv",
    ]
    for fname in spot_checks:
        path = OUTPUT_DIR / fname
        if not path.exists():
            print(f"    MISSING for spot-check: {fname}")
            continue
        df = pd.read_csv(path, sep="\t", nrows=500)
        has_col = "gt_event_type" in df.columns
        non_empty = df["gt_event_type"].notna().any() if has_col else False
        ok = has_col and non_empty
        print(f"    {'OK' if ok else 'FAIL'}: {fname} ({len(df)} rows sampled, gt_event_type={'present' if has_col else 'MISSING'})")

    with open(OUTPUT_DIR / "configs.json", encoding="utf-8") as f:
        json.load(f)
    print("    configs.json: valid JSON")

    print("\n=== Summary ===")
    print(f"  Extracted : {len(extracted)} files")
    print(f"  In configs: {n_cfg} entries")
    print(f"  Skipped   : {len(skipped)} files")
    for name in sorted(skipped):
        reason = "confounded (W and T both non-default)" if "Window50" in name else "duplicate of baseline"
        print(f"    - {name}  ({reason})")


if __name__ == "__main__":
    task1_setup()
    extracted, skipped = task2_extract()
    configs = task3_generate_configs(set(extracted))
    task4_verify(extracted, skipped, configs)
