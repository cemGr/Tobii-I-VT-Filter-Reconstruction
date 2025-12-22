# ivt_filter/io.py
from __future__ import annotations

from typing import Optional
import pandas as pd


def read_tsv(path: str) -> pd.DataFrame:
    """
    TSV Datei mit Tobii Export einlesen.

    - Tab als Separator
    - Komma als Dezimaltrennzeichen
    """
    return pd.read_csv(path, sep="\t", decimal=",", low_memory=False)


def write_tsv(df: pd.DataFrame, path: str) -> None:
    """
    DataFrame als TSV schreiben (Tobii kompatibel).
    """
    df.to_csv(path, sep="\t", index=False, decimal=",")
