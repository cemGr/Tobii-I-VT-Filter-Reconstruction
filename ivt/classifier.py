"""I-VT velocity threshold classifier."""
from __future__ import annotations

import math
from typing import Optional

import pandas as pd

from .config import IVTClassifierConfig


class IVTClassifier:
    def __init__(self, config: Optional[IVTClassifierConfig] = None) -> None:
        self.config = config or IVTClassifierConfig()

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if "velocity_deg_per_sec" not in df.columns:
            raise ValueError("DataFrame must contain 'velocity_deg_per_sec' before classification.")

        df = df.copy()
        df["velocity_deg_per_sec"] = pd.to_numeric(df["velocity_deg_per_sec"], errors="coerce")

        def classify_sample(v: float) -> str:
            if v is None or not math.isfinite(v):
                return "Unclassified"
            if v > cfg.velocity_threshold_deg_per_sec:
                return "Saccade"
            return "Fixation"

        df["ivt_sample_type"] = df["velocity_deg_per_sec"].apply(classify_sample)

        event_types = []
        event_indices = []
        current_type: Optional[str] = None
        current_index = 0

        for label in df["ivt_sample_type"]:
            if label not in ("Fixation", "Saccade"):
                event_types.append(label)
                event_indices.append(None)
                current_type = None
                continue

            if label != current_type:
                current_index += 1
                current_type = label

            event_types.append(label)
            event_indices.append(current_index)

        df["ivt_event_type"] = event_types
        df["ivt_event_index"] = pd.Series(event_indices, dtype="Int64")
        return df

    def classify_from_file(self, input_path: str, output_path: str) -> pd.DataFrame:
        df = pd.read_csv(input_path, sep="\t")
        result = self.classify(df)
        result.to_csv(output_path, sep="\t", index=False)
        return result


def apply_ivt_classifier(df: pd.DataFrame, cfg: Optional[IVTClassifierConfig] = None) -> pd.DataFrame:
    return IVTClassifier(cfg).classify(df)
