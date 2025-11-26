"""Backward-compatible wrapper for the new extractor module."""
from ivt.extractor import (
    TobiiTSVExtractor,
    convert_tobii_tsv_to_ivt_tsv,
    main as extractor_main,
)

__all__ = ["TobiiTSVExtractor", "convert_tobii_tsv_to_ivt_tsv"]


if __name__ == "__main__":
    extractor_main()
