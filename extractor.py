"""Backward-compatible wrapper for the new extractor module."""
from ivt.extractor import TobiiTSVExtractor, convert_tobii_tsv_to_ivt_tsv

__all__ = ["TobiiTSVExtractor", "convert_tobii_tsv_to_ivt_tsv"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Tobii TSV exports to slim IVT TSV")
    parser.add_argument("input", help="Raw Tobii TSV path")
    parser.add_argument("output", help="Output path for slim TSV")
    args = parser.parse_args()

    convert_tobii_tsv_to_ivt_tsv(args.input, args.output)
