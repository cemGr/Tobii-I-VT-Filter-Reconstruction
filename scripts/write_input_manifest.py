#!/usr/bin/env python3
"""Write a SHA-256 provenance manifest of the benchmark input data.

The manifest pins the exact bytes of the local Tobii study exports and external
benchmark data so the paper's numbers can be traced to a fixed data snapshot.
Most of these files are gitignored (large TSV / .mat), so the manifest is the
only record of which bytes produced a given result.

Usage:
    python scripts/write_input_manifest.py            # write the manifest
    python scripts/write_input_manifest.py --check    # verify present files

Manifest line format (tab-separated, sorted by path for determinism):
    <sha256>\t<size_bytes>\t<relative/path>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Data roots that feed the benchmarks. Missing roots are skipped silently so the
# script works in a fresh checkout that has not pulled the gitignored data yet.
DEFAULT_ROOTS = (
    "test_data/inputs",
    "test_data/external",
    "data/raw",
)
DEFAULT_MANIFEST = "results/input_manifest.sha256"

_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(roots: tuple[str, ...], repo_root: Path) -> list[Path]:
    """Return all files under the given roots, sorted, relative paths resolved."""
    found: list[Path] = []
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        found.extend(p for p in base.rglob("*") if p.is_file())
    return sorted(found)


def build_manifest(roots: tuple[str, ...], repo_root: Path) -> list[tuple[str, int, str]]:
    """Return (sha256, size_bytes, relpath) rows for every input file."""
    rows: list[tuple[str, int, str]] = []
    for path in iter_files(roots, repo_root):
        rel = path.relative_to(repo_root).as_posix()
        rows.append((sha256_file(path), path.stat().st_size, rel))
    return rows


def write_manifest(rows: list[tuple[str, int, str]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{sha}\t{size}\t{rel}" for sha, size, rel in rows]
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_manifest(manifest_path: Path) -> list[tuple[str, int, str]]:
    """Parse a manifest file back into (sha256, size_bytes, relpath) rows."""
    rows: list[tuple[str, int, str]] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sha, size, rel = line.split("\t")
        rows.append((sha, int(size), rel))
    return rows


def verify_manifest(
    manifest_path: Path,
    repo_root: Path,
    *,
    check_hash: bool = False,
) -> list[str]:
    """Return a list of human-readable mismatches for files present on disk.

    Files listed in the manifest but absent on disk are ignored (the data is
    gitignored and may not be present in CI). By default only file sizes are
    checked, which is cheap; pass ``check_hash=True`` to re-hash present files.
    """
    problems: list[str] = []
    for sha, size, rel in parse_manifest(manifest_path):
        path = repo_root / rel
        if not path.exists():
            continue
        actual_size = path.stat().st_size
        if actual_size != size:
            problems.append(f"size mismatch: {rel} ({actual_size} != {size})")
            continue
        if check_hash and sha256_file(path) != sha:
            problems.append(f"hash mismatch: {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    parser.add_argument("--output", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify present files against an existing manifest instead of writing",
    )
    parser.add_argument(
        "--check-hash",
        action="store_true",
        help="with --check, also re-hash present files (slow)",
    )
    args = parser.parse_args(argv)

    manifest_path = REPO_ROOT / args.output
    roots = tuple(args.roots)

    if args.check:
        if not manifest_path.exists():
            print(f"manifest not found: {manifest_path}", file=sys.stderr)
            return 2
        problems = verify_manifest(manifest_path, REPO_ROOT, check_hash=args.check_hash)
        if problems:
            print(f"{len(problems)} mismatch(es):", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        print("manifest OK (present files match)")
        return 0

    rows = build_manifest(roots, REPO_ROOT)
    write_manifest(rows, manifest_path)
    total_bytes = sum(size for _, size, _ in rows)
    print(f"wrote {len(rows)} entries ({total_bytes / 1e6:.1f} MB) to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
