"""WS-D: input-data provenance manifest round-trips and verifies."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = REPO_ROOT / "scripts" / "write_input_manifest.py"
_MANIFEST = REPO_ROOT / "results" / "input_manifest.sha256"


def _load_module():
    spec = importlib.util.spec_from_file_location("write_input_manifest", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_and_verify_round_trip(tmp_path) -> None:
    module = _load_module()
    root = tmp_path
    (root / "data").mkdir()
    (root / "data" / "a.tsv").write_text("hello\n", encoding="utf-8")
    (root / "data" / "b.tsv").write_text("world\n", encoding="utf-8")

    rows = module.build_manifest(("data",), root)
    assert len(rows) == 2
    manifest = root / "manifest.sha256"
    module.write_manifest(rows, manifest)

    # present files match (size + hash)
    assert module.verify_manifest(manifest, root, check_hash=True) == []

    # tampering with a file is detected
    (root / "data" / "a.tsv").write_text("tampered\n", encoding="utf-8")
    problems = module.verify_manifest(manifest, root, check_hash=True)
    assert any("a.tsv" in p for p in problems)


def test_committed_manifest_matches_present_files() -> None:
    """If a manifest exists, present input files must still match (size)."""
    if not _MANIFEST.exists():
        pytest.skip("input manifest not generated (run `make checksums`)")
    module = _load_module()
    problems = module.verify_manifest(_MANIFEST, REPO_ROOT, check_hash=False)
    assert problems == [], "manifest mismatch:\n" + "\n".join(problems)
