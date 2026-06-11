# Reproducibility and quality targets for the Tobii I-VT reconstruction.
#
# Most targets need the local Tobii study data under test_data/inputs (gitignored)
# and, for the external axis, the Lund2013 data under test_data/external. Quality
# targets (test/lint/typecheck/hooks) run anywhere.
#
# Override the interpreter or paths on the command line, e.g.:
#     make test PYTHON=.venv/bin/python
#     make gazetoolkit GTK_DIR=/path/to/GazeToolkit

PYTHON ?= python
COV_FAIL_UNDER ?= 70

INPUT_DIR   ?= test_data/inputs
RESULTS_DIR ?= results
# Directory holding the GazeToolkit (Konopka) input/output files for the
# head-to-head axis. Overridable; the script carries the same default.
GTK_DIR     ?= $(CURDIR)/../GazeToolkit
# Lund2013 / Andersson 2017 annotated data (img trials), gitignored.
LUND_DIR    ?= test_data/external/lund2013/img

.DEFAULT_GOAL := help

.PHONY: help test lint typecheck hooks baseline benchmark attribution \
        gazetoolkit external figures checksums reproduce

help:
	@echo "Quality:        test lint typecheck hooks"
	@echo "Evaluation:     benchmark attribution gazetoolkit external figures"
	@echo "Provenance:     checksums baseline"
	@echo "All numbers:    reproduce  (= checksums benchmark attribution gazetoolkit external figures)"

# --- Quality ---------------------------------------------------------------

test:
	$(PYTHON) -m pytest -q --cov=ivt_filter --cov-report=term-missing --cov-fail-under=$(COV_FAIL_UNDER)

lint:
	ruff check ivt_filter

typecheck:
	mypy ivt_filter

hooks:
	pre-commit install

# --- Provenance ------------------------------------------------------------

checksums:
	$(PYTHON) scripts/write_input_manifest.py

# Freeze the per-condition golden baseline. Run `make benchmark` first; this is a
# deliberate, rarely-run action that defines the WS-B no-regression gate.
baseline: benchmark
	$(PYTHON) scripts/freeze_baseline.py

# --- Evaluation (the three axes + attribution) -----------------------------

# Axis 1: fidelity vs Tobii Pro Lab over the 14 benchmark conditions.
benchmark:
	$(PYTHON) scripts/benchmark_filter_settings.py --input-dir $(INPUT_DIR) --results-dir $(RESULTS_DIR)/filter_benchmark

# WS-A: per-sample discrepancy attribution.
attribution:
	$(PYTHON) scripts/attribute_discrepancies.py --input-dir $(INPUT_DIR) --results-dir $(RESULTS_DIR)/attribution

# Axis 2: head-to-head vs GazeToolkit (Konopka).
gazetoolkit:
	$(PYTHON) scripts/evaluate_gazetoolkit.py --gtk-dir $(GTK_DIR) --results-dir $(RESULTS_DIR)/gazetoolkit

# Axis 3: external hand-annotated benchmark (Lund2013 / Andersson 2017).
external:
	$(PYTHON) scripts/benchmark_external.py --data-dir $(LUND_DIR) --all-protocols --coder both --results-dir $(RESULTS_DIR)/external_benchmark

figures:
	$(PYTHON) scripts/generate_sensitivity_figure.py

reproduce: checksums benchmark attribution gazetoolkit external figures
	@echo "Reproduced all evaluation artefacts under $(RESULTS_DIR)/"
