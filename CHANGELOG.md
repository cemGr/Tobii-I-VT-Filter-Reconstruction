# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-10

First stable release of the from-scratch reconstruction of Tobii's I-VT
velocity-threshold fixation filter.

### Added
- Configurable I-VT pipeline: preprocessing (gap filling, eye selection, noise
  reduction), velocity computation, classification, and post-processing
  (saccade/fixation merging, short-fixation discarding).
- Multiple velocity strategies (Olsen 2D, Ray 3D, Tobii gaze-direction angle)
  and window-selection policies, including sample-based and anchor windows.
- Command-line interface (`python -m ivt_filter.cli`) and a simple Python API.
- Evaluation tooling: agreement metrics, Cohen's kappa, event IoU, and plotting.
- Benchmark and parameter-sweep scripts with reproducible result outputs.
- Docker image and CI pipeline (tests, lint, type-check, packaging, container).

### Changed
- All source comments and docstrings standardized to English for consistency.

[1.0.0]: https://github.com/cemGr/Tobii-I-VT-Filter-Reconstruction/releases/tag/v1.0.0
