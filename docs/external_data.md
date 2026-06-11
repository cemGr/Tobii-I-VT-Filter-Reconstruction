# External benchmark data: Lund2013 / Andersson et al. 2017

The third evaluation axis (WS-C) runs the reconstruction's I-VT against the
hand-annotated dataset from:

> R. Andersson, L. Larsson, K. Holmqvist, M. Stridh, M. Nystrom. *One algorithm
> to rule them all? An evaluation and discussion of ten eye movement
> event-detection algorithms.* Behavior Research Methods 49(2):616-637, 2017.

## Provenance

- Source: <https://github.com/richardandersson/EyeMovementDetectorEvaluation>
- Pinned commit: `3e12416ab3fd6254c81811cf03f8e5d67c5d7129`
- Files used: `annotated_data/data used in the article/img/*.mat` (28 files = 14
  static-image trials x 2 human coders, MN and RA).
- Local copy (gitignored): `test_data/external/lund2013/img/`. SHA-256 of each
  file is recorded by `scripts/write_input_manifest.py` in
  `results/input_manifest.sha256`.

The `img` (static image) trials are used because they are fixation/saccade
dominated. The `dots` and `video` trials contain heavy smooth pursuit, which
I-VT cannot represent, and are out of scope for the primary benchmark.

## File format

Each `.mat` holds an `ETdata` struct:

- `pos`: N x 6 matrix, columns `[timestamp_us, pupil_h, pupil_v, x_px, y_px,
  label]`.
- `screenDim` (mm, [w, h]), `screenRes` (px, [w, h]), `viewDist` (mm),
  `sampFreq` (Hz). Geometry is read from the file by the loader.

Event-label integers: `0` undefined, `1` fixation, `2` saccade, `3` PSO,
`4` smooth pursuit, `5` blink, `6` other.

The loader (`ivt_filter/io/lund2013.py`) maps one file into the slim schema:
pixels to screen millimetres (origin centred), the single gaze stream duplicated
to both eyes, the eye origin synthesised at `(0, 0, viewDist)`, blink samples
marked invalid, and a regular time axis rebuilt from `sampFreq` (some files have
degenerate raw timestamps).

## Label-mapping protocols

I-VT has no smooth-pursuit or PSO class, so labels are mapped through a protocol.

| Protocol | PSO (3) | Pursuit (4) | Notes |
|----------|---------|-------------|-------|
| P0 (primary) | Saccade | excluded | PSO is part of saccadic dynamics; pursuit and undefined are dropped from scoring |
| S1 | Fixation | excluded | PSO as fixation tail (sensitivity) |
| S2 | Saccade | Fixation | pursuit folded into fixation (sensitivity) |

In every protocol: fixation -> Fixation, saccade -> Saccade, blink ->
EyesNotFound (and invalid), undefined/other -> excluded. Excluded samples are
dropped before scoring.

## Reproduce

```
make external          # or:
python scripts/benchmark_external.py --all-protocols --coder both
```

Writes `results/external_benchmark/{external_results.json, external_metrics.csv,
external_summary.txt}`.

## Interpretation

This axis measures agreement with **human coders**, a different target than the
Tobii-fidelity axis (axes 1 and 2). Exact parity is not expected: the
reconstruction reproduces Tobii Pro Lab, not the Lund hand-annotation. Reported
fixation agreement is in the typical range for velocity-threshold detectors on
static-image trials; saccade recall is lower because the human coders annotate
PSO and saccade onset/offset more finely than a pure velocity threshold can.
