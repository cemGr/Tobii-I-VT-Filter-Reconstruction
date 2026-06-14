# Tuning protocol and the no-regression gate (WS-B)

Every change to a reconstruction assumption is flag-gated, defaults to the
current behaviour, and is checked against a frozen per-condition baseline so no
condition loses fidelity. This document records the protocol and the evidence
that shapes which fixes are worth making.

## Baseline and the gate

- `results/filter_benchmark/baseline_v1.0.0.json` is the frozen per-condition
  snapshot (Cohen's kappa and the other agreement metrics) produced by
  `scripts/freeze_baseline.py` from a clean benchmark run.
- `tests/test_baseline_no_regression.py` is the single go/no-go gate: after a
  change, run `make benchmark` and then the test; it fails if any condition's
  kappa drops below its frozen value, or a condition errors or disappears.
- The A/B loop for any fix is: change code (flag defaulting to current
  behaviour) -> `make benchmark` -> run the gate. A fix is accepted only if it
  improves at least one condition and regresses none.

## Environment provenance

The committed benchmark numbers must be regenerated per environment. On the
current toolchain (numpy 1.26.4, pandas 3.0.3, scipy 1.11.4) the smoothing
conditions differ materially from older runs (for example the MA-9 condition
`LeftV30W20NoiseAvW9` moves from a previously recorded kappa near 0.68 to 0.98).
The baseline is therefore frozen on the current environment, not inherited from
stale committed results. The data snapshot is pinned by
`results/input_manifest.sha256`.

## Tune / hold-out split

`test_data/inputs/configs.json` carries no participant field, so a clean
per-participant split cannot be asserted from the benchmark set.

- Tune set: the 14 study-A1 robustness configurations. All parameter choices
  (thresholds, eps, strategy selection) are made here.
- Hold-out / report set: the controlled study (C1-C6, two participants x three
  eye settings x two rates). The headline fidelity numbers come from this set,
  untouched during tuning.
- Upgrade to leave-one-participant-out only if participant identifiers are
  recovered from the controlled-study filenames. State the regime used; do not
  overclaim.

The no-regression gate spans both sets, so a fix cannot win on the tune set
while quietly hurting the hold-out.

## What the per-sample attribution changes (evidence)

The thesis diagnoses three aggregate residual causes: Average-mode eye-offset
fallback (dominant in the occluded condition C5), Stage-4 boundary/half-window
off-by-one, and floating-point flips near the threshold. The per-sample
attribution (WS-A, `scripts/attribute_discrepancies.py`) refines this:

- On the clean binocular baseline (IVT30) the residual is dominated by
  **confident velocity-magnitude differences** (~64%, the reconstruction
  computes a velocity far from the threshold where Tobii's label is the
  opposite, typically brief post-saccadic velocity spikes Tobii absorbs) and
  **event-boundary timing** (~33%, off-by-one at fixation/saccade onset and
  offset). The eye-offset, near-threshold and smoothing mechanisms account for a
  small minority here, because the binocular benchmark files never trigger
  single-eye Average-mode fallback.
- The eye-offset cause is real but specific to the occluded-eye Average-mode
  conditions of the controlled study, which are not part of the 14-file
  benchmark.

## WS-B changes and their scope

1. `fixed_window_edge_fallback_limit` (default 50) makes the Stage-4 edge
   fallback search distance configurable and auditable. Default preserves
   behaviour; it governs the EOF-gap edge case (see
   `tests/test_velocity_fallback_limit.py` and the WS-D edge tests).
2. `offset_fallback_strategy` (`last_offset` default, `single_eye_direct`
   alternative) makes the Average-mode mono-interval reconstruction assumption
   selectable. `last_offset` is byte-identical to the previous behaviour. The
   alternative is exercised only when `tobii_eye_offset_interpolation` is on and
   one eye is missing, i.e. on the occluded-eye controlled conditions; it is a
   no-op on the binocular benchmark, so its A/B validation belongs on the
   controlled-study data rather than the 14-file benchmark.

3. `merge_absorb_saccades` (default `False`) makes the fixation-merge stage
   Tobii-faithful. The component-fidelity analysis
   (`scripts/component_fidelity.py`, `results/component_fidelity/`) found merge
   to be the single weakest-reproduced stage: pooled direction-faithful recall
   0.08 against >=0.90 for every other component, with high precision, i.e. pure
   under-triggering. The cause is the gap relabel loop's
   `max_gap_velocity_deg_per_sec` cap (35 deg/s, just above the 30 deg/s
   classification threshold): it blocks ~100% of the inter-fixation saccade
   samples, which is exactly what merge must absorb. Tobii's
   `MergeFixationsFilter` does not cap. With the flag on, a merged pair relabels
   its whole gap to Fixation (never overwriting EyesNotFound); the default keeps
   the conservative cap. A/B over the study_a1 merge sweep
   (`component_fidelity.py --component merge --source study_a1
   --merge-absorb-saccades`) lifts pooled merge recall 0.082 -> 0.901 at stable
   precision (0.843 -> 0.852). Unlike the other two flags this targets a
   dominant cause, so it is the strongest candidate for promotion to default
   (which requires re-freezing the baseline across all merge conditions).

4. `erode_saccade_edges` (default `False`, with `saccade_edge_velocity_band_deg`,
   default 2.0) targets the event-boundary off-by-one. The per-sample attribution
   makes it the largest single residual cause (~41% of all disagreements; 87-94%
   on the clean single-eye conditions), and on the proper 20 ms window it is
   directional: at fixation/saccade transitions our saccade runs ~1 sample too
   wide (GT-Fixation/pred-Saccade boundary errors outnumber the reverse ~7:1 on
   `LeftV30W20`). The centred Olsen window smears velocity across the transition,
   lifting the boundary sample just over threshold. With the flag on, the
   first/last sample of a saccade run that borders a fixation is relabelled to
   Fixation iff its velocity is in `[threshold, threshold + band)`; genuine fast
   saccade edges are untouched. **Tested and NOT recommended:** an A/B over the
   clean conditions and a band sweep (2.0/1.0/0.5 deg/s) shows net kappa never
   rises — it ranges from neutral to clearly negative (e.g. `LeftV30W20`
   0.9953->0.9925 at band 2.0, ->0.9943 at band 1.0; `left20ms30` 0.9980->0.9889).
   Genuine saccade edges share the same borderline-velocity band, so the rule
   erodes correct labels at least as often as it fixes incorrect ones. This is a
   reproducible negative result: the event-boundary residual is not correctable by
   a velocity-band edge convention and is confirmed as an intrinsic limitation.

All four default to current behaviour, so the benchmark gate stays green. Flags
1, 2 and 4 do not net-improve the dominant clean-condition residual
(velocity-magnitude and event-boundary timing); that residual reflects the
irreducible difference between the reconstruction's per-sample velocity and
Tobii's windowed classification, and is documented as a limitation rather than
forced. Flag 4 in particular has been tested and shown not to help, so it stays
opt-in only.
