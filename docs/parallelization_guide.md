# Future Work Design Notes: Velocity Parallelization

> **Status: future work only.** The package does not currently contain an active
> parallel velocity implementation. Production velocity calculation runs through
> the sequential `compute_olsen_velocity` implementation.

## Purpose

This document records design constraints for a possible future optimization. It
is not a user guide, benchmark report, installation guide, or description of a
currently supported feature. In particular:

- there is no `parallel` package extra;
- `joblib` is not a runtime dependency;
- `ivt_filter.processing.velocity_parallel` exists only as a deprecated import
  compatibility facade; and
- callers should use `ivt_filter.processing.velocity.compute_olsen_velocity`.

## Compatibility boundary

The historical wrapper names may need to remain importable for downstream code:

- `compute_olsen_velocity_parallel`
- `compute_velocity_auto`

Until a real implementation is designed, reviewed, and tested, both wrappers
must delegate to `compute_olsen_velocity`, ignore historical execution-control
arguments, and emit a deprecation warning when called. Their presence must not be
interpreted as evidence of multi-core execution.

## Design questions for a future implementation

Any future proposal should answer the following questions before exposing a
public option or dependency extra:

1. How are sample windows partitioned without changing boundary behavior?
2. How are index order, diagnostic columns, and DataFrame metadata preserved?
3. Which datasets demonstrate a measurable benefit after worker startup and
   serialization overhead?
4. How are sequential and optimized results compared for exact or documented
   numerical equivalence?
5. Which optional dependency, if any, is justified by the implementation?
6. How are fallbacks and platform-specific behavior documented and tested?

## Activation checklist

A future implementation should not be advertised until it includes:

- production code that performs the optimized execution path;
- regression tests for chunk boundaries and result equivalence;
- benchmarks with reproducible inputs and commands;
- an explicit dependency decision;
- updated package metadata and installation documentation; and
- replacement of the compatibility-only warnings with documented API behavior.
