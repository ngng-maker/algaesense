# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This repository currently contains no source code — only two planning/spec documents under `Resources/`. There is no package manifest, build system, linter, or test suite yet, so there are no build/lint/test commands to run. When implementation begins, this file should be updated with the actual commands (likely `pytest`, a linter, and packaging commands per the milestones below).

## What this project is

The two documents in `Resources/` together specify a not-yet-built software subsystem, `jaxsr-calibration`, that augments an existing upstream package called **JAXSR** (symbolic regression + active-learning design of experiments) with a pre-experimental sensor calibration/diagnostics/preprocessing pipeline. The motivating application is monitoring VOC emissions from *Arthrospira* (*Spirulina*) *platensis* cultures grown in Pioreactor vessels, using Alphasense PID sensors, but the design is intentionally sensor- and compound-agnostic.

- [Resources/jaxsr_augmentation_spec.md](Resources/jaxsr_augmentation_spec.md) — the authoritative software specification (module boundaries, function signatures, data schemas, CLI surface, milestones, testing strategy). Written for whoever implements the code (including Claude Code).
- [Resources/spirulina_voc_experimentalist_protocol.md](Resources/spirulina_voc_experimentalist_protocol.md) — the companion hands-on lab protocol for the human operator. Treat it as background/context, not an implementation spec; it references the same `jaxsr` CLI commands the software spec defines.

**When asked to implement any part of this system, read the spec document's "Quick Reference Summary" and "How to Use This Document as a Reference" sections first** — they define reading order and explicitly state that data schemas (Part VI) and function signatures (§34) are the API surface downstream code depends on and should not be deviated from without explicit user confirmation. Sections marked ⚠️ are open decisions (e.g. deployment model, exact PID sensor model, `ols` vs `symbolic` defaults) — ask the user rather than silently picking, per the spec's own guidance.

## Architecture (as specified)

The target system is a five-layer pipeline; only layers 2, 3, and 5 are in scope for this package (layer 1 is deferred external hardware, layer 4 is existing unmodified JAXSR):

| Layer | Function | New package |
|---|---|---|
| 2 | Per-sensor calibration by standard addition | `jaxsr.calibration` |
| 3 | Signal processing (covariate regression, common-mode subtraction, spectral filtering) | `jaxsr.processing` |
| 5 | Diagnostics (fleet-zero, ambient baseline, sensor-swap pilot, weekly audit) | `jaxsr.diagnostics` |

Plus `jaxsr.logging_` (trailing underscore avoids the stdlib clash) for raw sensor acquisition/writing.

Key design points:
- **Additive only** — must not change any existing JAXSR public API, defaults, or behavior. New heavy dependencies (`smbus2`, `pyserial`, `statsmodels`, `polars`) are gated behind a `jaxsr[calibration]` extra.
- **Recommended deployment**: a separate PyPI extension package (`jaxsr-calibration`) that installs into the `jaxsr.*` namespace via entry points, rather than an upstream contribution or a fork (spec §5, ⚠️ still open).
- **In-memory handoff**: the pipeline's output is a DataFrame passed in-process to `jaxsr.SymbolicRegressor.fit()`, not a required file round-trip. Parquet persistence exists for reproducibility, not as the primary API.
- **Compound-agnostic**: the calibration standard (VOC) is chosen interactively at calibration time, not hardcoded. Response factors relative to isobutylene are looked up from a built-in table (`jaxsr/calibration/data/response_factors.yaml`) or supplied by the user. Mixing calibration compounds across a dataset without a known response factor is explicitly guarded against (`MixedCalibrationCompoundError`).
- **Single bridge function** joins the new subsystem to existing JAXSR: `jaxsr.processing.load_features_for_jaxsr(features_df, target=...) -> (X, y, feature_names)`, consumed directly by `jaxsr.SymbolicRegressor.fit`.
- Some fitting steps optionally reuse JAXSR's own machinery (`jaxsr.SymbolicRegressor`, `jaxsr.Constraints`, `jaxsr.uncertainty.prediction_interval`) instead of plain `statsmodels.OLS`, when the user opts into `method="symbolic"` for covariate or sensitivity-surface fits.

Data flows through well-defined, versioned artifacts (see spec Part VI for full schemas):
- Raw per-sensor readings: `data/raw/experiments/{experiment_id}/sensor_id={sensor_id}/hour=YYYY-MM-DDTHH.parquet`
- Calibration results: `data/derived/calibrations/standard_addition/{calibration_run_id}.parquet`
- Final derived features consumed by JAXSR: `data/derived/features/{campaign_id}/{experiment_id}.parquet`, with an append-only `manifest.jsonl` tracking checksums.

Planned repository layout once implementation starts (spec §33, extension-package option):
```
jaxsr-calibration/
├── pyproject.toml
├── src/jaxsr_calibration/   # installs into jaxsr.* namespace via entry points
│   ├── calibration/
│   ├── diagnostics/
│   ├── processing/
│   ├── logging_/
│   └── cli.py
└── tests/{unit,integration,fixtures}/
```

Implementation is expected to proceed in the milestone order laid out in spec Part XIII (scaffold + acquisition → diagnostics → calibration → preprocessing → docs → dashboard), each with an independently testable definition-of-done — do not start a later milestone before the earlier one is done unless explicitly told to.
