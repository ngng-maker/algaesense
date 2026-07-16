# JAXSR Pre-Experimental Calibration Subsystem
## Specification for Augmenting JAXSR with Diagnostics, Sensor Calibration, and Preprocessing

---

## Quick Reference Summary

*This block exists so a reader loading this document as a reference can orient in under a minute. If you are Claude Code and a prompt just told you to consult this file, read this block first, then the "How to Use This Document as a Reference" section immediately after, then jump to whichever Part is relevant to your task.*

**What this document specifies:** a Python extension package named `jaxsr-calibration` that installs new subpackages into the `jaxsr.*` namespace — specifically `jaxsr.calibration`, `jaxsr.diagnostics`, `jaxsr.processing`, and `jaxsr.logging_` — and registers a CLI under the `jaxsr` command. The package prepares calibrated, preprocessed sensor data that feeds directly into `jaxsr.SymbolicRegressor` and `jaxsr.acquisition.ActiveLearner` (both of which already exist in JAXSR and are out of scope here).

**What it takes as input:** a physical rig of Pioreactor 20 mL reactors with Alphasense PID VOC sensors and Sensirion SHT35 or Bosch BME280 T+RH sensors, plus a directory of YAML configs (`sensors.yaml`, `reactors.yaml`, `rotation_schedule.yaml`, `preprocessing.yaml`, `diagnostic_thresholds.yaml`), plus interactive user input at calibration time (the user picks a calibration VOC standard from a menu).

**What it produces:** for each experiment run, a Parquet file at `data/derived/features/{campaign_id}/{experiment_id}.parquet` conforming to the schema in §19, plus an append-only JSONL manifest, plus a Python entry point `jaxsr.processing.load_features_for_jaxsr()` that returns `(X, y, feature_names)` directly consumable by `jaxsr.SymbolicRegressor.fit`.

**Motivating use case:** *Spirulina platensis* (also *Arthrospira platensis*) cultivation in Pioreactors, measuring VOC emission with PID sensors. The subsystem is designed to generalize to any noisy-sensor DoE problem and is fully VOC-agnostic — no compound identity is hardcoded anywhere.

**What is explicitly out of scope:**
- Symbolic regression fitting itself (existing `jaxsr.SymbolicRegressor` handles this).
- Active-learning acquisition planning (existing `jaxsr.acquisition` handles this).
- Physical hardware conditioning like Nafion dryers or heated sample lines ("Layer 1" in the architecture, deferred).
- Wet-lab biology like culture preparation or inoculation (see the companion experimentalist protocol).
- Compound identification via GC-MS (either done in an earlier stage or skipped).

**Runtime dependencies:** Python ≥ 3.11, `jaxsr` (upstream), `numpy`, `scipy`, `pandas`, `polars`, `pyarrow`, `statsmodels`, `pydantic` ≥ 2, and a click/typer CLI framework. Optional extras: `smbus2` and `pyserial` for hardware I/O, `streamlit` or `dash` for the live dashboard.

**Open decisions still to resolve** (marked ⚠️ where they appear in the body):
- Deployment as an extension package (current default), an upstream JAXSR contribution, or a fork.
- Exact Alphasense PID model per sensor slot — affects noise thresholds.
- Number of reactors and sensors at Milestone 1.
- Default covariate-regression method (`ols` current default; alternatives `robust`, `symbolic`).
- Whether the `jaxsr` CLI namespace is available or a `jaxsr-cal` fallback is needed.
- Off-machine backup cadence and retention for raw Parquet data.

**Companion document (also expected to be in reach):** `spirulina_voc_experimentalist_protocol.md` — the hands-on lab procedure that this software subsystem serves. It is written for the human operator, not for Claude Code, but it is the source of truth for what the operator physically does that this software then depends on.

**Position in a larger project:** this document specifies one stage of a larger multi-stage project. The parent orchestration prompt is expected to identify which specific stages sit upstream (typically hardware/rig setup and wet-lab culture preparation) and downstream (typically the JAXSR modeling and active-learning loop that consumes this stage's outputs). This document does not attempt to name them.

---

## How to Use This Document as a Reference

*This section is written directly for Claude Code (or any LLM/human) that has been told to read this file as a reference while working on a larger project.*

**Treat this document as authoritative for:** everything about how the diagnostics, sensor calibration, and preprocessing subsystem is designed, structured, and implemented. This includes module boundaries, function signatures (§34), data schemas (Part VI), configuration file formats (Part V), CLI surface (Part XI), directory layout (§33), and the ordering of implementation milestones (Part XIII).

**Do not treat this document as authoritative for:** anything outside its scope. In particular, if the parent prompt asks you to modify how JAXSR's `SymbolicRegressor` or `ActiveLearner` behave, look at JAXSR's own upstream documentation, not this file — those components are out of scope here and this document only describes how to feed data into them.

**When the parent prompt and this document conflict:** defer to the parent prompt on questions of scope, deadlines, package naming, deployment model, or which subset of the milestones to implement first. Defer to this document on questions of internal design — data schemas, function signatures, algorithms, testing strategy — unless the parent prompt explicitly overrides them. If the parent prompt makes a change that contradicts a data schema or function contract here, ask the user rather than silently reconciling; schema drift causes hard-to-diagnose downstream bugs.

**When implementing, follow this order:**
1. Read the "Quick Reference Summary" above (already done).
2. Read the "Stage Interface Contract" section immediately below — it defines the boundary of what you're building.
3. Read Part I (design principles) and Part II (integration constraints).
4. For any specific implementation task, jump to the corresponding Part: acquisition → Part III + §9 + §33; diagnostics → Part VII; calibration → Part VIII; preprocessing → Part IX; the bridge to JAXSR core → Part X; CLI → Part XI; testing → Part XIV.
5. When implementing a function, cross-reference the module contract in §34 (function signatures) and the data schemas in Part VI. These two sections are the API surface downstream code will program against; do not deviate from them without explicit user confirmation.
6. Follow the milestone ordering in Part XIII. Each milestone has an independently-testable "definition of done." Do not start a later milestone before the earlier one is at DoD, unless the parent prompt explicitly says to.

**How to reference specific parts back to the user or to the parent prompt:** use the section numbers and part numbers (e.g. "per §21, the ambient baseline diagnostic returns a `dict[sensor_id, CovariateModel]`"). These numbers are stable and unambiguous within this document.

**If sections marked ⚠️ block you:** these are open decisions where a default is stated but the final answer is up to the project owner. If the parent prompt has not resolved them, ask the user before implementing anything that depends on them. Do not silently pick.

---

**Version:** 0.1 (draft)
**Scope:** Extend JAXSR with a pre-experimental calibration subsystem (`jaxsr.calibration`, `jaxsr.diagnostics`, `jaxsr.processing`, `jaxsr.logging_`) that runs BEFORE the existing JAXSR DoE/symbolic-regression workflow.
**Motivating use case:** *Spirulina platensis* (also *Arthrospira platensis*) cultivation in Pioreactor 20 mL vessels with Alphasense PID VOC sensors. The subsystem is designed to generalize to any noisy-sensor DoE problem.
**Companion document:** `spirulina_voc_experimentalist_protocol.md` — hands-on lab procedure that this subsystem serves.
**Status:** working spec. Sections marked ⚠️ are open decisions.

---

## Stage Interface Contract

*This section is authoritative for orchestration. If it conflicts with detail elsewhere in the document, this section wins and the other should be corrected.*

### What this stage does

Given a physical rig of Pioreactor 20 mL reactors instrumented with Alphasense PID sensors and Sensirion/Bosch T+RH sensors, this stage acquires raw sensor streams, verifies sensor health, calibrates each sensor in-situ against a user-selected VOC standard, applies signal-processing corrections, and emits a per-experiment feature file that a downstream stage passes to `jaxsr.SymbolicRegressor.fit`.

### Preconditions (what upstream stages must have already established)

1. Physical rig exists per the experimentalist protocol Part III (reactors, PIDs, ancillary T/RH sensors, LEDs, gas lines, wiring).
2. `configs/` directory is populated with `sensors.yaml`, `reactors.yaml`, `rotation_schedule.yaml`, and `preprocessing.yaml`.
3. A cell-free reactor is available for standard-addition calibration immediately before each experiment run.
4. A calibration standard (gas cylinder or permeation tube) is on hand; its compound identity will be recorded at calibration time via the interactive UI (§25).
5. Biological culture preparation is complete and inoculum is ready (see experimentalist protocol Part IV).

### Postconditions (what this stage guarantees to downstream stages)

1. For every experiment run `{experiment_id}` in campaign `{campaign_id}`, a file exists at `data/derived/features/{campaign_id}/{experiment_id}.parquet` conforming to schema §19 with `features_schema_version: 1`.
2. The file contains one row per `(experiment_id, reactor_id, sensor_id)` triple; sensors flagged `EXCLUDED` in run metadata are omitted.
3. Every row carries provenance columns: `calibration_run_id`, `calibration_compound`, `calibration_response_factor`, `covariate_model_id`, `preprocessing_config_hash`.
4. The manifest at `data/derived/features/{campaign_id}/manifest.jsonl` has been appended with an entry recording the new file's SHA-256 checksum.
5. The Python function `jaxsr.processing.load_features_for_jaxsr(features_df, target="mean_voc_ppm_asgas")` returns `(X: np.ndarray, y: np.ndarray, feature_names: list[str])` in a form directly consumable by `jaxsr.SymbolicRegressor.fit`.

### Explicit non-goals (for orchestration clarity)

- This stage does not fit any symbolic-regression model; that is the job of the downstream stage using `jaxsr.SymbolicRegressor`.
- This stage does not plan the next batch of experimental conditions; that is the job of the downstream stage using `jaxsr.acquisition.ActiveLearner`.
- This stage does not manage biology (culture prep, inoculation); that is a wet-lab stage documented in the experimentalist protocol.
- This stage does not do compound identification via GC-MS; if the user needs compound identification, they run GC-MS in an earlier stage and choose an appropriate calibration standard at run time here.
- This stage does not perform hardware conditioning (Nafion drying, heated lines); that is a deferred hardware stage ("Layer 1" in the five-layer architecture, §2).

*(Guidance for how a Claude Code reader loading this file as a reference should navigate it is in the "How to Use This Document as a Reference" section near the top of the file.)*

---

## Table of Contents

- [Quick Reference Summary](#quick-reference-summary)
- [How to Use This Document as a Reference](#how-to-use-this-document-as-a-reference)
- [Stage Interface Contract](#stage-interface-contract)
- [Part I — Purpose, Context, Design Principles](#part-i--purpose-context-design-principles)
- [Part II — Integration with Existing JAXSR](#part-ii--integration-with-existing-jaxsr)
- [Part III — Subpackage Layout](#part-iii--subpackage-layout)
- [Part IV — End-to-End Workflow](#part-iv--end-to-end-workflow)
- [Part V — Configuration](#part-v--configuration)
- [Part VI — Data Schemas](#part-vi--data-schemas)
- [Part VII — Diagnostics Subpackage](#part-vii--diagnostics-subpackage)
- [Part VIII — Calibration Subpackage](#part-viii--calibration-subpackage)
- [Part IX — Preprocessing Subpackage](#part-ix--preprocessing-subpackage)
- [Part X — Bridge to Existing JAXSR DoE Workflow](#part-x--bridge-to-existing-jaxsr-doe-workflow)
- [Part XI — CLI Surface](#part-xi--cli-surface)
- [Part XII — Repository Structure & Module Contracts](#part-xii--repository-structure--module-contracts)
- [Part XIII — Task Backlog & Milestones](#part-xiii--task-backlog--milestones)
- [Part XIV — Testing Strategy](#part-xiv--testing-strategy)
- [Appendices](#appendices)

---

## Part I — Purpose, Context, Design Principles

### 1. Purpose

JAXSR currently assumes clean numeric inputs — feature matrix `X` and observation vector `y` — and provides symbolic regression + active-learning design of experiments on top of them. In real experimental workflows using noisy physical sensors (photoionization detectors, electrochemical cells, optical probes, etc.), the `X`/`y` matrix is not directly measured; it must be *reconstructed* from raw sensor voltages after diagnostics, calibration, and preprocessing.

This subsystem adds the missing pre-experimental phase. After it runs, the user can call `jaxsr.SymbolicRegressor().fit(X, y)` on data whose noise sources have been characterized and largely removed, with proper uncertainty attached.

### 2. Context: the five-layer architecture

For the motivating *Arthrospira* + PID application, the full stack is:

| Layer | Function | Where it lives |
|-------|----------|----------------|
| 1 | Physical sample conditioning (Nafion, heated lines) | External hardware; **deferred** |
| 2 | Per-sensor calibration by standard addition | **`jaxsr.calibration`** (new) |
| 3 | Signal processing: covariate regression, common-mode, spectral | **`jaxsr.processing`** (new) |
| 4 | Symbolic regression + active-learning DoE | `jaxsr.SymbolicRegressor`, `jaxsr.acquisition` (existing) |
| 5 | Diagnostics (fleet-zero, ambient baseline, swap pilot, weekly audit) | **`jaxsr.diagnostics`** (new) |

Layers 2, 3, 5 are new; Layer 4 is unchanged. Layer 1 is out of scope, compensated for in Layers 2 and 3.

### 3. Design principles

**Additive, not disruptive.** The subsystem must not change the behavior of any existing JAXSR API. Existing users of `BasisLibrary`, `SymbolicRegressor`, `ActiveLearner`, and the acquisition functions see zero change.

**Reuse JAXSR primitives where they fit naturally.** Where the calibration or covariate models involve nontrivial constrained fitting, they use `jaxsr.Constraints` and `jaxsr.uncertainty.prediction_interval` so uncertainty handling is unified across the pre-experimental and DoE phases. Where the models are trivially linear (e.g. the standard-addition sensitivity fit), plain `statsmodels.OLS` is used — pulling in the full symbolic-regression machinery for a two-parameter linear fit is overkill.

**In-memory handoff, not file-based handoff.** The calibrated features produced by `jaxsr.processing.features` are a pandas or polars DataFrame in memory; the caller passes them to `SymbolicRegressor.fit()` in the same Python process. File-based persistence exists for reproducibility and cross-session workflows, but the intra-process API is a normal Python call, not a Parquet round-trip.

**Optional import.** The new subpackages have heavier optional dependencies (`smbus2`, `pyserial`, `statsmodels`, `polars`). Users who only want JAXSR's symbolic regression should not need to install them. The subpackages register a `jaxsr[calibration]` extra in `pyproject.toml` and gate imports with clear error messages.

**Sensor-agnostic core.** Alphasense PID is the motivating case, but the abstractions (`SensorReader`, `AncillarySensorReader`, `Calibrator`) are generic. A user with a different sensor writes a small subclass or config entry.

### 4. What this subsystem does NOT do

- Does not modify `jaxsr.SymbolicRegressor` or `jaxsr.acquisition.*`.
- Does not do any symbolic regression or active learning itself — that stays in JAXSR core.
- Does not implement hardware conditioning (Nafion, heated lines) — external hardware only.
- Does not attempt compound-specific quantification unless the user provides a compound-specific calibration standard.
- Does not modify JAXSR's documentation build system beyond adding new pages under `docs/calibration/`.

---

## Part II — Integration with Existing JAXSR

### 5. Deployment model (⚠️ decide before Milestone 1)

Three options; all are viable. My default recommendation is (B) unless the JAXSR maintainers explicitly want (A).

**(A) Upstream contribution to JAXSR.** New subpackages live inside the JAXSR repo. Requires coordination with the Kitchin Group; slowest but produces the cleanest end state.

**(B) Extension package `jaxsr-calibration`.** Separate PyPI-installable package that imports `jaxsr` and installs its subpackages into the `jaxsr.*` namespace via an entry-point-based plugin mechanism. Independent release cadence. Recommended default.

**(C) Fork of JAXSR.** Full fork with the new subsystem integrated. Fast to prototype; hard to maintain long-term.

Whichever is chosen, the *import path from a user's perspective* is identical: `from jaxsr.calibration import run_standard_addition`. Users don't see the deployment choice.

### 6. Namespace layout after augmentation

```
jaxsr/
├── __init__.py                 # exports (unchanged public API + new names)
├── basis.py                    # existing
├── regressor.py                # existing SymbolicRegressor
├── constraints.py              # existing Constraints
├── acquisition/                # existing
│   ├── prediction_variance.py
│   ├── d_optimal.py
│   ├── expected_improvement.py
│   ├── model_discrimination.py
│   └── ...
├── uncertainty.py              # existing
├── sampling.py                 # existing latin_hypercube_sample etc.
│
├── calibration/                # NEW
│   ├── __init__.py
│   ├── standard_addition.py
│   ├── reference_jar.py
│   ├── apply.py
│   └── models.py               # dataclasses
├── diagnostics/                # NEW
│   ├── __init__.py
│   ├── fleet_zero.py
│   ├── ambient.py
│   ├── swap_pilot.py
│   ├── weekly.py
│   └── i2c.py
├── processing/                 # NEW
│   ├── __init__.py
│   ├── covariate.py
│   ├── common_mode.py
│   ├── spectral.py
│   ├── convert.py
│   └── features.py
├── logging_/                   # NEW  (trailing underscore avoids stdlib clash)
│   ├── __init__.py
│   ├── acquire.py
│   ├── schema.py
│   └── writer.py
└── cli.py                      # NEW top-level; extends any existing CLI
```

### 7. Backwards compatibility

- No existing name in `jaxsr.*` is renamed or removed.
- No default argument in any existing function is changed.
- No new required dependency is added to the base install; all new deps are gated behind `jaxsr[calibration]`.
- A user running the current JAXSR test suite after this change sees identical results.

### 8. Where the new subsystem uses existing JAXSR primitives

| New component | Uses JAXSR primitive | Purpose |
|---------------|---------------------|---------|
| `calibration.apply` | `jaxsr.uncertainty.prediction_interval` | uncertainty on inverted concentration |
| `processing.covariate` (JAXSR-augmented variant) | `jaxsr.BasisLibrary`, `jaxsr.SymbolicRegressor`, `jaxsr.Constraints` | optional symbolic covariate model when user opts in |
| `processing.features` (output) | fed directly to `jaxsr.SymbolicRegressor.fit` | no adapter needed |

Where JAXSR primitives are used, they are used as *first-class dependencies*, not wrapped or hidden. If the user wants to inspect a covariate model, they get back a real `SymbolicRegressor` instance.

---

## Part III — Subpackage Layout

### 9. `jaxsr.logging_` — sensor acquisition

Purpose: acquire raw sensor streams from physical hardware (PID via ADC, T/RH via I²C, Pioreactor state via HTTP), write to partitioned Parquet.

Public API:

```python
from jaxsr.logging_ import start_logging, stop_logging, LoggerHandle
```

Depends on: `smbus2`, `pyserial`, `pyarrow`, `polars`. All optional; imports fail with a helpful error if the extra isn't installed.

### 10. `jaxsr.diagnostics` — sensor-health tests

Purpose: characterize sensor health independent of biology. Fleet-zero, ambient baseline, sensor-swap Latin-square audit, weekly rollup, I²C bus scan.

Public API:

```python
from jaxsr.diagnostics import (
    run_fleet_zero, run_ambient_baseline, run_swap_pilot,
    run_weekly_audit, scan_i2c,
    FleetZeroResult, AmbientBaselineResult, SwapPilotResult,
)
```

### 11. `jaxsr.calibration` — in-situ sensor calibration

Purpose: per-sensor calibration by standard addition; reference-jar cross-sensor drift tracking; voltage-to-ppm inversion helper.

Public API:

```python
from jaxsr.calibration import (
    run_standard_addition, fit_sensitivity_per_sensor,
    run_reference_jar_rotation, compute_fleet_ratios,
    apply_calibration,
    SensitivityModel,
)
```

### 12. `jaxsr.processing` — signal processing pipeline

Purpose: covariate regression, common-mode subtraction, spectral filtering, voltage-to-ppm conversion, feature extraction for downstream JAXSR fitting.

Public API:

```python
from jaxsr.processing import (
    fit_covariate_model, apply_covariate_correction, CovariateModel,
    subtract_common_mode,
    lomb_scargle, notch_filter_known_artifacts,
    extract_features,
    load_features_for_jaxsr,       # convenience: returns (X, y, feature_names)
)
```

### 13. Extension points

Users with non-Alphasense sensors register their own reader by subclassing `jaxsr.logging_.SensorReader` and providing an entry point in `pyproject.toml`. The reader must produce records matching the schema in §17.

---

## Part IV — End-to-End Workflow

The workflow is now a **single Python session or pipeline** with two phases:

```python
import jaxsr
import jaxsr.calibration as jcal
import jaxsr.diagnostics as jdiag
import jaxsr.processing as jproc
import jaxsr.logging_ as jlog

# ─────────────────────────────────────────────────────────────
# Phase A: Pre-experimental calibration (new)
# ─────────────────────────────────────────────────────────────

# One-time / weekly diagnostics
zero_result = jdiag.run_fleet_zero(duration_min=60)
assert zero_result.summary_status in ("GREEN", "YELLOW")

baseline = jdiag.run_ambient_baseline(duration_h=12)   # sets per-sensor T/RH coefs

# Per-experiment: calibrate immediately before the run
cal_result = jcal.run_standard_addition(
    experiment_id="exp_2026-07-15_batch03",
    spike_ppm_list=[1.0, 5.0, 20.0],
    dwell_seconds=300,
)
sensitivity_models = jcal.fit_sensitivity_per_sensor(cal_result)

# Run the experiment (biology + sensor logging)
handle = jlog.start_logging(experiment_id="exp_2026-07-15_batch03")
# ... reactor runs for 12 h ...
raw_path = jlog.stop_logging(handle)

# Preprocess raw sensor data → calibrated feature vectors
timeseries = jproc.load_experiment(raw_path)
timeseries = jproc.apply_covariate_correction(timeseries, baseline.covariate_models)
timeseries = jproc.subtract_common_mode(timeseries)
timeseries = jproc.apply_calibration(timeseries, sensitivity_models)
features   = jproc.extract_features(timeseries, metadata)

# ─────────────────────────────────────────────────────────────
# Phase B: Existing JAXSR DoE workflow (unchanged)
# ─────────────────────────────────────────────────────────────

X, y, feature_names = jproc.load_features_for_jaxsr(
    features, target="mean_voc_ppm_asgas"     # or "mean_voc_ppm_iso_equiv" if RF known
)

lib = (
    jaxsr.BasisLibrary(n_features=X.shape[1], feature_names=feature_names)
    .add_constant().add_linear()
    .add_polynomials(max_degree=2).add_interactions(max_order=2)
    .add_transcendental(["log", "exp", "inv"])
)
cons = jaxsr.Constraints().add_bounds(target="y", lower=0)

model = jaxsr.SymbolicRegressor(basis_library=lib, max_terms=8, constraints=cons)
model.fit(X, y)

# Active learning for next batch (existing JAXSR API)
from jaxsr.acquisition import ActiveLearner, ExpectedImprovement, PredictionVariance
acq = 0.7 * ExpectedImprovement(minimize=False) + 0.3 * PredictionVariance()
learner = ActiveLearner(model, bounds=control_bounds, acquisition=acq)
next_batch = learner.suggest(n_points=4, batch_strategy="penalized")
```

The join between the two phases is a single line: `jproc.load_features_for_jaxsr(features, target=...)`.

---

## Part V — Configuration

Config lives in YAML under `configs/` in the *user's* project (not inside JAXSR itself). JAXSR provides Pydantic schemas users import.

```python
from jaxsr.calibration.config import SensorConfig, RotationSchedule
from jaxsr.processing.config import PreprocessingConfig
```

### 14. Files

- `configs/sensors.yaml` — one entry per PID + ancillary sensor.
- `configs/reactors.yaml` — one entry per Pioreactor.
- `configs/rotation_schedule.yaml` — sensor-to-reactor Latin-square rotation.
- `configs/preprocessing.yaml` — preprocessing parameters.
- `configs/diagnostic_thresholds.yaml` — pass/fail limits.

Schemas for each file are Pydantic models exported from `jaxsr.calibration.config` (SensorConfig, ReactorConfig, RotationSchedule) and `jaxsr.processing.config` (PreprocessingConfig, DiagnosticThresholds). The example below and the field-by-field descriptions in §17–§19 fully specify the required structure; the Pydantic classes are the runtime source of truth.

### 15. Example `configs/preprocessing.yaml`

```yaml
preprocessing_schema_version: 1
covariate_regression:
  method: ols                    # ols | robust | symbolic
  training_window: first_30min
  min_rh_range_pct: 20
  # if method: symbolic
  symbolic:
    max_terms: 4
    basis:
      polynomial_degree: 2
      transcendental: [log, exp]
common_mode:
  method: median
  outlier_std_threshold: 3.0
  min_healthy_fraction: 0.75
spectral:
  known_artifact_freqs_hz: [0.00028, 0.00056]
  notch_q: 30.0
  min_amplitude_to_flag: 0.05
concentration:
  extrapolation_policy: clip
features:
  analysis_window: last_2h_of_run
```

Note the `symbolic` option under `covariate_regression`: when set, the covariate model is fit with `jaxsr.SymbolicRegressor` instead of `statsmodels.OLS`. This is the first natural point where the pre-experimental subsystem leverages JAXSR's own machinery. Default remains `ols` for simplicity and speed.

---

## Part VI — Data Schemas

This section is authoritative for every persisted-file format the subsystem reads or writes. Types not shown inline live in `jaxsr.logging_.schema` (pyarrow) and `jaxsr.calibration.config` / `jaxsr.processing.config` (pydantic).

### 16. Raw record schema (input)

Written by `jaxsr.logging_.writer` to:

```
data/raw/experiments/{experiment_id}/sensor_id={sensor_id}/hour=YYYY-MM-DDTHH.parquet
```

Fields per row (one row per sensor per timestamp, ~1 Hz):

| Field | Type | Units | Nullable | Source |
|-------|------|-------|----------|--------|
| `timestamp` | `timestamp[ns, UTC]` | UTC | no | logger |
| `experiment_id` | `string` | — | no | logger |
| `sensor_id` | `string` | — | no | config |
| `reactor_id` | `string` | — | no | rotation schedule |
| `pid_voltage_mv` | `float64` | mV | no | PID ADC |
| `sample_t_c` | `float64` | °C | yes | SHT/BME |
| `sample_rh_pct` | `float64` | % RH | yes | SHT/BME |
| `sample_flow_sccm` | `float64` | sccm | yes | flow meter (if present) |
| `pump_pwm` | `float32` | 0–1 | yes | Pioreactor |
| `lamp_hours` | `float32` | h | no | config + elapsed |
| `reactor_par_umol_m2_s` | `float32` | µmol/m²/s | yes | LED command |
| `reactor_temp_c` | `float32` | °C | yes | Pioreactor probe |
| `reactor_od` | `float32` | — | yes | Pioreactor OD |
| `reactor_ph` | `float32` | — | yes | probe if present |
| `light_state` | `string` | `on`\|`off`\|`ramp` | no | LED command |
| `room_t_c` | `float32` | °C | yes | room ambient sensor |
| `room_rh_pct` | `float32` | % RH | yes | room ambient sensor |
| `acquisition_status` | `string` | `OK`\|`SENSOR_TIMEOUT`\|`I2C_ERROR`\|`ADC_ERROR`\|`INTERPOLATED` | no | logger |

Non-`OK` `acquisition_status` values are preserved through preprocessing; downstream stages decide how to weight or exclude them.

### 17. Experiment metadata

`data/raw/experiments/{experiment_id}/meta.yaml` — required fields:

```yaml
experiment_id: exp_2026-07-15_batch03      # string, must match directory
started_at: 2026-07-15T09:00:00Z            # ISO8601 UTC
ended_at:   2026-07-15T21:00:00Z            # ISO8601 UTC, null if aborted mid-run
operator: name-or-id                        # string
campaign_id: 2026-Q3-spirulina-voc          # string
proposed_by:                                # optional; present if this run came from active learning
  tool: modeling
  acquisition_run: acq_2026-07-14_003
  point_index: 2
conditions:                                 # dict[reactor_id, dict[str, float]]
  R01: {par_umol_m2_s: 200, reactor_temp_c: 32, n_nano3_g_l: 2.5, inoculum_od: 0.6, nacl_add_g_l: 5.0}
  R02: {...}
sensor_assignment:                          # dict[sensor_id, reactor_id]
  PID01: R01
  PID02: R02
calibration_run: cal_2026-07-15_pre         # string; foreign key to §18
excluded_sensors: []                        # list[str]; sensors omitted from feature file
notes:                                      # list; freeform operator annotations
  - {t: "2026-07-15T14:22Z", text: "R03 LED flicker, replaced 14:30"}
```

Loaded via `jaxsr.logging_.load_metadata(experiment_id) -> ExperimentMeta` (pydantic).

### 18. Calibration result schema

`data/derived/calibrations/standard_addition/{calibration_run_id}.parquet` — one row per (calibration_run_id, sensor_id):

| Field | Type | Description |
|-------|------|-------------|
| `calibration_run_id` | `string` | e.g. `cal_2026-07-15_pre` |
| `experiment_id` | `string` | run this calibration serves |
| `sensor_id` | `string` | — |
| `calibration_compound` | `string` | e.g. "isoprene" |
| `response_factor` | `float64` \| null | RF relative to isobutylene |
| `mw_g_mol` | `float64` | compound molecular weight |
| `mean_sample_t_c` | `float64` | during calibration |
| `mean_sample_rh_pct` | `float64` | during calibration |
| `lamp_hours` | `float64` | at time of calibration |
| `b0_mv` | `float64` | intercept |
| `b1_mv_per_ppm_asgas` | `float64` | primary sensitivity in cal-gas units |
| `b1_mv_per_ppm_iso_equiv` | `float64` \| null | null iff `response_factor` is null |
| `b1_stderr` | `float64` | 1-σ from OLS on `b1_mv_per_ppm_asgas` |
| `r_squared` | `float64` | fit quality |
| `n_spike_points` | `int32` | ≥ 3 required |
| `status` | `string` | `PASS` \| `SUSPECT` \| `FAIL` |
| `fit_method` | `string` | `ols` \| `robust` \| `polynomial_deg2` |

A sidecar `{calibration_run_id}.yaml` records the full `CalibrationGas` including `source` / provenance string.

Loaded via `jaxsr.calibration.load_calibration(calibration_run_id) -> dict[str, SensitivityModel]`.

### 19. Derived feature schema

`data/derived/features/{campaign_id}/{experiment_id}.parquet`. This is what feeds `jaxsr.SymbolicRegressor.fit`. One row per `(experiment_id, reactor_id, sensor_id)` triple.

**VOC-related columns are compound-agnostic in their names**; the compound identity is carried as metadata, not baked into column names. If a downstream user changes the calibration gas mid-campaign, the tool refuses to concatenate the two batches without an explicit `--allow-mixed-calibration` flag.

Column list:

| Field | Type | Description |
|-------|------|-------------|
| `experiment_id`, `campaign_id`, `reactor_id`, `sensor_id` | string | keys |
| `analysis_window_start`, `analysis_window_end` | timestamp | inclusive |
| **Controllable variables** (replicated from experiment metadata) | | |
| user-defined per campaign (e.g. `par_umol_m2_s`, `reactor_temp_c`, ...) | float | knobs |
| **Observed covariates** | | |
| `mean_sample_t_c`, `mean_sample_rh_pct`, `mean_room_t_c`, `mean_room_rh_pct` | float | window means |
| `lamp_hours` | float | at window midpoint |
| **Primary VOC target** (compound-agnostic column names) | | |
| `mean_voc_ppm_asgas` | float | in units of the calibration compound |
| `mean_voc_ppm_asgas_stderr` | float | propagated uncertainty |
| `mean_voc_ppm_iso_equiv` | float / NaN | RF-corrected; NaN iff RF unknown |
| `mean_voc_ppm_iso_equiv_stderr` | float / NaN | propagated uncertainty |
| `p95_voc_ppm_asgas` | float | 95th percentile in cal-gas units |
| `voc_slope_ppm_asgas_h` | float | linear slope over window |
| **Secondary targets** | | |
| `dominant_freq_hz`, `dominant_freq_amp` | float | Lomb-Scargle strongest surviving peak |
| `final_od`, `growth_rate_h_inv` | float | biology |
| **Provenance** | | |
| `calibration_run_id` | string | which calibration |
| `calibration_compound` | string | compound name |
| `calibration_response_factor` | float / NaN | RF relative to isobutylene (NaN if unknown) |
| `covariate_model_id`, `preprocessing_config_hash` | string | traceability |
| `calibration_status`, `flags` | string / list<string> | health |

**Convenience loader:**

```python
X, y, feature_names = jaxsr.processing.load_features_for_jaxsr(
    features_df,
    target="mean_voc_ppm_asgas",       # or "mean_voc_ppm_iso_equiv"
)
```

extracts numeric feature columns (dropping IDs and provenance) and returns `X: np.ndarray`, `y: np.ndarray`, `feature_names: list[str]` ready for `SymbolicRegressor`. If the requested target column contains NaNs (e.g. `mean_voc_ppm_iso_equiv` for a calibration with unknown RF), the loader raises `TargetContainsNaNError` with a suggested alternative.

**Metadata attributes** on the returned DataFrame (`features_df.attrs`):

```python
{
    "calibration_compound": "isoprene",
    "calibration_response_factor": 0.63,
    "compound_agnostic": True,          # True iff all rows share the same compound
    "features_schema_version": 1,
    "provenance": {...},
}
```

The `compound_agnostic` flag flips to `False` if any join or concatenation mixed calibration compounds — a check that runs at load time.

---

## Part VII — Diagnostics Subpackage

Behavior and acceptance criteria are specified inline below. Each diagnostic writes results to `data/derived/diagnostics/{diagnostic_name}/{run_id}.parquet` and (where applicable) a diagnostic PNG to `reports/diagnostics/`.

### 20. Fleet-zero

```python
from jaxsr.diagnostics import run_fleet_zero, FleetZeroResult

result: FleetZeroResult = run_fleet_zero(
    duration_min=60,
    thresholds=None,   # loaded from configs/diagnostic_thresholds.yaml if None
    output_dir=Path("data/derived/diagnostics/fleet_zero"),
)
print(result.summary_status)          # "GREEN" | "YELLOW" | "RED"
for sensor_id, stats in result.per_sensor.items():
    print(sensor_id, stats)
```

### 21. Ambient baseline

```python
from jaxsr.diagnostics import run_ambient_baseline

baseline = run_ambient_baseline(duration_h=12, method="ols")
# baseline.covariate_models is a dict[sensor_id, CovariateModel]
# ready to feed jaxsr.processing.apply_covariate_correction
```

### 22. Sensor-swap Latin-square pilot

```python
from jaxsr.diagnostics import run_swap_pilot

swap_result = run_swap_pilot(n_blocks=4, block_hours=4)
print(swap_result.variance_share)   # {"sensor_id": 0.18, "reactor_id": 0.12, "residual": 0.70}
assert swap_result.variance_share["sensor_id"] < 0.30
```

### 23. Weekly audit and I²C scan

```python
from jaxsr.diagnostics import run_weekly_audit, scan_i2c

audit = run_weekly_audit(output_markdown=Path("reports/weekly/2026-W28.md"))
i2c_status = scan_i2c()   # dict[bus_address, "OK" | "TIMEOUT" | "ERROR"]
```

---

## Part VIII — Calibration Subpackage

### 24. Calibration gas specification

The calibration gas / standard is user-specified. The package ships with a built-in table of common VOCs and their response factors (RF) relative to isobutylene for a 10.6 eV PID lamp. Users can select a built-in compound, override any field, or register a custom compound.

**Built-in table location:** `jaxsr/calibration/data/response_factors.yaml`. Ships with isobutylene, isoprene, acetone, methanol, ethanol, DMS, toluene, benzene, and a handful of others. Sources cited per entry (Alphasense AAN 305, RAE TN-106).

**User overrides:** if a project's `configs/response_factors_overrides.yaml` exists, it is merged over the built-in table at load time. Same schema.

**Python API:**

```python
from jaxsr.calibration import CalibrationGas

# Option 1: from the built-in table
gas = CalibrationGas.builtin("isoprene")
# → CalibrationGas(name="isoprene", rf=0.63, mw=68.12, ie_ev=8.85,
#                  source="Alphasense AAN 305", is_builtin=True)

# Option 2: custom, with known response factor
gas = CalibrationGas.custom(
    name="my_analyte", mw=88.15, response_factor=0.75,
    source="internal empirical measurement, 2026-05-14",
)

# Option 3: custom, response factor unknown
gas = CalibrationGas.custom(name="my_analyte", mw=88.15)
# → response_factor is None; downstream reports in compound units only
```

**Dataclass:**

```python
@dataclass(frozen=True)
class CalibrationGas:
    name: str
    mw: float                          # g/mol
    response_factor: float | None       # relative to isobutylene; None if unknown
    ie_ev: float | None = None          # ionization energy, informational
    source: str = "user"
    is_builtin: bool = False

    @classmethod
    def builtin(cls, name: str) -> "CalibrationGas": ...
    @classmethod
    def custom(cls, name: str, mw: float,
               response_factor: float | None = None, **kw) -> "CalibrationGas": ...

    @property
    def has_rf(self) -> bool:
        return self.response_factor is not None
```

### 25. Interactive UI: gas selection at calibration start

When `jaxsr calibrate --experiment {id}` is invoked without `--calibration-gas`, the CLI presents an interactive menu built from the response-factor table plus an "Other" option. Example transcript:

```
$ jaxsr calibrate --experiment exp_2026-07-15_batch03

Standard-addition calibration
─────────────────────────────
Calibration gas / standard?
  [1] Isobutylene       (RF = 1.00, reference)
  [2] Isoprene          (RF = 0.63)
  [3] Acetone           (RF = 1.10)
  [4] Methanol          (RF = 10.0)
  [5] Ethanol           (RF = 10.0)
  [6] DMS               (RF = 0.44)
  [7] Toluene           (RF = 0.53)
  [8] Other — enter manually
Selection [1]: 2

Confirmed: isoprene, RF = 0.63 (Alphasense AAN 305).
  → Output reported in isobutylene-equivalent ppm (RF-corrected).
  → Also record compound-specific ppm alongside? [Y/n]: Y

Note: RF = 0.63 assumes a 10.6 eV lamp in good condition. Published
values drift as lamps age. For strict cross-lab quantification, measure
RF empirically. For DoE / optimization use, published values are usually
adequate.

Proceed with spike-and-recover? [Y/n]: Y
```

For scripted / non-interactive use, `--calibration-gas` accepts the name:

```bash
jaxsr calibrate --experiment exp_03 --calibration-gas isoprene
jaxsr calibrate --experiment exp_03 --calibration-gas custom \
                --compound-name myVOC --mw 88.15 --response-factor 0.75
jaxsr calibrate --experiment exp_03 --calibration-gas custom \
                --compound-name myVOC --mw 88.15   # RF unknown; warning printed
```

Both entry paths converge on a single `CalibrationGas` instance that is stored with the calibration.

### 26. Standard-addition procedure

```python
from jaxsr.calibration import run_standard_addition, fit_sensitivity_per_sensor

df = run_standard_addition(
    experiment_id="exp_2026-07-15_batch03",
    calibration_gas=gas,               # CalibrationGas instance (§24)
    spike_ppm_list=[1.0, 5.0, 20.0],   # in units of the calibration gas
    dwell_seconds=300,
    method="ols",                       # "ols" | "robust" | "polynomial_deg2"
)
models: dict[str, SensitivityModel] = fit_sensitivity_per_sensor(df, method="ols")
```

**Concentration accounting** — `spike_ppm_list` values are always in units of the calibration gas being injected. The fitted slope `b1_mv_per_ppm_asgas` is therefore in `mV per ppm of the calibration compound`. If `gas.has_rf`, the tool also derives `b1_mv_per_ppm_isobutylene_equiv = b1_mv_per_ppm_asgas × gas.response_factor`. Both are stored in the `SensitivityModel`.

**`SensitivityModel` fields:**

```python
@dataclass
class SensitivityModel:
    sensor_id: str
    calibration_gas: CalibrationGas             # full record; not just the name
    b0_mv: float
    b1_mv_per_ppm_asgas: float                   # slope in calibration-gas units
    b1_mv_per_ppm_isobutylene_equiv: float | None  # None iff gas.has_rf is False
    b1_stderr: float                             # in mV per ppm-asgas
    r_squared: float
    fit_method: Literal["ols", "robust", "polynomial_deg2"]
    mean_sample_t_c: float
    mean_sample_rh_pct: float
    lamp_hours: float
    status: Literal["PASS", "SUSPECT", "FAIL"]
```

**Persistence:**

```python
from jaxsr.calibration import persist_calibration
path = persist_calibration(
    models,
    calibration_run_id="cal_2026-07-15_pre",
    experiment_id="exp_2026-07-15_batch03",
    out_dir=Path("data/derived/calibrations/standard_addition"),
)
```

The Parquet file gains three metadata columns (`calibration_compound`, `response_factor`, `mw_g_mol`) replicated per row for join convenience, plus a file-level YAML sidecar recording the full `CalibrationGas` including its provenance/source string.

### 27. Reference jar rotation

The reference jar contains a permeation tube of some VOC — chosen by the user, recorded as a `CalibrationGas` at jar-setup time. All weekly comparisons then use the same compound. The tool refuses to compare readings across sensors calibrated to different reference compounds and prints a diagnostic message pointing to the mismatch.

```python
from jaxsr.calibration import run_reference_jar_rotation, compute_fleet_ratios

readings = run_reference_jar_rotation(
    sensors="all",
    dwell_min=10,
    reference_gas=CalibrationGas.builtin("isoprene"),  # must match jar contents
)
ratios = compute_fleet_ratios(readings)   # dict[sensor_id, ratio_to_fleet_median]
```

### 28. Calibration inversion (used by preprocessing)

```python
from jaxsr.calibration import apply_calibration

ppm_series, ppm_stderr_series, unit = apply_calibration(
    voltage=timeseries["pid_voltage_mv_notch_filtered"],
    sensor_id="PID01",
    sample_t_c=timeseries["sample_t_c"],
    sample_rh_pct=timeseries["sample_rh_pct"],
    calibration_run_id="cal_2026-07-15_pre",
    extrapolation_policy="clip",
    output_unit="isobutylene_equiv",   # or "as_calibrated" | "both"
)
# unit is a string describing the returned unit;
# when output_unit="both", ppm_series is a 2-column DataFrame instead of Series.
```

**Output unit semantics:**

- `"as_calibrated"` — ppm in units of the calibration compound. Always available.
- `"isobutylene_equiv"` — RF-corrected ppm relative to isobutylene. Available iff the calibration used a gas with a known RF. Requesting it when RF is unknown raises `CalibrationUnitUnavailableError`.
- `"both"` — a DataFrame with both columns; the second is NaN-filled when RF is unknown.

**Uncertainty propagation:**

```
Var(ppm_asgas) = Var(voltage) / b1_asgas² + (voltage / b1_asgas²)² × Var(b1_asgas)
Var(ppm_iso)   = Var(ppm_asgas) × RF²                     (RF assumed exact)
```

RF is treated as exact because published values do not carry stated uncertainty. If the user wants to propagate RF uncertainty, they can supply it in the CalibrationGas record via the optional `response_factor_stderr` field, and the tool will use it.

### 29. Optional: T/RH-aware sensitivity as a JAXSR symbolic model

When ≥ 3 calibrations exist across a range of T and RH, `apply_calibration` fits `b1(T, RH) = c0 + c1·T + c2·RH + c3·T·RH` by OLS by default. Users can opt into a symbolic fit:

```python
from jaxsr.calibration import build_sensitivity_surface

surface = build_sensitivity_surface(
    historic_calibrations=[cal1, cal2, cal3, cal4],
    method="symbolic",       # uses jaxsr.SymbolicRegressor
    max_terms=4,
)
# surface is a full jaxsr.SymbolicRegressor with fitted expression;
# apply_calibration uses it transparently if present.
```

This is where the pre-experimental subsystem starts *using* JAXSR primitives, not just feeding them. It's opt-in because for typical sensor networks the linear surface is sufficient and easier to interpret.

---

## Part IX — Preprocessing Subpackage

Each preprocessing step is a pure function on a polars/pandas DataFrame. Side effects (Parquet writes, plot generation) are confined to the CLI wrapper and the top-level `jaxsr process` command. The pipeline is applied in the order §28 → §29 → §30 → §31 → §32.

### 28. Covariate correction

```python
from jaxsr.processing import fit_covariate_model, apply_covariate_correction, CovariateModel

model: CovariateModel | None = fit_covariate_model(
    df=raw_timeseries,
    training_mask=raw_timeseries["timestamp"] < training_end,
    method="ols",                       # "ols" | "robust" | "symbolic"
    min_rh_range_pct=20.0,
)
corrected = apply_covariate_correction(raw_timeseries, {"PID01": model, ...})
```

When `method="symbolic"`, `fit_covariate_model` returns a `CovariateModel` whose internal fit is a `jaxsr.SymbolicRegressor` — the constraints (monotonicity, sign) become `jaxsr.Constraints` objects natively. This is the second natural touchpoint with existing JAXSR primitives.

### 29. Common-mode subtraction

```python
from jaxsr.processing import subtract_common_mode

result = subtract_common_mode(
    corrected,
    method="median",
    outlier_std_threshold=3.0,
    min_healthy_fraction=0.75,
)
# Raises jaxsr.processing.CommonModeInsufficientFleetError on failure
```

### 30. Spectral diagnostic + notch filter

```python
from jaxsr.processing import lomb_scargle, notch_filter_known_artifacts

freqs, power = lomb_scargle(t=timeseries["timestamp_s"], y=timeseries["signal"])
filtered, flags = notch_filter_known_artifacts(
    t=timeseries["timestamp_s"], y=timeseries["signal"],
    artifact_freqs=[0.00028, 0.00056], q=30.0,
)
```

### 31. Voltage-to-ppm conversion and feature extraction

```python
from jaxsr.processing import apply_calibration_series, extract_features

timeseries = apply_calibration_series(timeseries, sensitivity_models)
features_df = extract_features(timeseries, metadata,
                               analysis_window=(t_start, t_end))
```

`features_df` is the row-per-`(experiment, reactor, sensor)` table (schema §19).

---

## Part X — Bridge to Existing JAXSR DoE Workflow

The bridge is one function:

```python
from jaxsr.processing import load_features_for_jaxsr

X, y, feature_names = load_features_for_jaxsr(
    features_df,
    target="mean_voc_ppm_asgas",    # or "mean_voc_ppm_iso_equiv" when RF is known
    feature_columns=None,           # None → auto-detect controllable + covariates
    include_categorical=True,       # sensor_id, reactor_id as one-hot
    return_stderr=False,            # if True, returns (X, y, y_stderr, feature_names)
)
```

Everything past this call is existing JAXSR functionality. The subsystem's job ends here.

Optionally, when `return_stderr=True`, downstream JAXSR fitting can use per-observation weights `w_i = 1 / stderr_i²` in the OLS fit inside `SymbolicRegressor.fit`. This uses the uncertainty JAXSR already tracks internally and does not require a change to the SymbolicRegressor API — the weights are passed as an existing optional argument.

### 32. Provenance tracking

Every row in `features_df` carries provenance columns:

- `calibration_run_id` — which calibration was applied.
- `calibration_compound` — which VOC standard the calibration used.
- `calibration_response_factor` — RF relative to isobutylene, or NaN if unknown.
- `covariate_model_id` — which covariate correction was applied.
- `preprocessing_config_hash` — first 8 hex of SHA-256 of `configs/preprocessing.yaml` at process time.

`load_features_for_jaxsr` strips these before returning `X`, but they remain accessible via `features_df.attrs["provenance"]` for reproducibility.

**Mixed-compound guard:** if the loaded DataFrame contains rows with more than one distinct `calibration_compound`, the loader raises `MixedCalibrationCompoundError` unless `allow_mixed=True` is passed. The user has to acknowledge that comparing rows across compounds requires both to have known response factors and that comparability is only meaningful in the `_iso_equiv` columns.

---

## Part XI — CLI Surface

JAXSR gains a `jaxsr` command with new subcommands. If a `jaxsr` CLI already exists (⚠️ check), the new subcommands are added under it; otherwise a new top-level entry is registered in `pyproject.toml`.

```
jaxsr preflight
jaxsr calibrate --experiment {id}
jaxsr calibrate --reference-jar {--sensor {id} | --all}
jaxsr start --experiment {id}
jaxsr stop --experiment {id}
jaxsr note --experiment {id} "{text}"
jaxsr process --experiment {id} [--force-version N]
jaxsr diagnose fleet-zero --duration-min {N}
jaxsr diagnose ambient --duration-h {H} [--method ols|robust]
jaxsr diagnose swap-pilot [--n-blocks N]
jaxsr diagnose weekly-audit
jaxsr diagnose i2c
jaxsr dashboard
```

All CLI commands are thin wrappers around the Python API. Every operation is scriptable from Python without the CLI.

⚠️ If there is a naming collision with an existing `jaxsr` CLI, fall back to `jaxsr-cal`.

---

## Part XII — Repository Structure & Module Contracts

### 33. If deploying as extension package (recommended)

```
jaxsr-calibration/
├── README.md
├── pyproject.toml              # depends on jaxsr, plus optional deps
├── docs/
│   ├── augmentation_spec.md    # this document
│   ├── experimentalist_protocol.md
│   └── examples/
├── src/jaxsr_calibration/       # installs into jaxsr.* namespace via entry points
│   ├── __init__.py
│   ├── calibration/
│   ├── diagnostics/
│   ├── processing/
│   ├── logging_/
│   ├── cli.py
│   └── configs/                 # example configs
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

### 34. Module contracts (function signatures)

Public function signatures and dataclasses that downstream stages and tests will program against. Import paths are all under `jaxsr.calibration`, `jaxsr.diagnostics`, `jaxsr.processing`, or `jaxsr.logging_`. Changes to these signatures are breaking changes and require a minor version bump.

Key public dataclasses:

```python
# jaxsr.calibration.models
@dataclass(frozen=True)
class CalibrationGas:
    name: str
    mw: float                                  # g/mol
    response_factor: float | None              # relative to isobutylene; None if unknown
    response_factor_stderr: float | None = None
    ie_ev: float | None = None                 # informational
    source: str = "user"
    is_builtin: bool = False

    @classmethod
    def builtin(cls, name: str) -> "CalibrationGas": ...
    @classmethod
    def custom(cls, name: str, mw: float,
               response_factor: float | None = None, **kw) -> "CalibrationGas": ...

    @property
    def has_rf(self) -> bool:
        return self.response_factor is not None

@dataclass
class SensitivityModel:
    sensor_id: str
    calibration_gas: CalibrationGas
    b0_mv: float
    b1_mv_per_ppm_asgas: float                       # slope in cal-gas units
    b1_mv_per_ppm_iso_equiv: float | None            # None iff not gas.has_rf
    b1_stderr: float                                 # in mV per ppm-asgas
    r_squared: float
    fit_method: Literal["ols", "robust", "polynomial_deg2"]
    mean_sample_t_c: float
    mean_sample_rh_pct: float
    lamp_hours: float
    status: Literal["PASS", "SUSPECT", "FAIL"]

# jaxsr.processing.models
@dataclass
class CovariateModel:
    sensor_id: str
    method: Literal["ols", "robust", "symbolic"]
    # populated for ols / robust
    alpha: float | None
    beta_rh: float | None
    gamma_t: float | None
    delta_rh_t: float | None
    covariance: np.ndarray | None                     # 4×4
    # populated for symbolic
    symbolic_regressor: "jaxsr.SymbolicRegressor | None"
    # common
    training_window: tuple[datetime, datetime]
    r_squared: float

# jaxsr.diagnostics
@dataclass
class FleetZeroResult:
    per_sensor: dict[str, dict]                       # {"mean", "std", "slope", "status"}
    summary_status: Literal["GREEN", "YELLOW", "RED"]

@dataclass
class AmbientBaselineResult:
    covariate_models: dict[str, CovariateModel]
    r_squared_per_sensor: dict[str, float]

@dataclass
class SwapPilotResult:
    variance_share: dict[str, float]                  # {"sensor_id", "reactor_id", "residual"}
    mixedlm_summary: str                              # human-readable statsmodels output
```

Key public functions:

```python
# jaxsr.logging_.acquire
def start_logging(experiment_id: str, config: AppConfig) -> LoggerHandle: ...
def stop_logging(handle: LoggerHandle) -> Path: ...
def running_experiments() -> list[str]: ...

# jaxsr.calibration.standard_addition
def run_standard_addition(
    experiment_id: str,
    calibration_gas: CalibrationGas,
    spike_ppm_list: list[float],
    dwell_seconds: int = 300,
    method: Literal["ols", "robust", "polynomial_deg2"] = "ols",
) -> pl.DataFrame: ...

def fit_sensitivity_per_sensor(
    df: pl.DataFrame,
    method: str = "ols",
) -> dict[str, SensitivityModel]: ...

def persist_calibration(
    models: dict[str, SensitivityModel],
    calibration_run_id: str,
    experiment_id: str,
    out_dir: Path,
) -> Path: ...

# jaxsr.calibration.apply
def apply_calibration(
    voltage: pl.Series,
    sensor_id: str,
    sample_t_c: pl.Series,
    sample_rh_pct: pl.Series,
    calibration_run_id: str,
    extrapolation_policy: Literal["clip", "linear", "nan"] = "clip",
    output_unit: Literal["as_calibrated", "isobutylene_equiv", "both"] = "as_calibrated",
) -> tuple[pl.Series, pl.Series, str]:      # (ppm, ppm_stderr, unit_string)
    ...

# jaxsr.processing.covariate
def fit_covariate_model(
    df: pl.DataFrame,
    training_mask: pl.Series,
    method: Literal["ols", "robust", "symbolic"] = "ols",
    min_rh_range_pct: float = 20.0,
) -> CovariateModel | None: ...

def apply_covariate_correction(
    df: pl.DataFrame,
    models: dict[str, CovariateModel],
) -> pl.DataFrame: ...

# jaxsr.processing.common_mode
def subtract_common_mode(
    df: pl.DataFrame,
    method: Literal["median", "trimmed_mean"] = "median",
    outlier_std_threshold: float = 3.0,
    min_healthy_fraction: float = 0.75,
) -> pl.DataFrame:
    """Raises jaxsr.processing.CommonModeInsufficientFleetError if too few sensors survive."""

# jaxsr.processing.spectral
def lomb_scargle(
    t: np.ndarray, y: np.ndarray, freq_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...

def notch_filter_known_artifacts(
    t: np.ndarray, y: np.ndarray, artifact_freqs: list[float], q: float = 30.0,
) -> tuple[np.ndarray, list[str]]:
    """Returns (filtered_signal, list_of_flag_strings)."""

# jaxsr.processing.features
def extract_features(
    timeseries: pl.DataFrame,
    metadata: ExperimentMeta,
    analysis_window: tuple[datetime, datetime],
) -> pl.DataFrame:
    """One row per (experiment_id, reactor_id, sensor_id). Schema per §19."""

def load_features_for_jaxsr(
    features_df: pl.DataFrame,
    target: str = "mean_voc_ppm_asgas",
    feature_columns: list[str] | None = None,
    include_categorical: bool = True,
    return_stderr: bool = False,
    allow_mixed: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Bridge to jaxsr.SymbolicRegressor.

    Raises:
        MixedCalibrationCompoundError: if multiple compounds present and not allow_mixed.
        TargetContainsNaNError: if target column has NaN and no fallback is available.
    """

# jaxsr.diagnostics
def run_fleet_zero(duration_min: int,
                   thresholds: DiagnosticThresholds | None = None) -> FleetZeroResult: ...
def run_ambient_baseline(duration_h: int,
                         method: Literal["ols", "robust"] = "ols") -> AmbientBaselineResult: ...
def run_swap_pilot(n_blocks: int = 4, block_hours: int = 4) -> SwapPilotResult: ...
def run_weekly_audit(output_markdown: Path | None = None) -> WeeklyAuditResult: ...
def scan_i2c() -> dict[str, Literal["OK", "TIMEOUT", "ERROR"]]: ...
```

**Custom exception types** (importable from `jaxsr.calibration.errors` and `jaxsr.processing.errors`):

```python
class CalibrationUnitUnavailableError(ValueError): ...
class MixedCalibrationCompoundError(ValueError): ...
class TargetContainsNaNError(ValueError): ...
class CommonModeInsufficientFleetError(RuntimeError): ...
class TrainingWindowInsufficientError(RuntimeError): ...
```

---

## Part XIII — Task Backlog & Milestones

### Milestone 1 — Package scaffold + acquisition (Week 1)

- [ ] Create `jaxsr-calibration` package (or fork/branch), pyproject with `jaxsr[calibration]` extra.
- [ ] `jaxsr.calibration.config`, `jaxsr.processing.config` — Pydantic schemas + YAML loaders.
- [ ] `jaxsr.logging_.schema` — pyarrow schemas.
- [ ] `jaxsr.logging_.acquire` — PID + SHT35/BME280 + Pioreactor streams.
- [ ] `jaxsr.logging_.writer` — partitioned Parquet writer, atomic renames.
- [ ] CLI: `jaxsr preflight`, `jaxsr start`, `jaxsr stop`, `jaxsr note`.
- [ ] Unit tests for schemas and writer.

**Done when:** on a live rig, `jaxsr start → wait → jaxsr stop` produces valid partitioned Parquet.

### Milestone 2 — Diagnostics (Week 1–2)

- [ ] `jaxsr.diagnostics.fleet_zero`
- [ ] `jaxsr.diagnostics.ambient` (OLS + robust)
- [ ] `jaxsr.diagnostics.swap_pilot` (mixed-effects via statsmodels)
- [ ] `jaxsr.diagnostics.weekly` (markdown rollup)
- [ ] `jaxsr.diagnostics.i2c`
- [ ] Diagnostic PNG reports.

**Done when:** all `jaxsr diagnose ...` commands run on synthetic data and produce documented artifacts.

### Milestone 3 — Calibration (Week 2–3)

- [ ] `jaxsr.calibration.standard_addition` — interactive spike-and-recover.
- [ ] `jaxsr.calibration.reference_jar`.
- [ ] `jaxsr.calibration.apply` — voltage→ppm with T/RH lookup, uncertainty propagation via `jaxsr.uncertainty.prediction_interval`.
- [ ] `jaxsr.calibration.build_sensitivity_surface` with `method="ols"` and `method="symbolic"` (the latter uses `jaxsr.SymbolicRegressor` under the hood).
- [ ] Unit tests with synthetic voltage streams and known ground truth.

**Done when:** calibrate → apply → recover known ppm within stated uncertainty on synthetic data, with both linear and symbolic sensitivity surfaces.

### Milestone 4 — Preprocessing pipeline (Week 3–4)

- [ ] `jaxsr.processing.covariate` — OLS, robust, symbolic variants.
- [ ] `jaxsr.processing.common_mode`.
- [ ] `jaxsr.processing.spectral` — Lomb-Scargle + notch.
- [ ] `jaxsr.processing.convert` with uncertainty propagation.
- [ ] `jaxsr.processing.features` — extraction to §19 schema.
- [ ] `jaxsr.processing.load_features_for_jaxsr` — the bridge to existing JAXSR.
- [ ] CLI `jaxsr process`, idempotent versioning.
- [ ] End-to-end integration test.

**Done when:** on synthetic data, `jaxsr process` → `SymbolicRegressor.fit` runs cleanly and recovers a known underlying function within the stated uncertainty.

### Milestone 5 — Documentation + example notebook (Week 4)

- [ ] Notebook: full workflow from synthetic sensor stream → calibrated features → `SymbolicRegressor` fit → `ActiveLearner` suggestion.
- [ ] Docs pages under `docs/calibration/`: overview, tutorial, API reference.
- [ ] README additions listing the new extra `jaxsr[calibration]`.

### Milestone 6 — Dashboard + hardening

- [ ] Streamlit/Dash live monitor.
- [ ] systemd units for the Pi.
- [ ] Docker image for developer machines.
- [ ] Backup script.

---

## Part XIV — Testing Strategy

### 35. Test tiers

**Unit (fast, no I/O):**
- Every pure function in `calibration/`, `processing/`, `diagnostics/`.
- Property-based tests via `hypothesis` for schema round-trips and correction invariants.
- ≥ 90% coverage on pure-math modules.

**Integration (medium, synthetic hardware):**
- `tests/fixtures/synthetic_pid_stream.py` generates raw PID data with known ground truth (see below).
- Scenarios: constant true VOC + sinusoidal RH; square-wave VOC + constant RH; drifting sensor; dead sensor; two coincident sinusoids at different frequencies.
- End-to-end assertion: after the full pipeline, the recovered `mean_voc_ppm_asgas` matches ground truth within stated uncertainty. Tests parametrize over calibration compounds (isobutylene with RF=1.0, isoprene with RF=0.63, and a custom compound with unknown RF) to verify VOC-agnostic behavior.

**JAXSR-integration (medium):**
- Fit a `SymbolicRegressor` on features extracted from synthetic data with a known underlying emission function.
- Assert the recovered symbolic expression matches (up to reordering / renaming) the ground-truth expression.
- Assert `ActiveLearner.suggest` returns points inside the specified bounds.

**Regression (fast):**
- Golden fixtures in `tests/fixtures/golden/`. Any change to preprocessing math altering these outputs requires an intentional bump of `preprocessing_schema_version`.

**Hardware-in-loop (slow, opt-in):**
- Marked `@pytest.mark.hardware`, skipped by default. Requires a real PID + reference source.

### 36. Synthetic PID stream fixture

```python
def synthetic_pid_stream(
    true_ppm: Callable[[float], float],
    rh_profile: Callable[[float], float],
    t_profile: Callable[[float], float],
    b0: float, b1: float,
    noise_std: float,
    drift_mv_per_hour: float,
    rh_gain_coef: float,           # β in the covariate model
    t_gain_coef: float,             # γ
    rh_t_gain_coef: float,          # δ
    duration_s: int,
    dt_s: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame: ...
```

Used to test every stage independently and in composition.

---

## Appendices

### Appendix A — Preprocessing config hash

Short hash (first 8 hex chars of SHA-256) of the canonicalized JSON serialization of `configs/preprocessing.yaml` at process time. Canonicalization: sorted keys, no whitespace, ISO datetimes.

### Appendix B — Open decisions

1. **Deployment model** — upstream contribution, extension package, or fork (§5). Default: extension package.
2. **CLI namespace** — extend `jaxsr` or use `jaxsr-cal`. Default: extend `jaxsr` if unclaimed.
3. **Exact Alphasense PID model** for each sensor slot.
4. **Target VOC** — the pipeline is VOC-agnostic; users specify at calibration time. No project-level default is required. Users should confirm their calibration compound has a published response factor if they want isobutylene-equivalent output.
5. **Number of reactors/sensors** at Milestone 1.
6. **Whether the covariate correction defaults to `ols` or `symbolic`** — my recommendation: default `ols`; users opt into `symbolic` when they want interpretability.
7. **Backup and archival policy** for raw Parquet data.
8. **Coordination with the Kitchin group** on upstream contribution (only if deployment model A is chosen).

### Appendix C — Relationship to the experimentalist protocol

The experimentalist protocol (`spirulina_voc_experimentalist_protocol.md`) is unchanged in content but has been updated to use `jaxsr ...` CLI commands instead of `spirulina-voc ...`. The lab workflow is identical; only the command names moved.
