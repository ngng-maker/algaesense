# Pre-calibration + JAXSR benchmark

This project has two genuinely different notions of "ground truth," and this benchmark tests both separately rather than conflating them:

- **Ground truth #2 -- how VOC varies ACROSS many different static settings.** Run many separate experiments, each at its own fixed (PAR, temp) setpoint, and ask whether a function captures how the resulting VOC value varies as the *setting itself* changes. This is the domain `suggest_next_experiments`/JAXSR active learning operates in. Tested by `calibration_recovery.py` (Test 1) and `doe_comparison.py` (Test 2).
- **Ground truth #1 -- the dynamic response to ONE specific time-varying profile.** Given a single, specific PAR(t)/temp(t) trajectory within one experiment, VOC(t) unfolds over *time*. This is the domain `discover_led_response_dynamics`/`jaxsr.discover_dynamics` operates in, and is completely untested by Tests 1/2. Tested by `dynamics_recovery.py` (Test 3).

Not part of the pytest suite -- this is an analysis/report, not a pass/fail correctness test.

1. **`calibration_recovery.py`** (Test 1) -- does running raw, deliberately-contaminated sensor data through the real `jaxsr_calibration` diagnostics/calibration pipeline (fleet-zero, ambient-baseline covariate correction, common-mode subtraction, standard-addition calibration) actually recover the true underlying `VOC(PAR, temp)` function better than using raw voltage directly?
2. **`doe_comparison.py`** (Test 2) -- given a fixed 10-experiment budget, does the real `suggest_next_experiments`/`suggest_next_experiments_with_context` active-learning workflow (informed by a labwiki finding via `bound_overrides`) characterize that same true function faster than classic Latin Hypercube / Sobol / Grid / Random designs?
3. **`dynamics_recovery.py`** (Test 3) -- given one real sinusoid control-profile schedule, does the real `discover_led_response_dynamics` recover the true within-experiment dynamic law (a first-order lag toward Test 1/2's own static surface as its steady-state target)? Also compares calling the real tool with vs. without its `ambient_baseline_run_id` parameter (added directly to the production tool after this benchmark first surfaced the gap -- see CLAUDE.md's dev log).

`ground_truth.py` defines the one nonlinear static "true" function (additive main effects -- a saturating light response and a linear temperature effect -- plus ONE genuine bilinear PAR x temperature interaction term, and a mild high-PAR photoinhibition decline), the dynamic relaxation law tied to it, and every synthetic noise generator (fleet-zero-style per-sensor bias, ambient RH/T covariate contamination, shared common-mode artifact, AR(1) autocorrelated noise). `doe_methods.py` holds the point-selection strategies for Test 2.

**On the functional form (2026-07-23):** an earlier version used a MULTIPLICATIVE `light_term * exp(BETA_T*(temp-TEMP_REF))` temperature modulation, which caused the interaction coefficient (`gamma`) to recover at 44-67% error -- reported at the time as "a real statistical limitation of this specific functional form." That diagnosis was wrong: the multiplicative term's own linearization is the exact same shape as the "interaction" term, a genuine design bug, not an inherent limit on testing interactions. The current additive-plus-bilinear-interaction form has no such collision, and `gamma` now recovers to <0.1% error. See CLAUDE.md's dev log for the full story and the general lesson.

## Running it

```
.venv/Scripts/python.exe packages/jaxsr-calibration/benchmarks/run_all.py
```

Takes about a minute. Writes `results/calibration_recovery.png`, `results/doe_comparison.png`, `results/dynamics_recovery.png`, and `results/REPORT.md`. The report includes the ground-truth equation and every noise source's exact parameters, a table of the actual generated data, an explanation of every metric used, and honest caveats about parameter identifiability and `jaxsr`'s own fit non-determinism at small sample sizes -- read it before quoting a specific number from this benchmark, the caveats matter as much as the headline verdict.

To just re-check one test alone (e.g. while iterating), run `calibration_recovery.py`, `doe_comparison.py`, or `dynamics_recovery.py` directly -- each has its own `if __name__ == "__main__"` printout.

## What this benchmark does NOT claim

- It's one synthetic ground-truth function, not a general proof the workflow always beats DoE (or vice versa) on real biological data -- the point is to test the pipeline's real mechanics (does correction actually help, does the active-learning/labwiki wiring actually work end to end), not to produce a universal ranking of experimental-design methods.
- Every method here fits with the SAME `jaxsr.SymbolicRegressor` basis library `mcp_pipeline/pipeline.py` itself uses, so the comparison isolates "which points get chosen," not "whose model is better."
