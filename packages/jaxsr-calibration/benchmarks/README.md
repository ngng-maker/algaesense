# Pre-calibration + JAXSR benchmark

This project has two genuinely different notions of "ground truth," and this benchmark tests both separately rather than conflating them:

- **Ground truth #2 -- how VOC varies ACROSS many different static settings.** Run many separate experiments, each at its own fixed (PAR, temp) setpoint, and ask whether a function captures how the resulting VOC value varies as the *setting itself* changes. This is the domain `suggest_next_experiments`/JAXSR active learning operates in. Tested by `calibration_recovery.py` (Test 1) and `doe_comparison.py` (Test 2).
- **Ground truth #1 -- the dynamic response within one experiment.** Given a single fixed static PAR level held for a whole experiment, VOC(t) unfolds over *time* on its way to a steady state. This is the domain `discover_led_response_dynamics`/`jaxsr.discover_dynamics` operates in, and is completely untested by Tests 1/2. Tested by `dynamics_recovery.py` (Test 3).

Not part of the pytest suite -- this is an analysis/report, not a pass/fail correctness test.

1. **`calibration_recovery.py`** (Test 1) -- does running raw, deliberately-contaminated sensor data through the real `jaxsr_calibration` diagnostics/calibration pipeline (fleet-zero, ambient-baseline covariate correction, common-mode subtraction, standard-addition calibration) actually recover the true underlying `VOC(PAR, temp)` function better than using raw voltage directly? Also includes a cross-sensor consistency check (Test 1b): 3 sensors observing the SAME condition simultaneously, each with its own distinct systematic-drift shape (a sluggish/flat sensor, a zigzag-rising sensor, a smooth-curve sensor) -- does correction make them converge with each other and with the truth, or does a real residual remain (since these drift shapes aren't the kind of contamination fleet-zero/ambient-baseline are built to model)?
2. **`doe_comparison.py`** (Test 2) -- given a shared 10-experiment budget, how many experiments does the real `suggest_next_experiments`/`suggest_next_experiments_with_context` active-learning workflow (informed by a labwiki finding via `bound_overrides`) actually NEED to characterize that true function to a target accuracy, vs. classic Latin Hypercube / Sobol / Grid / Random designs? Also reports a second, distinct metric (best conditions actually found) since the two can disagree about which method "wins."
3. **`dynamics_recovery.py`** (Test 3) -- runs several SEPARATE, independent experiments, each held at its own fixed static PAR level for >= 1 week (no single continuously-varying profile, and none pooled together before fitting -- see the module's own docstring for why), and asks whether the real `discover_led_response_dynamics` recovers each experiment's own within-experiment relaxation dynamics (a first-order lag toward Test 1/2's own static surface as its steady-state target) from just that one experiment's noisy trajectory. Also compares calling the real tool with vs. without its `ambient_baseline_run_id` parameter.

`ground_truth.py` defines the one nonlinear static "true" function (additive main effects -- a saturating light response and a linear temperature effect -- plus ONE genuine bilinear PAR x temperature interaction term, and a mild high-PAR photoinhibition decline), the dynamic relaxation law tied to it, and every synthetic noise generator (fleet-zero-style per-sensor bias, ambient RH/T covariate contamination, shared common-mode artifact, AR(1) autocorrelated noise). `doe_methods.py` holds the point-selection strategies for Test 2.

**On the functional form (2026-07-23):** an earlier version used a MULTIPLICATIVE `light_term * exp(BETA_T*(temp-TEMP_REF))` temperature modulation, which caused the interaction coefficient (`gamma`) to recover at 44-67% error -- reported at the time as "a real statistical limitation of this specific functional form." That diagnosis was wrong: the multiplicative term's own linearization is the exact same shape as the "interaction" term, a genuine design bug, not an inherent limit on testing interactions. The current additive-plus-bilinear-interaction form has no such collision, and `gamma` now recovers to <0.1% error. See CLAUDE.md's dev log for the full story and the general lesson.

**On Test 2's metric (2026-07-23):** originally reported ONLY as "how accurate is each method at a fixed 10-experiment budget." Reframed to lead with SPEED -- how many experiments a method actually needs to reach a target accuracy -- since that's usually the more practically relevant question; the old fixed-budget framing is still reported as supporting context.

**On Test 3's redesign (2026-07-23):** originally ran ONE experiment under a continuously-varying sinusoidal PAR(t) profile. Redesigned, at the user's explicit request, to several SEPARATE static-PAR step experiments (>= 1 week each), deliberately NOT pooled together before fitting. Real, honest consequences of this change, not glossed over: (1) within any single experiment PAR is literally constant, so a selected `reactor_par_umol_m2_s` term proves nothing about "detecting a PAR effect" the way it could when PAR genuinely varies within a run; (2) the true linear self-decay term is recovered less reliably than in the old sinusoid design (an honest, disclosed regression, not something tuned away); (3) some discovered equations are numerically unstable when integrated forward over the full week (a locally-accurate model isn't guaranteed to be globally stable -- a known, real SINDy-family limitation); (4) ambient-baseline correction, which reliably helped in the old design, does NOT reliably help here -- the report states whichever direction the actual numbers show, rather than assuming the old finding carries over. See REPORT.md's own Test 3 section and CLAUDE.md's dev log for the full detail.

**On Test 1b, cross-sensor consistency (2026-07-24):** added at the user's request to test a genuinely different question: do 3 sensors with wildly different-LOOKING raw response shapes (even when observing the identical true condition) converge after correction, and does the true value differ between them even once denoising is done? `ground_truth.py`'s `sluggish_flat_artifact_mv`/`zigzag_rising_artifact_mv`/`curvy_drift_artifact_mv` inject three physically-motivated, distinct time-varying drift shapes (a lagged/damped response, a periodic connector/thermal-cycling artifact, a slow aging-style exponential drift) on top of the same ambient-covariate/AR(1) noise every other recording uses. Result: correction meaningfully reduces cross-sensor disagreement but does NOT eliminate it -- a real, honest residual remains, because these drift shapes aren't the constant-bias/linear-RH-T contamination the pipeline actually models. See `cross_sensor_consistency.png` and REPORT.md's own section for the exact numbers.

## Running it

Needs `jaxsr-calibration` installed with its `benchmarks` extra (pulls in `matplotlib`, the only dependency this script needs beyond the package's own core deps):

```
pip install -e "packages/jaxsr-calibration[benchmarks]"
```

Then:

```
.venv/Scripts/python.exe packages/jaxsr-calibration/benchmarks/run_all.py
```

Takes 2-3 minutes (Test 3's week-long, multi-experiment static-PAR redesign is the slow part). Writes `results/calibration_recovery.png`, `results/cross_sensor_consistency.png`, `results/doe_comparison.png`, `results/dynamics_recovery.png`, and `results/REPORT.md`. The report includes the ground-truth equation and every noise source's exact parameters, a table of the actual generated data, an explanation of every metric used, and honest caveats about parameter identifiability and `jaxsr`'s own fit non-determinism at small sample sizes -- read it before quoting a specific number from this benchmark, the caveats matter as much as the headline verdict.

To just re-check one test alone (e.g. while iterating), run `calibration_recovery.py`, `doe_comparison.py`, or `dynamics_recovery.py` directly -- each has its own `if __name__ == "__main__"` printout.

## What this benchmark does NOT claim

- It's one synthetic ground-truth function, not a general proof the workflow always beats DoE (or vice versa) on real biological data -- the point is to test the pipeline's real mechanics (does correction actually help, does the active-learning/labwiki wiring actually work end to end), not to produce a universal ranking of experimental-design methods.
- Every method here fits with the SAME `jaxsr.SymbolicRegressor` basis library `mcp_pipeline/pipeline.py` itself uses, so the comparison isolates "which points get chosen," not "whose model is better."
