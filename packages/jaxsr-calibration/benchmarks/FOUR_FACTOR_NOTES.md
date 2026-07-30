# Four-factor discovery study — working notes

Written so this survives a context reset. If you are picking this up cold,
read this file first, then `discovery_speed_4d.py`.

## The question

Every 2D result says adaptive experiment design buys **no speed advantage**
over Sobol, only better extrapolation (~3 ppm vs 8–12). The standing
hypothesis for why: two factors is too small a space for point *choice* to
matter, because covering it evenly is already nearly optimal and that is
exactly what a space-filling design does.

This study tests that by moving to four factors, where even coverage stops
being cheap.

## What is already established (do not re-derive)

- 2D discovery needs ~17 experiments. 4D needs ~150. Measured, not guessed.
- 4D recoverability confirmed before building: exact 7-term recovery at 150
  points with realistic noise; at 80 points all true terms plus one decoy.
- Decoys were screened for collinearity against the true terms over the real
  operating ranges. `ph_x_temp` (0.995 vs linear temp) and `log_nut` (0.986
  vs the true nitrate term) were REJECTED as near-duplicates. Usable: `x2`,
  `x3`, `par_hump`, `temp_hump`, `nut_hump`, `inv_par`.
- Linear PAR (0.953 vs `sat_par`) is deliberately KEPT — it is the real
  experimental question, not an artifact.

## Smoothness � why the truth changed

Active learning assumes a SMOOTH response: it fits a model to what it has
seen and picks the next point from that model, which is only sound if
there are no kinks to blindside it. The photoinhibition term inherited
from the two-factor truth, `max(par - 380, 0)^2`, is C1 but NOT C2 -- its
second derivative jumps at the threshold (measured: 0.0208) -- and it is
identically zero across 90% of the light range.

Replaced with **Haldane substrate-inhibition kinetics**,
`VMAX * par / (K_M + par + par^2/K_I)`, which is the standard
photoinhibition model, is infinitely differentiable for par >= 0, and
declines at high light for the same physical reason. Verified: no
derivative discontinuity at orders 1-3.

This also gives a better discrimination test. The non-inhibited Monod
form is now a decoy, and telling it from Haldane requires sampling past
the turnover -- a question about the shape of a smooth curve rather than
about finding a kink.

**Geometry chosen by measurement.** With the turnover at 400 and light
ending at 420, only 5% of the range lay past the peak and Monod
correlated 0.984 with Haldane -- unwinnable. Turnover moved to 300 and
the range extended to 500: 42% past the peak, correlation 0.911. Hard but
answerable.

## Expensiveness

Satisfied by construction rather than by simulation cost: the metric is
experiments-to-discovery, so each evaluation is treated as the scarce
resource even though computing it is instant. Nothing to change.

## A real property found while checking recoverability

Discovery has an UPPER window as well as a lower one. Exact recovery at
100 and 150 points; at 300 and 800 the fit starts adding terms
(`inv_par`, `x3`, `temp_hump`). More data does not monotonically help --
jaxsr's selection over-includes once there is enough of it. MAX_EXPERIMENTS
is 180 partly for runtime and partly because beyond that the criterion
stops being reachable.

## Ground truth

`ground_truth.true_voc_ppm_4d_smooth(par, temp, ph, nutrient)` (the hinged
`true_voc_ppm_4d` is superseded and kept only for reference) — the existing 2D
truth plus a pH optimum and a Monod nitrate saturation. The 2D
`true_voc_ppm` is untouched; Tests 1–3 and the 2D benchmark depend on it.

True term set in the fitted basis (7 terms):

    {1, haldane_par, x1, par_x_dtemp, ph_hump, sat_nut}

## Traps already hit in this benchmark family — do not repeat

1. **Decoys that are not decoys.** Measure every distractor's correlation
   against every true term over the ACTUAL operating range before running.
   A collinear distractor makes strict discovery impossible by construction
   and fails silently, by making every method look equally bad.
2. **Survivorship bias.** Report the convergence count BEFORE any mean. A
   mean over the runs that converged is not comparable to a mean over all
   runs — the corners seed looked fastest purely because it averaged 12 of
   24.
3. **Ranking noise.** With spreads of ±4–7 experiments, a 1–2 experiment
   difference is nothing. Do not present a sorted table; state the overlap
   first, then use paired per-seed differences with a 95% interval.
4. **Discarding dispersion.** Never run the comparison with verbose off and
   no CSV. `_save_raw` exists for this reason.
5. **Clobbered outputs.** A focused run overwrote the full run's plot. Use
   distinct output names per study.

## Design decisions for this study

- Seed: Latin-square layout across all four factors. The 2D study showed
  corners-only halves the adaptive convergence rate, and clustered confines
  the search; Latin covers the interior and was 24/24.
- Labwiki declared bounds are always supplied to the adaptive arms — the 2D
  study showed 0/24 without them regardless of seed.
- Arms kept deliberately small, because each adaptive campaign is ~150
  sequential suggest-and-fit rounds: Sobol, Latin Hypercube, D-optimal +
  labwiki, model-discrimination + labwiki.
- `MAX_EXPERIMENTS = 200`, `MIN_EXPERIMENTS = 20` (7 terms cannot be
  determined below that).

## SUPERSEDED: the FIRST 4D run was not usable (kept for the lesson)

Raw result (6 repeats): Latin Hypercube 0/6, Sobol 0/6, Grid 6/6 at 32.7,
Random 1/6, D-optimal+labwiki 6/6 at 32.0, model-discrimination 6/6 at
39.3, Sobol-warmup-then-D-optimal 5/6 at 80.8.

**That headline is an artifact.** Diagnosed by printing the selected terms
rather than trusting the score: Sobol and Latin Hypercube find EVERY true
term at every budget from 40 to 160 experiments. They never miss one. They
fail only because they persistently also select `inv_par`, the
`1/(1+light)` decoy -- and no amount of extra data removes it. Their
prediction errors give it away: Sobol scores the best surface (0.20 ppm)
and best extrapolation (0.26 ppm) of any method while nominally having
failed outright.

`inv_par` measured 0.899 against the true Haldane term, just under the 0.9
cut-off, so it was kept. That was the wrong instrument. A fixed
correlation threshold does not answer the real question, which is whether
ANY design can exclude the decoy at realistic noise. Same trap as the
first 4D attempt where `temp^2` sat at 0.996 -- moving the threshold from
0.95 to 0.9 simply let a decoy through underneath it.

Grid escapes only because its coarse 3-4 levels per factor give it less
resolution at low light, which is where `inv_par` and Haldane differ. It
is passing for the wrong reason.

### Required before re-running

1. **Drop `inv_par`** -- justified by evidence (no design excluded it at any
   budget), not by a threshold.
2. **Report recall beside strict discovery.** "Found every true term" and
   "found every true term and nothing else" are different claims, and this
   run is the case that proves conflating them misleads: every method
   except Random had perfect recall throughout.
3. Re-screen any remaining decoy the same way -- by asking whether a
   well-resourced design can actually reject it, not by its correlation.

## Status

- [x] 4D ground truth committed
- [x] decoys screened, recoverability confirmed
- [x] campaign runner generalised to N factors (self-contained in discovery_speed_4d.py)
- [x] truth made smooth (Haldane), decoys re-screened, recoverability re-confirmed
- [x] 4D run executed -- RESULT NOT USABLE, see section above
- [x] inv_par removed, recall metric added, run repeated
- [x] results verified and reported

## FINAL 4D RESULT (6 repeats, inv_par removed, both metrics)

Exact structural discovery, then all-true recall:

  Grid                            exact 6/6 @ 30.0 (sd  8.7)   all-true 6/6 @ 21.3
  Model-discrimination + labwiki  exact 6/6 @ 37.3 (sd  9.4)   all-true 6/6 @ 20.0
  D-optimal + labwiki             exact 6/6 @ 39.3 (sd 21.2)   all-true 6/6 @ 20.7
  Sobol warm-up then D-optimal    exact 6/6 @ 72.0 (sd 37.9)   all-true 6/6 @ 24.0
  Random                          exact 3/6 @ 40.0 (sd 20.0)   all-true 6/6 @ 25.3
  Latin Hypercube                 exact 1/6 @ 48.0             all-true 6/6 @ 26.0
  Sobol                           exact 0/6                    all-true 6/6 @ 25.3

Prediction error (surface / extrapolation, ppm): Sobol 0.19/0.33 is the
BEST of any method despite scoring 0/6 on exact discovery.

### What is and is not supported

SUPPORTED -- convergence rate differs, decisively. 6/6 for Grid and all
three adaptive arms against 0/6 Sobol, 1/6 Latin Hypercube, 3/6 Random.
This reverses every two-factor result and is far too large to be noise.

SUPPORTED -- the Sobol warm-up hybrid costs more than it saves: 72.0
experiments against 39.3 for plain D-optimal, the only gap wide enough to
clear the scatter.

NOT SUPPORTED -- any ordering among the converging methods. Grid 30.0 and
D-optimal 39.3 carry standard deviations of 8.7 and 21.2 across six
repeats. Do not rank these without many more repeats.

NOT SUPPORTED -- "adaptive design learns the relationship faster". Every
method reached all-true recall in 20-26 experiments with no separation.
What differs is PARSIMONY: whether a method also drags along a wrong
light-response term (x0 or sat_par). That is a narrower and more accurate
claim than the convergence counts alone suggest.

## STOP — a fourth confound, and the one that drives the reported numbers

`jaxsr.SymbolicRegressor` defaults to `strategy='greedy_forward'`. Every fit
in every run of this benchmark family has used it, and the discovery
criterion measures WHICH TERMS GET SELECTED -- so the design comparison has
been measuring greedy selection's failure, not the designs.

Measured directly on Sobol at 48 experiments, identical data:

    greedy_forward + bic : exact=False, extra=['sat_par']
    greedy_forward + aic : exact=False, extra=['sat_par']
    exhaustive     + bic : exact=True
    exhaustive     + aic : exact=True

The information criterion changes nothing. The search strategy changes
everything. So "Sobol 0/6" was greedy selection failing on data that fully
determined the answer.

This project already learned this: the 2026-07-23 dev log records switching
`discover_dynamics` from greedy_forward to exhaustive, which tripled R^2 and
recovered a term greedy had been missing. It was not carried across.

### Required next, in order

1. Set `strategy="exhaustive"` on EVERY fit -- both `_fit` (scoring) and the
   learner's internal model via `suggest_next_experiments`. Check
   tractability first: 11 basis terms choose up to 8 should be fine, but
   confirm rather than assume.
2. Re-run. Recorded expectation so it can be checked against: the
   space-filling designs will converge, and the four-factor result will look
   much less like an adaptive-design win than currently reported.
3. Only then interpret the AdaptiveSampler (leverage/gradient) and
   Box-Behnken/central-composite arms -- their numbers carry the same
   confound.

### Full list of confounds found in the 4D work, all still to be cleared

- Learner reasoned over a degree-2 polynomial surrogate that cannot
  represent Haldane, the pH quadratic, or Monod nitrate. FIXED (basis_library
  parameter added), not yet re-run.
- Baselines were scipy reimplementations while jaxsr ships its own samplers
  AND the real classical designs. FIXED, not yet re-run.
- `jaxsr.AdaptiveSampler` (leverage, gradient strategies) was never used.
  ADDED, not yet run.
- Term selection defaulted to greedy_forward. NOT YET FIXED -- do this first.
