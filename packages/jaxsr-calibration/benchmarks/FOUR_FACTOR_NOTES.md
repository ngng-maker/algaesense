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

## Status

- [x] 4D ground truth committed
- [x] decoys screened, recoverability confirmed
- [x] campaign runner generalised to N factors (self-contained in discovery_speed_4d.py)
- [x] truth made smooth (Haldane), decoys re-screened, recoverability re-confirmed
- [ ] 4D run executed
- [ ] results verified and reported
