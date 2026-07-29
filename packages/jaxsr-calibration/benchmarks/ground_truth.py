"""The one 'true' VOC-response function this whole benchmark is built
around, plus synthetic raw-sensor data generators that inject the
specific noise sources this package's diagnostics are meant to remove
(a per-sensor fleet-zero-style bias, an ambient-RH/T covariate
contamination, a shared common-mode artifact, and autocorrelated
sensor noise) on top of it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import polars as pl


"""
Nothing here is meant to be biologically exact -- it is a stand-in
"real physics" the rest of the benchmark treats as ground truth, chosen
to be genuinely nonlinear (a saturating light response, a temperature
main effect, a genuine light-temperature interaction, and a high-PAR
photoinhibition decline) without being so exotic that no
polynomial-basis symbolic regressor could ever approximate it. Every
benchmark method (ours and every DoE baseline) fits the SAME jaxsr
basis library against this same truth, so the basis's own
expressiveness is a constant across the comparison, not a confound --
what varies is only which (PAR, temp) points got sampled.

**Design note (2026-07-23): an earlier version of this function used a
MULTIPLICATIVE light_term * exp(BETA_T*(temp-TEMP_REF)) temperature
modulation, plus a separate GAMMA*par*(temp-TEMP_REF)/(K_M+par)
"interaction" term.** Test 1's own benchmark run then showed `gamma`
recovering at 44-67% error every time, blamed at the time on "genuine
collinearity... a real statistical limitation of this specific
functional form." That diagnosis was correct but incomplete -- the
actual root cause was a real DESIGN BUG, not an unavoidable property of
testing interactions at all: linearizing exp(x) around x=0 gives
`light_term * (1 + BETA_T*(temp-TEMP_REF) + ...)`, so the multiplicative
term's own first-order Taylor expansion already contains
`light_term * BETA_T * (temp-TEMP_REF)` -- the EXACT SAME shape (up to
a constant) as the "interaction" term that was supposedly independent
of it. Two coefficients (`VMAX*BETA_T` and `GAMMA`) were being fit
against what is, to leading order, a single basis function -- textbook
non-identifiability, not a fundamental limit. The fix below uses
additive main effects (so there is no multiplicative term to linearize
into a collinear shape) plus ONE genuine bilinear interaction term
(`GAMMA * par * (temp - TEMP_REF)`) that has no other term in the
function proportional to it. Verified directly: the same curve_fit
recovery test that used to show 44-67% gamma error now recovers it to
<0.1% error. See CLAUDE.md's dev log for the full before/after.
"""

PAR_BOUNDS = (0.0, 500.0)
TEMP_BOUNDS = (20.0, 40.0)

VMAX = 800.0
"""Maximum light-driven VOC output (ppm) as PAR -> infinity."""
K_M = 150.0
"""Half-saturation PAR (umol/m^2/s) -- light_term reaches VMAX/2 here."""
TEMP_REF = 28.0
"""Reference temperature (degC) both the temperature main effect and
the interaction term are centered on -- not itself a fitted parameter,
same role as a fixed intercept-centering choice."""
TEMP_SLOPE = 3.0
"""Main effect of temperature alone (ppm per degC away from TEMP_REF) --
additive, not multiplicative, so it has no linearization that collides
with the interaction term below."""
GAMMA = 0.05
"""The genuine PAR x temperature interaction (ppm per umol/m^2/s per
degC) -- a plain bilinear term with no other term in this function
proportional to it, unlike the earlier design's collinear version."""
BASELINE = 30.0
"""A small constant offset -- keeps the function non-negative across
the whole (PAR, temp) domain now that temperature has its own additive
effect even at PAR=0 (a real, if modest, dark/respiration-driven VOC
baseline that varies with temperature is physically plausible)."""

"""
A mild photoinhibition decline above PHOTO_THRESHOLD_PAR -- Spirulina
does genuinely suffer photoinhibition at high light intensity (this
project's own hardware protocol already treats very high lux as a
safety concern, see LEDActuator's docstring history). This gives Test
2's labwiki-informed bound_overrides demonstration a real, physically
grounded finding to narrow around, rather than an arbitrary one.
"""
PHOTO_THRESHOLD_PAR = 380.0
PHOTO_K = 0.0104


def true_voc_ppm(par, temp):
    """The ground-truth mean VOC output (ppm) for a static (PAR, temp)
    setpoint -- what every method in this benchmark is trying to
    characterize using as few real experiments as possible.

    VOC(PAR, temp) = BASELINE
                    + VMAX * PAR / (K_M + PAR)                  [saturating light main effect]
                    + TEMP_SLOPE * (temp - TEMP_REF)             [temperature main effect]
                    + GAMMA * PAR * (temp - TEMP_REF)            [genuine PAR x temp interaction]
                    - PHOTO_K * max(PAR - PHOTO_THRESHOLD_PAR, 0)^2   [high-PAR photoinhibition]
    """
    par = np.asarray(par, dtype=float)
    temp = np.asarray(temp, dtype=float)
    light_term = VMAX * par / (K_M + par)
    temp_term = TEMP_SLOPE * (temp - TEMP_REF)
    interaction = GAMMA * par * (temp - TEMP_REF)
    photoinhibition = -PHOTO_K * np.maximum(par - PHOTO_THRESHOLD_PAR, 0.0) ** 2
    return BASELINE + light_term + temp_term + interaction + photoinhibition


@dataclass
class SensorCalibrationTruth:
    """The real, physical sensitivity line this sensor 'actually has' --
    what a careful standard-addition calibration should recover exactly,
    and what apply_calibration inverts against."""

    b0_mv: float
    b1_mv_per_ppm: float


@dataclass
class SensorNoiseProfile:
    """How this sensor's measurement noise scales with the concentration
    being measured.

    A single constant sigma is the wrong model for a PID: noise generally
    grows with signal (shot noise, amplifier gain, flow variation), so
    high-ppm points are intrinsically less precise than low-ppm ones. And
    the growth rate is a property of the individual unit -- two sensors
    from the same batch can share a sensitivity while differing in how
    fast their precision degrades with concentration.

    That matters for standard-addition calibration specifically.
    `fit_sensitivity_per_sensor` offers `ols` and `robust` (Theil-Sen);
    neither weights points by their variance, so with heteroscedastic
    data the fit is dominated by whichever points happen to be noisiest.
    A weighted least squares would be the textbook answer and does not
    exist in the package -- so this profile is what makes that gap
    measurable instead of hypothetical.
    """

    base_std_mv: float
    """Noise floor at zero concentration -- electronics, not chemistry."""

    std_per_ppm: float
    """Additional noise per ppm. This is the term that makes the variance
    profile sensor-specific."""

    def std_at(self, ppm) -> np.ndarray:
        return self.base_std_mv + self.std_per_ppm * np.asarray(ppm, dtype=float)


DEFAULT_NOISE_PROFILE = SensorNoiseProfile(base_std_mv=0.3, std_per_ppm=0.0)
"""Homoscedastic fallback -- reproduces the original constant-sigma
behaviour exactly for callers that pass no profile."""


@dataclass
class AmbientCovariateTruth:
    """A genuine nuisance relationship between the sensor housing's own
    ambient RH/T and its raw voltage, independent of the reactor's
    controlled internal PAR/temp -- the exact class of contamination
    run_ambient_baseline/fit_covariate_model exist to characterize and
    remove."""

    rh_ref_pct: float = 55.0
    t_ref_c: float = 28.0
    beta_rh: float = 1.2
    gamma_t: float = 3.0

    def effect_mv(self, sample_rh_pct: np.ndarray, sample_t_c: np.ndarray) -> np.ndarray:
        return self.beta_rh * (sample_rh_pct - self.rh_ref_pct) + self.gamma_t * (
            sample_t_c - self.t_ref_c
        )


def _ar1_noise(n: int, phi: float, sigma_mv: float, rng: np.random.Generator) -> np.ndarray:
    """One draw of a stationary AR(1) process -- the autocorrelated
    broadband sensor noise fleet-zero/ambient-baseline don't target
    directly, since it's not a systematic bias but a real 1/f-like
    memory in the raw signal that only per-window averaging reduces."""
    noise = np.empty(n)
    stationary_std = sigma_mv / np.sqrt(1.0 - phi**2)
    noise[0] = rng.normal(0.0, stationary_std)
    innovations = rng.normal(0.0, sigma_mv, size=n - 1)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + innovations[i - 1]
    return noise


def generate_calibration_recording(
    sensor_ids: list[str],
    truth: dict[str, SensorCalibrationTruth],
    spike_ppm_list: list[float],
    calibration_compound: str = "isoprene",
    mw_g_mol: float = 68.12,
    response_factor: float = 0.63,
    n_per_level: int = 10,
    noise_std_mv: float = 0.3,
    noise_profiles: dict[str, SensorNoiseProfile] | None = None,
    seed: int = 0,
) -> pl.DataFrame:
    """A clean, controlled standard-addition bench recording -- no
    ambient/common-mode contamination, matching how a real calibration
    is actually run (deliberately isolated from the reactor room). This
    is what `fit_sensitivity_per_sensor` should recover `truth` from.

    A known gas is delivered at several KNOWN, VARYING concentrations
    (`spike_ppm_list`) and each sensor reads every level. When
    `noise_profiles` is given, each sensor's scatter grows with the
    concentration according to its own SensorNoiseProfile, so the levels
    are not equally informative and the sensors are not equally
    well-calibrated -- which is the realistic case. Falls back to the
    original constant `noise_std_mv` when omitted."""
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 22, 6, 0, 0, tzinfo=dt.timezone.utc)

    n_levels = len(spike_ppm_list)
    total_rows = len(sensor_ids) * n_levels * n_per_level

    sensor_idx = np.repeat(np.arange(len(sensor_ids)), n_levels * n_per_level)
    level_idx = np.tile(np.repeat(np.arange(n_levels), n_per_level), len(sensor_ids))

    sensor_id_arr = np.array(sensor_ids)[sensor_idx]
    spike_ppm_arr = np.array(spike_ppm_list, dtype=float)[level_idx]

    b0_arr = np.array([truth[s].b0_mv for s in sensor_id_arr])
    b1_arr = np.array([truth[s].b1_mv_per_ppm for s in sensor_id_arr])

    if noise_profiles is None:
        sigma_arr = np.full(total_rows, noise_std_mv, dtype=float)
    else:
        sigma_arr = np.array(
            [
                noise_profiles.get(s_id, DEFAULT_NOISE_PROFILE).base_std_mv
                + noise_profiles.get(s_id, DEFAULT_NOISE_PROFILE).std_per_ppm * ppm
                for s_id, ppm in zip(sensor_id_arr, spike_ppm_arr)
            ],
            dtype=float,
        )

    """
    Drawn per row from that row's OWN sigma -- this is what makes the
    recording heteroscedastic rather than just noisy.
    """
    noise = rng.normal(0.0, sigma_arr)
    voltage = b0_arr + b1_arr * spike_ppm_arr + noise

    timestamps = [base_time + dt.timedelta(seconds=t) for t in range(total_rows)]

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": sensor_id_arr,
            "spike_ppm_asgas": spike_ppm_arr,
            "pid_voltage_mv": voltage,
            "sample_t_c": np.full(total_rows, 32.0),
            "sample_rh_pct": np.full(total_rows, 55.0),
            "lamp_hours": np.full(total_rows, 12.0),
            "calibration_compound": [calibration_compound] * total_rows,
            "mw_g_mol": np.full(total_rows, mw_g_mol),
            "response_factor": np.full(total_rows, response_factor),
            "response_factor_stderr": [None] * total_rows,
            "calibration_source": ["benchmark-synthetic"] * total_rows,
            "calibration_is_builtin": [True] * total_rows,
        }
    )


def generate_ambient_blank_recording(
    sensor_ids: list[str],
    ambient_truth: AmbientCovariateTruth,
    n_samples: int = 200,
    rh_range: tuple[float, float] = (30.0, 80.0),
    t_range: tuple[float, float] = (22.0, 34.0),
    noise_std_mv: float = 0.1,
    seed: int = 1,
) -> pl.DataFrame:
    """A zero-VOC ambient/blank recording spanning a real RH/T swing --
    what `fit_covariate_model` fits its nuisance-relationship model
    against, exactly the way a real ambient-baseline diagnostic run
    would (clean air, no spike, sensor exposed to the room's own
    RH/T drift)."""
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 22, 5, 0, 0, tzinfo=dt.timezone.utc)

    frames = []
    for sensor_id in sensor_ids:
        rh = rng.uniform(rh_range[0], rh_range[1], size=n_samples)
        temp = rng.uniform(t_range[0], t_range[1], size=n_samples)
        noise = rng.normal(0.0, noise_std_mv, size=n_samples)
        voltage = ambient_truth.effect_mv(rh, temp) + noise
        timestamps = [base_time + dt.timedelta(seconds=i) for i in range(n_samples)]
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_id": [sensor_id] * n_samples,
                    "sample_rh_pct": rh,
                    "sample_t_c": temp,
                    "pid_voltage_mv": voltage,
                }
            )
        )
    return pl.concat(frames)


@dataclass
class NoiseConfig:
    """Every contaminating noise source layered onto an experiment
    recording, beyond the sensor's own true calibration line."""

    """
    Deliberately no common-mode term here: subtract_common_mode's median-
    across-sensors estimate is only valid when every sensor is observing
    the SAME true value at a given timestamp (a synchronized blank/zero
    check, or a swap-pilot rotation) -- not true here, where each
    reactor genuinely differs in PAR/temp and therefore in true VOC
    output. Applying it to this data would subtract real signal
    differences, not noise. Common-mode contamination/removal is
    demonstrated separately, on its own valid same-true-value use case,
    via generate_common_mode_check_recording below.
    """
    ambient: AmbientCovariateTruth = field(default_factory=AmbientCovariateTruth)
    ar1_phi: float = 0.8
    ar1_sigma_mv: float = 2.6
    ambient_rh_swing_pct: float = 20.0
    ambient_t_swing_c: float = 4.0


def generate_experiment_recording(
    experiment_id: str,
    reactor_conditions: dict[str, tuple[float, float]],
    sensor_for_reactor: dict[str, str],
    calibration_truth: dict[str, SensorCalibrationTruth],
    noise: NoiseConfig,
    duration_s: int = 300,
    seed: int = 0,
) -> pl.DataFrame:
    """One experiment's raw recording, across however many
    reactors/sensors ran simultaneously. `reactor_conditions` maps
    reactor_id -> (par, temp), the real controlled setpoint driving
    `true_voc_ppm`."""
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 22, 8, 0, 0, tzinfo=dt.timezone.utc)
    timestamps = [base_time + dt.timedelta(seconds=t) for t in range(duration_s)]

    """
    One shared common-mode draw for the whole recording -- every
    sensor sees the identical value at a given timestamp index, same as
    a real shared HVAC/electrical artifact would produce, before each
    sensor's own independent AR(1) noise is added on top.
    """
    t_arr = np.arange(duration_s)

    frames = []
    for i, (reactor_id, (par, temp)) in enumerate(reactor_conditions.items()):
        sensor_id = sensor_for_reactor[reactor_id]
        truth = calibration_truth[sensor_id]

        true_ppm = float(true_voc_ppm(par, temp))

        sensor_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        sample_rh_pct = noise.ambient.rh_ref_pct + noise.ambient_rh_swing_pct * np.sin(
            2.0 * np.pi * t_arr / 240.0 + i
        )
        sample_t_c = noise.ambient.t_ref_c + noise.ambient_t_swing_c * np.cos(
            2.0 * np.pi * t_arr / 300.0 + i
        )
        ambient_effect = noise.ambient.effect_mv(sample_rh_pct, sample_t_c)
        ar1 = _ar1_noise(duration_s, noise.ar1_phi, noise.ar1_sigma_mv, sensor_rng)

        voltage = truth.b0_mv + truth.b1_mv_per_ppm * true_ppm + ambient_effect + ar1

        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "experiment_id": [experiment_id] * duration_s,
                    "sensor_id": [sensor_id] * duration_s,
                    "reactor_id": [reactor_id] * duration_s,
                    "pid_voltage_mv": voltage,
                    "sample_t_c": sample_t_c,
                    "sample_rh_pct": sample_rh_pct,
                    "lamp_hours": np.full(duration_s, 12.0),
                    "reactor_par_umol_m2_s": np.full(duration_s, par),
                    "reactor_temp_c": np.full(duration_s, temp),
                }
            )
        )

    return pl.concat(frames)


"""
============================================================================
The TIME axis: how VOC gets to its plateau, and how fast.
============================================================================

`true_voc_ppm` above is the PLATEAU -- where VOC settles once the culture
has fully adjusted to a (PAR, temp) setpoint. It says nothing about how
long that takes, and a real week-long run spends its first day or two
getting there.

The approach law is a first-order lag, derived rather than assumed: at
fixed conditions the culture's VOC production rate P is roughly constant
and removal from a well-mixed headspace is proportional to concentration,
so dC/dt = P - kC, which integrates to C(t) = (P/k)(1 - exp(-kt)). The
plateau is P/k (= true_voc_ppm) and the time constant is tau = 1/k.

TAU IS HOURS, NOT MINUTES, and that choice is load-bearing. A ~2-minute
tau would be headspace gas-mixing time; over a 168-hour run that transient
occupies the first 0.1% of the samples and is unrecoverable from noise
(exactly the failure the 2026-07-24 static-PAR redesign hit). At week
timescales the physically relevant process is photoacclimation -- the
culture remodelling pigment and metabolism for new light/temperature --
which in microalgae runs hours to days. ~12-22h gives 7-14 time constants
per week-long run, so tau is comfortably identifiable from the data.
"""

TAU_BASE_H = 20.0
"""Baseline photoacclimation time constant (hours) at PAR=0, temp=TEMP_REF."""
TAU_PAR_H = 6.0
"""Brighter light drives faster acclimation -- shares the SAME saturating
PAR/(K_M+PAR) shape as the plateau's light term (more light speeds the
response, with diminishing returns), so both surfaces need the same custom
basis term and structural recovery is scored on equal footing."""
TAU_TEMP_H_PER_C = 0.30
"""Warmer runs acclimate faster (faster metabolism), cooler ones slower."""


def true_tau_hours(par, temp):
    """Ground-truth photoacclimation time constant (hours) for a static
    (PAR, temp) setpoint -- the SECOND symbolic surface this benchmark
    asks each method to recover, alongside the plateau.

    tau(PAR, temp) = TAU_BASE_H
                    - TAU_PAR_H * PAR / (K_M + PAR)     [brighter -> faster]
                    - TAU_TEMP_H_PER_C * (temp - TEMP_REF)   [warmer -> faster]

    Stays strictly positive across the whole declared domain (~11.8h at
    PAR=500/temp=40, ~22.4h at PAR=0/temp=20).
    """
    par = np.asarray(par, dtype=float)
    temp = np.asarray(temp, dtype=float)
    light_term = TAU_PAR_H * par / (K_M + par)
    temp_term = TAU_TEMP_H_PER_C * (temp - TEMP_REF)
    return TAU_BASE_H - light_term - temp_term


def true_voc_timeseries(t_s, par, temp):
    """The true VOC concentration (ppm) at elapsed time t_s (seconds) into
    a run held at a fixed (PAR, temp), starting from a freshly-reset
    reactor at 0 ppm: plateau * (1 - exp(-t/tau))."""
    t_s = np.asarray(t_s, dtype=float)
    plateau = true_voc_ppm(par, temp)
    tau_s = true_tau_hours(par, temp) * 3600.0
    return plateau * (1.0 - np.exp(-t_s / tau_s))


"""
============================================================================
Per-sensor ambient micro-environments (Part 1)
============================================================================

Three sensors watching the SAME reactor produce three completely
different-LOOKING raw traces -- flat, haphazardly rising, and oscillating.
The shapes differ because each sensor sits somewhere physically different:

- "stable"  : a thermally quiet corner. Near-constant RH/T.
- "warming" : bolted near a component that heats through the week, so its
              own measured temperature ramps upward, with AR(1) roughness
              making the rise irregular rather than a clean line.
- "diurnal" : in an HVAC airflow, so its measured RH/T swing on a
              24-hour day/night cycle.

Every one of those differences is driven by the sensor's OWN MEASURED
RH/T, which is precisely the contamination class fit_covariate_model
characterizes and apply_covariate_correction removes. That is why the
three can genuinely converge after correction -- not because the pipeline
is magic, but because the shape differences are attributable to a measured
covariate.

The contrast case matters just as much: `drift_mv` injects the same three
SHAPES as unexplained sensor drift with no covariate signature. Those are
NOT correctable by anything in this pipeline (not a constant bias, not a
linear function of RH/T), and Part 1 runs it as an explicit negative
control so the boundary of the pipeline's competence is shown rather than
asserted.
"""

AMBIENT_PROFILE_KINDS = ("stable", "warming", "diurnal")

"""
Each micro-environment gets its own MEAN RH/T, not just its own shape.
That is what separates the three raw traces into visibly different bands
the way real multi-sensor recordings look -- a sensor sitting in a warmer
corner reads systematically higher, and its own nominal calibration
cannot remove that because the offset is ambient, not intrinsic to the
sensor. Covariate correction can, because each sensor logs the RH/T that
explains it.
"""
AMBIENT_MEANS = {
    "stable": {"t_offset_c": 0.0, "rh_offset_pct": 0.0},
    "warming": {"t_offset_c": 3.5, "rh_offset_pct": 6.0},
    "diurnal": {"t_offset_c": -2.5, "rh_offset_pct": -8.0},
}


def ambient_micro_environment(kind: str, t_s: np.ndarray, noise: NoiseConfig, rng):
    """Return (sample_rh_pct, sample_t_c) for one sensor's own housing,
    as a real instrument would log them alongside its voltage."""
    t_h = t_s / 3600.0
    offsets = AMBIENT_MEANS[kind]
    rh_mid = noise.ambient.rh_ref_pct + offsets["rh_offset_pct"]
    t_mid = noise.ambient.t_ref_c + offsets["t_offset_c"]

    if kind == "stable":
        """Quiet corner: essentially flat, but still real instrument
        noise on the RH/T channels themselves."""
        rh = np.full(t_s.shape, rh_mid) + rng.normal(0.0, 0.6, size=t_s.shape)
        temp = np.full(t_s.shape, t_mid) + rng.normal(0.0, 0.25, size=t_s.shape)
        return rh, temp

    if kind == "warming":
        """
        A monotone ramp plus AR(1) wander -- 'haphazardly increasing'
        rather than a clean straight line, which is what a real thermal
        drift against a noisy room actually looks like.
        """
        span_h = max(t_h[-1], 1.0)
        wander = _ar1_noise(t_s.size, 0.995, 0.09, rng)
        temp = t_mid - 2.0 + 7.0 * (t_h / span_h) + wander
        rh = rh_mid - 5.0 + 10.0 * (t_h / span_h) + 0.5 * wander
        return rh, temp

    if kind == "diurnal":
        phase = 2.0 * np.pi * t_h / 24.0
        temp = t_mid + noise.ambient_t_swing_c * np.sin(phase) + rng.normal(0.0, 0.2, size=t_s.shape)
        rh = rh_mid + noise.ambient_rh_swing_pct * np.sin(phase - 0.6) + rng.normal(0.0, 0.5, size=t_s.shape)
        return rh, temp

    raise ValueError(f"unknown ambient micro-environment kind: {kind!r}")


EVENT_RATE_PER_DAY = 1.5
"""Genuine transient VOC events -- a disturbance, a feed, a door opening.
These are REAL SIGNAL: every sensor watching the reactor sees the same
event at the same instant, and a correction pipeline that removed them
would be destroying data, not cleaning it."""

GLITCH_RATE_PER_DAY = 1.5
"""Instrument glitches -- electrical transients, dropouts, connector
noise. These are per-sensor and independent: no other sensor sees them.

That difference is the whole point of running three sensors. From ONE
trace a real event and a glitch are frequently indistinguishable; across
three simultaneous traces they are not, because coincidence across
independent instruments is the signature of something real. The benchmark
models them separately so the distinction can actually be measured
instead of assumed."""


def spike_train(t_s: np.ndarray, rng, rate_per_day: float, lo: float, hi: float) -> np.ndarray:
    """Sparse, sharp, short-lived excursions.

    Used for BOTH real VOC events (in ppm, shared across sensors, part of
    the true signal) and instrument glitches (in mV, independent per
    sensor, pure contamination) -- the shape is the same, which is exactly
    why a single trace cannot tell them apart.
    """
    duration_days = float(t_s[-1] - t_s[0]) / 86400.0
    n_spikes = rng.poisson(max(rate_per_day * duration_days, 0.0))
    out = np.zeros_like(t_s, dtype=float)
    if n_spikes == 0:
        return out

    dt_s = float(np.median(np.diff(t_s))) if t_s.size > 1 else 1.0
    for _ in range(int(n_spikes)):
        centre = int(rng.integers(0, t_s.size))
        """SHARP: one to three samples. Real PID excursions are brief
        vertical strokes, not hour-long plateaus -- an earlier version of
        this used widths up to an hour, which contaminated ~10% of the
        run and swamped every other effect in the benchmark."""
        width = int(rng.integers(0, 2))
        amplitude = rng.choice([-1.0, 1.0]) * rng.uniform(lo, hi)

        """
        `start`/`stop`, NOT lo/hi: those are this function's amplitude-bound
        parameters, and reassigning them here silently turned every
        subsequent rng.uniform(lo, hi) into a draw from the ARRAY INDEX
        range -- producing ~2000 ppm excursions and driving the true signal
        to -1380 ppm, which still looked superficially like 'spiky data'.
        """
        start = max(centre - width, 0)
        stop = min(centre + width + 1, t_s.size)
        out[start:stop] += amplitude
    return out


def uncorrectable_drift_mv(kind: str, t_s: np.ndarray, rng) -> np.ndarray:
    """The negative control: the same three visual shapes as
    `ambient_micro_environment`, but injected straight onto the voltage
    with NO covariate signature -- nothing in the logged RH/T explains
    them, so no amount of covariate correction can remove them."""
    t_h = t_s / 3600.0
    span_h = max(t_h[-1], 1.0)

    if kind == "stable":
        return np.full(t_s.shape, 12.0)
    if kind == "warming":
        return 25.0 * (t_h / span_h) + _ar1_noise(t_s.size, 0.995, 0.15, rng)
    if kind == "diurnal":
        return 18.0 * np.sin(2.0 * np.pi * t_h / 24.0)
    raise ValueError(f"unknown drift kind: {kind!r}")


def generate_week_long_sensor_recording(
    experiment_id: str,
    reactor_id: str,
    sensor_ids: list[str],
    sensor_environments: dict[str, str],
    calibration_truth: dict[str, SensorCalibrationTruth],
    noise: NoiseConfig,
    par: float,
    temp: float,
    duration_s: int = 7 * 24 * 3600,
    dt_s: float = 300.0,
    inject_uncorrectable_drift: bool = False,
    inject_events: bool = True,
    inject_glitches: bool = True,
    seed: int = 0,
) -> pl.DataFrame:
    """One week-long run at a fixed (PAR, temp), watched simultaneously by
    several sensors that each sit in their own ambient micro-environment.

    Every sensor observes the IDENTICAL true VOC(t) -- so any disagreement
    between their raw traces is contamination by construction, and how
    much of it survives correction is exactly what Part 1 measures.

    Sampled every `dt_s` (5 min by default, 2016 points/week): far coarser
    than the rig's real ~1 Hz VOC cadence, but a 12-22h time constant is
    massively oversampled at 5 min, and a week at 1 Hz would be 604,800
    rows per sensor carrying no additional information about tau.
    """
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 24, 10, 0, 0, tzinfo=dt.timezone.utc)
    n = int(duration_s / dt_s)
    t_arr = np.arange(n) * dt_s
    timestamps = [base_time + dt.timedelta(seconds=float(ti)) for ti in t_arr]

    """
    Real VOC events are generated ONCE, outside the per-sensor loop, and
    folded into `true_ppm` -- so they are part of the ground truth every
    sensor is trying to measure, identical and simultaneous across all
    three. A pipeline that removed them would be destroying signal, and
    scoring them as error would be measuring the wrong thing.
    """
    true_ppm = true_voc_timeseries(t_arr, par, temp)
    if inject_events:
        event_rng = np.random.default_rng(seed + 7919)
        true_ppm = true_ppm + spike_train(t_arr, event_rng, EVENT_RATE_PER_DAY, 60.0, 240.0)

    frames = []
    for sensor_id in sensor_ids:
        truth = calibration_truth[sensor_id]
        sensor_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        kind = sensor_environments[sensor_id]

        sample_rh_pct, sample_t_c = ambient_micro_environment(kind, t_arr, noise, sensor_rng)
        ambient_effect = noise.ambient.effect_mv(sample_rh_pct, sample_t_c)
        ar1 = _ar1_noise(n, noise.ar1_phi, noise.ar1_sigma_mv, sensor_rng)

        voltage = truth.b0_mv + truth.b1_mv_per_ppm * true_ppm + ambient_effect + ar1

        """
        Glitches are drawn per sensor from that sensor's OWN rng, so no
        two sensors share one. This is what makes cross-sensor coincidence
        a usable discriminator downstream.
        """
        if inject_glitches:
            voltage = voltage + spike_train(t_arr, sensor_rng, GLITCH_RATE_PER_DAY, 40.0, 160.0)
        if inject_uncorrectable_drift:
            voltage = voltage + uncorrectable_drift_mv(kind, t_arr, sensor_rng)

        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "experiment_id": [experiment_id] * n,
                    "sensor_id": [sensor_id] * n,
                    "reactor_id": [reactor_id] * n,
                    "pid_voltage_mv": voltage,
                    "sample_t_c": sample_t_c,
                    "sample_rh_pct": sample_rh_pct,
                    "lamp_hours": np.full(n, 12.0),
                    "reactor_par_umol_m2_s": np.full(n, par),
                    "reactor_temp_c": np.full(n, temp),
                    "true_voc_ppm": true_ppm,
                }
            )
        )

    return pl.concat(frames)


def generate_cross_sensor_consistency_recording(
    experiment_id: str,
    reactor_id: str,
    sensor_ids: list[str],
    calibration_truth: dict[str, SensorCalibrationTruth],
    noise: NoiseConfig,
    par: float,
    temp: float,
    duration_s: int = 600,
    dt_s: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Short-window, STATIC-VOC cross-sensor check retained for Test 1's
    own numeric sub-test.

    Part 1 (sensor_consistency.py) supersedes this for the headline
    convergence story: it runs a full week, gives each sensor its own
    ambient micro-environment so the raw traces differ in SHAPE rather
    than just phase, and drives a real time-varying VOC signal. This
    one holds the true VOC constant over 10 minutes, which is a weaker
    but still valid same-true-value agreement check.
    """
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 24, 10, 0, 0, tzinfo=dt.timezone.utc)
    n = int(duration_s / dt_s)
    t_arr = np.arange(n) * dt_s
    timestamps = [base_time + dt.timedelta(seconds=float(ti)) for ti in t_arr]

    true_ppm = float(true_voc_ppm(par, temp))

    frames = []
    for i, sensor_id in enumerate(sensor_ids):
        truth = calibration_truth[sensor_id]
        sensor_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        sample_rh_pct = noise.ambient.rh_ref_pct + noise.ambient_rh_swing_pct * np.sin(
            2.0 * np.pi * t_arr / 240.0 + i
        )
        sample_t_c = noise.ambient.t_ref_c + noise.ambient_t_swing_c * np.cos(
            2.0 * np.pi * t_arr / 300.0 + i
        )
        ambient_effect = noise.ambient.effect_mv(sample_rh_pct, sample_t_c)
        ar1 = _ar1_noise(n, noise.ar1_phi, noise.ar1_sigma_mv, sensor_rng)

        voltage = truth.b0_mv + truth.b1_mv_per_ppm * true_ppm + ambient_effect + ar1

        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "experiment_id": [experiment_id] * n,
                    "sensor_id": [sensor_id] * n,
                    "reactor_id": [reactor_id] * n,
                    "pid_voltage_mv": voltage,
                    "sample_t_c": sample_t_c,
                    "sample_rh_pct": sample_rh_pct,
                    "lamp_hours": np.full(n, 12.0),
                    "reactor_par_umol_m2_s": np.full(n, par),
                    "reactor_temp_c": np.full(n, temp),
                }
            )
        )

    return pl.concat(frames)


def generate_common_mode_check_recording(
    sensor_ids: list[str],
    reactor_ids: list[str],
    fleet_zero_bias_mv: dict[str, float],
    common_mode_amplitude_mv: float = 3.0,
    common_mode_period_s: float = 60.0,
    individual_noise_std_mv: float = 0.2,
    n_samples: int = 120,
    seed: int = 2,
) -> pl.DataFrame:
    """A synchronized fleet-wide zero/blank check -- every sensor exposed
    to the same true (zero-VOC) condition at the same instants, which is
    the one situation where subtract_common_mode's cross-sensor median
    is actually a valid estimate of the shared artifact rather than a
    corruption of genuine per-reactor signal differences (see
    NoiseConfig's docstring). Each sensor keeps its own fixed
    fleet-zero-style bias on top."""
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 22, 4, 0, 0, tzinfo=dt.timezone.utc)
    timestamps = [base_time + dt.timedelta(seconds=i) for i in range(n_samples)]

    t_arr = np.arange(n_samples)
    common_mode = common_mode_amplitude_mv * np.sin(2.0 * np.pi * t_arr / common_mode_period_s)

    frames = []
    for sensor_id, reactor_id in zip(sensor_ids, reactor_ids):
        noise = rng.normal(0.0, individual_noise_std_mv, size=n_samples)
        voltage = fleet_zero_bias_mv[sensor_id] + common_mode + noise
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_id": [sensor_id] * n_samples,
                    "reactor_id": [reactor_id] * n_samples,
                    "pid_voltage_mv": voltage,
                }
            )
        )
    return pl.concat(frames)


"""
Everything above is ground truth #2: how VOC varies ACROSS many
different static (PAR, temp) settings -- one number per experiment,
stacked across many experiments to see how the setting itself changes
the outcome. That's the domain suggest_next_experiments/JAXSR active
learning operates in.

Everything below is a SEPARATE, DISTINCT ground truth -- #1: given ONE
specific, time-varying PAR(t) schedule within a single experiment, how
does VOC unfold over TIME in response to it. That's the domain
discover_led_response_dynamics/jaxsr.discover_dynamics operates in, and
nothing above tests it at all. The two are tied together deliberately,
not left as two unrelated stories: the dynamic law below relaxes
toward true_voc_ppm(par(t), temp) as its steady-state target, so
holding any one (PAR, temp) setting constant forever would eventually
land exactly on the point already described by the static surface
above.
"""

DYNAMIC_RELAXATION_TAU_S = 120.0
"""Superseded by true_tau_hours (see simulate_true_dynamic_trajectory).
Kept only because a few older call sites still import it; it is no longer
what the trajectory integrator uses."""


def simulate_true_dynamic_trajectory(
    par_fn, temp: float, duration_s: int, dt_s: float = 1.0, voc0: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate dVOC/dt = (1/tau) * (true_voc_ppm(par(t), temp) - VOC(t))
    -- a first-order lag toward whatever true_voc_ppm says the CURRENT
    instantaneous (par(t), temp) setting's steady-state VOC would be, at
    a fixed relaxation time constant. `par_fn(elapsed_s) -> float` is
    typically `algaesense_edge.actuators.control_profiles.evaluate_control_profile`
    partially applied to a real profile dict -- the same function that
    actually drives the LED, not a re-implementation of profile shapes.
    Returns `(t, par_values, true_voc_values)`, all length `duration_s /
    dt_s`. Forward Euler at 1s resolution is adequate here since
    tau=120s is >> dt_s -- no stiff dynamics to worry about."""
    n = int(duration_s / dt_s)
    t = np.arange(n) * dt_s
    par_values = np.array([par_fn(float(ti)) for ti in t])
    voc = np.empty(n)
    voc[0] = voc0

    """
    tau is taken from `true_tau_hours(PAR(t), temp)` -- the SAME
    photoacclimation surface Part 2 asks methods to recover -- rather
    than a single fixed constant. That keeps one physics across the whole
    benchmark: hold PAR constant and this integrates to exactly the
    plateau*(1-exp(-t/tau)) curve Parts 1 and 2 use, while a time-varying
    PAR(t) makes tau itself vary along the trajectory, which is what
    gives Part 3 something real to discover.
    """
    for i in range(1, n):
        par_prev = par_values[i - 1]
        target = float(true_voc_ppm(par_prev, temp))
        tau_s = float(true_tau_hours(par_prev, temp)) * 3600.0
        voc[i] = voc[i - 1] + dt_s * (target - voc[i - 1]) / tau_s
    return t, par_values, voc


def generate_dynamic_experiment_recording(
    experiment_id: str,
    reactor_id: str,
    sensor_id: str,
    par_values: np.ndarray,
    temp: float,
    true_voc_values: np.ndarray,
    calibration_truth: SensorCalibrationTruth,
    noise: NoiseConfig,
    dt_s: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """One experiment's raw recording under a time-varying PAR(t)
    schedule -- same contamination model as generate_experiment_recording
    (ambient RH/T covariate nuisance + AR(1) noise; no common-mode term,
    same reasoning as there), just with `par_values`/`true_voc_values`
    varying per row instead of being constant for the whole recording."""
    n = len(par_values)
    rng = np.random.default_rng(seed)
    base_time = dt.datetime(2026, 7, 22, 9, 0, 0, tzinfo=dt.timezone.utc)
    timestamps = [base_time + dt.timedelta(seconds=i * dt_s) for i in range(n)]

    t_arr = np.arange(n)
    sample_rh_pct = noise.ambient.rh_ref_pct + noise.ambient_rh_swing_pct * np.sin(
        2.0 * np.pi * t_arr / 240.0
    )
    sample_t_c = noise.ambient.t_ref_c + noise.ambient_t_swing_c * np.cos(
        2.0 * np.pi * t_arr / 300.0
    )
    ambient_effect = noise.ambient.effect_mv(sample_rh_pct, sample_t_c)
    ar1 = _ar1_noise(n, noise.ar1_phi, noise.ar1_sigma_mv, rng)

    voltage = (
        calibration_truth.b0_mv
        + calibration_truth.b1_mv_per_ppm * true_voc_values
        + ambient_effect
        + ar1
    )

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "experiment_id": [experiment_id] * n,
            "sensor_id": [sensor_id] * n,
            "reactor_id": [reactor_id] * n,
            "pid_voltage_mv": voltage,
            "sample_t_c": sample_t_c,
            "sample_rh_pct": sample_rh_pct,
            "lamp_hours": np.full(n, 12.0),
            "reactor_par_umol_m2_s": par_values,
            "reactor_temp_c": np.full(n, temp),
        }
    )


# ---------------------------------------------------------------- four factors

"""
The two-factor truth above is what Tests 1-3 and the original discovery
benchmark are built on, and it stays exactly as it is. What follows is an
ADDITIONAL, wider ground truth for the question those cannot answer:
whether adaptive experiment design earns its keep once the space is big
enough that covering it evenly stops being cheap.

Two more factors, both chosen because they are real levers on a
photobioreactor and because their shapes differ from the light and
temperature terms already present -- a benchmark whose new factors behave
like the old ones would not test anything new.
"""

PH_OPT = 9.2
"""Spirulina is cultured alkaline; output falls away either side."""
PH_K = 1.8
PH_BOUNDS = (7.5, 10.5)

NUTRIENT_VMAX = 14.0
NUTRIENT_K_M = 4.0
"""Nitrate follows Monod saturation, the same family as the light
response -- deliberately, since two saturating terms in different
variables is exactly the sort of confusion a discovery method should have
to resolve."""
NUTRIENT_BOUNDS = (0.5, 30.0)


def true_voc_ppm_4d(par, temp, ph, nutrient):
    """VOC output as a function of light, temperature, pH and nitrate.

    The two-factor truth, plus a pH optimum and a saturating nutrient
    response:

        + the whole of true_voc_ppm(par, temp)
        - PH_K * (ph - PH_OPT)^2                       [pH optimum]
        + NUTRIENT_VMAX * nut / (NUTRIENT_K_M + nut)   [nitrate saturation]
    """
    par = np.asarray(par, dtype=float)
    temp = np.asarray(temp, dtype=float)
    ph = np.asarray(ph, dtype=float)
    nutrient = np.asarray(nutrient, dtype=float)

    ph_term = -PH_K * (ph - PH_OPT) ** 2
    nutrient_term = NUTRIENT_VMAX * nutrient / (NUTRIENT_K_M + nutrient)
    return true_voc_ppm(par, temp) + ph_term + nutrient_term


# ------------------------------------------------- smooth four-factor truth

"""
A second four-factor truth, differing from `true_voc_ppm_4d` in one
respect that matters for what active learning is actually good at.

Adaptive design assumes the response is SMOOTH -- it fits a model to what
it has seen and chooses the next point from that model, which is only
sound if the response has no kinks for the model to be blindsided by. The
photoinhibition term inherited from the two-factor truth,
`max(par - 380, 0)^2`, is continuous and has a continuous first
derivative but a DISCONTINUOUS second derivative at the threshold, and it
is identically zero across 90% of the light range. Measured, not assumed.

Haldane substrate-inhibition kinetics replaces it:

    VMAX * par / (K_M + par + par^2 / K_I)

which is the standard model for photoinhibition in the first place, is
infinitely differentiable for par >= 0, and declines at high light for
the same physical reason. Its peak sits at sqrt(K_M * K_I), so K_I is
chosen to put that peak where the hinged version's threshold was.

This also creates a better discrimination test than the hinge did. The
non-inhibited Monod form `par / (K_M + par)` is now available as a decoy,
and it is indistinguishable from Haldane unless the campaign samples high
light -- so "did you look where the response turns over" becomes a
question about the shape of a smooth curve rather than about finding a
kink.
"""

HALDANE_PEAK_PAR = 300.0
HALDANE_K_I = HALDANE_PEAK_PAR**2 / K_M
"""Haldane peaks at sqrt(K_M * K_I). Placed at 300 against a light range
running to 500, so 42% of the range lies past the turnover.

That fraction is the whole difficulty of the test and was chosen by
measurement, not taste. With the turnover at 400 and the range ending at
420, only 5% of the range showed any decline and the non-inhibited Monod
form correlated 0.984 with the true one -- indistinguishable in practice,
which would have made discovery an unwinnable question dressed up as a
hard one. At 300/500 that correlation is 0.911: still demanding, since a
campaign confined to low light genuinely cannot tell the two apart, but
answerable by one that looks past the peak."""


def haldane_light(par):
    """Light response with smooth photoinhibition built in."""
    par = np.asarray(par, dtype=float)
    return VMAX * par / (K_M + par + par**2 / HALDANE_K_I)


def true_voc_ppm_4d_smooth(par, temp, ph, nutrient):
    """The four-factor truth with every term infinitely differentiable.

        VOC = BASELINE
            + haldane_light(par)                            [light, with turnover]
            + TEMP_SLOPE * (temp - TEMP_REF)                [temperature]
            + GAMMA * par * (temp - TEMP_REF)               [light x temperature]
            - PH_K * (ph - PH_OPT)^2                        [pH optimum]
            + NUTRIENT_VMAX * nut / (NUTRIENT_K_M + nut)    [nitrate saturation]
    """
    par = np.asarray(par, dtype=float)
    temp = np.asarray(temp, dtype=float)
    ph = np.asarray(ph, dtype=float)
    nutrient = np.asarray(nutrient, dtype=float)

    return (
        BASELINE
        + haldane_light(par)
        + TEMP_SLOPE * (temp - TEMP_REF)
        + GAMMA * par * (temp - TEMP_REF)
        - PH_K * (ph - PH_OPT) ** 2
        + NUTRIENT_VMAX * nutrient / (NUTRIENT_K_M + nutrient)
    )
