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
from typing import Callable

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
    seed: int = 0,
) -> pl.DataFrame:
    """A clean, controlled standard-addition bench recording -- no
    ambient/common-mode contamination, matching how a real calibration
    is actually run (deliberately isolated from the reactor room). This
    is what `fit_sensitivity_per_sensor` should recover `truth` from
    almost exactly."""
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

    noise = rng.normal(0.0, noise_std_mv, size=total_rows)
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
    ar1_sigma_mv: float = 0.4
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
Three distinct, physically-motivated per-sensor systematic-drift
artifacts -- unlike the constant per-sensor bias (b0/b1) and the shared
ambient RH/T covariate nuisance already modeled above, these represent a
real, different class of sensor fault: a raw voltage trace whose SHAPE
over time is genuinely sensor-specific, even when every sensor is
observing the exact same (PAR, temp) condition. Deliberately NOT
something fleet-zero (a single constant offset) or ambient-baseline
covariate correction (a linear RH/T relationship) are built to remove --
the point of testing this is to see whether the existing correction
pipeline denoises these shapes into agreement anyway, or whether a real,
honest residual difference between sensors remains.
"""


def sluggish_flat_artifact_mv(t: np.ndarray, stale_offset_mv: float = -60.0, tau_s: float = 3000.0) -> np.ndarray:
    """A heavily-damped/lagged sensor -- e.g. a PID cell with a slow
    membrane response -- starts from a stale offset and relaxes toward
    zero (no extra artifact) only very slowly (tau_s=3000s is 5x the
    600s window this is actually observed over, so within that window it
    reads as almost flat/stagnant, persistently wrong, rather than
    tracking anything -- unlike a bare exp() with a short tau, which
    would visibly finish decaying and look like a normal rise instead)."""
    return stale_offset_mv * np.exp(-t / tau_s)


def zigzag_rising_artifact_mv(
    t: np.ndarray, amplitude_mv: float = 20.0, period_s: float = 90.0, rise_rate_mv_per_s: float = 0.05,
) -> np.ndarray:
    """A periodic triangle-wave drift -- e.g. intermittent connector/
    thermal-cycling contact-resistance changes -- superimposed on a slow
    rising trend, giving a genuinely jagged, non-smooth zigzag shape
    (distinct from a smooth sinusoid)."""
    from scipy.signal import sawtooth

    triangle = sawtooth(2.0 * np.pi * t / period_s, width=0.5)
    return amplitude_mv * triangle + rise_rate_mv_per_s * t


def curvy_drift_artifact_mv(t: np.ndarray, amplitude_mv: float = 45.0, tau_s: float = 150.0) -> np.ndarray:
    """A smooth, nonlinear exponential-approach drift -- e.g. slow
    thermal/aging equilibration -- visually distinct from both the flat
    (sluggish_flat) and jagged (zigzag_rising) shapes above."""
    return amplitude_mv * (1.0 - np.exp(-t / tau_s))


def generate_cross_sensor_consistency_recording(
    experiment_id: str,
    reactor_id: str,
    sensor_ids: list[str],
    calibration_truth: dict[str, SensorCalibrationTruth],
    noise: NoiseConfig,
    par: float,
    temp: float,
    artifact_mv_fns: dict[str, Callable[[np.ndarray], np.ndarray]],
    duration_s: int = 600,
    dt_s: float = 1.0,
    seed: int = 0,
) -> pl.DataFrame:
    """Unlike generate_experiment_recording (where each reactor/sensor
    observes its OWN different (PAR, temp) condition), every sensor here
    observes the EXACT SAME fixed condition simultaneously -- the true
    VOC value is identical for all of them. Each sensor's raw voltage
    additionally carries its own systematic drift artifact
    (artifact_mv_fns[sensor_id](t), one of the three functions above) on
    top of the same per-sensor calibration line and ambient-covariate/
    AR(1) noise every other recording in this module uses. Models a real
    fleet reality: three nominally-identical PID sensors can show wildly
    different raw traces over time due to sensor-specific faults, even
    when genuinely measuring the same thing."""
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
        artifact_mv = artifact_mv_fns[sensor_id](t_arr)

        voltage = truth.b0_mv + truth.b1_mv_per_ppm * true_ppm + ambient_effect + ar1 + artifact_mv

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
    k = 1.0 / DYNAMIC_RELAXATION_TAU_S
    for i in range(1, n):
        target = float(true_voc_ppm(par_values[i - 1], temp))
        voc[i] = voc[i - 1] + dt_s * k * (target - voc[i - 1])
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
