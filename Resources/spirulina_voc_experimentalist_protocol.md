# Spirulina VOC Emission Monitoring
## Experimentalist Protocol

**Version:** 0.1 (draft)
**Scope:** *Arthrospira platensis* cultivation in Pioreactor 20 mL reactors with Alphasense PID VOC sensors.
**Audience:** Lab experimentalist executing runs. This is a hands-on procedure document.
**Companion document:** `jaxsr_augmentation_spec.md` (for the software team building the diagnostics/calibration/preprocessing tool).
**Status:** working protocol; sections marked ⚠️ indicate open questions to resolve before first campaign.

---

## Table of Contents

- [Part I — Purpose, Scope, Safety](#part-i--purpose-scope-safety)
- [Part II — Materials & Hardware](#part-ii--materials--hardware)
- [Part III — Physical Setup (Day 0–1)](#part-iii--physical-setup-day-01)
- [Part IV — Culture Preparation](#part-iv--culture-preparation)
- [Part V — Running Baseline Diagnostics (Week 1)](#part-v--running-baseline-diagnostics-week-1)
- [Part VI — In-Situ Sensor Calibration](#part-vi--in-situ-sensor-calibration)
- [Part VII — Standard Operating Procedure for an Experiment Run](#part-vii--standard-operating-procedure-for-an-experiment-run)
- [Part VIII — Troubleshooting](#part-viii--troubleshooting)
- [Appendices](#appendices)

---

## Part I — Purpose, Scope, Safety

### 1. Purpose

Grow *Arthrospira platensis* in Pioreactor 20 mL vessels while monitoring VOC emissions with a network of Alphasense PID sensors, generating high-quality data for downstream modeling.

### 2. Scope

**In scope for this document:** everything you touch physically — hardware, tubing, medium preparation, inoculation, calibration procedure, running the experiment, reading dashboards, filing troubleshooting reports.

**Out of scope:** internal software architecture, JAXSR modeling, active-learning experiment planning. When the protocol says "run `jaxsr {command}`," you treat the tool as a black box; how it's built is documented separately.

### 3. The five-layer architecture (for orientation)

The pipeline has five layers. You interact directly with Layers 1, 2, and 5; Layers 3 and 4 happen inside the software.

| Layer | Function | Your involvement |
|-------|----------|------------------|
| 1 | Physical sample conditioning | **Deferred** — no Nafion dryer, no manifold, for now |
| 2 | Per-sensor calibration by standard addition | You do this before each run (Part VI) |
| 3 | Signal processing (software) | Runs automatically after each run |
| 4 | Modeling and active learning (existing JAXSR core) | Consumes your data; reads out next-batch conditions |
| 5 | Diagnostics | You run these weekly (Part VIII) |

Because Layer 1 is skipped, we compensate by measuring temperature and relative humidity at every sensor and correcting in software. That's why every PID has an SHT35 or BME280 sensor attached to its inlet — do not remove or bypass these.

### 4. Prerequisites

You should be comfortable with:
- Sterile pipetting at the mL scale.
- Basic aseptic technique (flame or laminar-flow hood).
- Reading a pH strip or calibrated pH probe.
- Running Python-CLI commands from a terminal.
- Following a checklist without skipping steps you don't understand — ask instead.

### 5. Safety

- Zarrouk medium contains ~16 g/L NaHCO₃ and reaches pH ~10; wear gloves and eye protection when mixing.
- The Alphasense PID lamp emits UV (10.6 eV / ~117 nm) but is enclosed. **Never disassemble a powered sensor.**
- LEDs may be bright enough to cause discomfort; do not stare into them at close range.
- No CBRN or high-hazard reagents are in scope. If you find yourself considering one, stop and consult the PI.

---

## Part II — Materials & Hardware

### 6. Bill of materials

**Reactors and culture:**
- Pioreactor 20 mL units (v1.1 preferred) — one per experimental condition plus at least one for cell-free control. Minimum recommended build: **4 units**.
- 20 mL glass vials with silicone/PTFE septum caps (Pioreactor stock).
- 15 mm PTFE-coated magnetic stir bars.
- 5 mm super-bright white photosynthetic LEDs (4000–5000 K) — 2 per reactor.
- Small air pump with adjustable flow + 0.2 µm PTFE syringe filters.

**VOC sensors and ancillaries:**
- Alphasense PID sensors — ⚠️ record model, serial number, and lamp hours for each unit.
- Sensirion SHT35 or Bosch BME280 T/RH sensor — **one per PID**, mounted immediately upstream of the PID inlet.
- Optional: Sensirion SFM3000 or Honeywell HAF flow sensor — one per PID.
- 3 mm ID PTFE or FEP tubing for gas sampling. **Do not use Tygon or silicone tubing** — both outgas VOCs badly.
- Luer-lock T-fittings.

**Calibration hardware:**
- **Calibration standard.** Pick one at project start; either works with the tool. Options: (a) certified gas cylinder — isobutylene is the community reference (RF = 1.0) and cheapest, but any compound with a published response factor is fine (isoprene, acetone, methanol, DMS, toluene are all supported out of the box); (b) permeation or diffusion tube of your chosen compound, held at a controlled temperature. If you don't know which to pick, isobutylene is the safe default. Whatever you choose, record it and the tool takes the response factor into account automatically.
- 500 mL Schott bottle with silicone/PTFE septum cap — the "reference jar" (§17).
- Gas-tight syringes: 100 µL, 1 mL, 10 mL.

**Data acquisition:**
- Raspberry Pi 4 or 5 (already used by Pioreactor Leader).
- ADS1115 or MCP3428 16-bit ADC if PID output is analog and not on a dedicated PID PCB.
- I²C multiplexer (TCA9548A) if multiple SHT/BME sensors share the same bus.

**Reagents:** see Appendix A for full Zarrouk medium recipe.

---

## Part III — Physical Setup (Day 0–1)

### 7. Bench layout

Reactors and sensors go on a vibration-isolated bench in a room with reasonably stable temperature (±2 °C over 24 h). Keep the setup away from HVAC vents, doors, and direct sunlight. Group all reactors within ~50 cm so room T/RH is common to all — this is what enables the software's common-mode rejection.

Photograph the layout after assembly. Save to `docs/setup_photos/YYYY-MM-DD/`.

### 8. Pioreactor assembly

Follow the manufacturer's assembly guide for each Pioreactor 20 mL. After assembly, verify each unit:

1. Stir plate spins a bare stir bar at 200, 500, and 800 RPM without wobble.
2. Heater setpoint of 32 °C reaches ±0.5 °C at the vial with an independent thermometer within 10 minutes.
3. Onboard OD reading of DI water reads within manufacturer's tolerance.

Label each unit with a permanent ID (`R01`, `R02`, …).

### 9. Photosynthetic LED installation

For each reactor, mount two 5 mm white LEDs on opposite sides of the vial holder at **1.5–2.0 cm** from the vial wall. Wire through the Pioreactor's 5 mm LED accessory ports so light is software-controllable.

With the vial in place but empty, measure illuminance at the vial surface with a phone lux meter. **Target ≤ 15,000 lux** to stay below the photoinhibition threshold at low biomass density. If higher, increase LED distance or use paper diffusers until in range.

Record the measurement.

### 10. PID and ancillary sensor installation

For each PID sensor, assemble the sample gas path as follows (from reactor to exhaust):

```
Reactor headspace luer port
   │
   ├── 0.2 µm PTFE sterile filter
   │
   ├── SHT35 / BME280 T+RH sensor (inline via I²C tee)
   │
   ├── Alphasense PID sensor inlet
   │
   ├── (optional) inline flow sensor
   │
   └── Exhaust to fume hood or activated-charcoal scrubber
```

**Rules:**
- Total tubing length between reactor and PID **≤ 30 cm**. Longer runs cause VOC adsorption on tubing walls.
- PTFE or FEP tubing only. No Tygon, no silicone.
- Label every fitting with sensor ID and orientation arrows. You will thank yourself later.

### 11. Ancillary sensor wiring

SHT35/BME280 modules communicate over I²C. If more than one shares the bus, use a TCA9548A multiplexer. Mount each T/RH module inside a small (~5 mL) PTFE T-fitting so sample gas flows over the sensor face. Press-fit with a PTFE gasket; avoid glue if possible.

### 12. Recording sensor configuration

For each PID, record in `configs/sensors.yaml`:

```yaml
sensors:
  - id: PID01
    model: PID-AH2            # ⚠️ confirm actual model
    serial: XXXXXXXX
    lamp_install_date: 2026-01-15
    lamp_hours_at_install: 0
    calibration_gas: isobutylene    # or isoprene, acetone, methanol, dms, custom, ...
    factory_sensitivity_mV_per_ppm: 20.0    # from Alphasense cert
    associated_rh_sensor: SHT01
    associated_reactor: R01                 # rotates per Latin square
```

**Sensor rotation:** sensor-to-reactor assignments rotate on a Latin-square schedule across experiments. Sensor identity is not permanently welded to reactor identity, because otherwise a bad sensor and a bad reactor cannot be distinguished. Read the rotation schedule for the coming week from `configs/rotation_schedule.yaml` before every run.

### 13. Cabling, grounding, noise

Route signal cables away from LED power lines and pump motor leads. Twist analog signal pairs. All grounds go to a single star point at the Pi. If PIDs share a 5 V rail, add 100 µF decoupling capacitors at each sensor.

---

## Part IV — Culture Preparation

### 14. Medium preparation (Zarrouk's medium)

Follow Appendix A. Autoclave at 121 °C for 20 min. Cool. Verify pH is 9.5–10.0 before use.

**Critical:** use distilled or RO water only. Tap water contains chlorine and chloramine (biocidal to cyanobacteria) and Ca/Mg salts (precipitate at alkaline pH).

### 15. Parent culture refresh

**Three to five days before any experiment run,** refresh the source jar culture:

1. Take ~10% of the jar volume of healthy dark blue-green culture from mid-depth. **Do not use surface scum or bottom sediment.**
2. Combine with fresh Zarrouk medium at 15–20% v/v inoculum.
3. Return to nominal growth (32 °C, ~100 μmol m⁻² s⁻¹ PAR, gentle mixing).
4. Confirm under 40× microscope that filaments are intact, helical, blue-green, ≥100 µm long.

**If the parent culture looks yellow, fragmented, or sedimented, do NOT proceed.** Grow it out further or restart from a fresh source. Continuing with a bad parent culture wastes days of experiment time.

### 16. Reactor inoculation

For each reactor scheduled in the run:

1. Sterilize vial and stir bar (autoclave or 70% ethanol rinse + UV).
2. Add fresh Zarrouk medium to fill volume minus inoculum volume.
3. Add inoculum from the refreshed parent culture to reach the target OD specified for that reactor in the run's condition list.
4. Cap the vial. Mount on the Pioreactor. Connect gas line. Start stir bar at 300 RPM **before** turning on LEDs.
5. Record initial OD, pH, and timestamp when prompted by the calibration/logging tool.

---

## Part V — Running Baseline Diagnostics (Week 1)

Before growing anything, characterize every PID against clean references. Any sensor failing these tests is cleaned or replaced before Part VI. The software team's tool provides three diagnostic commands; your job is to set up the physical conditions and interpret the pass/fail output.

### 17. Fleet zero test

**Objective:** every PID reads ~0 with low noise on clean air.

**Physical setup:**
1. Connect all PID inlets to a common activated-charcoal-scrubbed air source. A Drierite + charcoal column pushed by an aquarium pump is fine.
2. Let the manifold stabilize for 15 minutes before starting.

**Run the tool:**
```bash
jaxsr diagnose fleet-zero --duration-min 60
```

The tool logs at 1 Hz for 60 minutes and prints a per-sensor pass/fail summary. On failure, follow Alphasense AAN 306-03 to clean the lamp and repeat. If two clean-and-retest cycles still fail, tag the sensor `SUSPECT` and exclude from Part VI.

### 18. Ambient baseline test

**Objective:** characterize each sensor's response to room T and RH.

**Physical setup:**
1. Leave all PIDs sampling ambient room air overnight (≥ 12 h).
2. Ensure room T/RH logger is running.

**Run the tool:**
```bash
jaxsr diagnose ambient --duration-h 12
```

The tool fits a local regression per sensor and stores baseline covariate coefficients. These are used by the preprocessing pipeline. The tool prints an R² per sensor; anything below 0.6 means the sensor is dominated by drift or noise and needs investigation.

### 19. Sensor-swap pilot

**Objective:** detect reactor-specific effects and sensor-specific effects independently.

**Physical setup:**
1. Fill each reactor with sterile DI water + 16.8 g/L NaHCO₃ (Zarrouk without cells and micronutrients). This reproduces the alkaline aqueous headspace without biology.
2. Set reactors to nominal condition: 32 °C, LEDs on, stir 300 RPM, mild bubbling.

**Run the tool:**
```bash
jaxsr diagnose swap-pilot
```

The tool guides you through a Latin-square rotation: run 4 hours in assignment A, then it prompts you to physically rotate sensors one reactor-position to the left, then 4 more hours in assignment B, and so on. At the end it reports the fraction of variance explained by sensor identity, by reactor identity, and by residual. Both should be < 30%; higher than that means something is grossly asymmetric (leaky vial, bad tubing run, dying lamp) and must be fixed.

---

## Part VI — In-Situ Sensor Calibration

**Purpose:** build a per-sensor calibration relating voltage to VOC concentration, without a gas manifold. You do this immediately before every experiment run.

### 20. Standard-addition calibration

**Physical setup:**
1. Prepare a **cell-free** reactor with fresh Zarrouk medium at the same temperature as the coming experiment.
2. Ensure all sensors are connected to their assigned reactors per the rotation schedule.

**Run the tool:**
```bash
jaxsr calibrate --experiment {experiment_id}
```

The first thing the tool asks — before anything else happens — is what calibration compound you're using:

```
Calibration gas / standard?
  [1] Isobutylene       (RF = 1.00, reference)
  [2] Isoprene          (RF = 0.63)
  [3] Acetone           (RF = 1.10)
  [4] Methanol          (RF = 10.0)
  [5] Ethanol           (RF = 10.0)
  [6] DMS               (RF = 0.44)
  [7] Toluene           (RF = 0.53)
  [8] Other — enter manually
```

Pick whichever matches the standard you actually have on the bench — the response factor (RF) is applied automatically for compound-agnostic reporting. If you're using a compound that isn't listed, pick "Other" and the tool will ask for the name, molecular weight, and (if you know it) the response factor. If the RF is unknown, the tool still runs, but the output will be in "ppm of your compound" rather than "isobutylene-equivalent ppm" — and it will warn you accordingly.

If you already know your compound and want to skip the menu, use `--calibration-gas` at the CLI:
```bash
jaxsr calibrate --experiment exp_03 --calibration-gas isoprene
```

Then the tool walks you through spike-and-recover:

1. It records 10 min of baseline. You do nothing except wait.
2. It prompts you to inject `V₁` µL of the calibration standard through the septum. Use a gas-tight syringe; hold the plunger down for 3 seconds; withdraw slowly.
3. Wait 5 minutes while the tool records response.
4. Repeat with `V₂` and `V₃` at bracketing levels.
5. The tool fits per-sensor sensitivity, reports R² per sensor, and stores the calibration alongside the compound identity and RF.

**Acceptance:** R² ≥ 0.95 per sensor. Below that, clean the lamp and retest. If the same sensor fails again, either replace it or exclude it from this run (the tool has a `--exclude PID03` flag).

**⚠️ Injection volumes:** Appendix B gives the calculation from spike level in ppm to injected volume, using the molecular weight of whatever compound you selected. Bring the worksheet to the bench.

**Do not switch calibration compound mid-campaign** without a good reason. If you do, the tool will warn you when you try to combine data across compounds; the numeric comparability is only preserved when both compounds have known response factors, and even then only via the isobutylene-equivalent column.

### 21. Reference jar (weekly, not per-run)

**Purpose:** track sensor-to-sensor drift over time without recalibrating each session.

**Physical setup once:**
1. In a 500 mL Schott bottle, mount a diffusion vial or permeation tube of **the same VOC you use for standard-addition calibration**, held at controlled T (e.g. water bath at 25 °C). Using the same compound for both keeps comparisons apples-to-apples.
2. Fit cap with inlet and outlet luers plus a septum.
3. Sparge house air through it at a known flow rate.
4. Register the reference compound with the tool once: `jaxsr calibrate --reference-jar --setup --gas isoprene` (or whatever you chose). The tool refuses to compare weekly readings across sensors with mismatched reference compounds.

**Weekly:**
```bash
jaxsr calibrate --reference-jar --sensor PID01
```
Repeat for each sensor. The tool prompts you to disconnect one PID at a time, connect to the reference jar for 10 minutes, then reconnect. Do all sensors on the same day so ambient conditions are shared.

The tool computes a ratio for each sensor against the fleet median and plots the time series. If any sensor drifts by >20% between weekly checks, schedule lamp cleaning at the next opportunity.

---

## Part VII — Standard Operating Procedure for an Experiment Run

The day-to-day workflow. Print this section if you want; it's the working document.

### 22. Day-before checklist

- [ ] Parent culture confirmed healthy under microscope (§15).
- [ ] Fresh Zarrouk medium prepared and autoclaved (§14).
- [ ] Sensor-swap rotation for tomorrow's run confirmed by reading `configs/rotation_schedule.yaml`.
- [ ] Next batch of experimental conditions pulled from the JAXSR DoE workflow: `jaxsr plan --show-next`. (The plan is generated by JAXSR's `ActiveLearner`; you just read the suggested conditions.)
- [ ] Calibration gas / permeation tube inventory checked; refill if < 20%.

### 23. Run day: pre-flight (30–45 minutes)

1. Power on Pi, PIDs, LEDs, pumps. **Wait 15 min for sensor warm-up.**
2. Run:
   ```bash
   jaxsr preflight
   ```
   The tool verifies every sensor is streaming with non-null T and RH, runs a 5-minute zero-noise check against the Part-V baseline, and prints a red/yellow/green summary. Do not proceed if any sensor is red.
3. Perform standard-addition calibration (§20).
4. If any sensor fails calibration acceptance, either replace it or exclude it from this run.

### 24. Run day: setup and start

1. Set each reactor to the condition assigned in the acquisition YAML.
2. Inoculate per §16.
3. Confirm LED, stir, heater, bubbling are running per spec.
4. Start logging:
   ```bash
   jaxsr start --experiment {experiment_id}
   ```
5. Take a photo of the setup and save to `docs/setup_photos/{experiment_id}.jpg`.

### 25. During the run

- Check the live dashboard hourly for the first 4 h, then every 4 h. Launch with:
   ```bash
   jaxsr dashboard
   ```
- Note anything anomalous in the run's metadata file immediately. Use:
   ```bash
   jaxsr note --experiment {experiment_id} "R03 LED flicker at 14:22, replaced 14:30"
   ```
- **Do not adjust conditions once the run has started.** If you must abort, record the abort time and reason before touching anything.

### 26. End of run

1. Stop logging:
   ```bash
   jaxsr stop --experiment {experiment_id}
   ```
2. Take final OD, pH, photograph the vials.
3. Run preprocessing:
   ```bash
   jaxsr process --experiment {experiment_id}
   ```
   The tool applies covariate correction, common-mode subtraction, spectral filtering, and voltage-to-ppm conversion. It writes the derived feature file that JAXSR's DoE workflow consumes.
4. Refit and plan the next batch: `jaxsr fit --campaign {id}` retrains the yield model, then `jaxsr plan --campaign {id}` proposes the next set of conditions.
5. Archive raw data by syncing `data/raw/experiments/{experiment_id}/` to backup.

---

## Part VIII — Troubleshooting

### 27. Weekly diagnostics (every Monday)

Run each of the following. All should pass; if any fails, escalate to the software team lead and the PI.

```bash
jaxsr diagnose fleet-zero --duration-min 30
jaxsr diagnose ambient --duration-h 12
jaxsr calibrate --reference-jar --all
jaxsr diagnose weekly-audit
```

The `weekly-audit` command checks:
- Whether sensor-id variance-share has drifted upward across recent runs.
- Whether backup is current.
- Whether any sensor is due for lamp cleaning.

### 28. Common failure modes

**One sensor stuck at a flat voltage during a run.**
Likely causes: dead lamp, clogged inlet filter, disconnected SHT/BME I²C (which then reads NaN and kills covariate correction), stuck ADC channel.
Action: check the I²C log first (`jaxsr diagnose i2c`). If clean, swap the inlet filter. If still flat, clean the lamp per AAN 306-03.

**Sinusoidal artifact persists after processing.**
Likely causes: training window didn't cover the full RH range; a hidden variable (door-opening cycle, HVAC) modulates baseline.
Action: re-run covariate fit using a longer training window (`jaxsr process --experiment {id} --training-window 90m`); consider adding room-airflow logging.

**Square-wave artifact.**
Likely causes: pump duty-cycling, LED PWM, a solenoid somewhere.
Action: verify pump PWM and LED command state are being logged. If they aren't, that's a bug; file it with the software team.

**Culture browns / bleaches within 24 h.**
Likely causes: photoinhibition (LEDs too close/bright), medium not truly Zarrouk (bicarbonate missing?), inoculum too dilute, temperature overshoot.
Action: verify lux measurement at vial surface; confirm medium recipe and pH; confirm inoculum ratio ≥ 15% v/v; confirm liquid T with independent probe.

**All sensors read consistently high (drift over campaign).**
Likely causes: shared reference (charcoal scrubber) is exhausted; ambient VOC increased in the lab.
Action: replace the charcoal scrubber and rerun fleet zero.

### 29. When to escalate to Layer 1 (hardware conditioning)

Watch for these signs; if any persist across two full campaigns, tell the PI it's time to invest in Nafion drying, heated sample lines, and a gas manifold:

- Sensor-id variance-share consistently > 30% despite covariate correction.
- Model prediction intervals plateau above the target and won't shrink with more data.
- Persistent Lomb-Scargle artifact peaks that survive filtering.
- Standard-addition R² drifts below 0.9 within a single week.

---

## Appendices

### Appendix A — Zarrouk medium (per liter, distilled water)

| Component | Mass | Notes |
|-----------|------|-------|
| NaHCO₃ | 16.8 g | Primary carbon source + buffer |
| K₂HPO₄ | 0.5 g | Phosphate |
| NaNO₃ | 2.5 g | Nitrogen (varied by DoE) |
| K₂SO₄ | 1.0 g | Potassium |
| NaCl | 1.0 g | Sodium (baseline; variable additions in DoE) |
| MgSO₄·7H₂O | 0.2 g | Magnesium |
| CaCl₂·2H₂O | 0.04 g | Calcium (precipitates at high pH; add last) |
| FeSO₄·7H₂O | 0.01 g | Iron |
| EDTA (Na₂) | 0.08 g | Chelator |
| A5 trace metals | 1 mL | See below |

**A5 trace metals (per liter):** H₃BO₃ 2.86 g; MnCl₂·4H₂O 1.81 g; ZnSO₄·7H₂O 0.222 g; Na₂MoO₄·2H₂O 0.39 g; CuSO₄·5H₂O 0.079 g; Co(NO₃)₂·6H₂O 0.0494 g.

**Preparation order:**
1. Dissolve NaHCO₃ separately in ~200 mL distilled water.
2. Dissolve remaining salts (except CaCl₂) in ~700 mL distilled water.
3. Combine both solutions.
4. Add CaCl₂ solution last with vigorous stirring to prevent precipitate.
5. Adjust volume to 1 L.
6. Autoclave at 121 °C for 20 min.
7. Cool. Verify pH 9.5–10.0.

### Appendix B — Standard-addition calculation

These formulas work for **any** calibration compound; the resulting ppm is in units of the compound being injected. The tool applies the response factor separately, downstream.

**For a certified reference gas** of concentration `C_ref_ppm` injected as volume `V_inj_mL` into a headspace volume `V_hs_mL`:

```
ΔC_headspace_ppm  ≈  C_ref_ppm × V_inj_mL / V_hs_mL
```

Example: 10 ppm cylinder, inject 0.5 mL into a 20 mL headspace →
`ΔC ≈ 10 × 0.5 / 20 = 0.25 ppm` of the calibration compound.

**For a liquid standard** injected into a sealed reactor (assumes full headspace equilibration):

```
n_VOC     =  V_liquid_µL × ρ_liquid_g_per_mL × 10⁻³  /  MW_g_per_mol
n_gas_hs  =  V_headspace_mL × P_atm × 10⁻³            /  (R × T)
C_ppm     ≈  (n_VOC / n_gas_hs) × 10⁶
```

`MW_g_per_mol` is the molecular weight of your specific compound — the tool populates this from the `CalibrationGas` record you selected at the start of calibration. A worked example spreadsheet at `docs/calibration_worksheet.xlsx` recalculates automatically when you change the compound in cell B1.

**A note on response factors.** Photoionization detectors respond to different compounds with different sensitivities. Published response factors are relative to isobutylene = 1.0. If you calibrate with isoprene (RF = 0.63), a sensor reading equivalent to 10 ppm of the "isobutylene reference" corresponds to `10 / 0.63 ≈ 16 ppm` of isoprene at the sensor. The tool handles this conversion internally; you don't need to do the arithmetic. But you should know why the same physical signal produces different numeric ppm values depending on which standard you chose.

### Appendix C — Glossary

- **BME280 / SHT35** — small I²C temperature/humidity sensors.
- **JAXSR** — the symbolic regression + DoE package that provides both the pre-experimental calibration/preprocessing subsystem (via the augmentation) and the modeling/active-learning core.
- **PAR** — photosynthetically active radiation, 400–700 nm, in μmol photons m⁻² s⁻¹.
- **PID (sensor)** — photoionization detector; always Alphasense in this protocol.
- **Zarrouk medium** — the standard high-bicarbonate alkaline medium for *Arthrospira*.

### Appendix D — References

- Alphasense Application Notes AAN 301 (PID principles), AAN 306-03 (PID maintenance).
- Alphasense PID product datasheets (PID-A1, PID-AH, PID-AH2, PID-A15).
- Ma, Z. and Gao, K. (2010). Spiral breakage and photoinhibition of *Arthrospira platensis*. *Environ. Exp. Bot.* 68: 208–213.
- Achyuthan, K. E. *et al.* (2017). Volatile metabolites emission by *in vivo* microalgae. *Molecules* 22: 1330.
- Wang, F. *et al.* (2021). PID-type sensors performance evaluations in ambient air. *Sensors and Actuators B* 330: 129327.

### Appendix E — Open questions to resolve

1. Exact Alphasense PID model per sensor.
2. ~~Target VOC compound(s)~~ — resolved: the pipeline is VOC-agnostic and the calibration compound is picked at calibration time. Users decide per run; the tool records it and applies the response factor automatically.
3. Number of reactors and sensors in first build.
4. pH probes in each reactor, or offline sampling only?
5. Backup and archival policy for raw Parquet data.
