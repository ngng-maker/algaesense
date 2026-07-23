# End-to-end test runbook: benchmark -> hardware -> Slack -> JAXSR/labwiki

A single checklist for testing the whole real system in one pass -- the
synthetic pre-calibration/JAXSR/labwiki benchmark, powering on the Pi's
hardware, confirming acquisition, using Slack to start a fresh experiment run
with an LED profile, and running JAXSR/labwiki for real against that run's
data. This is **not something that can be run from this dev environment**
beyond Step 0: this machine has no network path to your Pi (no Tailscale
route through a controller machine reaches the controlled machine's own
network, confirmed earlier in this project), so every hardware/Slack step
below is something you run yourself, on your own laptop and Pi.

Each step links back to the doc that already covers it in detail -- this
runbook is the *order*, not a duplicate of the content.

## Before you start

- [ ] Read `docs/hardware_setup.md` section 5 (wiring/power sequencing) and
      section 8 (first-test order) if you haven't run the hardware in a while
      -- the ground-first, 12V-before-Pi sequencing and the "low brightness
      first" LED test both matter for avoiding a startup glitch, not just
      good practice.
- [ ] Confirm you know the Pi's real Tailscale/network address right now
      (`tailscale status`, not a value from memory) -- see the "verify the IP
      first" subsection of `docs/hardware_setup.md` section 2. A wrong address
      produces a pile of confusing, unrelated-looking symptoms (SSH refused,
      "service not running") that all vanish once the address itself is
      fixed, per this project's own dev log.
- [ ] Confirm the Windows laptop still has Streamlit reachable (see
      "Troubleshooting" in `docs/slack_and_hermes_setup.md` for the
      Startup-folder `.bat` if it isn't auto-starting) and Hermes's gateway
      running (`hermes gateway run`, or already running in the background).

## Step 0 -- run the synthetic benchmark suite (on the laptop, no hardware needed)

Confirms the pre-calibration pipeline, JAXSR active learning, labwiki
integration, and dynamics discovery all still work correctly against known
ground truth, before testing them against real (noisier, unverifiable) hardware
data. This step needs no Pi at all -- run it from your repo checkout:

```
.venv/Scripts/python.exe packages/jaxsr-calibration/benchmarks/run_all.py
```

Takes about a minute. Confirm it finishes with `results/REPORT.md` and three
PNGs written, and skim the verdicts (correction should recover the true VOC
value roughly 25x more accurately than raw; the active-learning/DoE comparison
and the dynamics-recovery numbers should be in the same ballpark reported in
`packages/jaxsr-calibration/benchmarks/README.md` -- exact numbers will differ
slightly run to run, since `jaxsr`'s own fits aren't perfectly deterministic,
see the report's own caveat on this). If any step in this script errors out,
fix that before touching real hardware -- a broken synthetic pipeline will
only be harder to diagnose once real, noisy Pi data is involved too.

## Step 1 -- power on and pre-check the hardware (on the Pi)

1. Follow the power sequence in `docs/hardware_setup.md` section 5 (grounds
   first, 12V supply before/with the Pi, not after).
2. `sudo .venv/bin/algaesense-edge scan-i2c` -- confirms the ADS1115 responds
   at 0x48 before anything else touches the LED or camera.
3. If it's been a while since the LED strip's pixel count or `--par-per-full-duty`
   were last confirmed, re-check them against `docs/hardware_setup.md`
   sections 6-7 rather than assuming last time's values still apply.

## Step 2 -- start algaesense-edge for a real, fresh experiment (on the Pi)

Run the full command from `docs/hardware_setup.md` section 9, with a fresh
`--experiment` id for this test run:

```
sudo .venv/bin/algaesense-edge start \
  --experiment exp_YYYY-MM-DD_e2e_test \
  --reactor R01 \
  --sensor PID01 \
  --camera CAM01 \
  --max-par <your reactor's safety ceiling> \
  --par-per-full-duty <your confirmed value> \
  --led-gpio-pin 18 \
  --led-num-pixels <your strip's actual count> \
  --led-pixel-order BRG \
  --voc-i2c-address 0x48
```

Watch the console for the first few VOC/camera ticks logging successfully
before moving on -- this is the same "does the service even start cleanly"
check `docs/hardware_setup.md`'s first-test order builds toward.

If you're running this as the systemd unit instead of manually (recommended
for anything Slack will later restart), start/restart it there instead:
`sudo systemctl restart algaesense-edge`, then `journalctl -u algaesense-edge -f`
to watch the same ticks.

## Step 3 -- confirm the dashboard sees it live (on the laptop)

1. Open the Streamlit dashboard's **Live** view, point the sidebar's
   "algaesense-edge URL" field at the Pi's real address (not `localhost`,
   unless the dashboard itself runs on the Pi).
2. Confirm the experiment-info header shows the fresh `exp_YYYY-MM-DD_e2e_test`
   id, and both the VOC and camera charts are updating with real numbers, not
   stuck/flat.
3. If a VOC ppm calibration run exists, confirm the ppm chart's `[0, 5]`
   fixed-axis view looks sane; if not, the raw mV view is still meaningful --
   don't expect a placeholder-calibrated ppm chart to mean much yet.

## Step 4 -- confirm Slack/Hermes can see the live tools

Before touching an actuator, confirm the read-only path works first --
smaller blast radius if something's misconfigured:

> *"What are the most recent VOC readings for R01?"*

Should return a real, current-looking snapshot (see `system_prompt.md`'s
note that this is a point-in-time snapshot, not a stream -- the dashboard is
the actual live view). If this fails, fix it before Step 5 -- an actuator
command failing for the same underlying config reason is a worse place to
discover it.

## Step 5 -- start a fresh experiment run from Slack, with an LED profile

This is the real test of `docs/remote_experiment_control.md`'s whole
propose/confirm/apply chain, end to end, over your actual Pi:

> *"Start a new experiment run for R01 with the LED ramping from 0 to 150 PAR over 10 minutes"*

Confirm:

- [ ] Hermes describes the proposed restart AND the LED profile in plain
      language, and explicitly waits for your "yes" -- if it doesn't wait,
      treat that as a bug (see the confirmation-gate rule in
      `system_prompt.md`), not something to route around.
- [ ] After you confirm, the reply includes a **clickable dashboard link**
      that actually loads (per `ensure_dashboard_running`'s auto-start
      behavior for a local URL -- give it a few seconds if it says it's
      starting the dashboard for you).
- [ ] The dashboard's experiment-info header now shows a NEW, freshly
      timestamp-generated `experiment_id` -- confirming the restart actually
      happened, not just that Slack said it would.
- [ ] The LED visibly ramps over the following ~10 minutes (or check the
      dashboard/`journalctl` if you're not standing next to the rig).

## Step 6 -- confirm the data actually lands correctly afterward

1. Let the fresh experiment run for at least a few minutes, then pull it down
   (`algaesense-dashboard-sync --pull-from-pi ...` or your configured remote
   storage backend, per `docs/remote_storage_setup.md`).
2. Switch the dashboard to **Past experiment**, select the new experiment id,
   and confirm the VOC/camera history looks continuous and sane, not full of
   gaps from the restart.

## Step 7 -- run JAXSR and labwiki for real, over this experiment's real data

This closes the loop the synthetic benchmark (Step 0) already proved works in
principle -- now confirm it against real, noisy hardware data for the first
time. You'll need at least one earlier experiment on the same campaign besides
this run's (a single data point can't fit a meaningful model) -- if this is
truly the first experiment ever run, note that and expect `fit_campaign_model`
to say so rather than fabricate a fit.

1. **Ingest this experiment's result into labwiki:**
   > *"Ingest the experiment we just ran into labwiki, with a note that [describe anything you actually observed -- e.g. 'VOC seemed to peak partway through the ramp']."*

   Confirm Hermes calls `ingest_experiment` and reports back the campaign/
   reactor/sensor pages it wrote or updated.

2. **Query labwiki directly, independent of any suggestion:**
   > *"What has labwiki recorded so far for R01/this campaign?"*

   Confirm the reply reflects real prior entries (including the one just
   ingested), not a generic non-answer.

3. **Fit a model and get a plain suggestion:**
   > *"Fit a model for this campaign and suggest the next experiment conditions."*

   Confirm a real expression/fit comes back (not an error), plus a concrete
   next-point suggestion with `search_bounds` reported.

4. **Test the labwiki-informed two-step workflow directly** (see
   `system_prompt.md`'s description of this, and the CLAUDE.md dev-log entries
   on `bound_overrides` for the full design):
   > *"What conditions should we try next? Check labwiki for any relevant findings first."*

   Confirm Hermes: (a) calls the suggestion tool once plain, (b) reads the
   fit's trends and the labwiki findings' full content for a concrete
   insight, (c) if one exists, calls again with `bound_overrides` reflecting
   it, and (d) shows you **both** results with its specific reasoning -- not
   silently one or the other. If it only calls the tool once, or narrows the
   range without quoting the actual finding that justified it, that's a
   prompting issue worth flagging (the underlying mechanism is what the
   synthetic benchmark already validated in Step 0 -- this step is testing
   whether Hermes *follows* the intended workflow against real data, not
   whether the tool itself works).

5. **If this experiment used a control profile (a ramp/sinusoid, not a static
   setpoint):**
   > *"Discover the LED response dynamics for this experiment."*

   Confirm `discover_led_response_dynamics` returns a real discovered
   equation with `reactor_par_umol_m2_s` among the selected features -- if it
   raises an error about `reactor_par_umol_m2_s` being null, that means PAR
   wasn't actually being recorded during this run (check the control profile
   was really started, per Step 5).

## If something breaks

- SSH/connection issues that look like "the Pi is down" -- re-verify the
  Tailscale address first (see "Before you start" above) before debugging
  anything else; this has been the actual root cause before.
- `CancelledError` across every MCP server on Hermes startup -- see the
  Troubleshooting section of `docs/slack_and_hermes_setup.md`; usually
  resolves on a retry.
- A dashboard link that doesn't load -- confirm Streamlit wasn't killed by an
  unrelated `Stop-Process -Name python` earlier in the same troubleshooting
  session (a real, confirmed gotcha -- see the same Troubleshooting section).
- Permission errors deleting files during a pull-from-Pi sync -- see the
  `UMask=0000` (not `0002`) fix in `docs/remote_storage_setup.md`/CLAUDE.md's
  dev log.
