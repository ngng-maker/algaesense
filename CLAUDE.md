# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Keep it current — see "Development log" at the bottom for the standing instruction to log mistakes and confirmed-good patterns here as they happen, not just at milestones.

## What this project is

AlgaeSense is an agentic monitoring/control system for an IoT algae (*Arthrospira*/*Spirulina platensis*) cultivation bioreactor: VOC emissions via an Alphasense PID sensor, biomass via a Pi camera (greenness-based), symbolic regression + active-learning experiment design via the upstream `jaxsr` package, and a conversational agent (Hermes Agent + Slack) that can fit models, suggest next experiments, and adjust the LED — with a human confirmation gate before anything touches hardware.

It started from two spec documents under `Resources/` (`jaxsr_augmentation_spec.md`, the authoritative software spec; `spirulina_voc_experimentalist_protocol.md`, the human lab protocol) describing a `jaxsr-calibration` package. The build went well beyond that original spec's scope — see Architecture below for what actually exists now.

## Repository status

All three planned packages are built, tested, and (for `algaesense-edge`) wired to specific, confirmed real hardware. This is a `uv` workspace monorepo; each package is also independently `pip install`-able.

```
algaesense/
├── CLAUDE.md
├── Resources/                      # original spec docs (background/history, not authoritative for current code)
├── docs/
│   ├── hardware_setup.md           # SSH setup + Pi/LED power sequencing for THIS project's confirmed wiring
│   └── slack_and_hermes_setup.md   # Hermes install, Anthropic key, Slack app creation walkthrough
├── packages/
│   ├── jaxsr-calibration/          # hardware-agnostic "brain": calibration, diagnostics, processing, camera
│   ├── algaesense-edge/            # Raspberry Pi hardware I/O: acquisition, actuators, network API
│   └── algaesense-agent/           # MCP servers Hermes connects to: mcp_pipeline, mcp_actuators, dashboard, labwiki, profile
└── pyproject.toml                  # uv workspace root (members = ["packages/*"], not itself installable)
```

## Architecture

Three packages, each with its own `pyproject.toml`, `src/` (src-layout), and `tests/{unit,integration,fixtures}/`:

- **`jaxsr-calibration`** — never touches hardware. Calibration (standard-addition, reference-jar cross-check, camera zero-point), diagnostics (fleet-zero, ambient baseline, swap-pilot, weekly audit), processing (covariate regression, common-mode subtraction, spectral filtering, dual-rate VOC/camera fusion), raw schemas (`logging_/schema.py`: `VOC_RAW_SCHEMA`, `CAMERA_RAW_SCHEMA`, `ExperimentMeta`). Depends on and calls the real upstream `jaxsr` package directly (`jaxsr.SymbolicRegressor`, `jaxsr.ActiveLearner`, `jaxsr.BasisLibrary`, etc.) — `jaxsr_calibration` is its own top-level import root, not merged into `jaxsr.*`'s namespace (upstream has no plugin mechanism for that). CLI: `jaxsr-cal`.
- **`algaesense-edge`** — runs ON the Raspberry Pi. Sensor acquisition (VOC PID via ADS1115 ADC, optional T/RH via BME280, camera via picamera2), actuator control (LED via WS2811 addressable strip), a small FastAPI network API so the brain machine never needs SSH into the Pi for data. Re-validates every actuator setpoint against configured safety bounds independently of whatever the caller sends (`UnsafeSetpointError`) — this is the actual safety boundary, not a suggestion. CLI: `algaesense-edge`.
- **`algaesense-agent`** — runs on the "brain" server, never on the Pi. Five MCP servers Hermes Agent (a real, separately-installed conversational agent) connects to as a client over stdio: `mcp_pipeline` (read-only: fit/suggest via jaxsr), `mcp_actuators` (propose/apply split — only `apply_led_change` has a side effect), `dashboard` (on-demand plot MCP tool + a separate live Streamlit app), `labwiki` (Karpathy LLM-wiki-pattern knowledge base: raw sources, index/log/summaries/entities pages, ingest/query/lint), `mcp_calibration` (guided step-by-step wizards for standard-addition/reference-jar/camera-zero calibration — durable file-backed session state per in-progress calibration, real math from `jaxsr_calibration.calibration`/`camera.calibration`, no hardware touched so no propose/apply gate needed). `profile/` holds the Hermes system prompt and config templates — not Python code Hermes imports.

Data layout (unchanged from the original spec):
- Raw: `data/raw/experiments/{experiment_id}/{sensor_id|camera_id}={value}/hour=YYYY-MM-DDTHH.parquet`
- Derived features: `data/derived/features/{campaign_id}/{experiment_id}.parquet`
- Labwiki: `data/labwiki/{campaign_id}/{raw/,wiki/}`

## Setup & running

Each package installs independently:
```
pip install -e packages/jaxsr-calibration
pip install -e "packages/algaesense-edge[hardware]"   # [hardware] only installs/imports on the Pi
pip install -e "packages/algaesense-agent[dev]"
```

Tests, **run per package separately** (not combined in one `pytest` invocation — each package defines its own top-level `tests/__init__.py`, and pytest's import system collides across them if both paths are given at once):
```
pytest packages/jaxsr-calibration/tests
pytest packages/algaesense-edge/tests
pytest packages/algaesense-agent/tests
```

`algaesense-edge` and `algaesense-agent` both use `@pytest.mark.hardware` for tests that need real hardware (WS2811 strip, ADS1115, camera) — skipped by default (`addopts = "-m 'not hardware'"`), run for real on the Pi with `pytest -m hardware`.

For the Streamlit live dashboard: `streamlit run packages/algaesense-agent/src/algaesense_agent/dashboard/streamlit_app.py` (a `.claude/launch.json` config named `streamlit-dashboard` exists for `preview_start`).

For Hermes/Slack/Anthropic setup and Pi hardware/SSH setup: see `docs/slack_and_hermes_setup.md` and `docs/hardware_setup.md`.

## Confirmed real hardware (algaesense-edge)

- **VOC sensor**: Alphasense PID + ISB (analog 0-3.3V output) → **ADS1115 ADC** (Adafruit STEMMA QT), I2C address **0x48**, channel 0 (A0), single-ended.
- **T/RH sensor**: not yet acquired. `Bme280TRHSensorReader` is a real, ready driver for when one is added — until then, `trh_reader` is `None` throughout and `sample_t_c`/`sample_rh_pct` are recorded as null (schema already supports this).
- **LED**: WS2811 addressable RGB strip (ALITOVE), **GPIO18** (BCM), through a 74AHCT125 level shifter (3.3V→5V) + 470ohm resistor, powered from a separate 12V supply with common ground to the Pi. Driven via `adafruit-circuitpython-neopixel`, **pixel_order="BRG"** (confirmed by testing — not the library's GRB default). `num_pixels` has no default; it must match the real strip's actual count.
- **Camera**: Raspberry Pi Camera Module v1 (OV5647), CSI connection, via `picamera2`.

## Coding conventions

- **Comments**: a short, simple docstring for `help()`/introspection (what it does, no technical detail), plus a *separate* triple-quoted string block placed above the corresponding code for developer-facing rationale/non-obvious detail — not `#` comments, and not folded into the docstring. Blank-line spacing between chunks for readability. This is a deliberate, explicit project convention (not the tool's usual "minimal comments" default).
  - **Exception: Streamlit entry-point scripts** (anything run via `streamlit run x.py`, e.g. `dashboard/streamlit_app.py`). Streamlit's "magic commands" feature renders every bare top-level string expression as literal page content — confirmed by testing — including ones inside a function body once called; only the true first-statement docstring is exempt. Use plain `#` comments in those files instead. This does NOT apply to modules merely imported by a Streamlit app, only to the script actually passed to `streamlit run`.
- **No markdown tables in chat responses to this user** — use lists/headers (horizontal scroll on tables is inconvenient for them). This is about chat responses, not file content — tables inside project markdown docs are fine.
- **Plain-English explanation before code** when explaining how a system/codebase works, especially for "explain X to me" style requests — lead with a narrative story (no file names, no code), then go to code/file-level detail only once asked or for "show me the code" style requests.
- Ousterhout "A Philosophy of Software Design" principles apply throughout: deep modules over shallow ones, avoid needless duplication across subpackages, separate general-purpose from special-purpose code, pull complexity downward, define errors out of existence where reasonable, don't merge genuinely distinct ideas into one file/class just to reduce file count.
- Mock/fake hardware or service stand-ins are **not used** in this codebase as of 2026-07-16 (see Development log) — real hardware classes and real in-process apps (e.g. `httpx.ASGITransport` against the actual FastAPI app) are used instead, even in tests. Pure numeric/text test fixtures for hardware-independent logic (e.g. a synthetic video file to test `process_clip`, synthetic Parquet rows to test a fit) are fine and are not "mocks" in the sense that was ruled out — the distinction is stand-in *hardware/service* implementations vs. plain *input data* for pure functions.

## External tools & accounts this project depends on

- **`jaxsr`** — real, installed upstream package (github.com/jkitchin/jaxsr) providing `SymbolicRegressor`, `ActiveLearner`, `BasisLibrary`, `Constraints`, etc. `jaxsr_calibration` depends on and calls it directly.
- **`mcp`** (official Model Context Protocol Python SDK) — used to build the four `algaesense-agent` MCP servers (`mcp.server.fastmcp.FastMCP`).
- **Hermes Agent** (`pip install hermes-agent`, Nous Research) — real, pip-installable, MIT-licensed. Acts as an **MCP client** (connects to our servers via `~/.hermes/config.yaml`, stdio subprocesses) plus owns its own Slack gateway and cron scheduler. We do not write code that runs inside Hermes; we write servers it connects to.
- **Anthropic API key** — needed for Hermes's own LLM calls (`hermes config set ANTHROPIC_API_KEY ...`, never pasted into a chat session).
- **Slack app + bot token** — see `docs/slack_and_hermes_setup.md` for creating it (manifest generator: `hermes slack manifest --agent-view --write`).
- **Streamlit**, **slack_sdk**, **pandas** — the live dashboard's dependencies (in `algaesense-agent`'s main deps, not an extra, since the dashboard ships as part of the package).

## Known gotchas (found the hard way — don't reintroduce)

- **pyarrow Hive-partitioning false positive**: `pyarrow.parquet.read_table(path)` auto-detects Hive-style partitioning from ANY `key=value` segment in a file's path — including this project's own raw-data layout (`sensor_id=PID01/hour=...parquet`) — and invents a phantom partition column that collides with the real same-named data column already in the file (`ArrowTypeError: Unable to merge: Field sensor_id has incompatible types: string vs dictionary<...>`). Fix: read via `polars.read_parquet(path)` instead (no such auto-detection), `.to_arrow().cast(schema)` if you need it back as a pyarrow Table for `concat_tables`. Affects `algaesense_edge/acquisition/writer.py`; watch for it anywhere else that reads files back from this partition layout with raw pyarrow.
- **Streamlit magic commands** — see Coding conventions above.
- **This repo has no git history before 2026-07-16** — it was a plain directory until a safety checkpoint commit was made right before a large, hard-to-reverse deletion pass (removing all mock/fake test doubles). If you're about to do something similarly sweeping and destructive, check `git status`/commit a checkpoint first, same as then.
- **Cross-package `pytest` collision** — see Setup & running above; always run each package's tests as a separate invocation.

## Standing/deferred requirements (not yet built — confirm scope before starting)

- **Agent-generated LED control profiles** (ramps, sinusoidal cycles, etc., per JAXSR's DoE recommendation) — NOT a fixed profile-type library; the intended design is Hermes generating a Python control script per-experiment and it running on the Pi with results/scripts logged per run, still re-validated by `LEDActuator`'s bounds-check. Real design work, not started.
- **Physical hardware verification** — the hardware-backed code (`voc.py`, `camera.py`, `actuators.py`, etc.) is written against confirmed real specs but has not been run on the actual Pi yet (this dev environment is a remote laptop, not the Pi). Run `pytest -m hardware` on the Pi itself once physically accessible, per `docs/hardware_setup.md`.
- **`run_weekly_audit` sensor-health diagnostics are not wrapped as an MCP tool** — only the labwiki-consistency lint pass is cron-able today; wiring the diagnostics themselves in is a natural follow-up (new `mcp_diagnostics` server or an addition to `mcp_pipeline`).

## Development log

**Standing instruction**: whenever something fails and gets fixed, an approach is confirmed correct after genuine uncertainty, or a non-obvious lesson surfaces during work on this repo, add a dated entry below. Keep entries short: what happened, why, what to do differently (or keep doing) as a result. This is a shared engineering log for this repo, not a personal memory — write it so any future session (human or Claude) can use it without re-deriving the lesson.

### 2026-07-16

- **Removed all mock/fake hardware and service test-doubles project-wide** (`MockLEDHardware`, `MockVOCSensorReader`, `MockTRHSensorReader`, `MockCameraCapture`, `GpiozeroLEDHardware`, `httpx.MockTransport` fake-handler tests, a `FakeSlackChannel` test harness) at the user's explicit request, after they confirmed understanding it would reduce automated test coverage on non-Pi dev machines. Replaced with: real hardware classes (tests split via `@pytest.mark.hardware`), real in-process FastAPI apps via `httpx.ASGITransport` instead of hand-written fake HTTP handlers, and removing the fake-Slack simulation entirely (the propose/apply tool split is what's actually verified, not a simulated chat). **Lesson**: this kind of instruction has large blast radius across the whole codebase — always init/checkpoint git first if no version control safety net exists yet, and read every affected file's real, current content before rewriting (don't rely on memory of what was there).
- **`pip install streamlit` silently downgraded `pyarrow` (25.0.0 → 24.0.0)** in the shared venv, which then surfaced the Hive-partitioning bug above as a test failure. **Lesson**: after installing a new dependency into a shared environment, always re-run the full test suite for every package, not just the one being worked on — a transitive dependency conflict can break unrelated code silently.
- **Confirmed real Hermes Agent architecture via direct web research before writing any MCP server code**, rather than assuming from training knowledge: it's a real pip package (`hermes-agent`), acts as an MCP *client* (not a place to run server code), and has its own Slack gateway + cron scheduler configured via `~/.hermes/config.yaml` and its own CLI. **Lesson**: for any library/framework claim central to an architecture decision, verify via WebSearch/WebFetch against current docs rather than relying on training-data recall, especially when the claim is old (from an earlier planning session) and might have changed or been imprecise.
- **`neopixel.NeoPixel`'s `pixel_order` accepts a plain string directly** (e.g. `"BRG"`) — it is not required to be one of the library's predefined name constants (`RGB`/`GRB`/`BGR`/`RGBW`/`GRBW`), which don't include `BRG` at all. Caught before shipping by checking the library's real source rather than assuming a `getattr(neopixel, "BRG")` lookup would work.

### 2026-07-17

- **Built the guided calibration wizard** (`algaesense-agent`'s new `mcp_calibration` server): step-by-step standard-addition/reference-jar/camera-zero flows, each backed by a durable YAML session file (`mcp_calibration/sessions.py`) rather than in-memory state, since a session can span many minutes across separate tool calls and an in-memory dict would be lost if the MCP server subprocess restarted mid-session. All the actual math/persistence reuses existing, already-tested `jaxsr_calibration` functions (`fit_sensitivity_per_sensor`, `compute_fleet_ratios`, `compute_blank_baseline` + their real `persist_*` functions) — the wizard's own code is only sequencing and data collection.
- **A tool function's `from x import Y` placed *inside* the function body can't be monkeypatched via `monkeypatch.setattr(module, "Y", ...)`** — the patch target has to be a module-level name for that to work, since a function-local import re-resolves `Y` fresh from its own source module on every call, ignoring whatever the test patched onto the *calling* module's namespace. Hit this writing `record_camera_zero_step_from_edge`'s test; fixed by moving the `EdgeClient` import to module level and introducing an overridable `_build_edge_client()` factory function (mirroring the pattern `mcp_actuators/server.py` already used for exactly this reason) as the actual patch target. **Lesson**: any MCP tool that constructs a network client internally should build it via a small local factory function, not inline in the tool body, purely so tests can inject a fake transport through it later.
- **`EdgeClient` only had `recent_voc_readings`, no `recent_camera_readings`**, even though `algaesense-edge`'s API has exposed `GET /sensors/camera/recent` since Phase 1b — the gap only became visible when the camera-zero calibration wizard needed to pull a live camera reading. **Lesson**: when a new feature needs to call an existing service, double-check the client wrapper actually has a method for every endpoint the service exposes, don't assume symmetry with a sibling method (`recent_voc_readings`) that happens to already exist.
