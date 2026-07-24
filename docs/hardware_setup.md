# Hardware setup: SSH access and powering the rig

This covers two things you asked for help with: getting SSH working between your remote laptop and the Pi, and safely powering the Pi + the 12V LED strip supply together. Written for your confirmed hardware: Raspberry Pi 4, Alphasense PID + ISB → ADS1115 (0x48) → Pi, WS2811 (ALITOVE) strip on GPIO18 through a 74AHCT125 level shifter, OV5647 camera module.

## 1. Enabling SSH on the Pi

If the Pi already has an OS installed and you have physical/keyboard access to it at some point:

1. `sudo raspi-config` → `Interface Options` → `SSH` → `Enable`.
2. Find its IP address: `hostname -I` on the Pi itself, or check your router's connected-devices list.

If you're flashing a fresh SD card (or the Pi isn't set up yet) and only have remote access to this laptop:

1. Use **Raspberry Pi Imager** (on this laptop) to write Raspberry Pi OS to the SD card.
2. Before writing, click the gear icon (⚙) / "Edit Settings" — this lets you pre-configure, on the same laptop, without ever touching the Pi directly:
   - Hostname (e.g. `algaesense-r01`)
   - Enable SSH, and set a password (or paste a public key if you already have one)
   - Wi-Fi SSID/password, if the Pi will connect wirelessly
3. Write the image, move the SD card to the Pi, power it on. It should appear on the network within a minute or two, reachable at `algaesense-r01.local` (mDNS) or whatever IP your router assigns it.

## 2. Connecting from this laptop

Once SSH is enabled and you know the Pi's hostname or IP:

```
ssh pi@algaesense-r01.local
```

(or `ssh pi@<ip-address>` if `.local` mDNS resolution doesn't work on your network — Windows sometimes needs [Bonjour](https://support.apple.com/kb/dl999) installed for `.local` names to resolve, or just use the IP directly.)

First connection will ask you to confirm the host key — type `yes`.

**Important:** I (Claude Code) don't have a way to reach your Pi directly from this session — I have no network path to it, and I'm not going to ask you to expose it to the internet or hand me credentials to store. The practical workflow is: I prepare code changes here, you `git pull` (or `scp`/copy) them onto the Pi and run the actual hardware-facing commands yourself over your own SSH session, and tell me what happened (errors, output) so I can help debug from there. If you'd rather, you can paste terminal output back to me directly.

### Connecting over Tailscale specifically: verify the IP first, don't assume it

If you're connecting from a Tailscale-joined "brain" machine rather than the same local network, **confirm the Pi's actual Tailscale IP before troubleshooting anything else** — don't assume an address is correct just because it's the one you remember or the one you happened to try first. On a real bring-up, an incorrect address here can produce a confusing, multi-layered troubleshooting session that looks like an SSH problem, a firewall problem, or a service-not-running problem, when it's actually none of those.

Confirm the real address with either:
```
tailscale status
```
(run on either machine — lists every device on your tailnet with its actual assigned IP), or the Tailscale admin console (`login.tailscale.com/admin/machines`), which shows the same list in a browser.

**A concrete tell that you have the wrong address**: if a plain `ping` to the address you're using succeeds instantly (sub-1ms round trip) and reports `TTL=128`, you're very likely pinging a Windows machine, not the Pi — Raspberry Pi OS (Linux) replies to pings with a default TTL of **64**, not 128. If everything else (SSH, the network API, any other service) then gets "connection refused" against that same address while the Pi itself looks completely healthy when you check it directly, that combination is a strong sign the address itself is wrong, not that anything on the Pi is broken. Re-verify with `tailscale status` before going any further down other troubleshooting paths.

## 3. Getting the code onto the Pi

Simplest path, once the Pi has internet access and git installed:

```
git clone <your repo's remote, once you push one> algaesense
cd algaesense
python3 -m venv .venv
source .venv/bin/activate
pip install -e packages/jaxsr-calibration
pip install -e "packages/algaesense-edge[hardware]"
```

(This repo isn't pushed to a remote yet — it only exists locally on this laptop right now, as a local git repository. If you want, I can help you push it to a private GitHub repo so `git clone`/`git pull` onto the Pi is straightforward; just say so.)

## 4. OS-level setup the pip install alone won't handle

These are real, known requirements for this exact hardware combination on a Raspberry Pi 4 running Raspberry Pi OS — none of this is hypothetical, but none of it has been verified against your actual Pi yet either, since this dev environment has no way to run it for real.

**GPIO/PWM access needs root.** Driving the WS2811 strip goes through `rpi_ws281x` (underneath `adafruit-circuitpython-neopixel`), which needs direct access to the Pi's PWM/DMA hardware — this normally requires root. Run `algaesense-edge start` with `sudo` (using the venv's own Python explicitly, since `sudo` resets `PATH` and won't automatically use your activated venv):

```
sudo .venv/bin/algaesense-edge start ...
```

If you see a permission error mentioning `/dev/mem` or DMA, this is why.

**Onboard audio conflicts with the LED's PWM channel.** The Pi's built-in audio output uses the same PWM peripheral `rpi_ws281x` needs for the LED strip — with both active, the strip can flicker, show wrong colors, or not respond at all. Disable onboard audio before testing the LED: edit `/boot/config.txt` (or `/boot/firmware/config.txt` on newer Raspberry Pi OS releases) and set:

```
dtparam=audio=off
```

then reboot. You don't need Pi audio output for this project, so there's no downside to leaving it off permanently.

**`picamera2` needs a system package, not just pip.** It depends on `libcamera`'s Python bindings, which aren't purely pip-installable — the `[hardware]` extra may report success while the actual import still fails on the Pi. Install the system package too:

```
sudo apt update
sudo apt install -y python3-picamera2
```

If your virtualenv doesn't see it afterward, recreate the venv with `--system-site-packages` so it can see the system-installed `picamera2`, rather than trying to `pip install picamera2` into an isolated venv.

**Confirmed real failure**: pip's own `picamera2` (the one `pip install -e "packages/algaesense-edge[hardware]"` tries to fetch) pulls in a transitive dependency, `python-prctl`, which needs to compile a C extension against `libcap`'s development headers — and that build fails with `error: subprocess-exited-with-error ... You need to install libcap development headers to build this module` if they aren't installed. Fix, in order:

```
sudo apt install -y python3-picamera2 libcap-dev
```

then recreate the venv with `--system-site-packages` (delete and redo the `python3 -m venv` step from section 3 above with `--system-site-packages` added) and re-run the two `pip install -e ...` commands. With `--system-site-packages`, pip should report `picamera2` as "Requirement already satisfied" from the apt-installed one rather than trying to build it from source again — which is what you actually want anyway, since the apt package is the one with real `libcamera` bindings.

**`ffmpeg` is a required system dependency, not an optional remux tool.** Confirmed on real hardware: `picamera2`'s `H264Encoder` always writes a raw H.264 elementary stream regardless of the output path's extension, and `run_camera_tick` (this project's own code) names every clip `.mp4` — so `process_clip`'s `cv2.VideoCapture` reliably failed to open them (`moov atom not found`; the file genuinely wasn't a valid MP4, not an OpenCV-build quirk). Fixed in code: `Picamera2CameraCapture.record_clip` now records through picamera2's `FfmpegOutput`, which shells out to the real `ffmpeg` binary to mux a genuine MP4 container directly — no manual remux step needed anymore, but `ffmpeg` itself must be installed for this to work:

```
sudo apt install -y ffmpeg
```

**Raspberry Pi 4 specifically**: no additional compatibility concerns beyond the above — `rpi_ws281x` has long-standing, solid support for the Pi 4's GPIO/PWM architecture. (This is a real, current concern on a Raspberry Pi 5 instead, which moved GPIO handling to a separate RP1 chip that older PWM+DMA-based libraries don't uniformly support yet — not applicable to your setup.)

## 5. Wiring and power sequencing

Your confirmed wiring:

- **VOC (Alphasense PID + ISB → ADS1115 @ 0x48)**: ISB signal out → ADS1115 A0, ISB 3.3V power → Pi 3.3V rail, grounds common.
- **LED (WS2811 strip)**: GPIO18 → 74AHCT125 input → 470Ω resistor → strip data in. Strip's +V from a **separate 12V supply**, not the Pi. **Grounds must be common** between the Pi, the level shifter, and the strip — the level shifter can't translate the data signal correctly without a shared ground reference, and an ungrounded strip can develop stray voltage that damages the Pi's GPIO pin.
- **Camera**: CSI ribbon cable, powered through the cable itself — no separate supply.

Power-on sequence that avoids surprises:

1. **Grounds first**: with everything unpowered, double-check the common ground connection between Pi, level shifter, and LED strip's 12V supply ground.
2. **Power the 12V supply first, Pi second** (or simultaneously) — powering the Pi first with the LED strip's data line floating (level shifter unpowered) can occasionally cause the strip to flash garbage colors briefly; not damaging, just a startup glitch worth knowing about rather than being alarmed by.
3. Once both are up, run `sudo .venv/bin/algaesense-edge scan-i2c` first (cheap, non-destructive) to confirm the ADS1115 shows up at 0x48 before doing anything with the LED.
4. Test the LED at low brightness first (e.g. `par_per_full_duty` calibration run's first spike should be a small one) rather than commanding full brightness on the very first real test — standard practice for any new wiring, not specific to this project.

## 6. Confirming the pixel count

`create_hardware_led(gpio_pin, num_pixels, ...)` needs the strip's actual pixel count — I don't know this number for your ALITOVE strip. Count the individually-addressable LEDs on it (usually printed on the reel or in the product listing, e.g. "60 LED/m × length"), and use that value everywhere `--led-num-pixels` appears (the CLI) or `_TEST_NUM_PIXELS` (the hardware-marked tests, which you should update to match before running `pytest -m hardware` for real).

**`pytest -m hardware` needs `sudo` too, same as `algaesense-edge start`/`scan-i2c`** — it's easy to forget since `pytest` itself feels like "just running tests." Without root, the LED tests don't just fail cleanly; a first attempt raises a clean `RuntimeError: NeoPixel support requires running with sudo`, but a *second* LED test running afterward in the same process can hit a hard `Segmentation fault` instead — the underlying `rpi_ws281x` C library's module-level state is left partially initialized by the first failed attempt, and the next one crashes the whole Python process rather than raising cleanly. Confirmed real on this hardware. Run it as:
```
sudo .venv/bin/pytest -m hardware packages/algaesense-edge/tests
```

## 7. Deriving `--par-per-full-duty` without a real PAR meter

`--par-per-full-duty` (`par_per_full_duty_umol_m2_s` in code) is "how much PAR this specific LED/vial setup produces at 100% duty cycle" — every requested PAR value gets converted to a duty cycle by dividing by this number, so getting it wrong scales every light setpoint in the system by the same error.

This is **not** the same measurement as the experimentalist protocol's lux-meter step (which only checks you're under a ~15,000 lux photoinhibition ceiling, once, during LED installation). A plain lux meter can't give you PAR directly — lux is weighted to human eye sensitivity (peaks green, ~555nm), while PAR is a flat photon count over 400-700nm, so the conversion between them depends on your specific LED's spectrum, not a universal constant.

Without a dedicated quantum/PAR meter, derive an approximate value:

1. With the vial in place and empty, run the LED at 100% duty cycle (e.g. `create_hardware_led(...).set_duty_cycle(1.0)` directly, same one-off approach as the first-test order below) and measure illuminance at the vial surface with a phone lux meter — the same physical setup as the protocol's ceiling check, just read at full brightness instead of only checking it's under the limit.
2. Multiply that lux reading by a conversion factor appropriate for your LED's spectrum. For generic white LEDs, roughly **0.014–0.02 μmol·m⁻²·s⁻¹ per lux** is a commonly cited starting range — but this varies by color temperature/spectrum and isn't a value you should treat as precise. If your LED's datasheet or product listing states a color temperature (e.g. "6500K daylight white" vs "3000K warm white"), lean toward the lower end of that range for warmer/redder LEDs and the higher end for cooler/bluer ones, as a rough rule of thumb — but treat this as a starting estimate, not a substitute for a real measurement.
3. Use the result as `--par-per-full-duty`.

**Every PAR value this system reports or acts on inherits this approximation's uncertainty** until it's replaced with a real quantum/PAR meter reading (e.g. an Apogee MQ-500 series sensor, ~$100-200) — worth remembering if you're ever comparing a PAR number here against another lab's instrument-calibrated value, or trusting a discovered light-response equation's coefficients as physically exact.

## 8. A sane first-test order (don't jump straight to the full service)

Each of these isolates one piece, so a failure tells you exactly where to look instead of a tangle of "it didn't work":

1. `sudo .venv/bin/algaesense-edge scan-i2c` — confirms the ADS1115 responds, no LED/camera involved.
2. A one-off Python check reading the VOC voltage directly (`create_hardware_voc_reader().read_voltage_mv()`), to confirm real sensor data before anything else touches it.
3. The LED alone, at low brightness, via `create_hardware_led(...).set_duty_cycle(0.1)` directly, before wiring it into the full service.
4. The camera alone — record one clip, then manually try `process_clip` on it to check for the H.264/mp4 issue above before it's buried inside a running service.
5. Only then run the full `algaesense-edge start` below.

## 9. Running it for real

```
sudo .venv/bin/algaesense-edge start \
  --experiment exp_2026-07-XX_batch01 \
  --reactor R01 \
  --sensor PID01 \
  --camera CAM01 \
  --max-par <your reactor's safety ceiling> \
  --par-per-full-duty <measured, see calibration protocol> \
  --led-gpio-pin 18 \
  --led-num-pixels <your strip's actual count> \
  --led-pixel-order BRG \
  --voc-i2c-address 0x48
```

No `--trh-i2c-address` flag is passed here — the service runs fine without a temperature/humidity sensor (those columns just come back null), until you add a BME280 later.

## 10. Running it as a systemd service, so Slack can restart it

Running the command above directly in your SSH session works, but it dies the moment you close that session, and there's no way for `apply_start_new_experiment_run` (the Slack tool that "starts a new experiment") to restart it — that tool works by running `sudo systemctl restart algaesense-edge`, which needs a real systemd service by that name to already exist.

A template unit file is committed at `packages/algaesense-edge/deploy/algaesense-edge.service`. To install it:

1. If you already `git pull`ed the latest code on the Pi, the template is already present at that path. **Don't edit it in place and commit the result** — copy it somewhere local first (or just edit the copy in step 2 below), since `WorkingDirectory`/`ExecStart`'s repo-path placeholder is specific to your machine (fill it in with `pwd`'s actual output from your SSH session) and shouldn't get pushed back to the shared template.
2. Copy and install it, filling in the placeholders in the copy:
   ```
   sudo cp packages/algaesense-edge/deploy/algaesense-edge.service /etc/systemd/system/algaesense-edge.service
   sudo nano /etc/systemd/system/algaesense-edge.service   # fill in WorkingDirectory/ExecStart's path, --max-par, --par-per-full-duty
   sudo systemctl daemon-reload
   sudo systemctl enable --now algaesense-edge
   ```
3. Confirm it's actually running: `sudo systemctl status algaesense-edge` (should show `active (running)`), and `curl localhost:8000/health` should return healthy.

Every restart of this unit generates a fresh `experiment_id` from the current date/time (the `$(date ...)` inside `ExecStart`) — that's what lets a Slack-triggered restart count as "starting a new experiment," with no other state to reset.

**Confirmed real bug, fixed in the template (2026-07-24): `ExecStart` must use `exec`.** An earlier version of the template read `ExecStart=/bin/bash -c '<path>/.venv/bin/algaesense-edge start ...'` (no `exec`) — this leaves bash running as a separate parent process with `algaesense-edge` as its *child*, and systemd tracks bash's PID as the unit's main process. Bash has no SIGTERM trap of its own, so on `systemctl restart` it can exit almost immediately; once systemd sees the tracked main PID gone, it can forcefully clean up the rest of the cgroup, killing the still-running Python child *before* it ever finishes its own graceful shutdown -- confirmed directly: the acquisition loop's flush-on-shutdown code never ran (verified via raw file-write diagnostics that never got created), even though uvicorn's own internal shutdown logged as completely clean. **Every restart silently dropped that run's last partial hour of VOC/camera data** until this was fixed. If you installed the unit before this date, edit `/etc/systemd/system/algaesense-edge.service` and add `exec` right after the opening quote in `ExecStart` (`/bin/bash -c 'exec <path>/.venv/bin/algaesense-edge start ...'`), then `sudo systemctl daemon-reload` -- no restart needed immediately, but the fix won't take effect until the next one.

For the one-time `NOPASSWD` sudoers rule that lets a scripted SSH connection (from your brain machine, via `apply_start_new_experiment_run`) actually run `sudo systemctl restart algaesense-edge` without a password prompt, see `docs/remote_experiment_control.md`.
