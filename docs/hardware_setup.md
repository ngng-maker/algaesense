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

**A likely camera format mismatch, not yet resolved in code.** `picamera2`'s `H264Encoder` writes a raw H.264 elementary stream. `process_clip` (this project's own code) reads recorded clips via `cv2.VideoCapture`, which may not open a raw `.h264` file directly depending on the OpenCV build's codec support — you may see `process_clip: could not open video file at ...` even though the clip recorded fine. If that happens, remux it into a real container first:

```
ffmpeg -i clip.h264 -c copy clip.mp4
```

(`sudo apt install -y ffmpeg` if it's not already there.) This is a real, known gap — if it turns out to happen every time rather than being an edge case, tell me and I'll fix `Picamera2CameraCapture` to write directly into an `.mp4` container instead (picamera2 supports this via `FfmpegOutput`), so this manual remux step goes away entirely.

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
