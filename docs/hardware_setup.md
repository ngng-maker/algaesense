# Hardware setup: SSH access and powering the rig

This covers two things you asked for help with: getting SSH working between your remote laptop and the Pi, and safely powering the Pi + the 12V LED strip supply together. Written for your confirmed hardware: Alphasense PID + ISB → ADS1115 (0x48) → Pi, WS2811 (ALITOVE) strip on GPIO18 through a 74AHCT125 level shifter, OV5647 camera module.

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

## 4. Wiring and power sequencing

Your confirmed wiring:

- **VOC (Alphasense PID + ISB → ADS1115 @ 0x48)**: ISB signal out → ADS1115 A0, ISB 3.3V power → Pi 3.3V rail, grounds common.
- **LED (WS2811 strip)**: GPIO18 → 74AHCT125 input → 470Ω resistor → strip data in. Strip's +V from a **separate 12V supply**, not the Pi. **Grounds must be common** between the Pi, the level shifter, and the strip — the level shifter can't translate the data signal correctly without a shared ground reference, and an ungrounded strip can develop stray voltage that damages the Pi's GPIO pin.
- **Camera**: CSI ribbon cable, powered through the cable itself — no separate supply.

Power-on sequence that avoids surprises:

1. **Grounds first**: with everything unpowered, double-check the common ground connection between Pi, level shifter, and LED strip's 12V supply ground.
2. **Power the 12V supply first, Pi second** (or simultaneously) — powering the Pi first with the LED strip's data line floating (level shifter unpowered) can occasionally cause the strip to flash garbage colors briefly; not damaging, just a startup glitch worth knowing about rather than being alarmed by.
3. Once both are up, run `algaesense-edge scan-i2c` first (cheap, non-destructive) to confirm the ADS1115 shows up at 0x48 before doing anything with the LED.
4. Test the LED at low brightness first (e.g. `par_per_full_duty` calibration run's first spike should be a small one) rather than commanding full brightness on the very first real test — standard practice for any new wiring, not specific to this project.

## 5. Confirming the pixel count

`create_hardware_led(gpio_pin, num_pixels, ...)` needs the strip's actual pixel count — I don't know this number for your ALITOVE strip. Count the individually-addressable LEDs on it (usually printed on the reel or in the product listing, e.g. "60 LED/m × length"), and use that value everywhere `--led-num-pixels` appears (the CLI) or `_TEST_NUM_PIXELS` (the hardware-marked tests, which you should update to match before running `pytest -m hardware` for real).

## 6. Running it for real

```
algaesense-edge start \
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
