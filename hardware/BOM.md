# Bill of materials

Components confirmed for this project's build (see [`CLAUDE.md`](../CLAUDE.md)'s "Confirmed real hardware" section and [`docs/hardware_setup.md`](../docs/hardware_setup.md) for the full wiring and setup detail this list is drawn from). Quantities are per single reactor; multiply out for a multi-reactor setup.

| Component | Spec / part | Qty | Notes |
|---|---|---|---|
| Single-board computer | Raspberry Pi 4 | 1 | Runs `algaesense-edge` — sensor acquisition, LED control, and the network API. |
| VOC (gas) sensor | Alphasense PID + ISB (analog 0–3.3V output) | 1 | Interchangeable Sensor Board (ISB) provides the conditioned analog output the ADC reads. |
| ADC | ADS1115 (Adafruit STEMMA QT breakout) | 1 | I2C address `0x48`. Reads the VOC sensor on channel 0 (A0), single-ended. |
| Temperature/humidity sensor | BME280 | 0–1 | Not yet acquired as of this writing — the software already supports it (`Bme280TRHSensorReader`); the system runs fine without one, with those readings recorded as null. |
| LED strip | WS2811 addressable RGB strip (ALITOVE) | 1 | Pixel count varies by reel/length purchased — count the strip's actual addressable LEDs and set `--led-num-pixels` accordingly (see `docs/hardware_setup.md` §6). |
| Level shifter | 74AHCT125 (3.3V → 5V) | 1 | Between the Pi's GPIO18 data line and the LED strip's data input — the Pi's 3.3V logic level isn't reliably read by the 5V WS2811 protocol without this. |
| Resistor | 470Ω | 1 | In series on the level-shifted data line into the strip's data-in pin — standard practice for WS2811/NeoPixel-style strips to reduce reflections/ringing on the data line. |
| LED power supply | 12V DC supply, current rating sized to strip length/brightness | 1 | Powers the LED strip directly — **not** drawn from the Pi. Must share a common ground with the Pi and level shifter (see wiring notes below). |
| Camera | Raspberry Pi Camera Module v1 (OV5647 sensor), CSI connector | 1 | Connects via the Pi's CSI ribbon cable; powered through the cable itself, no separate supply needed. |
| PAR/quantum meter (optional but recommended) | e.g. Apogee MQ-500 series, ~$100–200 | 0–1 | Not currently used — `par_per_full_duty_umol_m2_s` is presently derived from a phone lux meter and an approximate lux-to-PPFD conversion factor (see `docs/hardware_setup.md` §7). Every PAR value in this system inherits that approximation's uncertainty until a real PAR meter replaces this step. |

## Wiring summary

(Full detail in [`docs/hardware_setup.md`](../docs/hardware_setup.md) section 5 — this is the short version.)

- **VOC sensor**: ISB signal out → ADS1115 A0; ISB 3.3V power → Pi's 3.3V rail; grounds common.
- **LED strip**: Pi GPIO18 → 74AHCT125 input → 470Ω resistor → strip data-in. Strip's +V comes from the separate 12V supply, not the Pi. Grounds must be common across the Pi, the level shifter, and the LED supply.
- **Camera**: CSI ribbon cable only, no separate power.

## CAD files

Physical enclosure/mounting CAD files go in [`cad/`](cad/) alongside this BOM — see that folder for the expected format.
