"""Command-line interface for jaxsr-calibration."""

from __future__ import annotations

from pathlib import Path

import click
import polars as pl

from jaxsr_calibration.calibration.models import CalibrationGas
from jaxsr_calibration.diagnostics.ambient import run_ambient_baseline
from jaxsr_calibration.diagnostics.fleet_zero import run_fleet_zero
from jaxsr_calibration.diagnostics.swap_pilot import run_swap_pilot


"""
Several commands here only have real behavior for their "analyze
already-collected data" half (--input <parquet file>); the "drive live
hardware" half raises LiveAcquisitionNotAvailableError until
algaesense-edge exists to actually run it, same split documented in
jaxsr_calibration.errors.

We use `click` rather than `typer` for this project: click's decorators
are more explicit about what they do, which suits a codebase meant to be
read and learned from line by line.
"""


"""
The interactive menu's builtin options, in the exact order spec §25's
worked transcript lists them (note: the underlying response-factor table
also has "benzene", reachable via --calibration-gas=benzene, but the
interactive menu itself only shows what the spec's transcript shows).
"""
_MENU_BUILTIN_GASES = ["isobutylene", "isoprene", "acetone", "methanol", "ethanol", "dms", "toluene"]


def _resolve_calibration_gas(
    calibration_gas: str | None,
    compound_name: str | None,
    mw: float | None,
    response_factor: float | None,
) -> CalibrationGas:
    """Resolve which calibration compound this run uses, from flags or an
    interactive menu."""

    """
    Implements spec §25's gas-selection step: either resolve directly from
    CLI flags (scripted/non-interactive use), or fall back to the
    interactive menu transcript the spec shows verbatim.
    """

    if calibration_gas == "custom":
        if compound_name is None or mw is None:
            raise click.ClickException(
                "--calibration-gas=custom requires --compound-name and --mw."
            )
        return CalibrationGas.custom(name=compound_name, mw=mw, response_factor=response_factor)

    if calibration_gas is not None:
        try:
            return CalibrationGas.builtin(calibration_gas)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    """
    No --calibration-gas given at all: interactive menu, matching spec
    §25's transcript.
    """
    click.echo("Standard-addition calibration")
    click.echo("-" * 31)
    click.echo("Calibration gas / standard?")
    menu_gases = [CalibrationGas.builtin(name) for name in _MENU_BUILTIN_GASES]
    for i, gas in enumerate(menu_gases, start=1):
        click.echo(f"  [{i}] {gas.name.capitalize():<18}(RF = {gas.response_factor:.2f})")
    other_choice = len(menu_gases) + 1
    click.echo(f"  [{other_choice}] Other -- enter manually")

    """
    `click.IntRange(min, max)` is a type converter (like the Path type used
    elsewhere in this file) that restricts the parsed integer to a range --
    click re-prompts automatically if the user types something outside [1,
    other_choice] rather than us having to validate it by hand.
    """
    selection = click.prompt(
        "Selection", type=click.IntRange(1, other_choice), default=1
    )
    if selection == other_choice:
        name = click.prompt("Compound name")
        entered_mw = click.prompt("Molecular weight (g/mol)", type=float)
        knows_rf = click.confirm("Do you know the response factor?", default=False)
        entered_rf = click.prompt("Response factor", type=float) if knows_rf else None
        gas = CalibrationGas.custom(name=name, mw=entered_mw, response_factor=entered_rf)
    else:
        gas = menu_gases[selection - 1]

    rf_display = f"{gas.response_factor:.2f}" if gas.has_rf else "unknown"
    click.echo(f"Confirmed: {gas.name}, RF = {rf_display} ({gas.source}).")
    return gas


"""
`click.Path(exists=True, dir_okay=False, path_type=Path)` is a *type
converter* for an option's value: click will (1) verify the given path
exists on disk, (2) reject it if it's a directory, and (3) hand our
function a real `pathlib.Path` object rather than a plain string -- all
three checks happen before our command body ever runs, same idea as
`required=True` below.
"""
_PARQUET_INPUT_OPTION = click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Parquet file of already-collected readings, for offline analysis "
    "(live acquisition isn't available until algaesense-edge exists).",
)


def _require_input(input_path: Path | None, command_name: str) -> pl.DataFrame:
    """Shared guard for the diagnose subcommands that can analyze offline
    data: fail clearly if --input wasn't given, otherwise load the Parquet
    file into a DataFrame."""

    if input_path is None:
        raise click.ClickException(
            f"'{command_name}' has no live-acquisition backend yet (needs "
            "algaesense-edge, a later milestone). Pass --input <parquet file> "
            "to analyze already-collected data instead."
        )
    return pl.read_parquet(input_path)


def _not_implemented(command_name: str) -> None:
    """Shared helper for every stub command below."""

    """
    `click.ClickException` is click's built-in way of failing a command
    *cleanly*: click catches it, prints `message` to stderr, and exits the
    process with status code 1 -- unlike letting a raw Exception propagate,
    which would also print a full Python traceback that's confusing to a
    user who just typed a command wrong or hit an unimplemented feature.
    """
    raise click.ClickException(
        f"'{command_name}' is not implemented yet (this is a Milestone 1 scaffold)."
    )


"""
`@click.group()` turns the plain function below it into a *command group*:
a parent command whose only job is to hold other subcommands (like how
`git` itself does nothing, but `git commit` and `git push` are subcommands
of it). Every `@cli.command()` or `@cli.group()` below attaches itself to
this one.
"""


@click.group()
def cli() -> None:
    """jaxsr-cal: pre-experimental calibration, diagnostics, and preprocessing
    for JAXSR (VOC PID sensors + camera-based biomass estimation)."""

    """
    A group function's body normally does nothing (as here) -- click calls
    it once, then dispatches to whichever subcommand the user actually
    asked for. This docstring becomes the `--help` text for `jaxsr-cal`
    with no subcommand given.
    """


@cli.command()
def preflight() -> None:
    """Verify every sensor is streaming and run a zero-noise check."""
    _not_implemented("preflight")


@cli.command()
@click.option("--experiment", required=True, help="Experiment ID, e.g. exp_2026-07-15_batch03.")
@click.option(
    "--calibration-gas",
    default=None,
    help="Calibration compound name (e.g. isoprene), or 'custom'. Omit for an interactive menu.",
)
@click.option("--compound-name", default=None, help="Required when --calibration-gas=custom.")
@click.option("--mw", type=float, default=None, help="Molecular weight (g/mol). Required when --calibration-gas=custom.")
@click.option("--response-factor", type=float, default=None, help="RF relative to isobutylene, if known.")
@click.option("--reference-jar", is_flag=True, default=False, help="Manage the reference jar instead of running standard-addition.")
def calibrate(
    experiment: str,
    calibration_gas: str | None,
    compound_name: str | None,
    mw: float | None,
    response_factor: float | None,
    reference_jar: bool,
) -> None:
    """Run standard-addition calibration for the given experiment."""

    if reference_jar:
        """
        Reference-jar setup/rotation (spec's `--reference-jar --setup
        --gas ...` / `--reference-jar --sensor ... | --all` variants) is
        entirely live-hardware-driven with no offline analysis split, same
        situation as the spike-and-recover procedure below -- stays a stub.
        """
        _not_implemented("calibrate --reference-jar")

    """
    Gas selection itself needs no hardware at all, so it's fully real --
    only the spike-and-recover procedure after it requires live
    acquisition.
    """
    gas = _resolve_calibration_gas(calibration_gas, compound_name, mw, response_factor)

    raise click.ClickException(
        f"Calibration gas resolved: {gas.name} (RF="
        f"{f'{gas.response_factor:.2f}' if gas.has_rf else 'unknown'}). "
        "Standard-addition spike-and-recover requires live acquisition "
        "(needs algaesense-edge, a later phase). Once you have collected "
        "spike-and-recover data, call fit_sensitivity_per_sensor(df, "
        "method=...) directly instead of this CLI command."
    )


@cli.command()
@click.option("--experiment", required=True)
def start(experiment: str) -> None:
    """Start raw sensor logging for an experiment."""
    _not_implemented("start")


@cli.command()
@click.option("--experiment", required=True)
def stop(experiment: str) -> None:
    """Stop raw sensor logging for an experiment."""
    _not_implemented("stop")


@cli.command()
@click.option("--experiment", required=True)
@click.argument("text")
def note(experiment: str, text: str) -> None:
    """Append a freeform operator note to an experiment's metadata."""

    """
    `click.argument` (unlike `click.option`) is a *positional* value -- the
    operator types `jaxsr-cal note --experiment exp_03 "R03 LED flicker"`
    rather than needing a `--text` flag for the note's contents.
    """
    _not_implemented("note")


@cli.command()
@click.option("--experiment", required=True)
@click.option("--force-version", type=int, default=None)
def process(experiment: str, force_version: int | None) -> None:
    """Run the full preprocessing/fusion pipeline for a completed experiment."""
    _not_implemented("process")


@cli.command()
def dashboard() -> None:
    """Launch the live monitoring dashboard (Milestone 6 / stretch goal)."""
    _not_implemented("dashboard")


"""
A *nested* group: `jaxsr-cal diagnose fleet-zero`, `jaxsr-cal diagnose
ambient`, etc. all live under this "diagnose" parent, mirroring spec Part
XI exactly.
"""


@cli.group()
def diagnose() -> None:
    """Sensor-health diagnostics (fleet-zero, ambient, swap-pilot, weekly)."""


@diagnose.command("fleet-zero")
@click.option("--duration-min", type=int, default=60)
@_PARQUET_INPUT_OPTION
def diagnose_fleet_zero(duration_min: int, input_path: Path | None) -> None:
    """Check every sensor reads ~0 with low noise on clean air."""

    readings = _require_input(input_path, "diagnose fleet-zero")

    """
    LiveAcquisitionNotAvailableError can't actually happen here (we always
    pass readings=), but fit_covariate_model-adjacent errors could still
    surface from bad/insufficient data, so we don't blanket-catch beyond
    what _require_input already guards.
    """
    result = run_fleet_zero(duration_min=duration_min, readings=readings)

    click.echo(f"Fleet-zero summary: {result.summary_status}")
    for sensor_id, stats in result.per_sensor.items():
        click.echo(
            f"  {sensor_id}: {stats['status']} "
            f"(mean={stats['mean_mv']:.2f}mV std={stats['std_mv']:.2f}mV "
            f"slope={stats['slope_mv_per_min']:.3f}mV/min)"
        )


@diagnose.command("ambient")
@click.option("--duration-h", type=int, default=12)
@click.option("--method", type=click.Choice(["ols", "robust"]), default="ols")
@_PARQUET_INPUT_OPTION
def diagnose_ambient(duration_h: int, method: str, input_path: Path | None) -> None:
    """Characterize each sensor's response to room temperature/humidity."""

    readings = _require_input(input_path, "diagnose ambient")
    result = run_ambient_baseline(duration_h=duration_h, method=method, readings=readings)

    for sensor_id, r_squared in result.r_squared_per_sensor.items():
        flag = "" if r_squared >= 0.6 else "  (below 0.6 -- needs investigation)"
        click.echo(f"  {sensor_id}: R²={r_squared:.3f}{flag}")


@diagnose.command("swap-pilot")
@click.option("--n-blocks", type=int, default=4)
@_PARQUET_INPUT_OPTION
def diagnose_swap_pilot(n_blocks: int, input_path: Path | None) -> None:
    """Latin-square sensor/reactor swap to separate sensor vs. reactor effects."""

    readings = _require_input(input_path, "diagnose swap-pilot")
    result = run_swap_pilot(n_blocks=n_blocks, readings=readings)

    for source, share in result.variance_share.items():
        click.echo(f"  {source}: {share:.1%}")


@diagnose.command("weekly-audit")
def diagnose_weekly_audit() -> None:
    """Run the full weekly diagnostic rollup."""

    """
    Unlike fleet-zero/ambient/swap-pilot, weekly-audit composes *other
    diagnostics' already-computed results* (a list of past
    SwapPilotResults plus SensorConfigs) rather than one raw readings
    table -- there's no single obvious --input file for that yet, so this
    command stays a stub until the config/results loading story is built.
    """
    _not_implemented("diagnose weekly-audit")


"""
There is deliberately no "diagnose i2c" command here: scanning a physical
I2C bus is real hardware I/O, which belongs in algaesense_edge (the
Raspberry-Pi-side package), not in this hardware-agnostic analysis
package. Run `algaesense-edge scan-i2c` instead once that package is
installed on the Pi.
"""


"""
This is what `[project.scripts]` in pyproject.toml points at
("jaxsr_calibration.cli:cli") -- when someone types `jaxsr-cal` in a
terminal, this is the object that actually runs.
"""
if __name__ == "__main__":
    """
    Lets you also run this file directly for local debugging, e.g.
    `python -m jaxsr_calibration.cli --help`, without installing the
    package.
    """
    cli()
