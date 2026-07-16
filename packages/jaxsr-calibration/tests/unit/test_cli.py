"""Unit tests for the CLI (cli.py).

Most commands are still Milestone-1-style stubs (every command/group is
registered, `--help` works, required options are enforced, but the body just
reports "not implemented"). The three `diagnose` subcommands with an offline
`--input` path (fleet-zero, ambient, swap-pilot) are real as of Milestone 2 --
tested here against synthetic Parquet fixtures. There is no `diagnose i2c`
here -- real I2C bus scanning lives in algaesense_edge instead (see that
package's own CLI tests).
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from jaxsr_calibration.cli import cli
from tests.fixtures.synthetic_readings import make_fleet_readings


def test_top_level_help_lists_all_commands() -> None:
    # `CliRunner` lets us invoke a click command exactly like a user would
    # from a real terminal, but in-process (no subprocess, no real stdin/stdout) --
    # `result.output` captures whatever would have been printed, and
    # `result.exit_code` captures the process exit status.
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command_name in ["preflight", "calibrate", "start", "stop", "note", "process", "diagnose", "dashboard"]:
        assert command_name in result.output


def test_diagnose_group_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["diagnose", "--help"])
    assert result.exit_code == 0
    for subcommand in ["fleet-zero", "ambient", "swap-pilot", "weekly-audit"]:
        assert subcommand in result.output


def test_preflight_stub_exits_nonzero_with_clear_message() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["preflight"])
    # click.ClickException causes exit code 1, not a raw traceback.
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_calibrate_requires_experiment_option() -> None:
    runner = CliRunner()
    # Omitting the required --experiment option should be rejected by click
    # itself, before our stub body ever runs -- so the failure message talks
    # about a missing option, not "not implemented yet".
    result = runner.invoke(cli, ["calibrate"])
    assert result.exit_code != 0
    assert "--experiment" in result.output


def test_calibrate_with_builtin_gas_flag_resolves_and_reports_no_live_backend() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["calibrate", "--experiment", "exp_03", "--calibration-gas", "isoprene"])

    assert result.exit_code == 1
    assert "isoprene" in result.output
    assert "RF=0.63" in result.output
    assert "live acquisition" in result.output


def test_calibrate_with_unknown_builtin_gas_reports_clear_error() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["calibrate", "--experiment", "exp_03", "--calibration-gas", "unobtainium"])

    assert result.exit_code == 1
    assert "not in the response-factor table" in result.output


def test_calibrate_custom_gas_requires_compound_name_and_mw() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["calibrate", "--experiment", "exp_03", "--calibration-gas", "custom"])

    assert result.exit_code == 1
    assert "--compound-name" in result.output


def test_calibrate_custom_gas_with_known_response_factor() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "calibrate", "--experiment", "exp_03", "--calibration-gas", "custom",
            "--compound-name", "myVOC", "--mw", "88.15", "--response-factor", "0.75",
        ],
    )

    assert result.exit_code == 1
    assert "myVOC" in result.output
    assert "RF=0.75" in result.output


def test_calibrate_interactive_menu_selects_second_option() -> None:
    runner = CliRunner()
    # `input="2\n"` feeds "2" (then Enter) to the first click.prompt() call
    # the command makes -- CliRunner simulates real terminal stdin this way,
    # so we're exercising the exact interactive path spec §25 describes,
    # including the printed menu, without a real terminal.
    result = runner.invoke(cli, ["calibrate", "--experiment", "exp_03"], input="2\n")

    assert "Calibration gas / standard?" in result.output
    assert "[2] Isoprene" in result.output
    assert "Confirmed: isoprene, RF = 0.63" in result.output


def test_calibrate_interactive_menu_other_option_prompts_for_custom_details() -> None:
    runner = CliRunner()
    # Sequence: select "Other" (8), then compound name, MW, "no" to knowing
    # RF -- four prompts in the order _resolve_calibration_gas asks them.
    result = runner.invoke(
        cli, ["calibrate", "--experiment", "exp_03"], input="8\nmyVOC\n88.15\nn\n"
    )

    assert "Confirmed: myVOC, RF = unknown" in result.output


def test_calibrate_reference_jar_flag_is_still_a_stub() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["calibrate", "--experiment", "exp_03", "--reference-jar"])

    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_note_accepts_experiment_and_positional_text() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["note", "--experiment", "exp_03", "R03 LED flicker"])
    # Reaches the stub (proving both the option and positional argument
    # parsed correctly) and then reports not-implemented, same as any other
    # stub command.
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_diagnose_fleet_zero_without_input_reports_no_live_backend() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["diagnose", "fleet-zero"])
    assert result.exit_code == 1
    assert "--input" in result.output


def test_diagnose_fleet_zero_with_input_prints_summary(tmp_path: Path) -> None:
    readings = make_fleet_readings(
        {"PID01": {"mean_mv": 0.5, "std_mv": 0.1, "slope_mv_per_min": 0.0}}, seed=42
    )
    parquet_path = tmp_path / "readings.parquet"
    readings.write_parquet(parquet_path)

    runner = CliRunner()
    result = runner.invoke(cli, ["diagnose", "fleet-zero", "--input", str(parquet_path)])

    assert result.exit_code == 0
    assert "Fleet-zero summary: GREEN" in result.output
    assert "PID01: PASS" in result.output


