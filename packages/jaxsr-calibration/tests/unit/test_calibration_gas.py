"""Unit tests for CalibrationGas and the response-factor table loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jaxsr_calibration.calibration.models import CalibrationGas


def test_builtin_isobutylene_is_the_reference_compound() -> None:
    gas = CalibrationGas.builtin("isobutylene")

    assert gas.response_factor == 1.00
    assert gas.is_builtin is True
    assert gas.has_rf is True


def test_builtin_isoprene_matches_spec_worked_example() -> None:
    # Spec §24's own worked example: isoprene, RF=0.63, MW=68.12,
    # ie_ev=8.85, source="Alphasense AAN 305".
    gas = CalibrationGas.builtin("isoprene")

    assert gas.response_factor == pytest.approx(0.63)
    assert gas.mw == pytest.approx(68.12)
    assert gas.ie_ev == pytest.approx(8.85)
    assert "AAN 305" in gas.source


def test_builtin_lookup_is_case_insensitive() -> None:
    assert CalibrationGas.builtin("Isoprene").name == "isoprene"
    assert CalibrationGas.builtin("ISOPRENE").name == "isoprene"


def test_builtin_unknown_compound_raises_helpful_error() -> None:
    with pytest.raises(ValueError, match="not in the response-factor table"):
        CalibrationGas.builtin("unobtainium")


def test_custom_gas_with_known_response_factor() -> None:
    gas = CalibrationGas.custom(name="my_analyte", mw=88.15, response_factor=0.75)

    assert gas.has_rf is True
    assert gas.is_builtin is False
    assert gas.source == "user"  # dataclass default, not overridden here


def test_custom_gas_with_unknown_response_factor() -> None:
    # spec §24 Option 3: response_factor omitted entirely -- has_rf must be
    # False, not accidentally 0.0 or some other falsy-but-wrong value.
    gas = CalibrationGas.custom(name="my_analyte", mw=88.15)

    assert gas.response_factor is None
    assert gas.has_rf is False


def test_calibration_gas_is_frozen() -> None:
    gas = CalibrationGas.builtin("isobutylene")
    # `dataclasses.FrozenInstanceError` (a subclass of AttributeError) is
    # raised when you try to assign to a field of a frozen dataclass instance
    # after construction.
    with pytest.raises(AttributeError):
        gas.mw = 999.0


def test_overrides_file_replaces_builtin_entry_and_adds_new_one(tmp_path: Path) -> None:
    overrides_path = tmp_path / "response_factors_overrides.yaml"
    overrides_path.write_text(
        yaml.safe_dump(
            {
                "compounds": {
                    # Override the built-in isoprene RF with a lab-measured value.
                    "isoprene": {"mw": 68.12, "response_factor": 0.70, "source": "in-house 2026-06-01"},
                    # A brand-new compound not in the built-in table at all.
                    "limonene": {"mw": 136.23, "response_factor": 0.35, "source": "in-house 2026-06-01"},
                }
            }
        ),
        encoding="utf-8",
    )

    overridden_isoprene = CalibrationGas.builtin("isoprene", overrides_path=overrides_path)
    limonene = CalibrationGas.builtin("limonene", overrides_path=overrides_path)
    # Untouched built-in entries should still resolve normally even when an
    # overrides file is given.
    isobutylene = CalibrationGas.builtin("isobutylene", overrides_path=overrides_path)

    assert overridden_isoprene.response_factor == pytest.approx(0.70)
    assert overridden_isoprene.source == "in-house 2026-06-01"
    assert limonene.mw == pytest.approx(136.23)
    assert isobutylene.response_factor == 1.00
