"""Unit tests for jaxsr_calibration.validation's shared input-validation helpers."""

from __future__ import annotations

import polars as pl
import pytest

from jaxsr_calibration.validation import require_columns, require_implemented_method


def test_require_columns_passes_when_all_present() -> None:
    df = pl.DataFrame({"a": [1], "b": [2]})
    require_columns(df, {"a", "b"}, "some_function")  # should not raise


def test_require_columns_raises_naming_function_and_missing_columns() -> None:
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match=r"some_function.*\['b', 'c'\]"):
        require_columns(df, {"a", "b", "c"}, "some_function")


def test_require_implemented_method_passes_for_implemented() -> None:
    require_implemented_method("ols", {"ols", "robust"}, "some_function")  # should not raise


def test_require_implemented_method_raises_naming_function_method_and_implemented_set() -> None:
    with pytest.raises(NotImplementedError, match=r"some_function\(method='symbolic'\).*\['ols', 'robust'\]"):
        require_implemented_method("symbolic", {"ols", "robust"}, "some_function")
