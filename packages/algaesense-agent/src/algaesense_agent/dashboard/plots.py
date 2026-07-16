"""On-demand plot rendering: turns a campaign's fitted model into a PNG
image, for the agent to post into Slack.
"""

from __future__ import annotations

import io

import matplotlib

"""
`matplotlib.use("Agg")` selects the non-interactive, file/bytes-only
rendering backend, before `pyplot` is ever imported -- this process has
no display to draw a window onto (it's a headless MCP server), and Agg is
the standard choice for exactly that situation.
"""
matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt


def render_fit_curve_png(
    x: np.ndarray,
    y: np.ndarray,
    predict_fn,
    x_label: str,
    y_label: str,
    title: str,
) -> bytes:
    """Plot observed (x, y) points against a fitted model's predicted
    curve, returned as PNG image bytes."""

    """
    `predict_fn` is any callable `x_grid -> y_pred` (deliberately not
    typed as `jaxsr.SymbolicRegressor` specifically) -- keeps this
    function usable from a unit test with a plain lambda, without needing
    a real fitted model or any of jaxsr_calibration's data-loading
    machinery just to test that the plot itself renders correctly.
    """

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    y_grid = predict_fn(x_grid)

    figure, axes = plt.subplots(figsize=(6, 4))
    axes.scatter(x, y, label="observed", color="tab:blue")
    axes.plot(x_grid, y_grid, label="fitted model", color="tab:orange")
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()

    """
    Rendering to an in-memory `io.BytesIO` buffer (rather than a temp
    file on disk) keeps this function side-effect-free -- the caller
    decides whether the returned bytes get written to disk, embedded
    directly in an MCP `Image` result, or something else.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)

    return buffer.getvalue()
