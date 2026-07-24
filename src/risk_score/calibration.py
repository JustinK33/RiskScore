"""Calibration analysis stubs for credit default probability models."""

from typing import Any

import pandas as pd


def compute_calibration_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute observed default rates versus predicted probabilities by bin.

    TODO:
        Use scikit-learn calibration utilities or a custom binning strategy and
        return a dataframe suitable for plotting and reporting.
    """
    raise NotImplementedError("Compute calibration curve.")


def plot_calibration_curve(
    calibration_data: pd.DataFrame,
    *,
    output_path: str | None = None,
) -> Any:
    """Create a calibration plot and optionally save it to disk.

    TODO:
        Plot predicted probability against observed default frequency, include
        a diagonal reference line, and save to `reports/figures/`.
    """
    raise NotImplementedError("Plot calibration curve.")


def calibrate_model(
    model: Any,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    *,
    method: str = "isotonic",
) -> Any:
    """Fit a post-training probability calibration wrapper.

    TODO:
        Support isotonic and sigmoid calibration after a validation split has
        been selected without contaminating the final test set.
    """
    raise NotImplementedError("Calibrate model probabilities.")
