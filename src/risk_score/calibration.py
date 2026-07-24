"""Calibration analysis stubs for credit default probability models."""

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def compute_calibration_curve(
    y_true: pd.Series,
    y_score: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute observed default rates versus predicted probabilities by bin.

    TODO:
        Add sample counts per bin for richer diagnostics.
    """
    observed, predicted = calibration_curve(
        y_true,
        y_score,
        n_bins=n_bins,
        strategy="quantile",
    )
    return pd.DataFrame(
        {
            "mean_predicted_probability": predicted,
            "observed_default_rate": observed,
        }
    )


def plot_calibration_curve(
    calibration_data: pd.DataFrame,
    *,
    output_path: str | None = None,
) -> Any:
    """Create a calibration plot and optionally save it to disk.

    TODO:
        Add styling conventions for final report charts.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(
        calibration_data["mean_predicted_probability"],
        calibration_data["observed_default_rate"],
        marker="o",
        label="Model",
    )
    ax.set_xlabel("Mean predicted default probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration Plot")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    return fig


def calibrate_model(
    model: Any,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    *,
    method: str = "isotonic",
) -> Any:
    """Fit a post-training probability calibration wrapper.

    TODO:
        Ensure callers reserve a calibration split distinct from the final test
        set.
    """
    if method not in {"isotonic", "sigmoid"}:
        raise ValueError("Calibration method must be `isotonic` or `sigmoid`.")

    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    return calibrated.fit(x_calibration, y_calibration)
