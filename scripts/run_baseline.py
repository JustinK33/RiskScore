"""Command-line entrypoint for the baseline pipeline.

Slated for deletion: the ``riskscore`` CLI replaces this and
``scripts/serve_dashboard.py`` together. Kept working in the meantime so the
project has a runnable entrypoint at every commit.
"""

import argparse

from risk_score.config import RunConfig, load_run_config
from risk_score.pipeline import run_baseline_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fit and report one credit risk model.")
    parser.add_argument("--raw-data-path", required=True, help="Path to a Lending Club CSV.")
    parser.add_argument(
        "--config",
        default="configs/run.yaml",
        help="Run configuration: split windows, costs, hyperparameters, feature tier.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "xgboost"],
        default="logistic_regression",
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Directory for metrics and figures."
    )
    return parser.parse_args()


def main() -> None:
    """Run the pipeline and print the holdout metrics."""
    args = parse_args()
    # Every knob lives in the config file, so there is exactly one place a split
    # window or a cost can be set and no chance of two disagreeing.
    config = load_run_config(args.config) if args.config else RunConfig()
    metrics = run_baseline_pipeline(
        args.raw_data_path,
        config=config,
        output_dir=args.output_dir,
        model_type=args.model_type,
    )
    print(metrics)


if __name__ == "__main__":
    main()
