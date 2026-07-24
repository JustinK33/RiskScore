"""Command-line entrypoint for the MVP logistic regression baseline."""

import argparse

from risk_score.config import load_yaml_config
from risk_score.evaluation import CostMatrix
from risk_score.pipeline import run_baseline_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the credit risk baseline MVP pipeline.")
    parser.add_argument(
        "--raw-data-path",
        required=True,
        help="Path to Lending Club CSV or parquet.",
    )
    parser.add_argument("--train-end-date", required=True, help="Last issue date in the train set.")
    parser.add_argument(
        "--test-start-date",
        required=True,
        help="First issue date in the test set.",
    )
    parser.add_argument("--date-column", default="issue_d", help="Origination date column.")
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "xgboost"],
        default="logistic_regression",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model_config.yaml",
        help="YAML config with split, model, and threshold settings.",
    )
    parser.add_argument(
        "--schema-config",
        default="configs/dataset_schema.yaml",
        help="YAML config with dataset column aliases.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory for metrics and figures.",
    )
    parser.add_argument("--false-negative-cost", type=float, default=5.0)
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    """Run the MVP baseline pipeline."""
    args = parse_args()
    config = load_yaml_config(args.model_config)
    schema_config = load_yaml_config(args.schema_config)
    selected_model_config = config.get(args.model_type, {})
    try:
        metrics = run_baseline_pipeline(
            args.raw_data_path,
            train_end_date=args.train_end_date,
            test_start_date=args.test_start_date,
            date_column=args.date_column,
            output_dir=args.output_dir,
            model_type=args.model_type,
            model_config=selected_model_config,
            schema_config=schema_config,
            cost_matrix=CostMatrix(
                false_negative_cost=args.false_negative_cost,
                false_positive_cost=args.false_positive_cost,
            ),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from None
    print(metrics)


if __name__ == "__main__":
    main()
