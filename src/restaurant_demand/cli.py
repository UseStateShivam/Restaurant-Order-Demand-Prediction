"""Command-line interface (CLI) for Restaurant Order Demand Prediction workflows.

Provides entry points for:
- data validation and quality reporting
- model training and evaluation
- batch inference from CSV files
- launching the FastAPI serving interface and Streamlit web dashboard
"""

import argparse
import sys
from pathlib import Path

from restaurant_demand.config import get_project_root, load_data_config, load_model_config
from restaurant_demand.data.ingestion import DataIngestion
from restaurant_demand.data.validation import DataValidator
from restaurant_demand.exceptions import RestaurantDemandError
from restaurant_demand.logging_config import configure_logger
from restaurant_demand.models.predict import DemandPredictor
from restaurant_demand.models.train import ModelTrainer

logger = configure_logger(__name__)


def validate_command(args: argparse.Namespace) -> int:
    """Validates raw data files against expected schema constraints."""
    logger.info("Executing data validation command...")
    data_config = load_data_config(args.data_config)
    ingestion = DataIngestion(config=data_config)
    validator = DataValidator(config=data_config)

    try:
        merged_df = ingestion.run(save_interim=True)
        report = validator.validate_raw_schema(merged_df)
        print("Data validation SUCCESSFUL.")
        print(f"Total observations: {report.row_count}")
        print(f"Total features/columns: {report.column_count}")
        if report.target_summary:
            print("Target 'num_orders' summary:")
            print(f"  Mean: {report.target_summary['mean']:.2f}")
            print(f"  Median: {report.target_summary['median']:.2f}")
            print(f"  Min: {report.target_summary['min']:.2f}")
            print(f"  Max: {report.target_summary['max']:.2f}")
        return 0
    except RestaurantDemandError as exc:
        logger.error("Validation failed: %s", exc)
        print(f"Data validation FAILED: {exc}", file=sys.stderr)
        return 1


def train_command(args: argparse.Namespace) -> int:
    """Executes the full training, benchmark comparison, and artifact saving pipeline."""
    logger.info("Executing training pipeline command...")
    data_config = load_data_config(args.data_config)
    model_config = load_model_config(args.model_config)

    try:
        trainer = ModelTrainer(data_config=data_config, model_config=model_config)
        metrics = trainer.train_pipeline()
        print("Model training COMPLETED successfully.")
        print(f"Baseline test MAE: {metrics['baseline']['test']['mae']}")
        print(f"Candidate test MAE: {metrics['candidate_hist_gradient_boosting']['test']['mae']}")
        print(f"Relative MAE reduction: +{metrics['relative_improvement_pct']['mae_reduction']}%")
        return 0
    except RestaurantDemandError as exc:
        logger.error("Training failed: %s", exc)
        print(f"Model training FAILED: {exc}", file=sys.stderr)
        return 1


def predict_command(args: argparse.Namespace) -> int:
    """Scores an input CSV file and writes results with predicted_demand."""
    logger.info("Executing batch prediction from file...")
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output) if args.output else input_path.parent / f"scored_{input_path.name}"
    )

    try:
        predictor = DemandPredictor()
        predictor.predict_file(input_csv_path=input_path, output_csv_path=output_path)
        print(f"Scoring COMPLETED. Output written to: {output_path}")
        return 0
    except RestaurantDemandError as exc:
        logger.error("Prediction failed: %s", exc)
        print(f"Prediction FAILED: {exc}", file=sys.stderr)
        return 1


def serve_command(args: argparse.Namespace) -> int:
    """Starts the FastAPI inference service using uvicorn."""
    import uvicorn

    logger.info("Starting FastAPI service on %s:%d...", args.host, args.port)
    uvicorn.run("restaurant_demand.api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def ui_command(args: argparse.Namespace) -> int:
    """Launches the Streamlit interactive dashboard."""
    import subprocess

    root = get_project_root()
    app_path = root / "app.py"
    logger.info("Starting Streamlit web interface on port %d...", args.port)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(args.port)]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="restaurant-demand",
        description="CLI workflows for Restaurant Order Demand Prediction",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # validate
    val_p = subparsers.add_parser("validate", help="Validate raw data integrity and schema")
    val_p.add_argument("--data-config", type=str, default=None, help="Path to custom data.yaml")
    val_p.set_defaults(func=validate_command)

    # train
    train_p = subparsers.add_parser("train", help="Train baseline and candidate models")
    train_p.add_argument("--data-config", type=str, default=None, help="Path to custom data.yaml")
    train_p.add_argument("--model-config", type=str, default=None, help="Path to custom model.yaml")
    train_p.set_defaults(func=train_command)

    # predict
    pred_p = subparsers.add_parser("predict", help="Generate predictions for an input CSV")
    pred_p.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV file")
    pred_p.add_argument("--output", "-o", type=str, default=None, help="Path to output CSV file")
    pred_p.set_defaults(func=predict_command)

    # serve
    serve_p = subparsers.add_parser("serve", help="Launch FastAPI REST service")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    serve_p.add_argument("--port", type=int, default=8000, help="Listening port")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_p.set_defaults(func=serve_command)

    # ui
    ui_p = subparsers.add_parser("ui", help="Launch Streamlit interactive web dashboard")
    ui_p.add_argument("--port", type=int, default=8501, help="Streamlit port")
    ui_p.set_defaults(func=ui_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
