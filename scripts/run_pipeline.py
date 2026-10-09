"""End-to-end execution script for the Restaurant Order Demand Prediction pipeline.

Runs data ingestion, schema validation, feature engineering, chronological partitioning,
baseline and candidate model training, held-out evaluation, and test scoring.
"""

import sys

from restaurant_demand.config import load_data_config, load_model_config
from restaurant_demand.logging_config import configure_logger
from restaurant_demand.models.predict import DemandPredictor
from restaurant_demand.models.train import ModelTrainer

logger = configure_logger(__name__)


def main() -> int:
    """Executes the complete pipeline."""
    logger.info("Executing end-to-end Restaurant Demand ML Pipeline...")
    data_config = load_data_config()
    model_config = load_model_config()

    try:
        # 1. Run trainer
        trainer = ModelTrainer(data_config=data_config, model_config=model_config)
        metrics = trainer.train_pipeline()

        # 2. Verify inference predictor
        logger.info("Verifying model scoring on sample transaction...")
        predictor = DemandPredictor()
        sample_input = {
            "week": 140,
            "center_id": 55,
            "meal_id": 1885,
            "checkout_price": 136.83,
            "base_price": 152.29,
            "emailer_for_promotion": 0,
            "homepage_featured": 0,
        }
        prediction = predictor.predict_single(sample_input)
        logger.info(
            "Sample Prediction Result: Center %d, Meal %d, Week %d -> Predicted Demand: %.1f %s",
            prediction["center_id"],
            prediction["meal_id"],
            prediction["week"],
            prediction["predicted_demand"],
            prediction["target_units"],
        )

        print("\n=== PIPELINE RUN SUMMARY ===")
        print(f"Candidate Model: {prediction['model_name']} (v{prediction['model_version']})")
        print(f"Test MAE: {metrics['candidate_hist_gradient_boosting']['test']['mae']}")
        print(f"Test RMSE: {metrics['candidate_hist_gradient_boosting']['test']['rmse']}")
        print(f"Test R2: {metrics['candidate_hist_gradient_boosting']['test']['r2']}")
        print(f"Test WAPE: {metrics['candidate_hist_gradient_boosting']['test']['wape']}%")
        print(
            f"Relative MAE improvement over baseline: +{metrics['relative_improvement_pct']['mae_reduction']}%\n"
        )
        return 0

    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        print(f"Pipeline execution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
