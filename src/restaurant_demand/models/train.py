"""Training pipeline orchestration module for Restaurant Order Demand Prediction.

Executes end-to-end data ingestion, validation, chronological splitting,
baseline estimation, candidate tree-based model training, held-out evaluation,
and atomic model artifact persistence.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from restaurant_demand.config import (
    DataConfig,
    ModelSettings,
    get_project_root,
    load_data_config,
    load_model_config,
)
from restaurant_demand.data.ingestion import DataIngestion
from restaurant_demand.data.preprocessing import DataPreprocessor
from restaurant_demand.data.validation import DataValidator
from restaurant_demand.logging_config import configure_logger
from restaurant_demand.models.baseline import BaselineDemandModel
from restaurant_demand.models.evaluate import ModelEvaluator
from restaurant_demand.models.registry import ModelRegistry

logger = configure_logger(__name__)


class ModelTrainer:
    """Orchestrates end-to-end model training, comparative evaluation, and artifact saving."""

    def __init__(
        self,
        data_config: DataConfig | None = None,
        model_config: ModelSettings | None = None,
        base_dir: Path | None = None,
    ):
        """Initializes the trainer with data and modeling configurations.

        Args:
            data_config: Data pipeline configuration.
            model_config: Model hyperparameters and architecture settings.
            base_dir: Optional project root path override.
        """
        self.root = base_dir or get_project_root()
        self.data_config = data_config or load_data_config()
        self.model_config = model_config or load_model_config()

        self.ingestion = DataIngestion(config=self.data_config, base_dir=self.root)
        self.validator = DataValidator(config=self.data_config)
        self.preprocessor = DataPreprocessor(config=self.data_config, base_dir=self.root)
        self.evaluator = ModelEvaluator(figures_dir=self.root / "reports" / "figures")
        self.registry = ModelRegistry(model_dir=self.root / self.model_config.artifacts.model_dir)

    def train_pipeline(self) -> dict[str, Any]:
        """Runs the complete training and evaluation pipeline.

        Returns:
            Dict[str, Any]: Results dictionary containing metrics, artifact paths, and error analyses.

        Raises:
            ModelTrainingError: If training or evaluation encounters unrecoverable errors.
        """
        logger.info("Starting Restaurant Order Demand Prediction Training Pipeline...")

        # 1. Ingest raw tables and perform relational merge
        merged_df = self.ingestion.run(save_interim=True)

        # 2. Validate raw and merged schema constraints
        logger.info("Validating dataset integrity...")
        self.validator.validate_raw_schema(merged_df)

        # 3. Feature engineering and chronological splitting
        logger.info("Preparing features and executing chronological partition...")
        train_df, val_df, test_df = self.preprocessor.prepare_data(merged_df, save_processed=True)

        target_col = self.data_config.target_column
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        # 4. Fit Baseline Heuristic Model
        logger.info("Training Domain Baseline (Center-Meal Group Median)...")
        baseline = BaselineDemandModel(
            group_cols=self.model_config.baseline.group_columns,
            strategy=self.model_config.baseline.strategy,
        )
        baseline.fit(train_df, y_train)

        baseline_val_preds = baseline.predict(val_df)
        baseline_test_preds = baseline.predict(test_df)

        baseline_val_metrics = self.evaluator.compute_metrics(
            y_val, baseline_val_preds, model_name="Baseline (Val)"
        )
        baseline_test_metrics = self.evaluator.compute_metrics(
            y_test, baseline_test_preds, model_name="Baseline (Test)"
        )

        # 5. Build and Fit Candidate Model (HistGradientBoostingRegressor inside Pipeline)
        num_cols, cat_cols, bin_cols = self.preprocessor.get_feature_lists(train_df)
        feature_columns = num_cols + cat_cols + bin_cols

        logger.info(
            "Feature subset for ML: %d numerical, %d categorical, %d binary (total=%d)",
            len(num_cols),
            len(cat_cols),
            len(bin_cols),
            len(feature_columns),
        )

        transformer = self.preprocessor.build_transformer(
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            binary_cols=bin_cols,
        )

        hgb_params = self.model_config.candidate.hyperparameters
        regressor = HistGradientBoostingRegressor(**hgb_params)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("regressor", regressor),
            ]
        )

        logger.info(
            "Fitting HistGradientBoosting candidate pipeline on training partition (%d rows)...",
            len(train_df),
        )
        X_train = train_df[feature_columns]
        pipeline.fit(X_train, y_train)
        logger.info("Pipeline training complete.")

        # 6. Evaluate Candidate Model on Validation and Final Test Horizons
        X_val = val_df[feature_columns]
        X_test = test_df[feature_columns]

        val_preds = pipeline.predict(X_val)
        test_preds = pipeline.predict(X_test)

        # Enforce domain non-negativity constraint
        # Order demand cannot be negative; minimum historical demand is positive
        val_preds = np.maximum(1.0, val_preds)
        test_preds = np.maximum(1.0, test_preds)

        candidate_val_metrics = self.evaluator.compute_metrics(
            y_val, val_preds, model_name="Candidate HGB (Val)"
        )
        candidate_test_metrics = self.evaluator.compute_metrics(
            y_test, test_preds, model_name="Candidate HGB (Test)"
        )

        # 7. Perform Error and Residual Analysis on Held-Out Test Set
        logger.info("Conducting error segmentation and generating diagnostic figures...")
        error_analysis = self.evaluator.analyze_errors(
            y_test, test_preds, df_context=test_df, top_k=10
        )
        saved_plots = self.evaluator.generate_evaluation_plots(
            y_test, test_preds, prefix="candidate_hgb"
        )

        # 8. Compare baseline vs candidate improvement
        mae_improvement = round(
            (baseline_test_metrics["mae"] - candidate_test_metrics["mae"])
            / baseline_test_metrics["mae"]
            * 100.0,
            2,
        )
        rmse_improvement = round(
            (baseline_test_metrics["rmse"] - candidate_test_metrics["rmse"])
            / baseline_test_metrics["rmse"]
            * 100.0,
            2,
        )
        wape_improvement = round(
            (baseline_test_metrics["wape"] - candidate_test_metrics["wape"])
            / baseline_test_metrics["wape"]
            * 100.0,
            2,
        )

        logger.info(
            "Candidate vs Baseline Performance Delta: MAE reduction: +%.2f%% | RMSE reduction: +%.2f%% | WAPE reduction: +%.2f%%",
            mae_improvement,
            rmse_improvement,
            wape_improvement,
        )

        split_summary = {
            "strategy": "chronological",
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_weeks": f"1 to {self.data_config.split.train_week_max}",
            "val_weeks": f"{self.data_config.split.train_week_max + 1} to {self.data_config.split.val_week_max}",
            "test_weeks": f"{self.data_config.split.val_week_max + 1} to {self.data_config.split.test_week_max}",
        }

        # 9. Atomically Persist the Production Pipeline and Metadata
        model_path, meta_path = self.registry.save_artifact(
            pipeline=pipeline,
            model_name="HistGradientBoostingRegressor",
            feature_names=feature_columns,
            metrics=candidate_test_metrics,
            hyperparameters=hgb_params,
            split_summary=split_summary,
            model_filename=self.model_config.artifacts.pipeline_filename,
            metadata_filename=self.model_config.artifacts.metadata_filename,
        )

        # 10. Write Machine-Readable Metrics Report and Markdown Summary
        metrics_report = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "baseline": {
                "val": baseline_val_metrics,
                "test": baseline_test_metrics,
            },
            "candidate_hist_gradient_boosting": {
                "val": candidate_val_metrics,
                "test": candidate_test_metrics,
            },
            "relative_improvement_pct": {
                "mae_reduction": mae_improvement,
                "rmse_reduction": rmse_improvement,
                "wape_reduction": wape_improvement,
            },
            "split_summary": split_summary,
            "error_analysis": error_analysis,
            "saved_plots": saved_plots,
        }

        metrics_file = (
            self.root
            / self.model_config.artifacts.model_dir
            / self.model_config.artifacts.metrics_filename
        )

        def _json_default(o):
            if isinstance(o, (np.integer, np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.floating, np.float64, np.float32)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2, default=_json_default)

        # Write markdown evaluation report in reports/model_evaluation.md
        self._write_markdown_evaluation_report(metrics_report, model_path)

        logger.info("Training pipeline completed successfully.")
        return metrics_report

    def _write_markdown_evaluation_report(self, report: dict[str, Any], model_path: Path) -> None:
        """Writes human-readable markdown report to reports/model_evaluation.md."""
        report_path = self.root / "reports" / "model_evaluation.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        base_t = report["baseline"]["test"]
        cand_t = report["candidate_hist_gradient_boosting"]["test"]
        imp = report["relative_improvement_pct"]
        bias = report["error_analysis"]["bias_summary"]
        tiers = report["error_analysis"]["demand_tiers"]

        md_content = f"""# Model Evaluation and Benchmark Report

## 1. Executive Summary

This report documents the empirical evaluation of the **Restaurant Order Demand Prediction** system.
A tree-based **HistGradientBoostingRegressor** pipeline was benchmarked against a domain **Group Median Baseline**
(center and meal historical volume) using a strict chronological holdout to prevent temporal data leakage.

- **Model Artifact**: `{model_path.name}`
- **Evaluation Horizon**: Weeks 136 to 145 ({report["split_summary"]["test_rows"]} unseen transactions)
- **Target Variable**: `num_orders` (Weekly orders per meal per fulfillment center)

## 2. Test Set Metrics Comparison

| Metric | Group Median Baseline | HistGradientBoosting | Relative Improvement |
|---|---|---|---|
| **MAE** (Mean Absolute Error) | {base_t["mae"]} | {cand_t["mae"]} | +{imp["mae_reduction"]}% |
| **RMSE** (Root Mean Squared Error) | {base_t["rmse"]} | {cand_t["rmse"]} | +{imp["rmse_reduction"]}% |
| **R² Score** | {base_t["r2"]} | {cand_t["r2"]} | Improved |
| **MedAE** (Median Absolute Error) | {base_t["medae"]} | {cand_t["medae"]} | Lower is better |
| **WAPE** (Weighted Absolute % Error) | {base_t["wape"]}% | {cand_t["wape"]}% | +{imp["wape_reduction"]}% |

## 3. Demand Tier Breakdown

Forecasting error characteristics across order volume tiers:

- **Low Demand (< 100 orders)**: Count: {tiers["low_demand_under_100"]["count"]}, MAE: {tiers["low_demand_under_100"]["mae"]}, WAPE: {tiers["low_demand_under_100"]["wape"]}%
- **Medium Demand (100 - 500 orders)**: Count: {tiers["medium_demand_100_to_500"]["count"]}, MAE: {tiers["medium_demand_100_to_500"]["mae"]}, WAPE: {tiers["medium_demand_100_to_500"]["wape"]}%
- **High Demand (> 500 orders)**: Count: {tiers["high_demand_above_500"]["count"]}, MAE: {tiers["high_demand_above_500"]["mae"]}, WAPE: {tiers["high_demand_above_500"]["wape"]}%

## 4. Bias and Residual Characteristics

- **Mean Residual (Actual - Predicted)**: {bias["mean_residual"]} orders
- **Median Residual**: {bias["median_residual"]} orders
- **Underprediction Rate**: {bias["underpredicted_pct"]}% of transactions
- **Overprediction Rate**: {bias["overpredicted_pct"]}% of transactions

## 5. Visualizations

- Actual vs. Predicted Scatter: `reports/figures/candidate_hgb_actual_vs_predicted.png`
- Residual Distribution: `reports/figures/candidate_hgb_residuals_distribution.png`
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Saved evaluation markdown report to %s", report_path)
