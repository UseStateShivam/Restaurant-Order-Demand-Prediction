"""Unit tests for baseline model, metrics, and artifact registry."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from restaurant_demand.models.baseline import BaselineDemandModel
from restaurant_demand.models.evaluate import ModelEvaluator
from restaurant_demand.models.registry import ModelRegistry


def test_baseline_demand_model_fit_predict(sample_merged_df):
    """Verifies that BaselineDemandModel groups historical orders correctly."""
    X = sample_merged_df[["center_id", "meal_id"]].copy()
    y = sample_merged_df["num_orders"].values

    baseline = BaselineDemandModel(group_cols=["center_id", "meal_id"], strategy="median")
    baseline.fit(X, y)

    preds = baseline.predict(X)
    assert len(preds) == len(X)
    assert (preds > 0).all()


def test_baseline_demand_model_fallback(sample_merged_df):
    """Verifies that unseen center/meal combinations use the global median fallback."""
    X = sample_merged_df[["center_id", "meal_id"]].copy()
    y = sample_merged_df["num_orders"].values

    baseline = BaselineDemandModel(group_cols=["center_id", "meal_id"], strategy="median")
    baseline.fit(X, y)

    # Completely unseen center and meal
    unseen_df = pd.DataFrame([{"center_id": 99999, "meal_id": 99999}])
    pred_unseen = baseline.predict(unseen_df)

    assert len(pred_unseen) == 1
    assert pred_unseen[0] == baseline.global_estimate_


def test_model_evaluator_metrics():
    """Verifies metric calculations against known inputs."""
    evaluator = ModelEvaluator()
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])

    metrics = evaluator.compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 10.0
    assert metrics["medae"] == 10.0
    assert metrics["wape"] == round(30.0 / 600.0 * 100.0, 2)
    assert metrics["r2"] > 0.9


def test_model_registry_save_load(tmp_path: Path):
    """Verifies atomic model saving and compatibility metadata loading."""
    registry = ModelRegistry(model_dir=tmp_path)
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(np.array([[1], [2]]), np.array([10, 20]))

    model_p, meta_p = registry.save_artifact(
        pipeline=dummy_model,
        model_name="DummyRegressor",
        feature_names=["feature_1"],
        metrics={"mae": 5.0},
        hyperparameters={"strategy": "mean"},
        split_summary={"train_rows": 2},
    )

    assert model_p.exists()
    assert meta_p.exists()

    loaded_model, loaded_meta = registry.load_artifact()
    assert loaded_meta["model_name"] == "DummyRegressor"
    assert loaded_meta["test_metrics"]["mae"] == 5.0

    test_pred = loaded_model.predict(np.array([[3]]))
    assert test_pred[0] == 15.0
