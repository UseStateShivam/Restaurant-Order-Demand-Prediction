"""Model artifact persistence and metadata registry module.

Implements safe, atomic persistence of complete fitted pipelines using Joblib,
and tracks comprehensive machine-readable provenance metadata in JSON format.
"""

import importlib.metadata
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from restaurant_demand.config import get_project_root
from restaurant_demand.exceptions import ModelNotFoundError
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


class ModelRegistry:
    """Manages saving, loading, and versioning of trained model pipelines and metadata."""

    def __init__(self, model_dir: Path | None = None):
        """Initializes ModelRegistry.

        Args:
            model_dir: Optional path to the directory where models are saved.
        """
        self.root = get_project_root()
        self.model_dir = model_dir or (self.root / "models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_installed_versions(self) -> dict[str, str]:
        """Collects exact versions of core ML dependencies for auditability."""
        packages = ["scikit-learn", "pandas", "numpy", "joblib", "pydantic"]
        versions = {}
        for pkg in packages:
            try:
                versions[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                versions[pkg] = "unknown"
        return versions

    def save_artifact(
        self,
        pipeline: Any,
        model_name: str,
        feature_names: list[str],
        metrics: dict[str, float],
        hyperparameters: dict[str, Any],
        split_summary: dict[str, Any],
        model_filename: str = "restaurant_demand_pipeline.joblib",
        metadata_filename: str = "model_metadata.json",
    ) -> tuple[Path, Path]:
        """Atomically saves the trained pipeline and metadata to disk.

        Using atomic file replacement (writing to a temporary file then renaming)
        prevents incomplete or corrupted artifacts if writes are interrupted.

        Args:
            pipeline: Fitted scikit-learn Pipeline or estimator.
            model_name: Name of the algorithm (e.g., 'HistGradientBoostingRegressor').
            feature_names: List of expected input features.
            metrics: Dictionary of test set evaluation metrics.
            hyperparameters: Hyperparameters used during training.
            split_summary: Summary of train/val/test splits.
            model_filename: Name of the joblib model artifact file.
            metadata_filename: Name of the metadata JSON file.

        Returns:
            Tuple of (model_path, metadata_path).
        """
        target_model_path = self.model_dir / model_filename
        target_meta_path = self.model_dir / metadata_filename

        metadata = {
            "model_name": model_name,
            "version": "1.0.0",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "target_variable": "num_orders",
            "target_units": "weekly_orders",
            "feature_names": feature_names,
            "expected_input_schema": {
                "required_columns": [
                    "week",
                    "center_id",
                    "meal_id",
                    "checkout_price",
                    "base_price",
                    "emailer_for_promotion",
                    "homepage_featured",
                ],
                "optional_enrichment_columns": [
                    "city_code",
                    "region_code",
                    "center_type",
                    "op_area",
                    "category",
                    "cuisine",
                ],
            },
            "hyperparameters": hyperparameters,
            "test_metrics": metrics,
            "split_summary": split_summary,
            "environment_packages": self._get_installed_versions(),
            "security_notice": "Load only from trusted internal repository sources.",
        }

        # 1. Atomic write for Joblib model artifact
        temp_dir = self.model_dir / "temp_atomic"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".joblib") as tmp_file:
            temp_model_path = Path(tmp_file.name)

        try:
            joblib.dump(pipeline, temp_model_path, compress=3)
            # Atomic replace
            os.replace(temp_model_path, target_model_path)
            logger.info("Atomically saved model pipeline to %s", target_model_path)
        finally:
            if temp_model_path.exists():
                temp_model_path.unlink()

        # 2. Write metadata JSON
        def _json_default(o):
            if hasattr(o, "item"):
                return o.item()
            return str(o)

        with open(target_meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=_json_default)
        logger.info("Saved model metadata to %s", target_meta_path)

        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()

        return target_model_path, target_meta_path

    def load_artifact(
        self,
        model_filename: str = "restaurant_demand_pipeline.joblib",
        metadata_filename: str = "model_metadata.json",
    ) -> tuple[Any, dict[str, Any]]:
        """Loads and validates a persisted model pipeline and its metadata.

        Args:
            model_filename: Name of the model file.
            metadata_filename: Name of the metadata JSON file.

        Returns:
            Tuple of (pipeline, metadata).

        Raises:
            ModelNotFoundError: If the requested artifact cannot be found.
        """
        model_path = self.model_dir / model_filename
        meta_path = self.model_dir / metadata_filename

        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model artifact not found at {model_path}. "
                "Train the model first via 'python -m restaurant_demand.cli train'."
            )

        if not meta_path.exists():
            logger.warning(
                "Metadata file not found at %s. Loading pipeline without metadata.", meta_path
            )
            metadata = {}
        else:
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        logger.info("Loading model artifact from %s", model_path)
        pipeline = joblib.load(model_path)
        return pipeline, metadata
