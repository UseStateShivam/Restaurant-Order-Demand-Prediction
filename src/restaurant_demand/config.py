"""Configuration management module using Pydantic and PyYAML.

Loads and validates project settings from YAML files with graceful fallbacks
to sensible defaults and environment variable overrides.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from restaurant_demand.exceptions import ConfigurationError
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


class RawDataPaths(BaseModel):
    """Filepaths for raw dataset components."""

    train_path: str = "data/raw/train.csv"
    center_path: str = "data/raw/fulfilment_center_info.csv"
    meal_path: str = "data/raw/meal_info.csv"


class InterimDataPaths(BaseModel):
    """Filepaths for interim merged datasets."""

    merged_path: str = "data/interim/merged_data.csv"


class ProcessedDataPaths(BaseModel):
    """Filepaths for split processed datasets."""

    train_path: str = "data/processed/train.csv"
    val_path: str = "data/processed/val.csv"
    test_path: str = "data/processed/test.csv"


class SplitConfig(BaseModel):
    """Chronological split bounds."""

    strategy: str = "chronological"
    train_week_max: int = 125
    val_week_max: int = 135
    test_week_max: int = 145


class FeatureConfig(BaseModel):
    """Input feature group specifications."""

    numerical: list[str] = Field(
        default_factory=lambda: [
            "checkout_price",
            "base_price",
            "discount_amount",
            "discount_ratio",
            "op_area",
        ]
    )
    categorical: list[str] = Field(
        default_factory=lambda: [
            "center_type",
            "category",
            "cuisine",
            "city_code",
            "region_code",
        ]
    )
    binary: list[str] = Field(
        default_factory=lambda: [
            "emailer_for_promotion",
            "homepage_featured",
        ]
    )
    identifiers: list[str] = Field(
        default_factory=lambda: [
            "center_id",
            "meal_id",
        ]
    )


class DataConfig(BaseModel):
    """Global data configuration container."""

    raw: RawDataPaths = Field(default_factory=RawDataPaths)
    interim: InterimDataPaths = Field(default_factory=InterimDataPaths)
    processed: ProcessedDataPaths = Field(default_factory=ProcessedDataPaths)
    target_column: str = "num_orders"
    id_column: str = "id"
    temporal_column: str = "week"
    split: SplitConfig = Field(default_factory=SplitConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)


class BaselineConfig(BaseModel):
    """Configuration for benchmark baseline model."""

    strategy: str = "group_median"
    fallback_strategy: str = "global_median"
    group_columns: list[str] = Field(default_factory=lambda: ["center_id", "meal_id"])


class CandidateConfig(BaseModel):
    """Configuration for primary candidate model (HistGradientBoosting)."""

    algorithm: str = "hist_gradient_boosting"
    hyperparameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_iter": 200,
            "learning_rate": 0.08,
            "max_depth": 12,
            "min_samples_leaf": 25,
            "l2_regularization": 0.5,
            "random_state": 42,
        }
    )


class ComparisonConfig(BaseModel):
    """Configuration for secondary comparative model (RandomForest)."""

    algorithm: str = "random_forest"
    hyperparameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "n_estimators": 100,
            "max_depth": 14,
            "min_samples_leaf": 20,
            "random_state": 42,
            "n_jobs": -1,
        }
    )


class ArtifactsConfig(BaseModel):
    """Paths for saved model artifacts and evaluation metrics."""

    model_dir: str = "models"
    pipeline_filename: str = "restaurant_demand_pipeline.joblib"
    metadata_filename: str = "model_metadata.json"
    metrics_filename: str = "metrics_report.json"


class ModelSettings(BaseModel):
    """Global model configuration container."""

    target_name: str = "num_orders"
    target_units: str = "orders_per_week"
    random_seed: int = 42
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    candidate: CandidateConfig = Field(default_factory=CandidateConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    evaluation_metrics: list[str] = Field(
        default_factory=lambda: ["mae", "rmse", "r2", "wape", "medae"]
    )
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # Assuming config.py is located at src/restaurant_demand/config.py
    return Path(__file__).resolve().parent.parent.parent


def load_yaml(filepath: Path) -> dict[str, Any]:
    """Reads and parses a YAML configuration file safely.

    Args:
        filepath: Absolute or relative Path to the YAML file.

    Returns:
        dict: Parsed dictionary from YAML contents.

    Raises:
        ConfigurationError: If the file does not exist or has malformed syntax.
    """
    if not filepath.exists():
        raise ConfigurationError(f"Configuration file not found at: {filepath}")

    try:
        with open(filepath, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML file at {filepath}: {exc}") from exc


def load_data_config(config_path: str | None = None) -> DataConfig:
    """Loads and validates data configuration.

    Args:
        config_path: Optional path to data.yaml. If None, uses default path.

    Returns:
        DataConfig: Validated Pydantic DataConfig instance.
    """
    root = get_project_root()
    path = Path(config_path) if config_path else root / "configs" / "data.yaml"

    if path.exists():
        raw_dict = load_yaml(path)
        data_section = raw_dict.get("data", raw_dict)
        return DataConfig.model_validate(data_section)

    logger.warning("data.yaml not found at %s. Using default configuration.", path)
    return DataConfig()


def load_model_config(config_path: str | None = None) -> ModelSettings:
    """Loads and validates model configuration.

    Args:
        config_path: Optional path to model.yaml. If None, uses default path.

    Returns:
        ModelSettings: Validated Pydantic ModelSettings instance.
    """
    root = get_project_root()
    path = Path(config_path) if config_path else root / "configs" / "model.yaml"

    if path.exists():
        raw_dict = load_yaml(path)
        model_section = raw_dict.get("model", raw_dict)
        return ModelSettings.model_validate(model_section)

    logger.warning("model.yaml not found at %s. Using default configuration.", path)
    return ModelSettings()
