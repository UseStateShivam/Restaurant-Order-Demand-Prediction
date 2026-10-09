"""Custom domain-specific exceptions for the Restaurant Order Demand Prediction system.

Defining structured exception hierarchies enables precise failure diagnosis across
boundaries (ingestion, validation, preprocessing, training, inference, and API).
"""


class RestaurantDemandError(Exception):
    """Base exception for all domain-specific errors in the system."""

    pass


class ConfigurationError(RestaurantDemandError):
    """Raised when application or model configuration is missing or malformed."""

    pass


class DataIngestionError(RestaurantDemandError):
    """Raised when reading or downloading raw datasets fails."""

    pass


class DataValidationError(RestaurantDemandError):
    """Raised when an incoming dataset violates required schemas or integrity constraints."""

    pass


class PreprocessingError(RestaurantDemandError):
    """Raised during data cleaning, transformation, or alignment failures."""

    pass


class FeatureEngineeringError(RestaurantDemandError):
    """Raised when feature generation fails or detects invalid values."""

    pass


class ModelTrainingError(RestaurantDemandError):
    """Raised when model fitting encounters numerical or structural errors."""

    pass


class ModelEvaluationError(RestaurantDemandError):
    """Raised when calculating evaluation metrics or generating reports fails."""

    pass


class ModelNotFoundError(RestaurantDemandError):
    """Raised when attempting to load a non-existent or inaccessible model artifact."""

    pass


class ModelInferenceError(RestaurantDemandError):
    """Raised when scoring input observations fails during inference."""

    pass
