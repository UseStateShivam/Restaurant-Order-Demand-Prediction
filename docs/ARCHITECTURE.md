# System Architecture and Design Document

## 1. System Overview

The **Restaurant Order Demand Prediction System** is an end-to-end machine learning platform designed to forecast weekly order demand (`num_orders`) for specific meal items across a network of fulfillment centers. 

The architecture strictly adheres to high-cohesion, low-coupling design principles, separating offline training pipelines from online, low-latency inference services.

```mermaid
flowchart TD
    subgraph Data Layer
        A1[train.csv\nWeekly Transactions] --> M[Data Ingestion & Referential Merge]
        A2[fulfilment_center_info.csv\nCenter Logistics] --> M
        A3[meal_info.csv\nMeal Catalog] --> M
        M --> V[Schema & Data Validation]
    end

    subgraph Feature & Partitioning Layer
        V --> FE[Feature Engineering\nPrices, Promotions, Fourier Seasonality]
        FE --> CS[Chronological Partitioning\nTrain: W1-125 | Val: W126-135 | Test: W136-145]
    end

    subgraph Modeling & Benchmark Layer
        CS --> B[Domain Baseline Estimator\nCenter-Meal Group Median]
        CS --> CT[ColumnTransformer Pipeline\nImputation + Scaling + OneHot]
        CT --> HGB[HistGradientBoostingRegressor]
    end

    subgraph Evaluation & Artifacts
        B --> EV[Model Evaluator & Benchmark Comparison]
        HGB --> EV
        EV --> MR[Model Registry\nAtomic Joblib + JSON Metadata]
    end

    subgraph Serving & Applications
        MR --> DP[Demand Predictor Engine]
        DP --> API[FastAPI Microservice\n/predict & /predict/batch]
        DP --> UI[Streamlit Interactive Dashboard]
        DP --> CLI[Command-Line Interface]
    end
```

## 2. Directory Structure and Boundaries

```text
restaurant-order-demand-prediction/
├── configs/                  # Declarative configurations (data.yaml, model.yaml)
├── data/
│   ├── raw/                  # Immutable primary CSV files
│   ├── interim/              # Relational joined data (merged_data.csv)
│   ├── processed/            # Cleaned chronological partitions (train, val, test)
│   └── README.md             # Dataset documentation
├── docker/                   # Dockerfile and container build artifacts
├── docs/                     # Architectural and operational documentation
│   ├── ARCHITECTURE.md
│   ├── DATA_DICTIONARY.md
│   ├── EXPERIMENTS.md
│   ├── INTERVIEW_GUIDE.md
│   └── MODEL_CARD.md
├── models/                   # Persisted serialized pipelines and metadata
│   ├── restaurant_demand_pipeline.joblib
│   ├── model_metadata.json
│   └── metrics_report.json
├── notebooks/                # Exploratory analysis and model exploration
├── reports/
│   ├── figures/              # Diagnostic scatter and residual plots
│   └── model_evaluation.md   # Benchmark performance report
├── scripts/                  # Standalone execution scripts
│   ├── download_data.py
│   └── run_pipeline.py
├── src/                      # Core installable Python package
│   └── restaurant_demand/
│       ├── api/              # FastAPI application and Pydantic schemas
│       ├── data/             # Ingestion, validation, and preprocessing
│       ├── features/         # Feature transformations and catalog lookups
│       ├── models/           # Baseline, training, evaluation, predictor
│       ├── monitoring/       # Statistical drift detection (KS-test, PSI)
│       ├── cli.py            # Command-line interface entry points
│       ├── config.py         # Pydantic configuration loader
│       ├── exceptions.py     # Domain-specific exception hierarchy
│       └── logging_config.py # Structured logging setup
├── tests/                    # Pytest unit and integration test suite
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── app.py                    # Streamlit interactive web application
├── Makefile                  # Developer workflow automation
├── pyproject.toml            # Package build and dependency metadata
└── requirements.txt          # Pinned dependency requirements
```

## 3. Component Boundaries and Responsibilities

### Data Ingestion (`data/ingestion.py`)
- Responsible for fetching raw CSVs from verified repository endpoints if missing locally.
- Executes referential left joins across `center_id` and `meal_id`.
- Enforces referential integrity: ensures every transactional record corresponds to known center and meal dimensions.

### Data Validation (`data/validation.py`)
- Serves as the system boundary firewall.
- Validates structural requirements (column names, types, missing values).
- Enforces domain business rules:
  - Non-negative order counts (`num_orders > 0`).
  - Positive pricing (`checkout_price > 0`, `base_price > 0`).
  - Binary indicator fidelity (`emailer_for_promotion`, `homepage_featured` in `{0, 1}`).

### Preprocessing and Partitioning (`data/preprocessing.py`)
- Partitions datasets chronologically by `week` number.
- Fits scikit-learn `ColumnTransformer` exclusively on training data to prevent future leakage into validation or test partitions.
- Encapsulates OneHotEncoder with `handle_unknown="ignore"` to safeguard real-time inference against unseen categorical levels.

### Feature Engineering (`features/build_features.py`)
- Derives economic and temporal signals:
  - `discount_amount`: `base_price - checkout_price`.
  - `discount_ratio`: `(base_price - checkout_price) / base_price`.
  - `price_ratio`: `checkout_price / base_price`.
  - Promotional interaction: `has_any_promotion` and `dual_promotion`.
  - Cyclical seasonality: Fourier sine/cosine transformations on 52-week annual cycles.
- Catalogs fallback lookup: automatically enriches inference requests containing only transaction IDs with static attributes.

### Model Training and Registry (`models/train.py`, `models/registry.py`)
- Trains a transparent historical central-tendency benchmark (`BaselineDemandModel`).
- Fits candidate tree-based gradient boosted regressor (`HistGradientBoostingRegressor`).
- Persists pipelines using atomic file replacement (`os.replace`) to eliminate risk of partially written corrupted weights.
- Emits accompanying machine-readable JSON metadata tracking package dependencies, timestamps, and evaluation metrics.

### Serving Layer (`models/predict.py`, `api/app.py`, `app.py`)
- Lazy-loads fitted pipeline once into memory at application startup.
- Validates request payloads through typed Pydantic models.
- Enforces physical non-negativity constraint on predictions (`max(1.0, prediction)`).
- Provides OpenAPI endpoints (`/predict`, `/predict/batch`, `/model-info`, `/health`) and interactive Streamlit UI.
