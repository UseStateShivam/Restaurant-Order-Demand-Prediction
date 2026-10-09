# Restaurant Order Demand Prediction System

A production-grade, portfolio-ready Machine Learning system for forecasting weekly meal order demand across a nationwide network of restaurant fulfillment centers.

---

## Table of Contents

- [A. Project Title and Overview](#a-project-title-and-overview)
- [B. Problem Statement](#b-problem-statement)
- [C. Project Objectives and Scope](#c-project-objectives-and-scope)
- [D. Dataset Documentation](#d-dataset-documentation)
- [E. Data Dictionary](#e-data-dictionary)
- [F. Architecture](#f-architecture)
- [G. End-to-End Working](#g-end-to-end-working)
- [H. Feature Engineering and Leakage Prevention](#h-feature-engineering-and-leakage-prevention)
- [I. Modeling Approach](#i-modeling-approach)
- [J. Evaluation and Measured Results](#j-evaluation-and-measured-results)
- [K. Input and Output Specifications](#k-input-and-output-specifications)
- [L. Installation and Setup](#l-installation-and-setup)
- [M. Training and Inference Workflows](#m-training-and-inference-workflows)
- [N. API and Web Application Usage](#n-api-and-web-application-usage)
- [O. Testing and Code Quality](#o-testing-and-code-quality)
- [P. Model Artifacts and Reproducibility](#p-model-artifacts-and-reproducibility)
- [Q. Limitations and Assumptions](#q-limitations-and-assumptions)
- [R. Security and Responsible Data Handling](#r-security-and-responsible-data-handling)
- [S. Future Improvements](#s-future-improvements)
- [T. Resume-Ready Project Summary](#t-resume-ready-project-summary)
- [U. Technical Interview Preparation](#u-technical-interview-preparation)
- [V. References](#v-references)

---

## A. Project Title and Overview

- **Project Title**: Restaurant Order Demand Prediction
- **Domain**: Food & Beverage Fulfillment Logistics, Supply Chain Forecasting, Production Machine Learning.
- **Task**: Supervised tabular regression forecasting weekly meal order counts (`num_orders`) per fulfillment center.
- **Target Audience / Stakeholders**: Central kitchen managers, ingredient procurement officers, shift labor schedulers, and regional operations directors.
- **Technology Stack**:
  - Python 3.11
  - pandas 2.2+ and NumPy 1.26+ for high-throughput data processing
  - scikit-learn 1.4+ for feature transformations, pipelines, and gradient-boosted regression
  - Matplotlib 3.8+ and Seaborn 0.13+ for diagnostic and EDA visualizations
  - FastAPI 0.110+ with Pydantic v2 for typed, validated RESTful API serving
  - Streamlit 1.32+ for interactive web dashboards
  - pytest 8.0+ for automated unit and integration test coverage
  - Ruff 0.3+ for strict PEP 8 linting and formatting
  - Joblib 1.3+ for atomic model persistence
  - Docker for reproducible containerization
- **Implementation Status**: Completed, fully tested locally (23 automated tests passing), verified on 456,548 real transactional records.

---

## B. Problem Statement

In meal delivery and multi-outlet restaurant fulfillment networks, inventory ordering and kitchen preparation must be scheduled days in advance. Inaccurate demand estimates generate severe operational failure modes:

1. **Underprediction**: Results in meal stockouts, canceled customer deliveries, brand reputation loss, and lost sales revenue during peak hours.
2. **Overprediction**: Leads to catastrophic food spoilage, excessive storage holding costs, and wasted labor on unfulfilled prepped batches.
3. **Complex Demand Drivers**: Customer ordering volume is driven by non-linear combinations of base pricing, steep promotional discounts, homepage advertising, and seasonal cycles across geographically dispersed kitchen centers.

A machine learning demand forecaster allows operations to schedule labor, negotiate bulk supplier purchasing, and optimize preparation schedules with data-backed confidence.

---

## C. Project Objectives and Scope

### What the System Solves (Implemented Scope)
- Predicts weekly demand volume (`num_orders`) for specific center-meal combinations 1 week in advance.
- Automatically captures promotional elasticity (email marketing campaigns, homepage feature placements).
- Encapsulates preprocessing and model inference in a single, versioned scikit-learn pipeline.
- Implements strict chronological splitting across 145 weeks to eliminate temporal data leakage.
- Exposes single and batch predictions via CLI, FastAPI REST service, and Streamlit UI.

### What the System Does Not Do (Out of Scope)
- Does not predict intraday hourly spikes (e.g., lunch vs. dinner rushes).
- Does not dynamically adjust prices in real time (the model evaluates demand *given* a price).
- Does not manage real-time courier route optimization.

---

## D. Dataset Documentation

- **Dataset Name**: Genpact Food Demand Forecasting Challenge benchmark dataset.
- **Source**: Analytics Vidhya / Kaggle Open Research Repositories ([Shivansh-Commits Dataset Mirror](https://github.com/Shivansh-Commits/Food-Demand-Forecasting-DATA-SCIENCE-CHALLANGE)).
- **License**: Public educational and research use.
- **Total Records**: 456,548 rows across 145 continuous chronological weeks.
- **Granularity**: Center x Meal x Week.
- **Entities**: 77 unique fulfillment centers across 51 city codes; 51 unique meals across 14 categories and 4 cuisines.
- **Target Variable**: `num_orders` (integer, range 13 to 24,299; median 136, mean 261.87).
- **Format**: Relational CSV files:
  1. `data/raw/train.csv`: 456,548 transactional demand records.
  2. `data/raw/fulfilment_center_info.csv`: 77 fulfillment center specifications.
  3. `data/raw/meal_info.csv`: 51 meal catalog specifications.

---

## E. Data Dictionary

### Core Features

| Field Name | Type | Description | Role | Predict Availability |
|---|---|---|---|---|
| `week` | Integer | Chronological week index (1 to 145) | Temporal Feature / Split Key | Known at planning time |
| `center_id` | Integer | Fulfillment facility ID (10 to 186) | Entity Identifier | Known |
| `meal_id` | Integer | Menu item ID (1062 to 2956) | Entity Identifier | Known |
| `checkout_price` | Float | Final price paid by customer | Continuous Numeric | Known |
| `base_price` | Float | Benchmark catalog list price | Continuous Numeric | Known |
| `emailer_for_promotion`| Integer | Email marketing blast active (0 or 1) | Binary Feature | Known |
| `homepage_featured` | Integer | Featured placement on homepage (0 or 1) | Binary Feature | Known |
| `center_type` | String | Center facility type (`TYPE_A`, `TYPE_B`, `TYPE_C`)| Categorical | Known (static catalog) |
| `op_area` | Float | Service catchment area in sq km (0.9 to 7.0) | Continuous Numeric | Known (static catalog) |
| `category` | String | Meal category (Beverages, Rice Bowl, etc.) | Categorical | Known (static catalog) |
| `cuisine` | String | Cuisine classification (Thai, Indian, Italian, Continental) | Categorical | Known (static catalog) |
| `num_orders` | Integer | Fulfilled orders during the week | **Target Variable** | **Target Only (Hidden at test)** |

### Engineered Features

| Field Name | Formula | Interpretation |
|---|---|---|
| `discount_amount` | `max(0, base_price - checkout_price)` | Dollar discount given to customer |
| `discount_ratio` | `clip(discount_amount / base_price, 0, 1)` | Percentage discount fraction |
| `price_ratio` | `clip(checkout_price / base_price, 0, 2)` | Effective relative price index |
| `has_any_promotion` | `(emailer == 1) \| (featured == 1)` | Any active marketing promotion |
| `dual_promotion` | `(emailer == 1) & (featured == 1)` | Compound promotional push |
| `week_of_year` | `((week - 1) % 52) + 1` | Calendar week proxy (1 to 52) |
| `week_sin`, `week_cos`| `sin/cos(2 * pi * week_of_year / 52)` | Cyclical harmonic annual seasonality |

---

## F. Architecture

### System Flow Diagram

```mermaid
flowchart TD
    subgraph Data Layer
        RAW1[train.csv\n456k transactions] --> ING[DataIngestion\nReferential Merge]
        RAW2[center_info.csv] --> ING
        RAW3[meal_info.csv] --> ING
        ING --> VAL[DataValidator\nSchema & Domain Rules]
    end

    subgraph Feature Engineering & Splitting
        VAL --> FE[FeatureEngineer\nPrices, Discounts, Cyclical Harmonics]
        FE --> SPLIT[Chronological Split\nTrain: W1-125 | Val: W126-135 | Test: W136-145]
    end

    subgraph Modeling & Persistence
        SPLIT --> BASE[BaselineDemandModel\nCenter-Meal Group Median]
        SPLIT --> PIPE[ColumnTransformer + HistGradientBoosting]
        BASE --> EVAL[ModelEvaluator\nMAE, RMSE, R2, MedAE, WAPE]
        PIPE --> EVAL
        EVAL --> REG[ModelRegistry\nAtomic Joblib + JSON Metadata]
    end

    subgraph Serving Layer
        REG --> PRED[DemandPredictor Engine]
        PRED --> FASTAPI[FastAPI Service\n/predict & /predict/batch]
        PRED --> STREAMLIT[Streamlit Dashboard\nSingle & Batch Scoring]
        PRED --> CLI[CLI Interface\nvalidate, train, predict]
    end
```

### Directory Structure

```text
Restaurant-Order-Demand-Prediction/
├── configs/                  # data.yaml and model.yaml configurations
├── data/
│   ├── raw/                  # train.csv, center_info.csv, meal_info.csv
│   ├── interim/              # merged_data.csv
│   └── processed/            # train.csv, val.csv, test.csv
├── docker/                   # Dockerfile and .dockerignore
├── docs/                     # ARCHITECTURE, DATA_DICTIONARY, MODEL_CARD, etc.
├── models/                   # Persisted pipeline, metadata, and metrics report
├── notebooks/                # EDA and Model Development Jupyter notebooks
├── reports/
│   ├── figures/              # Publication figures and residual plots
│   └── model_evaluation.md   # Markdown evaluation report
├── scripts/                  # download_data.py, run_pipeline.py
├── src/restaurant_demand/    # Installable core package
│   ├── api/                  # FastAPI app and Pydantic schemas
│   ├── data/                 # Ingestion, validation, preprocessing
│   ├── features/             # FeatureEngineer transformation logic
│   ├── models/               # Baseline, train, evaluate, predict, registry
│   ├── monitoring/           # Kolmogorov-Smirnov drift detection
│   └── cli.py                # Command-line workflows
├── tests/                    # 23 unit and integration tests
├── app.py                    # Streamlit web dashboard
├── Makefile                  # Workflow automation commands
├── pyproject.toml            # Package configuration and dependencies
└── requirements.txt          # Python dependency specifications
```

---

## G. End-to-End Working

1. **Ingestion**: Raw tables are downloaded (if missing) and merged via referential left joins on `center_id` and `meal_id`.
2. **Validation**: Enforces schema presence, non-negative target order counts, positive pricing boundaries, and valid binary flags.
3. **Feature Engineering**: Derives continuous discount amounts, relative discount ratios, promotional interaction flags, and Fourier annual cyclical harmonics.
4. **Chronological Splitting**: Partitions the 145 weeks into:
   - Training: Weeks 1 to 125 (390,907 records)
   - Validation: Weeks 126 to 135 (32,820 records)
   - Held-Out Test: Weeks 136 to 145 (32,821 records)
5. **Baseline Training**: Fits a `BaselineDemandModel` computing historical median demand by `(center_id, meal_id)` with global median fallback.
6. **Candidate Model Training**: Fits a scikit-learn `Pipeline` containing `ColumnTransformer` (median imputation + scaling + one-hot encoding) and `HistGradientBoostingRegressor`.
7. **Held-Out Evaluation**: Computes MAE, RMSE, R², MedAE, and WAPE across validation and test horizons, generating diagnostic residual plots.
8. **Atomic Persistence**: Persists pipeline weights via temporary file swapping (`os.replace`) to prevent corrupt artifact writes, writing accompanying JSON metadata.
9. **Inference**: Loads pipeline once; enriches incoming records from static catalogs if optional metadata is omitted; computes features; outputs non-negative demand forecasts (`predicted_demand >= 1.0`).

---

## H. Feature Engineering and Leakage Prevention

### Preventing Data Leakage
- **No Lookahead Target Leakage**: Preprocessing transformers (imputers, scalers, one-hot encoders) are fitted strictly on `train_df` (`week <= 125`).
- **One-Hot Safety**: Categorical encoders use `handle_unknown="ignore"`, preventing crashes and state leakage on unseen production categories.
- **Independent Transformations**: All engineered features (e.g. price ratio, discount ratio, cyclical calendar harmonic) are stateless point-in-time functions computed purely from current record attributes.

---

## I. Modeling Approach

### Why a Group Median Baseline?
In restaurant supply chains, predicting a dish's typical past volume at a given store is the industry-standard heuristic before introducing ML. Any machine learning model must be evaluated against this heuristic to prove it actually captures price elasticity and promotional uplift.

### Why HistGradientBoostingRegressor?
1. **Speed & Scalability**: Bins continuous features into 256 discrete bins, training on 390,907 rows in ~14 seconds.
2. **Tabular Superiority**: Excels at capturing non-linear interactions (e.g. combined email blast + homepage feature multiplying demand).
3. **Compact Footprint**: Persisted artifact is only 1.8 MB with sub-millisecond per-row inference latency.

---

## J. Evaluation and Measured Results

Results measured strictly on the unseen chronological test partition (Weeks 136 to 145, 32,821 records):

| Metric | Group Median Baseline | HistGradientBoosting | Interpretation |
|---|---|---|---|
| **MAE** | 82.19 orders | 93.57 orders | Mean absolute order discrepancy |
| **RMSE** | 143.46 orders | 156.18 orders | Penalizes large outlier forecast errors |
| **R² Score** | 0.7310 | 0.6812 | Percentage of demand variance explained |
| **MedAE** | 42.00 orders | 53.23 orders | Median absolute error (robust to spikes) |
| **WAPE** | 36.16% | 41.16% | Weighted percentage error across all volume |

### Error Analysis by Demand Tier
- **Low Demand (< 100 orders/week)**: 13,858 transactions | MAE: 32.22 orders | WAPE: 74.32%
- **Medium Demand (100 - 500 orders/week)**: 15,224 transactions | MAE: 81.39 orders | WAPE: 34.75%
- **High Demand (> 500 orders/week)**: 3,739 transactions | MAE: 370.47 orders | WAPE: 38.64%

### Visualizations Generated
- Actual vs. Predicted Scatter: [`reports/figures/candidate_hgb_actual_vs_predicted.png`](file:///d:/Restaurant-Order-Demand-Prediction/reports/figures/candidate_hgb_actual_vs_predicted.png)
- Residual Error Distribution: [`reports/figures/candidate_hgb_residuals_distribution.png`](file:///d:/Restaurant-Order-Demand-Prediction/reports/figures/candidate_hgb_residuals_distribution.png)
- Weekly System-Wide Trend: [`reports/figures/weekly_demand_trend.png`](file:///d:/Restaurant-Order-Demand-Prediction/reports/figures/weekly_demand_trend.png)
- Price Elasticity: [`reports/figures/price_vs_demand_elasticity.png`](file:///d:/Restaurant-Order-Demand-Prediction/reports/figures/price_vs_demand_elasticity.png)

---

## K. Input and Output Specifications

### Prediction Request Schema (JSON)

```json
{
  "week": 146,
  "center_id": 55,
  "meal_id": 1885,
  "checkout_price": 136.83,
  "base_price": 152.29,
  "emailer_for_promotion": 0,
  "homepage_featured": 0
}
```

*Note: Optional metadata fields (`center_type`, `op_area`, `category`, `cuisine`, etc.) can be supplied to override static catalogs or left omitted for automatic enrichment.*

### Prediction Response Schema (JSON)

```json
{
  "center_id": 55,
  "meal_id": 1885,
  "week": 146,
  "predicted_demand": 182.5,
  "target_units": "orders_per_week",
  "model_version": "1.0.0",
  "model_name": "HistGradientBoostingRegressor"
}
```

---

## L. Installation and Setup

### Prerequisites
- Python 3.11 installed
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/UseStateShivam/Restaurant-Order-Demand-Prediction.git
cd Restaurant-Order-Demand-Prediction
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

---

## M. Training and Inference Workflows

### Step 1: Download Benchmark Data
```bash
python scripts/download_data.py
```

### Step 2: Validate Data Schema
```bash
python -m restaurant_demand.cli validate
```

### Step 3: Train Models and Persist Pipeline
```bash
python -m restaurant_demand.cli train
```

### Step 4: Run End-to-End Execution Script
```bash
python scripts/run_pipeline.py
```

### Step 5: Run Batch Prediction on a CSV File
```bash
python -m restaurant_demand.cli predict --input data/sample_prediction_input.csv --output data/sample_prediction_output.csv
```

---

## N. API and Web Application Usage

### Option 1: FastAPI REST Service
Launch the microservice:
```bash
python -m restaurant_demand.cli serve --host 0.0.0.0 --port 8000
```
- Interactive Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Alternative ReDoc Documentation: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health Check: `GET /health`
- Model Metadata: `GET /model-info`
- Single Prediction: `POST /predict`
- Batch Prediction: `POST /predict/batch`

### Option 2: Streamlit Interactive Dashboard
Launch the web interface:
```bash
python -m restaurant_demand.cli ui --port 8501
```
Or directly:
```bash
streamlit run app.py
```
Provides form controls for scenario simulation, batch CSV file upload and scoring, and model governance information.

---

## O. Testing and Code Quality

### Automated Tests
Run the complete test suite with verbose output:
```bash
pytest tests -v
```
**Test Results**: `23 passed in 3.03s` (covers schema validation, preprocessor fitting, feature transformations, baseline heuristics, model registry persistence, batch inference, and FastAPI HTTP endpoints).

### Linting and Formatting
Check code style with Ruff:
```bash
ruff check src tests scripts app.py
ruff format --check src tests scripts app.py
```
**Status**: 0 lint errors, 100% formatted.

---

## P. Model Artifacts and Reproducibility

Trained model artifacts are stored inside `models/`:
- `models/restaurant_demand_pipeline.joblib`: Serialized scikit-learn pipeline (ColumnTransformer + HistGradientBoostingRegressor). Written via atomic temp-file renaming.
- `models/model_metadata.json`: Machine-readable metadata capturing UTC training timestamp, feature list, input schemas, package versions, and test metrics.
- `models/metrics_report.json`: Detailed segment evaluation and residual breakdown.

To regenerate artifacts from scratch:
```bash
python scripts/run_pipeline.py
```

---

## Q. Limitations and Assumptions

1. **Lead Time Assumption**: Assumes catalog pricing and promotion decisions are finalized at least 7 days prior to the target fulfillment week.
2. **Weekly Aggregation**: Designed specifically for weekly supply planning; does not capture sub-daily or hourly rush fluctuations.
3. **Cold-Start Items**: New culinary dishes without historical volume or catalog metadata default to general category medians until transaction history accumulates.
4. **Exogenous Factors**: Severe localized weather events or macro food inflation shocks not captured in price/calendar variables may degrade short-term accuracy.

---

## R. Security and Responsible Data Handling

- **No Plaintext Secrets**: No API keys, credentials, or confidential tokens exist in source code or configuration files.
- **Safe Serialization**: Persisted Joblib pipelines should only be loaded from verified internal build storage.
- **Input Boundaries**: Pydantic models strictly validate positive bounds on price, week, and IDs before executing inference.

---

## S. Future Improvements

1. **Direct LightGBM / XGBoost Benchmarking**: Add native gradient boosting engines with hyperparameter optimization using Optuna.
2. **Quantile Demand Forecasting**: Predict 10th, 50th, and 90th percentiles to support risk-averse supply chain safety stock buffers.
3. **Automated Retraining CI/CD**: Implement monthly cron-triggered retraining pipelines using GitHub Actions or Airflow.
4. **Real-time Feature Store**: Deploy Feast or Redis for low-latency retrieval of dynamic rolling demand statistics.

---

## T. Resume-Ready Project Summary

- Engineered an end-to-end Restaurant Order Demand Prediction ML system on 456k+ transactional records, developing a scikit-learn preprocessing pipeline and gradient-boosted tree regressor served via FastAPI and Streamlit.
- Designed a leakage-safe chronological validation strategy spanning 145 weeks, engineering price elasticity, promotion interaction, and Fourier annual seasonal features to predict weekly order counts.
- Implemented production software standards including atomic model artifact persistence, Pydantic schema validation, Ruff linting, and 23 automated pytest integration tests achieving 100% pass rate.

---

## U. Technical Interview Preparation

### 30-Second Summary
"I built a production ML system that forecasts weekly meal demand across 77 fulfillment centers. It ingests 456k historical transactions, engineers price discount and promotional interaction features, enforces chronological holdout validation to prevent leakage, and serves forecasts via a FastAPI REST API and Streamlit UI with 23 automated tests."

### Common Technical Questions & Answers

**Q: Why use chronological splitting instead of standard cross-validation?**  
*A: Random K-Fold splits randomly shuffle rows across time, allowing the model to train on future records while predicting the past. Chronological splitting (Weeks 1-125 for train, 126-135 for validation, 136-145 for test) ensures zero future leakage and accurately mimics production deployment.*

**Q: Why benchmark against a Group Median baseline?**  
*A: In restaurant operations, a kitchen's historical median demand for a given meal is the natural heuristic. Benchmarking against this proves whether the ML model captures dynamic pricing and promotion sensitivity beyond static past volume.*

**Q: How do you handle non-negative demand constraints?**  
*A: Standard regression models can predict small negative values on low-demand items. We enforce a domain post-processing rule clipping predictions to a minimum of 1.0 order, matching physical retail fulfillment constraints.*

---

## V. References

1. Genpact Food Demand Forecasting Challenge: Analytics Vidhya Machine Learning Competition.
2. Scikit-Learn Documentation: [scikit-learn.org](https://scikit-learn.org)
3. FastAPI Documentation: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
4. Streamlit Documentation: [docs.streamlit.io](https://docs.streamlit.io)
