# Experiment Tracking and Benchmark Log

## 1. Experimental Setup

- **Objective**: Accurately predict weekly order count (`num_orders`) for each fulfillment center and meal combination.
- **Data Splitting**: Strict chronological split to prevent data leakage:
  - **Train**: Weeks 1 to 125 (390,907 records)
  - **Validation**: Weeks 126 to 135 (32,820 records)
  - **Test**: Weeks 136 to 145 (32,821 records)
- **Primary Metric**: MAE (Mean Absolute Error) and WAPE (Weighted Absolute Percentage Error).
- **Secondary Metrics**: RMSE, R², MedAE.

## 2. Model Iteration Log

### Experiment 1: Global Heuristic (Mean / Median)
- **Algorithm**: Global central tendency.
- **Description**: Predicts the single overall median order count across all historical training records (136.0 orders).
- **Validation Results**: MAE: 198.42, RMSE: 368.10, WAPE: 75.38%
- **Takeaway**: Far too coarse; completely misses center catchment size and meal popularity variations.

### Experiment 2: Domain Entity Baseline (Group Median)
- **Algorithm**: `BaselineDemandModel` (Group Median by `center_id` + `meal_id` with fallback to global median).
- **Description**: Represents the typical operational rule-of-thumb: predicting typical historical volume for a given dish at a given fulfillment center.
- **Validation Results**: MAE: 111.44, RMSE: 322.29, R²: 0.4139, WAPE: 42.32%
- **Test Results**: MAE: 82.19, RMSE: 143.46, R²: 0.7310, WAPE: 36.16%
- **Takeaway**: Strong baseline on normal non-promotional weeks, but unable to dynamically respond to discount pricing or email campaigns.

### Experiment 3: Tree-Based Regressor (Random Forest)
- **Algorithm**: `RandomForestRegressor(n_estimators=100, max_depth=14, n_jobs=-1)`
- **Observation**: Training time on 390k rows was high (> 120 seconds on CPU), memory footprint was large (> 1.2 GB), with moderate gains on continuous interaction splits.
- **Takeaway**: High inference latency and heavy artifact size made it less suitable for real-time microservice serving.

### Experiment 4: Primary Candidate Model (HistGradientBoostingRegressor)
- **Algorithm**: `HistGradientBoostingRegressor` with ColumnTransformer.
- **Hyperparameters**:
  - `max_iter`: 200
  - `learning_rate`: 0.08
  - `max_depth`: 12
  - `min_samples_leaf`: 25
  - `l2_regularization`: 0.5
  - `random_state`: 42
- **Training Time**: ~14 seconds on 390,907 rows.
- **Artifact Size**: ~1.8 MB (Joblib compressed).
- **Validation Results**:
  - MAE: 108.45 orders
  - RMSE: 237.42 orders
  - R²: 0.6819
  - WAPE: 41.18%
- **Test Results**:
  - MAE: 93.57 orders
  - RMSE: 156.18 orders
  - R²: 0.6812
  - WAPE: 41.16%
  - MedAE: 53.23 orders
- **Takeaway**: Selected as the primary production model due to its rapid training time, compact artifact size, sub-millisecond inference latency, and ability to model non-linear promotional spikes.

## 3. Comparative Summary Table

| Model | Partition | MAE | RMSE | R² | WAPE | Training Time | Artifact Size |
|---|---|---|---|---|---|---|---|
| **Global Median** | Val | 198.42 | 368.10 | 0.05 | 75.38% | < 0.1s | < 1 KB |
| **Group Median Baseline** | Val | 111.44 | 322.29 | 0.4139 | 42.32% | 0.5s | 150 KB |
| **Group Median Baseline** | Test | 82.19 | 143.46 | 0.7310 | 36.16% | - | - |
| **HistGradientBoosting** | Val | 108.45 | 237.42 | 0.6819 | 41.18% | 14s | 1.8 MB |
| **HistGradientBoosting** | Test | 93.57 | 156.18 | 0.6812 | 41.16% | - | - |
