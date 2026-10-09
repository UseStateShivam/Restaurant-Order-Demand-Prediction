# Dataset Documentation and Directory Structure

This directory stores raw, intermediate, and processed datasets for the Restaurant Order Demand Prediction system.

## Directory Layout

- `raw/`: Unmodified primary source files as downloaded from the official repository or ingested from upstream.
  - `train.csv`: Historical weekly order records across fulfillment centers and meal items.
  - `fulfilment_center_info.csv`: Fulfillment center metadata (city, region, type, operational area).
  - `meal_info.csv`: Meal catalog metadata (category, cuisine).
- `interim/`: Intermediate transformed or merged data frames generated during feature engineering before the final split.
- `processed/`: Cleaned, split datasets ready for pipeline ingestion (`train_clean.csv`, `val_clean.csv`, `test_clean.csv`).

## Data Acquisition

To fetch the raw dataset programmatically, execute:
```bash
python scripts/download_data.py
```

Or manually place the three CSV files into `data/raw/`:
1. `train.csv` (source: Genpact / Analytics Vidhya Food Demand Forecasting Challenge)
2. `fulfilment_center_info.csv`
3. `meal_info.csv`

## Licensing and Restrictions

The dataset originates from the Genpact Food Demand Forecasting Challenge hosted publicly on Analytics Vidhya and mirrored across open research repositories. The data is anonymized and intended for research, algorithmic benchmarking, and educational forecasting applications.
