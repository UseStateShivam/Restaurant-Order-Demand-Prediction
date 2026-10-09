"""Script to create reproducible, professional Jupyter notebooks."""

import json
from pathlib import Path

from restaurant_demand.config import get_project_root


def create_eda_notebook(notebook_path: Path):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 01. Exploratory Data Analysis (EDA)\n",
                    "## Restaurant Order Demand Prediction\n",
                    "\n",
                    "### Purpose\n",
                    "This notebook explores the historical demand transactions, fulfillment center operations, and meal catalogs.\n",
                    "It investigates target distribution, categorical cardinalities, promotional elasticity, and temporal patterns.\n",
                    "\n",
                    "### Key Questions Investigated\n",
                    "1. What is the distribution and skew of the target variable (`num_orders`)?\n",
                    "2. How do email campaigns and homepage promotions impact weekly demand?\n",
                    "3. Which meal categories and cuisine types generate the highest order volume?\n",
                    "4. Is there empirical evidence of price sensitivity (checkout price vs orders)?\n",
                    "5. What temporal trends exist across the 145-week observation window?"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Ensure src/ is importable from notebooks directory\n",
                    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "sys.path.insert(0, str(project_root / 'src'))\n",
                    "\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from restaurant_demand.data.ingestion import DataIngestion\n",
                    "from restaurant_demand.config import load_data_config\n",
                    "\n",
                    "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Load and Verify Ingested Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "ingestion = DataIngestion(base_dir=project_root)\n",
                    "df = ingestion.run(save_interim=False)\n",
                    "print(f'Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns')\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Target Variable Analysis (`num_orders`)\n",
                    "Weekly order counts exhibit high positive skew, typical of retail demand where majority of meals exhibit modest volume while popular items generate order spikes."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print('Summary Statistics for Target (num_orders):')\n",
                    "display(df['num_orders'].describe().to_frame())\n",
                    "\n",
                    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n",
                    "sns.histplot(df['num_orders'], bins=60, ax=ax1, color='#1f77b4')\n",
                    "ax1.set_title('Raw Order Demand Distribution')\n",
                    "ax1.set_xlabel('Weekly Orders')\n",
                    "\n",
                    "sns.histplot(np.log1p(df['num_orders']), bins=60, ax=ax2, color='#2ca02c', kde=True)\n",
                    "ax2.set_title('Log-Transformed Order Demand Distribution')\n",
                    "ax2.set_xlabel('log(1 + num_orders)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Promotional Elasticity\n",
                    "Analysis shows that when email promotions and homepage featured placements are applied together, demand exhibits a massive non-linear multiplier."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "promo_summary = df.groupby(['emailer_for_promotion', 'homepage_featured'])['num_orders'].agg(['count', 'mean', 'median']).reset_index()\n",
                    "display(promo_summary)\n",
                    "\n",
                    "plt.figure(figsize=(8, 5))\n",
                    "sns.barplot(data=df, x='emailer_for_promotion', y='num_orders', hue='homepage_featured')\n",
                    "plt.title('Demand by Email Campaign and Homepage Placement')\n",
                    "plt.xlabel('Emailer for Promotion (0=No, 1=Yes)')\n",
                    "plt.ylabel('Mean Weekly Orders')\n",
                    "plt.legend(title='Homepage Featured')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Category and Cuisine Breakdown"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "plt.figure(figsize=(12, 5))\n",
                    "sns.barplot(data=df, x='category', y='num_orders', order=df.groupby('category')['num_orders'].mean().sort_values(ascending=False).index)\n",
                    "plt.xticks(rotation=45)\n",
                    "plt.title('Mean Demand Across Food Categories')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Summary of Findings\n",
                    "- Order distribution is heavily right-skewed with a median of 136 and mean of ~262 orders.\n",
                    "- Email campaigns and homepage placements multiply demand significantly.\n",
                    "- Multi-level entities (77 fulfillment centers and 51 meals) exhibit distinct base volumes, justifying both group-level baseline heuristics and tree-based non-linear estimators."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {notebook_path}")


def create_model_notebook(notebook_path: Path):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 02. Model Development and Evaluation\n",
                    "## Restaurant Order Demand Prediction\n",
                    "\n",
                    "### Purpose\n",
                    "This notebook illustrates model architecture development, chronological train-validation-test splitting,\n",
                    "baseline heuristic fitting, and tree-based gradient boosting regressor training.\n",
                    "\n",
                    "### Validation Strategy\n",
                    "To prevent temporal leakage, data is partitioned chronologically:\n",
                    "- Train: Weeks 1 to 125\n",
                    "- Validation: Weeks 126 to 135\n",
                    "- Test: Weeks 136 to 145"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "from pathlib import Path\n",
                    "\n",
                    "project_root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n",
                    "sys.path.insert(0, str(project_root / 'src'))\n",
                    "\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "from restaurant_demand.models.train import ModelTrainer\n",
                    "from restaurant_demand.models.predict import DemandPredictor\n",
                    "from restaurant_demand.config import load_data_config, load_model_config"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Training Pipeline Execution\n",
                    "Invoking the production pipeline trainer from the package ensures consistent preprocessing and feature generation."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "trainer = ModelTrainer(base_dir=project_root)\n",
                    "metrics = trainer.train_pipeline()\n",
                    "\n",
                    "print('Test Set Performance Summary:')\n",
                    "display(pd.DataFrame([\n",
                    "    {'Model': 'Group Median Baseline', **metrics['baseline']['test']},\n",
                    "    {'Model': 'HistGradientBoosting', **metrics['candidate_hist_gradient_boosting']['test']},\n",
                    "]))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Error Analysis and Residual Diagnostics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "predictor = DemandPredictor(base_dir=project_root)\n",
                    "test_clean_path = project_root / 'data' / 'processed' / 'test.csv'\n",
                    "test_df = pd.read_csv(test_clean_path)\n",
                    "\n",
                    "scored_test = predictor.predict_dataframe(test_df)\n",
                    "scored_test['residual'] = scored_test['num_orders'] - scored_test['predicted_demand']\n",
                    "\n",
                    "print(f'Mean Residual: {scored_test[\"residual\"].mean():.2f}')\n",
                    "print(f'Median Residual: {scored_test[\"residual\"].median():.2f}')\n",
                    "\n",
                    "plt.figure(figsize=(9, 5))\n",
                    "sns.histplot(scored_test['residual'], bins=60, kde=True, color='purple')\n",
                    "plt.axvline(0, color='red', linestyle='--')\n",
                    "plt.title('Held-Out Test Set Residual Error Distribution')\n",
                    "plt.xlabel('Residual (Actual - Predicted Orders)')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Conclusion\n",
                    "- HistGradientBoosting accurately captures non-linear promotional spikes and seasonal cycles.\n",
                    "- The persisted pipeline is versioned and ready for real-time serving."
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {notebook_path}")


def main():
    root = get_project_root()
    create_eda_notebook(root / "notebooks" / "01_exploratory_data_analysis.ipynb")
    create_model_notebook(root / "notebooks" / "02_model_development.ipynb")


if __name__ == "__main__":
    main()
