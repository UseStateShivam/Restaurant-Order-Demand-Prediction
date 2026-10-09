"""Script to download raw Food Demand Forecasting Challenge dataset files.

Downloads the three relational CSV files:
- train.csv
- fulfilment_center_info.csv
- meal_info.csv
into data/raw/.
"""

import sys

from restaurant_demand.config import load_data_config
from restaurant_demand.data.ingestion import DataIngestion
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)


def main() -> int:
    """Executes dataset acquisition."""
    logger.info("Initiating dataset download script...")
    data_config = load_data_config()
    ingestion = DataIngestion(config=data_config)
    try:
        ingestion.download_raw_data_if_missing()
        print("Data download completed successfully. All files present in data/raw/.")
        return 0
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        print(f"Error during dataset acquisition: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
