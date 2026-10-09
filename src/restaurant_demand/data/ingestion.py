"""Data ingestion module for Restaurant Order Demand Prediction.

Handles downloading, loading, and merging the three relational raw datasets:
1. train.csv: Weekly order demand transactions by center and meal.
2. fulfilment_center_info.csv: Metadata describing center logistics and catchment area.
3. meal_info.csv: Metadata describing culinary classification and category.

Separating ingestion from feature engineering ensures raw data remains immutable,
and network retrieval failures are isolated with clear, actionable diagnostics.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from restaurant_demand.config import DataConfig, get_project_root
from restaurant_demand.exceptions import DataIngestionError
from restaurant_demand.logging_config import configure_logger

logger = configure_logger(__name__)

# Public repository raw URLs hosting the verified benchmark dataset
REMOTE_DATA_URLS: dict[str, str] = {
    "meal_info.csv": (
        "https://raw.githubusercontent.com/Shivansh-Commits/"
        "Food-Demand-Forecasting-DATA-SCIENCE-CHALLANGE/master/meal_info.csv"
    ),
    "fulfilment_center_info.csv": (
        "https://raw.githubusercontent.com/Shivansh-Commits/"
        "Food-Demand-Forecasting-DATA-SCIENCE-CHALLANGE/master/fulfilment_center_info.csv"
    ),
    "train.csv": (
        "https://raw.githubusercontent.com/Shivansh-Commits/"
        "Food-Demand-Forecasting-DATA-SCIENCE-CHALLANGE/master/train.csv"
    ),
}


class DataIngestion:
    """Manages raw dataset acquisition, loading, and initial relational joins."""

    def __init__(self, config: DataConfig | None = None, base_dir: Path | None = None):
        """Initializes the DataIngestion manager.

        Args:
            config: Optional DataConfig instance. Uses default if None.
            base_dir: Optional project root path override.
        """
        self.config = config or DataConfig()
        self.root = base_dir or get_project_root()
        self.raw_dir = self.root / Path(self.config.raw.train_path).parent
        self.interim_dir = self.root / Path(self.config.interim.merged_path).parent

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def download_raw_data_if_missing(self) -> None:
        """Downloads raw CSV files from verified remote sources if not already on disk.

        Raises:
            DataIngestionError: If network retrieval or file writing fails.
        """
        for filename, url in REMOTE_DATA_URLS.items():
            destination = self.raw_dir / filename
            if destination.exists() and destination.stat().st_size > 0:
                logger.info(
                    "File already exists locally: %s (%d bytes)",
                    destination,
                    destination.stat().st_size,
                )
                continue

            logger.info("Downloading %s from %s...", filename, url)
            try:
                # Using standard library urllib with user agent to prevent 403 Forbidden errors
                req = urllib.request.Request(url, headers={"User-Agent": "RestaurantDemandML/1.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    content = response.read()
                    destination.write_bytes(content)
                logger.info("Successfully saved %s (%d bytes)", filename, len(content))
            except Exception as exc:
                raise DataIngestionError(
                    f"Failed to download {filename} from {url}. Error: {exc}. "
                    f"Please place {filename} manually inside {self.raw_dir}."
                ) from exc

    def load_raw_tables(
        self,
        train_path: Path | None = None,
        center_path: Path | None = None,
        meal_path: Path | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads the individual raw CSV tables into DataFrames.

        Args:
            train_path: Optional override path for train.csv.
            center_path: Optional override path for fulfilment_center_info.csv.
            meal_path: Optional override path for meal_info.csv.

        Returns:
            Tuple of (train_df, center_df, meal_df).

        Raises:
            DataIngestionError: If any CSV cannot be found or read.
        """
        t_path = train_path or (self.root / self.config.raw.train_path)
        c_path = center_path or (self.root / self.config.raw.center_path)
        m_path = meal_path or (self.root / self.config.raw.meal_path)

        for p in (t_path, c_path, m_path):
            if not p.exists():
                raise DataIngestionError(
                    f"Required raw file does not exist: {p}. "
                    "Run 'python scripts/download_data.py' or provide valid path."
                )

        try:
            logger.info("Loading raw train data from %s", t_path)
            train_df = pd.read_csv(t_path)
            logger.info("Loading center info from %s", c_path)
            center_df = pd.read_csv(c_path)
            logger.info("Loading meal info from %s", m_path)
            meal_df = pd.read_csv(m_path)
            return train_df, center_df, meal_df
        except Exception as exc:
            raise DataIngestionError(f"Error reading CSV files: {exc}") from exc

    def merge_datasets(
        self,
        train_df: pd.DataFrame,
        center_df: pd.DataFrame,
        meal_df: pd.DataFrame,
        save_interim: bool = True,
    ) -> pd.DataFrame:
        """Performs referential joins between transactional demand and entity catalogs.

        Left joins are executed on center_id and meal_id so that all demand records
        are preserved while enriching each row with facility and meal attributes.

        Args:
            train_df: Demand transaction DataFrame.
            center_df: Center metadata DataFrame.
            meal_df: Meal metadata DataFrame.
            save_interim: If True, writes the merged table to data/interim/.

        Returns:
            pd.DataFrame: Merged relational DataFrame.

        Raises:
            DataIngestionError: If key columns are missing during join.
        """
        if "center_id" not in train_df.columns or "center_id" not in center_df.columns:
            raise DataIngestionError(
                "Cannot merge: 'center_id' missing from train or center metadata."
            )
        if "meal_id" not in train_df.columns or "meal_id" not in meal_df.columns:
            raise DataIngestionError("Cannot merge: 'meal_id' missing from train or meal metadata.")

        logger.info("Merging train records (%d rows) with centers and meals...", len(train_df))
        merged = train_df.merge(center_df, on="center_id", how="left")
        merged = merged.merge(meal_df, on="meal_id", how="left")

        # Referential integrity check: ensuring no unmatched centers or meals created NaNs
        if merged["center_type"].isnull().any():
            missing_centers = merged[merged["center_type"].isnull()]["center_id"].unique()
            raise DataIngestionError(
                f"Referential integrity violation: unknown center_ids encountered in train data: {missing_centers}"
            )
        if merged["category"].isnull().any():
            missing_meals = merged[merged["category"].isnull()]["meal_id"].unique()
            raise DataIngestionError(
                f"Referential integrity violation: unknown meal_ids encountered in train data: {missing_meals}"
            )

        if save_interim:
            interim_path = self.root / self.config.interim.merged_path
            interim_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(interim_path, index=False)
            logger.info(
                "Saved interim merged dataset to %s (%d rows, %d cols)", interim_path, *merged.shape
            )

        return merged

    def run(self, save_interim: bool = True) -> pd.DataFrame:
        """Executes the full raw data ingestion and merge pipeline.

        Returns:
            pd.DataFrame: Complete merged interim dataset.
        """
        self.download_raw_data_if_missing()
        train_df, center_df, meal_df = self.load_raw_tables()
        return self.merge_datasets(train_df, center_df, meal_df, save_interim=save_interim)
