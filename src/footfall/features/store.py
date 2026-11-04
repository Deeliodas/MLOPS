# Feature storage for the Zurich Footfall Forecast.
# Saves versioned training and live feature snapshots as Parquet files
#and loads the most recent snapshot from the feature store.

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from footfall.common.config import ROOT_DIR, SETTINGS


def get_feature_store_directory(store_type: str) -> Path:
    """Return the configured feature store directory."""

    base_directory = ROOT_DIR / SETTINGS["feature_store"]["directory"]

    if store_type == "training":
        subdirectory = SETTINGS["feature_store"]["training_subdirectory"]
    elif store_type == "live":
        subdirectory = SETTINGS["feature_store"]["live_subdirectory"]
    else:
        raise ValueError(
            f"Unknown feature store type: {store_type}"
        )

    return base_directory / subdirectory


def save_feature_snapshot(
    dataframe: pd.DataFrame,
    store_type: str,
) -> Path:
    """Save a versioned feature snapshot as a Parquet file."""

    feature_store_directory = get_feature_store_directory(
        store_type
    )

    feature_store_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename_prefix = SETTINGS["feature_store"]["filename_prefix"]

    # creates the timestamp for version control
    version = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    file_path = feature_store_directory / (
        f"{filename_prefix}_{version}.parquet"
    )

    #saves DataFrame as Parquet File
    dataframe.to_parquet(
        file_path,
        index=False,
    )

    return file_path


def load_latest_feature_snapshot(
    store_type: str,
) -> pd.DataFrame:
    """Load the most recent feature snapshot."""

    feature_store_directory = get_feature_store_directory(
        store_type
    )

    filename_prefix = SETTINGS["feature_store"]["filename_prefix"]

    feature_files = sorted(
        #find all matching Parquet files
        feature_store_directory.glob(
            f"{filename_prefix}_*.parquet"
        )
    )

    if not feature_files:
        raise FileNotFoundError(
            f"No {store_type} feature snapshots found."
        )

    # takes the last file in the sorted list.
    latest_file = feature_files[-1]

    return pd.read_parquet(latest_file)