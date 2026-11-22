# Feature storage for the Zurich Footfall Forecast.
# Saves versioned training and live feature snapshots as Parquet files
#and loads the most recent snapshot from the feature store.

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from google.cloud import storage

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


def get_feature_store_blob():
    """Return the GCS blob used as the persistent feature store."""

    bucket_name = SETTINGS["feature_store"]["bucket_name"]
    object_name = SETTINGS["feature_store"]["object_name"]

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    return bucket.blob(object_name)


def load_feature_store() -> pd.DataFrame:
    """Load the complete persistent feature store from GCS."""

    blob = get_feature_store_blob()

    # If the feature store does not exist yet, return an empty DataFrame.
    if not blob.exists():
        return pd.DataFrame()

    parquet_bytes = blob.download_as_bytes()

    return pd.read_parquet(
        BytesIO(parquet_bytes)
    )


def save_feature_rows(
    dataframe: pd.DataFrame,
) -> str:
    """Insert or update feature rows in the persistent GCS feature store."""

    key_column = SETTINGS["feature_store"]["key_column"]

    if key_column not in dataframe.columns:
        raise ValueError(
            f"Missing feature store key column: {key_column}"
        )

    # Keep only the newest version if the input contains the same key twice.
    new_rows = (
        dataframe
        .drop_duplicates(
            subset=[key_column],
            keep="last",
        )
        .copy()
    )

    existing_dataframe = load_feature_store()

    if existing_dataframe.empty:
        combined_dataframe = new_rows

    else:
        # Use target_timestamp as the unique index for updating rows.
        existing_dataframe = (
            existing_dataframe
            .drop_duplicates(
                subset=[key_column],
                keep="last",
            )
            .set_index(key_column)
        )

        new_rows = new_rows.set_index(key_column)

        # New non-null values replace old values.
        # Existing values are kept if the new row contains NaN.
        combined_dataframe = (
            new_rows
            .combine_first(existing_dataframe)
            .reset_index()
        )

    combined_dataframe = (
        combined_dataframe
        .sort_values(key_column)
        .reset_index(drop=True)
    )

    # Write the complete DataFrame into an in-memory Parquet file.
    parquet_buffer = BytesIO()

    combined_dataframe.to_parquet(
        parquet_buffer,
        index=False,
    )

    parquet_buffer.seek(0)

    blob = get_feature_store_blob()

    # Upload the updated Parquet dataset to Google Cloud Storage.
    blob.upload_from_file(
        parquet_buffer,
        content_type="application/octet-stream",
    )

    bucket_name = SETTINGS["feature_store"]["bucket_name"]
    object_name = SETTINGS["feature_store"]["object_name"]

    return f"gs://{bucket_name}/{object_name}"