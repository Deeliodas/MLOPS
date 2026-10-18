import pandas as pd

from footfall.common.config import SETTINGS


def load_source_data() -> pd.DataFrame:
    """Load the hystreet fussgaengerfrequenz seit 2021 dataset from Open Data Zürich"""

    source_url = SETTINGS["data"]["source_url"]

    dataframe = pd.read_csv(source_url)

    return dataframe



def filter_location(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter the dataset for the configured location."""

    location_id = SETTINGS["data"]["location_id"]

    filtered_dataframe = dataframe[
        dataframe["location_id"] == location_id
    ].copy()

    return filtered_dataframe



def prepare_timestamps(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamps to UTC datetime and sort the dataset chronologically"""

    prepared_dataframe = dataframe.copy()

    prepared_dataframe["timestamp"] = pd.to_datetime(
        prepared_dataframe["timestamp"],
        utc=True,
    )

    prepared_dataframe = prepared_dataframe.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return prepared_dataframe



def validate_data(dataframe: pd.DataFrame) -> None:
    """Validate the basic structure and hourly continuity of the dataset."""

    required_columns = {
        "timestamp",
        "location_id",
        "location_name",
        "pedestrians_count",
        "unverified",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    duplicate_timestamps = dataframe["timestamp"].duplicated().sum()

    if duplicate_timestamps > 0:
        raise ValueError(
            f"Found {duplicate_timestamps} duplicate timestamps."
        )

    time_differences = dataframe["timestamp"].diff().dropna()

    expected_difference = pd.Timedelta(hours=1)

    invalid_differences = time_differences[
        time_differences != expected_difference
    ]

    if not invalid_differences.empty:
        raise ValueError(
            f"Found {len(invalid_differences)} non-hourly time gaps."
        )
        



def ingest_pedestrian_data() -> pd.DataFrame:
    """Load, filter, prepare, and validate pedestrian data."""

    dataframe = load_source_data()
    dataframe = filter_location(dataframe)
    dataframe = prepare_timestamps(dataframe)

    validate_data(dataframe)

    return dataframe