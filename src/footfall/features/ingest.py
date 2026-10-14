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