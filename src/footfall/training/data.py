import pandas as pd

from footfall.common.config import SETTINGS
from footfall.features.store import load_feature_store


def load_training_data() -> pd.DataFrame:
    """Load and prepare verified, complete rows for model training."""

    dataframe = load_feature_store()

    model_columns = SETTINGS["features"]["model_columns"]

    required_columns = [
        "target_timestamp",
        *model_columns,
        "target_pedestrians",
        "target_unverified",
    ]

    missing_columns = set(required_columns) - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required training columns: {sorted(missing_columns)}"
        )

    # Keep only rows for which the true pedestrian count is already known.
    dataframe = dataframe[
        dataframe["target_pedestrians"].notna()
    ].copy()

    #Use only verified pedestrian counts as training targets.
    dataframe = dataframe[
        ~dataframe["target_unverified"].astype(bool)
    ].copy()

    #Remove rows with incomplete model features.
    dataframe = dataframe.dropna(
        subset=model_columns
    ).copy()

    dataframe = dataframe.sort_values(
        "target_timestamp"
    ).reset_index(drop=True)

    return dataframe[required_columns].copy()




def split_training_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data chronologically into train and test sets."""

    train_fraction = SETTINGS["training"]["train_fraction"]

    if not 0 < train_fraction < 1:
        raise ValueError(
            "training.train_fraction must be between 0 and 1."
        )

    split_index = int(
        len(dataframe) * train_fraction
    )

    train_dataframe = dataframe.iloc[
        :split_index
    ].copy()

    test_dataframe = dataframe.iloc[
        split_index:
    ].copy()

    if train_dataframe.empty or test_dataframe.empty:
        raise ValueError(
            "Train/test split produced an empty dataset."
        )

    if (
        train_dataframe["target_timestamp"].max()
        >= test_dataframe["target_timestamp"].min()
    ):
        raise ValueError(
            "Training data overlaps with test data in time."
        )

    return train_dataframe, test_dataframe