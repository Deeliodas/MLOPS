import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from footfall.common.config import SETTINGS


def split_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the prediction target."""

    model_columns = SETTINGS["features"]["model_columns"]

    features = dataframe[model_columns].copy()

    target = dataframe["target_pedestrians"].copy()

    return features, target


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> HistGradientBoostingRegressor:
    """Train a reproducible gradient boosting regression model."""

    model = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=200,
        learning_rate=0.05,
        random_state=42,
    )

    model.fit(
        features,
        target,
    )

    return model