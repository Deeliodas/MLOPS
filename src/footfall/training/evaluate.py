import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def evaluate_model(
    model: HistGradientBoostingRegressor,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Evaluate a trained regression model."""

    predictions = model.predict(features)

    mae = mean_absolute_error(
        target,
        predictions,
    )

    return {
        "mae": float(mae),
    }