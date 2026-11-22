import pandas as pd


def evaluate_seasonal_naive(
    dataframe: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate a weekly seasonal-naive pedestrian forecast."""

    actual = dataframe["target_pedestrians"]
    prediction = dataframe["pedestrians_lag_168h"]

    absolute_error = (
        actual - prediction
    ).abs()

    mae = absolute_error.mean()

    return {
        "mae": float(mae),
    }