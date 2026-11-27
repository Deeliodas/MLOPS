import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_pinball_loss
from sklearn.ensemble import HistGradientBoostingRegressor


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


def evaluate_quantile_models(
    models: dict[str, HistGradientBoostingRegressor],
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Evaluate P10, P50, and P90 quantile regression models."""

    p10 = models["p10"].predict(features)
    p50 = models["p50"].predict(features)
    p90 = models["p90"].predict(features)

    p10_loss = mean_pinball_loss(
        target,
        p10,
        alpha=0.10,
    )

    p50_loss = mean_pinball_loss(
        target,
        p50,
        alpha=0.50,
    )

    p90_loss = mean_pinball_loss(
        target,
        p90,
        alpha=0.90,
    )

    coverage = np.mean(
        (target.to_numpy() >= p10)
        & (target.to_numpy() <= p90)
    )

    ordered_fraction = np.mean(
        (p10 <= p50)
        & (p50 <= p90)
    )

    return {
        "p10_pinball_loss": float(p10_loss),
        "p50_pinball_loss": float(p50_loss),
        "p90_pinball_loss": float(p90_loss),
        "p10_p90_coverage": float(coverage),
        "ordered_fraction": float(ordered_fraction),
    }