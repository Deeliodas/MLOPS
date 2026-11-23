from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = "zurich-footfall-model"
CHAMPION_ALIAS = "champion"


def promote_if_best(
    model_version: int,
    model_mae: float,
) -> bool:
    """Promote a model version to champion if it has the best MAE."""

    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
        )

    except MlflowException:
        # No champion exists yet, so the first model becomes champion.
        client.set_registered_model_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
            str(model_version),
        )

        return True

    champion_run = client.get_run(
        champion.run_id
    )

    champion_mae = champion_run.data.metrics[
        "model_mae"
    ]

    if model_mae < champion_mae:
        client.set_registered_model_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
            str(model_version),
        )

        return True

    return False