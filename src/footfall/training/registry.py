from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


MODEL_NAME = "zurich-footfall-model"
CHAMPION_ALIAS = "champion"


def promote_if_best(
    model_version: int,
    validation_mae: float,
) -> bool:
    """Promote a model version if it beats the current champion."""

    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
        )
    except MlflowException:
        client.set_registered_model_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
            str(model_version),
        )
        return True

    champion_run = client.get_run(
        champion.run_id
    )

    champion_validation_mae = (
        champion_run.data.metrics.get(
            "validation_mae"
        )
    )

    if champion_validation_mae is None:
        raise ValueError(
            "Current champion does not have a "
            "'validation_mae' metric."
        )

    if validation_mae < champion_validation_mae:
        client.set_registered_model_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
            str(model_version),
        )
        return True

    return False