import mlflow
import mlflow.sklearn


from footfall.training.baseline import (
    evaluate_seasonal_naive,
)
from footfall.training.data import (
    load_training_data,
    split_training_data,
)
from footfall.training.evaluate import (
    evaluate_model,
)
from footfall.training.model import (
    split_features_and_target,
    train_model,
)

from footfall.training.registry import (
    promote_if_best,
)

def run_training_pipeline() -> None:
    """Run the complete model training and evaluation pipeline."""

    mlflow.set_experiment(
        "zurich-footfall-training"
    )

    dataframe = load_training_data()

    train_dataframe, test_dataframe = split_training_data(
        dataframe
    )

    X_train, y_train = split_features_and_target(
        train_dataframe
    )

    X_test, y_test = split_features_and_target(
        test_dataframe
    )

    with mlflow.start_run():

        model = train_model(
            X_train,
            y_train,
        )

        model_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        baseline_metrics = evaluate_seasonal_naive(
            test_dataframe
        )

        mlflow.log_params(
            {
                "model_type": "HistGradientBoostingRegressor",
                "loss": "absolute_error",
                "max_iter": 200,
                "learning_rate": 0.05,
                "random_state": 42,
                "train_fraction": 0.8,
            }
        )

        mlflow.log_metrics(
            {
                "model_mae": model_metrics["mae"],
                "baseline_mae": baseline_metrics["mae"],
            }
        )
        
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="zurich-footfall-model",
            skops_trusted_types=[
                "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
            ],
        )

        promoted = promote_if_best(
            model_version=logged_model.registered_model_version,
            model_mae=model_metrics["mae"],
        )

        if promoted:
            print(
                f"Model version "
                f"{logged_model.registered_model_version} "
                f"promoted to champion."
            )
        else:
            print(
                f"Model version "
                f"{logged_model.registered_model_version} "
                f"did not beat the current champion."
            )
        
        promoted = promote_if_best(
            model_version=logged_model.registered_model_version,
            model_mae=model_metrics["mae"],
        )

        print(
            f"Training rows: {len(train_dataframe)}"
        )

        print(
            f"Test rows: {len(test_dataframe)}"
        )

        print(
            f"Seasonal-naive MAE: "
            f"{baseline_metrics['mae']:.2f}"
        )

        print(
            f"ML model MAE: "
            f"{model_metrics['mae']:.2f}"
        )


def main() -> None:
    """Run the training pipeline."""

    run_training_pipeline()


if __name__ == "__main__":
    main()