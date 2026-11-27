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
    evaluate_quantile_models,
)
from footfall.training.model import (
    split_features_and_target,
    train_model,
    train_quantile_models,
)
from footfall.training.registry import (
    promote_if_best,
)


def run_training_pipeline() -> None:
    """Run the complete model training and evaluation pipeline."""

    mlflow.set_experiment(
        "zurich-footfall-training"
    )

    # Load verified and complete training rows from the GCS feature store.
    dataframe = load_training_data()

    # Chronological split:
    # 70% train, 15% validation, 15% test.
    (
        train_dataframe,
        validation_dataframe,
        test_dataframe,
    ) = split_training_data(
        dataframe
    )

    # Separate model features from target values.
    X_train, y_train = split_features_and_target(
        train_dataframe
    )

    X_validation, y_validation = split_features_and_target(
        validation_dataframe
    )

    X_test, y_test = split_features_and_target(
        test_dataframe
    )

    with mlflow.start_run():

        # Train the main point prediction model.
        model = train_model(
            X_train,
            y_train,
        )

        # Train P10, P50, and P90 quantile models.
        quantile_models = train_quantile_models(
            X_train,
            y_train,
        )

        # Evaluate the main model on validation data.
        validation_metrics = evaluate_model(
            model,
            X_validation,
            y_validation,
        )

        validation_baseline_metrics = evaluate_seasonal_naive(
            validation_dataframe
        )

        # Evaluate quantile models on validation data.
        validation_quantile_metrics = evaluate_quantile_models(
            quantile_models,
            X_validation,
            y_validation,
        )

        # Evaluate the main model on the independent test data.
        test_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        test_baseline_metrics = evaluate_seasonal_naive(
            test_dataframe
        )

        # Evaluate quantile models on the independent test data.
        test_quantile_metrics = evaluate_quantile_models(
            quantile_models,
            X_test,
            y_test,
        )

        # Log training configuration.
        mlflow.log_params(
            {
                "model_type": "HistGradientBoostingRegressor",
                "loss": "absolute_error",
                "max_iter": 200,
                "learning_rate": 0.05,
                "random_state": 42,
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "p10_quantile": 0.10,
                "p50_quantile": 0.50,
                "p90_quantile": 0.90,
            }
        )

        # Log point-model and baseline metrics.
        mlflow.log_metrics(
            {
                "validation_mae": validation_metrics["mae"],
                "validation_baseline_mae": (
                    validation_baseline_metrics["mae"]
                ),
                "test_mae": test_metrics["mae"],
                "test_baseline_mae": (
                    test_baseline_metrics["mae"]
                ),

                "validation_p10_pinball_loss": (
                    validation_quantile_metrics[
                        "p10_pinball_loss"
                    ]
                ),
                "validation_p50_pinball_loss": (
                    validation_quantile_metrics[
                        "p50_pinball_loss"
                    ]
                ),
                "validation_p90_pinball_loss": (
                    validation_quantile_metrics[
                        "p90_pinball_loss"
                    ]
                ),
                "validation_p10_p90_coverage": (
                    validation_quantile_metrics[
                        "p10_p90_coverage"
                    ]
                ),
                "validation_ordered_fraction": (
                    validation_quantile_metrics[
                        "ordered_fraction"
                    ]
                ),

                "test_p10_pinball_loss": (
                    test_quantile_metrics[
                        "p10_pinball_loss"
                    ]
                ),
                "test_p50_pinball_loss": (
                    test_quantile_metrics[
                        "p50_pinball_loss"
                    ]
                ),
                "test_p90_pinball_loss": (
                    test_quantile_metrics[
                        "p90_pinball_loss"
                    ]
                ),
                "test_p10_p90_coverage": (
                    test_quantile_metrics[
                        "p10_p90_coverage"
                    ]
                ),
                "test_ordered_fraction": (
                    test_quantile_metrics[
                        "ordered_fraction"
                    ]
                ),
            }
        )

        # Store and register the main model.
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="zurich-footfall-model",
            skops_trusted_types=[
                (
                    "sklearn.ensemble._hist_gradient_boosting."
                    "predictor.TreePredictor"
                ),
            ],
        )

        # Store P10, P50, and P90 models as artifacts of the same run.
        for name, quantile_model in quantile_models.items():
            mlflow.sklearn.log_model(
                sk_model=quantile_model,
                name=f"{name}_model",
                skops_trusted_types=[
                    (
                        "sklearn.ensemble._hist_gradient_boosting."
                        "predictor.TreePredictor"
                    ),
                ],
            )

        # Champion selection uses validation MAE only.
        # The test set is not used for model selection.
        promoted = promote_if_best(
            model_version=logged_model.registered_model_version,
            validation_mae=validation_metrics["mae"],
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

        print(
            f"Training rows: {len(train_dataframe)}"
        )

        print(
            f"Validation rows: {len(validation_dataframe)}"
        )

        print(
            f"Test rows: {len(test_dataframe)}"
        )

        print(
            f"Validation baseline MAE: "
            f"{validation_baseline_metrics['mae']:.2f}"
        )

        print(
            f"Validation model MAE: "
            f"{validation_metrics['mae']:.2f}"
        )

        print(
            f"Test baseline MAE: "
            f"{test_baseline_metrics['mae']:.2f}"
        )

        print(
            f"Test model MAE: "
            f"{test_metrics['mae']:.2f}"
        )

        print(
            f"Validation P10-P90 coverage: "
            f"{validation_quantile_metrics['p10_p90_coverage']:.4f}"
        )

        print(
            f"Validation ordered fraction: "
            f"{validation_quantile_metrics['ordered_fraction']:.4f}"
        )

        print(
            f"Test P10-P90 coverage: "
            f"{test_quantile_metrics['p10_p90_coverage']:.4f}"
        )

        print(
            f"Test ordered fraction: "
            f"{test_quantile_metrics['ordered_fraction']:.4f}"
        )


def main() -> None:
    """Run the training pipeline."""

    run_training_pipeline()


if __name__ == "__main__":
    main()