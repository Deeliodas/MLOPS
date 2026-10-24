import argparse

import pandas as pd

from footfall.features.build_features import (
    build_base_features,
    build_training_features,
)
from footfall.features.ingest import ingest_pedestrian_data
from footfall.features.ingest_weather import (
    load_historical_weather,
    load_live_weather,
    select_target_weather,
)
from footfall.features.store import save_feature_snapshot


def run_backfill(
    start_date: str,
    end_date: str,
) -> None:
    """Build and store historical training features for a given date range."""

    pedestrian_dataframe = ingest_pedestrian_data()

    weather_dataframe = load_historical_weather(
        start_date=start_date,
        end_date=end_date,
    )

    feature_dataframe = build_training_features(
        pedestrian_dataframe,
        weather_dataframe,
    )

    start_timestamp = pd.Timestamp(
        start_date,
        tz="UTC",
    )

    end_timestamp = (
        pd.Timestamp(
            end_date,
            tz="UTC",
        )
        + pd.Timedelta(days=1)
    )

    feature_dataframe = feature_dataframe[
        (feature_dataframe["target_timestamp"] >= start_timestamp)
        & (feature_dataframe["target_timestamp"] < end_timestamp)
    ].copy()

    file_path = save_feature_snapshot(
        feature_dataframe,
        store_type="training",
    )

    print(
        f"Saved {len(feature_dataframe)} training feature rows "
        f"to {file_path}"
    )


def run_live_pipeline() -> pd.DataFrame:
    """Build and store the latest feature row for live inference."""

    pedestrian_dataframe = ingest_pedestrian_data()

    latest_timestamp = pedestrian_dataframe[
        "timestamp"
    ].max()

    weather_dataframe = load_live_weather()

    target_weather = select_target_weather(
        weather_dataframe,
        prediction_timestamp=latest_timestamp,
    )

    feature_dataframe = build_base_features(
        pedestrian_dataframe,
        target_weather,
    )

    live_feature = feature_dataframe[
        feature_dataframe["timestamp"] == latest_timestamp
    ].copy()

    if live_feature.empty:
        raise ValueError(
            f"No live feature row found for {latest_timestamp}."
        )

    file_path = save_feature_snapshot(
        live_feature,
        store_type="live",
    )

    print(
        f"Saved live feature row to {file_path}"
    )

    return live_feature


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the feature pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the Zurich footfall feature pipeline."
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        required=True,
    )

    backfill_parser = subparsers.add_parser(
        "backfill",
        help="Build historical training features.",
    )

    backfill_parser.add_argument(
        "--start-date",
        required=True,
        help="Backfill start date in YYYY-MM-DD format.",
    )

    backfill_parser.add_argument(
        "--end-date",
        required=True,
        help="Backfill end date in YYYY-MM-DD format.",
    )

    subparsers.add_parser(
        "live",
        help="Build the latest live feature row.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the selected feature pipeline mode."""

    arguments = parse_arguments()

    if arguments.mode == "backfill":
        run_backfill(
            start_date=arguments.start_date,
            end_date=arguments.end_date,
        )

    elif arguments.mode == "live":
        run_live_pipeline()


if __name__ == "__main__":
    main()