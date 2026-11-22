# Pipeline orchestration for the Zurich Footfall Forecast.
# Runs historical backfills and the live feature pipeline, then stores the resulting training or live feature snapshots.

import argparse

import pandas as pd
from footfall.common.config import SETTINGS


from footfall.features.build_features import (
    build_base_features,
    build_training_features,
)
from footfall.features.ingest import ingest_pedestrian_data
from footfall.features.ingest_weather import (
    load_historical_weather,
    load_live_weather,
)
from footfall.features.store import save_feature_rows


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

    # Saves or updates the historical feature rows in the persistent GCS feature store.
    feature_store_path = save_feature_rows(
        feature_dataframe,
    )

    print(
        f"Saved {len(feature_dataframe)} training feature rows "
        f"to {feature_store_path}"
    )


def create_live_prediction_timestamps(
    reference_timestamp: pd.Timestamp,
) -> pd.DatetimeIndex:
    """Create prediction timestamps for all target hours of the next local day."""

    local_timezone = SETTINGS["time"]["local_timezone"]
    forecast_horizon = SETTINGS["target"]["forecast_horizon_hours"]

    reference_timestamp = pd.Timestamp(reference_timestamp)

    if reference_timestamp.tzinfo is None:
        reference_timestamp = reference_timestamp.tz_localize("UTC")
    else:
        reference_timestamp = reference_timestamp.tz_convert("UTC")

    reference_local = reference_timestamp.tz_convert(local_timezone)

    next_day_start_local = (
        reference_local.normalize()
        + pd.DateOffset(days=1)
    )

    next_day_end_local = (
        next_day_start_local
        + pd.DateOffset(days=1)
    )

    target_timestamps_local = pd.date_range(
        start=next_day_start_local,
        end=next_day_end_local,
        freq="h",
        inclusive="left",
    )

    prediction_timestamps_utc = (
        target_timestamps_local.tz_convert("UTC")
        - pd.Timedelta(hours=forecast_horizon)
    )

    return prediction_timestamps_utc



def run_live_pipeline() -> pd.DataFrame:
    """Build and store feature rows for all target hours of the next local day."""

    pedestrian_dataframe = ingest_pedestrian_data()

    latest_timestamp = pedestrian_dataframe[
        "timestamp"
    ].max()

    prediction_timestamps = create_live_prediction_timestamps(
        latest_timestamp
    )

    # Add timestamp-only anchor rows for hours that are not yet present
    # in the Hystreet data.
    anchor_dataframe = pd.DataFrame(
        {
            "timestamp": prediction_timestamps,
        }
    )

    extended_pedestrian_dataframe = pd.concat(
        [
            pedestrian_dataframe,
            anchor_dataframe,
        ],
        ignore_index=True,
    )

    # Keep the real Hystreet row if a timestamp already exists.
    extended_pedestrian_dataframe = (
        extended_pedestrian_dataframe
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    weather_dataframe = load_live_weather()

    feature_dataframe = build_base_features(
        extended_pedestrian_dataframe,
        weather_dataframe,
    )

    live_features = feature_dataframe[
        feature_dataframe["timestamp"].isin(
            prediction_timestamps
        )
    ].copy()

    if len(live_features) != len(prediction_timestamps):
        raise ValueError(
            "Could not build all required live feature rows."
        )

    weather_columns = [
        "forecast_temperature",
        "forecast_precipitation",
        "forecast_weather_code",
    ]

    if live_features[weather_columns].isna().any().any():
        raise ValueError(
            "Live feature rows contain missing weather values."
        )

    feature_store_path = save_feature_rows(
        live_features
    )

    print(
        f"Saved {len(live_features)} live feature rows "
        f"to {feature_store_path}"
    )

    return live_features




def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the feature pipeline."""

    parser = argparse.ArgumentParser(
        description="Run the Zurich footfall feature pipeline."
    )

    subparsers = parser.add_subparsers(
        dest="mode",                                  # mode: backfill or live
        required=True,
    )

    #Defines backfille mode
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

    # Defines Live Mode
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