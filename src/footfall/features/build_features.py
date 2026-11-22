#Feature engineering for the Zurich Footfall Forecast.
#Creates target-time, lag, rolling, weather, holiday, and event features
#and builds the final feature tables for training and live inference.

import pandas as pd
import holidays

from footfall.common.config import SETTINGS


def add_target_time_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add target timestamp and calendar features for the forecasted hour."""

    feature_dataframe = dataframe.copy()

    forecast_horizon = SETTINGS["target"]["forecast_horizon_hours"]
    local_timezone = SETTINGS["time"]["local_timezone"]

    feature_dataframe["target_timestamp"] = (
        feature_dataframe["timestamp"]
        + pd.Timedelta(hours=forecast_horizon)
    )

    #Convert UTC to Europe/Zürich
    local_target_time = feature_dataframe[
        "target_timestamp"
    ].dt.tz_convert(local_timezone)

    feature_dataframe["target_hour"] = local_target_time.dt.hour
    #Saves the day of the week as number (e.g. Monday = 0)
    feature_dataframe["target_day_of_week"] = local_target_time.dt.dayofweek

    feature_dataframe["target_is_weekend"] = (
        local_target_time.dt.dayofweek >= 5
    ).astype(int)

    feature_dataframe["target_month"] = local_target_time.dt.month

    return feature_dataframe




def add_pedestrian_lag_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add timestamp-based pedestrian lag features."""

    feature_dataframe = dataframe.copy()

    forecast_horizon = SETTINGS["target"]["forecast_horizon_hours"]
    verified_only = SETTINGS["data_quality"]["verified_only"]

    lag_source = dataframe[
        [
            "timestamp",
            "pedestrians_count",
            "unverified",
        ]
    ].copy()

    # Only use verified pedestrian counts if configured.
    if verified_only:
        lag_source["pedestrians_count"] = (
            lag_source["pedestrians_count"].where(
                ~lag_source["unverified"].astype(bool)
            )
        )

    # 24-hour lag relative to prediction time.
    lag_24h_dataframe = lag_source[
        ["timestamp", "pedestrians_count"]
    ].copy()

    lag_24h_dataframe["timestamp"] = (
        lag_24h_dataframe["timestamp"]
        + pd.Timedelta(hours=24)
    )

    lag_24h_dataframe = lag_24h_dataframe.rename(
        columns={
            "pedestrians_count": "pedestrians_lag_24h"
        }
    )

    feature_dataframe = feature_dataframe.merge(
        lag_24h_dataframe,
        on="timestamp",
        how="left",
    )

    # Weekly lag relative to target time.
    # target_timestamp = prediction timestamp + forecast horizon.
    # Therefore, target - 168h equals prediction timestamp - 144h
    # for a 24-hour forecast horizon.
    weekly_lag_from_prediction = (
        168 - forecast_horizon
    )

    weekly_lag_dataframe = lag_source[
        ["timestamp", "pedestrians_count"]
    ].copy()

    weekly_lag_dataframe["timestamp"] = (
        weekly_lag_dataframe["timestamp"]
        + pd.Timedelta(
            hours=weekly_lag_from_prediction
        )
    )

    weekly_lag_dataframe = weekly_lag_dataframe.rename(
        columns={
            "pedestrians_count": "pedestrians_lag_168h"
        }
    )

    feature_dataframe = feature_dataframe.merge(
        weekly_lag_dataframe,
        on="timestamp",
        how="left",
    )

    return feature_dataframe




def add_target(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add the future pedestrian count used as the training target."""

    feature_dataframe = dataframe.copy()

    forecast_horizon = SETTINGS["target"]["forecast_horizon_hours"]

    target_dataframe = dataframe[
        ["timestamp", "pedestrians_count", "unverified"]
    ].copy()

    #Move tiemstamp 24hours back to get the target value
    target_dataframe["timestamp"] = (
        target_dataframe["timestamp"]
        - pd.Timedelta(hours=forecast_horizon)
    )

    target_dataframe = target_dataframe.rename(
        columns={
            "pedestrians_count": "target_pedestrians",
            "unverified": "target_unverified",
        }
    )

    feature_dataframe = feature_dataframe.merge(
        target_dataframe,
        on="timestamp",
        how="left",
    )

    return feature_dataframe




def add_weather_features(
    dataframe: pd.DataFrame,
    weather_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Add weather forecasts for the target timestamp."""

    feature_dataframe = dataframe.copy()

    weather_features = weather_dataframe[
        [
            "timestamp",
            "forecast_temperature",
            "forecast_precipitation",
            "forecast_weather_code",
        ]
    ].copy()

    weather_features = weather_features.rename(
        columns={
            "timestamp": "target_timestamp",
        }
    )

    feature_dataframe = feature_dataframe.merge(
        weather_features,
        on="target_timestamp",
        how="left",
    )

    return feature_dataframe




def add_holiday_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a holiday flag for the target date in Zurich."""

    feature_dataframe = dataframe.copy()

    local_timezone = SETTINGS["time"]["local_timezone"]

    local_target_time = feature_dataframe[
        "target_timestamp"
    ].dt.tz_convert(local_timezone)

    # removes Date & timezone, keeps only the date
    target_dates = local_target_time.dt.date

    years = sorted({
        target_date.year
        for target_date in target_dates
    })

    zurich_holidays = holidays.country_holidays(
        "CH",
        subdiv="ZH",
        years=years,
    )

    feature_dataframe["target_is_holiday"] = target_dates.apply(
        lambda target_date: int(target_date in zurich_holidays)
    )

    return feature_dataframe



def add_event_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add major Zurich event features for the target date."""

    feature_dataframe = dataframe.copy()

    local_timezone = SETTINGS["time"]["local_timezone"]

    local_target_time = feature_dataframe[
        "target_timestamp"
    ].dt.tz_convert(local_timezone)

    target_dates = local_target_time.dt.strftime("%Y-%m-%d")

    street_parade_dates = set(
        SETTINGS["events"]["street_parade_impact"]
    )

    sechselaeuten_dates = set(
        SETTINGS["events"]["sechselaeuten"]
    )

    feature_dataframe["target_is_street_parade_impact"] = (
        target_dates.isin(street_parade_dates)
    ).astype(int)

    feature_dataframe["target_is_sechselaeuten"] = (
        target_dates.isin(sechselaeuten_dates)
    ).astype(int)

    return feature_dataframe




def add_rolling_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add verified rolling pedestrian means ending 24 hours before prediction time."""

    feature_dataframe = dataframe.copy()

    rolling_windows = SETTINGS["features"]["rolling_windows"]
    end_lag_hours = SETTINGS["features"]["rolling_end_lag_hours"]

    rolling_source = dataframe[
        [
            "timestamp",
            "pedestrians_count",
            "unverified",
        ]
    ].copy()

    # Only keep verified footfall values
    verified_counts = rolling_source["pedestrians_count"].where(
        ~rolling_source["unverified"].astype(bool)
    )

    for window_hours in rolling_windows:
        rolling_source[f"pedestrians_mean_{window_hours}h"] = (
            verified_counts
            .rolling(
                window=window_hours,
                min_periods=window_hours,
            )
            .mean()
        )

    rolling_columns = [
        f"pedestrians_mean_{window_hours}h"
        for window_hours in rolling_windows
    ]

    rolling_features = rolling_source[
        ["timestamp", *rolling_columns]
    ].copy()

    rolling_features["timestamp"] = (
        rolling_features["timestamp"]
        + pd.Timedelta(hours=end_lag_hours)
    )

    rolling_features = rolling_features.rename(
        columns={
            column: f"{column}_lag{end_lag_hours}h"
            for column in rolling_columns
        }
    )

    feature_dataframe = feature_dataframe.merge(
        rolling_features,
        on="timestamp",
        how="left",
    )

    return feature_dataframe




def select_feature_columns(
    dataframe: pd.DataFrame,
    include_target: bool = False,
) -> pd.DataFrame:
    """Keep only configured model features and required metadata columns."""

    model_columns = SETTINGS["features"]["model_columns"]

    selected_columns = [
        "timestamp",
        "target_timestamp",
        *model_columns,
    ]

    if include_target:
        selected_columns.extend(
            [
                "target_pedestrians",
                "target_unverified",
            ]
        )

    return dataframe[selected_columns].copy()




def build_base_features(
    dataframe: pd.DataFrame,
    weather_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build features shared by training and inference."""

    feature_dataframe = add_target_time_features(dataframe)

    feature_dataframe = add_pedestrian_lag_features(
        feature_dataframe
    )

    feature_dataframe = add_rolling_features(
        feature_dataframe
    )

    feature_dataframe = add_weather_features(
        feature_dataframe,
        weather_dataframe,
    )

    feature_dataframe = add_holiday_feature(
        feature_dataframe
    )

    feature_dataframe = add_event_features(
        feature_dataframe
    )

    validate_feature_data(
        feature_dataframe,
        require_target=False,
    )

    # Remove raw input columns and keep only model features.
    feature_dataframe = select_feature_columns(
        feature_dataframe,
        include_target=False,
    )

    return feature_dataframe




def build_training_features(
    dataframe: pd.DataFrame,
    weather_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Build the complete feature table including the training target."""

    feature_dataframe = build_base_features(
        dataframe,
        weather_dataframe,
    )

    # Build the target from the original pedestrian data.
    target_dataframe = add_target(
        dataframe
    )

    target_dataframe = target_dataframe[
        [
            "timestamp",
            "target_pedestrians",
            "target_unverified",
        ]
    ].copy()

    feature_dataframe = feature_dataframe.merge(
        target_dataframe,
        on="timestamp",
        how="left",
    )

    validate_feature_data(
        feature_dataframe,
        require_target=True,
    )

    feature_dataframe = select_feature_columns(
        feature_dataframe,
        include_target=True,
    )

    return feature_dataframe 




def validate_feature_data(
    dataframe: pd.DataFrame,
    require_target: bool = False,
) -> None:
    """Validate the structure of the engineered feature data."""

    model_columns = SETTINGS["features"]["model_columns"]

    required_columns = {
        "timestamp",
        "target_timestamp",
        *model_columns,
    }

    if require_target:
        required_columns.update(
            {
                "target_pedestrians",
                "target_unverified",
            }
        )

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {sorted(missing_columns)}"
        )

    duplicate_timestamps = dataframe["timestamp"].duplicated().sum()

    if duplicate_timestamps > 0:
        raise ValueError(
            f"Found {duplicate_timestamps} duplicate feature timestamps."
        )

    expected_target_timestamp = (
        dataframe["timestamp"]
        + pd.Timedelta(
            hours=SETTINGS["target"]["forecast_horizon_hours"]
        )
    )

    invalid_target_timestamps = (
        dataframe["target_timestamp"] != expected_target_timestamp
    )

    if invalid_target_timestamps.any():
        raise ValueError(
            "Found target timestamps that do not match the forecast horizon."
        )