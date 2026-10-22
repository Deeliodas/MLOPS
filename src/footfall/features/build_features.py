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
    feature_dataframe["target_day_of_week"] = local_target_time.dt.dayofweek

    feature_dataframe["target_is_weekend"] = (
        local_target_time.dt.dayofweek >= 5
    ).astype(int)

    feature_dataframe["target_month"] = local_target_time.dt.month

    return feature_dataframe




def add_pedestrian_lag_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add timestamp-based pedestrian lag features."""

    feature_dataframe = dataframe.copy()

    pedestrian_lags = SETTINGS["features"]["pedestrian_lags"]

    for lag_hours in pedestrian_lags:
        lagged_dataframe = dataframe[
            ["timestamp", "pedestrians_count"]
        ].copy()

        lagged_dataframe["timestamp"] = (
            lagged_dataframe["timestamp"]
            + pd.Timedelta(hours=lag_hours)
        )

        lagged_dataframe = lagged_dataframe.rename(
            columns={
                "pedestrians_count": f"pedestrians_lag_{lag_hours}h"
            }
        )

        feature_dataframe = feature_dataframe.merge(
            lagged_dataframe,
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