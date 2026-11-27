# Weather ingestion for the Zurich Footfall Forecast.
#Loads historical point-in-time and live weather forecasts from Open-Meteo,
#validates the hourly forecast data, and selects the weather for the target hour.

import pandas as pd
import requests
import warnings
import time

from footfall.common.config import SETTINGS
from footfall.common.http import get_with_retry


def load_historical_weather(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load archived 24-hour-ahead weather forecasts from Open-Meteo."""

    weather_settings = SETTINGS["weather_data"]

    parameters = {
        "latitude": weather_settings["latitude"],
        "longitude": weather_settings["longitude"],
        "hourly": ",".join(
            weather_settings["historical_variables"]
        ),
        "timezone": weather_settings["timezone"],
        "models": weather_settings["model"],
        "start_date": start_date,
        "end_date": end_date,
    }


    response = get_with_retry(
        weather_settings["historical_api_url"],
        params=parameters,
        timeout=30,
    )                

    #converts the API's JSON response into Python data.
    payload = response.json()                    

    dataframe = pd.DataFrame(payload["hourly"])

    dataframe["time"] = pd.to_datetime(
        dataframe["time"],
        utc=True,
    )

    dataframe = dataframe.rename(
        columns={
            "time": "timestamp",
            "temperature_2m_previous_day1": "forecast_temperature",
            "precipitation_previous_day1": "forecast_precipitation",
            "weather_code_previous_day1": "forecast_weather_code",
        }
    )

    dataframe = dataframe.sort_values(
        "timestamp"
    ).reset_index(drop=True)
    
    validate_weather_data(dataframe)

    return dataframe




def load_live_weather(
    required_timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Load live weather and require complete forecasts for needed timestamps."""

    weather_settings = SETTINGS["weather_data"]

    parameters = {
        "latitude": weather_settings["latitude"],
        "longitude": weather_settings["longitude"],
        "hourly": ",".join(
            weather_settings["live_variables"]
        ),
        "timezone": weather_settings["timezone"],
        "models": weather_settings["model"],
        "forecast_hours": 48,
    }

    weather_columns = [
        "forecast_temperature",
        "forecast_precipitation",
        "forecast_weather_code",
    ]

    max_attempts = 4

    for attempt in range(1, max_attempts + 1):
        response = get_with_retry(
            weather_settings["live_api_url"],
            params=parameters,
            timeout=30,
        )

        payload = response.json()

        dataframe = pd.DataFrame(
            payload["hourly"]
        )

        dataframe["time"] = pd.to_datetime(
            dataframe["time"],
            utc=True,
        )

        dataframe = dataframe.rename(
            columns={
                "time": "timestamp",
                "temperature_2m": "forecast_temperature",
                "precipitation": "forecast_precipitation",
                "weather_code": "forecast_weather_code",
            }
        )

        dataframe = dataframe.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        validate_weather_data(dataframe)

        required_weather = dataframe[
            dataframe["timestamp"].isin(
                required_timestamps
            )
        ].copy()

        missing_timestamps = required_timestamps.difference(
            required_weather["timestamp"]
        )

        missing_value_rows = required_weather[
            weather_columns
        ].isna().any(axis=1)

        incomplete_timestamps = required_weather.loc[
            missing_value_rows,
            "timestamp",
        ]

        problem_timestamps = (
            missing_timestamps
            .union(
                pd.DatetimeIndex(
                    incomplete_timestamps
                )
            )
        )

        if len(problem_timestamps) == 0:
            return dataframe

        if attempt == max_attempts:
            raise ValueError(
                "Required live weather is still incomplete after "
                f"{max_attempts} attempts. Problem timestamps: "
                f"{problem_timestamps.tolist()}"
            )

        delay_seconds = 15 * (2 ** (attempt - 1))

        warnings.warn(
            "Required live weather is incomplete. "
            f"Attempt {attempt}/{max_attempts}. "
            f"Problem timestamps: {problem_timestamps.tolist()}. "
            f"Retrying in {delay_seconds} seconds."
        )

        time.sleep(delay_seconds)

    raise RuntimeError(
        "Unexpected end of live weather retry loop."
    )




def validate_weather_data(dataframe: pd.DataFrame) -> None:
    """Validate the basic structure and hourly continuity of weather data."""

    required_columns = {
        "timestamp",
        "forecast_temperature",
        "forecast_precipitation",
        "forecast_weather_code",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required weather columns: {sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Weather dataframe is empty."
        )

    weather_columns = [
        "forecast_temperature",
        "forecast_precipitation",
        "forecast_weather_code",
    ]

    # Stop if the API returned timestamps but no usable weather forecasts.
    if dataframe[weather_columns].isna().all().all():
        raise ValueError(
            "Weather data contains no usable forecast values."
        )

    # Partial missing values are reported but do not stop the complete backfill.
    rows_with_missing_weather = (
        dataframe[weather_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    if rows_with_missing_weather > 0:
        warnings.warn(
            f"Found {rows_with_missing_weather} weather rows "
            "with missing forecast values.",
            UserWarning,
        )

    duplicate_timestamps = dataframe["timestamp"].duplicated().sum()

    if duplicate_timestamps > 0:
        raise ValueError(
            f"Found {duplicate_timestamps} duplicate weather timestamps."
        )

    time_differences = dataframe["timestamp"].diff().dropna()

    invalid_differences = time_differences[
        time_differences != pd.Timedelta(hours=1)
    ]

    if not invalid_differences.empty:
        raise ValueError(
            f"Found {len(invalid_differences)} non-hourly weather gaps."
        )




def select_target_weather(
    dataframe: pd.DataFrame,
    prediction_timestamp: pd.Timestamp,
) -> pd.DataFrame:
    """Select the weather forecast for the configured prediction horizon."""

    forecast_horizon = SETTINGS["target"]["forecast_horizon_hours"]

    prediction_timestamp = pd.Timestamp(prediction_timestamp)

    if prediction_timestamp.tzinfo is None:
        prediction_timestamp = prediction_timestamp.tz_localize("UTC")
    else:
        prediction_timestamp = prediction_timestamp.tz_convert("UTC")

    #round the timestamp down to the full hour
    prediction_timestamp = prediction_timestamp.floor("h")

    # calculate target-hour
    target_timestamp = prediction_timestamp + pd.Timedelta(
        hours=forecast_horizon
    )

    #give out the weahter value of this hour
    target_weather = dataframe[
        dataframe["timestamp"] == target_timestamp
    ].copy()

    if target_weather.empty:
        raise ValueError(
            f"No weather forecast found for {target_timestamp}."
        )

    weather_columns = [
        "forecast_temperature",
        "forecast_precipitation",
        "forecast_weather_code",
    ]

    if target_weather[weather_columns].isna().any().any():
        raise ValueError(
            f"Weather forecast contains missing values for {target_timestamp}."
        )

    return target_weather.reset_index(drop=True)
