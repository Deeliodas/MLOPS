import pandas as pd
import requests

from footfall.common.config import SETTINGS


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


    response = requests.get(
        weather_settings["historical_api_url"],
        params=parameters,
        timeout=30,
    )

    response.raise_for_status()                  

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




def load_live_weather() -> pd.DataFrame:
    """Load the current weather forecast from Open-Meteo."""

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

    response = requests.get(
        weather_settings["live_api_url"],
        params=parameters,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    dataframe = pd.DataFrame(payload["hourly"])

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

    return dataframe




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

    target_timestamp = prediction_timestamp + pd.Timedelta(
        hours=forecast_horizon
    )

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




