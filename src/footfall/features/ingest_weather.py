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

    return dataframe