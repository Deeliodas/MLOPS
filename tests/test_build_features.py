# Tests for the feature engineering functions.
# Verifies target creation, lag and rolling features, calendar features,
# holidays, events, and correct target-time handling.


import pandas as pd
from footfall.common.config import SETTINGS

from footfall.features.build_features import (
    add_event_features,
    add_holiday_feature,
    add_pedestrian_lag_features,
    add_rolling_features,
    add_target,
    add_target_time_features,
    select_feature_columns,
)

def test_add_target() -> None:
    """Test that the target contains the observed count 24 hours later."""

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01 00:00:00",
                periods=30,
                freq="h",
                tz="UTC",
            ),
            "pedestrians_count": range(30),
            "unverified": False,
        }
    )

    result = add_target(dataframe)

    first_row = result.iloc[0]

    assert first_row["target_pedestrians"] == 24
    assert first_row["target_unverified"] == False





def test_add_pedestrian_lag_features() -> None:
    """Test that pedestrian lag features reference the correct past hours."""

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01 00:00:00",
                periods=200,
                freq="h",
                tz="UTC",
            ),
            "pedestrians_count": range(200),
            "unverified": False,
        }
    )

    result = add_pedestrian_lag_features(dataframe)

    row = result.iloc[168]

    assert row["pedestrians_lag_24h"] == 144
    assert row["pedestrians_lag_168h"] == 24
    
    
    
def test_lag_features_ignore_unverified_values() -> None:
    """Test that unverified pedestrian counts are excluded from lag features."""

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01 00:00:00",
                periods=200,
                freq="h",
                tz="UTC",
            ),
            "pedestrians_count": range(200),
            "unverified": False,
        }
    )

    # These two values would be used by row 168.
    dataframe.loc[24, "unverified"] = True
    dataframe.loc[144, "unverified"] = True

    result = add_pedestrian_lag_features(dataframe)

    row = result.iloc[168]

    assert pd.isna(row["pedestrians_lag_24h"])
    assert pd.isna(row["pedestrians_lag_168h"])    
    
    
    
    
def test_add_rolling_features() -> None:
    """Test that rolling features end 24 hours before prediction time."""

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01 00:00:00",
                periods=60,
                freq="h",
                tz="UTC",
            ),
            "pedestrians_count": range(60),
            "unverified": False,
        }
    )

    result = add_rolling_features(dataframe)

    row = result.iloc[50]

    assert row["pedestrians_mean_3h_lag24h"] == 25
    assert row["pedestrians_mean_6h_lag24h"] == 23.5
    assert row["pedestrians_mean_24h_lag24h"] == 14.5
    
    
    
    

def test_rolling_features_ignore_unverified_values() -> None:
    """Test that unverified pedestrian counts are excluded from rolling features."""

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start="2026-01-01 00:00:00",
                periods=60,
                freq="h",
                tz="UTC",
            ),
            "pedestrians_count": range(60),
            "unverified": False,
        }
    )

    dataframe.loc[25, "unverified"] = True

    result = add_rolling_features(dataframe)

    row = result.iloc[50]

    assert pd.isna(row["pedestrians_mean_3h_lag24h"])
    
    
    

def test_add_holiday_feature() -> None:
    """Test that Zurich public holidays are detected for the target date."""

    dataframe = pd.DataFrame(
        {
            "target_timestamp": pd.to_datetime(
                [
                    "2026-12-25 12:00:00+00:00",
                    "2026-12-26 12:00:00+00:00",
                    "2026-12-27 12:00:00+00:00",
                ]
            )
        }
    )

    result = add_holiday_feature(dataframe)

    assert result.loc[0, "target_is_holiday"] == 1
    assert result.loc[1, "target_is_holiday"] == 1
    assert result.loc[2, "target_is_holiday"] == 0


def test_add_event_features() -> None:
    """Test that configured Zurich event impact dates are detected."""

    dataframe = pd.DataFrame(
        {
            "target_timestamp": pd.to_datetime(
                [
                    "2026-08-08 12:00:00+00:00",
                    "2026-08-09 12:00:00+00:00",
                    "2026-04-20 12:00:00+00:00",
                    "2026-05-01 12:00:00+00:00",
                ]
            )
        }
    )

    result = add_event_features(dataframe)

    assert result.loc[0, "target_is_street_parade_impact"] == 1
    assert result.loc[1, "target_is_street_parade_impact"] == 1

    assert result.loc[2, "target_is_sechselaeuten"] == 1

    assert result.loc[3, "target_is_street_parade_impact"] == 0
    assert result.loc[3, "target_is_sechselaeuten"] == 0
    
    
    
    

def test_add_target_time_features() -> None:
    """Test that target time and local Zurich calendar features are correct."""

    dataframe = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-07-04 22:00:00+00:00",
                    "2026-07-05 22:00:00+00:00",
                ]
            )
        }
    )

    result = add_target_time_features(dataframe)

    first_row = result.iloc[0]
    second_row = result.iloc[1]

    assert first_row["target_timestamp"] == pd.Timestamp(
        "2026-07-05 22:00:00+00:00"
    )
    assert first_row["target_hour"] == 0
    assert first_row["target_day_of_week"] == 0
    assert first_row["target_is_weekend"] == 0
    assert first_row["target_month"] == 7

    assert second_row["target_timestamp"] == pd.Timestamp(
        "2026-07-06 22:00:00+00:00"
    )
    assert second_row["target_hour"] == 0
    assert second_row["target_day_of_week"] == 1
    assert second_row["target_is_weekend"] == 0
    assert second_row["target_month"] == 7
    
    


def test_live_and_training_feature_columns_match() -> None:
    """Test that live and training use the same model feature columns."""

    model_columns = SETTINGS["features"]["model_columns"]

    dataframe = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
            "target_timestamp": [pd.Timestamp("2026-01-02", tz="UTC")],
            **{
                column: [1]
                for column in model_columns
            },
            "target_pedestrians": [1000],
            "target_unverified": [False],
            "pedestrians_count": [999],
        }
    )

    live_features = select_feature_columns(
        dataframe,
        include_target=False,
    )

    training_features = select_feature_columns(
        dataframe,
        include_target=True,
    )

    expected_live_columns = {
        "timestamp",
        "target_timestamp",
        *model_columns,
    }

    expected_training_columns = {
        *expected_live_columns,
        "target_pedestrians",
        "target_unverified",
    }

    assert set(live_features.columns) == expected_live_columns
    assert set(training_features.columns) == expected_training_columns

    assert "pedestrians_count" not in live_features.columns
    assert "pedestrians_count" not in training_features.columns