import pandas as pd

from footfall.features.pipeline import (
    create_live_prediction_timestamps,
)


def test_create_live_prediction_timestamps() -> None:
    """Test that all target hours of the next local day are covered."""

    reference_timestamp = pd.Timestamp(
        "2026-11-22 18:00:00",
        tz="UTC",
    )

    prediction_timestamps = create_live_prediction_timestamps(
        reference_timestamp
    )

    assert len(prediction_timestamps) == 24

    target_timestamps = (
        prediction_timestamps
        + pd.Timedelta(hours=24)
    )

    target_timestamps_local = target_timestamps.tz_convert(
        "Europe/Zurich"
    )

    assert set(target_timestamps_local.hour) == set(range(24))

    assert all(
        target_timestamps_local.date
        == pd.Timestamp("2026-11-23").date()
    )