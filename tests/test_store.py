from pathlib import Path

import pandas as pd
import pytest

import footfall.features.store as store


def create_test_dataframe() -> pd.DataFrame:
    """Create a small feature dataframe for store tests."""

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00+00:00",
                    "2026-01-01 01:00:00+00:00",
                ]
            ),
            "pedestrians_lag_24h": [
                100,
                120,
            ],
            "forecast_temperature": [
                5.0,
                5.5,
            ],
        }
    )


def test_save_and_load_training_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test saving and loading a training feature snapshot."""

    monkeypatch.setattr(
        store,
        "ROOT_DIR",
        tmp_path,
    )

    dataframe = create_test_dataframe()

    file_path = store.save_feature_snapshot(
        dataframe,
        store_type="training",
    )

    assert file_path.exists()
    assert file_path.suffix == ".parquet"
    assert file_path.parent.name == "training"

    loaded_dataframe = store.load_latest_feature_snapshot(
        store_type="training",
    )

    pd.testing.assert_frame_equal(
        loaded_dataframe,
        dataframe,
    )


def test_save_and_load_live_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test saving and loading a live feature snapshot."""

    monkeypatch.setattr(
        store,
        "ROOT_DIR",
        tmp_path,
    )

    dataframe = create_test_dataframe()

    file_path = store.save_feature_snapshot(
        dataframe,
        store_type="live",
    )

    assert file_path.exists()
    assert file_path.suffix == ".parquet"
    assert file_path.parent.name == "live"

    loaded_dataframe = store.load_latest_feature_snapshot(
        store_type="live",
    )

    pd.testing.assert_frame_equal(
        loaded_dataframe,
        dataframe,
    )


def test_invalid_store_type_raises_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that an unknown feature store type raises an error."""

    monkeypatch.setattr(
        store,
        "ROOT_DIR",
        tmp_path,
    )

    dataframe = create_test_dataframe()

    with pytest.raises(ValueError):
        store.save_feature_snapshot(
            dataframe,
            store_type="invalid",
        )