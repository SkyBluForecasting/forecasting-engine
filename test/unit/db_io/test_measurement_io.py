import pytest
import pandas as pd
from forecasting_engine.db_io.measurement_io import MeasurementsIO
from forecasting_db.models import Measurement


# ----------------- to_df tests -----------------


def test_to_df_returns_pivoted_dataframe(in_memory_session):
    io = MeasurementsIO(in_memory_session)

    measurements = [
        Measurement(
            timestamp=pd.Timestamp("2025-01-01 00:00"),
            metric="load",
            value=10,
            asset_uuid="ASSET1",
        ),
        Measurement(
            timestamp=pd.Timestamp("2025-01-01 01:00"),
            metric="temp",
            value=5,
            asset_uuid="ASSET1",
        ),
        Measurement(
            timestamp=pd.Timestamp("2025-01-01 02:00"),
            metric="load",
            value=20,
            asset_uuid="ASSET1",
        ),
    ]
    in_memory_session.add_all(measurements)
    in_memory_session.commit()

    df = io.to_df(asset_uuid="ASSET1")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "load", "temp"]
    assert pd.isna(df.loc[1, "load"])
    assert df.loc[0, "load"] == 10
    assert df.loc[2, "load"] == 20


def test_to_df_with_metrics_filter(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    in_memory_session.add(
        Measurement(
            timestamp=pd.Timestamp("2025-01-01 00:00"),
            metric="load",
            value=10,
            asset_uuid="ASSET1",
        )
    )
    in_memory_session.commit()

    df = io.to_df(asset_uuid="ASSET1", metrics=["load"])
    assert list(df.columns) == ["timestamp", "load"]
    assert df.shape[0] == 1


def test_to_df_raises_if_no_rows(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    with pytest.raises(ValueError, match="No measurements found for asset ASSET1"):
        io.to_df(asset_uuid="ASSET1")


# ----------------- from_df tests -----------------


def test_from_df_long_form(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="H"),
            "metric": ["load", "temp"],
            "value": [10, 20],
        }
    )
    io.from_df(df, asset_uuid="ASSET1")

    rows = in_memory_session.query(Measurement).filter_by(asset_uuid="ASSET1").all()
    assert len(rows) == 2
    assert set(r.metric for r in rows) == {"load", "temp"}


def test_from_df_wide_form(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="H"),
            "load": [1, 2],
            "temp": [10, 20],
        }
    )
    io.from_df(df, asset_uuid="ASSET1")

    rows = in_memory_session.query(Measurement).filter_by(asset_uuid="ASSET1").all()
    assert set(r.metric for r in rows) == {"load", "temp"}
    assert all(r.asset_uuid == "ASSET1" for r in rows)


# ----------------- resample_to_frequency tests -----------------


def test_resample_to_frequency_empty(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    df = pd.DataFrame()
    resampled = io.resample_to_frequency(df, freq="H")
    assert resampled.empty


def test_resample_to_frequency_interpolation(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="30min"),
            "load": [1, 3, 5],
            "temp": [10, 20, 30],
        }
    )

    resampled = io.resample_to_frequency(df, freq="H")
    expected_index = pd.date_range("2025-01-01", periods=2, freq="H", name="timestamp")
    pd.testing.assert_index_equal(resampled.index, expected_index, check_names=False)
    assert resampled.loc[pd.Timestamp("2025-01-01 00:00"), "load"] == 2.0
    assert resampled.loc[pd.Timestamp("2025-01-01 01:00"), "load"] == 5.0


def test_resample_to_frequency_raises_without_timestamp(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    df = pd.DataFrame({"load": [1, 2, 3]})
    with pytest.raises(
        ValueError,
        match="DataFrame must have either a DatetimeIndex or 'timestamp' column",
    ):
        io.resample_to_frequency(df, freq="H")


def test_resample_to_frequency_with_datetime_index(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    timestamps = pd.date_range("2025-01-01", periods=3, freq="30min")
    df = pd.DataFrame({"load": [1, 2, 3]}, index=timestamps)

    resampled = io.resample_to_frequency(df, freq="H")
    assert isinstance(resampled.index, pd.DatetimeIndex)
    assert len(resampled) > 0
    assert "load" in resampled.columns


# ----------------- get_latest_load_per_asset tests -----------------


def test_get_latest_load_per_asset_subset(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    io.from_df(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2, freq="H"),
                "load": [100, 200],
            }
        ),
        asset_uuid="A1",
    )
    io.from_df(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2, freq="H"),
                "load": [300, 400],
            }
        ),
        asset_uuid="A2",
    )

    result = io.get_latest_load_per_asset(asset_uuids=["A1"])
    assert set(result.keys()) == {"A1"}
    assert result["A1"] == (pd.Timestamp("2025-01-01 01:00"), 200)


def test_get_latest_load_per_asset_all(in_memory_session):
    io = MeasurementsIO(in_memory_session)
    io.from_df(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2, freq="H"),
                "load": [100, 200],
            }
        ),
        asset_uuid="A1",
    )
    io.from_df(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=2, freq="H"),
                "load": [300, 400],
            }
        ),
        asset_uuid="A2",
    )

    result = io.get_latest_load_per_asset(asset_uuids=None)
    assert set(result.keys()) == {"A1", "A2"}
    assert result["A1"] == (pd.Timestamp("2025-01-01 01:00"), 200)
    assert result["A2"] == (pd.Timestamp("2025-01-01 01:00"), 400)
