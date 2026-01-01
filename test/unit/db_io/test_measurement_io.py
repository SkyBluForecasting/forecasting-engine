import pytest
from datetime import datetime
import pandas as pd
from unittest.mock import MagicMock
from forecasting_engine.db_io.measurement_io import MeasurementsIO


def test_to_df_returns_pivoted_dataframe(make_io, mock_session):
    """to_df should pivot long measurements to wide format."""
    io = make_io(MeasurementsIO)

    class MockRow:
        def __init__(self, timestamp, metric, value):
            self.timestamp = timestamp
            self.metric = metric
            self.value = value

    rows = [
        MockRow(pd.Timestamp("2025-01-01 00:00"), "load", 10),
        MockRow(pd.Timestamp("2025-01-01 01:00"), "temp", 5),
        MockRow(pd.Timestamp("2025-01-01 02:00"), "load", 20),
    ]
    mock_session.query.return_value.filter.return_value.all.return_value = rows

    df = io.to_df(asset_uuid="ASSET1")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["timestamp", "load", "temp"]
    # NaNs are expected in pivoted DataFrame
    assert pd.isna(df.loc[1, "load"])
    assert df.loc[0, "load"] == 10
    assert df.loc[2, "load"] == 20


def test_to_df_with_metrics_filter(make_io, mock_session):
    """to_df applies metrics filter if provided."""
    io = make_io(MeasurementsIO)

    class MockRow:
        def __init__(self, timestamp, metric, value):
            self.timestamp = timestamp
            self.metric = metric
            self.value = value

    rows = [
        MockRow(pd.Timestamp("2025-01-01"), "load", 10),
    ]
    mock_session.query.return_value.filter.return_value.filter.return_value.all.return_value = (
        rows
    )

    df = io.to_df(asset_uuid="ASSET1", metrics=["load"])
    assert list(df.columns) == ["timestamp", "load"]
    assert df.shape[0] == 1


def test_to_df_raises_if_no_rows(make_io, mock_session):
    """to_df should raise ValueError if no rows returned."""
    io = make_io(MeasurementsIO)
    mock_session.query.return_value.filter.return_value.all.return_value = []

    with pytest.raises(ValueError, match="No measurements found for asset ASSET1"):
        io.to_df(asset_uuid="ASSET1")


def test_from_df_long_form(make_io, mock_session, measurement_df):
    """from_df should insert long-form DataFrame correctly."""
    io = make_io(MeasurementsIO)
    io.from_df(measurement_df, asset_uuid="ASSET1")

    args, _ = mock_session.bulk_insert_mappings.call_args
    records = args[1] if len(args) > 1 else args[0]
    assert all(r["asset_uuid"] == "ASSET1" for r in records)
    assert set(r["metric"] for r in records) == {"load", "temp"}


def test_from_df_wide_form(make_io, mock_session):
    """from_df should melt wide-form DataFrame correctly."""
    io = make_io(MeasurementsIO)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="H"),
            "load": [1, 2],
            "temp": [10, 20],
        }
    )
    io.from_df(df, asset_uuid="ASSET1")

    args, _ = mock_session.bulk_insert_mappings.call_args
    records = args[1] if len(args) > 1 else args[0]
    assert set(r["metric"] for r in records) == {"load", "temp"}
    assert all(r["asset_uuid"] == "ASSET1" for r in records)


def test_resample_to_frequency_empty(make_io):
    """resample_to_frequency should return empty DataFrame if input is empty."""
    io = make_io(MeasurementsIO)
    df = pd.DataFrame()
    resampled = io.resample_to_frequency(df, freq="H")
    assert resampled.empty


def test_resample_to_frequency_with_timestamp_column(make_io):
    """resample_to_frequency should resample and interpolate correctly."""
    io = make_io(MeasurementsIO)
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
    # Interpolated values
    assert resampled.loc[pd.Timestamp("2025-01-01 00:00"), "load"] == 2.0
    assert resampled.loc[pd.Timestamp("2025-01-01 01:00"), "load"] == 5.0


def test_resample_to_frequency_raises_without_timestamp(make_io):
    """Should raise ValueError if DataFrame has no timestamp column and not DatetimeIndex."""
    io = make_io(MeasurementsIO)
    df = pd.DataFrame({"load": [1, 2, 3]})  # no timestamp column

    with pytest.raises(
        ValueError,
        match="DataFrame must have either a DatetimeIndex or 'timestamp' column",
    ):
        io.resample_to_frequency(df, freq="H")


def test_resample_to_frequency_with_non_datetime_index_and_timestamp_column(make_io):
    """Should handle case where index is not DatetimeIndex but timestamp column exists."""
    io = make_io(MeasurementsIO)

    # Create DataFrame with timestamp column but non-datetime index
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="30min"),
            "load": [1, 2, 3],
            "temp": [10, 20, 30],
        }
    )
    # Force a non-DatetimeIndex to trigger the set_index path
    df.index = [0, 1, 2]  # Integer index

    resampled = io.resample_to_frequency(df, freq="H")

    # Should successfully resample - this tests lines 76-82
    assert isinstance(resampled.index, pd.DatetimeIndex)
    assert len(resampled) > 0
    assert "load" in resampled.columns


def test_resample_to_frequency_exception_handling(make_io):
    """Should handle exceptions during resampling and wrap them appropriately."""
    io = make_io(MeasurementsIO)

    # Create a DataFrame that will cause an issue during resampling
    # Use invalid frequency to trigger exception path
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="30min"),
            "load": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError, match="Failed to resample data to frequency invalid_freq:"
    ):
        io.resample_to_frequency(df, freq="invalid_freq")


def test_resample_to_frequency_with_datetime_index(make_io):
    """Should handle case where DataFrame already has DatetimeIndex (skips the if condition)."""
    io = make_io(MeasurementsIO)

    # Create DataFrame with DatetimeIndex - this will make isinstance(df.index, pd.DatetimeIndex) True
    timestamps = pd.date_range("2025-01-01", periods=3, freq="30min")
    df = pd.DataFrame(
        {"load": [1, 2, 3], "temp": [10, 20, 30]}, index=timestamps
    )  # Set DatetimeIndex directly

    resampled = io.resample_to_frequency(df, freq="H")

    # Should successfully resample and skip the set_index() call
    assert isinstance(resampled.index, pd.DatetimeIndex)
    assert len(resampled) > 0
    assert "load" in resampled.columns


def _make_mock_subquery():
    """Helper to mock SQLAlchemy subquery object with .c attributes."""
    mock_subq = MagicMock()
    mock_subq.c.asset_uuid = "asset_uuid_col"
    mock_subq.c.latest_ts = "latest_ts_col"
    return mock_subq


def test_get_latest_load_per_asset_no_filter(make_io, mock_session):
    io = make_io(MeasurementsIO)

    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter = MagicMock(side_effect=lambda *args, **kwargs: mock_query)
    mock_subq = _make_mock_subquery()
    mock_query.group_by.return_value.subquery.return_value = mock_subq

    mock_session.query.return_value.join.return_value.all.return_value = [
        ("A1", datetime(2025, 1, 1, 12, 0), 100.0),
        ("A2", datetime(2025, 1, 1, 11, 0), 200.0),
    ]

    result = io.get_latest_load_per_asset()

    assert result == {
        "A1": (datetime(2025, 1, 1, 12, 0), 100.0),
        "A2": (datetime(2025, 1, 1, 11, 0), 200.0),
    }

    # Ensure first filter for metric was called
    first_filter_arg = mock_query.filter.call_args_list[0][0][0]
    assert "metric" in str(first_filter_arg)


def test_get_latest_load_per_asset_with_filter(make_io, mock_session):
    io = make_io(MeasurementsIO)

    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter = MagicMock(side_effect=lambda *args, **kwargs: mock_query)
    mock_subq = _make_mock_subquery()
    mock_query.group_by.return_value.subquery.return_value = mock_subq

    mock_session.query.return_value.join.return_value.all.return_value = [
        ("A1", datetime(2025, 1, 1, 10, 0), 123.4)
    ]

    result = io.get_latest_load_per_asset(["A1", "A2"])
    assert result == {"A1": (datetime(2025, 1, 1, 10, 0), 123.4)}

    # Ensure second filter for asset_uuids applied
    second_filter_arg = mock_query.filter.call_args_list[1][0][0]
    assert "asset_uuid" in str(second_filter_arg)


def test_get_latest_load_per_asset_empty_result(make_io, mock_session):
    """Should return empty dict if no rows found."""
    io = make_io(MeasurementsIO)

    mock_query = MagicMock()
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query
    mock_subq = _make_mock_subquery()
    mock_query.group_by.return_value.subquery.return_value = mock_subq

    mock_session.query.return_value.join.return_value.all.return_value = []

    result = io.get_latest_load_per_asset()
    assert result == {}
