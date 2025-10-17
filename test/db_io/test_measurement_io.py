import pytest
import pandas as pd
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
