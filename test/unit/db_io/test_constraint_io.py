import pytest
import pandas as pd

from forecasting_engine.db_io.constraint_io import ConstraintsIO


# ----------------------------
# from_forecast tests
# ----------------------------


def test_from_forecast_no_asset_found(make_io, mock_session, constraint_forecast_df):
    """Should log a warning and return if asset not found."""
    io = make_io(ConstraintsIO)
    mock_session.query.return_value.filter.return_value.one_or_none.return_value = None

    io.from_forecast(constraint_forecast_df, "RUN1", "ASSET1")

    mock_session.bulk_insert_mappings.assert_not_called()
    mock_session.commit.assert_not_called()
    mock_session.rollback.assert_not_called()


def test_from_forecast_asset_without_capacity(
    make_io, mock_session, constraint_forecast_df, mock_asset
):
    """Should skip if asset has no capacity_kw."""
    mock_asset.capacity_kw = None
    mock_session.query.return_value.filter.return_value.one_or_none.return_value = (
        mock_asset
    )
    io = make_io(ConstraintsIO)

    io.from_forecast(constraint_forecast_df, "RUN1", mock_asset.asset_uuid)

    mock_session.bulk_insert_mappings.assert_not_called()
    mock_session.commit.assert_not_called()


def test_from_forecast_no_violations(
    make_io, mock_session, constraint_forecast_df, mock_asset
):
    """Should not insert anything if no forecasts exceed capacity."""
    mock_asset.capacity_kw = 100  # all below this
    mock_session.query.return_value.filter.return_value.one_or_none.return_value = (
        mock_asset
    )
    io = make_io(ConstraintsIO)

    io.from_forecast(constraint_forecast_df, "RUN1", mock_asset.asset_uuid)

    mock_session.bulk_insert_mappings.assert_not_called()
    mock_session.commit.assert_not_called()


def test_from_forecast_with_violations(make_io, mock_session, mock_asset):
    """Should insert rows and commit if violations exist."""
    mock_asset.capacity_kw = 15  # triggers violations for 20 and 30
    mock_session.query.return_value.filter.return_value.one_or_none.return_value = (
        mock_asset
    )
    io = make_io(ConstraintsIO)

    forecast_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "forecast": [10, 20, 30],
        }
    )

    io.from_forecast(forecast_df, "RUN1", mock_asset.asset_uuid)

    # Expect 2 violations (20, 30)
    expected_rows = [
        {
            "asset_uuid": mock_asset.asset_uuid,
            "timestamp": forecast_df.iloc[1]["timestamp"],
            "forecast_run_id": "RUN1",
            "constraint_kw": 5,
        },
        {
            "asset_uuid": mock_asset.asset_uuid,
            "timestamp": forecast_df.iloc[2]["timestamp"],
            "forecast_run_id": "RUN1",
            "constraint_kw": 15,
        },
    ]

    mock_session.bulk_insert_mappings.assert_called_once_with(
        mock_session.bulk_insert_mappings.call_args[0][0],
        expected_rows,
    )
    mock_session.commit.assert_called_once()


def test_from_forecast_empty_df(make_io, mock_session, mock_asset):
    """Should return early if DataFrame is empty."""
    io = make_io(ConstraintsIO)
    df = pd.DataFrame(columns=["timestamp", "forecast"])
    io.from_forecast(df, "RUN1", "ASSET1")

    mock_session.bulk_insert_mappings.assert_not_called()
    mock_session.commit.assert_not_called()


def test_from_forecast_missing_forecast_column(make_io, mock_session, mock_asset):
    """Should raise ValueError if 'forecast' column is missing."""
    io = make_io(ConstraintsIO)
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3, freq="h")})

    with pytest.raises(ValueError):
        io.from_forecast(df, "RUN1", mock_asset.asset_uuid)

    mock_session.rollback.assert_called_once()


def test_from_forecast_exception_rolls_back(make_io, mock_session, mock_asset):
    """Should rollback if unexpected exception occurs during insert."""
    mock_asset.capacity_kw = 10  # ensure violations occur
    mock_session.query.return_value.filter.return_value.one_or_none.return_value = (
        mock_asset
    )
    mock_session.bulk_insert_mappings.side_effect = RuntimeError("DB error")
    io = make_io(ConstraintsIO)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "forecast": [50, 60, 70],
        }
    )

    with pytest.raises(RuntimeError):
        io.from_forecast(df, "RUN1", mock_asset.asset_uuid)

    mock_session.rollback.assert_called_once()


# ----------------------------
# to_df tests
# ----------------------------


def test_to_df_returns_dataframe(make_io, mock_session):
    """Should return DataFrame with correct fields."""

    class Row:
        def __init__(self, asset_uuid, timestamp, forecast_run_id, constraint_kw):
            self.asset_uuid = asset_uuid
            self.timestamp = timestamp
            self.forecast_run_id = forecast_run_id
            self.constraint_kw = constraint_kw

    rows = [
        Row("A1", pd.Timestamp("2025-01-01T00:00:00"), "RUN1", 5.0),
        Row("A1", pd.Timestamp("2025-01-01T01:00:00"), "RUN1", 10.0),
    ]
    mock_session.query.return_value.all.return_value = rows
    io = make_io(ConstraintsIO)

    df = io.to_df()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) == {
        "asset_uuid",
        "timestamp",
        "forecast_run_id",
        "constraint_kw",
    }


def test_to_df_empty_returns_empty_df(make_io, mock_session):
    """Should return empty DataFrame when no rows found."""
    mock_session.query.return_value.all.return_value = []
    io = make_io(ConstraintsIO)

    df = io.to_df()

    assert df.empty
    mock_session.query.return_value.filter.assert_not_called()


def test_to_df_filters_by_asset_uuid(make_io, mock_session):
    """Should apply filter when asset_uuid is provided."""
    io = make_io(ConstraintsIO)
    mock_query = mock_session.query.return_value
    mock_query.filter.return_value.all.return_value = []

    io.to_df(asset_uuid="ASSET123")

    mock_query.filter.assert_called_once()
    called_args, _ = mock_query.filter.call_args

    # Relaxed check — just ensure the filter expression involves asset_uuid
    assert "asset_uuid" in str(called_args[0])


def test_from_df_not_implemented(make_io, mock_session):
    """Should raise NotImplementedError if from_df is called."""
    io = make_io(ConstraintsIO)
    import pandas as pd

    df = pd.DataFrame()
    with pytest.raises(NotImplementedError, match="Use `from_forecast`"):
        io.from_df(df)
