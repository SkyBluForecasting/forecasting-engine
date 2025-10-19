import pytest
import pandas as pd
from forecasting_engine.db_io.forecast_io import ForecastIO


# ----------------------------
# from_df tests
# ----------------------------


def test_from_df_empty_df_raises_valueerror(make_io, mock_session):
    """Should raise ValueError if forecast_df is empty."""
    io = make_io(ForecastIO)
    df = pd.DataFrame(columns=["timestamp", "forecast"])
    with pytest.raises(ValueError, match="Empty forecast DataFrame"):
        io.from_df(df, forecast_run_id=1)
    mock_session.rollback.assert_called_once()


def test_from_df_missing_required_columns_raises(make_io, mock_session):
    """Should raise ValueError if required columns are missing."""
    io = make_io(ForecastIO)
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3, freq="h")})
    with pytest.raises(ValueError, match="Missing required columns"):
        io.from_df(df, forecast_run_id=1)
    mock_session.rollback.assert_called_once()


def test_from_df_inserts_and_commits(make_io, mock_session, forecast_df):
    """Should insert forecast rows and commit."""
    io = make_io(ForecastIO)
    io.from_df(forecast_df, forecast_run_id=1)

    # Expect one bulk insert with 3 mappings
    args, kwargs = mock_session.bulk_insert_mappings.call_args
    called_model, called_rows = args
    assert len(called_rows) == len(forecast_df)
    assert all("forecast_value" in r for r in called_rows)
    mock_session.commit.assert_called_once()


def test_from_df_exception_rolls_back(make_io, mock_session, forecast_df):
    """Should rollback and re-raise on DB error."""
    io = make_io(ForecastIO)
    mock_session.bulk_insert_mappings.side_effect = RuntimeError("DB failure")
    with pytest.raises(RuntimeError, match="DB failure"):
        io.from_df(forecast_df, forecast_run_id=1)
    mock_session.rollback.assert_called_once()


# ----------------------------
# to_df tests
# ----------------------------


def test_to_df_returns_dataframe(make_io, mock_session):
    """Should return DataFrame with expected columns."""

    class Row:
        def __init__(self):
            self.timestamp = pd.Timestamp("2025-01-01T00:00:00Z")
            self.forecast_value = 10
            self.p05 = 5
            self.p10 = 7
            self.p30 = 9
            self.p50 = 10
            self.p70 = 11
            self.p90 = 13
            self.p95 = 15
            self.description = "test"

    rows = [Row(), Row()]
    mock_query = mock_session.query.return_value
    mock_query.filter.return_value.all.return_value = rows

    io = make_io(ForecastIO)
    df = io.to_df(forecast_run_id="RUN1")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(["timestamp", "forecast", "p05", "p95", "description"]).issubset(
        df.columns
    )
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    mock_query.filter.assert_called_once()


def test_to_df_returns_empty_df_when_no_rows(make_io, mock_session):
    """Should return empty DataFrame if no rows found."""
    mock_query = mock_session.query.return_value
    mock_query.filter.return_value.all.return_value = []

    io = make_io(ForecastIO)
    df = io.to_df(forecast_run_id="RUN1")

    assert df.empty


def test_to_df_exception_reraises(make_io, mock_session):
    """Should re-raise any exception and not swallow it."""
    mock_query = mock_session.query.return_value
    mock_query.filter.side_effect = RuntimeError("query failed")

    io = make_io(ForecastIO)
    with pytest.raises(RuntimeError, match="query failed"):
        io.to_df(forecast_run_id="RUN1")
