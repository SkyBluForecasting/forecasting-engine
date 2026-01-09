import pytest
import pandas as pd
from forecasting_db.models import Forecast


# ----------------------------
# from_df tests
# ----------------------------


@pytest.fixture
def forecast_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "forecast": [10.0, 12.0, 15.0],
            "p05": [8, 9, 10],
            "p95": [14, 16, 19],
        }
    )


# ----------------------------
# from_df tests
# ----------------------------
def test_from_df_empty_df_raises_valueerror(forecast_io, in_memory_session):
    df = pd.DataFrame(columns=["timestamp", "forecast"])

    with pytest.raises(ValueError, match="Empty forecast DataFrame"):
        forecast_io.from_df(df, forecast_run_id="RUN1")

    # No rows written
    assert in_memory_session.query(Forecast).count() == 0


def test_from_df_missing_required_columns_raises(forecast_io, in_memory_session):
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3, freq="h")})

    with pytest.raises(ValueError, match="Missing required columns"):
        forecast_io.from_df(df, forecast_run_id="RUN1")

    assert in_memory_session.query(Forecast).count() == 0


def test_from_df_inserts_rows(forecast_io, in_memory_session, forecast_df):
    forecast_io.from_df(forecast_df, forecast_run_id="RUN1")

    rows = in_memory_session.query(Forecast).all()

    assert len(rows) == len(forecast_df)
    assert all(r.forecast_run_id == "RUN1" for r in rows)


def test_from_df_partial_columns_still_insert(forecast_io, in_memory_session):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "forecast": [10, 20],
        }
    )

    forecast_io.from_df(df, forecast_run_id="RUN1")

    rows = in_memory_session.query(Forecast).all()
    assert len(rows) == 2


def test_from_df_exception_rolls_back(
    forecast_io, in_memory_session, forecast_df, monkeypatch
):
    # Force DB failure
    monkeypatch.setattr(
        in_memory_session,
        "bulk_insert_mappings",
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("DB failure")),
    )

    with pytest.raises(RuntimeError, match="DB failure"):
        forecast_io.from_df(forecast_df, forecast_run_id="RUN1")

    assert in_memory_session.query(Forecast).count() == 0


# # ----------------------------
# # to_df tests
# # ----------------------------


def test_to_df_returns_dataframe(forecast_io, in_memory_session, forecast_df):
    forecast_io.from_df(forecast_df, forecast_run_id="RUN1")

    df = forecast_io.to_df(forecast_run_id="RUN1")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(["timestamp", "forecast", "p05", "p95", "forecast_run_id"]).issubset(
        df.columns
    )
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_to_df_empty_returns_empty_df(forecast_io):
    df = forecast_io.to_df(forecast_run_id="MISSING")
    assert df.empty


def test_to_df_filters_by_forecast_run_id(forecast_io, in_memory_session, forecast_df):
    forecast_io.from_df(forecast_df, forecast_run_id="RUN1")
    forecast_io.from_df(forecast_df, forecast_run_id="RUN2")

    df = forecast_io.to_df(forecast_run_id="RUN2")

    assert df["forecast_run_id"].nunique() == 1
    assert df["forecast_run_id"].iloc[0] == "RUN2"


def test_to_df_exception_reraises(forecast_io, in_memory_session, monkeypatch):
    # Monkeypatch session.query to raise an error
    def raise_error(*args, **kwargs):
        raise RuntimeError("DB failure")

    monkeypatch.setattr(forecast_io.session, "query", raise_error)

    # Only assert that the exception is raised
    with pytest.raises(RuntimeError, match="DB failure"):
        forecast_io.to_df(forecast_run_id="RUN1")
