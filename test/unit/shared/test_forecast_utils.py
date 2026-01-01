import pytest
import pandas as pd
from unittest.mock import MagicMock
from forecasting_engine.shared.forecast_utils import (
    ForecastDataProcessor,
    normalize_forecast_columns,
    QUANTILE_MAP,
)

# --------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "load": [10, 20, 30],
        }
    )


@pytest.fixture
def mock_pj():
    pj = MagicMock()
    pj.resolution_minutes = 60
    pj.horizon_minutes = 180
    return pj


# --------------------------------------------------------------------
# ForecastDataProcessor.preprocess
# --------------------------------------------------------------------


def test_preprocess_with_timestamp_column(simple_df, mock_pj):
    processor = ForecastDataProcessor(simple_df, mock_pj)
    df_out = processor.preprocess()

    assert isinstance(df_out.index, pd.DatetimeIndex)

    tzinfo = df_out.index.tz
    assert tzinfo is not None
    assert tzinfo.utcoffset(None).total_seconds() == 0

    assert list(df_out["load"]) == [10, 20, 30]
    assert df_out.index.is_monotonic_increasing
    assert df_out.index.is_unique


def test_preprocess_with_datetime_index(simple_df, mock_pj):
    df = simple_df.set_index("timestamp")
    processor = ForecastDataProcessor(df, mock_pj)
    df_out = processor.preprocess()
    pd.testing.assert_index_equal(df_out.index, df.index)


def test_preprocess_missing_timestamp_raises(simple_df, mock_pj):
    df = simple_df.drop(columns=["timestamp"])
    processor = ForecastDataProcessor(df, mock_pj)
    with pytest.raises(ValueError, match="does not have a datetime index"):
        processor.preprocess()


def test_preprocess_deduplicates_and_sorts(mock_pj):
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 01:00", "2024-01-01 00:00", "2024-01-01 01:00"],
            "load": [20, 10, 99],
        }
    )
    processor = ForecastDataProcessor(df, mock_pj)
    df_out = processor.preprocess()
    assert list(df_out["load"]) == [10, 99]
    assert df_out.index[0] < df_out.index[1]


# --------------------------------------------------------------------
# normalize_forecast_columns
# --------------------------------------------------------------------


def test_forecast_already_exists():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "forecast": [10, 20],
            "p50": [99, 100],
        }
    )
    out = normalize_forecast_columns(df)
    # Line: df["forecast"] = df["p50"] should NOT run
    assert out["forecast"].equals(df["forecast"])  # unchanged


def test_forecast_missing_filled_from_p50():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "p50": [100, 200],
        }
    )
    df = df.drop(columns=["forecast"], errors="ignore")  # ensure missing
    out = normalize_forecast_columns(df)
    assert "forecast" in out.columns
    assert out["forecast"].equals(df["p50"])


def test_forecast_and_p50_missing_raises():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "load": [1, 2],
        }
    )
    with pytest.raises(ValueError, match="missing 'forecast' and 'p50' columns"):
        normalize_forecast_columns(df)


@pytest.mark.parametrize("ts_col", ["timestamp", "Timestamp", "ds", "  TimeStamp  "])
def test_timestamp_column_handling(ts_col):
    df = pd.DataFrame(
        {
            ts_col: pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "p50": [1, 2],
        }
    )
    out = normalize_forecast_columns(df)
    assert "timestamp" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])
    assert "forecast" in out.columns


def test_quantile_mapping_branch():
    # Use only first quantile for simplicity
    mlflow_col, db_col = next(iter(QUANTILE_MAP.items()))
    df = pd.DataFrame(
        {
            mlflow_col: [1],
            "p50": [2],
            "timestamp": pd.date_range("2024-01-01", periods=1, freq="h", tz="UTC"),
        }
    )
    out = normalize_forecast_columns(df)
    assert db_col in out.columns


def test_normalize_forecast_columns_index_to_timestamp():
    # DataFrame with datetime index but no timestamp column
    df = pd.DataFrame(
        {"p50": [1, 2]},
        index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
    )
    # 'timestamp' not in columns, so reset_index branch runs
    out = normalize_forecast_columns(df)
    assert "timestamp" in out.columns
    assert "forecast" in out.columns
    assert out["forecast"].equals(out["p50"])


def test_normalize_forecast_columns_missing_timestamp_raises():
    # No timestamp column and no recognized alias
    df = pd.DataFrame({"p50": [1, 2], "load": [10, 20]})
    with pytest.raises(ValueError, match="Forecast DataFrame missing timestamp column"):
        normalize_forecast_columns(df)


# --------------------------------------------------------------------
# ForecastDataProcessor.add_forecast_horizon_nans and helpers
# --------------------------------------------------------------------


def test_compute_horizon_times(mock_pj, simple_df):
    processor = ForecastDataProcessor(simple_df, mock_pj)
    horizon_times = processor._compute_horizon_times()
    expected_steps = int(mock_pj.horizon_minutes / mock_pj.resolution_minutes)
    assert len(horizon_times) == expected_steps
    assert horizon_times.freqstr in {
        f"{mock_pj.resolution_minutes}T",
        f"{mock_pj.resolution_minutes}min",
    }


def test_clip_and_null_horizon_with_existing_load(simple_df, mock_pj):
    df = simple_df.set_index("timestamp")
    processor = ForecastDataProcessor(df, mock_pj)

    now = pd.Timestamp.now(tz="UTC").floor(f"{mock_pj.resolution_minutes}min")
    horizon_times = pd.date_range(
        start=now, periods=3, freq=f"{mock_pj.resolution_minutes}min", tz="UTC"
    )
    df_out = processor._clip_and_null_horizon(df, horizon_times)

    assert df_out.loc[df_out.index.isin(horizon_times), "load"].isna().all()
    hist_mask = ~df_out.index.isin(horizon_times)
    if hist_mask.any():
        assert df_out.loc[hist_mask, "load"].notna().all()


def test_clip_and_null_horizon_adds_load_column_if_missing(mock_pj):
    df = pd.DataFrame(index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    processor = ForecastDataProcessor(df, mock_pj)
    horizon_times = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    df_out = processor._clip_and_null_horizon(df, horizon_times)
    assert "load" in df_out.columns
    assert df_out["load"].isna().all()


def test_fill_missing_horizon_rows_adds_missing(simple_df, mock_pj):
    df = simple_df.set_index("timestamp").iloc[:-1]
    processor = ForecastDataProcessor(df, mock_pj)
    now = df.index[-1]
    horizon_times = pd.date_range(
        start=now, periods=3, freq=f"{mock_pj.resolution_minutes}min", tz="UTC"
    )
    df_out = processor._fill_missing_horizon_rows(df, horizon_times)
    for ts in horizon_times:
        assert ts in df_out.index
    missing_ts = [t for t in horizon_times if t not in df.index]
    assert df_out.loc[missing_ts, "load"].isna().all()


def test_fill_missing_horizon_rows_no_missing(simple_df, mock_pj):
    df = simple_df.set_index("timestamp")
    processor = ForecastDataProcessor(df, mock_pj)
    horizon_times = df.index
    df_out = processor._fill_missing_horizon_rows(df, horizon_times)
    pd.testing.assert_frame_equal(df_out, df)


def test_add_forecast_horizon_nans_full(simple_df, mock_pj):
    df = simple_df.set_index("timestamp")
    processor = ForecastDataProcessor(df, mock_pj)
    df_out = processor.add_forecast_horizon_nans()
    expected_steps = int(mock_pj.horizon_minutes / mock_pj.resolution_minutes)
    future_rows = df_out.index[-expected_steps:]
    assert df_out.loc[future_rows, "load"].isna().all()
    historical_rows = df_out.index[:-expected_steps]
    if len(historical_rows) > 0:
        assert df_out.loc[historical_rows, "load"].notna().all()


def test_add_forecast_horizon_nans_missing_load_column(simple_df, mock_pj):
    df = simple_df.set_index("timestamp").drop(columns=["load"])
    processor = ForecastDataProcessor(df, mock_pj)
    df_out = processor.add_forecast_horizon_nans()
    assert "load" in df_out.columns
    expected_steps = int(mock_pj.horizon_minutes / mock_pj.resolution_minutes)
    future_rows = df_out.index[-expected_steps:]
    assert df_out.loc[future_rows, "load"].isna().all()
