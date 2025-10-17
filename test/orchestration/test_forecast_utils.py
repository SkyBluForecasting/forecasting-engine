import pytest
import pandas as pd
from datetime import timezone
from unittest.mock import MagicMock
from forecasting_engine.orchestration.forecast_utils import (
    ForecastDataProcessor,
    normalize_forecast_columns,
    QUANTILE_MAP,
)


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
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
    assert list(df_out["load"]) == [10, 20, 30]
    # Should be sorted and unique
    assert df_out.index.is_monotonic_increasing
    assert df_out.index.is_unique


def test_preprocess_with_datetime_index(simple_df, mock_pj):
    df = simple_df.set_index("timestamp")
    processor = ForecastDataProcessor(df, mock_pj)
    df_out = processor.preprocess()

    # Should return identical index if already datetime
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

    # Should sort ascending and keep last duplicate (99)
    assert list(df_out["load"]) == [10, 99]
    assert df_out.index[0] < df_out.index[1]


# ------------------------------------- -------------------------------
# ForecastDataProcessor.add_forecast_horizon_nans
# --------------------------------------------------------------------


def test_add_forecast_horizon_nans_appends_future(simple_df, mock_pj):
    processor = ForecastDataProcessor(simple_df, mock_pj)
    df_out = processor.add_forecast_horizon_nans()

    # Horizon: 3 hours (horizon_minutes / resolution_minutes)
    expected_steps = int(mock_pj.horizon_minutes / mock_pj.resolution_minutes)

    assert len(df_out) >= len(simple_df)
    assert df_out.index[-expected_steps:].is_monotonic_increasing
    # Future section should be NaNs
    assert df_out.tail(expected_steps)["load"].isna().all()


def test_add_forecast_horizon_nans_filters_future_data(simple_df, mock_pj):
    # Push timestamps into the future
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range(pd.Timestamp.utcnow(), periods=3, freq="H"),
            "load": [1, 2, 3],
        }
    )
    processor = ForecastDataProcessor(df, mock_pj)
    df_out = processor.add_forecast_horizon_nans()

    # All original rows should be filtered out (only horizon rows remain)
    assert df_out["load"].isna().all()


def test_add_forecast_horizon_nans_without_load(mock_pj):
    mock_pj.resolution_minutes = 15
    mock_pj.horizon_minutes = 60

    # Historical dataframe WITHOUT 'load' column
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC"),
            "other_col": [1, 2, 3, 4],
        }
    )

    processor = ForecastDataProcessor(df, mock_pj)
    combined = processor.add_forecast_horizon_nans()

    # Ensure 'load' was NOT added
    assert "load" not in combined.columns
    # Horizon rows should exist for future timestamps
    now = (
        pd.Timestamp.utcnow()
        .floor(f"{mock_pj.resolution_minutes}min")
        .replace(tzinfo=timezone.utc)
    )

    future_rows = combined.index >= now
    assert len(combined.loc[future_rows]) == int(
        mock_pj.horizon_minutes / mock_pj.resolution_minutes
    )


# def test_add_forecast_horizon_nans_load_branch(mock_pj):
#     # Mock prediction job
#     mock_pj.resolution_minutes = 15
#     mock_pj.horizon_minutes = 60

#     # Historical dataframe with 'load'
#     df = pd.DataFrame({
#         "timestamp": pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC"),
#         "load": [10, 20, 30, 40],
#     })

#     processor = ForecastDataProcessor(df, mock_pj)
#     combined = processor.add_forecast_horizon_nans()

#     # Make 'now' consistent with processor logic
#     now = pd.Timestamp.utcnow().floor(f"{mock_pj.resolution_minutes}min").tz_convert("UTC") if pd.Timestamp.utcnow().tzinfo else pd.Timestamp.utcnow().floor(f"{mock_pj.resolution_minutes}min").tz_localize("UTC")
#     future_rows = combined.index >= now

#     # Check horizon rows exist and have 'load' column with NaN values
#     assert "load" in combined.columns
#     assert combined.loc[future_rows, "load"].isna().all()

#     # Ensure historical rows are preserved correctly
#     historical_rows = combined.index < now
#     assert combined.loc[historical_rows, "load"].notna().all()


# --------------------------------------------------------------------
# normalize_forecast_columns
# --------------------------------------------------------------------


def test_normalize_forecast_columns_from_index():
    df = pd.DataFrame(
        {
            "p50": [10, 20],
            "p05": [5, 10],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="H"),
    )

    out = normalize_forecast_columns(df)
    assert "timestamp" in out.columns
    assert "forecast" in out.columns
    assert out["forecast"].equals(out["p50"])
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])


def test_normalize_forecast_columns_maps_quantiles():
    df = pd.DataFrame(
        {
            "quantile_p05": [1],
            "quantile_p50": [2],
            "quantile_p95": [3],
            "timestamp": ["2024-01-01 00:00"],
        }
    )

    out = normalize_forecast_columns(df)
    for mlflow_col, db_col in QUANTILE_MAP.items():
        if mlflow_col in df.columns:
            assert db_col in out.columns


def test_normalize_forecast_columns_with_ds_alias():
    df = pd.DataFrame(
        {
            "ds": ["2024-01-01 00:00"],
            "p50": [10],
        }
    )

    out = normalize_forecast_columns(df)
    assert "timestamp" in out.columns
    assert out["forecast"].iloc[0] == 10


def test_normalize_forecast_columns_missing_timestamp_column():
    df = pd.DataFrame({"p50": [1, 2]})
    with pytest.raises(ValueError, match="missing timestamp"):
        normalize_forecast_columns(df)


def test_normalize_forecast_columns_missing_forecast_and_p50():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00"],
            "load": [100],
        }
    )
    with pytest.raises(ValueError, match="missing 'forecast' and 'p50'"):
        normalize_forecast_columns(df)


def test_normalize_forecast_columns_lowercase_and_strip():
    df = pd.DataFrame(
        {
            " TimeStamp  ": ["2024-01-01 00:00"],
            " Quantile_P50 ": [100],
        }
    )

    out = normalize_forecast_columns(df)
    assert "timestamp" in out.columns
    assert "forecast" in out.columns
    assert out["forecast"].iloc[0] == 100
