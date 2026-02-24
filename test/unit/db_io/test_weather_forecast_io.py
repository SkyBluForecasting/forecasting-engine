"""Tests for WeatherForecastIO."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from forecasting_db.models import WeatherForecast, WeatherVariable
from forecasting_engine.db_io.weather_forecast_io import WeatherForecastIO


@pytest.fixture
def weather_forecast_io(in_memory_session):
    return WeatherForecastIO(in_memory_session)


@pytest.fixture
def forecast_factory(in_memory_session):
    """
    Factory to create weather forecast records.

    Usage:
      forecast_factory(
        weather_site_id="SITE1",
        timestamp=datetime(...),
        issue_time=datetime(...),
        variable=WeatherVariable.temp_air_c,
        value=22.5
      )
    """

    def _make(
        weather_site_id="SITE1",
        timestamp=None,
        issue_time=None,
        variable=WeatherVariable.temp_air_c,
        value=20.0,
    ):
        if timestamp is None:
            timestamp = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
        if issue_time is None:
            issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        forecast = WeatherForecast(
            weather_site_id=weather_site_id,
            timestamp=timestamp,
            issue_time=issue_time,
            variable=variable,
            value=value,
        )
        in_memory_session.add(forecast)
        in_memory_session.flush()
        return forecast

    return _make


# ----------------------------
# get_latest_issue_time
# ----------------------------


def test_get_latest_issue_time_returns_max_issue_time(
    weather_forecast_io, forecast_factory
):
    """Should return the maximum issue_time for a site/variable combo."""
    issue_time1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    issue_time2 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    issue_time3 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, issue_time1, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts, issue_time2, WeatherVariable.temp_air_c, 21.0)
    forecast_factory("SITE1", ts, issue_time3, WeatherVariable.temp_air_c, 22.0)

    result = weather_forecast_io.get_latest_issue_time(
        "SITE1", WeatherVariable.temp_air_c
    )

    assert result.replace(tzinfo=None) == issue_time3.replace(tzinfo=None)


def test_get_latest_issue_time_returns_none_if_no_data(weather_forecast_io):
    """Should return None if no forecasts exist for site/variable."""
    result = weather_forecast_io.get_latest_issue_time(
        "NONEXISTENT", WeatherVariable.temp_air_c
    )
    assert result is None


def test_get_latest_issue_time_different_variables(
    weather_forecast_io, forecast_factory
):
    """Should only return max issue_time for specified variable."""
    issue_time1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    issue_time2 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Add temp forecasts
    forecast_factory("SITE1", ts, issue_time1, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts, issue_time2, WeatherVariable.temp_air_c, 21.0)

    # Add wind forecasts at different times
    forecast_factory("SITE1", ts, issue_time1, WeatherVariable.wind_speed_ms, 5.0)

    temp_it = weather_forecast_io.get_latest_issue_time(
        "SITE1", WeatherVariable.temp_air_c
    )
    wind_it = weather_forecast_io.get_latest_issue_time(
        "SITE1", WeatherVariable.wind_speed_ms
    )

    assert temp_it.replace(tzinfo=None) == issue_time2.replace(tzinfo=None)
    assert wind_it.replace(tzinfo=None) == issue_time1.replace(tzinfo=None)


# ----------------------------
# get_latest_issue_times
# ----------------------------


def test_get_latest_issue_times_returns_all_variables(
    weather_forecast_io, forecast_factory
):
    """Should return max issue_time for each variable in one query."""
    it1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    it2 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    it3 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, it1, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts, it2, WeatherVariable.wind_speed_ms, 5.0)
    forecast_factory("SITE1", ts, it3, WeatherVariable.ghi_wm2, 500.0)

    result = weather_forecast_io.get_latest_issue_times("SITE1")

    assert len(result) == 3
    assert result[WeatherVariable.temp_air_c].replace(tzinfo=None) == it1.replace(
        tzinfo=None
    )
    assert result[WeatherVariable.wind_speed_ms].replace(tzinfo=None) == it2.replace(
        tzinfo=None
    )
    assert result[WeatherVariable.ghi_wm2].replace(tzinfo=None) == it3.replace(
        tzinfo=None
    )


def test_get_latest_issue_times_empty_site(weather_forecast_io):
    """Should return empty dict for site with no forecasts."""
    result = weather_forecast_io.get_latest_issue_times("NONEXISTENT")
    assert result == {}


def test_get_latest_issue_times_partial_variables(
    weather_forecast_io, forecast_factory
):
    """Should only include variables with data."""
    it1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, it1, WeatherVariable.temp_air_c, 20.0)

    result = weather_forecast_io.get_latest_issue_times("SITE1")

    assert len(result) == 1
    assert WeatherVariable.temp_air_c in result
    assert WeatherVariable.wind_speed_ms not in result


# ----------------------------
# forecasts_exist_for_range
# ----------------------------


def test_forecasts_exist_for_range_true(weather_forecast_io, forecast_factory):
    """Should return True if forecasts exist in range."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts1 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts1, issue_time, WeatherVariable.temp_air_c, 20.0)

    exists = weather_forecast_io.forecasts_exist_for_range(
        "SITE1",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        WeatherVariable.temp_air_c,
    )

    assert exists is True


def test_forecasts_exist_for_range_false(weather_forecast_io):
    """Should return False if no forecasts exist in range."""
    exists = weather_forecast_io.forecasts_exist_for_range(
        "NONEXISTENT",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        WeatherVariable.temp_air_c,
    )

    assert exists is False


def test_forecasts_exist_for_range_with_issue_time_filter(
    weather_forecast_io, forecast_factory
):
    """Should filter by issue_time when provided."""
    issue_time1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    issue_time2 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, issue_time1, WeatherVariable.temp_air_c, 20.0)

    exists_it1 = weather_forecast_io.forecasts_exist_for_range(
        "SITE1",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        WeatherVariable.temp_air_c,
        issue_time=issue_time1,
    )

    exists_it2 = weather_forecast_io.forecasts_exist_for_range(
        "SITE1",
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        WeatherVariable.temp_air_c,
        issue_time=issue_time2,
    )

    assert exists_it1 is True
    assert exists_it2 is False


# ----------------------------
# bulk_insert_forecasts
# ----------------------------


def test_bulk_insert_forecasts_inserts_all_records(weather_forecast_io):
    """Should insert all records from the list."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    forecasts = [
        {
            "weather_site_id": "SITE1",
            "timestamp": ts,
            "issue_time": issue_time,
            "variable": WeatherVariable.temp_air_c,
            "value": 20.0,
        },
        {
            "weather_site_id": "SITE1",
            "timestamp": ts,
            "issue_time": issue_time,
            "variable": WeatherVariable.wind_speed_ms,
            "value": 5.0,
        },
    ]

    count = weather_forecast_io.bulk_insert_forecasts(forecasts)

    assert count == 2
    rows = weather_forecast_io.session.query(WeatherForecast).all()
    assert len(rows) == 2


def test_bulk_insert_forecasts_empty_list_returns_zero(weather_forecast_io):
    """Should return 0 for empty input."""
    count = weather_forecast_io.bulk_insert_forecasts([])
    assert count == 0


def test_bulk_insert_forecasts_idempotent_on_duplicates(weather_forecast_io):
    """Should ignore duplicates (ON CONFLICT DO NOTHING)."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    forecast = {
        "weather_site_id": "SITE1",
        "timestamp": ts,
        "issue_time": issue_time,
        "variable": WeatherVariable.temp_air_c,
        "value": 20.0,
    }

    # Insert once
    count1 = weather_forecast_io.bulk_insert_forecasts([forecast])
    assert count1 == 1

    # Try to insert same forecast again
    count2 = weather_forecast_io.bulk_insert_forecasts([forecast])
    assert count2 == 0  # Should not insert duplicate

    rows = weather_forecast_io.session.query(WeatherForecast).all()
    assert len(rows) == 1


# ----------------------------
# get_forecasts_for_site
# ----------------------------


def test_get_forecasts_for_site_returns_dataframe(
    weather_forecast_io, forecast_factory
):
    """Should return DataFrame with all forecasts for site."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts1 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts1, issue_time, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts2, issue_time, WeatherVariable.temp_air_c, 21.0)

    df = weather_forecast_io.get_forecasts_for_site("SITE1")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert "variable" in df.columns
    assert "value" in df.columns
    assert "issue_time" in df.columns
    assert "weather_site_id" in df.columns


def test_get_forecasts_for_site_empty_returns_empty_dataframe(weather_forecast_io):
    """Should return empty DataFrame if no forecasts."""
    df = weather_forecast_io.get_forecasts_for_site("NONEXISTENT")
    assert df.empty


def test_get_forecasts_for_site_with_time_filter(weather_forecast_io, forecast_factory):
    """Should filter by start_time and end_time."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    ts3 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts1, issue_time, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts2, issue_time, WeatherVariable.temp_air_c, 21.0)
    forecast_factory("SITE1", ts3, issue_time, WeatherVariable.temp_air_c, 22.0)

    df = weather_forecast_io.get_forecasts_for_site(
        "SITE1",
        start_time=ts1,
        end_time=ts2,
    )

    assert len(df) == 2


def test_get_forecasts_for_site_with_variable_filter(
    weather_forecast_io, forecast_factory
):
    """Should filter by specific variables."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, issue_time, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts, issue_time, WeatherVariable.wind_speed_ms, 5.0)
    forecast_factory("SITE1", ts, issue_time, WeatherVariable.ghi_wm2, 500.0)

    df = weather_forecast_io.get_forecasts_for_site(
        "SITE1",
        variables=[WeatherVariable.temp_air_c, WeatherVariable.wind_speed_ms],
    )

    assert len(df) == 2
    assert set(df["variable"]) == {"temp_air_c", "wind_speed_ms"}


def test_get_forecasts_for_site_with_issue_time_filter(
    weather_forecast_io, forecast_factory
):
    """Should filter by specific issue_time."""
    issue_time1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    issue_time2 = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, issue_time1, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts, issue_time2, WeatherVariable.temp_air_c, 21.0)

    df = weather_forecast_io.get_forecasts_for_site(
        "SITE1",
        issue_time=issue_time1,
    )

    assert len(df) == 1
    assert df["value"].iloc[0] == 20.0


# ----------------------------
# from_df
# ----------------------------


def test_from_df_inserts_records(weather_forecast_io):
    """Should insert records from DataFrame."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1", "SITE1"],
            "timestamp": [ts, ts],
            "issue_time": [issue_time, issue_time],
            "variable": ["temp_air_c", "wind_speed_ms"],
            "value": [20.0, 5.0],
        }
    )

    count = weather_forecast_io.from_df(df)

    assert count == 2
    rows = weather_forecast_io.session.query(WeatherForecast).all()
    assert len(rows) == 2


def test_from_df_raises_on_missing_columns(weather_forecast_io):
    """Should raise if required columns are missing."""
    df = pd.DataFrame({"timestamp": [], "variable": []})

    with pytest.raises(ValueError, match="must contain columns"):
        weather_forecast_io.from_df(df)


def test_from_df_converts_string_variables(weather_forecast_io):
    """Should convert string variable names to enums."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1"],
            "timestamp": [ts],
            "issue_time": [issue_time],
            "variable": ["temp_air_c"],
            "value": [20.0],
        }
    )

    count = weather_forecast_io.from_df(df)

    assert count == 1
    row = weather_forecast_io.session.query(WeatherForecast).first()
    assert row.variable == WeatherVariable.temp_air_c


def test_from_df_converts_timestamps_to_utc(weather_forecast_io):
    """Should ensure timestamps are converted to UTC before storage."""
    issue_time_naive = datetime(2026, 1, 1, 0, 0)
    ts_naive = datetime(2026, 1, 1, 6, 0)

    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1"],
            "timestamp": [ts_naive],
            "issue_time": [issue_time_naive],
            "variable": ["temp_air_c"],
            "value": [20.0],
        }
    )

    count = weather_forecast_io.from_df(df)

    assert count == 1
    row = weather_forecast_io.session.query(WeatherForecast).first()
    # Verify the timestamps were stored
    assert row.timestamp == ts_naive
    assert row.issue_time == issue_time_naive


def test_from_df_handles_enum_variables_directly(weather_forecast_io):
    """Should handle variables that are already WeatherVariable enums."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1"],
            "timestamp": [ts],
            "issue_time": [issue_time],
            "variable": [WeatherVariable.temp_air_c],  # Pass enum directly, not string
            "value": [20.0],
        }
    )

    count = weather_forecast_io.from_df(df)

    assert count == 1
    row = weather_forecast_io.session.query(WeatherForecast).first()
    assert row.variable == WeatherVariable.temp_air_c


# ----------------------------
# to_df
# ----------------------------


def test_to_df_returns_all_forecasts(weather_forecast_io, forecast_factory):
    """Should return DataFrame with all forecasts."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, issue_time, WeatherVariable.temp_air_c, 20.0)
    forecast_factory("SITE1", ts, issue_time, WeatherVariable.wind_speed_ms, 5.0)

    df = weather_forecast_io.to_df()

    assert len(df) == 2
    assert set(df["weather_site_id"]) == {"SITE1"}
    assert set(df["variable"]) == {"temp_air_c", "wind_speed_ms"}


def test_to_df_empty_returns_empty_dataframe(weather_forecast_io):
    """Should return empty DataFrame if no forecasts."""
    df = weather_forecast_io.to_df()
    assert df.empty


def test_to_df_includes_all_columns(weather_forecast_io, forecast_factory):
    """Should include all required columns."""
    issue_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts = datetime(2026, 1, 1, 6, 0, tzinfo=timezone.utc)

    forecast_factory("SITE1", ts, issue_time, WeatherVariable.temp_air_c, 20.0)

    df = weather_forecast_io.to_df()

    expected_cols = {"weather_site_id", "timestamp", "variable", "value", "issue_time"}
    assert expected_cols.issubset(df.columns)


# ----------------------------
# _convert_variable
# ----------------------------
# Note: variable conversion is tested implicitly via from_df tests
# (test_from_df_converts_string_variables and test_from_df_handles_enum_variables_directly)
