"""Tests for WeatherActualIO."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from forecasting_db.models import WeatherActual, WeatherVariable
from forecasting_engine.db_io.weather_actual_io import WeatherActualIO


@pytest.fixture
def weather_actual_io(in_memory_session):
    return WeatherActualIO(in_memory_session)


@pytest.fixture
def reading_factory(in_memory_session):
    """
    Factory to create weather actual readings.

    Usage:
      reading_factory(
        weather_site_id="SITE1",
        timestamp=datetime(...),
        variable=WeatherVariable.temp_air_c,
        value=22.5
      )
    """

    def _make(
        weather_site_id="SITE1",
        timestamp=None,
        variable=WeatherVariable.temp_air_c,
        value=20.0,
    ):
        if timestamp is None:
            timestamp = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

        reading = WeatherActual(
            weather_site_id=weather_site_id,
            timestamp=timestamp,
            variable=variable,
            value=value,
        )
        in_memory_session.add(reading)
        in_memory_session.flush()
        return reading

    return _make


# ----------------------------
# get_latest_reading_timestamp
# ----------------------------


def test_get_latest_reading_timestamp_returns_max_timestamp(
    weather_actual_io, reading_factory
):
    """Should return the maximum timestamp for a site/variable combo."""
    ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    ts3 = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)

    reading_factory("SITE1", ts1, WeatherVariable.temp_air_c, 20.0)
    reading_factory("SITE1", ts2, WeatherVariable.temp_air_c, 21.0)
    reading_factory("SITE1", ts3, WeatherVariable.temp_air_c, 22.0)

    result = weather_actual_io.get_latest_reading_timestamp(
        "SITE1", WeatherVariable.temp_air_c
    )

    # SQLite doesn't preserve tzinfo, so compare without timezone
    assert result.replace(tzinfo=None) == ts3.replace(tzinfo=None)


def test_get_latest_reading_timestamp_returns_none_if_no_data(weather_actual_io):
    """Should return None if no readings exist for site/variable."""
    result = weather_actual_io.get_latest_reading_timestamp(
        "NONEXISTENT", WeatherVariable.temp_air_c
    )
    assert result is None


def test_get_latest_reading_timestamp_different_variables(
    weather_actual_io, reading_factory
):
    """Should only return max timestamp for specified variable."""
    ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)

    # Add temp readings
    reading_factory("SITE1", ts1, WeatherVariable.temp_air_c, 20.0)
    reading_factory("SITE1", ts2, WeatherVariable.temp_air_c, 21.0)

    # Add wind readings at different times
    reading_factory("SITE1", ts1, WeatherVariable.wind_speed_ms, 5.0)

    temp_ts = weather_actual_io.get_latest_reading_timestamp(
        "SITE1", WeatherVariable.temp_air_c
    )
    wind_ts = weather_actual_io.get_latest_reading_timestamp(
        "SITE1", WeatherVariable.wind_speed_ms
    )

    # SQLite doesn't preserve tzinfo, so compare without timezone
    assert temp_ts.replace(tzinfo=None) == ts2.replace(tzinfo=None)
    assert wind_ts.replace(tzinfo=None) == ts1.replace(tzinfo=None)


# ----------------------------
# get_latest_reading_timestamps
# ----------------------------


def test_get_latest_reading_timestamps_returns_all_variables(
    weather_actual_io, reading_factory
):
    """Should return max timestamp for each variable in one query."""
    ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    ts3 = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)

    reading_factory("SITE1", ts1, WeatherVariable.temp_air_c, 20.0)
    reading_factory("SITE1", ts2, WeatherVariable.wind_speed_ms, 5.0)
    reading_factory("SITE1", ts3, WeatherVariable.ghi_wm2, 500.0)

    result = weather_actual_io.get_latest_reading_timestamps("SITE1")

    assert len(result) == 3
    # SQLite doesn't preserve tzinfo, so compare without timezone
    assert result[WeatherVariable.temp_air_c].replace(tzinfo=None) == ts1.replace(
        tzinfo=None
    )
    assert result[WeatherVariable.wind_speed_ms].replace(tzinfo=None) == ts2.replace(
        tzinfo=None
    )
    assert result[WeatherVariable.ghi_wm2].replace(tzinfo=None) == ts3.replace(
        tzinfo=None
    )


def test_get_latest_reading_timestamps_empty_site(weather_actual_io):
    """Should return empty dict for site with no readings."""
    result = weather_actual_io.get_latest_reading_timestamps("NONEXISTENT")
    assert result == {}


def test_get_latest_reading_timestamps_partial_variables(
    weather_actual_io, reading_factory
):
    """Should only include variables with data."""
    ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    reading_factory("SITE1", ts1, WeatherVariable.temp_air_c, 20.0)

    result = weather_actual_io.get_latest_reading_timestamps("SITE1")

    assert len(result) == 1
    assert WeatherVariable.temp_air_c in result
    assert WeatherVariable.wind_speed_ms not in result


# ----------------------------
# bulk_insert_readings
# ----------------------------


def test_bulk_insert_readings_inserts_all_records(weather_actual_io):
    """Should insert all records from the list."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    readings = [
        {
            "weather_site_id": "SITE1",
            "timestamp": ts,
            "variable": WeatherVariable.temp_air_c,
            "value": 20.0,
        },
        {
            "weather_site_id": "SITE1",
            "timestamp": ts,
            "variable": WeatherVariable.wind_speed_ms,
            "value": 5.0,
        },
    ]

    count = weather_actual_io.bulk_insert_readings(readings)

    assert count == 2

    rows = weather_actual_io.session.query(WeatherActual).all()
    assert len(rows) == 2


def test_bulk_insert_readings_empty_list_returns_zero(weather_actual_io):
    """Should return 0 for empty input."""
    count = weather_actual_io.bulk_insert_readings([])
    assert count == 0


def test_bulk_insert_readings_idempotent_on_duplicates(weather_actual_io):
    """Should ignore duplicates (ON CONFLICT DO NOTHING)."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    reading = {
        "weather_site_id": "SITE1",
        "timestamp": ts,
        "variable": WeatherVariable.temp_air_c,
        "value": 20.0,
    }

    # Insert once
    count1 = weather_actual_io.bulk_insert_readings([reading])
    assert count1 == 1

    # Try to insert same reading again
    count2 = weather_actual_io.bulk_insert_readings([reading])
    assert count2 == 0  # Should not insert duplicate

    rows = weather_actual_io.session.query(WeatherActual).all()
    assert len(rows) == 1


# ----------------------------
# get_readings_for_site
# ----------------------------


def test_get_readings_for_site_returns_dataframe(weather_actual_io, reading_factory):
    """Should return DataFrame with all readings for site."""
    ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)

    reading_factory("SITE1", ts1, WeatherVariable.temp_air_c, 20.0)
    reading_factory("SITE1", ts2, WeatherVariable.temp_air_c, 21.0)

    df = weather_actual_io.get_readings_for_site("SITE1")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "timestamp" in df.columns
    assert "variable" in df.columns
    assert "value" in df.columns
    assert "weather_site_id" in df.columns


def test_get_readings_for_site_empty_returns_empty_dataframe(weather_actual_io):
    """Should return empty DataFrame if no readings."""
    df = weather_actual_io.get_readings_for_site("NONEXISTENT")
    assert df.empty


def test_get_readings_for_site_with_time_filter(weather_actual_io, reading_factory):
    """Should filter by start_time and end_time."""
    ts1 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    ts3 = datetime(2026, 1, 1, 2, 0, tzinfo=timezone.utc)

    reading_factory("SITE1", ts1, WeatherVariable.temp_air_c, 20.0)
    reading_factory("SITE1", ts2, WeatherVariable.temp_air_c, 21.0)
    reading_factory("SITE1", ts3, WeatherVariable.temp_air_c, 22.0)

    df = weather_actual_io.get_readings_for_site(
        "SITE1",
        start_time=ts1,
        end_time=ts2,
    )

    assert len(df) == 2
    # Convert pandas Timestamp to datetime for comparison
    assert pd.Timestamp(df["timestamp"].min()).to_pydatetime().replace(
        tzinfo=None
    ) == ts1.replace(tzinfo=None)
    assert pd.Timestamp(df["timestamp"].max()).to_pydatetime().replace(
        tzinfo=None
    ) == ts2.replace(tzinfo=None)


def test_get_readings_for_site_with_variable_filter(weather_actual_io, reading_factory):
    """Should filter by specific variables."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    reading_factory("SITE1", ts, WeatherVariable.temp_air_c, 20.0)
    reading_factory("SITE1", ts, WeatherVariable.wind_speed_ms, 5.0)
    reading_factory("SITE1", ts, WeatherVariable.ghi_wm2, 500.0)

    df = weather_actual_io.get_readings_for_site(
        "SITE1",
        variables=[WeatherVariable.temp_air_c, WeatherVariable.wind_speed_ms],
    )

    assert len(df) == 2
    assert set(df["variable"]) == {"temp_air_c", "wind_speed_ms"}


# ----------------------------
# from_df
# ----------------------------


def test_from_df_inserts_records(weather_actual_io):
    """Should insert records from DataFrame."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1", "SITE1"],
            "timestamp": [ts, ts],
            "variable": ["temp_air_c", "wind_speed_ms"],
            "value": [20.0, 5.0],
        }
    )

    count = weather_actual_io.from_df(df)

    assert count == 2
    rows = weather_actual_io.session.query(WeatherActual).all()
    assert len(rows) == 2


def test_from_df_raises_on_missing_columns(weather_actual_io):
    """Should raise if required columns are missing."""
    df = pd.DataFrame({"timestamp": [], "variable": []})

    with pytest.raises(ValueError, match="must contain columns"):
        weather_actual_io.from_df(df)


def test_from_df_converts_string_variables(weather_actual_io):
    """Should convert string variable names to enums."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1"],
            "timestamp": [ts],
            "variable": ["temp_air_c"],
            "value": [20.0],
        }
    )

    count = weather_actual_io.from_df(df)

    assert count == 1
    row = weather_actual_io.session.query(WeatherActual).first()
    assert row.variable == WeatherVariable.temp_air_c


def test_from_df_converts_timestamps_to_utc(weather_actual_io):
    """Should ensure timestamps are converted to UTC before storage."""
    ts_naive = datetime(2026, 1, 1, 0, 0)
    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1"],
            "timestamp": [ts_naive],
            "variable": ["temp_air_c"],
            "value": [20.0],
        }
    )

    count = weather_actual_io.from_df(df)

    assert count == 1
    row = weather_actual_io.session.query(WeatherActual).first()
    # Verify the timestamp was stored (SQLite doesn't preserve tzinfo, but data is in UTC)
    assert row.timestamp == ts_naive


def test_from_df_handles_enum_variables_directly(weather_actual_io):
    """Should handle variables that are already WeatherVariable enums."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "weather_site_id": ["SITE1"],
            "timestamp": [ts],
            "variable": [WeatherVariable.temp_air_c],  # Pass enum directly, not string
            "value": [20.0],
        }
    )

    count = weather_actual_io.from_df(df)

    assert count == 1
    row = weather_actual_io.session.query(WeatherActual).first()
    assert row.variable == WeatherVariable.temp_air_c


# ----------------------------
# to_df
# ----------------------------


def test_to_df_returns_all_readings(weather_actual_io, reading_factory):
    """Should return DataFrame with all readings."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    reading_factory("SITE1", ts, WeatherVariable.temp_air_c, 20.0)
    reading_factory("SITE1", ts, WeatherVariable.wind_speed_ms, 5.0)

    df = weather_actual_io.to_df()

    assert len(df) == 2
    assert set(df["weather_site_id"]) == {"SITE1"}
    assert set(df["variable"]) == {"temp_air_c", "wind_speed_ms"}


def test_to_df_empty_returns_empty_dataframe(weather_actual_io):
    """Should return empty DataFrame if no readings."""
    df = weather_actual_io.to_df()
    assert df.empty


def test_to_df_includes_all_columns(weather_actual_io, reading_factory):
    """Should include all required columns."""
    ts = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    reading_factory("SITE1", ts, WeatherVariable.temp_air_c, 20.0)

    df = weather_actual_io.to_df()

    expected_cols = {"weather_site_id", "timestamp", "variable", "value"}
    assert expected_cols.issubset(df.columns)


# ----------------------------
# _convert_variable
# ----------------------------
# Note: _convert_variable is tested implicitly via from_df tests
# (test_from_df_converts_string_variables and test_from_df_handles_enum_variables_directly)
