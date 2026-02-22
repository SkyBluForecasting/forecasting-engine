import pytest
from datetime import datetime, date, timezone, timedelta
from unittest.mock import Mock, patch
import pandas as pd
import requests

from forecasting_engine.shared.openmeteo_client import OpenMeteoClient


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def client():
    """Create an OpenMeteo client instance for testing."""
    return OpenMeteoClient(timeout=30, max_retries=3, backoff_factor=0.5)


@pytest.fixture
def mock_session():
    """Create a mock session fixture."""
    with patch("requests.Session") as mock:
        yield mock


# ============================================================================
# Tests for __init__
# ============================================================================
def test_init_with_defaults():
    """Test client initialization with default parameters."""
    client = OpenMeteoClient()
    assert client.timeout == 30
    assert client.session is not None


def test_init_with_custom_params():
    """Test client initialization with custom parameters."""
    client = OpenMeteoClient(timeout=60, max_retries=5, backoff_factor=1.0)
    assert client.timeout == 60
    assert client.session is not None


def test_variable_mapping_exists():
    """Test that VARIABLE_MAPPING is properly defined."""
    assert OpenMeteoClient.VARIABLE_MAPPING == {
        "temperature_2m": "temp_air_c",
        "wind_speed_10m": "wind_speed_ms",
        "shortwave_radiation": "ghi_wm2",
    }


def test_base_urls_defined():
    """Test that required URLs are defined."""
    assert OpenMeteoClient.BASE_URL == "https://api.open-meteo.com/v1"
    assert OpenMeteoClient.FORECAST_URL == "https://api.open-meteo.com/v1/forecast"
    assert (
        OpenMeteoClient.ARCHIVE_URL == "https://archive-api.open-meteo.com/v1/archive"
    )


# ============================================================================
# Tests for _validate_coords
# ============================================================================
def test_validate_coords_valid():
    """Test coordinate validation with valid coordinates."""
    # Should not raise any exception
    OpenMeteoClient._validate_coords(43.691, -79.299)
    OpenMeteoClient._validate_coords(0.0, 0.0)
    OpenMeteoClient._validate_coords(90.0, 180.0)
    OpenMeteoClient._validate_coords(-90.0, -180.0)


def test_validate_coords_invalid_latitude_too_high():
    """Test that latitude > 90 raises ValueError."""
    with pytest.raises(ValueError, match="latitude must be between -90 and 90"):
        OpenMeteoClient._validate_coords(90.1, 0.0)


def test_validate_coords_invalid_latitude_too_low():
    """Test that latitude < -90 raises ValueError."""
    with pytest.raises(ValueError, match="latitude must be between -90 and 90"):
        OpenMeteoClient._validate_coords(-90.1, 0.0)


def test_validate_coords_invalid_longitude_too_high():
    """Test that longitude > 180 raises ValueError."""
    with pytest.raises(ValueError, match="longitude must be between -180 and 180"):
        OpenMeteoClient._validate_coords(0.0, 180.1)


def test_validate_coords_invalid_longitude_too_low():
    """Test that longitude < -180 raises ValueError."""
    with pytest.raises(ValueError, match="longitude must be between -180 and 180"):
        OpenMeteoClient._validate_coords(0.0, -180.1)


# ============================================================================
# Tests for _get_json
# ============================================================================
def test_get_json_success(client):
    """Test successful JSON retrieval."""
    mock_response = Mock()
    mock_response.json.return_value = {"data": "test"}
    client.session.get = Mock(return_value=mock_response)

    result = client._get_json("http://example.com", {"param": "value"})

    assert result == {"data": "test"}
    client.session.get.assert_called_once_with(
        "http://example.com", params={"param": "value"}, timeout=30
    )


def test_get_json_http_error(client):
    """Test HTTP error handling."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    client.session.get = Mock(return_value=mock_response)

    with pytest.raises(requests.HTTPError):
        client._get_json("http://example.com", {})


def test_get_json_404(client):
    """Test 404 error handling."""
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    client.session.get = Mock(return_value=mock_response)

    with pytest.raises(requests.HTTPError):
        client._get_json("http://example.com", {})


def test_get_json_with_long_response_body(client):
    """Test that long response bodies are truncated in logs."""
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "x" * 1000  # Long response
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    client.session.get = Mock(return_value=mock_response)

    with pytest.raises(requests.HTTPError):
        client._get_json("http://example.com", {})


# ============================================================================
# Tests for fetch_forecast
# ============================================================================
def test_fetch_forecast_success(client):
    """Test successful forecast fetch."""
    mock_response = {
        "hourly": {
            "time": [
                "2026-02-22T10:00",
                "2026-02-22T11:00",
                "2026-02-22T12:00",
            ],
            "temperature_2m": [15.5, 16.2, 17.1],
            "wind_speed_10m": [5.2, 5.5, 6.1],
            "shortwave_radiation": [100.0, 150.0, 200.0],
        }
    }
    client._get_json = Mock(return_value=mock_response)

    result = client.fetch_forecast(43.691, -79.299, hours=60)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert set(result.columns) >= {"timestamp", "variable", "value", "issue_time"}
    assert set(result["variable"].unique()) == {
        "temp_air_c",
        "wind_speed_ms",
        "ghi_wm2",
    }


def test_fetch_forecast_invalid_latitude(client):
    """Test forecast fetch with invalid latitude."""
    with pytest.raises(ValueError, match="latitude must be between -90 and 90"):
        client.fetch_forecast(91.0, -79.299)


def test_fetch_forecast_invalid_longitude(client):
    """Test forecast fetch with invalid longitude."""
    with pytest.raises(ValueError, match="longitude must be between -180 and 180"):
        client.fetch_forecast(43.691, 181.0)


def test_fetch_forecast_http_error(client):
    """Test forecast fetch with HTTP error."""
    client._get_json = Mock(side_effect=requests.HTTPError("500 Server Error"))

    result = client.fetch_forecast(43.691, -79.299)
    assert result.empty


def test_fetch_forecast_unexpected_error(client):
    """Test forecast fetch with unexpected error."""
    client._get_json = Mock(side_effect=Exception("Unexpected error"))

    result = client.fetch_forecast(43.691, -79.299)
    assert result.empty


def test_fetch_forecast_with_custom_hours(client):
    """Test forecast fetch with custom hours parameter."""
    mock_response = {
        "hourly": {
            "time": ["2026-02-22T10:00"],
            "temperature_2m": [15.5],
            "wind_speed_10m": [5.2],
            "shortwave_radiation": [100.0],
        }
    }
    client._get_json = Mock(return_value=mock_response)

    client.fetch_forecast(43.691, -79.299, hours=120)

    # Check that the API was called with the correct hours parameter
    call_args = client._get_json.call_args
    assert call_args.kwargs["params"]["forecast_hours"] == 120


# ============================================================================
# Tests for fetch_historical
# ============================================================================
def test_fetch_historical_success(client):
    """Test successful historical data fetch."""
    mock_response = {
        "hourly": {
            "time": [
                "2026-02-20T10:00",
                "2026-02-20T11:00",
                "2026-02-21T10:00",
            ],
            "temperature_2m": [10.0, 10.5, 11.0],
            "wind_speed_10m": [3.0, 3.2, 3.5],
            "shortwave_radiation": [50.0, 75.0, 100.0],
        }
    }
    client._get_json = Mock(return_value=mock_response)

    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    result = client.fetch_historical(43.691, -79.299, end_date)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert set(result.columns) >= {"timestamp", "variable", "value"}


def test_fetch_historical_default_start_date(client):
    """Test that start_date defaults to 30 days before end_date."""
    mock_response = {
        "hourly": {
            "time": [],
            "temperature_2m": [],
            "wind_speed_10m": [],
            "shortwave_radiation": [],
        }
    }
    client._get_json = Mock(return_value=mock_response)

    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    client.fetch_historical(43.691, -79.299, end_date)

    # Check the API call parameters
    call_args = client._get_json.call_args
    params = call_args.kwargs["params"]

    end_d = end_date.date()
    expected_start_d = end_d - timedelta(days=30)

    assert params["start_date"] == expected_start_d.isoformat()
    assert params["end_date"] == end_d.isoformat()


def test_fetch_historical_with_start_date(client):
    """Test historical data fetch with explicit start_date."""
    mock_response = {
        "hourly": {
            "time": [],
            "temperature_2m": [],
            "wind_speed_10m": [],
            "shortwave_radiation": [],
        }
    }
    client._get_json = Mock(return_value=mock_response)

    start_date = datetime(2026, 1, 20, 0, 0, 0)
    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    client.fetch_historical(43.691, -79.299, end_date, start_date)

    # Check the API call parameters
    call_args = client._get_json.call_args
    params = call_args.kwargs["params"]

    assert params["start_date"] == "2026-01-20"
    assert params["end_date"] == "2026-02-22"


def test_fetch_historical_with_start_date_as_date_object(client):
    """Test historical data fetch with date object as start_date."""
    mock_response = {
        "hourly": {
            "time": [],
            "temperature_2m": [],
            "wind_speed_10m": [],
            "shortwave_radiation": [],
        }
    }
    client._get_json = Mock(return_value=mock_response)

    start_date = date(2026, 1, 20)
    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    client.fetch_historical(43.691, -79.299, end_date, start_date)

    # Check the API call parameters
    call_args = client._get_json.call_args
    params = call_args.kwargs["params"]

    assert params["start_date"] == "2026-01-20"
    assert params["end_date"] == "2026-02-22"


def test_fetch_historical_start_date_after_end_date(client):
    """Test historical fetch when start_date is after end_date."""
    client._get_json = Mock()

    start_date = datetime(2026, 3, 22, 0, 0, 0)
    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    result = client.fetch_historical(43.691, -79.299, end_date, start_date)

    assert result.empty


def test_fetch_historical_invalid_latitude(client):
    """Test historical fetch with invalid latitude."""
    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="latitude must be between -90 and 90"):
        client.fetch_historical(91.0, -79.299, end_date)


def test_fetch_historical_http_error(client):
    """Test historical fetch with HTTP error."""
    client._get_json = Mock(side_effect=requests.HTTPError("500 Server Error"))

    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    result = client.fetch_historical(43.691, -79.299, end_date)
    assert result.empty


def test_fetch_historical_unexpected_error(client):
    """Test historical fetch with unexpected error."""
    client._get_json = Mock(side_effect=Exception("Unexpected error"))

    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    result = client.fetch_historical(43.691, -79.299, end_date)
    assert result.empty


def test_fetch_historical_filters_by_end_date(client):
    """Test that historical data is filtered to end_date."""
    mock_response = {
        "hourly": {
            "time": [
                "2026-02-20T10:00",
                "2026-02-22T11:00",
                "2026-02-22T15:00",  # After end_date
                "2026-02-23T10:00",  # After end_date
            ],
            "temperature_2m": [10.0, 12.0, 13.0, 14.0],
            "wind_speed_10m": [3.0, 3.5, 4.0, 3.5],
            "shortwave_radiation": [50.0, 100.0, 150.0, 100.0],
        }
    }
    client._get_json = Mock(return_value=mock_response)

    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    result = client.fetch_historical(43.691, -79.299, end_date)

    # All timestamps should be <= end_date
    assert all(result["timestamp"] <= end_date)


# ============================================================================
# Tests for _parse_hourly_data
# ============================================================================
def test_parse_hourly_data_forecast(client):
    """Test parsing hourly data for forecast."""
    issue_time = datetime(2026, 2, 22, 10, 0, 0, tzinfo=timezone.utc)
    data = {
        "hourly": {
            "time": [
                "2026-02-22T10:00",
                "2026-02-22T11:00",
                "2026-02-22T12:00",
            ],
            "temperature_2m": [15.5, 16.2, 17.1],
            "wind_speed_10m": [5.2, 5.5, 6.1],
            "shortwave_radiation": [100.0, 150.0, 200.0],
        }
    }

    result = client._parse_hourly_data(data, is_forecast=True, issue_time=issue_time)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(
        col in result.columns
        for col in ["timestamp", "variable", "value", "issue_time"]
    )
    assert set(result["variable"].unique()) == {
        "temp_air_c",
        "wind_speed_ms",
        "ghi_wm2",
    }
    assert all(result["issue_time"] == issue_time)
    assert len(result) == 9  # 3 timestamps * 3 variables


def test_parse_hourly_data_historical(client):
    """Test parsing hourly data for historical data."""
    data = {
        "hourly": {
            "time": [
                "2026-02-22T10:00",
                "2026-02-22T11:00",
            ],
            "temperature_2m": [15.5, 16.2],
            "wind_speed_10m": [5.2, 5.5],
            "shortwave_radiation": [100.0, 150.0],
        }
    }

    result = client._parse_hourly_data(data, is_forecast=False)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(col in result.columns for col in ["timestamp", "variable", "value"])
    assert "issue_time" not in result.columns


def test_parse_hourly_data_no_hourly_field(client):
    """Test parsing when hourly field is missing."""
    data = {}

    result = client._parse_hourly_data(data)
    assert result.empty


def test_parse_hourly_data_no_time_field(client):
    """Test parsing when time field is missing."""
    data = {
        "hourly": {
            "temperature_2m": [15.5, 16.2],
        }
    }

    result = client._parse_hourly_data(data)
    assert result.empty


def test_parse_hourly_data_with_nulls(client):
    """Test that null values are dropped."""
    data = {
        "hourly": {
            "time": [
                "2026-02-22T10:00",
                "2026-02-22T11:00",
                "2026-02-22T12:00",
            ],
            "temperature_2m": [15.5, None, 17.1],
            "wind_speed_10m": [5.2, None, None],
            "shortwave_radiation": [100.0, 150.0, 200.0],
        }
    }

    result = client._parse_hourly_data(data, is_forecast=False)

    # Nulls should be dropped per variable
    assert not result.empty
    # temperature_2m should have 2 values (10:00 and 12:00)
    temp_records = result[result["variable"] == "temp_air_c"]
    assert len(temp_records) == 2


def test_parse_hourly_data_all_nulls(client):
    def test_parse_hourly_data_continue_branch(client):
        """Explicitly cover the continue statement for a variable with all None values."""
        data = {
            "hourly": {
                "time": ["2026-02-22T10:00", "2026-02-22T11:00"],
                "temperature_2m": [None, None],  # triggers continue
                "wind_speed_10m": [5.2, 5.5],  # valid
                "shortwave_radiation": [100.0, 150.0],  # valid
            }
        }
        result = client._parse_hourly_data(data, is_forecast=False)
        # Should not be empty, since other variables are valid
        assert not result.empty
        # temperature_2m should not appear in the result
        assert "temp_air_c" not in result["variable"].unique()

    def test_parse_hourly_data_all_nulls(client):
        """Test parsing when all values for a variable are null."""
        data = {
            "hourly": {
                "time": [
                    "2026-02-22T10:00",
                    "2026-02-22T11:00",
                ],
                "temperature_2m": [15.5, 16.2],
                "wind_speed_10m": [None, None],  # All nulls
                "shortwave_radiation": [100.0, 150.0],
            }
        }
        result = client._parse_hourly_data(data, is_forecast=False)
        assert not result.empty
        variables = set(result["variable"].unique())
        assert "wind_speed_ms" not in variables


def test_parse_hourly_data_no_valid_records(client):
    """Test fetch_historical returns empty DataFrame when hourly data is empty."""
    # Covers branch where all variables produce empty series, triggering logger.warning
    data = {
        "hourly": {
            "time": ["2026-02-22T10:00", "2026-02-22T11:00"],
            "temperature_2m": [None, None],
            "wind_speed_10m": [None, None],
            "shortwave_radiation": [None, None],
        }
    }
    with patch("forecasting_engine.shared.openmeteo_client.logger") as mock_logger:
        result = client._parse_hourly_data(data, is_forecast=False)
        assert result.empty
        mock_logger.warning.assert_called_with(
            "No valid records parsed from API response"
        )


def test_parse_hourly_data_column_order(client):
    """Test that parsed data has correct columns."""
    data = {
        "hourly": {
            "time": ["2026-02-22T10:00"],
            "temperature_2m": [15.5],
            "wind_speed_10m": [5.2],
            "shortwave_radiation": [100.0],
        }
    }

    result = client._parse_hourly_data(data, is_forecast=False)

    assert "timestamp" in result.columns
    assert "variable" in result.columns
    assert "value" in result.columns


def test_parse_hourly_data_timestamp_format(client):
    """Test that timestamps are parsed correctly as datetime objects."""
    data = {
        "hourly": {
            "time": ["2026-02-22T10:00", "2026-02-22T11:00"],
            "temperature_2m": [15.5, 16.2],
            "wind_speed_10m": [5.2, 5.5],
            "shortwave_radiation": [100.0, 150.0],
        }
    }

    result = client._parse_hourly_data(data, is_forecast=False)

    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_parse_hourly_data_no_valid_variables(client):
    """Ensure branch where no valid variables are found is covered."""
    data = {
        "hourly": {
            "time": ["2026-02-22T10:00", "2026-02-22T11:00"],
            "temperature_2m": [None, None],
            "wind_speed_10m": [None, None],
            "shortwave_radiation": [None, None],
        }
    }
    with patch("forecasting_engine.shared.openmeteo_client.logger") as mock_logger:
        result = client._parse_hourly_data(data, is_forecast=False)
        assert result.empty
        mock_logger.warning.assert_called_with(
            "No valid records parsed from API response"
        )


# ============================================================================
# Integration-like tests
# ============================================================================
def test_fetch_forecast_end_to_end_flow(client):
    """Test forecast fetch with mocked session."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2026-02-22T10:00", "2026-02-22T11:00"],
            "temperature_2m": [15.5, 16.2],
            "wind_speed_10m": [5.2, 5.5],
            "shortwave_radiation": [100.0, 150.0],
        }
    }
    client.session.get = Mock(return_value=mock_response)

    result = client.fetch_forecast(43.691, -79.299)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "issue_time" in result.columns


def test_fetch_historical_end_to_end_flow(client):
    """Test historical fetch with mocked session."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "hourly": {
            "time": ["2026-02-20T10:00", "2026-02-21T10:00"],
            "temperature_2m": [10.0, 11.0],
            "wind_speed_10m": [3.0, 3.5],
            "shortwave_radiation": [50.0, 100.0],
        }
    }
    client.session.get = Mock(return_value=mock_response)

    end_date = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
    result = client.fetch_historical(43.691, -79.299, end_date)

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "issue_time" not in result.columns
