import pytest
import pandas as pd
from unittest.mock import MagicMock
from sqlalchemy.orm import Session


# ----------------------------
# Mock SQLAlchemy session fixture
# ----------------------------
@pytest.fixture
def mock_session():
    """Return a mock SQLAlchemy session with default behavior."""
    session = MagicMock(spec=Session)
    session.commit = MagicMock()
    session.rollback = MagicMock()
    return session


# ----------------------------
# Generic factory fixture for IO classes
# ----------------------------
@pytest.fixture
def make_io(mock_session):
    """
    Factory to create an IO instance with a mock session.

    Usage in tests:
        def test_something(make_io):
            io = make_io(ForecastRunIO)
    """

    def _make(io_class, session=None):
        return io_class(session or mock_session)

    return _make


# ----------------------------
# Sample test data fixtures
# ----------------------------


@pytest.fixture
def forecast_df():

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="H"),
            "forecast": [10, 20, 30],
            "p05": [5, 10, 15],
            "p95": [15, 25, 35],
            "description": ["desc"] * 3,
        }
    )


@pytest.fixture
def constraint_forecast_df():

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="H"),
            "forecast": [10, 20, 30],
        }
    )


@pytest.fixture
def measurement_df():

    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="H"),
            "metric": ["load", "temp", "load"],
            "value": [10, 5, 20],
        }
    )


@pytest.fixture
def mock_asset():

    asset = MagicMock()
    asset.asset_uuid = "ASSET1"
    asset.capacity_kw = 100
    return asset
