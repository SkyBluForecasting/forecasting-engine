import sys
import pytest
from unittest.mock import MagicMock, patch
import importlib


# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def mock_models(monkeypatch):
    """Mock forecasting_db.models.Base only (keep SQLAlchemy real)."""
    mock_base = MagicMock()
    mock_base.metadata.create_all = MagicMock()
    monkeypatch.setitem(sys.modules, "forecasting_db.models", MagicMock(Base=mock_base))
    return mock_base


@pytest.fixture
def mock_sqlite_config(monkeypatch):
    """Mock DATABASE_URL to use SQLite (normal path)."""
    monkeypatch.setitem(
        sys.modules,
        "forecasting_engine.config",
        MagicMock(DATABASE_URL="sqlite:///:memory:"),
    )


@pytest.fixture
def mock_postgres_config(monkeypatch):
    """Mock DATABASE_URL to look like Postgres."""
    monkeypatch.setitem(
        sys.modules,
        "forecasting_engine.config",
        MagicMock(DATABASE_URL="postgresql://user:pass@localhost/db"),
    )


# ----------------------------
# Tests: SQLite / normal import
# ----------------------------


def test_session_module_initialization(mock_models, mock_sqlite_config):
    """Ensure session.py initializes engine, SessionLocal, and calls Base.metadata.create_all."""
    # Ensure clean re-import
    if "forecasting_engine.db_io.session" in sys.modules:
        del sys.modules["forecasting_engine.db_io.session"]

    module = importlib.import_module("forecasting_engine.db_io.session")

    assert hasattr(module, "engine")
    assert hasattr(module, "SessionLocal")
    assert hasattr(module, "Base")
    mock_models.metadata.create_all.assert_called_once_with(bind=module.engine)


def test_session_module_reimport_does_not_error(mock_models, mock_sqlite_config):
    """Ensure re-importing session.py does not raise errors."""
    # Force a clean import
    if "forecasting_engine.db_io.session" in sys.modules:
        del sys.modules["forecasting_engine.db_io.session"]

    module = importlib.import_module("forecasting_engine.db_io.session")

    assert hasattr(module, "engine")
    assert hasattr(module, "SessionLocal")
    assert hasattr(module, "Base")
    mock_models.metadata.create_all.assert_called_once_with(bind=module.engine)


# ----------------------------
# Tests: Postgres schema creation
# ----------------------------


def test_postgres_schema_creation(mock_postgres_config, mock_models):
    """Test that schema creation SQL executes for Postgres."""
    # Patch create_engine to return a mock engine with a mock connection
    mock_conn = MagicMock()
    mock_engine = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn

    with patch("sqlalchemy.create_engine", return_value=mock_engine):
        # Force clean import
        if "forecasting_engine.db_io.session" in sys.modules:
            del sys.modules["forecasting_engine.db_io.session"]

        importlib.import_module("forecasting_engine.db_io.session")

    # Assert engine.connect() was used
    mock_engine.connect.assert_called_once()

    # Check that the correct SQL was executed
    executed_sql = [
        call[1][0].text for call in mock_conn.method_calls if call[0] == "execute"
    ]
    assert "CREATE SCHEMA IF NOT EXISTS openstef" in executed_sql[0]
    assert "CREATE SCHEMA IF NOT EXISTS product" in executed_sql[1]

    # Ensure commit was called
    mock_conn.commit.assert_called_once()

    # Ensure Base.metadata.create_all is still called
    mock_models.metadata.create_all.assert_called_once_with(bind=mock_engine)
