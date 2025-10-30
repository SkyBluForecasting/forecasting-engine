import sys
import pytest
from unittest.mock import MagicMock
import importlib


@pytest.fixture
def mock_models(monkeypatch):
    """Mock forecasting_db.models.Base only (keep SQLAlchemy real)."""
    mock_base = MagicMock()
    mock_base.metadata.create_all = MagicMock()
    monkeypatch.setitem(sys.modules, "forecasting_db.models", MagicMock(Base=mock_base))
    return mock_base


@pytest.fixture
def mock_config(monkeypatch):
    """Mock forecasting_engine.config.DATABASE_URL."""
    monkeypatch.setitem(
        sys.modules,
        "forecasting_engine.config",
        MagicMock(DATABASE_URL="sqlite:///:memory:"),
    )


def test_session_module_initialization(mock_models, mock_config):
    """Ensure session.py initializes engine, SessionLocal, and calls Base.metadata.create_all."""
    import importlib

    # Ensure clean re-import
    if "forecasting_engine.db_io.session" in sys.modules:
        del sys.modules["forecasting_engine.db_io.session"]

    module = importlib.import_module("forecasting_engine.db_io.session")

    # Assertions
    assert hasattr(module, "engine")
    assert hasattr(module, "SessionLocal")
    assert hasattr(module, "Base")
    mock_models.metadata.create_all.assert_called_once_with(bind=module.engine)


def test_session_module_reimport_does_not_error(monkeypatch):
    """Ensure re-importing session.py does not raise errors."""
    # Mock Base so create_all doesn't run real SQL
    mock_base = MagicMock()
    mock_base.metadata.create_all = MagicMock()

    # Patch both modules before import
    monkeypatch.setitem(sys.modules, "forecasting_db.models", MagicMock(Base=mock_base))
    monkeypatch.setitem(
        sys.modules,
        "forecasting_engine.config",
        MagicMock(DATABASE_URL="sqlite:///:memory:"),
    )

    # Force a clean import
    if "forecasting_engine.db_io.session" in sys.modules:
        del sys.modules["forecasting_engine.db_io.session"]

    module = importlib.import_module("forecasting_engine.db_io.session")

    assert hasattr(module, "engine")
    assert hasattr(module, "SessionLocal")
    assert hasattr(module, "Base")
    mock_base.metadata.create_all.assert_called_once_with(bind=module.engine)
