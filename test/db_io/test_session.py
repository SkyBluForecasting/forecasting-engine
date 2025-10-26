import sys
import pytest
from unittest.mock import MagicMock


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


def test_session_module_reimport_does_not_error():
    """Ensure re-importing session.py does not raise errors."""
    import importlib
    import forecasting_engine.db_io.session as session

    importlib.reload(session)

    assert hasattr(session, "engine")
    assert hasattr(session, "SessionLocal")
    assert hasattr(session, "Base")
