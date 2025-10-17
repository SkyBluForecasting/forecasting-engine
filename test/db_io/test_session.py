import sys
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_modules(monkeypatch):
    """Mock all external dependencies imported by forecasting_engine.db_io.session."""
    # Mock sqlalchemy imports
    mock_sqlalchemy = MagicMock()
    mock_engine = MagicMock(name="engine")
    mock_session_factory = MagicMock(name="SessionLocal")

    mock_sqlalchemy.create_engine.return_value = mock_engine
    mock_sqlalchemy.orm.sessionmaker.return_value = mock_session_factory

    monkeypatch.setitem(sys.modules, "sqlalchemy", mock_sqlalchemy)
    monkeypatch.setitem(sys.modules, "sqlalchemy.orm", mock_sqlalchemy.orm)

    # Mock forecasting_db.models.Base
    mock_base = MagicMock(name="Base")
    mock_base.metadata.create_all = MagicMock()
    mock_models = MagicMock(Base=mock_base)
    monkeypatch.setitem(sys.modules, "forecasting_db.models", mock_models)

    # Mock forecasting_engine.config
    mock_config = MagicMock(DATABASE_URL="sqlite:///:memory:")
    monkeypatch.setitem(sys.modules, "forecasting_engine.config", mock_config)

    yield {
        "mock_sqlalchemy": mock_sqlalchemy,
        "mock_engine": mock_engine,
        "mock_session_factory": mock_session_factory,
        "mock_base": mock_base,
        "mock_models": mock_models,
        "mock_config": mock_config,
    }


def test_session_module_initialization(mock_modules):
    """Ensure importing session.py initializes engine, sessionmaker, and Base metadata properly."""
    import importlib

    # Ensure clean re-import
    if "forecasting_engine.db_io.session" in sys.modules:
        del sys.modules["forecasting_engine.db_io.session"]

    module = importlib.import_module("forecasting_engine.db_io.session")

    mock_sqlalchemy = mock_modules["mock_sqlalchemy"]
    mock_base = mock_modules["mock_base"]
    mock_config = mock_modules["mock_config"]

    # Assertions
    mock_sqlalchemy.create_engine.assert_called_once_with(
        mock_config.DATABASE_URL, echo=False, future=True
    )
    mock_sqlalchemy.orm.sessionmaker.assert_called_once_with(
        bind=mock_modules["mock_engine"], autoflush=False, autocommit=False
    )
    mock_base.metadata.create_all.assert_called_once_with(
        bind=mock_modules["mock_engine"]
    )

    # Verify globals
    assert module.engine == mock_modules["mock_engine"]
    assert module.SessionLocal == mock_modules["mock_session_factory"]
    assert module.Base == mock_modules["mock_models"].Base


def test_session_module_reimport_does_not_error(mock_modules):
    """Ensure re-importing the module does not raise errors."""
    import importlib

    # First import
    import forecasting_engine.db_io.session as session

    # Force reimport
    importlib.reload(session)

    assert hasattr(session, "engine")
    assert hasattr(session, "SessionLocal")
    assert hasattr(session, "Base")
