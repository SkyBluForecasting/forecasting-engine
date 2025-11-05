import sys
import pytest
from unittest.mock import patch, MagicMock
import importlib


@pytest.fixture
def mock_trainer(monkeypatch):
    """Mock TrainingManager to avoid real training."""
    mock_trainer_class = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "forecasting_engine.shared.model_trainer",
        MagicMock(TrainingManager=mock_trainer_class),
    )
    return mock_trainer_class


@pytest.fixture
def mock_logger(monkeypatch):
    """Mock logger to avoid real logging."""
    mock_log = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "forecasting_engine.shared.logger_factory",
        MagicMock(get_logger=lambda name: mock_log),
    )
    return mock_log


# ----------------------------
# Helper to run main() with arguments
# ----------------------------
def run_main_with_args(args_list):
    with patch.object(sys, "argv", ["progname"] + args_list):
        # Re-import script to reset main()
        module_name = "forecasting_engine.services.model_trainer.main"
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = importlib.import_module(module_name)
        module.main()
        return module


# ----------------------------
# Tests
# ----------------------------
def test_main_with_asset_id(mock_trainer, mock_logger):
    """Passing --asset-id calls train_asset with uppercase asset_id."""
    mock_session_instance = MagicMock()
    sys.modules["forecasting_engine.db_io.session"].SessionLocal.return_value = (
        mock_session_instance
    )
    mock_trainer.return_value = MagicMock()

    asset_id = "asset123"
    run_main_with_args(["--asset-id", asset_id])

    # Assert TrainingManager initialized with session
    mock_trainer.assert_called_once_with(mock_session_instance)

    # Assert train_asset called with uppercased ID
    mock_trainer.return_value.train_asset.assert_called_once_with(asset_id.upper())
    mock_trainer.return_value.train_all_assets.assert_not_called()

    # Assert session.close called
    mock_session_instance.close.assert_called_once()


def test_main_without_asset_id(mock_trainer, mock_logger):
    """No --asset-id calls train_all_assets."""
    mock_session_instance = MagicMock()
    sys.modules["forecasting_engine.db_io.session"].SessionLocal.return_value = (
        mock_session_instance
    )
    mock_trainer.return_value = MagicMock()

    run_main_with_args([])

    mock_trainer.assert_called_once_with(mock_session_instance)
    mock_trainer.return_value.train_all_assets.assert_called_once()
    mock_trainer.return_value.train_asset.assert_not_called()

    mock_session_instance.close.assert_called_once()
