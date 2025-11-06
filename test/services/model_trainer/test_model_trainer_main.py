import pytest
from unittest.mock import MagicMock
from forecasting_engine.services.model_trainer.main import run_training


@pytest.fixture
def mock_trainer():
    return MagicMock()


def test_run_training_with_asset_id(mock_trainer):
    """train_asset is called when asset_id is provided."""
    run_training(mock_trainer, "asset123")

    mock_trainer.train_asset.assert_called_once_with("ASSET123")
    mock_trainer.train_all_assets.assert_not_called()


def test_run_training_without_asset_id(mock_trainer):
    """train_all_assets is called when asset_id is None."""
    run_training(mock_trainer, None)

    mock_trainer.train_all_assets.assert_called_once()
    mock_trainer.train_asset.assert_not_called()
