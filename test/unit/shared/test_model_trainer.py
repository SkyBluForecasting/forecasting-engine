import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import forecasting_engine.shared.model_trainer as mt
from forecasting_engine.shared.model_trainer import (
    TrainingManager,
    HORIZON_MIN,
    RES_MIN,
)


# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture
def mock_session():
    from unittest.mock import MagicMock

    return MagicMock()


@pytest.fixture
def trainer(mock_session):
    """Return TrainingManager with MeasurementsIO and AssetsIO mocked."""
    mock_measurements_io = MagicMock()
    mock_assets_io = MagicMock()

    with patch(
        "forecasting_engine.shared.model_trainer.MeasurementsIO",
        return_value=mock_measurements_io,
    ), patch(
        "forecasting_engine.shared.model_trainer.AssetsIO", return_value=mock_assets_io
    ):
        tm = TrainingManager(mock_session)
        tm.measurements_io = mock_measurements_io
        tm.assets_io = mock_assets_io
        yield tm


# ----------------------------
# Test _slice_train_data
# ----------------------------


class TestSliceTrainData:
    def test_slices_correctly(self, trainer):
        # Must have >= test_len rows
        test_len = int(HORIZON_MIN // RES_MIN)
        df = pd.DataFrame({"load": range(test_len + 10)})
        result = trainer._slice_train_data(df)
        assert len(result) == len(df) - test_len

    def test_not_enough_rows_raises(self, trainer):
        df = pd.DataFrame({"load": range(10)})  # fewer than test_len
        with pytest.raises(ValueError):
            trainer._slice_train_data(df)


# ----------------------------
# Test _prepare_train_df
# ----------------------------


class TestPrepareTrainDf:
    def test_prepare_train_df_converts_and_renames(self, trainer):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
                "load_kW": [1, 2, 3],
                "temp_C": [10, 20, 30],
            }
        )
        result = trainer._prepare_train_df(df)
        assert "load" in result.columns
        assert "temp" in result.columns
        assert result.index.name == "timestamp"
        assert pd.api.types.is_datetime64tz_dtype(
            result.index
        ) or pd.api.types.is_datetime64_dtype(result.index)


# ----------------------------
# Test train_asset
# ----------------------------


class TestTrainAsset:
    @patch("forecasting_engine.shared.model_trainer.train_model_pipeline")
    def test_successful_training(self, mock_train, trainer):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=300, freq="h"),
                "load_kW": range(300),
                "temp_C": range(300),
            }
        )
        trainer.measurements_io.to_df.return_value = df
        trainer.train_asset("A")
        trainer.measurements_io.to_df.assert_called_once_with("A")
        mock_train.assert_called_once()

    @patch("forecasting_engine.shared.model_trainer.train_model_pipeline")
    def test_lookup_error(self, mock_train, trainer):
        mock_train.side_effect = LookupError("model missing")
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=300, freq="h"),
                "load_kW": range(300),
                "temp_C": range(300),
            }
        )
        trainer.measurements_io.to_df.return_value = df
        with pytest.raises(LookupError):
            trainer.train_asset("A")

    @patch("forecasting_engine.shared.model_trainer.train_model_pipeline")
    def test_other_error(self, mock_train, trainer):
        mock_train.side_effect = Exception("generic fail")
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=300, freq="h"),
                "load_kW": range(300),
                "temp_C": range(300),
            }
        )
        trainer.measurements_io.to_df.return_value = df
        with pytest.raises(Exception):
            trainer.train_asset("A")

    def test_no_measurements(self, trainer):
        trainer.measurements_io.to_df.return_value = pd.DataFrame()
        with pytest.raises(ValueError):
            trainer.train_asset("A")


# ----------------------------
# Test train_all_assets
# ----------------------------


class TestTrainAllAssets:

    @patch("forecasting_engine.shared.model_trainer.TrainingManager.train_asset")
    def test_trains_multiple_assets(self, mock_train_asset, trainer):
        mock_assets = [MagicMock(asset_uuid="A"), MagicMock(asset_uuid="B")]
        trainer.assets_io.list_assets.return_value = mock_assets
        trainer.train_all_assets()

        mock_train_asset.assert_any_call("A")
        mock_train_asset.assert_any_call("B")
        assert mock_train_asset.call_count == 2

    def test_train_all_assets_logs_exception(self, trainer):
        # Simulate one asset that fails training
        failing_asset = MagicMock(asset_uuid="A")
        trainer.assets_io.list_assets.return_value = [failing_asset]

        with patch.object(mt.logger, "exception") as mock_log:
            # Patch train_asset to raise an exception
            with patch.object(
                trainer, "train_asset", side_effect=Exception("fail for A")
            ):
                trainer.train_all_assets()

            # Ensure logger.exception was called
            mock_log.assert_called()
            args, kwargs = mock_log.call_args
            # The log message should contain asset id and the exception message
            assert "A" in args[0]
            assert "fail for A" in args[0]
