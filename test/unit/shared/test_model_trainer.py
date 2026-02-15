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
# Test _is_valid_asset_to_train
# ----------------------------


class TestIsValidAssetToTrain:
    def test_measured_false_returns_false(self, trainer):
        asset = MagicMock(measured=False, children=[], asset_type="system")
        assert trainer._is_valid_asset_to_train(asset) is False

    def test_measured_true_with_children_returns_false(self, trainer):
        child = MagicMock()
        asset = MagicMock(measured=True, children=[child], asset_type="system")
        assert trainer._is_valid_asset_to_train(asset) is False

    def test_measured_true_no_children_pv_returns_false(self, trainer):
        # IMPORTANT: measured True + no children ensures we hit the pv check
        asset = MagicMock(measured=True, children=[], asset_type="pv")
        assert trainer._is_valid_asset_to_train(asset) is False

    def test_measured_true_no_children_nonpv_returns_true(self, trainer):
        asset = MagicMock(measured=True, children=[], asset_type="system")
        assert trainer._is_valid_asset_to_train(asset) is True


# ----------------------------
# Test train_asset
# ----------------------------


class TestTrainAsset:
    @patch("forecasting_engine.shared.model_trainer.train_model_pipeline")
    @patch.object(TrainingManager, "_is_valid_asset_to_train", return_value=True)
    def test_successful_training(self, mock_valid, mock_train, trainer):
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
    @patch.object(TrainingManager, "_is_valid_asset_to_train", return_value=True)
    def test_lookup_error(self, mock_valid, mock_train, trainer):
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
    @patch.object(TrainingManager, "_is_valid_asset_to_train", return_value=True)
    def test_other_error(self, mock_valid, mock_train, trainer):
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

    @patch.object(TrainingManager, "_is_valid_asset_to_train", return_value=True)
    def test_no_measurements(self, mock_valid, trainer):
        trainer.measurements_io.to_df.return_value = pd.DataFrame()
        with pytest.raises(ValueError):
            trainer.train_asset("A")


# ----------------------------
# Test train_asset when asset is skipped
# ----------------------------


class TestTrainAssetSkipped:
    def test_skipped_asset_logs_info(self, trainer):
        asset_id = "A"

        # Patch _is_valid_asset_to_train to return False
        with patch.object(
            TrainingManager, "_is_valid_asset_to_train", return_value=False
        ):
            # Patch logger.info to capture logs
            with patch.object(mt.logger, "info") as mock_log_info:
                trainer.train_asset(asset_id)

        # Assert the skip log was called
        mock_log_info.assert_any_call(
            f"Skipping training for asset {asset_id} (not eligible)"
        )


# ----------------------------
# Test train_all_assets
# ----------------------------


class TestTrainAllAssets:

    @patch("forecasting_engine.shared.model_trainer.TrainingManager.train_asset")
    def test_trains_multiple_assets_only_valid(self, mock_train_asset, trainer):
        asset1 = MagicMock(
            asset_uuid="A", measured=True, children=[], asset_type="system"
        )
        asset2 = MagicMock(
            asset_uuid="B", measured=False, children=[], asset_type="system"
        )
        asset3 = MagicMock(
            asset_uuid="C", measured=True, children=[MagicMock()], asset_type="system"
        )
        asset4 = MagicMock(asset_uuid="D", measured=True, children=[], asset_type="pv")

        trainer.assets_io.list_assets.return_value = [asset1, asset2, asset3, asset4]

        with patch.object(mt.logger, "info") as mock_log_info:
            trainer.train_all_assets()

        # Only A should train
        mock_train_asset.assert_called_once_with("A")

        # Check that skipped assets are logged (including pv)
        log_messages = [args[0] for args, kwargs in mock_log_info.call_args_list]
        assert any("Skipping training for asset B" in msg for msg in log_messages)
        assert any("Skipping training for asset C" in msg for msg in log_messages)
        assert any("Skipping training for asset D" in msg for msg in log_messages)

    def test_train_all_assets_logs_exception(self, trainer):
        # Simulate one asset that fails training
        failing_asset = MagicMock(asset_uuid="A", measured=True, children=[])
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
            assert "A" in args[0]
            assert "fail for A" in args[0]
