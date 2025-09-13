import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from forecasting_engine.orchestration import forecast_runner
from forecasting_engine.orchestration.forecast_runner import (
    generate_forecast_for_asset,
    cleanup_temp_folder,
)


@pytest.fixture
def mock_training_data():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    return pd.DataFrame({"load": [1, 2, 3]}, index=idx)


@pytest.fixture(autouse=True)
def patch_get_s3_bucket():
    with patch(
        "forecasting_engine.orchestration.s3_utils.get_s3_bucket",
        return_value="dummy-bucket",
    ):
        yield


@pytest.fixture(autouse=True)
def patch_s3_client():
    with patch(
        "forecasting_engine.orchestration.s3_utils.get_s3_client"
    ) as mock_client:
        mock = MagicMock()
        mock_client.return_value = mock
        yield mock


@pytest.fixture
def patch_load_training_pd_from_s3(mock_training_data):
    with patch(
        "forecasting_engine.orchestration.forecast_runner.load_training_pd_from_s3"
    ) as mock_load:
        mock_load.return_value = mock_training_data
        yield mock_load


@pytest.fixture
def patch_create_forecast_pipeline():
    with patch(
        "forecasting_engine.orchestration.forecast_runner.create_forecast_pipeline"
    ) as mock_create:
        yield mock_create


@pytest.fixture
def patch_cleanup_temp_folder():
    with patch(
        "forecasting_engine.orchestration.forecast_runner.cleanup_temp_folder"
    ) as mock_cleanup:
        yield mock_cleanup


@pytest.fixture
def tmp_dir(tmp_path):
    # Create a temporary directory with some files
    d = tmp_path / "asset123"
    d.mkdir()
    (d / "file1.txt").write_text("test")
    (d / "file2.txt").write_text("test2")
    return d


class TestCleanupTempFiles:

    def test_cleanup_existing_folder(self, tmp_dir):
        # The folder exists
        cleanup_temp_folder(tmp_dir)
        assert not tmp_dir.exists()  # folder should be removed

    def test_cleanup_nonexistent_folder(self, tmp_path):
        non_exist = tmp_path / "does_not_exist"
        cleanup_temp_folder(non_exist)  # should not raise
        assert not non_exist.exists()

    def test_cleanup_exception_logged(self, tmp_dir):
        with patch.object(forecast_runner.logger, "warning") as mock_warning:
            with patch("shutil.rmtree") as mock_rmtree:
                mock_rmtree.side_effect = Exception("cannot delete")
                forecast_runner.cleanup_temp_folder(tmp_dir)
                mock_warning.assert_called_once()


class TestGenerateForecastForAsset:

    def test_generate_forecast_success(
        self,
        patch_load_training_pd_from_s3,
        patch_create_forecast_pipeline,
        mock_training_data,
    ):
        asset_id = "ASSET123"
        mock_forecast = pd.DataFrame(
            {"forecast": [10, 20, 30]}, index=mock_training_data.index
        )
        patch_create_forecast_pipeline.return_value = mock_forecast

        result = generate_forecast_for_asset(asset_id)

        assert isinstance(result, pd.DataFrame)
        assert "forecast" in result.columns
        pd.testing.assert_frame_equal(result, mock_forecast)
        patch_load_training_pd_from_s3.assert_called_once_with(asset_id=asset_id)
        patch_create_forecast_pipeline.assert_called_once()

    def test_generate_forecast_training_data_error(
        self, patch_load_training_pd_from_s3
    ):
        patch_load_training_pd_from_s3.side_effect = FileNotFoundError(
            "Missing training data"
        )
        with pytest.raises(FileNotFoundError):
            generate_forecast_for_asset("ASSET_MISSING")

    def test_generate_forecast_training_data_value_error(
        self, patch_load_training_pd_from_s3
    ):
        patch_load_training_pd_from_s3.side_effect = ValueError(
            "Malformed training data"
        )
        with pytest.raises(ValueError, match="Malformed training data"):
            generate_forecast_for_asset("ASSET_MALFORMED")

    def test_generate_forecast_training_data_unexpected_exception(
        self, patch_load_training_pd_from_s3
    ):
        patch_load_training_pd_from_s3.side_effect = RuntimeError("Unexpected S3 error")
        with pytest.raises(RuntimeError, match="Unexpected S3 error"):
            generate_forecast_for_asset("ASSET_ERROR")

    def test_generate_forecast_pipeline_error(
        self,
        patch_load_training_pd_from_s3,
        patch_create_forecast_pipeline,
        mock_training_data,
    ):
        patch_create_forecast_pipeline.side_effect = LookupError("No model found")
        with pytest.raises(LookupError):
            generate_forecast_for_asset("ASSET_NO_MODEL")

    def test_generate_forecast_pipeline_exception(
        self,
        patch_load_training_pd_from_s3,
        patch_create_forecast_pipeline,
        mock_training_data,
    ):
        patch_create_forecast_pipeline.side_effect = RuntimeError(
            "Some pipeline failure"
        )
        with pytest.raises(RuntimeError, match="Some pipeline failure"):
            generate_forecast_for_asset("ASSET_FAIL")

    def test_cleanup_called_on_success(
        self,
        patch_load_training_pd_from_s3,
        patch_create_forecast_pipeline,
        patch_cleanup_temp_folder,
        mock_training_data,
    ):
        asset_id = "ASSET123"
        mock_forecast = pd.DataFrame(
            {"forecast": [10, 20, 30]}, index=mock_training_data.index
        )
        patch_create_forecast_pipeline.return_value = mock_forecast

        result = generate_forecast_for_asset(asset_id)

        pd.testing.assert_frame_equal(result, mock_forecast)
        patch_cleanup_temp_folder.assert_called_once()

    def test_cleanup_called_on_exception(
        self,
        patch_load_training_pd_from_s3,
        patch_create_forecast_pipeline,
        patch_cleanup_temp_folder,
        mock_training_data,
    ):
        asset_id = "ASSET_ERROR"
        patch_create_forecast_pipeline.side_effect = RuntimeError("Pipeline failed")

        with pytest.raises(RuntimeError, match="Pipeline failed"):
            generate_forecast_for_asset(asset_id)

        patch_cleanup_temp_folder.assert_called_once()
